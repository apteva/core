package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// OpenAINativeProvider uses the OpenAI Responses API for web_search,
// code_interpreter, and other OpenAI-specific features.
// For OpenAI-compatible endpoints (Fireworks, Ollama, etc.), use OpenAICompatProvider.
type OpenAINativeProvider struct {
	name              string
	apiKey            string
	responsesURL      string
	forceStoreFalse   bool
	runtimeTokenURL   string
	serverAPIKey      string
	accountID         string
	models            map[ModelTier]string
	modelCapabilities map[string]ModelCapabilities
	builtinTools      []string
	reasoning         ReasoningSettings
	tokenMu           sync.RWMutex
	refreshMu         sync.Mutex
	lastTokenRefresh  time.Time
	cacheStateMu      sync.Mutex
	cacheState        *openAIPromptCacheState
}

func NewOpenAINativeProvider(apiKey string) LLMProvider {
	return &OpenAINativeProvider{
		name:         "openai",
		apiKey:       apiKey,
		responsesURL: openAIBaseURL() + "/responses",
		models: map[ModelTier]string{
			ModelLarge:  "gpt-5.4-mini",
			ModelMedium: "gpt-5.4-mini",
			ModelSmall:  "gpt-5.4-mini",
		},
	}
}

func NewOpenAICodexProvider(accessToken string) LLMProvider {
	serverURL := strings.TrimRight(os.Getenv("SERVER_URL"), "/")
	providerID := strings.TrimSpace(os.Getenv("OPENAI_CODEX_PROVIDER_ID"))
	runtimeTokenURL := ""
	if serverURL != "" && providerID != "" {
		runtimeTokenURL = serverURL + "/api/providers/" + providerID + "/auth/runtime-token"
	}
	return &OpenAINativeProvider{
		name:            "openai-codex",
		apiKey:          accessToken,
		responsesURL:    "https://chatgpt.com/backend-api/codex/responses",
		forceStoreFalse: true,
		runtimeTokenURL: runtimeTokenURL,
		serverAPIKey:    os.Getenv("APTEVA_API_KEY"),
		accountID:       strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCOUNT_ID")),
		models: map[ModelTier]string{
			ModelLarge:  "gpt-5.5",
			ModelMedium: "gpt-5.5",
			ModelSmall:  "gpt-5.5",
		},
	}
}

func (p *OpenAINativeProvider) Name() string {
	if p.name != "" {
		return p.name
	}
	return "openai"
}
func (p *OpenAINativeProvider) Models() map[ModelTier]string { return p.models }
func (p *OpenAINativeProvider) SupportsNativeTools() bool    { return true }
func (p *OpenAINativeProvider) CostPer1M() (float64, float64, float64) {
	if p.Name() == "openai-codex" {
		return 0, 0, 0
	}
	// Default to gpt-5.4-mini pricing
	return 0.75, 0.375, 4.50
}

func (p *OpenAINativeProvider) AvailableBuiltinTools() []BuiltinTool {
	if p.Name() == "openai-codex" {
		return nil
	}
	return []BuiltinTool{
		{Type: "code_interpreter", Name: "code_interpreter"},
		{Type: "web_search_preview", Name: "web_search"},
	}
}

func (p *OpenAINativeProvider) SetBuiltinTools(tools []string) {
	p.builtinTools = tools
}

func (p *OpenAINativeProvider) WithBuiltins(builtins []string) LLMProvider {
	clone := p.clone()
	clone.builtinTools = builtins
	return clone
}

func (p *OpenAINativeProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	clone := p.clone()
	clone.reasoning = settings
	return clone
}

func (p *OpenAINativeProvider) clone() *OpenAINativeProvider {
	cacheState := p.promptCacheState()
	p.tokenMu.RLock()
	apiKey := p.apiKey
	lastRefresh := p.lastTokenRefresh
	accountID := p.accountID
	p.tokenMu.RUnlock()
	return &OpenAINativeProvider{
		name: p.name, apiKey: apiKey, responsesURL: p.responsesURL,
		forceStoreFalse: p.forceStoreFalse, runtimeTokenURL: p.runtimeTokenURL,
		serverAPIKey: p.serverAPIKey, accountID: accountID, models: p.models,
		modelCapabilities: cloneModelCapabilitiesMap(p.modelCapabilities),
		builtinTools:      append([]string(nil), p.builtinTools...), reasoning: p.reasoning,
		lastTokenRefresh: lastRefresh, cacheState: cacheState,
	}
}

func (p *OpenAINativeProvider) promptCacheState() *openAIPromptCacheState {
	p.cacheStateMu.Lock()
	defer p.cacheStateMu.Unlock()
	if p.cacheState == nil {
		p.cacheState = &openAIPromptCacheState{}
	}
	return p.cacheState
}

func (p *OpenAINativeProvider) token() string {
	p.tokenMu.RLock()
	defer p.tokenMu.RUnlock()
	return p.apiKey
}

func (p *OpenAINativeProvider) account() string {
	p.tokenMu.RLock()
	defer p.tokenMu.RUnlock()
	return p.accountID
}

func (p *OpenAINativeProvider) requestReasoning(model string) *oaiReasoning {
	level := normalizeReasoningLevel(p.reasoning.Level)
	summariesSupported := true
	if caps, ok := p.modelCapabilities[model]; ok && caps.SupportsReasoningSummaries != nil {
		summariesSupported = *caps.SupportsReasoningSummaries
	}
	if p.Name() == "openai-codex" && level == ReasoningAuto {
		if summariesSupported {
			return &oaiReasoning{Summary: "auto"}
		}
		return nil
	}
	if level == ReasoningAuto {
		return nil
	}
	effort := openAIReasoningEffort(level, p.Name(), model)
	if caps, ok := p.modelCapabilities[model]; ok {
		effort = reasoningEffortForCapabilities(caps, effort)
	} else {
		effort = modelReasoningEffort(model, effort)
	}
	out := &oaiReasoning{Effort: effort}
	if level != ReasoningNone && summariesSupported {
		out.Summary = "auto"
	}
	return out
}

func openAIReasoningEffort(level ReasoningLevel, providerName, model string) string {
	switch normalizeReasoningLevel(level) {
	case ReasoningNone:
		return "none"
	case ReasoningMinimal:
		if openAIModelSupportsXHigh(model) || providerName == "openai-codex" {
			return "low"
		}
		return "minimal"
	case ReasoningLow:
		return "low"
	case ReasoningMedium:
		return "medium"
	case ReasoningHigh:
		return "high"
	case ReasoningXHigh:
		if openAIModelSupportsXHigh(model) || providerName == "openai-codex" {
			return "xhigh"
		}
		return "high"
	default:
		return ""
	}
}

func openAIModelSupportsXHigh(model string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(model)), "gpt-5.5")
}

// --- Responses API types ---

type oaiResponsesRequest struct {
	Model                string         `json:"model"`
	Instructions         string         `json:"instructions,omitempty"`
	Input                []oaiInputItem `json:"input"`
	Tools                []any          `json:"tools,omitempty"`
	Stream               bool           `json:"stream"`
	Store                *bool          `json:"store,omitempty"`
	Reasoning            *oaiReasoning  `json:"reasoning,omitempty"`
	PromptCacheKey       string         `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string         `json:"prompt_cache_retention,omitempty"`
	ParallelToolCalls    *bool          `json:"parallel_tool_calls,omitempty"`
	Include              []string       `json:"include,omitempty"`
}

type oaiReasoning struct {
	Summary string `json:"summary,omitempty"`
	Effort  string `json:"effort,omitempty"`
}

// oaiInputItem is a polymorphic input item for the Responses API.
type oaiInputItem struct {
	Raw     json.RawMessage `json:"-"`                 // exact Responses output item replay
	Type    string          `json:"type"`              // "message", "function_call", "function_call_output"
	ID      string          `json:"id,omitempty"`      // for replaying Responses output items
	Status  string          `json:"status,omitempty"`  // for replaying Responses output items
	Role    string          `json:"role,omitempty"`    // for type=message
	Content any             `json:"content,omitempty"` // string or []oaiContentBlock
	Name    string          `json:"name,omitempty"`    // for type=function_call

	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Output    any    `json:"output,omitempty"`
	Arguments string `json:"arguments,omitempty"` // for type=function_call (JSON string)
}

func (i oaiInputItem) MarshalJSON() ([]byte, error) {
	if len(i.Raw) > 0 {
		if !json.Valid(i.Raw) {
			return nil, fmt.Errorf("invalid raw Responses input item")
		}
		return i.Raw, nil
	}
	type wireItem oaiInputItem
	return json.Marshal(wireItem(i))
}

type oaiContentBlock struct {
	Type     string `json:"type"` // "input_text", "input_image", "input_file"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"` // data:image/png;base64,...
	Detail   string `json:"detail,omitempty"`    // "original", "high", "low"
	FileURL  string `json:"file_url,omitempty"`  // URL or data URI for audio/files
}

type oaiFunctionTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      bool           `json:"strict"`
}

// --- Streaming response types ---

type oaiStreamEvent struct {
	Type     string          `json:"type"`
	Sequence int             `json:"sequence,omitempty"`
	Item     json.RawMessage `json:"item,omitempty"`
	Delta    json.RawMessage `json:"delta,omitempty"`
}

type oaiOutputItem struct {
	Type    string `json:"type"` // "message", "function_call"
	ID      string `json:"id,omitempty"`
	Status  string `json:"status,omitempty"`
	Role    string `json:"role,omitempty"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content,omitempty"`

	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// reasoning fields
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"summary,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

const openAIResponsesStateProvider = "openai-responses"

func (p *OpenAINativeProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	if p.Name() == "openai-codex" {
		_ = p.refreshRuntimeToken(ctx, false)
	}
	// Convert messages to Responses API input items
	input := p.buildInput(messages)

	// Convert tools
	apiTools := p.buildAPITools(model, tools)

	// Add builtin tools (only those supported by Responses API)
	supportedBuiltins := map[string]bool{
		"code_interpreter": true, "web_search_preview": true,
		"file_search": true, "image_generation": true,
	}
	for _, bt := range p.builtinTools {
		if supportedBuiltins[bt] {
			apiTools = append(apiTools, map[string]string{"type": bt})
		}
	}

	reqBody := oaiResponsesRequest{
		Model:  model,
		Input:  input,
		Tools:  apiTools,
		Stream: true,
	}
	reqBody.Reasoning = p.requestReasoning(model)
	if caps, ok := p.modelCapabilities[model]; ok && caps.SupportsParallelToolCalls != nil {
		value := *caps.SupportsParallelToolCalls
		reqBody.ParallelToolCalls = &value
	}
	if p.forceStoreFalse {
		store := false
		reqBody.Store = &store
		reqBody.Instructions = p.instructionsFromMessages(messages)
		// Stateless Responses calls must carry opaque reasoning forward
		// themselves. Request it on every Codex call so completed output
		// items can be replayed exactly with later function outputs.
		reqBody.Include = []string{"reasoning.encrypted_content"}
	}
	stablePrefix := reqBody.Instructions
	if stablePrefix == "" {
		stablePrefix = p.instructionsFromMessages(messages)
	}
	cacheHints := openAIPromptCacheHintsForScope(p.Name(), model, stablePrefix, apiTools, openAIPromptCacheScopeFromContext(ctx))
	cacheState := p.promptCacheState()
	if !cacheState.enabled() {
		cacheHints = openAIPromptCacheHints{}
	}
	reqBody.PromptCacheKey = cacheHints.Key
	reqBody.PromptCacheRetention = cacheHints.Retention

	body, err := json.Marshal(reqBody)
	if err != nil {
		return ChatResponse{}, err
	}

	// Log first tool for debugging
	if len(apiTools) > 0 {
		toolJSON, _ := json.Marshal(apiTools[0])
		logMsg("OPENAI-NATIVE", fmt.Sprintf("model=%s input_items=%d tools=%d first_tool=%s", model, len(input), len(apiTools), string(toolJSON)))
	} else {
		logMsg("OPENAI-NATIVE", fmt.Sprintf("model=%s input_items=%d tools=0", model, len(input)))
	}

	responsesURL := p.responsesURL
	if responsesURL == "" {
		responsesURL = openAIBaseURL() + "/responses"
	}
	doRequest := func(payload []byte) (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, "POST", responsesURL, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+p.token())
		if accountID := p.account(); p.Name() == "openai-codex" && accountID != "" {
			req.Header.Set("ChatGPT-Account-ID", accountID)
		}
		return llmHTTPClient.Do(req)
	}

	resp, err := doRequest(body)
	if err != nil {
		return ChatResponse{}, err
	}
	if p.Name() == "openai-codex" && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) && p.refreshRuntimeToken(ctx, true) == nil {
		_ = resp.Body.Close()
		resp, err = doRequest(body)
		if err != nil {
			return ChatResponse{}, err
		}
	}

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if cacheHints.Key != "" && openAICacheHintsUnsupported(resp.StatusCode, string(respBody)) {
			cacheState.disable()
			reqBody.PromptCacheKey = ""
			reqBody.PromptCacheRetention = ""
			retryBody, err := json.Marshal(reqBody)
			if err != nil {
				return ChatResponse{}, err
			}
			logMsg("OPENAI-NATIVE", fmt.Sprintf("cache hints unsupported, retrying without them: %d %s", resp.StatusCode, string(respBody)))
			resp, err = doRequest(retryBody)
			if err != nil {
				return ChatResponse{}, err
			}
			if p.Name() == "openai-codex" && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) && p.refreshRuntimeToken(ctx, true) == nil {
				_ = resp.Body.Close()
				resp, err = doRequest(retryBody)
				if err != nil {
					return ChatResponse{}, err
				}
			}
			if resp.StatusCode == http.StatusOK {
				idleBody := newIdleReader(resp.Body, streamIdleTimeout(), func() {
					logMsg("OPENAI-NATIVE", fmt.Sprintf("stream idle for %s on model=%s", streamIdleTimeout(), model))
				})
				defer idleBody.Close()
				return p.streamResponse(idleBody, onChunk, onThinking, onToolChunk)
			}
			respBody, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()
		}
		logMsg("OPENAI-NATIVE", fmt.Sprintf("ERROR %d: %s", resp.StatusCode, string(respBody)))
		return ChatResponse{}, fmt.Errorf("OpenAI Responses API error %d: %s", resp.StatusCode, string(respBody))
	}

	idleBody := newIdleReader(resp.Body, streamIdleTimeout(), func() {
		logMsg("OPENAI-NATIVE", fmt.Sprintf("stream idle for %s on model=%s", streamIdleTimeout(), model))
	})
	defer idleBody.Close()
	return p.streamResponse(idleBody, onChunk, onThinking, onToolChunk)
}

func (p *OpenAINativeProvider) buildAPITools(_ string, tools []NativeTool) []any {
	var apiTools []any
	for _, t := range tools {
		apiTools = append(apiTools, oaiFunctionTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
			Strict:      false,
		})
	}
	return apiTools
}

func (p *OpenAINativeProvider) refreshRuntimeToken(ctx context.Context, force bool) error {
	if p.runtimeTokenURL == "" || p.serverAPIKey == "" {
		return nil
	}
	before := p.token()
	p.refreshMu.Lock()
	defer p.refreshMu.Unlock()
	if force {
		p.tokenMu.RLock()
		changed := p.apiKey != before
		recent := !p.lastTokenRefresh.IsZero() && time.Since(p.lastTokenRefresh) < time.Second
		p.tokenMu.RUnlock()
		if changed || recent {
			return nil
		}
	}
	url := p.runtimeTokenURL
	if force {
		url += "?force=1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+p.serverAPIKey)
	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("runtime token refresh failed: HTTP %d", resp.StatusCode)
	}
	var payload struct {
		AccessToken string `json:"access_token"`
		AccountID   string `json:"account_id"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return err
	}
	if strings.TrimSpace(payload.AccessToken) == "" {
		return fmt.Errorf("runtime token refresh returned empty access_token")
	}
	p.tokenMu.Lock()
	p.apiKey = payload.AccessToken
	if strings.TrimSpace(payload.AccountID) != "" {
		p.accountID = strings.TrimSpace(payload.AccountID)
	}
	p.lastTokenRefresh = time.Now()
	p.tokenMu.Unlock()
	return nil
}

func (p *OpenAINativeProvider) instructionsFromMessages(messages []Message) string {
	var parts []string
	for _, m := range messages {
		if m.Role == "system" && strings.TrimSpace(m.TextContent()) != "" {
			parts = append(parts, m.TextContent())
		}
	}
	if len(parts) == 0 {
		return "You are an Apteva agent. Follow the provided tools and user instructions."
	}
	return strings.Join(parts, "\n\n")
}

// buildInput converts our Message slice to Responses API input items.
func (p *OpenAINativeProvider) buildInput(messages []Message) []oaiInputItem {
	var items []oaiInputItem

	for _, m := range messages {
		// System message
		if m.Role == "system" {
			if p.forceStoreFalse {
				// Codex receives system/developer text through the
				// top-level instructions field. Repeating it in input[]
				// bloats the uncached prefix and weakens cache locality.
				continue
			}
			items = append(items, oaiInputItem{
				Type:    "message",
				Role:    "developer",
				Content: m.TextContent(),
			})
			continue
		}

		// Responses output items are valid subsequent input items. Prefer
		// the exact provider payload (reasoning, message, function calls,
		// IDs and status) over the lossy legacy reconstruction below.
		if m.ProviderState != nil &&
			m.ProviderState.Provider == openAIResponsesStateProvider &&
			len(m.ProviderState.Items) > 0 {
			valid := true
			for _, raw := range m.ProviderState.Items {
				if len(raw) == 0 || !json.Valid(raw) {
					valid = false
					break
				}
			}
			if valid {
				for _, raw := range m.ProviderState.Items {
					items = append(items, oaiInputItem{Raw: append(json.RawMessage(nil), raw...)})
				}
				continue
			}
		}

		// Tool results
		if len(m.ToolResults) > 0 {
			for _, tr := range m.ToolResults {
				if tr.Image != nil {
					imageURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(tr.Image)
					items = append(items, oaiInputItem{
						Type:   "function_call_output",
						CallID: tr.CallID,
						Output: []oaiContentBlock{
							{Type: "input_text", Text: tr.Content},
							{Type: "input_image", ImageURL: imageURL, Detail: "original"},
						},
					})
				} else {
					// Function call output
					items = append(items, oaiInputItem{
						Type:   "function_call_output",
						CallID: tr.CallID,
						Output: tr.Content,
					})
				}
			}
			continue
		}

		// Assistant message with tool calls — re-emit as output items
		if len(m.ToolCalls) > 0 {
			// First add any text content
			if m.Content != "" {
				items = append(items, oaiInputItem{
					Type:    "message",
					Role:    "assistant",
					Content: m.Content,
				})
			}
			// Then add each tool call as its original output item
			for _, tc := range m.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Args)
				items = append(items, oaiInputItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Name,
					Arguments: string(argsJSON),
				})
			}
			continue
		}

		// Regular user/assistant message
		role := m.Role
		if role == "assistant" {
			role = "assistant"
		}

		if m.HasParts() {
			logMsg("OPENAI-NATIVE", fmt.Sprintf("message with %d parts (role=%s)", len(m.Parts), m.Role))
			var blocks []oaiContentBlock
			for _, part := range m.Parts {
				logMsg("OPENAI-NATIVE", fmt.Sprintf("  part type=%s", part.Type))
				switch part.Type {
				case "text":
					blocks = append(blocks, oaiContentBlock{Type: "input_text", Text: part.Text})
				case "image_url":
					if part.ImageURL != nil {
						blocks = append(blocks, oaiContentBlock{Type: "input_image", ImageURL: part.ImageURL.URL, Detail: "original"})
					}
				case "audio_url", "input_audio":
					// OpenAI Responses API does not support audio input — skip silently
				}
			}
			items = append(items, oaiInputItem{Type: "message", Role: role, Content: blocks})
		} else if m.Content != "" {
			items = append(items, oaiInputItem{Type: "message", Role: role, Content: m.Content})
		}
	}

	return items
}

// streamResponse parses the Responses API SSE stream.
func (p *OpenAINativeProvider) streamResponse(body io.Reader, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	var full strings.Builder
	var fullReasoning strings.Builder
	var usage TokenUsage
	var toolCalls []NativeToolCall
	var providerItems []json.RawMessage
	completedStream := false

	// Track pending items
	type pendingFunc struct {
		id         string
		name       string
		args       strings.Builder
		pendingBuf strings.Builder
	}
	pendingFuncs := map[string]*pendingFunc{} // by item ID

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

streamLoop:
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event oaiStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if os.Getenv("APTEVA_DUMP_STREAM_EVENTS") == "1" {
			logMsg("OPENAI-NATIVE-STREAM", event.Type)
			logOpenAINativeStreamItemMeta(event.Type, event.Item)
		}

		switch event.Type {
		// Text content delta
		case "response.output_text.delta":
			var delta struct {
				Delta string `json:"delta"`
			}
			json.Unmarshal([]byte(data), &delta)
			if delta.Delta != "" {
				full.WriteString(delta.Delta)
				if onChunk != nil {
					onChunk(delta.Delta)
				}
			}

		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			var delta struct {
				Delta string `json:"delta"`
			}
			json.Unmarshal([]byte(data), &delta)
			if delta.Delta != "" {
				fullReasoning.WriteString(delta.Delta)
				if onThinking != nil {
					onThinking(delta.Delta)
				}
			}

		case "response.reasoning_summary_text.done", "response.reasoning_text.done":
			var done struct {
				Text string `json:"text"`
			}
			json.Unmarshal([]byte(data), &done)
			if done.Text != "" && fullReasoning.Len() == 0 {
				fullReasoning.WriteString(done.Text)
				if onThinking != nil {
					onThinking(done.Text)
				}
			}

		case "response.reasoning_summary_part.done":
			var done struct {
				Part struct {
					Text string `json:"text"`
				} `json:"part"`
			}
			json.Unmarshal([]byte(data), &done)
			if done.Part.Text != "" && fullReasoning.Len() == 0 {
				fullReasoning.WriteString(done.Part.Text)
				if onThinking != nil {
					onThinking(done.Part.Text)
				}
			}

		// Function call started
		case "response.function_call_arguments.delta":
			var delta struct {
				ItemID string `json:"item_id"`
				Delta  string `json:"delta"`
			}
			json.Unmarshal([]byte(data), &delta)
			pf, ok := pendingFuncs[delta.ItemID]
			if !ok && delta.ItemID != "" {
				pf = &pendingFunc{}
				pendingFuncs[delta.ItemID] = pf
				ok = true
			}
			if ok {
				pf.args.WriteString(delta.Delta)
				if pf.id == "" {
					pf.pendingBuf.WriteString(delta.Delta)
				} else if onToolChunk != nil && pf.name != "" {
					if pf.pendingBuf.Len() > 0 {
						onToolChunk(pf.name, pf.id, pf.pendingBuf.String())
						pf.pendingBuf.Reset()
					}
					onToolChunk(pf.name, pf.id, delta.Delta)
				}
			}

		// Output item added (function_call, message)
		case "response.output_item.added":
			var item oaiOutputItem
			json.Unmarshal(event.Item, &item)

			switch item.Type {
			case "function_call":
				pendingFuncs[item.ID] = &pendingFunc{id: item.CallID, name: item.Name}
			}

		// Output item done — finalize
		case "response.output_item.done":
			var item oaiOutputItem
			json.Unmarshal(event.Item, &item)
			if len(event.Item) > 0 && json.Valid(event.Item) {
				providerItems = append(providerItems, append(json.RawMessage(nil), event.Item...))
			}

			switch item.Type {
			case "function_call":
				pf, ok := pendingFuncs[item.ID]
				if ok {
					if item.Name != "" {
						pf.name = item.Name
					}
					if item.CallID != "" {
						pf.id = item.CallID
					}
					if pf.id == "" {
						pf.id = item.ID
					}
					if onToolChunk != nil && pf.name != "" && pf.pendingBuf.Len() > 0 {
						onToolChunk(pf.name, pf.id, pf.pendingBuf.String())
						pf.pendingBuf.Reset()
					}
					args := make(map[string]string)
					var rawArgs map[string]any
					argsJSON := pf.args.String()
					if argsJSON == "" {
						argsJSON = item.Arguments
						if argsJSON != "" && onToolChunk != nil && pf.name != "" {
							onToolChunk(pf.name, pf.id, argsJSON)
						}
					}
					if err := json.Unmarshal([]byte(argsJSON), &rawArgs); err != nil || rawArgs == nil || pf.id == "" || pf.name == "" {
						return ChatResponse{}, fmt.Errorf("invalid OpenAI tool call %q arguments or identity", pf.id)
					}
					if json.Unmarshal([]byte(argsJSON), &rawArgs) == nil {
						for k, v := range rawArgs {
							switch val := v.(type) {
							case string:
								args[k] = val
							default:
								b, _ := json.Marshal(v)
								args[k] = string(b)
							}
						}
					}
					toolCalls = append(toolCalls, NativeToolCall{
						ID:            pf.id,
						OutputItemID:  item.ID,
						Status:        item.Status,
						Name:          pf.name,
						Args:          args,
						RawArgs:       argsJSON,
						CanonicalArgs: json.RawMessage(argsJSON),
					})
					delete(pendingFuncs, item.ID)
				}

			case "reasoning":
				if fullReasoning.Len() == 0 {
					for _, summary := range item.Summary {
						if summary.Text == "" {
							continue
						}
						fullReasoning.WriteString(summary.Text)
						if onThinking != nil {
							onThinking(summary.Text)
						}
					}
				}

			}

		// Usage
		case "response.completed":
			completedStream = true
			// Log raw usage for debugging
			var rawCompleted map[string]any
			json.Unmarshal([]byte(data), &rawCompleted)
			if resp, ok := rawCompleted["response"].(map[string]any); ok {
				if u, ok := resp["usage"].(map[string]any); ok {
					logMsg("OPENAI-NATIVE", fmt.Sprintf("raw usage: %v", u))
				}
			}

			var completed struct {
				Response struct {
					Usage struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
						InputDetails struct {
							CachedTokens     int `json:"cached_tokens"`
							CacheWriteTokens int `json:"cache_write_tokens"`
						} `json:"input_tokens_details"`
					} `json:"usage"`
				} `json:"response"`
			}
			json.Unmarshal([]byte(data), &completed)
			usage.PromptTokens = completed.Response.Usage.InputTokens
			usage.CompletionTokens = completed.Response.Usage.OutputTokens
			usage.CachedTokens = completed.Response.Usage.InputDetails.CachedTokens
			usage.CacheWriteTokens = completed.Response.Usage.InputDetails.CacheWriteTokens
			break streamLoop

		case "response.failed", "response.incomplete", "error":
			var failed struct {
				Error *struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error,omitempty"`
				Response struct {
					Error *struct {
						Code    string `json:"code"`
						Message string `json:"message"`
					} `json:"error,omitempty"`
					IncompleteDetails json.RawMessage `json:"incomplete_details,omitempty"`
				} `json:"response,omitempty"`
			}
			_ = json.Unmarshal([]byte(data), &failed)
			if failed.Error != nil {
				return ChatResponse{}, fmt.Errorf("OpenAI stream %s (%s): %s", event.Type, failed.Error.Code, failed.Error.Message)
			}
			if failed.Response.Error != nil {
				return ChatResponse{}, fmt.Errorf("OpenAI stream %s (%s): %s", event.Type, failed.Response.Error.Code, failed.Response.Error.Message)
			}
			return ChatResponse{}, fmt.Errorf("OpenAI stream %s: %s", event.Type, strings.TrimSpace(string(failed.Response.IncompleteDetails)))
		}
	}
	if err := scanner.Err(); err != nil {
		return ChatResponse{}, fmt.Errorf("OpenAI stream read error: %w", err)
	}

	if !completedStream || len(pendingFuncs) != 0 {
		return ChatResponse{}, fmt.Errorf("OpenAI stream ended before response.completed or with unfinished calls")
	}
	response := full.String()
	logMsg("OPENAI-NATIVE", fmt.Sprintf("done tokens_in=%d tokens_out=%d tools=%d len=%d", usage.PromptTokens, usage.CompletionTokens, len(toolCalls), len(response)))
	var providerState *ProviderResponseState
	if len(providerItems) > 0 {
		providerState = &ProviderResponseState{
			Provider: openAIResponsesStateProvider,
			Items:    providerItems,
		}
	}
	return ChatResponse{
		Text:          response,
		ToolCalls:     toolCalls,
		Reasoning:     fullReasoning.String(),
		ProviderState: providerState,
		Usage:         usage,
	}, nil
}

func logOpenAINativeStreamItemMeta(eventType string, raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var item oaiOutputItem
	if err := json.Unmarshal(raw, &item); err != nil {
		logMsg("OPENAI-NATIVE-STREAM", fmt.Sprintf("%s item_meta_unparseable bytes=%d", eventType, len(raw)))
		return
	}
	contentTypes := make(map[string]int)
	contentTextChars := 0
	for _, c := range item.Content {
		contentTypes[c.Type]++
		contentTextChars += len(c.Text)
	}
	summaryTypes := make(map[string]int)
	summaryTextChars := 0
	for _, s := range item.Summary {
		summaryTypes[s.Type]++
		summaryTextChars += len(s.Text)
	}
	logMsg("OPENAI-NATIVE-STREAM", fmt.Sprintf(
		"%s item_meta type=%q status=%q id_present=%t call_id_present=%t name=%q content_count=%d content_types=%v content_text_chars=%d summary_count=%d summary_types=%v summary_text_chars=%d arguments_chars=%d",
		eventType,
		item.Type,
		item.Status,
		item.ID != "",
		item.CallID != "",
		item.Name,
		len(item.Content),
		contentTypes,
		contentTextChars,
		len(item.Summary),
		summaryTypes,
		summaryTextChars,
		len(item.Arguments),
	))
}
