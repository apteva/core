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

	"github.com/apteva/core/pkg/computer"
)

// OpenAINativeProvider uses the OpenAI Responses API for native computer use,
// web_search, code_interpreter, and other OpenAI-specific features.
// For OpenAI-compatible endpoints (Fireworks, Ollama, etc.), use OpenAICompatProvider.
type OpenAINativeProvider struct {
	name            string
	apiKey          string
	responsesURL    string
	forceStoreFalse bool
	runtimeTokenURL string
	serverAPIKey    string
	models          map[ModelTier]string
	builtinTools    []string
}

func NewOpenAINativeProvider(apiKey string) LLMProvider {
	return &OpenAINativeProvider{
		name:         "openai",
		apiKey:       apiKey,
		responsesURL: "https://api.openai.com/v1/responses",
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
	clone := *p
	clone.builtinTools = builtins
	return &clone
}

// --- Responses API types ---

type oaiResponsesRequest struct {
	Model        string         `json:"model"`
	Instructions string         `json:"instructions,omitempty"`
	Input        []oaiInputItem `json:"input"`
	Tools        []any          `json:"tools,omitempty"`
	Stream       bool           `json:"stream"`
	Store        *bool          `json:"store,omitempty"`
	Reasoning    *oaiReasoning  `json:"reasoning,omitempty"`
}

type oaiReasoning struct {
	Summary string `json:"summary,omitempty"`
}

// oaiInputItem is a polymorphic input item for the Responses API.
type oaiInputItem struct {
	Type    string `json:"type"`              // "message", "computer_call_output", "function_call", "function_call_output"
	ID      string `json:"id,omitempty"`      // for replaying Responses output items
	Status  string `json:"status,omitempty"`  // for replaying Responses output items
	Role    string `json:"role,omitempty"`    // for type=message
	Content any    `json:"content,omitempty"` // string or []oaiContentBlock
	Name    string `json:"name,omitempty"`    // for type=function_call

	// computer_call_output / function_call fields
	CallID    string `json:"call_id,omitempty"`
	Output    any    `json:"output,omitempty"`    // screenshot etc.
	Arguments string `json:"arguments,omitempty"` // for type=function_call (JSON string)
	Actions   any    `json:"actions,omitempty"`   // for type=computer_call (GA computer tool)
	Action    any    `json:"action,omitempty"`    // for legacy computer-use-preview
}

type oaiContentBlock struct {
	Type     string `json:"type"` // "input_text", "input_image", "input_file"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"` // data:image/png;base64,...
	Detail   string `json:"detail,omitempty"`    // "original", "high", "low"
	FileURL  string `json:"file_url,omitempty"`  // URL or data URI for audio/files
}

type oaiComputerTool struct {
	Type          string `json:"type"` // "computer"
	DisplayWidth  int    `json:"display_width,omitempty"`
	DisplayHeight int    `json:"display_height,omitempty"`
	Environment   string `json:"environment,omitempty"`
}

type oaiFunctionTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// --- Streaming response types ---

type oaiStreamEvent struct {
	Type     string          `json:"type"`
	Sequence int             `json:"sequence,omitempty"`
	Item     json.RawMessage `json:"item,omitempty"`
	Delta    json.RawMessage `json:"delta,omitempty"`
}

type oaiOutputItem struct {
	Type    string `json:"type"` // "message", "computer_call", "function_call"
	ID      string `json:"id,omitempty"`
	Status  string `json:"status,omitempty"`
	Role    string `json:"role,omitempty"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content,omitempty"`

	// computer_call fields
	CallID  string          `json:"call_id,omitempty"`
	Actions json.RawMessage `json:"actions,omitempty"`
	Action  json.RawMessage `json:"action,omitempty"`

	// function_call fields
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// reasoning fields
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"summary,omitempty"`
}

type oaiComputerAction struct {
	Type     string   `json:"type"` // "click", "type", "keypress", "scroll", "drag", "move", "screenshot", "wait", "double_click"
	X        int      `json:"x,omitempty"`
	Y        int      `json:"y,omitempty"`
	Button   string   `json:"button,omitempty"` // "left", "right", "middle"
	Text     string   `json:"text,omitempty"`   // for "type"
	Key      string   `json:"key,omitempty"`    // for "keypress"
	ScrollX  int      `json:"scroll_x,omitempty"`
	ScrollY  int      `json:"scroll_y,omitempty"`
	StartX   int      `json:"start_x,omitempty"` // drag
	StartY   int      `json:"start_y,omitempty"`
	EndX     int      `json:"end_x,omitempty"`
	EndY     int      `json:"end_y,omitempty"`
	Duration int      `json:"duration,omitempty"` // wait ms
	Keys     []string `json:"keys,omitempty"`     // modifier keys
}

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
	if p.Name() == "openai-codex" {
		reqBody.Reasoning = &oaiReasoning{Summary: "auto"}
	}
	if p.forceStoreFalse {
		store := false
		reqBody.Store = &store
		reqBody.Instructions = p.instructionsFromMessages(messages)
	}

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
		responsesURL = "https://api.openai.com/v1/responses"
	}
	req, err := http.NewRequestWithContext(ctx, "POST", responsesURL, bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return ChatResponse{}, err
	}
	if p.Name() == "openai-codex" && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) && p.refreshRuntimeToken(ctx, true) == nil {
		_ = resp.Body.Close()
		req, err = http.NewRequestWithContext(ctx, "POST", responsesURL, bytes.NewReader(body))
		if err != nil {
			return ChatResponse{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
		resp, err = llmHTTPClient.Do(req)
		if err != nil {
			return ChatResponse{}, err
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		logMsg("OPENAI-NATIVE", fmt.Sprintf("ERROR %d: %s", resp.StatusCode, string(respBody)))
		return ChatResponse{}, fmt.Errorf("OpenAI Responses API error %d: %s", resp.StatusCode, string(respBody))
	}

	return p.streamResponse(resp.Body, onChunk, onThinking, onToolChunk)
}

func (p *OpenAINativeProvider) buildAPITools(model string, tools []NativeTool) []any {
	var apiTools []any
	computerMode := p.openAIComputerMode()
	for _, t := range tools {
		if t.Name == "computer_use" {
			if computerMode == "off" {
				continue
			}
			if computerMode != "custom" && openAIModelSupportsNativeComputer(model) {
				tool := oaiComputerTool{Type: "computer"}
				if strings.HasPrefix(model, "computer-use-preview") {
					tool.Type = "computer_use_preview"
					tool.DisplayWidth, tool.DisplayHeight = 1280, 800
					tool.Environment = "browser"
					if w, ok := t.Parameters["_display_width"].(int); ok {
						tool.DisplayWidth = w
					}
					if h, ok := t.Parameters["_display_height"].(int); ok {
						tool.DisplayHeight = h
					}
				}
				apiTools = append(apiTools, tool)
				logMsg("OPENAI-NATIVE", fmt.Sprintf("native computer tool enabled mode=%s model=%s", computerMode, model))
				continue
			}
			logMsg("OPENAI-NATIVE", fmt.Sprintf("computer_use exposed as function mode=%s model=%s", computerMode, model))
		}
		apiTools = append(apiTools, oaiFunctionTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}
	return apiTools
}

func (p *OpenAINativeProvider) openAIComputerMode() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("APTEVA_OPENAI_COMPUTER_MODE"))) {
	case "custom":
		return "custom"
	case "off":
		return "off"
	case "native":
		return "native"
	default:
		if p.Name() == "openai-codex" {
			return "custom"
		}
		return "native"
	}
}

func openAIModelSupportsNativeComputer(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(model, "gpt-5.5") ||
		strings.HasPrefix(model, "gpt-5.4") ||
		strings.HasPrefix(model, "computer-use-preview")
}

func (p *OpenAINativeProvider) refreshRuntimeToken(ctx context.Context, force bool) error {
	if p.runtimeTokenURL == "" || p.serverAPIKey == "" {
		return nil
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
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return err
	}
	if strings.TrimSpace(payload.AccessToken) == "" {
		return fmt.Errorf("runtime token refresh returned empty access_token")
	}
	p.apiKey = payload.AccessToken
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
	computerCallIDs := map[string]bool{}
	useNativeComputerReplay := p.openAIComputerMode() != "custom"
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			if useNativeComputerReplay && (computer.IsGeminiComputerAction(tc.Name) || isComputerUseTool(tc.Name)) {
				computerCallIDs[tc.ID] = true
			}
		}
	}

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

		// Tool results
		if len(m.ToolResults) > 0 {
			for _, tr := range m.ToolResults {
				if tr.Image != nil {
					imageURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(tr.Image)
					if computerCallIDs[tr.CallID] && p.openAIComputerMode() != "custom" {
						// Native OpenAI computer result.
						items = append(items, oaiInputItem{
							Type:   "computer_call_output",
							CallID: tr.CallID,
							Output: map[string]any{
								"type":      "computer_screenshot",
								"image_url": imageURL,
							},
						})
					} else {
						// Custom function-tool fallback: keep the screenshot in
						// the tool result so vision-capable OpenAI models can
						// inspect the page without the native computer tool.
						items = append(items, oaiInputItem{
							Type:   "function_call_output",
							CallID: tr.CallID,
							Output: []oaiContentBlock{
								{Type: "input_text", Text: tr.Content},
								{Type: "input_image", ImageURL: imageURL, Detail: "original"},
							},
						})
					}
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
				if useNativeComputerReplay && (computer.IsGeminiComputerAction(tc.Name) || isComputerUseTool(tc.Name)) {
					// Computer call — reconstruct as computer_call item
					argsAny := make(map[string]any, len(tc.Args))
					for k, v := range tc.Args {
						argsAny[k] = v
					}
					items = append(items, oaiInputItem{
						Type:    "computer_call",
						CallID:  tc.ID,
						Status:  "completed",
						Actions: []map[string]any{argsAny},
					})
				} else {
					argsJSON, _ := json.Marshal(tc.Args)
					items = append(items, oaiInputItem{
						Type:      "function_call",
						CallID:    tc.ID,
						Name:      tc.Name,
						Arguments: string(argsJSON),
					})
				}
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

	// Track pending items
	type pendingFunc struct {
		id         string
		name       string
		args       strings.Builder
		pendingBuf strings.Builder
	}
	pendingFuncs := map[string]*pendingFunc{} // by item ID

	// Track computer calls
	type pendingComputer struct {
		callID  string
		actions []oaiComputerAction
	}
	var currentComputer *pendingComputer

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

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

		// Output item added (function_call, computer_call, message)
		case "response.output_item.added":
			var item oaiOutputItem
			json.Unmarshal(event.Item, &item)

			switch item.Type {
			case "function_call":
				pendingFuncs[item.ID] = &pendingFunc{id: item.CallID, name: item.Name}
			case "computer_call":
				currentComputer = &pendingComputer{callID: item.CallID}
			}

		// Output item done — finalize
		case "response.output_item.done":
			var item oaiOutputItem
			json.Unmarshal(event.Item, &item)

			switch item.Type {
			case "function_call":
				pf, ok := pendingFuncs[item.ID]
				if ok {
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
						ID:   pf.id,
						Name: pf.name,
						Args: args,
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

			case "computer_call":
				if currentComputer != nil {
					// Parse actions from the completed item
					var actions []oaiComputerAction
					if len(item.Actions) > 0 {
						json.Unmarshal(item.Actions, &actions)
					}
					if len(actions) == 0 && len(item.Action) > 0 {
						var action oaiComputerAction
						if json.Unmarshal(item.Action, &action) == nil && action.Type != "" {
							actions = append(actions, action)
						}
					}

					// Convert each action to a NativeToolCall
					for i, action := range actions {
						args := oaiActionToArgs(action)
						callID := currentComputer.callID
						if i > 0 {
							callID = fmt.Sprintf("%s_%d", callID, i)
						}
						toolCalls = append(toolCalls, NativeToolCall{
							ID:   callID,
							Name: "computer_use",
							Args: args,
						})
					}
					currentComputer = nil
				}
			}

		// Usage
		case "response.completed":
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
							CachedTokens int `json:"cached_tokens"`
						} `json:"input_tokens_details"`
					} `json:"usage"`
				} `json:"response"`
			}
			json.Unmarshal([]byte(data), &completed)
			usage.PromptTokens = completed.Response.Usage.InputTokens
			usage.CompletionTokens = completed.Response.Usage.OutputTokens
			usage.CachedTokens = completed.Response.Usage.InputDetails.CachedTokens
		}
	}

	response := full.String()
	logMsg("OPENAI-NATIVE", fmt.Sprintf("done tokens_in=%d tokens_out=%d tools=%d len=%d", usage.PromptTokens, usage.CompletionTokens, len(toolCalls), len(response)))
	return ChatResponse{Text: response, ToolCalls: toolCalls, Reasoning: fullReasoning.String(), Usage: usage}, nil
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
		"%s item_meta type=%q status=%q id_present=%t call_id_present=%t name=%q content_count=%d content_types=%v content_text_chars=%d summary_count=%d summary_types=%v summary_text_chars=%d arguments_chars=%d actions_bytes=%d",
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
		len(item.Actions),
	))
}

// oaiActionToArgs converts an OpenAI computer action to our standard args map.
func oaiActionToArgs(a oaiComputerAction) map[string]string {
	args := map[string]string{"action": a.Type}
	switch a.Type {
	case "click", "double_click", "move":
		args["coordinate"] = fmt.Sprintf("[%d, %d]", a.X, a.Y)
		if a.Button != "" {
			args["button"] = a.Button
		}
	case "type":
		args["text"] = a.Text
	case "keypress":
		args["key"] = a.Key
	case "scroll":
		args["coordinate"] = fmt.Sprintf("[%d, %d]", a.X, a.Y)
		args["direction"] = "down"
		if a.ScrollY < 0 {
			args["direction"] = "up"
		}
		amount := a.ScrollY
		if amount < 0 {
			amount = -amount
		}
		if amount == 0 {
			amount = 3
		}
		args["amount"] = fmt.Sprintf("%d", amount)
	case "drag":
		args["coordinate"] = fmt.Sprintf("[%d, %d]", a.StartX, a.StartY)
		args["end_coordinate"] = fmt.Sprintf("[%d, %d]", a.EndX, a.EndY)
	case "wait":
		dur := a.Duration
		if dur == 0 {
			dur = 1000
		}
		args["duration"] = fmt.Sprintf("%d", dur)
	case "screenshot":
		// No extra args needed
	}
	return args
}
