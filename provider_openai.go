package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// OpenAICompatProvider works with any OpenAI-compatible API:
// Fireworks, OpenAI, Ollama, Together, Groq, etc.
type OpenAICompatProvider struct {
	name       string
	apiKey     string
	url        string
	models     map[ModelTier]string
	inputCost  float64 // per 1M tokens
	cachedCost float64
	outputCost float64
	authHeader string // "Bearer" or empty for no auth (Ollama)
	cacheMu    sync.RWMutex
	cacheOff   bool
}

// openAICompatRequestOptions carries provider-specific additions through the
// shared Chat Completions transport. Keeping these options request-scoped lets
// thin wrappers such as XAIProvider reuse all message, tool-call, streaming,
// and usage parsing without teaching every compatible provider about xAI.
type openAICompatRequestOptions struct {
	Fields         map[string]any
	Headers        map[string]string
	OptionalFields map[string]openAICompatOptionalField
}

// openAICompatOptionalField is a provider-specific request field that can be
// removed safely when a model rejects it. The transport retries the identical
// prepared turn without the field and lets the provider wrapper remember the
// model-specific incompatibility for later calls.
type openAICompatOptionalField struct {
	Value         any
	OnUnsupported func()
}

type openAICompatRequestOptionsContextKey struct{}

func withOpenAICompatRequestOptions(ctx context.Context, options openAICompatRequestOptions) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, openAICompatRequestOptionsContextKey{}, options)
}

func openAICompatRequestOptionsFromContext(ctx context.Context) openAICompatRequestOptions {
	if ctx == nil {
		return openAICompatRequestOptions{}
	}
	options, _ := ctx.Value(openAICompatRequestOptionsContextKey{}).(openAICompatRequestOptions)
	return options
}

func openAIOptionalFieldUnsupported(status int, responseBody, field string) bool {
	if status != http.StatusBadRequest && status != http.StatusUnprocessableEntity {
		return false
	}
	body := strings.ToLower(responseBody)
	field = strings.ToLower(strings.TrimSpace(field))
	if field == "" || !strings.Contains(body, field) {
		return false
	}
	for _, marker := range []string{
		"unsupported", "not supported", "does not support", "unknown",
		"unrecognized", "invalid", "only supports", "must be", "expected",
	} {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
}

func (p *OpenAICompatProvider) promptCacheHintsEnabled() bool {
	p.cacheMu.RLock()
	defer p.cacheMu.RUnlock()
	return !p.cacheOff
}

func (p *OpenAICompatProvider) disablePromptCacheHints() {
	p.cacheMu.Lock()
	p.cacheOff = true
	p.cacheMu.Unlock()
}

func (p *OpenAICompatProvider) Name() string                 { return p.name }
func (p *OpenAICompatProvider) Models() map[ModelTier]string { return p.models }
func (p *OpenAICompatProvider) CostPer1M() (float64, float64, float64) {
	return p.inputCost, p.cachedCost, p.outputCost
}
func (p *OpenAICompatProvider) SupportsNativeTools() bool {
	// All OpenAI-compatible Chat Completions endpoints accept the
	// `tools` field. Ollama is the lone exception in practice — tool
	// support is gated per-model and most local models don't honor it
	// reliably, so we keep our prompt-level fallback there. Anything
	// else (OpenAI, Fireworks, OpenCode Go, NVIDIA NIM, Together,
	// Groq, …) gets native tool calls.
	return p.name != "ollama"
}

func (p *OpenAICompatProvider) AvailableBuiltinTools() []BuiltinTool {
	if p.name == "openai" {
		return []BuiltinTool{
			{Type: "code_interpreter", Name: "code_interpreter"},
		}
	}
	return nil
}

func (p *OpenAICompatProvider) SetBuiltinTools(tools []string) {
	// OpenAI built-in tools handled via Responses API, not Chat Completions
	// Placeholder for future support
}

func (p *OpenAICompatProvider) WithBuiltins(builtins []string) LLMProvider {
	return p // OpenAI compat providers don't use builtins in Chat Completions
}

// openaiMessage serializes a Message for the OpenAI API.
// When Parts is set, content becomes the array (native format).
type openaiMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []ContentPart
}

// convertAudioURLParts converts audio_url parts to input_audio (OpenAI format).
func convertAudioURLParts(parts []ContentPart) []ContentPart {
	var out []ContentPart
	for _, p := range parts {
		if p.Type == "audio_url" && p.AudioURL != nil {
			if strings.HasPrefix(p.AudioURL.URL, "data:") {
				data, mime := parseDataURI(p.AudioURL.URL)
				format := "wav"
				if strings.Contains(mime, "mp3") || strings.Contains(mime, "mpeg") {
					format = "mp3"
				}
				out = append(out, ContentPart{Type: "input_audio", InputAudio: &InputAudio{Data: data, Format: format}})
			} else {
				// Fetch and convert
				b64, mime, err := fetchMediaAsBase64(p.AudioURL.URL)
				if err != nil {
					logMsg("OPENAI", fmt.Sprintf("audio fetch error: %v", err))
					out = append(out, ContentPart{Type: "text", Text: fmt.Sprintf("[audio fetch failed: %s]", p.AudioURL.URL)})
				} else {
					format := "wav"
					if strings.Contains(mime, "mp3") || strings.Contains(mime, "mpeg") {
						format = "mp3"
					}
					out = append(out, ContentPart{Type: "input_audio", InputAudio: &InputAudio{Data: b64, Format: format}})
				}
			}
		} else {
			out = append(out, p)
		}
	}
	return out
}

// openaiToolCallDelta tracks streaming tool call assembly.
type openaiToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function *struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function,omitempty"`
}

// openaiToolDef is the OpenAI tool format for the request.
type openaiToolDef struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
		Strict      bool           `json:"strict,omitempty"`
	} `json:"function"`
}

// openaiToolResultMsg is a tool result message.
type openaiToolResultMsg struct {
	Role       string `json:"role"` // "tool"
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

func toOpenAIMessages(messages []Message) []any {
	var out []any
	for _, m := range messages {
		// Tool result messages
		if len(m.ToolResults) > 0 {
			for _, tr := range m.ToolResults {
				if tr.Image != nil {
					// Tool result with image (screenshot) — send as multimodal content
					// Use "original" detail for computer use to preserve full resolution
					out = append(out, map[string]any{
						"role":         "tool",
						"tool_call_id": tr.CallID,
						"content": []map[string]any{
							{"type": "text", "text": tr.Content},
							{"type": "image_url", "image_url": map[string]any{
								"url":    "data:image/png;base64," + base64Encode(tr.Image),
								"detail": "original",
							}},
						},
					})
				} else {
					out = append(out, openaiToolResultMsg{
						Role:       "tool",
						Content:    tr.Content,
						ToolCallID: tr.CallID,
					})
				}
			}
			continue
		}

		// Assistant message with tool calls.
		//
		// `content` is ALWAYS included, even as empty string. The
		// OpenAI spec allows omitting it when tool_calls is present,
		// and OpenAI/Fireworks accept that — but Moonshot Kimi K2.6
		// (which OpenCode Go proxies for the kimi-k2.6 slug) rejects
		// the message with HTTP 400 unless `content` is on the wire.
		// Empty string is interop-safe across every backend we've
		// tested (OpenAI, Fireworks, Moonshot, NVIDIA NIM, OpenRouter,
		// Together, Groq); `null` is not (older NIM builds reject it).
		//
		// `reasoning_content` is included when the message carries
		// reasoning captured from the prior turn. Moonshot with
		// thinking enabled requires it on assistant tool_call
		// messages — without it the next request 400s with
		// "thinking is enabled but reasoning_content is missing in
		// assistant tool call message". Other backends ignore the
		// field (it's a known reasoning-model extension).
		if len(m.ToolCalls) > 0 {
			toolCalls := make([]map[string]any, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Args)
				toolCalls[i] = map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": string(argsJSON),
					},
				}
			}
			msg := map[string]any{
				"role":       "assistant",
				"content":    m.Content, // always present; "" when only tool_calls
				"tool_calls": toolCalls,
			}
			if m.Reasoning != "" {
				msg["reasoning_content"] = m.Reasoning
			}
			out = append(out, msg)
			continue
		}

		// Regular message.
		//
		// Skip assistant turns that have nothing to say AND no tool
		// calls AND no parts. Those are dead-air entries (e.g. left
		// behind when an upstream Chat() errored after the message was
		// already appended) and Moonshot rejects them with HTTP 400
		// "Invalid request: the message at position N with role
		// 'assistant' must not be empty" — which then poisons every
		// subsequent iteration. User and system messages are kept even
		// when empty (rare but legitimate signals like "[admin]"
		// directives).
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) == 0 && !m.HasParts() {
			continue
		}
		if m.HasParts() {
			out = append(out, openaiMessage{Role: m.Role, Content: convertAudioURLParts(m.Parts)})
		} else {
			out = append(out, openaiMessage{Role: m.Role, Content: m.Content})
		}
	}
	return out
}

func (p *OpenAICompatProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	callStart := time.Now()
	timing := ProviderTiming{}
	elapsedMs := func() int64 {
		return time.Since(callStart).Milliseconds()
	}
	markMs := func() *int64 {
		ms := elapsedMs()
		return &ms
	}
	failed := func(phase string, err error) (ChatResponse, error) {
		timing.CompletionMs = elapsedMs()
		timing.TerminalPhase = phase
		return ChatResponse{ProviderTiming: timing}, err
	}

	requestOptions := openAICompatRequestOptionsFromContext(ctx)
	// Build request
	openAIMessages := toOpenAIMessages(messages)
	reqMap := map[string]any{
		"model":    model,
		"messages": openAIMessages,
		"stream":   true,
	}
	// OpenAI supports stream_options for usage in streaming; Fireworks may not
	if p.name == "openai" || p.name == "managed" {
		reqMap["stream_options"] = map[string]any{"include_usage": true}
	}
	// Add tools if provider supports them.
	if len(tools) > 0 && p.SupportsNativeTools() {
		var defs []openaiToolDef
		for _, t := range tools {
			def := openaiToolDef{Type: "function"}
			def.Function.Name = t.Name
			def.Function.Description = t.Description
			def.Function.Parameters = t.Parameters
			// Note: Strict mode not supported by all providers (Fireworks ignores it)
			defs = append(defs, def)
		}
		reqMap["tools"] = defs
	}
	for key, value := range requestOptions.Fields {
		reqMap[key] = value
	}
	for key, field := range requestOptions.OptionalFields {
		reqMap[key] = field.Value
	}
	if hints := openAIPromptCacheHintsForScope(p.name, model, openAIPromptCacheStablePrefix(openAIMessages), reqMap["tools"], openAIPromptCacheScopeFromContext(ctx)); hints.Key != "" && p.promptCacheHintsEnabled() {
		reqMap["prompt_cache_key"] = hints.Key
		reqMap["prompt_cache_retention"] = hints.Retention
	}

	// Log message count and types for debugging
	if msgs, ok := reqMap["messages"].([]any); ok {
		for i, m := range msgs {
			switch v := m.(type) {
			case map[string]any:
				if v["role"] == "tool" {
					logMsg("OPENAI", fmt.Sprintf("msg[%d] role=tool call_id=%v content_type=%T", i, v["tool_call_id"], v["content"]))
				}
			}
		}
	}

	doRequest := func(payload []byte) (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, "POST", p.url, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if p.apiKey != "" && p.authHeader != "" {
			req.Header.Set("Authorization", p.authHeader+" "+p.apiKey)
		}
		for key, value := range requestOptions.Headers {
			if strings.TrimSpace(key) != "" {
				req.Header.Set(key, value)
			}
		}
		return llmHTTPClient.Do(req)
	}

	var resp *http.Response
	var reqIDs map[string]string
	for {
		body, err := json.Marshal(reqMap)
		if err != nil {
			return failed("request_encode", err)
		}
		timing.RequestAttempts++
		resp, err = doRequest(body)
		if err != nil {
			return failed("response_headers", err)
		}
		timing.ResponseHeadersMs = markMs()

		// Capture provider-side request identifiers so a future stall /
		// hang can be cross-referenced with the provider's own logs without
		// another round-trip. Different vendors use different header names
		// (Fireworks ships x-request-id; some return x-fw-request-id). Log
		// whatever we find.
		reqIDs = extractProviderRequestIDs(resp.Header)
		timing.ProviderRequestIDs = reqIDs
		if len(reqIDs) > 0 {
			logMsg("PROVIDER", fmt.Sprintf("model=%s request_ids=%v", model, reqIDs))
		}
		if resp.StatusCode == http.StatusOK {
			break
		}

		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if _, ok := reqMap["prompt_cache_key"]; ok && openAICacheHintsUnsupported(resp.StatusCode, string(respBody)) {
			p.disablePromptCacheHints()
			delete(reqMap, "prompt_cache_key")
			delete(reqMap, "prompt_cache_retention")
			logMsg("OPENAI", fmt.Sprintf("cache hints unsupported, retrying without them: %d %s", resp.StatusCode, string(respBody)))
			continue
		}

		retriedOptional := false
		for key, field := range requestOptions.OptionalFields {
			if _, present := reqMap[key]; !present || !openAIOptionalFieldUnsupported(resp.StatusCode, string(respBody), key) {
				continue
			}
			delete(reqMap, key)
			if field.OnUnsupported != nil {
				field.OnUnsupported()
			}
			logMsg("OPENAI", fmt.Sprintf("optional field %s unsupported for model=%s, retrying prepared turn without it: %d %s", key, model, resp.StatusCode, string(respBody)))
			retriedOptional = true
			break
		}
		if retriedOptional {
			continue
		}
		return failed("http_status", fmt.Errorf("API error %d: %s (request_ids=%v)", resp.StatusCode, string(respBody), reqIDs))
	}

	// Wrap the streaming body in an idle-read monitor. Any pause longer
	// than streamIdleTimeout without a single byte arriving is treated
	// as a provider stall — we close the body so the scanner unblocks
	// with an error, and the caller returns ErrStreamIdleTimeout (with
	// request_ids folded in) so the think loop can retry.
	idleBody := newIdleReader(resp.Body, streamIdleTimeout(), func() {
		logMsg("FIREWORKS-STALL", fmt.Sprintf("stream idle for %s on model=%s request_ids=%v — aborting",
			streamIdleTimeout(), model, reqIDs))
	})
	resp.Body = idleBody
	defer resp.Body.Close()

	var full strings.Builder
	// Accumulate reasoning chunks too. We forward each one through
	// onThinking for live UI rendering, but also capture the full
	// transcript so the caller can write it back onto the assistant
	// Message it appends — Moonshot via OpenCode Go requires the
	// `reasoning_content` field on the next-turn assistant tool_call
	// message, otherwise it 400s with "thinking is enabled but
	// reasoning_content is missing".
	var fullReasoning strings.Builder
	var usage TokenUsage
	// Track streamed tool calls by index
	pendingTools := make(map[int]*struct {
		id         string
		name       string
		argsJSON   strings.Builder
		pendingBuf strings.Builder
	})

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		timing.StreamChunks++
		if timing.FirstChunkMs == nil {
			timing.FirstChunkMs = markMs()
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		// Debug: dump the raw delta so we can see what fields Fireworks is
		// actually returning (reasoning_content, thinking, etc.). Enable via
		// APTEVA_DUMP_STREAM=1 to avoid log spam in production.
		if os.Getenv("APTEVA_DUMP_STREAM") == "1" {
			logMsg("OPENAI-STREAM", data)
		}

		var event struct {
			Error *struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    any    `json:"code"`
			} `json:"error,omitempty"`
			Choices []struct {
				Delta struct {
					Content          string                `json:"content"`
					ReasoningContent string                `json:"reasoning_content,omitempty"`
					Reasoning        string                `json:"reasoning,omitempty"`
					ToolCalls        []openaiToolCallDelta `json:"tool_calls,omitempty"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *Usage `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if event.Error != nil {
			return failed("stream_error", fmt.Errorf("provider stream error (%s/%v): %s", event.Error.Type, event.Error.Code, event.Error.Message))
		}
		if len(event.Choices) > 0 {
			delta := event.Choices[0].Delta
			// Reasoning chain-of-thought arrives under different field
			// names depending on the gateway:
			//   reasoning_content — Fireworks, DeepSeek, OpenAI o-series
			//   reasoning         — OpenRouter-style proxies (OpenCode Go)
			// Both go through onThinking (not onChunk) so the UI can
			// distinguish reasoning from output, and neither is appended
			// to `full` (the visible answer).
			reasoning := delta.ReasoningContent
			if reasoning == "" {
				reasoning = delta.Reasoning
			}
			if reasoning != "" {
				fullReasoning.WriteString(reasoning)
				if onThinking != nil {
					onThinking(reasoning)
				}
			}
			if delta.Content != "" {
				full.WriteString(delta.Content)
				if onChunk != nil {
					onChunk(delta.Content)
				}
			}
			for _, tc := range delta.ToolCalls {
				if timing.FirstToolCallMs == nil {
					timing.FirstToolCallMs = markMs()
				}
				pt, ok := pendingTools[tc.Index]
				if !ok {
					pt = &struct {
						id         string
						name       string
						argsJSON   strings.Builder
						pendingBuf strings.Builder // chunks accumulated before tc.ID arrived
					}{}
					pendingTools[tc.Index] = pt
				}
				if tc.ID != "" {
					pt.id = tc.ID
				}
				if tc.Function != nil {
					if tc.Function.Name != "" {
						pt.name = tc.Function.Name
					}
					if tc.Function.Arguments != "" {
						pt.argsJSON.WriteString(tc.Function.Arguments)
						// Only emit chunks once pt.id is known so the call_id
						// on every llm.tool_chunk event matches the eventual
						// tool.call (both use the upstream provider id). Using
						// an index-based fallback split streaming rows in the
						// dashboard because the fallback and the real id were
						// different strings.
						if pt.id == "" {
							pt.pendingBuf.WriteString(tc.Function.Arguments)
						} else if onToolChunk != nil && pt.name != "" {
							if pt.pendingBuf.Len() > 0 {
								onToolChunk(pt.name, pt.id, pt.pendingBuf.String())
								pt.pendingBuf.Reset()
							}
							onToolChunk(pt.name, pt.id, tc.Function.Arguments)
						}
					}
				}
			}
		}
		if event.Usage != nil {
			usage.PromptTokens = event.Usage.PromptTokens
			usage.CompletionTokens = event.Usage.CompletionTokens
			if event.Usage.PromptTokensDetails != nil {
				usage.CachedTokens = event.Usage.PromptTokensDetails.CachedTokens
				usage.CacheWriteTokens = event.Usage.PromptTokensDetails.CacheWriteTokens
			}
		}
	}

	// Surface any scanner error (stall, I/O error) with the request IDs
	// attached so the operator can cross-reference with the provider's
	// own logs. Idle-timeout stalls get a dedicated sentinel so callers
	// (and tests) can tell them apart from transport errors.
	if scerr := scanner.Err(); scerr != nil {
		if errors.Is(scerr, ErrStreamIdleTimeout) {
			return failed("stream_read", fmt.Errorf("%w (model=%s request_ids=%v)", ErrStreamIdleTimeout, model, reqIDs))
		}
		return failed("stream_read", fmt.Errorf("stream read error: %w (model=%s request_ids=%v)", scerr, model, reqIDs))
	}

	// Assemble completed tool calls
	var toolCalls []NativeToolCall
	for i := 0; i < len(pendingTools); i++ {
		pt, ok := pendingTools[i]
		if !ok {
			continue
		}
		// Flush any chunks buffered before pt.id arrived. By now the stream
		// is done so pt.id is either set (happy path — flush under real id)
		// or never arrived (pathological provider — nothing to emit since
		// the dashboard has no way to match anyway).
		if onToolChunk != nil && pt.id != "" && pt.pendingBuf.Len() > 0 && pt.name != "" {
			onToolChunk(pt.name, pt.id, pt.pendingBuf.String())
			pt.pendingBuf.Reset()
		}
		args := make(map[string]string)
		var raw map[string]any
		if err := json.Unmarshal([]byte(pt.argsJSON.String()), &raw); err == nil {
			for k, v := range raw {
				switch v.(type) {
				case string:
					args[k] = v.(string)
				default:
					// Preserve arrays/objects/numbers as JSON strings
					b, _ := json.Marshal(v)
					args[k] = string(b)
				}
			}
		}
		toolCalls = append(toolCalls, NativeToolCall{
			ID:   pt.id,
			Name: pt.name,
			Args: args,
		})
	}

	timing.CompletionMs = elapsedMs()
	timing.TerminalPhase = "completed"
	return ChatResponse{
		Text:           full.String(),
		Reasoning:      fullReasoning.String(),
		ToolCalls:      toolCalls,
		Usage:          usage,
		ProviderTiming: timing,
	}, nil
}

// --- Factory functions ---

// NewManagedProvider talks only to the local Apteva Server gateway. The token
// is the core process's row-scoped credential, not an upstream provider key;
// the server resolves the platform-managed connection and applies policy.
func NewManagedProvider(endpoint, token string) LLMProvider {
	return &OpenAICompatProvider{
		name: "managed", apiKey: token, url: endpoint, authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge: "managed", ModelMedium: "managed", ModelSmall: "managed",
		},
	}
}

func NewFireworksProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "fireworks",
		apiKey:     apiKey,
		url:        "https://api.fireworks.ai/inference/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge:  "accounts/fireworks/models/kimi-k2p6",
			ModelMedium: "accounts/fireworks/models/kimi-k2p6",
			ModelSmall:  "accounts/fireworks/models/kimi-k2p6",
		},
		inputCost:  0.60,
		cachedCost: 0.10,
		outputCost: 3.00,
	}
}

// NewOpenCodeGoProvider — flat-rate subscription gateway from
// opencode.ai/go that fronts the same Kimi K2.6 we use via Fireworks
// plus Qwen / GLM / MiMo variants under one OpenAI-compatible endpoint.
//
// Per-token costs are placeholders (0/0/0) because OpenCode Go bills
// per subscription, not per call — the server's model_fetch.go pricing
// table reports the same so the dashboard's per-call $ figure stays
// blank rather than misleadingly nonzero.
//
// Defaults: glm-5.2 across all three tiers. With a flat-rate plan the
// per-iteration cost incentive for choosing weaker small/medium models does
// not apply, so the agent uses the same capable model end-to-end. Users can
// still override individual tiers in the dashboard provider settings.
func NewOpenCodeGoProvider(apiKey string) LLMProvider {
	return &OpenCodeGoProvider{
		compat: &OpenAICompatProvider{
			name:       "opencode-go",
			apiKey:     apiKey,
			url:        "https://opencode.ai/zen/go/v1/chat/completions",
			authHeader: "Bearer",
			models: map[ModelTier]string{
				ModelLarge:  "glm-5.2",
				ModelMedium: "glm-5.2",
				ModelSmall:  "glm-5.2",
			},
			inputCost:  0,
			cachedCost: 0,
			outputCost: 0,
		},
		support: newOpenCodeGoReasoningSupport(),
	}
}

func NewOpenAIProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "openai",
		apiKey:     apiKey,
		url:        "https://api.openai.com/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge:  "gpt-4.1",
			ModelMedium: "gpt-4.1-mini",
			ModelSmall:  "gpt-4.1-nano",
		},
		inputCost:  2.50,
		cachedCost: 1.25,
		outputCost: 10.00,
	}
}

// NewNvidiaProvider wires up NVIDIA's NIM hosted catalog. They expose an
// OpenAI-compatible Chat Completions endpoint at integrate.api.nvidia.com,
// so we reuse OpenAICompatProvider verbatim — just the base URL and default
// model slugs are NVIDIA-specific. Pricing is left at zero by default because
// NIM billing is account-scoped rather than per-token-listed, so cost
// projections in the dashboard stats bar will stay at $0 unless the user
// manually overrides these costs via config.
func NewNvidiaProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "nvidia",
		apiKey:     apiKey,
		url:        "https://integrate.api.nvidia.com/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			// Defaults picked from NVIDIA's public NIM catalog. Users will
			// typically override via Config.Providers[].Models on the
			// dashboard settings page.
			ModelLarge:  "nvidia/llama-3.1-nemotron-70b-instruct",
			ModelMedium: "meta/llama-3.1-70b-instruct",
			ModelSmall:  "meta/llama-3.1-8b-instruct",
		},
		// NIM pricing is account-plan dependent — leave the per-token cost
		// at 0 so calculateCostForProvider() returns 0 instead of a
		// misleading number. Users wanting real cost tracking can edit
		// the struct directly from a fork or set costs via a future
		// config field.
		inputCost:  0,
		cachedCost: 0,
		outputCost: 0,
	}
}

// NewVeniceProvider — privacy-focused inference gateway at venice.ai.
// OpenAI-compatible /chat/completions endpoint, large rotating catalog
// (Llama, Qwen, GLM, Mistral, plus Claude/Grok/Gemini reseller variants).
// Pricing varies per model and is set per-account in their dashboard, so
// per-token costs are left at zero here — the model picker / picker
// dropdown is the source of truth for what's available right now.
func NewVeniceProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "venice",
		apiKey:     apiKey,
		url:        "https://api.venice.ai/api/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge:  "qwen3-coder-480b-a35b-instruct",
			ModelMedium: "qwen3-6-27b",
			ModelSmall:  "mistral-small-3-2-24b-instruct",
		},
		inputCost:  0,
		cachedCost: 0,
		outputCost: 0,
	}
}

func NewOllamaProvider(host string) LLMProvider {
	url := strings.TrimRight(host, "/") + "/v1/chat/completions"
	model := strings.TrimSpace(os.Getenv("OLLAMA_MODEL"))
	if model == "" {
		model = "llama3.1"
	}
	return &OpenAICompatProvider{
		name:       "ollama",
		apiKey:     "",
		url:        url,
		authHeader: "",
		models: map[ModelTier]string{
			ModelLarge:  model,
			ModelMedium: model,
			ModelSmall:  model,
		},
		inputCost:  0,
		cachedCost: 0,
		outputCost: 0,
	}
}
