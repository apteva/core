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
	"reflect"
	"strings"
)

// sanitizeToolID ensures tool call IDs only contain characters Anthropic accepts: [a-zA-Z0-9_-]
// toMap converts a struct to map[string]any via JSON round-trip.
func toMap(v any) (map[string]any, bool) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, false
	}
	var m map[string]any
	if json.Unmarshal(data, &m) != nil {
		return nil, false
	}
	return m, true
}

// applyMsgCacheControl tags the last text block of an anthropic message
// with cache_control: ephemeral. Used to set a cache breakpoint at the
// most recent turn so subsequent requests can prefix-match the entire
// conversation up through it.
//
// Handles three content shapes:
//   - string content: rewrite into a single text block carrying cache
//   - []anthropicContentBlock: tag the last text block (or fall through
//     to a structural rewrite if there isn't one to tag)
//   - any other slice (e.g. []map[string]any from JSON round-tripping):
//     no-op, the conversation tail will simply not be cached and the
//     existing tools+system breakpoints still apply.
func applyMsgCacheControl(m *anthropicMessage) {
	if m == nil {
		return
	}
	switch c := m.Content.(type) {
	case string:
		m.Content = []anthropicContentBlock{
			{Type: "text", Text: c, CacheControl: &anthropicCacheControl{Type: "ephemeral"}},
		}
	case []anthropicContentBlock:
		// Tag the last text block — doing so on a tool_use/tool_result
		// block is invalid; if there's no text block, leave it alone.
		for i := len(c) - 1; i >= 0; i-- {
			if c[i].Type == "text" {
				c[i].CacheControl = &anthropicCacheControl{Type: "ephemeral"}
				m.Content = c
				return
			}
		}
	}
}

func sanitizeToolID(id string) string {
	var b strings.Builder
	for _, c := range id {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			b.WriteRune(c)
		} else {
			b.WriteRune('_')
		}
	}
	s := b.String()
	if s == "" {
		s = "tool_0"
	}
	return s
}

type AnthropicProvider struct {
	apiKey          string
	url             string
	models          map[ModelTier]string
	builtinTools    []string // enabled built-in tools: "code_execution", "web_search"
	builtinConfigs  map[string]BuiltinToolConfig
	maxOutputTokens int
	reasoning       ReasoningSettings
}

func NewAnthropicProvider(apiKey string) LLMProvider {
	return &AnthropicProvider{
		apiKey: apiKey,
		url:    "https://api.anthropic.com/v1/messages",
		models: map[ModelTier]string{
			ModelLarge:  "claude-opus-5-5",
			ModelMedium: "claude-sonnet-5-5",
			ModelSmall:  "claude-haiku-5-5",
		},
	}
}

// anthropicPricing is first-party API pricing per 1M tokens (input, cache
// read, output). Haiku 5.5 is the up-to-100K-prompt rate.
var anthropicPricing = map[string][3]float64{
	"claude-fable-5-1":  {10.00, 0.25, 50.00},
	"claude-fable-5":    {10.00, 1.00, 50.00},
	"claude-opus-5-5":   {4.00, 0.20, 20.00},
	"claude-opus-5":     {5.00, 0.50, 25.00},
	"claude-opus-4-8":   {5.00, 0.50, 25.00},
	"claude-opus-4-7":   {5.00, 0.50, 25.00},
	"claude-opus-4-6":   {5.00, 0.50, 25.00},
	"claude-sonnet-5-5": {2.00, 0.20, 10.00},
	"claude-sonnet-5":   {2.00, 0.20, 10.00},
	"claude-sonnet-4-6": {3.00, 0.30, 15.00},
	"claude-haiku-5-5":  {0.10, 0.01, 0.50},
	"claude-haiku-4-5":  {1.00, 0.10, 5.00},
}

// anthropicModelKey strips a trailing date snapshot (claude-haiku-4-5-20251001).
func anthropicModelKey(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if i := strings.LastIndex(model, "-"); i > 0 && len(model)-i-1 == 8 && strings.Trim(model[i+1:], "0123456789") == "" {
		return model[:i]
	}
	return model
}

func (p *AnthropicProvider) Name() string                 { return "anthropic" }
func (p *AnthropicProvider) Models() map[ModelTier]string { return p.models }
func (p *AnthropicProvider) CostPer1M() (float64, float64, float64) {
	if price, ok := anthropicPricing[anthropicModelKey(p.models[ModelLarge])]; ok {
		return price[0], price[1], price[2]
	}
	return 3.00, 0.30, 15.00
}
func (p *AnthropicProvider) SupportsNativeTools() bool { return true }

func (p *AnthropicProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	clone := *p
	clone.reasoning = settings
	return &clone
}

// anthropicGeneration returns the family and major version of a Claude model
// ID ("opus", 5 for claude-opus-5-5), or "", 0 when it isn't one.
func anthropicGeneration(model string) (string, int) {
	parts := strings.Split(anthropicModelKey(model), "-")
	if len(parts) < 3 || parts[0] != "claude" {
		return "", 0
	}
	major := 0
	for _, c := range parts[2] {
		if c < '0' || c > '9' {
			return "", 0
		}
		major = major*10 + int(c-'0')
	}
	return parts[1], major
}

// anthropicAlwaysThinks reports models where thinking runs by default (all
// Claude 5.x). Their thinking blocks must be replayed in tool loops.
func anthropicAlwaysThinks(model string) bool {
	family, major := anthropicGeneration(model)
	return major >= 5 && family != ""
}

// anthropicEffort maps an agent reasoning level to output_config.effort, or ""
// to leave the model default. Claude 5.x cannot turn thinking off, so none and
// minimal become low. Effort is only sent to models that accept it.
func anthropicEffort(level ReasoningLevel, model string) string {
	family, major := anthropicGeneration(model)
	key := anthropicModelKey(model)
	is46 := key == "claude-opus-4-6" || key == "claude-sonnet-4-6"
	if major < 5 && !is46 && key != "claude-opus-4-7" && key != "claude-opus-4-8" {
		return ""
	}
	if family == "" {
		return ""
	}
	switch normalizeReasoningLevel(level) {
	case ReasoningNone, ReasoningMinimal, ReasoningLow:
		return "low"
	case ReasoningMedium:
		return "medium"
	case ReasoningHigh:
		return "high"
	case ReasoningXHigh:
		if is46 {
			return "high"
		}
		return "xhigh"
	}
	return ""
}

func anthropicMaxTokens(model string) int {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(model)), "-")
	if len(parts) >= 3 && parts[0] == "claude" {
		switch parts[1] {
		case "fable", "haiku", "mythos", "opus", "sonnet":
			if parts[2] == "4" || parts[2] == "5" {
				return 64000
			}
		}
	}
	return 4096
}

func (p *AnthropicProvider) outputTokenLimit(model string) int {
	limit := p.maxOutputTokens
	if limit == 0 {
		limit = 16384
	}
	return min(limit, anthropicMaxTokens(model))
}

func (p *AnthropicProvider) AvailableBuiltinTools() []BuiltinTool {
	return []BuiltinTool{
		{Type: "code_execution_20250825", Name: "code_execution"},
		{Type: "web_search_20250305", Name: "web_search"},
	}
}

func (p *AnthropicProvider) SetBuiltinTools(tools []string) {
	p.builtinTools = tools
}

func (p *AnthropicProvider) WithBuiltins(builtins []string) LLMProvider {
	if builtins == nil {
		return p // nil = inherit all
	}
	clone := *p // shallow copy — shares apiKey, models, httpClient
	clone.builtinTools = append([]string(nil), builtins...)
	clone.builtinConfigs = narrowBuiltinConfigs(p.builtinConfigs, builtins)
	return &clone
}

// --- Request types ---

type anthropicRequest struct {
	Model        string                 `json:"model"`
	MaxTokens    int                    `json:"max_tokens"`
	Stream       bool                   `json:"stream"`
	System       any                    `json:"system,omitempty"` // string or []anthropicSystemBlock
	Messages     []anthropicMessage     `json:"messages"`
	Tools        []any                  `json:"tools,omitempty"` // mixed: anthropicTool or provider builtin maps
	Thinking     *anthropicThinking     `json:"thinking,omitempty"`
	OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
}

type anthropicThinking struct {
	Type         string                 `json:"type"` // "adaptive"
	BlockBinding *anthropicBlockBinding `json:"block_binding,omitempty"`
}

type anthropicBlockBinding struct {
	PrefixMismatchBehavior string `json:"prefix_mismatch_behavior"`
}

type anthropicOutputConfig struct {
	Effort string `json:"effort,omitempty"`
}

// Core rebuilds system prompts and compacts history, which invalidates earlier
// thinking blocks. drop_block lets those requests continue without the stale
// reasoning instead of failing with a 400.
const anthropicThinkingBindingBeta = "thinking-binding-controls-2026-08-01"

type anthropicSystemBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

type anthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  map[string]any         `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []anthropicContentBlock
}

type anthropicContentBlock struct {
	Type         string                 `json:"type"`                    // "text", "image", "tool_use", "tool_result"
	Text         string                 `json:"text,omitempty"`          // type=text
	Source       *anthropicSource       `json:"source,omitempty"`        // type=image
	ID           string                 `json:"id,omitempty"`            // type=tool_use
	Name         string                 `json:"name,omitempty"`          // type=tool_use
	Input        json.RawMessage        `json:"input,omitempty"`         // type=tool_use — use RawMessage to preserve empty {}
	ToolUseID    string                 `json:"tool_use_id,omitempty"`   // type=tool_result
	Content      any                    `json:"content,omitempty"`       // type=tool_result (string or blocks)
	IsError      bool                   `json:"is_error,omitempty"`      // type=tool_result
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"` // mark this block as a cache breakpoint
}

type anthropicSource struct {
	Type      string `json:"type"` // "url", "base64"
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// --- Streaming response types ---

type anthropicStreamEvent struct {
	Type         string               `json:"type"`
	Index        int                  `json:"index,omitempty"`
	Delta        *anthropicDelta      `json:"delta,omitempty"`
	ContentBlock *anthropicBlockStart `json:"content_block,omitempty"`
	Message      *struct {
		Usage *struct {
			InputTokens   int `json:"input_tokens"`
			OutputTokens  int `json:"output_tokens"`
			CacheRead     int `json:"cache_read_input_tokens"`
			CacheCreation int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	} `json:"message,omitempty"`
	Usage *struct {
		InputTokens   int `json:"input_tokens"`
		OutputTokens  int `json:"output_tokens"`
		CacheRead     int `json:"cache_read_input_tokens"`
		CacheCreation int `json:"cache_creation_input_tokens"`
	} `json:"usage,omitempty"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type anthropicDelta struct {
	Type         string `json:"type"`
	Text         string `json:"text,omitempty"`
	Thinking     string `json:"thinking,omitempty"`     // thinking_delta
	Signature    string `json:"signature,omitempty"`    // signature_delta
	PartialJSON  string `json:"partial_json,omitempty"` // for tool_use input streaming
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
}

type anthropicBlockStart struct {
	Type  string         `json:"type"` // "text", "thinking", "redacted_thinking", "tool_use", "server_tool_use", "code_execution_tool_result"
	Text  string         `json:"text,omitempty"`
	Data  string         `json:"data,omitempty"` // redacted_thinking
	ID    string         `json:"id,omitempty"`   // tool_use
	Name  string         `json:"name,omitempty"` // tool_use / server_tool_use
	Input map[string]any `json:"input"`          // tool_use (may be empty at start)
	// code_execution_tool_result fields
	Content []struct {
		Type   string `json:"type"`
		Text   string `json:"text,omitempty"`
		Output struct {
			Stdout     string `json:"stdout,omitempty"`
			Stderr     string `json:"stderr,omitempty"`
			ReturnCode int    `json:"return_code"`
		} `json:"output,omitempty"`
	} `json:"content,omitempty"`
}

// --- Chat implementation ---

func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	// Convert messages: extract system prompt, convert rest to Anthropic format
	var system string
	var anthropicMsgs []anthropicMessage
	for _, m := range messages {
		if m.Role == "system" {
			system = m.TextContent()
			continue
		}

		// Message with tool results
		if len(m.ToolResults) > 0 {
			var blocks []anthropicContentBlock
			for _, tr := range m.ToolResults {
				block := anthropicContentBlock{
					Type:      "tool_result",
					ToolUseID: sanitizeToolID(tr.CallID),
					IsError:   tr.IsError,
				}
				if tr.Image != nil {
					// Image result (screenshot etc.) — detect MIME from magic bytes
					mime := "image/png"
					if len(tr.Image) > 2 && tr.Image[0] == 0xFF && tr.Image[1] == 0xD8 {
						mime = "image/jpeg"
					}
					logMsg("ANTHROPIC", fmt.Sprintf("tool_result image: call_id=%s mime=%s size=%d bytes first_bytes=%x", tr.CallID, mime, len(tr.Image), tr.Image[:4]))
					block.Content = []anthropicContentBlock{
						{Type: "image", Source: &anthropicSource{
							Type:      "base64",
							MediaType: mime,
							Data:      base64.StdEncoding.EncodeToString(tr.Image),
						}},
					}
				} else {
					block.Content = tr.Content
				}
				blocks = append(blocks, block)
			}
			anthropicMsgs = append(anthropicMsgs, anthropicMessage{Role: "user", Content: blocks})
			continue
		}

		// Assistant turns that carried thinking go back exactly as produced.
		if replay := anthropicReplayContent(m); replay != nil {
			anthropicMsgs = append(anthropicMsgs, anthropicMessage{Role: "assistant", Content: replay})
			continue
		}

		// Message with tool calls (assistant)
		if len(m.ToolCalls) > 0 {
			var blocks []anthropicContentBlock
			if m.Content != "" {
				blocks = append(blocks, anthropicContentBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				inputJSON := toolCallArguments(tc)
				blocks = append(blocks, anthropicContentBlock{
					Type:  "tool_use",
					ID:    sanitizeToolID(tc.ID),
					Name:  tc.Name,
					Input: inputJSON,
				})
			}
			anthropicMsgs = append(anthropicMsgs, anthropicMessage{Role: "assistant", Content: blocks})
			continue
		}

		// Regular message
		if m.HasParts() {
			anthropicMsgs = append(anthropicMsgs, anthropicMessage{
				Role:    m.Role,
				Content: toAnthropicBlocks(m.Parts),
			})
		} else {
			anthropicMsgs = append(anthropicMsgs, anthropicMessage{
				Role:    m.Role,
				Content: m.Content,
			})
		}
	}

	if len(anthropicMsgs) == 0 {
		anthropicMsgs = append(anthropicMsgs, anthropicMessage{Role: "user", Content: "Begin."})
	}

	// Convert tools.
	var anthropicTools []any
	for _, t := range tools {
		anthropicTools = append(anthropicTools, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}

	anthropicTools = append(anthropicTools, p.configuredAnthropicBuiltins()...)

	// System prompt as cacheable block
	var systemContent any
	if system != "" {
		systemContent = []anthropicSystemBlock{
			{Type: "text", Text: system, CacheControl: &anthropicCacheControl{Type: "ephemeral"}},
		}
	}

	// Mark last tool as cacheable (tools rarely change between calls)
	// Must handle all tool types: anthropicTool, map[string]string (builtin), AnthropicToolSpec (computer)
	if len(anthropicTools) > 0 {
		idx := len(anthropicTools) - 1
		switch last := anthropicTools[idx].(type) {
		case anthropicTool:
			last.CacheControl = &anthropicCacheControl{Type: "ephemeral"}
			anthropicTools[idx] = last
		default:
			// For builtins/computer spec (maps), inject cache_control via generic map
			if m, ok := toMap(last); ok {
				m["cache_control"] = map[string]string{"type": "ephemeral"}
				anthropicTools[idx] = m
			}
		}
	}

	// Third cache breakpoint: the last message's terminal text block.
	// Tools + system + (optionally) the conversation up through the
	// most recent turn = up to 4 cache breakpoints (Anthropic's limit).
	// Marking the last message means the next request's prefix can
	// be served from cache up to and including this turn — only the
	// tail (next user input) is uncached.
	if n := len(anthropicMsgs); n > 0 {
		applyMsgCacheControl(&anthropicMsgs[n-1])
	}

	reqBody := anthropicRequest{
		Model:     model,
		MaxTokens: p.outputTokenLimit(model),
		Stream:    true,
		System:    systemContent,
		Messages:  anthropicMsgs,
		Tools:     anthropicTools,
	}
	alwaysThinks := anthropicAlwaysThinks(model)
	if alwaysThinks {
		reqBody.Thinking = &anthropicThinking{Type: "adaptive", BlockBinding: &anthropicBlockBinding{PrefixMismatchBehavior: "drop_block"}}
	}
	requestedEffort := normalizeReasoningLevel(p.reasoning.Level).String()
	effectiveEffort := "provider-default"
	if effort := anthropicEffort(p.reasoning.Level, model); effort != "" {
		reqBody.OutputConfig = &anthropicOutputConfig{Effort: effort}
		effectiveEffort = effort
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return ChatResponse{}, err
	}
	if err := observeProviderRequest(ctx, p.Name(), model, body); err != nil {
		return ChatResponse{}, err
	}

	url := p.url
	if url == "" {
		url = "https://api.anthropic.com/v1/messages"
	}
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	if alwaysThinks {
		req.Header.Set("anthropic-beta", anthropicThinkingBindingBeta)
	}
	resp, err := tracedProviderHTTP(req)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return ChatResponse{}, fmt.Errorf("Anthropic API error %d: %s", resp.StatusCode, string(respBody))
	}

	var full strings.Builder
	var usage TokenUsage
	var toolCalls []NativeToolCall
	var serverResults []ServerToolResult
	var currentServerTool string // name of server tool being executed
	var stopReason string
	completedStream := false

	// Track current tool_use block being streamed
	type pendingTool struct {
		id   string
		name string
		json strings.Builder
	}
	var currentTool *pendingTool
	var reasoning strings.Builder
	// replay holds the assistant content blocks in stream order so a turn with
	// thinking can be sent back unchanged. Unknown block types disable replay.
	var replay []map[string]any
	var currentBlock map[string]any
	replayable := true

	// Wrap resp.Body so a silent stream is killed after the idle window
	// instead of hanging the thinker indefinitely. Each byte received
	// resets the timer, so healthy slow streams are unaffected; only a
	// fully silent stall (no SSE events, not even keepalives) trips it.
	streamBody := newIdleReader(resp.Body, streamIdleTimeout(), nil)
	defer streamBody.Close()

	scanner := bufio.NewScanner(streamBody)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "error":
			if event.Error != nil {
				return ChatResponse{}, fmt.Errorf("Anthropic stream error (%s): %s", event.Error.Type, event.Error.Message)
			}
			return ChatResponse{}, fmt.Errorf("Anthropic stream error")
		case "content_block_start":
			if event.ContentBlock != nil {
				currentBlock = nil
				switch event.ContentBlock.Type {
				case "thinking":
					currentBlock = map[string]any{"type": "thinking", "thinking": "", "signature": ""}
				case "redacted_thinking":
					currentBlock = map[string]any{"type": "redacted_thinking", "data": event.ContentBlock.Data}
				case "text":
					currentBlock = map[string]any{"type": "text", "text": event.ContentBlock.Text}
				case "tool_use":
				default:
					replayable = false
				}
				if currentBlock != nil {
					replay = append(replay, currentBlock)
				}
				switch event.ContentBlock.Type {
				case "tool_use":
					currentTool = &pendingTool{
						id:   event.ContentBlock.ID,
						name: event.ContentBlock.Name,
					}
				case "server_tool_use":
					// Built-in tool being executed server-side
					currentServerTool = event.ContentBlock.Name
					code, _ := event.ContentBlock.Input["code"].(string)
					if code != "" && onChunk != nil {
						onChunk("\n→ " + currentServerTool + ": executing...\n")
					}
				case "code_execution_tool_result":
					// Server tool result
					var output, stderr string
					for _, c := range event.ContentBlock.Content {
						if c.Type == "text" {
							output += c.Text
						}
						if c.Output.Stdout != "" {
							output += c.Output.Stdout
						}
						if c.Output.Stderr != "" {
							stderr += c.Output.Stderr
						}
					}
					serverResults = append(serverResults, ServerToolResult{
						ToolName: currentServerTool,
						Output:   output,
						Error:    stderr,
					})
					if onChunk != nil {
						preview := output
						if len(preview) > 200 {
							preview = preview[:200] + "..."
						}
						onChunk("\n← " + currentServerTool + ": " + preview + "\n")
					}
					currentServerTool = ""
				}
			}
		case "content_block_delta":
			if event.Delta != nil {
				if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
					full.WriteString(event.Delta.Text)
					if onChunk != nil {
						onChunk(event.Delta.Text)
					}
					if currentBlock != nil && currentBlock["type"] == "text" {
						currentBlock["text"] = currentBlock["text"].(string) + event.Delta.Text
					}
				}
				if event.Delta.Type == "thinking_delta" && currentBlock != nil && currentBlock["type"] == "thinking" {
					currentBlock["thinking"] = currentBlock["thinking"].(string) + event.Delta.Thinking
					if event.Delta.Thinking != "" {
						reasoning.WriteString(event.Delta.Thinking)
						if onThinking != nil {
							onThinking(event.Delta.Thinking)
						}
					}
				}
				if event.Delta.Type == "signature_delta" && currentBlock != nil && currentBlock["type"] == "thinking" {
					currentBlock["signature"] = currentBlock["signature"].(string) + event.Delta.Signature
				}
				if event.Delta.Type == "input_json_delta" && currentTool != nil {
					logMsg("ANTHROPIC-STREAM", fmt.Sprintf("tool=%s chunk_len=%d chunk=%q", currentTool.name, len(event.Delta.PartialJSON), truncateStr(event.Delta.PartialJSON, 80)))
					currentTool.json.WriteString(event.Delta.PartialJSON)
					if onToolChunk != nil {
						onToolChunk(currentTool.name, currentTool.id, event.Delta.PartialJSON)
					}
				}
			}
		case "content_block_stop":
			if currentTool != nil {
				// Parse accumulated JSON into args
				args := make(map[string]string)
				var raw map[string]any
				if currentTool.json.Len() == 0 {
					currentTool.json.WriteString("{}")
				}
				if err := json.Unmarshal([]byte(currentTool.json.String()), &raw); err != nil || raw == nil || currentTool.id == "" || currentTool.name == "" {
					return ChatResponse{}, fmt.Errorf("invalid Anthropic tool call arguments or identity")
				}
				if err := json.Unmarshal([]byte(currentTool.json.String()), &raw); err == nil {
					for k, v := range raw {
						switch v.(type) {
						case string:
							args[k] = v.(string)
						default:
							b, _ := json.Marshal(v)
							args[k] = string(b)
						}
					}
				}
				toolCalls = append(toolCalls, NativeToolCall{
					ID:            currentTool.id,
					Name:          currentTool.name,
					Args:          args,
					RawArgs:       currentTool.json.String(),
					CanonicalArgs: json.RawMessage(currentTool.json.String()),
				})
				replay = append(replay, map[string]any{"type": "tool_use", "id": currentTool.id, "name": currentTool.name, "input": json.RawMessage(currentTool.json.String())})
				currentTool = nil
			}
			currentBlock = nil
		case "message_start":
			if event.Message != nil && event.Message.Usage != nil {
				u := event.Message.Usage
				// Anthropic reports three separate counts:
				//   input_tokens                — NEW tokens not served from cache
				//   cache_read_input_tokens     — tokens served from cache
				//   cache_creation_input_tokens — tokens newly written to cache
				// The real "prompt size" the model processed is the sum
				// of all three. We historically set PromptTokens =
				// input_tokens alone, which is wrong: on a turn with a
				// big cached prefix the cached count was legitimately
				// larger than "PromptTokens", making cached > in look
				// impossible. Downstream pricing (uncached = in − cached)
				// stays correct because we include creation in the
				// uncached bucket (creation is billed at ~1.25× input;
				// we approximate at 1×).
				usage.PromptTokens = u.InputTokens + u.CacheRead + u.CacheCreation
				usage.CachedTokens = u.CacheRead
				if u.CacheRead > 0 || u.CacheCreation > 0 {
					logMsg("ANTHROPIC", fmt.Sprintf("cache: read=%d creation=%d fresh=%d total=%d",
						u.CacheRead, u.CacheCreation, u.InputTokens, usage.PromptTokens))
				}
			}
		case "message_stop":
			completedStream = true
		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != "" {
				stopReason = event.Delta.StopReason
			}
			if event.Usage != nil {
				usage.CompletionTokens = event.Usage.OutputTokens
				// Cache tokens can also appear in message_delta — refresh
				// the prompt total too, not just the cached count, so
				// we don't drift from message_start.
				if event.Usage.CacheRead > 0 || event.Usage.CacheCreation > 0 {
					usage.CachedTokens = event.Usage.CacheRead
					usage.PromptTokens = event.Usage.InputTokens + event.Usage.CacheRead + event.Usage.CacheCreation
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return ChatResponse{}, fmt.Errorf("Anthropic stream read error: %w", err)
	}
	if currentTool != nil {
		reason := stopReason
		if reason == "" {
			reason = "stream ended"
		}
		return ChatResponse{}, fmt.Errorf("Anthropic response ended with incomplete tool input for %s after %d bytes (stop_reason=%s)", currentTool.name, currentTool.json.Len(), reason)
	}

	if !completedStream {
		return ChatResponse{}, fmt.Errorf("Anthropic stream ended before message_stop")
	}
	return validateProviderToolOutput(ChatResponse{
		Text:                     full.String(),
		Reasoning:                reasoning.String(),
		RequestedReasoningEffort: requestedEffort,
		EffectiveReasoningEffort: effectiveEffort,
		ToolCalls:                toolCalls,
		ServerResults:            serverResults,
		ProviderState:            anthropicResponseState(replay, replayable),
		Usage:                    usage,
	})
}

// anthropicResponseState keeps the turn's content blocks only when it holds
// thinking; turns without thinking are rebuilt from structured history.
func anthropicResponseState(blocks []map[string]any, replayable bool) *ProviderResponseState {
	if !replayable {
		return nil
	}
	thinking := false
	for _, b := range blocks {
		thinking = thinking || b["type"] == "thinking" || b["type"] == "redacted_thinking"
	}
	if !thinking {
		return nil
	}
	state := &ProviderResponseState{Provider: "anthropic"}
	for _, b := range blocks {
		if b["type"] == "text" && b["text"] == "" {
			continue // the API rejects empty text blocks
		}
		raw, err := json.Marshal(b)
		if err != nil {
			return nil
		}
		state.Items = append(state.Items, raw)
	}
	return state
}

// anthropicReplayContent returns the stored blocks of an assistant turn when
// they still match its structured history (same tool calls in order, same
// text). Any divergence means history was edited, so the turn is rebuilt.
func anthropicReplayContent(m Message) []json.RawMessage {
	if m.Role != "assistant" || m.ProviderState == nil || m.ProviderState.Provider != "anthropic" || len(m.ProviderState.Items) == 0 {
		return nil
	}
	var text strings.Builder
	calls := 0
	for _, raw := range m.ProviderState.Items {
		var block struct {
			Type  string          `json:"type"`
			Text  string          `json:"text"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(raw, &block) != nil {
			return nil
		}
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			if calls >= len(m.ToolCalls) || m.ToolCalls[calls].ID != block.ID || m.ToolCalls[calls].Name != block.Name {
				return nil
			}
			var inputs [2]any
			for i, raw := range []json.RawMessage{block.Input, toolCallArguments(m.ToolCalls[calls])} {
				decoder := json.NewDecoder(bytes.NewReader(raw))
				decoder.UseNumber()
				if decoder.Decode(&inputs[i]) != nil {
					return nil
				}
			}
			if !reflect.DeepEqual(inputs[0], inputs[1]) {
				return nil
			}
			calls++
		}
	}
	if calls != len(m.ToolCalls) || strings.TrimSpace(text.String()) != strings.TrimSpace(m.Content) || m.HasParts() {
		return nil
	}
	return append([]json.RawMessage(nil), m.ProviderState.Items...)
}

// toAnthropicBlocks converts our ContentParts to Anthropic content blocks.
func toAnthropicBlocks(parts []ContentPart) []anthropicContentBlock {
	parts = fileRefsForModel(parts)
	var blocks []anthropicContentBlock
	for _, p := range parts {
		switch p.Type {
		case "text":
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: p.Text})
		case "image_url":
			if p.ImageURL != nil {
				if strings.HasPrefix(p.ImageURL.URL, "data:") {
					segments := strings.SplitN(p.ImageURL.URL, ",", 2)
					mediaType := strings.TrimPrefix(strings.TrimSuffix(segments[0], ";base64"), "data:")
					data := ""
					if len(segments) > 1 {
						data = segments[1]
					}
					blocks = append(blocks, anthropicContentBlock{
						Type:   "image",
						Source: &anthropicSource{Type: "base64", MediaType: mediaType, Data: data},
					})
				} else {
					blocks = append(blocks, anthropicContentBlock{
						Type:   "image",
						Source: &anthropicSource{Type: "url", URL: p.ImageURL.URL},
					})
				}
			}
		case "audio_url", "input_audio":
			blocks = append(blocks, anthropicContentBlock{Type: "text", Text: "[audio input not supported by this provider]"})
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, anthropicContentBlock{Type: "text", Text: ""})
	}
	return blocks
}
