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
	"strings"

	"github.com/apteva/core/pkg/computer"
)

// OpenAINativeProvider uses the OpenAI Responses API for native computer use,
// web_search, code_interpreter, and other OpenAI-specific features.
// For OpenAI-compatible endpoints (Fireworks, Ollama, etc.), use OpenAICompatProvider.
type OpenAINativeProvider struct {
	apiKey       string
	models       map[ModelTier]string
	builtinTools []string
}

func NewOpenAINativeProvider(apiKey string) LLMProvider {
	return &OpenAINativeProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge:  "gpt-5.4-mini",
			ModelMedium: "gpt-5.4-mini",
			ModelSmall:  "gpt-5.4-mini",
		},
	}
}

func (p *OpenAINativeProvider) Name() string                            { return "openai" }
func (p *OpenAINativeProvider) Models() map[ModelTier]string            { return p.models }
func (p *OpenAINativeProvider) SupportsNativeTools() bool               { return true }
func (p *OpenAINativeProvider) CostPer1M() (float64, float64, float64) {
	// Default to gpt-5.4-mini pricing
	return 0.75, 0.375, 4.50
}

func (p *OpenAINativeProvider) AvailableBuiltinTools() []BuiltinTool {
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
	Model  string           `json:"model"`
	Input  []oaiInputItem   `json:"input"`
	Tools  []any            `json:"tools,omitempty"`
	Stream bool             `json:"stream"`
}

// oaiInputItem is a polymorphic input item for the Responses API.
type oaiInputItem struct {
	Type    string `json:"type"`              // "message", "computer_call_output", "function_call", "function_call_output"
	Role    string `json:"role,omitempty"`    // for type=message
	Content any    `json:"content,omitempty"` // string or []oaiContentBlock
	Name    string `json:"name,omitempty"`    // for type=function_call

	// computer_call_output / function_call fields
	CallID     string `json:"call_id,omitempty"`
	Output     any    `json:"output,omitempty"` // screenshot etc.
	Arguments  string `json:"arguments,omitempty"` // for type=function_call (JSON string)
}

type oaiContentBlock struct {
	Type     string `json:"type"`                  // "input_text", "input_image", "input_file"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`    // data:image/png;base64,...
	Detail   string `json:"detail,omitempty"`       // "original", "high", "low"
	FileURL  string `json:"file_url,omitempty"`     // URL or data URI for audio/files
}

type oaiComputerTool struct {
	Type          string `json:"type"`           // "computer"
	DisplayWidth  int    `json:"display_width"`
	DisplayHeight int    `json:"display_height"`
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

	// function_call fields
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type oaiComputerAction struct {
	Type      string `json:"type"` // "click", "type", "keypress", "scroll", "drag", "move", "screenshot", "wait", "double_click"
	X         int    `json:"x,omitempty"`
	Y         int    `json:"y,omitempty"`
	Button    string `json:"button,omitempty"`    // "left", "right", "middle"
	Text      string `json:"text,omitempty"`      // for "type"
	Key       string `json:"key,omitempty"`       // for "keypress"
	ScrollX   int    `json:"scroll_x,omitempty"`
	ScrollY   int    `json:"scroll_y,omitempty"`
	StartX    int    `json:"start_x,omitempty"` // drag
	StartY    int    `json:"start_y,omitempty"`
	EndX      int    `json:"end_x,omitempty"`
	EndY      int    `json:"end_y,omitempty"`
	Duration  int    `json:"duration,omitempty"` // wait ms
	Keys      []string `json:"keys,omitempty"`   // modifier keys
}

func (p *OpenAINativeProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	// Convert messages to Responses API input items
	input := p.buildInput(messages)

	// Convert tools
	var apiTools []any
	hasComputerUse := false
	for _, t := range tools {
		if t.Name == "computer_use" {
			// Only gpt-5.4 supports native computer use
			if !strings.Contains(model, "gpt-5.4") || strings.Contains(model, "mini") || strings.Contains(model, "nano") {
				logMsg("OPENAI-NATIVE", fmt.Sprintf("skipping computer_use — not supported by %s", model))
				continue
			}
			width, height := 1280, 800
			if w, ok := t.Parameters["_display_width"].(int); ok {
				width = w
			}
			if h, ok := t.Parameters["_display_height"].(int); ok {
				height = h
			}
			apiTools = append(apiTools, oaiComputerTool{
				Type:          "computer",
				DisplayWidth:  width,
				DisplayHeight: height,
			})
			hasComputerUse = true
			logMsg("OPENAI-NATIVE", "native computer tool enabled")
		} else if t.Name == "browser_session" {
			if !hasComputerUse {
				continue // skip if computer not enabled
			}
			continue
		} else {
			apiTools = append(apiTools, oaiFunctionTool{
				Type:        "function",
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			})
		}
	}
	_ = hasComputerUse

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

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/responses", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		logMsg("OPENAI-NATIVE", fmt.Sprintf("ERROR %d: %s", resp.StatusCode, string(respBody)))
		return ChatResponse{}, fmt.Errorf("OpenAI Responses API error %d: %s", resp.StatusCode, string(respBody))
	}

	return p.streamResponse(resp.Body, onChunk, onToolChunk)
}

// buildInput converts our Message slice to Responses API input items.
func (p *OpenAINativeProvider) buildInput(messages []Message) []oaiInputItem {
	var items []oaiInputItem

	for _, m := range messages {
		// System message
		if m.Role == "system" {
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
					// Computer call output with screenshot
					items = append(items, oaiInputItem{
						Type:   "computer_call_output",
						CallID: tr.CallID,
						Output: []oaiContentBlock{
							{Type: "input_image", ImageURL: "data:image/png;base64," + base64.StdEncoding.EncodeToString(tr.Image), Detail: "original"},
						},
					})
				} else {
					// Function call output
					items = append(items, oaiInputItem{
						Type:    "function_call_output",
						CallID:  tr.CallID,
						Output:  tr.Content,
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
				if computer.IsGeminiComputerAction(tc.Name) || isComputerUseTool(tc.Name) {
					// Computer call — reconstruct as computer_call item
					argsAny := make(map[string]any, len(tc.Args))
					for k, v := range tc.Args {
						argsAny[k] = v
					}
					actionsJSON, _ := json.Marshal([]map[string]any{argsAny})
					items = append(items, oaiInputItem{
						Type:   "computer_call",
						CallID: tc.ID,
						Output: json.RawMessage(actionsJSON),
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
func (p *OpenAINativeProvider) streamResponse(body io.Reader, onChunk func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	var full strings.Builder
	var usage TokenUsage
	var toolCalls []NativeToolCall

	// Track pending items
	type pendingFunc struct {
		id   string
		name string
		args strings.Builder
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
				if onToolChunk != nil && pf.name != "" {
					onToolChunk(pf.name, delta.ItemID, delta.Delta)
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
					args := make(map[string]string)
					var rawArgs map[string]any
					if json.Unmarshal([]byte(pf.args.String()), &rawArgs) == nil {
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

			case "computer_call":
				if currentComputer != nil {
					// Parse actions from the completed item
					var actions []oaiComputerAction
					json.Unmarshal(item.Actions, &actions)

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
	return ChatResponse{Text: response, ToolCalls: toolCalls, Usage: usage}, nil
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
