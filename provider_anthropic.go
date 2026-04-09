package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/apteva/core/pkg/computer"
)

// sanitizeToolID ensures tool call IDs only contain characters Anthropic accepts: [a-zA-Z0-9_-]
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
	apiKey       string
	models       map[ModelTier]string
	builtinTools []string // enabled built-in tools: "code_execution", "web_search"
}

func NewAnthropicProvider(apiKey string) LLMProvider {
	return &AnthropicProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge:  "claude-opus-4-6",
			ModelMedium: "claude-sonnet-4-6",
			ModelSmall:  "claude-haiku-4-5-20251001",
		},
	}
}

func (p *AnthropicProvider) Name() string                            { return "anthropic" }
func (p *AnthropicProvider) Models() map[ModelTier]string            { return p.models }
func (p *AnthropicProvider) CostPer1M() (float64, float64, float64) { return 3.00, 0.30, 15.00 }
func (p *AnthropicProvider) SupportsNativeTools() bool               { return true }

func (p *AnthropicProvider) AvailableBuiltinTools() []BuiltinTool {
	return []BuiltinTool{
		{Type: "code_execution_20250825", Name: "code_execution"},
		{Type: "web_search_20250305", Name: "web_search"},
	}
}

func (p *AnthropicProvider) SetBuiltinTools(tools []string) {
	p.builtinTools = tools
}

// --- Request types ---

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Stream    bool               `json:"stream"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Tools     []any              `json:"tools,omitempty"` // mixed: anthropicTool or anthropicBuiltinTool
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// anthropicBuiltinTool is for Anthropic-specific tool types (computer_use, text_editor, bash).
type anthropicBuiltinTool struct {
	Type           string `json:"type"`                      // "computer_20250124"
	Name           string `json:"name"`                      // "computer"
	DisplayWidthPx  int   `json:"display_width_px,omitempty"`
	DisplayHeightPx int   `json:"display_height_px,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string or []anthropicContentBlock
}

type anthropicContentBlock struct {
	Type       string           `json:"type"`                    // "text", "image", "tool_use", "tool_result"
	Text       string           `json:"text,omitempty"`          // type=text
	Source     *anthropicSource `json:"source,omitempty"`        // type=image
	ID         string           `json:"id,omitempty"`            // type=tool_use
	Name       string           `json:"name,omitempty"`          // type=tool_use
	Input      map[string]any   `json:"input,omitempty"`         // type=tool_use
	ToolUseID  string           `json:"tool_use_id,omitempty"`   // type=tool_result
	Content    any              `json:"content,omitempty"`       // type=tool_result (string or blocks)
	IsError    bool             `json:"is_error,omitempty"`      // type=tool_result
}

type anthropicSource struct {
	Type      string `json:"type"` // "url", "base64"
	URL       string `json:"url,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// --- Streaming response types ---

type anthropicStreamEvent struct {
	Type         string `json:"type"`
	Index        int    `json:"index,omitempty"`
	Delta        *anthropicDelta `json:"delta,omitempty"`
	ContentBlock *anthropicBlockStart `json:"content_block,omitempty"`
	Message      *struct {
		Usage *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			CacheRead    int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	} `json:"message,omitempty"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

type anthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"` // for tool_use input streaming
}

type anthropicBlockStart struct {
	Type  string `json:"type"` // "text", "tool_use", "server_tool_use", "code_execution_tool_result"
	Text  string `json:"text,omitempty"`
	ID    string `json:"id,omitempty"`    // tool_use
	Name  string `json:"name,omitempty"`  // tool_use / server_tool_use
	Input map[string]any `json:"input,omitempty"` // tool_use (may be empty at start)
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

func (p *AnthropicProvider) Chat(messages []Message, model string, tools []NativeTool, onChunk func(string), onToolChunk func(string, string)) (ChatResponse, error) {
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
					// Image result (screenshot etc.)
					block.Content = []anthropicContentBlock{
						{Type: "image", Source: &anthropicSource{
							Type:      "base64",
							MediaType: "image/png",
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

		// Message with tool calls (assistant)
		if len(m.ToolCalls) > 0 {
			var blocks []anthropicContentBlock
			if m.Content != "" {
				blocks = append(blocks, anthropicContentBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				input := make(map[string]any)
				for k, v := range tc.Args {
					input[k] = v
				}
				blocks = append(blocks, anthropicContentBlock{
					Type:  "tool_use",
					ID:    sanitizeToolID(tc.ID),
					Name:  tc.Name,
					Input: input,
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

	// Convert tools — separate computer_use (builtin) from regular tools
	var anthropicTools []any
	hasComputerUse := false
	computerBeta := ""
	for _, t := range tools {
		if t.Name == "computer_use" {
			// Use the native Anthropic computer use format
			// Parse display dimensions from parameters or defaults
			width, height := 1280, 800
			if w, ok := t.Parameters["_display_width"].(int); ok {
				width = w
			}
			if h, ok := t.Parameters["_display_height"].(int); ok {
				height = h
			}
			display := computer.DisplaySize{Width: width, Height: height}
			// Computer tool version depends on model
			toolVersion := "20250124" // default for older models
			if strings.Contains(model, "opus-4-6") || strings.Contains(model, "sonnet-4-6") || strings.Contains(model, "opus-4-5") {
				toolVersion = "20251124" // enhanced computer use for 4.5+ models
			}
			spec := computer.GetAnthropicToolSpec(display, toolVersion)
			anthropicTools = append(anthropicTools, spec)
			computerBeta = computer.AnthropicBetaHeader(toolVersion)
			hasComputerUse = true
		} else {
			anthropicTools = append(anthropicTools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: t.Parameters,
			})
		}
	}

	// Add enabled built-in tools
	for _, btName := range p.builtinTools {
		for _, available := range p.AvailableBuiltinTools() {
			if available.Name == btName {
				anthropicTools = append(anthropicTools, map[string]string{
					"type": available.Type,
					"name": available.Name,
				})
				break
			}
		}
	}

	reqBody := anthropicRequest{
		Model:     model,
		MaxTokens: 4096,
		Stream:    true,
		System:    system,
		Messages:  anthropicMsgs,
		Tools:     anthropicTools,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return ChatResponse{}, err
	}

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	if hasComputerUse {
		req.Header.Set("anthropic-beta", computerBeta)
	}

	resp, err := llmHTTPClient.Do(req)
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

	// Track current tool_use block being streamed
	type pendingTool struct {
		id   string
		name string
		json strings.Builder
	}
	var currentTool *pendingTool

	scanner := bufio.NewScanner(resp.Body)
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
		case "content_block_start":
			if event.ContentBlock != nil {
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
				}
				if event.Delta.Type == "input_json_delta" && currentTool != nil {
					logMsg("ANTHROPIC-STREAM", fmt.Sprintf("tool=%s chunk_len=%d chunk=%q", currentTool.name, len(event.Delta.PartialJSON), truncateStr(event.Delta.PartialJSON, 80)))
					currentTool.json.WriteString(event.Delta.PartialJSON)
					if onToolChunk != nil {
						onToolChunk(currentTool.name, event.Delta.PartialJSON)
					}
				}
			}
		case "content_block_stop":
			if currentTool != nil {
				// Parse accumulated JSON into args
				args := make(map[string]string)
				var raw map[string]any
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
					ID:   currentTool.id,
					Name: currentTool.name,
					Args: args,
				})
				currentTool = nil
			}
		case "message_start":
			if event.Message != nil && event.Message.Usage != nil {
				usage.PromptTokens = event.Message.Usage.InputTokens
				usage.CachedTokens = event.Message.Usage.CacheRead
			}
		case "message_delta":
			if event.Usage != nil {
				usage.CompletionTokens = event.Usage.OutputTokens
			}
		}
	}

	return ChatResponse{
		Text:          full.String(),
		ToolCalls:     toolCalls,
		ServerResults: serverResults,
		Usage:         usage,
	}, nil
}

// toAnthropicBlocks converts our ContentParts to Anthropic content blocks.
func toAnthropicBlocks(parts []ContentPart) []anthropicContentBlock {
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
