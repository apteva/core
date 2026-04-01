package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
}

func (p *OpenAICompatProvider) Name() string                            { return p.name }
func (p *OpenAICompatProvider) Models() map[ModelTier]string            { return p.models }
func (p *OpenAICompatProvider) CostPer1M() (float64, float64, float64) { return p.inputCost, p.cachedCost, p.outputCost }
func (p *OpenAICompatProvider) SupportsNativeTools() bool {
	// OpenAI and Fireworks support tools; Ollama may not
	return p.name == "openai" || p.name == "fireworks"
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
					// Tool result with image — send as multimodal content
					out = append(out, map[string]any{
						"role":         "tool",
						"tool_call_id": tr.CallID,
						"content": []map[string]any{
							{"type": "text", "text": tr.Content},
							{"type": "image_url", "image_url": map[string]string{
								"url": "data:image/png;base64," + base64Encode(tr.Image),
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

		// Assistant message with tool calls
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
				"tool_calls": toolCalls,
			}
			if m.Content != "" {
				msg["content"] = m.Content
			}
			out = append(out, msg)
			continue
		}

		// Regular message
		if m.HasParts() {
			out = append(out, openaiMessage{Role: m.Role, Content: convertAudioURLParts(m.Parts)})
		} else {
			out = append(out, openaiMessage{Role: m.Role, Content: m.Content})
		}
	}
	return out
}

func (p *OpenAICompatProvider) Chat(messages []Message, model string, tools []NativeTool, onChunk func(string)) (ChatResponse, error) {
	// Build request
	reqMap := map[string]any{
		"model":    model,
		"messages": toOpenAIMessages(messages),
		"stream":   true,
	}
	// OpenAI supports stream_options for usage in streaming; Fireworks may not
	if p.name == "openai" {
		reqMap["stream_options"] = map[string]any{"include_usage": true}
	}

	// Add tools if provider supports them
	if len(tools) > 0 && p.SupportsNativeTools() {
		var defs []openaiToolDef
		for _, t := range tools {
			def := openaiToolDef{Type: "function"}
			def.Function.Name = t.Name
			def.Function.Description = t.Description
			def.Function.Parameters = t.Parameters
			defs = append(defs, def)
		}
		reqMap["tools"] = defs
	}

	body, err := json.Marshal(reqMap)
	if err != nil {
		return ChatResponse{}, err
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

	req, err := http.NewRequest("POST", p.url, bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" && p.authHeader != "" {
		req.Header.Set("Authorization", p.authHeader+" "+p.apiKey)
	}

	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return ChatResponse{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var full strings.Builder
	var usage TokenUsage
	// Track streamed tool calls by index
	pendingTools := make(map[int]*struct {
		id       string
		name     string
		argsJSON strings.Builder
	})

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event struct {
			Choices []struct {
				Delta struct {
					Content   string                 `json:"content"`
					ToolCalls []openaiToolCallDelta   `json:"tool_calls,omitempty"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *Usage `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if len(event.Choices) > 0 {
			delta := event.Choices[0].Delta
			if delta.Content != "" {
				full.WriteString(delta.Content)
				if onChunk != nil {
					onChunk(delta.Content)
				}
			}
			for _, tc := range delta.ToolCalls {
				pt, ok := pendingTools[tc.Index]
				if !ok {
					pt = &struct {
						id       string
						name     string
						argsJSON strings.Builder
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
					}
				}
			}
		}
		if event.Usage != nil {
			usage.PromptTokens = event.Usage.PromptTokens
			usage.CompletionTokens = event.Usage.CompletionTokens
			if event.Usage.PromptTokensDetails != nil {
				usage.CachedTokens = event.Usage.PromptTokensDetails.CachedTokens
			}
		}
	}

	// Assemble completed tool calls
	var toolCalls []NativeToolCall
	for i := 0; i < len(pendingTools); i++ {
		pt, ok := pendingTools[i]
		if !ok {
			continue
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

	return ChatResponse{
		Text:      full.String(),
		ToolCalls: toolCalls,
		Usage:     usage,
	}, nil
}

// --- Factory functions ---

func NewFireworksProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "fireworks",
		apiKey:     apiKey,
		url:        "https://api.fireworks.ai/inference/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge: "accounts/fireworks/models/kimi-k2p5",
			ModelSmall: "accounts/fireworks/routers/kimi-k2p5-turbo",
		},
		inputCost:  0.60,
		cachedCost: 0.10,
		outputCost: 3.00,
	}
}

func NewOpenAIProvider(apiKey string) LLMProvider {
	return &OpenAICompatProvider{
		name:       "openai",
		apiKey:     apiKey,
		url:        "https://api.openai.com/v1/chat/completions",
		authHeader: "Bearer",
		models: map[ModelTier]string{
			ModelLarge: "gpt-4o",
			ModelSmall: "gpt-4o-mini",
		},
		inputCost:  2.50,
		cachedCost: 1.25,
		outputCost: 10.00,
	}
}

func NewOllamaProvider(host string) LLMProvider {
	url := strings.TrimRight(host, "/") + "/v1/chat/completions"
	return &OpenAICompatProvider{
		name:       "ollama",
		apiKey:     "",
		url:        url,
		authHeader: "",
		models: map[ModelTier]string{
			ModelLarge: "llama3.1",
			ModelSmall: "llama3.1",
		},
		inputCost:  0,
		cachedCost: 0,
		outputCost: 0,
	}
}
