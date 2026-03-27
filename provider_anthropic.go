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

type AnthropicProvider struct {
	apiKey string
	models map[ModelTier]string
}

func NewAnthropicProvider(apiKey string) LLMProvider {
	return &AnthropicProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge: "claude-sonnet-4-20250514",
			ModelSmall: "claude-haiku-4-20250414",
		},
	}
}

func (p *AnthropicProvider) Name() string                          { return "anthropic" }
func (p *AnthropicProvider) Models() map[ModelTier]string          { return p.models }
func (p *AnthropicProvider) CostPer1M() (float64, float64, float64) { return 3.00, 0.30, 15.00 }

// Anthropic Messages API request format
type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Stream    bool               `json:"stream"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Anthropic streaming event types
type anthropicStreamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index,omitempty"`
	Delta *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content_block,omitempty"`
	Message *struct {
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

func (p *AnthropicProvider) Chat(messages []Message, model string, onChunk func(string)) (string, TokenUsage, error) {
	// Convert messages: extract system prompt, convert rest to Anthropic format
	var system string
	var anthropicMsgs []anthropicMessage
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
			continue
		}
		anthropicMsgs = append(anthropicMsgs, anthropicMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	// Anthropic requires at least one message
	if len(anthropicMsgs) == 0 {
		anthropicMsgs = append(anthropicMsgs, anthropicMessage{Role: "user", Content: "Begin."})
	}

	reqBody := anthropicRequest{
		Model:     model,
		MaxTokens: 4096,
		Stream:    true,
		System:    system,
		Messages:  anthropicMsgs,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", TokenUsage{}, err
	}

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("Anthropic API error %d: %s", resp.StatusCode, string(respBody))
	}

	var full strings.Builder
	var usage TokenUsage
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
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Text != "" {
				full.WriteString(event.Delta.Text)
				if onChunk != nil {
					onChunk(event.Delta.Text)
				}
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

	return full.String(), usage, nil
}
