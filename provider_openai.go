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

func (p *OpenAICompatProvider) Name() string                          { return p.name }
func (p *OpenAICompatProvider) Models() map[ModelTier]string          { return p.models }
func (p *OpenAICompatProvider) CostPer1M() (float64, float64, float64) { return p.inputCost, p.cachedCost, p.outputCost }

func (p *OpenAICompatProvider) Chat(messages []Message, model string, onChunk func(string)) (string, TokenUsage, error) {
	reqBody := Request{
		Model:    model,
		Messages: messages,
		Stream:   true,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", TokenUsage{}, err
	}

	req, err := http.NewRequest("POST", p.url, bytes.NewReader(body))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" && p.authHeader != "" {
		req.Header.Set("Authorization", p.authHeader+" "+p.apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var full strings.Builder
	var usage TokenUsage
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

		var event StreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if len(event.Choices) > 0 {
			chunk := event.Choices[0].Delta.Content
			if chunk != "" {
				full.WriteString(chunk)
				if onChunk != nil {
					onChunk(chunk)
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

	return full.String(), usage, nil
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
			ModelSmall: "accounts/fireworks/models/kimi-k2p5",
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
