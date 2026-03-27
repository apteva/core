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

type GoogleProvider struct {
	apiKey string
	models map[ModelTier]string
}

func NewGoogleProvider(apiKey string) LLMProvider {
	return &GoogleProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge: "gemini-2.5-flash",
			ModelSmall: "gemini-2.5-flash",
		},
	}
}

func (p *GoogleProvider) Name() string                          { return "google" }
func (p *GoogleProvider) Models() map[ModelTier]string          { return p.models }
func (p *GoogleProvider) CostPer1M() (float64, float64, float64) { return 0.15, 0.0375, 0.60 }

// Gemini API request format
type geminiRequest struct {
	Contents         []geminiContent        `json:"contents"`
	SystemInstruction *geminiContent        `json:"systemInstruction,omitempty"`
	GenerationConfig  map[string]any        `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

// Gemini streaming response
type geminiStreamResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
}

func (p *GoogleProvider) Chat(messages []Message, model string, onChunk func(string)) (string, TokenUsage, error) {
	// Convert messages to Gemini format
	var systemContent *geminiContent
	var contents []geminiContent

	for _, m := range messages {
		if m.Role == "system" {
			systemContent = &geminiContent{
				Parts: []geminiPart{{Text: m.Content}},
			}
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: m.Content}},
		})
	}

	if len(contents) == 0 {
		contents = append(contents, geminiContent{
			Role:  "user",
			Parts: []geminiPart{{Text: "Begin."}},
		})
	}

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig: map[string]any{
			"maxOutputTokens": 4096,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", TokenUsage{}, err
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", model, p.apiKey)

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("Gemini API error %d: %s", resp.StatusCode, string(respBody))
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

		var event geminiStreamResponse
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		for _, candidate := range event.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					full.WriteString(part.Text)
					if onChunk != nil {
						onChunk(part.Text)
					}
				}
			}
		}

		if event.UsageMetadata != nil {
			usage.PromptTokens = event.UsageMetadata.PromptTokenCount
			usage.CompletionTokens = event.UsageMetadata.CandidatesTokenCount
			usage.CachedTokens = event.UsageMetadata.CachedContentTokenCount
		}
	}

	return full.String(), usage, nil
}
