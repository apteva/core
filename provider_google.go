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
)

func base64Encode(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// GoogleModel holds metadata for a Gemini model.
type GoogleModel struct {
	ID              string
	InputPer1M      float64
	CachedPer1M     float64
	OutputPer1M     float64
	MaxOutputTokens int
}

// Available Gemini models (March 2026)
var geminiModels = map[string]GoogleModel{
	// Gemini 3.1 series
	"gemini-3.1-pro-preview": {
		ID: "gemini-3.1-pro-preview", InputPer1M: 2.00, CachedPer1M: 0.20, OutputPer1M: 12.00, MaxOutputTokens: 65536,
	},
	"gemini-3.1-flash-lite-preview": {
		ID: "gemini-3.1-flash-lite-preview", InputPer1M: 0.25, CachedPer1M: 0.025, OutputPer1M: 1.50, MaxOutputTokens: 65536,
	},
	// Gemini 3 series
	"gemini-3-flash-preview": {
		ID: "gemini-3-flash-preview", InputPer1M: 0.50, CachedPer1M: 0.05, OutputPer1M: 3.00, MaxOutputTokens: 65536,
	},
	// Gemini 2.5 series
	"gemini-2.5-pro": {
		ID: "gemini-2.5-pro", InputPer1M: 1.00, CachedPer1M: 0.10, OutputPer1M: 10.00, MaxOutputTokens: 65536,
	},
	"gemini-2.5-flash": {
		ID: "gemini-2.5-flash", InputPer1M: 0.30, CachedPer1M: 0.03, OutputPer1M: 2.50, MaxOutputTokens: 65536,
	},
	// Computer Use model
	"gemini-2.5-computer-use-preview-10-2025": {
		ID: "gemini-2.5-computer-use-preview-10-2025", InputPer1M: 1.00, CachedPer1M: 0.10, OutputPer1M: 4.00, MaxOutputTokens: 64000,
	},
}

// GeminiModelOrder defines the cycle order for model switching in the TUI.
var GeminiModelOrder = []string{
	"gemini-3.1-pro-preview",
	"gemini-3-flash-preview",
	"gemini-3.1-flash-lite-preview",
	"gemini-2.5-pro",
	"gemini-2.5-flash",
}

type GoogleProvider struct {
	apiKey       string
	models       map[ModelTier]string
	activeModel  string // current model ID for cost tracking
}

func NewGoogleProvider(apiKey string) LLMProvider {
	return &GoogleProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge:  "gemini-2.5-pro-preview-05-06",
			ModelMedium: "gemini-2.5-flash-preview-04-17",
			ModelSmall:  "gemini-2.5-flash-preview-04-17",
		},
		activeModel: "gemini-3.1-pro-preview",
	}
}

func (p *GoogleProvider) Name() string                 { return "google" }
func (p *GoogleProvider) Models() map[ModelTier]string  { return p.models }
func (p *GoogleProvider) SupportsNativeTools() bool     { return true }

func (p *GoogleProvider) AvailableBuiltinTools() []BuiltinTool {
	return []BuiltinTool{
		{Type: "code_execution", Name: "code_execution"},
	}
}

func (p *GoogleProvider) SetBuiltinTools(tools []string) {
	// Google built-in tools handled via API config
}

func (p *GoogleProvider) WithBuiltins(builtins []string) LLMProvider {
	return p // Google builtins handled separately
}

func (p *GoogleProvider) CostPer1M() (float64, float64, float64) {
	if m, ok := geminiModels[p.activeModel]; ok {
		return m.InputPer1M, m.CachedPer1M, m.OutputPer1M
	}
	// Fallback to gemini-3.1-pro-preview pricing
	return 2.00, 0.20, 12.00
}

// SetModel updates the active model. Called from TUI model cycling.
func (p *GoogleProvider) SetModel(modelID string) {
	if _, ok := geminiModels[modelID]; ok {
		p.activeModel = modelID
		p.models[ModelLarge] = modelID
		p.models[ModelSmall] = modelID
	}
}

// ActiveModel returns the current model ID.
func (p *GoogleProvider) ActiveModel() string { return p.activeModel }

// AvailableModels returns all supported Gemini model IDs in cycle order.
func (p *GoogleProvider) AvailableModels() []string { return GeminiModelOrder }

// Gemini API request format
type geminiRequest struct {
	Contents          []geminiContent        `json:"contents"`
	SystemInstruction *geminiContent         `json:"systemInstruction,omitempty"`
	GenerationConfig  map[string]any         `json:"generationConfig,omitempty"`
	Tools             []geminiToolDecl       `json:"tools,omitempty"`
}

type geminiToolDecl struct {
	FunctionDeclarations []geminiFunctionDecl `json:"functionDeclarations,omitempty"`
	ComputerUse          *geminiComputerUse   `json:"computerUse,omitempty"`
}

// geminiComputerUse is the native Gemini Computer Use tool config.
type geminiComputerUse struct {
	Environment                string   `json:"environment"`                          // "ENVIRONMENT_BROWSER"
	ExcludedPredefinedFunctions []string `json:"excludedPredefinedFunctions,omitempty"` // e.g. ["drag_and_drop"]
}

type geminiFunctionDecl struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInline           `json:"inlineData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type geminiInline struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
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

func (p *GoogleProvider) Chat(messages []Message, model string, tools []NativeTool, onChunk func(string), onToolChunk func(string, string)) (ChatResponse, error) {
	// Track active model for cost calculation
	p.activeModel = model

	// Convert messages to Gemini format
	// Gemini requires: one optional systemInstruction, then strictly alternating user/model turns.
	// We merge consecutive same-role messages and fold extra system messages into user context.
	var systemParts []geminiPart
	var contents []geminiContent

	for _, m := range messages {
		if m.Role == "system" {
			// First system message becomes systemInstruction; subsequent ones merge into next user turn
			if len(contents) == 0 && len(systemParts) == 0 {
				systemParts = append(systemParts, geminiPart{Text: m.TextContent()})
			} else {
				// Fold into a user message as context
				text := "[system context] " + m.TextContent()
				if len(contents) > 0 && contents[len(contents)-1].Role == "user" {
					// Merge into existing user turn
					contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, geminiPart{Text: text})
				} else {
					contents = append(contents, geminiContent{
						Role:  "user",
						Parts: []geminiPart{{Text: text}},
					})
				}
			}
			continue
		}
		// Handle tool result messages (user → functionResponse)
		if len(m.ToolResults) > 0 {
			var parts []geminiPart
			for _, tr := range m.ToolResults {
				response := map[string]any{"result": tr.Content}
				parts = append(parts, geminiPart{
					FunctionResponse: &geminiFunctionResponse{
						Name:     tr.CallID,
						Response: response,
					},
				})
				// Gemini Computer Use: screenshots go as separate inlineData parts
				if tr.Image != nil {
					parts = append(parts, geminiPart{
						InlineData: &geminiInline{
							MimeType: "image/png",
							Data:     base64Encode(tr.Image),
						},
					})
				}
			}
			if len(contents) > 0 && contents[len(contents)-1].Role == "user" {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
			} else {
				contents = append(contents, geminiContent{Role: "user", Parts: parts})
			}
			continue
		}

		// Handle assistant messages with tool calls (model → functionCall)
		if len(m.ToolCalls) > 0 {
			var parts []geminiPart
			if m.Content != "" {
				parts = append(parts, geminiPart{Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				args := make(map[string]any)
				for k, v := range tc.Args {
					args[k] = v
				}
				parts = append(parts, geminiPart{
					FunctionCall: &geminiFunctionCall{Name: tc.Name, Args: args},
				})
			}
			if len(contents) > 0 && contents[len(contents)-1].Role == "model" {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
			} else {
				contents = append(contents, geminiContent{Role: "model", Parts: parts})
			}
			continue
		}

		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}

		var parts []geminiPart
		if m.HasParts() {
			parts = toGeminiParts(m.Parts)
		} else if m.Content != "" {
			parts = []geminiPart{{Text: m.Content}}
		} else {
			// Skip empty messages
			continue
		}

		// Merge consecutive same-role messages (Gemini requirement)
		if len(contents) > 0 && contents[len(contents)-1].Role == role {
			contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
		} else {
			contents = append(contents, geminiContent{Role: role, Parts: parts})
		}
	}

	if len(contents) == 0 {
		contents = append(contents, geminiContent{
			Role:  "user",
			Parts: []geminiPart{{Text: "Begin."}},
		})
	}

	// Ensure conversation starts with user (Gemini requirement)
	if contents[0].Role == "model" {
		contents = append([]geminiContent{{Role: "user", Parts: []geminiPart{{Text: "Begin."}}}}, contents...)
	}
	// Ensure conversation ends with user (Gemini requirement)
	if contents[len(contents)-1].Role == "model" {
		contents = append(contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: "Continue."}}})
	}

	var systemContent *geminiContent
	if len(systemParts) > 0 {
		systemContent = &geminiContent{Parts: systemParts}
	}

	// Log turn structure for debugging
	var turnLog strings.Builder
	for i, c := range contents {
		partLen := 0
		for _, p := range c.Parts {
			partLen += len(p.Text)
		}
		if i > 0 {
			turnLog.WriteString(", ")
		}
		turnLog.WriteString(fmt.Sprintf("%s(%d)", c.Role, partLen))
	}
	logMsg("GEMINI", fmt.Sprintf("model=%s msgs=%d contents=%d sys=%d turns=[%s]", model, len(messages), len(contents), len(systemParts), turnLog.String()))

	// Use model-specific max output tokens
	maxTokens := 65536
	if m, ok := geminiModels[model]; ok {
		maxTokens = m.MaxOutputTokens
	}

	// Convert native tools to Gemini function declarations
	// Separate computer_use (native) from regular tools
	var geminiTools []geminiToolDecl
	hasComputerUse := false
	if len(tools) > 0 {
		var funcs []geminiFunctionDecl
		for _, t := range tools {
			if t.Name == "computer_use" {
				// Use native Gemini Computer Use tool
				geminiTools = append(geminiTools, geminiToolDecl{
					ComputerUse: &geminiComputerUse{
						Environment: "ENVIRONMENT_BROWSER",
					},
				})
				hasComputerUse = true
				logMsg("GEMINI", "native computer_use tool enabled")
			} else if t.Name == "browser_session" {
				// browser_session is handled by Gemini's native navigate/search/go_back etc.
				// Skip — these are built into the Computer Use tool
				continue
			} else {
				funcs = append(funcs, geminiFunctionDecl{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
				})
			}
		}
		if len(funcs) > 0 {
			geminiTools = append(geminiTools, geminiToolDecl{FunctionDeclarations: funcs})
		}
	}
	_ = hasComputerUse

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig: map[string]any{
			"maxOutputTokens": maxTokens,
		},
		Tools: geminiTools,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return ChatResponse{}, err
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", model, p.apiKey)

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		logMsg("GEMINI", fmt.Sprintf("ERROR %d: %s", resp.StatusCode, string(respBody)))
		return ChatResponse{}, fmt.Errorf("Gemini API error %d: %s", resp.StatusCode, string(respBody))
	}
	logMsg("GEMINI", "streaming response started")

	var full strings.Builder
	var usage TokenUsage
	var toolCalls []NativeToolCall
	toolCallSeq := 0

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
				if part.FunctionCall != nil {
					toolCallSeq++
					args := make(map[string]string)
					for k, v := range part.FunctionCall.Args {
						switch v.(type) {
						case string:
							args[k] = v.(string)
						default:
							b, _ := json.Marshal(v)
							args[k] = string(b)
						}
					}
					toolCalls = append(toolCalls, NativeToolCall{
						ID:   fmt.Sprintf("gemini_%d", toolCallSeq),
						Name: part.FunctionCall.Name,
						Args: args,
					})
					// Gemini delivers complete tool calls, emit the full args as one chunk
					if onToolChunk != nil {
						argsJSON, _ := json.Marshal(part.FunctionCall.Args)
						onToolChunk(part.FunctionCall.Name, string(argsJSON))
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

	response := full.String()
	preview := response
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	logMsg("GEMINI", fmt.Sprintf("done tokens_in=%d tokens_out=%d len=%d tools=%d response=%q", usage.PromptTokens, usage.CompletionTokens, len(response), len(toolCalls), preview))
	return ChatResponse{Text: response, ToolCalls: toolCalls, Usage: usage}, nil
}

// audioMimeTypes maps file extensions to MIME types for audio.
var audioMimeTypes = map[string]string{
	"mp3": "audio/mp3", "wav": "audio/wav", "aac": "audio/aac",
	"ogg": "audio/ogg", "flac": "audio/flac", "aiff": "audio/aiff",
	"m4a": "audio/mp4",
}

// imageMimeTypes maps file extensions to MIME types for images.
var imageMimeTypes = map[string]string{
	"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
	"gif": "image/gif", "webp": "image/webp",
}

// fetchMediaAsBase64 downloads a URL and returns (base64data, mimeType, error).
func fetchMediaAsBase64(url string) (string, string, error) {
	logMsg("GEMINI", fmt.Sprintf("fetching media: %s", url))
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", "", fmt.Errorf("fetch failed: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; AptevaCore/1.0)")
	resp, err := llmHTTPClient.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("fetch failed: HTTP %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, 20*1024*1024)) // 20MB max
	if err != nil {
		return "", "", fmt.Errorf("read failed: %w", err)
	}

	// Detect MIME from Content-Type header or URL extension
	mime := resp.Header.Get("Content-Type")
	if mime == "" || mime == "application/octet-stream" {
		ext := ""
		if idx := strings.LastIndex(url, "."); idx >= 0 {
			ext = strings.ToLower(url[idx+1:])
			if qIdx := strings.Index(ext, "?"); qIdx >= 0 {
				ext = ext[:qIdx]
			}
		}
		if m, ok := audioMimeTypes[ext]; ok {
			mime = m
		} else if m, ok := imageMimeTypes[ext]; ok {
			mime = m
		}
	}

	encoded := base64Encode(data)
	logMsg("GEMINI", fmt.Sprintf("fetched media: %d bytes, mime=%s", len(data), mime))
	return encoded, mime, nil
}

// parseDataURI parses "data:mime;base64,DATA" and returns (data, mimeType).
func parseDataURI(uri string) (string, string) {
	segments := strings.SplitN(uri, ",", 2)
	mimeType := strings.TrimPrefix(strings.TrimSuffix(segments[0], ";base64"), "data:")
	data := ""
	if len(segments) > 1 {
		data = segments[1]
	}
	return data, mimeType
}

// toGeminiParts converts our ContentParts to Gemini parts.
func toGeminiParts(parts []ContentPart) []geminiPart {
	var out []geminiPart
	for _, p := range parts {
		switch p.Type {
		case "text":
			out = append(out, geminiPart{Text: p.Text})
		case "image_url":
			if p.ImageURL == nil {
				continue
			}
			if strings.HasPrefix(p.ImageURL.URL, "data:") {
				data, mime := parseDataURI(p.ImageURL.URL)
				out = append(out, geminiPart{InlineData: &geminiInline{MimeType: mime, Data: data}})
			} else {
				// Fetch URL and encode
				data, mime, err := fetchMediaAsBase64(p.ImageURL.URL)
				if err != nil {
					logMsg("GEMINI", fmt.Sprintf("image fetch error: %v", err))
					out = append(out, geminiPart{Text: fmt.Sprintf("[image fetch failed: %s]", p.ImageURL.URL)})
				} else {
					out = append(out, geminiPart{InlineData: &geminiInline{MimeType: mime, Data: data}})
				}
			}
		case "audio_url":
			if p.AudioURL == nil {
				continue
			}
			if strings.HasPrefix(p.AudioURL.URL, "data:") {
				data, mime := parseDataURI(p.AudioURL.URL)
				out = append(out, geminiPart{InlineData: &geminiInline{MimeType: mime, Data: data}})
			} else {
				data, mime, err := fetchMediaAsBase64(p.AudioURL.URL)
				if err != nil {
					logMsg("GEMINI", fmt.Sprintf("audio fetch error: %v", err))
					out = append(out, geminiPart{Text: fmt.Sprintf("[audio fetch failed: %s]", p.AudioURL.URL)})
				} else {
					if p.AudioURL.MimeType != "" {
						mime = p.AudioURL.MimeType
					}
					out = append(out, geminiPart{InlineData: &geminiInline{MimeType: mime, Data: data}})
				}
			}
		case "input_audio":
			if p.InputAudio != nil {
				mimeType := "audio/wav"
				if m, ok := audioMimeTypes[p.InputAudio.Format]; ok {
					mimeType = m
				}
				out = append(out, geminiPart{InlineData: &geminiInline{MimeType: mimeType, Data: p.InputAudio.Data}})
			}
		}
	}
	if len(out) == 0 {
		out = append(out, geminiPart{Text: ""})
	}
	return out
}
