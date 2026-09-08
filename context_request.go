package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Estimates are deliberately labelled: serialized bytes are exact, tokens are
// conservative approximations (including opaque state), not tokenizer counts.
type requestBudget struct {
	Provider           string `json:"provider"`
	Model              string `json:"model"`
	Stage              string `json:"stage"`
	Fingerprint        string `json:"request_fingerprint"`
	SerializedBytes    int    `json:"serialized_bytes"`
	SystemTokens       int    `json:"system_tokens_est"`
	ConversationTokens int    `json:"conversation_tokens_est"`
	ToolTokens         int    `json:"tool_tokens_est"`
	ImageTokens        int    `json:"image_tokens_est"`
	OpaqueTokens       int    `json:"opaque_tokens_est"`
	InputTokens        int    `json:"input_tokens_est"`
	ReservedOutput     int    `json:"reserved_output_tokens"`
	ContextWindow      int    `json:"context_window"`
	InputBudget        int    `json:"input_budget"`
	OverBudget         bool   `json:"over_budget"`
}

type contextBudgetError struct{ Budget requestBudget }

func (e *contextBudgetError) Error() string {
	return fmt.Sprintf("context_length_exceeded: estimated input %d exceeds safe input budget %d (window %d, reserved output %d, stage %s)", e.Budget.InputTokens, e.Budget.InputBudget, e.Budget.ContextWindow, e.Budget.ReservedOutput, e.Budget.Stage)
}

type contextManagementError struct{ Cause error }

func (e *contextManagementError) Error() string {
	return "context_management_failed: " + e.Cause.Error()
}
func (e *contextManagementError) Unwrap() error { return e.Cause }
func isContextLengthError(err error) bool {
	if err == nil {
		return false
	}
	var budget *contextBudgetError
	if errors.As(err, &budget) {
		return true
	}
	s := strings.ToLower(err.Error())
	for _, needle := range []string{"context_length_exceeded", "maximum context length", "exceeds the context window", "prompt is too long", "input is too long", "input token count exceeds", "request too large for model"} {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}
func providerExecutionFailureReason(err error) string {
	var management *contextManagementError
	if errors.As(err, &management) || isContextLengthError(err) {
		return "context_management_failed: " + err.Error()
	}
	return "provider_retry_budget_exhausted: " + err.Error()
}

func (b *requestBudget) finish(output int) {
	b.ContextWindow = ModelEffectiveContextWindow(b.Model)
	if b.ContextWindow <= 0 {
		b.ContextWindow = contextPressureCharFallback / 4
	}
	if output <= 0 {
		output = 16384
	}
	b.ReservedOutput = output
	b.InputBudget = b.ContextWindow - output - b.ContextWindow/10
	if b.InputBudget < 0 {
		b.InputBudget = 0
	}
	b.InputTokens = b.SystemTokens + b.ConversationTokens + b.ToolTokens + b.ImageTokens + b.OpaqueTokens
	b.OverBudget = b.InputTokens > b.InputBudget
}
func requestFingerprint(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}
func estimatePreparedRequest(provider, model string, messages []Message, tools []NativeTool) requestBudget {
	b := requestBudget{Provider: provider, Model: model, Stage: "prepared"}
	for _, msg := range messages {
		n := estimatedContextTokens([]Message{msg}) + 8
		if msg.Role == "system" {
			b.SystemTokens += n
		} else {
			b.ConversationTokens += n
		}
	}
	rawTools, _ := json.Marshal(tools)
	b.ToolTokens = (len(rawTools) + 3) / 4
	raw, _ := json.Marshal(struct {
		Messages []Message
		Tools    []NativeTool
	}{messages, tools})
	b.SerializedBytes = len(raw)
	b.Fingerprint = requestFingerprint(raw)
	reserve := 16384
	if provider == "anthropic" {
		reserve = anthropicMaxTokens(model)
	}
	b.finish(reserve)
	return b
}

type requestObserverKey struct{}
type requestObserver func(requestBudget)

// The adapter calls this on the exact bytes immediately before HTTP submission,
// after schemas, provider state, images and provider-specific options are added.
func observeProviderRequest(ctx context.Context, provider, model string, body []byte) error {
	observer, ok := ctx.Value(requestObserverKey{}).(requestObserver)
	if !ok {
		return nil
	} // Independent provider clients keep their existing API.
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return err
	}
	b := requestBudget{Provider: provider, Model: model, Stage: "serialized", SerializedBytes: len(body), Fingerprint: requestFingerprint(body)}
	// Walk provider structures rather than dividing base64 media bytes by four.
	// A base64 string inside ordinary tool arguments IS text and is counted.
	var walk func(any) int
	walk = func(v any) int {
		switch x := v.(type) {
		case string:
			return (len(x) + 3) / 4
		case []any:
			n := 0
			for _, item := range x {
				n += walk(item)
			}
			return n
		case map[string]any:
			role, _ := x["role"].(string)
			if role == "system" || role == "developer" {
				b.SystemTokens += walk(x["content"]) + 4
				return 0
			}
			typ, _ := x["type"].(string)
			if typ == "input_image" || typ == "image_url" || typ == "image" {
				imageTokens := defaultImageTokenEstimate
				url, _ := x["image_url"].(string)
				if im, ok := x["image_url"].(map[string]any); ok {
					url, _ = im["url"].(string)
				}
				if data, ok := decodeImageDataURL(url); ok {
					imageTokens = estimateImageTokens(data)
				}
				b.ImageTokens += imageTokens
				return 0
			}
			if inline, ok := x["inlineData"].(map[string]any); ok {
				mime, _ := inline["mimeType"].(string)
				if strings.HasPrefix(mime, "image/") {
					b.ImageTokens += defaultImageTokenEstimate
					return 0
				}
			}
			n := 4
			for k, val := range x {
				if k == "encrypted_content" {
					b.OpaqueTokens += walk(val)
					continue
				}
				n += walk(val) + 1
			}
			return n
		default:
			return 1
		}
	}
	b.SystemTokens = walk(root["instructions"]) + walk(root["system"]) + walk(root["systemInstruction"])
	b.ToolTokens = walk(root["tools"])
	for _, key := range []string{"input", "messages", "contents"} {
		b.ConversationTokens += walk(root[key])
	}
	reserve := 0
	for _, key := range []string{"max_tokens", "max_output_tokens", "max_completion_tokens"} {
		if n, ok := root[key].(float64); ok && int(n) > reserve {
			reserve = int(n)
		}
	}
	if cfg, ok := root["generationConfig"].(map[string]any); ok {
		if n, ok := cfg["maxOutputTokens"].(float64); ok {
			reserve = int(n)
		}
	}
	b.finish(reserve)
	observer(b)
	if b.OverBudget {
		return &contextBudgetError{Budget: b}
	}
	return nil
}

// Preserve both causal identities; errors.As/Is must see primary AND fallback.
type providerChainError struct {
	PrimaryName, FallbackName string
	Primary, Fallback         error
}

func (e *providerChainError) Error() string {
	return fmt.Sprintf("primary %s: %v; fallback %s: %v", e.PrimaryName, e.Primary, e.FallbackName, e.Fallback)
}
func (e *providerChainError) Unwrap() []error { return []error{e.Primary, e.Fallback} }
