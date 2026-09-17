package core

import (
	"context"
	"strings"
	"testing"
	"time"
)

// Real coding-agent traffic, including a native tool result continuation.
// RUN_LLM_INTEGRATION_TESTS=1 OPENCODE_GO_API_KEY=... go test -run
// TestIntegration_OpenCodeGo_KimiK3SessionToolRoundTrip -v .
func TestIntegration_OpenCodeGo_KimiK3SessionToolRoundTrip(t *testing.T) {
	p := NewOpenCodeGoProvider(getOpenCodeGoKey(t))
	owner := &Thinker{threadID: "opencode-session-regression"}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	ctx = owner.providerSessionContext(ctx, "inference")
	tools := []NativeTool{{
		Name:        "inspect_source",
		Description: "Read the Go function under review. Call this before proposing a correction.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	messages := []Message{
		{Role: "system", Content: "You are reviewing a small Go function. First call inspect_source exactly once. After receiving its result, output only the corrected Go function. Do not call the tool again."},
		{Role: "user", Content: "The add function subtracts instead of adding. Inspect the source using the tool, then correct it."},
	}
	started := time.Now()
	first, err := p.Chat(ctx, messages, "kimi-k3", tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("initial request: %v", err)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].Name != "inspect_source" || first.ToolCalls[0].ID == "" {
		t.Fatalf("expected one inspect_source call, got %d tool calls", len(first.ToolCalls))
	}
	if err := validateToolCalls(first.ToolCalls, "provider"); err != nil {
		t.Fatal(err)
	}
	messages = append(messages,
		Message{Role: "assistant", Content: first.Text, Reasoning: first.Reasoning, ToolCalls: first.ToolCalls},
		Message{Role: "tool", ToolResults: []ToolResult{{CallID: first.ToolCalls[0].ID, Content: "package sample\nfunc add(a, b int) int { return a - b }"}}},
	)
	if err := validateToolHistory(messages); err != nil {
		t.Fatal(err)
	}
	second, err := p.WithBuiltins(nil).Chat(ctx, messages, "kimi-k3", tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("tool-result continuation: %v", err)
	}
	if len(second.ToolCalls) != 0 || !strings.Contains(strings.Join(strings.Fields(second.Text), " "), "return a + b") {
		t.Fatal("continuation did not produce the corrected addition function")
	}
	t.Logf("kimi-k3: two accepted requests, native tool round trip passed; elapsed=%s input_tokens=%d output_tokens=%d", time.Since(started).Round(time.Millisecond), first.Usage.PromptTokens+second.Usage.PromptTokens, first.Usage.CompletionTokens+second.Usage.CompletionTokens)
}
