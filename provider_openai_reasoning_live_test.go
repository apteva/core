package core

import (
	"context"
	"strings"
	"testing"
	"time"
)

// Opt in with RUN_LLM_INTEGRATION_TESTS=1 and OPENCODE_GO_API_KEY. The
// conversation replays a real ordinary assistant reply with reasoning before
// exercising a tool call and its continuation, with tools supplied throughout.
func TestIntegration_OpenCodeGo_DeepSeekReasoningReplay(t *testing.T) {
	p := NewOpenCodeGoProvider(getOpenCodeGoKey(t))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	ctx = (&Thinker{threadID: "deepseek-reasoning-replay"}).providerSessionContext(ctx, "inference")
	const model = "deepseek-v4.1-flash"
	tools := []NativeTool{{
		Name:        "inspect_source",
		Description: "Read the Go function currently under review.",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
	}}
	messages := []Message{
		{Role: "system", Content: "Review small Go functions. When the user supplies source, answer directly without tools. When asked to inspect source, call inspect_source exactly once, then output the corrected function after receiving the result. Keep answers brief."},
		{Role: "user", Content: "The add function should add its arguments. Correct this source, returning only the function: func add(a, b int) int { return a - b }"},
	}
	first, err := p.Chat(ctx, messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("initial request: %v", err)
	}
	if len(first.ToolCalls) != 0 || strings.TrimSpace(first.Reasoning) == "" || strings.TrimSpace(first.Text) == "" {
		t.Fatal("expected an ordinary assistant reply with reasoning and no tool calls")
	}
	messages = append(messages,
		Message{Role: "assistant", Content: first.Text, Reasoning: first.Reasoning},
		Message{Role: "user", Content: "Now inspect the current source using inspect_source, then correct it."},
	)
	second, err := p.Chat(ctx, messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("ordinary assistant reasoning replay: %v", err)
	}
	if len(second.ToolCalls) != 1 || second.ToolCalls[0].Name != "inspect_source" || strings.TrimSpace(second.Reasoning) == "" {
		t.Fatal("expected one inspect_source call with reasoning")
	}
	if err := validateToolCalls(second.ToolCalls, "provider"); err != nil {
		t.Fatal(err)
	}
	messages = append(messages,
		Message{Role: "assistant", Content: second.Text, Reasoning: second.Reasoning, ToolCalls: second.ToolCalls},
		Message{Role: "tool", ToolResults: []ToolResult{{CallID: second.ToolCalls[0].ID, Content: "func add(a, b int) int { return a - b }"}}},
	)
	if err := validateToolHistory(messages); err != nil {
		t.Fatal(err)
	}
	third, err := p.Chat(ctx, messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("ordinary assistant and tool-call reasoning replay: %v", err)
	}
	if len(third.ToolCalls) != 0 || !strings.Contains(strings.Join(strings.Fields(third.Text), " "), "return a + b") {
		t.Fatal("continuation did not return the corrected addition function")
	}
	t.Logf("%s: three accepted requests, ordinary reply and tool-call reasoning replay passed", model)
}
