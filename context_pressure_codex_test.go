package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestCodexLargeToolResultPreservedSmoke is a release-gate smoke for
// the real Codex provider and the context-pressure guard. Deterministic
// unit tests prove pressure compaction preserves retained tool results;
// this smoke proves a live model can still see data past the old local
// tool-result cap boundary.
//
//	RUN_CODEX_CONTEXT_PRESSURE_SMOKE=1 OPENAI_CODEX_ACCESS_TOKEN=... go test -run TestCodexLargeToolResultPreservedSmoke -timeout 5m .
func TestCodexLargeToolResultPreservedSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_CONTEXT_PRESSURE_SMOKE") == "" {
		t.Skip("set RUN_CODEX_CONTEXT_PRESSURE_SMOKE=1 to run the Codex context-pressure smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex context-pressure smoke in short mode")
	}
	token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"))
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set")
	}
	runLargeToolResultPreservedSmoke(t, NewOpenAICodexProvider(token))
}

func runLargeToolResultPreservedSmoke(t *testing.T, provider LLMProvider) {
	t.Helper()
	const sentinel = "FULL_RESULT_SENTINEL_9F4C2A"
	fullResult := "BEGIN-CATALOG\n" +
		strings.Repeat("catalog item before sentinel\n", 320) +
		sentinel + "\n" +
		strings.Repeat("catalog item after sentinel\n", 80)
	if strings.Index(fullResult, sentinel) < 4000 {
		t.Fatalf("test setup failed: sentinel offset=%d, want past old cap boundary", strings.Index(fullResult, sentinel))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	resp, err := provider.Chat(ctx, []Message{
		{Role: "system", Content: "You verify tool-result handling. Follow the user's requested exact output."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "huge_catalog", Args: map[string]string{}}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: fullResult}}},
		{Role: "user", Content: "If the tool result contains " + sentinel + ", reply exactly: RESULT: full tool result preserved"},
	}, provider.Models()[ModelMedium], nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("Codex context-pressure smoke failed: %v", err)
	}
	if !strings.Contains(resp.Text, "RESULT: full tool result preserved") {
		t.Fatalf("Codex did not acknowledge full tool result; text=%q", resp.Text)
	}
}

// TestCodexSemanticCompactionSmoke proves the live Codex provider can
// produce a continuation summary from older context while core preserves
// the latest full tool result verbatim in the retained tail.
//
//	RUN_CODEX_CONTEXT_PRESSURE_SMOKE=1 OPENAI_CODEX_ACCESS_TOKEN=... go test -run TestCodexSemanticCompactionSmoke -timeout 5m .
func TestCodexSemanticCompactionSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_CONTEXT_PRESSURE_SMOKE") == "" {
		t.Skip("set RUN_CODEX_CONTEXT_PRESSURE_SMOKE=1 to run the Codex context-pressure smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex context-pressure smoke in short mode")
	}
	token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"))
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set")
	}
	runSemanticCompactionSmoke(t, NewOpenAICodexProvider(token))
}

func runSemanticCompactionSmoke(t *testing.T, provider LLMProvider) {
	t.Helper()
	const oldIdentifier = "CONSULTING_LEAD_ALPHA_77"
	const sentinel = "FULL_RESULT_SENTINEL_2B85D1"
	fullResult := "BEGIN-CATALOG\n" +
		strings.Repeat("retained item before sentinel\n", 320) +
		sentinel + "\n" +
		strings.Repeat("retained item after sentinel\n", 80)

	messages := []Message{{Role: "system", Content: "You are a lead-management agent."}}
	for i := 0; i < contextPressureKeepRecent+8; i++ {
		content := "Older work log: reviewed consulting leads and maintained outreach state."
		if i == 4 {
			content += " Exact lead identifier to preserve in compaction: " + oldIdentifier + ". Email alpha77@example.com. Status: qualified for follow-up."
		}
		messages = append(messages, Message{Role: "user", Content: content})
	}
	messages = append(messages,
		Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "huge_catalog", Args: map[string]string{}}}},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: fullResult}}},
	)

	thinker := &Thinker{
		provider: provider,
		threadID: "main",
		messages: messages,
		quit:     make(chan struct{}),
	}
	reduced := thinker.compactForContextPressure("live_smoke", TokenUsage{PromptTokens: 205_000}, 1)
	if !reduced {
		t.Fatal("semantic compaction did not reduce context")
	}
	if len(thinker.messages) < 4 {
		t.Fatalf("messages after compaction too short: %d", len(thinker.messages))
	}
	if !strings.Contains(thinker.messages[1].Content, oldIdentifier) {
		t.Fatalf("semantic compaction summary did not preserve %s: %q", oldIdentifier, thinker.messages[1].Content)
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 || !strings.Contains(last.ToolResults[0].Content, sentinel) {
		t.Fatalf("latest full tool result was not preserved: %+v", last.ToolResults)
	}
}
