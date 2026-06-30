package core

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

type testCompactionProvider struct {
	response string
	err      error
	calls    int
	prompts  [][]Message
}

func (p *testCompactionProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(toolName, callID, chunk string)) (ChatResponse, error) {
	p.calls++
	p.prompts = append(p.prompts, append([]Message(nil), messages...))
	if p.err != nil {
		return ChatResponse{}, p.err
	}
	return ChatResponse{
		Text: p.response,
		Usage: TokenUsage{
			PromptTokens:     123,
			CompletionTokens: 45,
		},
	}, nil
}

func (p *testCompactionProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{
		ModelLarge:  "large-test",
		ModelMedium: "medium-test",
		ModelSmall:  "small-test",
	}
}

func (p *testCompactionProvider) Name() string { return "test-compactor" }
func (p *testCompactionProvider) CostPer1M() (float64, float64, float64) {
	return 0, 0, 0
}
func (p *testCompactionProvider) SupportsNativeTools() bool { return true }
func (p *testCompactionProvider) AvailableBuiltinTools() []BuiltinTool {
	return nil
}
func (p *testCompactionProvider) SetBuiltinTools(tools []string) {}
func (p *testCompactionProvider) WithBuiltins(builtins []string) LLMProvider {
	return p
}

func TestShouldCompactBeforeLLMWhenContextFallbackExceeded(t *testing.T) {
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+5; i++ {
		messages = append(messages, Message{Role: "user", Content: strings.Repeat("x", 24_000)})
	}
	if !shouldCompactBeforeLLM("unknown-model", messages) {
		t.Fatal("expected pre-LLM compaction when fallback context pressure is exceeded")
	}
}

func TestShouldCompactBeforeLLMFalseWhenNoCompactionRoom(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: strings.Repeat("x", contextPressureCharFallback+1)},
	}
	if shouldCompactBeforeLLM("unknown-model", messages) {
		t.Fatal("did not expect pre-LLM compaction when there is no older context to remove")
	}
}

func TestShouldRecoverFromEmptyResponseNearContextLimit(t *testing.T) {
	messages := []Message{{Role: "system", Content: "small"}}
	if !shouldRecoverFromEmptyResponse(TokenUsage{PromptTokens: 205_000}, "kimi-k2.6", messages, 1) {
		t.Fatal("expected recovery near 256k context limit")
	}
}

func TestShouldRecoverFromEmptyResponseAfterRepeatedEmptyUnknownUsage(t *testing.T) {
	messages := []Message{{Role: "system", Content: "small"}}
	if shouldRecoverFromEmptyResponse(TokenUsage{}, "unknown-model", messages, 1) {
		t.Fatal("did not expect recovery on first empty response without pressure signal")
	}
	if !shouldRecoverFromEmptyResponse(TokenUsage{}, "unknown-model", messages, contextPressureEmptyStreak) {
		t.Fatal("expected recovery after repeated empty responses")
	}
}

func TestSessionForceCompactPreservesRecentOversizedToolResult(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	for i := 0; i < contextPressureKeepRecent+5; i++ {
		session.Append(SessionEntry{Role: "user", Content: "message"})
	}
	session.Append(SessionEntry{
		Role:      "assistant",
		ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}},
	})
	large := strings.Repeat("y", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("z", 16_000)
	session.Append(SessionEntry{
		Role: "user",
		ToolResults: []ToolResult{{
			CallID:  "call-1",
			Content: large,
		}},
	})

	session.ForceCompact(contextPressureKeepRecent, func(text string) string {
		return "pressure summary"
	})

	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) != contextPressureKeepRecent+1 {
		t.Fatalf("entry count = %d, want %d", len(entries), contextPressureKeepRecent+1)
	}
	last := entries[len(entries)-1]
	if len(last.ToolResults) != 1 {
		t.Fatalf("expected retained tool result, got %+v", last.ToolResults)
	}
	if last.ToolResults[0].Content != large {
		t.Fatalf("tool result changed: len=%d want=%d contains_sentinel=%v",
			len(last.ToolResults[0].Content), len(large), strings.Contains(last.ToolResults[0].Content, "FULL_RESULT_SENTINEL"))
	}
}

func TestTrimMessagesForContextPressurePreservesFullToolResultAndPair(t *testing.T) {
	large := strings.Repeat("a", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("b", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < 30; i++ {
		messages = append(messages, Message{Role: "user", Content: strings.Repeat("old", 100)})
	}
	messages = append(messages,
		Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}},
	)
	for i := 0; i < contextPressureKeepRecent-1; i++ {
		messages = append(messages, Message{Role: "user", Content: "later"})
	}

	trimmed := trimMessagesForContextPressure(messages, contextPressureKeepRecent)
	if len(trimmed) != contextPressureKeepRecent+2 {
		t.Fatalf("trimmed messages = %d, want %d", len(trimmed), contextPressureKeepRecent+2)
	}

	foundCall := false
	foundResult := false
	for _, msg := range trimmed {
		for _, call := range msg.ToolCalls {
			if call.ID == "call-1" {
				foundCall = true
			}
		}
		for _, result := range msg.ToolResults {
			if result.CallID == "call-1" {
				foundResult = true
				if result.Content != large {
					t.Fatalf("tool result changed: len=%d want=%d", len(result.Content), len(large))
				}
			}
		}
	}
	if !foundCall {
		t.Fatal("matching tool call was not preserved")
	}
	if !foundResult {
		t.Fatal("tool result was not preserved")
	}
}

func TestThinkerCompactForContextPressurePreservesOversizedToolResult(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	for i := 0; i < contextPressureKeepRecent+10; i++ {
		msg := Message{Role: "user", Content: "message"}
		if i == contextPressureKeepRecent+8 {
			msg = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
		}
		if i == contextPressureKeepRecent+9 {
			msg = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
		}
		session.AppendMessage(msg, i, TokenUsage{})
	}
	thinker := &Thinker{
		session:  session,
		threadID: "main",
		messages: append([]Message{{Role: "system", Content: "system"}},
			make([]Message, contextPressureKeepRecent+10)...),
	}
	for i := 1; i < len(thinker.messages); i++ {
		thinker.messages[i] = Message{Role: "user", Content: strings.Repeat("m", 100)}
	}
	thinker.messages[len(thinker.messages)-2] = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	thinker.messages[len(thinker.messages)-1] = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}

	reduced := thinker.compactForContextPressure("test", TokenUsage{PromptTokens: 205_000}, 1)
	if !reduced {
		t.Fatal("expected context pressure compaction to report progress")
	}

	if len(thinker.messages) != contextPressureKeepRecent+1 {
		t.Fatalf("messages = %d, want %d", len(thinker.messages), contextPressureKeepRecent+1)
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 {
		t.Fatalf("expected last message tool result, got %+v", last)
	}
	if last.ToolResults[0].Content != large {
		t.Fatalf("in-memory tool result changed: len=%d want=%d", len(last.ToolResults[0].Content), len(large))
	}
}

func TestThinkerSemanticCompactionSummarizesOldContextAndPreservesLatestToolResult(t *testing.T) {
	provider := &testCompactionProvider{
		response: strings.Join([]string{
			"# Objective",
			"- Continue the lead workflow.",
			"## Completed Work",
			"- CONSULTING_LEAD_ALPHA was qualified from the older context.",
			"## Current State",
			"- Latest tool result remains in the retained tail.",
			"## Important Tool Results",
			"- None known",
			"## Decisions And Constraints",
			"- Preserve emails.",
			"## Open Tasks",
			"- Follow up.",
			"## Risks And Failed Attempts",
			"- None known",
		}, "\n"),
	}
	session := NewSession(t.TempDir(), "main")
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+8; i++ {
		content := strings.Repeat("older context ", 300)
		if i == 3 {
			content += " CONSULTING_LEAD_ALPHA email alpha@example.com qualified"
		}
		msg := Message{Role: "user", Content: content}
		messages = append(messages, msg)
		session.AppendMessage(msg, i, TokenUsage{})
	}
	callMsg := Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	resultMsg := Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
	messages = append(messages, callMsg, resultMsg)
	session.AppendMessage(callMsg, 99, TokenUsage{})
	session.AppendMessage(resultMsg, 100, TokenUsage{})

	thinker := &Thinker{
		provider:  provider,
		session:   session,
		threadID:  "main",
		messages:  messages,
		quit:      make(chan struct{}),
		telemetry: &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{PromptTokens: 205_000}, 1)
	if !reduced {
		t.Fatal("expected semantic compaction to reduce context")
	}
	if provider.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls)
	}
	if len(provider.prompts) != 1 || !strings.Contains(provider.prompts[0][1].Content, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("compaction prompt did not include old context: %+v", provider.prompts)
	}
	if len(thinker.messages) < 4 {
		t.Fatalf("messages after compaction too short: %d", len(thinker.messages))
	}
	if !strings.Contains(thinker.messages[1].Content, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("semantic summary missing from compacted context: %q", thinker.messages[1].Content)
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 || last.ToolResults[0].Content != large {
		t.Fatalf("latest full tool result was not preserved: %+v", last.ToolResults)
	}

	events, _ := thinker.telemetry.StoredEvents(0)
	seenStarted := false
	seenDone := false
	seenContextCompat := false
	for _, ev := range events {
		switch ev.Type {
		case "llm.compaction_started":
			seenStarted = true
		case "llm.compaction_done":
			seenDone = true
			var data map[string]any
			if err := json.Unmarshal(ev.Data, &data); err != nil {
				t.Fatalf("done telemetry JSON: %v", err)
			}
			if data["mode"] != "semantic" {
				t.Fatalf("done telemetry mode = %v, want semantic", data["mode"])
			}
		case "llm.context_compacted":
			seenContextCompat = true
		}
	}
	if !seenStarted || !seenDone || !seenContextCompat {
		t.Fatalf("missing compaction telemetry: started=%v done=%v context=%v", seenStarted, seenDone, seenContextCompat)
	}

	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) == 0 || entries[0].Role != "_compacted" {
		t.Fatalf("session was not compacted with summary: %+v", entries)
	}
	if !strings.Contains(entries[0].Summary, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("session summary missing semantic content: %q", entries[0].Summary)
	}
}

func TestThinkerSemanticCompactionNoReductionFallsBackToEmergencyTrim(t *testing.T) {
	provider := &testCompactionProvider{response: strings.Repeat("oversized summary ", 80_000)}
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+10; i++ {
		messages = append(messages, Message{Role: "user", Content: "old"})
	}
	messages[len(messages)-2] = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	messages[len(messages)-1] = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
	thinker := &Thinker{
		provider:  provider,
		threadID:  "main",
		messages:  messages,
		quit:      make(chan struct{}),
		telemetry: &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{}, 1)
	if !reduced {
		t.Fatal("expected fallback trim to reduce context")
	}
	if strings.Contains(thinker.messages[1].Content, "oversized summary") {
		t.Fatal("accepted semantic summary that did not reduce context")
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 || last.ToolResults[0].Content != large {
		t.Fatalf("fallback did not preserve latest full tool result: %+v", last.ToolResults)
	}
}

func TestThinkerCompactForContextPressureReportsNoProgress(t *testing.T) {
	large := strings.Repeat("x", contextPressureCharFallback+1)
	thinker := &Thinker{
		threadID: "main",
		messages: []Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: large},
		},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{}, 2)
	if reduced {
		t.Fatal("did not expect progress when there is no old context to remove")
	}
	if thinker.messages[1].Content != large {
		t.Fatal("expected full message content to remain unchanged")
	}
}
