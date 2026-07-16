package core

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestOpenAIPromptCacheScopeSeparatesAgentThreadAndEpoch(t *testing.T) {
	base := openAIPromptCacheHintsForScope("openai", "gpt-5.5", "system", []any{"tool"}, openAIPromptCacheScope{
		Identity: "agent:10/thread:main",
		Epoch:    3,
	})
	if !strings.HasPrefix(base.Key, "apteva-v2-") || base.Retention != "24h" {
		t.Fatalf("base hints = %+v, want scoped v2 key with existing 24h retention", base)
	}
	if same := openAIPromptCacheHintsForScope("openai", "gpt-5.5", "system", []any{"tool"}, openAIPromptCacheScope{
		Identity: "agent:10/thread:main",
		Epoch:    3,
	}); same != base {
		t.Fatalf("same scope was not stable: %+v vs %+v", base, same)
	}
	for _, changed := range []openAIPromptCacheScope{
		{Identity: "agent:11/thread:main", Epoch: 3},
		{Identity: "agent:10/thread:worker", Epoch: 3},
		{Identity: "agent:10/thread:main", Epoch: 4},
	} {
		got := openAIPromptCacheHintsForScope("openai", "gpt-5.5", "system", []any{"tool"}, changed)
		if got.Key == base.Key {
			t.Fatalf("scope %+v reused base key %q", changed, got.Key)
		}
		if got.Retention != "24h" {
			t.Fatalf("scope %+v changed retention to %q", changed, got.Retention)
		}
	}
}

func TestPromptCacheDiagnosticsTrackAppendOnlyRequestsAndToolReset(t *testing.T) {
	t.Setenv("AGENT_ID", "10")
	thinker := &Thinker{threadID: "main", promptCacheResetReason: "startup"}
	tools := []NativeTool{{Name: "lookup", Description: "Lookup", Parameters: map[string]any{"type": "object"}}}
	first := []Message{
		{Role: "system", Content: "stable system"},
		{Role: "user", Content: "task"},
	}
	ctx := thinker.preparePromptCacheContext(context.Background(), first, tools)
	scope := openAIPromptCacheScopeFromContext(ctx)
	if scope.Identity != "agent:10/thread:main" || scope.Epoch != 0 {
		t.Fatalf("scope = %+v", scope)
	}
	firstBytes := len(thinker.promptCachePreviousRequest)
	firstStableHash := thinker.promptCacheStableHash

	second := append(cloneMessages(first),
		Message{Role: "assistant", Content: "working"},
		Message{Role: "user", Content: "next"},
	)
	thinker.preparePromptCacheContext(context.Background(), second, tools)
	if thinker.promptCacheCommonPrefixBytes != firstBytes {
		t.Fatalf("common prefix = %d, want complete prior request %d", thinker.promptCacheCommonPrefixBytes, firstBytes)
	}
	if thinker.promptCacheEpoch != 0 || thinker.promptCacheStableHash != firstStableHash {
		t.Fatalf("normal append changed cache epoch/hash: epoch=%d stable=%q", thinker.promptCacheEpoch, thinker.promptCacheStableHash)
	}

	changedTools := append(tools, NativeTool{Name: "second", Description: "Second", Parameters: map[string]any{"type": "object"}})
	ctx = thinker.preparePromptCacheContext(context.Background(), second, changedTools)
	if thinker.promptCacheEpoch != 1 || thinker.promptCacheResetReason != "stable_prefix_changed" {
		t.Fatalf("tool prefix change did not advance epoch: epoch=%d reason=%q", thinker.promptCacheEpoch, thinker.promptCacheResetReason)
	}
	if scope := openAIPromptCacheScopeFromContext(ctx); scope.Epoch != 1 {
		t.Fatalf("provider scope epoch = %d, want 1", scope.Epoch)
	}

	requestHash := thinker.promptCacheRequestHash
	thinker.resetPromptCache("directive_evolved")
	if thinker.promptCacheRequestEpoch != 1 || thinker.promptCacheRequestReason != "stable_prefix_changed" || thinker.promptCacheRequestHash != requestHash {
		t.Fatalf("post-response reset rewrote diagnostics for the completed request: epoch=%d reason=%q hash=%q",
			thinker.promptCacheRequestEpoch, thinker.promptCacheRequestReason, thinker.promptCacheRequestHash)
	}
}

func TestPrepareComputerScreenshotTailDoesNotMutateHistory(t *testing.T) {
	original := []Message{{Role: "system", Content: "system"}}
	for i, pixels := range [][]byte{computerScreenshotJPEG(t, 90_000), computerScreenshotJPEG(t, 100_000)} {
		id := "call-" + string(rune('1'+i))
		original = append(original,
			Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "computer_computer_use"}}},
			Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: "screen metadata " + id, Image: pixels}}},
		)
	}
	before := cloneMessages(original)
	projected := prepareComputerScreenshotTail(original)
	if !reflect.DeepEqual(original, before) {
		t.Fatal("screenshot projection mutated durable history")
	}
	if len(projected) != len(original)+1 {
		t.Fatalf("projected messages = %d, want %d", len(projected), len(original)+1)
	}
	for _, message := range projected[:len(original)] {
		for _, result := range message.ToolResults {
			if len(result.Image) != 0 {
				t.Fatalf("historical tool result retained inline pixels: %s", result.CallID)
			}
			if !strings.Contains(result.Content, "screen metadata") {
				t.Fatalf("historical tool text was rewritten: %q", result.Content)
			}
		}
	}
	tail := projected[len(projected)-1]
	if !tail.RequestContext || len(tail.Parts) != 2 || tail.Parts[1].ImageURL == nil {
		t.Fatalf("screenshot tail = %+v", tail)
	}
	if !strings.Contains(tail.Parts[0].Text, "call-2") || !strings.HasPrefix(tail.Parts[1].ImageURL.URL, "data:image/jpeg;base64,") {
		t.Fatalf("tail did not contain latest frame: %+v", tail.Parts)
	}

	input := (&OpenAINativeProvider{}).buildInput(projected)
	if got := input[len(input)-1]; got.Type != "message" || got.Role != "user" {
		t.Fatalf("native screenshot tail = %+v", got)
	}
}

func TestCheckpointHistoryWindowUsesHysteresis(t *testing.T) {
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < 120; i++ {
		messages = append(messages, Message{Role: "user", Content: "message"})
	}
	if got, dropped := checkpointHistoryWindow(messages, maxHistoryMain); dropped != 0 || len(got) != len(messages) {
		t.Fatalf("checkpoint triggered at upper boundary: dropped=%d len=%d", dropped, len(got))
	}
	messages = append(messages, Message{Role: "user", Content: "trigger"})
	got, dropped := checkpointHistoryWindow(messages, maxHistoryMain)
	if dropped == 0 {
		t.Fatal("checkpoint did not trigger above upper boundary")
	}
	if len(got) != 81 {
		t.Fatalf("checkpoint retained %d messages, want system + 80", len(got))
	}
	if got[0].Role != "system" || got[len(got)-1].Content != "trigger" {
		t.Fatalf("checkpoint lost system/latest messages: first=%+v last=%+v", got[0], got[len(got)-1])
	}
}

func TestPromptCacheResetTelemetryIncludesEpochAndReason(t *testing.T) {
	t.Setenv("AGENT_ID", "10")
	telemetry := NewTelemetry()
	defer telemetry.Stop()
	thinker := &Thinker{threadID: "main", telemetry: telemetry}
	thinker.advancePromptCacheEpoch("history_checkpoint", true, map[string]any{"dropped_messages": 40})
	events, _ := telemetry.StoredEvents(0)
	if len(events) != 1 || events[0].Type != "llm.prompt_cache_reset" {
		t.Fatalf("events = %+v", events)
	}
	var data map[string]any
	if err := json.Unmarshal(events[0].Data, &data); err != nil {
		t.Fatal(err)
	}
	if data["cache_epoch"] != float64(1) || data["reset_reason"] != "history_checkpoint" || data["dropped_messages"] != float64(40) {
		t.Fatalf("reset telemetry = %+v", data)
	}
	if data["identity_hash"] == "" {
		t.Fatalf("reset telemetry omitted identity hash: %+v", data)
	}
}
