package core

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAppendEphemeralTurnContextPreservesDurablePrefix(t *testing.T) {
	base := []Message{
		{Role: "system", Content: "stable system"},
		{Role: "user", Content: "raw event"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "lookup"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: "result"}}},
	}
	request := appendEphemeralTurnContext(base, "[memories — surfaced]\n- relevant fact", "2026-07-10 14:00", false)
	if len(request) != len(base)+1 {
		t.Fatalf("request messages = %d, want %d", len(request), len(base)+1)
	}
	for i := range base {
		if request[i].Role != base[i].Role || request[i].Content != base[i].Content {
			t.Fatalf("durable prefix changed at %d: got=%+v want=%+v", i, request[i], base[i])
		}
	}
	if !strings.HasPrefix(request[len(request)-1].Content, ephemeralContextHeader) {
		t.Fatalf("last request message is not ephemeral context: %q", request[len(request)-1].Content)
	}
	if len(base) != 4 || strings.Contains(base[len(base)-1].Content, "memories") {
		t.Fatal("base conversation was mutated")
	}

	wire := toOpenAIMessages(request)
	if len(wire) < 3 {
		t.Fatalf("OpenAI wire messages too short: %+v", wire)
	}
	nativeInput := (&OpenAINativeProvider{}).buildInput(request)
	if len(nativeInput) < 3 {
		t.Fatalf("OpenAI native input too short: %+v", nativeInput)
	}
	last := nativeInput[len(nativeInput)-1]
	if last.Type != "message" || last.Role != "user" || !strings.HasPrefix(last.Content.(string), ephemeralContextHeader) {
		t.Fatalf("OpenAI native ephemeral context is not last: %+v", last)
	}
	foundToolResult := false
	for _, item := range nativeInput[:len(nativeInput)-1] {
		if item.Type == "function_call_output" && item.CallID == "call-1" {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Fatalf("tool result missing before ephemeral tail: %+v", nativeInput)
	}
}

func TestEphemeralTurnContextStatePreservesAppendOnlyPrefix(t *testing.T) {
	state := ephemeralTurnContextState{}
	durable := []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "task"}}
	first := state.prepare(durable, "[memories — surfaced]\n- relevant", "2026-07-10 14:00", false, true)
	if len(first) != 3 || !first[2].RequestContext {
		t.Fatalf("first request context = %+v", first)
	}

	durable = append(durable,
		Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "lookup"}}},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: "result"}}},
	)
	second := state.prepare(durable, "[memories — surfaced]\n- relevant", "2026-07-10 14:01", false, false)
	if !reflect.DeepEqual(first, second[:len(first)]) {
		t.Fatalf("prior request is not an exact prefix:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if !second[2].RequestContext || second[2].Content != first[2].Content {
		t.Fatalf("request context moved or changed: first=%+v second=%+v", first[2], second[2])
	}
	firstInput := (&OpenAINativeProvider{}).buildInput(first)
	secondInput := (&OpenAINativeProvider{}).buildInput(second)
	if len(secondInput) < len(firstInput) || !reflect.DeepEqual(firstInput, secondInput[:len(firstInput)]) {
		t.Fatalf("native Responses input lost append-only prefix:\nfirst=%+v\nsecond=%+v", firstInput, secondInput)
	}
}

func TestRecallQueryPriorityNeverUsesAssistantFiller(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "[2026-07-10 13:00] Events:\n• [console] Investigate storage uploads"},
		{Role: "assistant", Content: "I'll wait for follow-up."},
	}
	query, source := recallQueryForTurn(nil, messages, "Monitor deployments")
	if source != "active_task" || !strings.Contains(query, "storage uploads") {
		t.Fatalf("query=%q source=%q", query, source)
	}
	if strings.Contains(query, "wait for follow-up") {
		t.Fatalf("assistant filler became recall query: %q", query)
	}

	query, source = recallQueryForTurn([]string{"[console] Check invoices"}, messages, "Monitor deployments")
	if source != "event" || query != "[console] Check invoices" {
		t.Fatalf("event query=%q source=%q", query, source)
	}

	query, source = recallQueryForTurn(nil, []Message{{Role: "system", Content: "system"}}, "Monitor deployments")
	if source != "directive" || query != "Monitor deployments" {
		t.Fatalf("directive query=%q source=%q", query, source)
	}
}

func TestStripLegacyDynamicContextKeepsOnlyRealEvent(t *testing.T) {
	legacy := "[ACTIVE THREADS]\n- worker\n\n[memories — surfaced because they may be relevant]\n- large capability\n\n[2026-07-10 14:00] Events:\n• [console] Real request\n"
	cleaned, synthetic := stripLegacyDynamicContext(legacy)
	if !synthetic {
		t.Fatal("legacy dynamic context was not detected")
	}
	want := "[2026-07-10 14:00] Events:\n• [console] Real request"
	if cleaned != want {
		t.Fatalf("cleaned=%q want=%q", cleaned, want)
	}

	cleaned, synthetic = stripLegacyDynamicContext("[memories — surfaced because they may be relevant]\n- large capability")
	if !synthetic || cleaned != "" {
		t.Fatalf("synthetic-only block cleaned=%q detected=%v", cleaned, synthetic)
	}
}

func TestSessionLoadTailFiltersLegacyDynamicMessages(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "main")
	if err := session.Append(SessionEntry{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- duplicate capability"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- fact\n\n[2026-07-10 14:00] Events:\n• [console] Keep me"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "assistant", Content: "done"}); err != nil {
		t.Fatal(err)
	}

	messages, _ := session.LoadTail(10)
	if len(messages) != 2 {
		t.Fatalf("loaded %d messages, want 2: %+v", len(messages), messages)
	}
	if strings.Contains(messages[0].Content, "memories") || !strings.Contains(messages[0].Content, "Keep me") {
		t.Fatalf("combined legacy message not cleaned: %q", messages[0].Content)
	}

	combined, _, recent := buildCompactionParts([]SessionEntry{
		{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- duplicate capability"},
		{Role: "user", Content: "real task"},
		{Role: "assistant", Content: "done"},
	}, 1)
	if strings.Contains(combined, "duplicate capability") || strings.Contains(combined, "[memories") {
		t.Fatalf("compaction input retained synthetic memory: %q", combined)
	}
	if len(recent) != 1 || recent[0].Content != "done" {
		t.Fatalf("unexpected recent entries: %+v", recent)
	}
}

type retrievalCaptureProvider struct {
	mu       sync.Mutex
	requests [][]Message
	called   chan struct{}
}

func (p *retrievalCaptureProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	copyMessages := append([]Message(nil), messages...)
	p.mu.Lock()
	p.requests = append(p.requests, copyMessages)
	callNumber := len(p.requests)
	p.mu.Unlock()
	select {
	case p.called <- struct{}{}:
	default:
	}
	sleep := "1ms"
	if callNumber >= 4 {
		sleep = "1h"
	}
	return ChatResponse{
		Text: "Continuing standing work.",
		ToolCalls: []NativeToolCall{{
			ID: "pace-" + string(rune('0'+callNumber)), Name: "pace",
			Args: map[string]string{"sleep": sleep, "_reason": "Waiting for work"},
		}},
		Usage: TokenUsage{PromptTokens: contextChars(messages) / 4, CompletionTokens: 8},
	}, nil
}

func (p *retrievalCaptureProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "capture", ModelMedium: "capture", ModelSmall: "capture"}
}
func (p *retrievalCaptureProvider) Name() string                           { return "retrieval-capture" }
func (p *retrievalCaptureProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *retrievalCaptureProvider) SupportsNativeTools() bool              { return true }
func (p *retrievalCaptureProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *retrievalCaptureProvider) SetBuiltinTools([]string)               {}
func (p *retrievalCaptureProvider) WithBuiltins([]string) LLMProvider      { return p }

func TestThinkerRecallIsEphemeralAcrossTurns(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := &retrievalCaptureProvider{called: make(chan struct{}, 8)}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Monitor storage uploads and signed URLs.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	if _, err := thinker.memory.Remember("Use signed URLs for storage uploads.", []string{"procedure", "storage"}, 0.95); err != nil {
		t.Fatalf("remember: %v", err)
	}
	defer thinker.Stop()
	go thinker.Run()

	for i := 0; i < 4; i++ {
		select {
		case <-provider.called:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for provider call %d", i+1)
		}
	}
	thinker.Stop()
	time.Sleep(50 * time.Millisecond)

	provider.mu.Lock()
	requests := append([][]Message(nil), provider.requests...)
	provider.mu.Unlock()
	if len(requests) < 4 {
		t.Fatalf("captured %d requests", len(requests))
	}
	contextIndex := -1
	var contextContent string
	for i, request := range requests[:4] {
		markers := 0
		foundIndex := -1
		for j, message := range request {
			markers += strings.Count(message.Content, "[memories — surfaced")
			if message.RequestContext {
				foundIndex = j
			}
		}
		if markers != 1 {
			t.Fatalf("request %d has %d memory blocks, want exactly 1", i+1, markers)
		}
		if foundIndex < 0 || !strings.HasPrefix(request[foundIndex].Content, ephemeralContextHeader) {
			t.Fatalf("request %d is missing marked ephemeral context: %+v", i+1, request)
		}
		if i == 0 {
			contextIndex = foundIndex
			contextContent = request[foundIndex].Content
		} else {
			if foundIndex != contextIndex || request[foundIndex].Content != contextContent {
				t.Fatalf("request %d moved or changed transient context", i+1)
			}
			previous := requests[i-1]
			if len(request) < len(previous) || !reflect.DeepEqual(previous, request[:len(previous)]) {
				t.Fatalf("request %d is not an append-only extension of request %d", i+1, i)
			}
		}
	}
	history, err := os.ReadFile(filepath.Join("history", "main.jsonl"))
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if strings.Contains(string(history), "[memories — surfaced") || strings.Contains(string(history), ephemeralContextHeader) {
		t.Fatalf("session history contains retrieval context: %s", history)
	}
	events, _ := thinker.telemetry.StoredEvents(0)
	foundRecall := false
	for _, event := range events {
		if event.Type == "memory.recall" && strings.Contains(string(event.Data), `"ephemeral":true`) {
			foundRecall = true
			break
		}
	}
	if !foundRecall {
		t.Fatal("missing ephemeral memory.recall telemetry")
	}
}
