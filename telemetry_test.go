package main

import (
	"encoding/json"
	"testing"
	"time"
)

func TestTelemetry_Emit(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	tel.Emit("llm.done", "main", LLMDoneData{
		Model:    "test-model",
		TokensIn: 100, TokensOut: 50,
		DurationMs: 1500, CostUSD: 0.001,
		Iteration: 1,
	})

	events, cursor := tel.Events(0)
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if cursor != 1 {
		t.Fatalf("expected cursor 1, got %d", cursor)
	}

	ev := events[0]
	if ev.Type != "llm.done" {
		t.Errorf("expected llm.done, got %s", ev.Type)
	}
	if ev.ThreadID != "main" {
		t.Errorf("expected main, got %s", ev.ThreadID)
	}
	if ev.ID == "" {
		t.Error("expected non-empty ID")
	}

	// Verify data
	var data LLMDoneData
	if err := json.Unmarshal(ev.Data, &data); err != nil {
		t.Fatalf("unmarshal data: %v", err)
	}
	if data.TokensIn != 100 {
		t.Errorf("expected 100, got %d", data.TokensIn)
	}
	if data.CostUSD != 0.001 {
		t.Errorf("expected 0.001, got %f", data.CostUSD)
	}
}

func TestTelemetry_EventsSince(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	tel.Emit("llm.done", "main", map[string]string{"a": "1"})
	tel.Emit("thread.spawn", "t1", map[string]string{"b": "2"})
	tel.Emit("tool.call", "main", map[string]string{"c": "3"})

	// Get all
	all, _ := tel.Events(0)
	if len(all) != 3 {
		t.Fatalf("expected 3, got %d", len(all))
	}

	// Get since cursor 1
	since1, cursor := tel.Events(1)
	if len(since1) != 2 {
		t.Fatalf("expected 2, got %d", len(since1))
	}
	if cursor != 3 {
		t.Fatalf("expected cursor 3, got %d", cursor)
	}
	if since1[0].Type != "thread.spawn" {
		t.Errorf("expected thread.spawn, got %s", since1[0].Type)
	}

	// Get since end — should be empty
	empty, _ := tel.Events(3)
	if len(empty) != 0 {
		t.Fatalf("expected 0, got %d", len(empty))
	}
}

func TestTelemetry_BufferLimit(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	for i := 0; i < 2500; i++ {
		tel.Emit("llm.done", "main", map[string]int{"i": i})
	}

	events, _ := tel.Events(0)
	if len(events) > 1500 {
		t.Errorf("expected buffer trimmed, got %d events", len(events))
	}
}

func TestTelemetry_AllEventTypes(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	// LLM events
	tel.Emit("llm.done", "main", LLMDoneData{
		Model: "m", TokensIn: 10, TokensOut: 5, DurationMs: 100,
		CostUSD: 0.0001, Iteration: 1, Rate: "normal",
	})
	tel.Emit("llm.error", "main", LLMErrorData{
		Model: "m", Error: "timeout", Iteration: 2,
	})

	// Thread events
	tel.Emit("thread.spawn", "research", ThreadSpawnData{
		ParentID: "main", Directive: "investigate X", Tools: []string{"web", "send"},
	})
	tel.Emit("thread.message", "research", ThreadMessageData{
		From: "research", To: "main", Message: "found something",
	})
	tel.Emit("thread.done", "research", ThreadDoneData{
		ParentID: "main", Result: "done investigating",
	})

	// Tool events
	tel.Emit("tool.call", "main", ToolCallData{
		Name: "web", Args: "url=https://example.com",
	})
	tel.Emit("tool.result", "main", ToolResultData{
		Name: "web", DurationMs: 500, Success: true, Result: "page content",
	})

	events, _ := tel.Events(0)
	if len(events) != 7 {
		t.Fatalf("expected 7 events, got %d", len(events))
	}

	types := make([]string, len(events))
	for i, e := range events {
		types[i] = e.Type
	}

	expected := []string{"llm.done", "llm.error", "thread.spawn", "thread.message", "thread.done", "tool.call", "tool.result"}
	for i, exp := range expected {
		if types[i] != exp {
			t.Errorf("event %d: expected %s, got %s", i, exp, types[i])
		}
	}
}

func TestTelemetry_ThreadMessageData(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	tel.Emit("thread.message", "worker-1", ThreadMessageData{
		From: "worker-1", To: "main", Message: "reporting results",
	})

	events, _ := tel.Events(0)
	ev := events[0]

	var data ThreadMessageData
	if err := json.Unmarshal(ev.Data, &data); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if data.From != "worker-1" || data.To != "main" {
		t.Errorf("expected worker-1→main, got %s→%s", data.From, data.To)
	}
	if data.Message != "reporting results" {
		t.Errorf("unexpected message: %s", data.Message)
	}
}

func TestCalculateCost(t *testing.T) {
	usage := TokenUsage{
		PromptTokens:     1000,
		CachedTokens:     800,
		CompletionTokens: 200,
	}
	cost := calculateCost(usage)

	// 200 uncached * 0.60/1M + 800 cached * 0.10/1M + 200 output * 3.00/1M
	expected := (200*0.60 + 800*0.10 + 200*3.00) / 1_000_000
	if cost < expected*0.99 || cost > expected*1.01 {
		t.Errorf("expected ~%f, got %f", expected, cost)
	}
}

func TestTelemetry_EmitLive(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	// EmitLive should appear in Events() but NOT in StoredEvents()
	tel.EmitLive("llm.chunk", "main", LLMChunkData{Text: "hello", Iteration: 1})
	tel.EmitLive("llm.chunk", "main", LLMChunkData{Text: " world", Iteration: 1})
	tel.Emit("llm.done", "main", map[string]string{"msg": "done"})

	// Events (SSE) should see all 3
	all, _ := tel.Events(0)
	if len(all) != 3 {
		t.Fatalf("Events: expected 3, got %d", len(all))
	}

	// StoredEvents (backplane forward) should only see 1 (llm.done)
	stored, _ := tel.StoredEvents(0)
	if len(stored) != 1 {
		t.Fatalf("StoredEvents: expected 1, got %d", len(stored))
	}
	if stored[0].Type != "llm.done" {
		t.Errorf("expected llm.done, got %s", stored[0].Type)
	}
}

func TestTelemetry_NotifyChannel(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	tel.Emit("llm.done", "main", map[string]string{})

	// Notify should have a message
	select {
	case <-tel.notify:
		// ok
	case <-time.After(100 * time.Millisecond):
		t.Error("expected notify signal")
	}
}
