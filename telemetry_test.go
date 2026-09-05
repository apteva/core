package core

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestDurableTerminalTelemetryReplaysOnceAfterRestart(t *testing.T) {
	outbox := t.TempDir()
	t.Setenv("APTEVA_TELEMETRY_OUTBOX_DIR", outbox)
	t.Setenv("TELEMETRY_URL", "")
	t.Setenv("TELEMETRY_LIVE_URL", "")

	first := NewTelemetry()
	if err := first.EmitDurable("tool.result", "worker", ToolResultData{
		ID: "done-call", Name: "done", Success: true,
		ExecutionIDs: []string{"exe-replay"},
	}); err != nil {
		t.Fatalf("persist terminal result: %v", err)
	}
	stored, _ := first.StoredEvents(0)
	if len(stored) != 1 {
		t.Fatalf("initial stored events=%d want 1", len(stored))
	}
	wantID := stored[0].ID
	first.Stop()
	if entries, err := os.ReadDir(outbox); err != nil || len(entries) != 1 {
		t.Fatalf("durable outbox before restart: entries=%d err=%v", len(entries), err)
	}

	received := make(chan []TelemetryEvent, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var events []TelemetryEvent
		if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
			t.Errorf("decode replay batch: %v", err)
		}
		received <- events
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	t.Setenv("TELEMETRY_URL", server.URL)

	second := NewTelemetry()
	select {
	case events := <-received:
		if len(events) != 1 || events[0].ID != wantID || events[0].Type != "tool.result" {
			t.Fatalf("replayed events=%+v want terminal event %s", events, wantID)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("durable terminal result was not replayed")
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		entries, err := os.ReadDir(outbox)
		if err == nil && len(entries) == 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if entries, err := os.ReadDir(outbox); err != nil || len(entries) != 0 {
		t.Fatalf("acknowledged outbox: entries=%d err=%v", len(entries), err)
	}
	second.Stop()

	third := NewTelemetry()
	defer third.Stop()
	select {
	case events := <-received:
		t.Fatalf("acknowledged completion replayed twice: %+v", events)
	case <-time.After(1200 * time.Millisecond):
	}
}

func TestTelemetryLiveForwardBatchesBurstIntoOneRequest(t *testing.T) {
	requests := make(chan []TelemetryEvent, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var events []TelemetryEvent
		if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
			t.Errorf("decode live batch: %v", err)
		}
		requests <- events
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	tel := &Telemetry{
		forwardCh:        make(chan TelemetryEvent, 100),
		quit:             make(chan struct{}),
		telemetryLiveURL: server.URL,
	}
	for i := 0; i < 20; i++ {
		tel.forwardCh <- TelemetryEvent{ID: string(rune('a' + i)), Type: "llm.chunk"}
	}
	go tel.liveForwardLoop()
	defer close(tel.quit)
	select {
	case events := <-requests:
		if len(events) != 20 {
			t.Fatalf("live batch size = %d, want 20", len(events))
		}
	case <-time.After(time.Second):
		t.Fatal("live telemetry batch was not posted")
	}
	select {
	case extra := <-requests:
		t.Fatalf("burst produced an extra request with %d events", len(extra))
	case <-time.After(75 * time.Millisecond):
	}
}

func TestTelemetryLiveForwardRetriesBatchWithoutReordering(t *testing.T) {
	requests := make(chan []TelemetryEvent, 4)
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var events []TelemetryEvent
		if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
			t.Errorf("decode live batch: %v", err)
		}
		requests <- events
		if attempts.Add(1) == 1 {
			http.Error(w, "restart", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	tel := &Telemetry{
		forwardCh:        make(chan TelemetryEvent, 100),
		quit:             make(chan struct{}),
		telemetryLiveURL: server.URL,
	}
	tel.forwardCh <- TelemetryEvent{ID: "first", Type: "llm.start"}
	tel.forwardCh <- TelemetryEvent{ID: "second", Type: "llm.chunk"}
	go tel.liveForwardLoop()
	defer close(tel.quit)

	for i := 0; i < 2; i++ {
		select {
		case events := <-requests:
			if len(events) != 2 || events[0].ID != "first" || events[1].ID != "second" {
				t.Fatalf("attempt %d reordered batch: %+v", i+1, events)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for attempt %d", i+1)
		}
	}
}

func TestTelemetryNotificationsBroadcastToAllSubscribers(t *testing.T) {
	tel := &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	first, cancelFirst := tel.Subscribe()
	defer cancelFirst()
	second, cancelSecond := tel.Subscribe()
	defer cancelSecond()
	tel.EmitLive("llm.chunk", "main", map[string]string{"text": "x"})
	for i, ch := range []<-chan struct{}{first, second} {
		select {
		case <-ch:
		case <-time.After(100 * time.Millisecond):
			t.Fatalf("subscriber %d was not notified", i)
		}
	}
}

func TestTelemetry_Emit(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	tel.Emit("llm.done", "main", LLMDoneData{
		Provider: "test-provider",
		Model:    "test-model",
		TokensIn: 100, TokensCached: 40, CacheWriteTokens: 25, TokensOut: 50,
		DurationMs: 1500,
		ProviderTiming: ProviderTiming{
			RequestAttempts: 1,
			TerminalPhase:   "completed",
			CompletionMs:    1400,
		},
		Iteration:       1,
		NativeToolCount: 12,
		ActiveMCPCount:  3,
		ToolMode:        "discovery",
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
	if data.TokensCached != 40 || data.CacheWriteTokens != 25 {
		t.Errorf("cache usage missing from llm.done: %+v", data)
	}
	if data.NativeToolCount != 12 || data.ActiveMCPCount != 3 || data.ToolMode != "discovery" {
		t.Errorf("cache diagnostics missing from llm.done: %+v", data)
	}
	if data.Provider != "test-provider" || data.RequestAttempts != 1 || data.TerminalPhase != "completed" || data.CompletionMs != 1400 {
		t.Errorf("provider timing diagnostics missing from llm.done: %+v", data)
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
		Iteration: 1, Rate: "normal",
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
		Name: "web", Args: map[string]string{"url": "https://example.com"},
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

func TestModelContextWindow(t *testing.T) {
	cases := []struct {
		model string
		want  int
	}{
		// Fireworks Kimi router — what the prod deployment actually
		// runs against, so this row is the canary.
		{"accounts/fireworks/routers/kimi-k2p5-turbo", 256_000},
		{"accounts/fireworks/models/kimi-k2-instruct", 128_000},
		// OpenCode Go uses dotted ids; bare "kimi-k2" must NOT
		// shadow the dotted entries (was the bug).
		{"kimi-k2.6", 256_000},
		{"glm-5.2", 128_000},
		{"kimi-k2.5", 256_000},
		{"minimax-m2.7", 196_608},
		{"minimax-m2.5", 196_608},
		// Anthropic — both default and 1M-context variants.
		{"claude-opus-4-7", 200_000},
		{"claude-opus-4-7[1m]", 1_000_000},
		{"claude-sonnet-4-6", 200_000},
		{"claude-sonnet-4-5[1m]", 1_000_000},
		{"claude-haiku-4-5-20251001", 200_000},
		{"claude-3-5-sonnet-20241022", 200_000},
		// OpenAI.
		{"gpt-4o", 128_000},
		{"gpt-4o-mini", 128_000},
		{"gpt-4.1", 1_000_000},
		{"o1-preview", 200_000}, // matches "o1" prefix
		{"o3-mini", 200_000},
		// Gemini.
		{"gemini-1.5-pro-002", 2_000_000},
		{"gemini-2.5-flash", 1_000_000},
		// Local.
		{"llama3.1:8b", 128_000},
		// Unknown — must return 0 (UI uses 0 as the "no max known" sentinel).
		{"some-future-model-9000", 0},
		{"", 0},
	}
	for _, c := range cases {
		got := ModelContextWindow(c.model)
		if got != c.want {
			t.Errorf("ModelContextWindow(%q) = %d, want %d", c.model, got, c.want)
		}
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

func TestToolArgumentsTelemetryDistinguishesProviderParsedAndTypedPayloads(t *testing.T) {
	tel := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}
	raw := `{"action":"open","persist":false,"timeout":60}`
	parsed := map[string]string{"action": "open", "persist": "false", "timeout": "60"}
	typed := mcpArgumentsFromStrings(parsed, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action":  map[string]any{"type": "string"},
			"persist": map[string]any{"type": "boolean"},
			"timeout": map[string]any{"type": "integer"},
		},
	})

	tel.Emit("tool.arguments", "worker", newToolArgumentsData("call-1", "computer_browser_session", "provider_raw", raw))
	tel.Emit("tool.arguments", "worker", newToolArgumentsData("call-1", "computer_browser_session", "core_parsed", parsed))
	tel.Emit("tool.arguments", "worker", newToolArgumentsData("call-1", "computer_browser_session", "mcp_typed", typed))

	events, _ := tel.Events(0)
	if len(events) != 3 {
		t.Fatalf("tool argument events = %d, want 3", len(events))
	}
	wantStages := []string{"provider_raw", "core_parsed", "mcp_typed"}
	for i, event := range events {
		if event.Type != "tool.arguments" {
			t.Fatalf("event %d type = %q", i, event.Type)
		}
		var data ToolArgumentsData
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatalf("decode event %d: %v", i, err)
		}
		if data.Stage != wantStages[i] || data.ID != "call-1" || data.Name != "computer_browser_session" {
			t.Fatalf("event %d = %+v", i, data)
		}
		if data.JSON == "" || data.SHA256 == "" || data.OriginalBytes == 0 || data.Truncated {
			t.Fatalf("event %d missing bounded payload metadata: %+v", i, data)
		}
	}
	var final ToolArgumentsData
	if err := json.Unmarshal(events[2].Data, &final); err != nil {
		t.Fatal(err)
	}
	if final.Types["action"] != "string" || final.Types["persist"] != "boolean" || final.Types["timeout"] != "number" {
		t.Fatalf("typed argument types = %#v", final.Types)
	}
}

func TestToolArgumentsTelemetryBoundsLargeRawProviderJSON(t *testing.T) {
	data := newToolArgumentsData("call-large", "upload", "provider_raw", strings.Repeat("x", toolArgumentsTelemetryPreviewBytes+500))
	if !data.Truncated || data.OriginalBytes != toolArgumentsTelemetryPreviewBytes+500 || data.PreviewBytes != toolArgumentsTelemetryPreviewBytes {
		t.Fatalf("bounded raw argument telemetry = %+v", data)
	}
	if len(data.JSON) != toolArgumentsTelemetryPreviewBytes || data.SHA256 == "" {
		t.Fatalf("bounded preview/hash missing: %+v", data)
	}
}
