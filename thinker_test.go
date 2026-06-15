package core

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestThinkRate_String(t *testing.T) {
	tests := []struct {
		rate ThinkRate
		want string
	}{
		{RateReactive, "reactive"},
		{RateFast, "fast"},
		{RateNormal, "normal"},
		{RateSlow, "slow"},
		{RateSleep, "sleep"},
	}
	for _, tt := range tests {
		if got := tt.rate.String(); got != tt.want {
			t.Errorf("ThinkRate(%d).String() = %q, want %q", tt.rate, got, tt.want)
		}
	}
}

func TestThinkRate_Delay(t *testing.T) {
	tests := []struct {
		rate ThinkRate
		want time.Duration
	}{
		{RateReactive, 500 * time.Millisecond},
		{RateFast, 2 * time.Second},
		{RateNormal, 10 * time.Second},
		{RateSlow, 30 * time.Second},
		{RateSleep, 120 * time.Second},
	}
	for _, tt := range tests {
		if got := tt.rate.Delay(); got != tt.want {
			t.Errorf("ThinkRate(%d).Delay() = %v, want %v", tt.rate, got, tt.want)
		}
	}
}

func TestRateNames(t *testing.T) {
	for name, rate := range rateNames {
		if rate.String() != name {
			t.Errorf("rateNames[%q] = %d, String() = %q", name, rate, rate.String())
		}
	}
}

func TestParseSleepDuration(t *testing.T) {
	tests := []struct {
		input string
		want  time.Duration
		ok    bool
	}{
		// Named aliases
		{"fast", 2 * time.Second, true},
		{"normal", 10 * time.Second, true},
		{"slow", 30 * time.Second, true},
		{"sleep", 2 * time.Minute, true},
		{"reactive", 500 * time.Millisecond, true},
		// Go duration strings
		{"5s", 5 * time.Second, true},
		{"30s", 30 * time.Second, true},
		{"5m", 5 * time.Minute, true},
		{"1h", 1 * time.Hour, true},
		{"2h30m", 2*time.Hour + 30*time.Minute, true},
		{"500ms", 500 * time.Millisecond, true},
		// Clamping
		{"100ms", 500 * time.Millisecond, true}, // clamped to min
		{"48h", 24 * time.Hour, true},           // clamped to max
		// Invalid
		{"garbage", 0, false},
		{"", 0, false},
	}
	for _, tt := range tests {
		got, ok := parseSleepDuration(tt.input)
		if ok != tt.ok {
			t.Errorf("parseSleepDuration(%q): ok=%v, want %v", tt.input, ok, tt.ok)
			continue
		}
		if ok && got != tt.want {
			t.Errorf("parseSleepDuration(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}

func TestFormatSleep(t *testing.T) {
	tests := []struct {
		dur  time.Duration
		want string
	}{
		{500 * time.Millisecond, "0.5s"},
		{2 * time.Second, "2.0s"},
		{30 * time.Second, "30.0s"},
		{5 * time.Minute, "5.0m"},
		{1 * time.Hour, "1.0h"},
		{90 * time.Minute, "1.5h"},
	}
	for _, tt := range tests {
		if got := formatSleep(tt.dur); got != tt.want {
			t.Errorf("formatSleep(%v) = %q, want %q", tt.dur, got, tt.want)
		}
	}
}

func TestModelTier_String(t *testing.T) {
	if ModelLarge.String() != "large" {
		t.Errorf("expected 'large', got %q", ModelLarge.String())
	}
	if ModelSmall.String() != "small" {
		t.Errorf("expected 'small', got %q", ModelSmall.String())
	}
}

func TestModelTier_ProviderID(t *testing.T) {
	provider := NewFireworksProvider("test")
	models := provider.Models()
	if models[ModelLarge] == "" {
		t.Error("large model ID should not be empty")
	}
	if models[ModelSmall] == "" {
		t.Error("small model ID should not be empty")
	}
}

func TestModelNames(t *testing.T) {
	for name, tier := range modelNames {
		if tier.String() != name {
			t.Errorf("modelNames[%q] = %d, String() = %q", name, tier, tier.String())
		}
	}
}

func TestDrainEvents_Empty(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 10),
		threadID: "test",
	}
	items := thinker.drainEventTexts()
	if len(items) != 0 {
		t.Errorf("expected empty, got %d items", len(items))
	}
}

func TestDrainEvents_WithMessages(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 10),
		threadID: "test",
	}
	bus.Publish(Event{Type: EventInbox, To: "test", Text: "msg1"})
	bus.Publish(Event{Type: EventInbox, To: "test", Text: "msg2"})
	bus.Publish(Event{Type: EventInbox, To: "test", Text: "msg3"})

	// Small sleep to let publishes land
	time.Sleep(10 * time.Millisecond)

	items := thinker.drainEventTexts()
	if len(items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(items))
	}
	if items[0] != "msg1" || items[1] != "msg2" || items[2] != "msg3" {
		t.Errorf("unexpected items: %v", items)
	}

	// Should be empty now
	items2 := thinker.drainEventTexts()
	if len(items2) != 0 {
		t.Errorf("expected empty after drain, got %d", len(items2))
	}
}

func TestInject(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 10),
		threadID: "test",
	}
	thinker.Inject("test event")
	time.Sleep(10 * time.Millisecond)
	items := thinker.drainEventTexts()
	if len(items) != 1 || items[0] != "test event" {
		t.Errorf("unexpected items: %v", items)
	}
}

func TestInjectConsoleMessage(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 10),
		threadID: "test",
	}
	thinker.InjectConsole("Hello")
	time.Sleep(10 * time.Millisecond)
	items := thinker.drainEventTexts()
	if len(items) != 1 || items[0] != "[console] Hello" {
		t.Errorf("expected '[console] Hello', got %v", items)
	}
}

func TestMainEvolveKicksNextTurn(t *testing.T) {
	events := []APIEvent{}
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "old directive"},
		messages:  []Message{{Role: "system", Content: "old prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"directive": "new directive"},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	if thinker.config.GetDirective() != "new directive" {
		t.Fatalf("directive = %q, want new directive", thinker.config.GetDirective())
	}
	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after evolve")
	}
	if len(results) != 1 || results[0].CallID != "call-1" || results[0].Content != "directive updated" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestMainKillKicksNextTurn(t *testing.T) {
	events := []APIEvent{}
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "test"},
		messages:  []Message{{Role: "system", Content: "test"}},
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		quit:      make(chan struct{}),
	}
	thinker.threads = NewThreadManager(thinker)

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "kill",
		Args:     map[string]string{"id": "worker"},
		Raw:      "kill",
		NativeID: "call-1",
	}}, nil)

	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after kill")
	}
	if len(results) != 1 || results[0].CallID != "call-1" || !strings.Contains(results[0].Content, "thread worker killed") {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestMainUpdateKicksNextTurn(t *testing.T) {
	events := []APIEvent{}
	bus := NewEventBus()
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "test"},
		messages:  []Message{{Role: "system", Content: "test"}},
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		quit:      make(chan struct{}),
	}
	thinker.threads = NewThreadManager(thinker)
	if err := thinker.threads.SpawnWithOpts("worker", "old directive", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "update",
		Args:     map[string]string{"id": "worker", "directive": "new directive"},
		Raw:      "update",
		NativeID: "call-1",
	}}, nil)

	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after update")
	}
	if len(results) != 1 || results[0].CallID != "call-1" || results[0].Content != "thread worker updated" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestPublish_NonBlocking(t *testing.T) {
	bus := NewEventBus()
	// Small buffer — should not block even when full
	bus.Subscribe("slow", 1)
	// Publish multiple times — should never block
	bus.Publish(Event{Type: EventInbox, To: "slow", Text: "1"})
	bus.Publish(Event{Type: EventInbox, To: "slow", Text: "2"})
	bus.Publish(Event{Type: EventInbox, To: "slow", Text: "3"})
	// If we get here without hanging, it works
}

func TestThinkerStop(t *testing.T) {
	thinker := &Thinker{
		quit: make(chan struct{}),
	}
	thinker.Stop()
	select {
	case <-thinker.quit:
		// ok
	default:
		t.Error("quit channel should be closed")
	}
}

// TestWaitForPendingTools_DrainsAllBeforeDeadline verifies that the
// iter-boundary barrier drains every pending tool result that arrives
// before the deadline. Baseline happy-path: 4 parallel tools all
// finish within the window, waitForPendingTools returns with every
// result appended and pendingTools empty.
func TestWaitForPendingTools_DrainsAllBeforeDeadline(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 100),
		threadID: "test",
		quit:     make(chan struct{}),
	}

	// Four in-flight tool dispatches.
	ids := []string{"call-A", "call-B", "call-C", "call-D"}
	for _, id := range ids {
		thinker.pendingTools.Store(id, "mock_tool")
	}

	// Goroutines simulate each tool finishing at staggered times within
	// the 3s window, publishing a ToolResult and clearing pendingTools.
	for i, id := range ids {
		go func(id string, delay time.Duration) {
			time.Sleep(delay)
			bus.Publish(Event{
				Type: EventInbox, To: "test",
				ToolResult: &ToolResult{CallID: id, Content: "ok-" + id},
			})
			thinker.pendingTools.Delete(id)
		}(id, time.Duration(50+i*50)*time.Millisecond)
	}

	var toolResults []ToolResult
	var consumed []string
	var mediaParts []ContentPart
	thinker.waitForPendingTools(&toolResults, &consumed, &mediaParts, 3*time.Second)

	if len(toolResults) != 4 {
		t.Fatalf("expected 4 tool results after drain, got %d", len(toolResults))
	}
	if n := thinker.pendingToolCount(); n != 0 {
		t.Errorf("expected 0 pending after drain, got %d", n)
	}
	seen := map[string]bool{}
	for _, tr := range toolResults {
		seen[tr.CallID] = true
	}
	for _, id := range ids {
		if !seen[id] {
			t.Errorf("missing tool result for %s", id)
		}
	}
}

// TestWaitForPendingTools_DeadlineThenPlaceholder verifies the critical
// race fix: one tool is genuinely slow (2.5s > 800ms deadline), so the
// barrier returns early and injectPlaceholdersForPending synthesises a
// "⏳ in progress" result for the laggard. When the slow goroutine
// eventually publishes its real result, the late-result routing in
// executeTool sends it as a text [late-result] event instead of a
// second ToolResult for the same id — so the model never sees two
// paired tool_results and never retries the call.
func TestWaitForPendingTools_DeadlineThenPlaceholder(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 100),
		threadID: "test",
		quit:     make(chan struct{}),
	}

	ids := []string{"call-A", "call-B", "call-C", "call-SLOW"}
	for _, id := range ids {
		thinker.pendingTools.Store(id, "mock_tool")
	}

	// A, B, C finish fast (50-150ms). SLOW finishes at 2.5s — long
	// after the 800ms barrier deadline. We drive tools.go-style
	// publish logic manually so we can exercise the late-result path
	// end-to-end without a real registry.
	publishResult := func(id string, delay time.Duration) {
		go func() {
			time.Sleep(delay)
			if _, has := thinker.placeholdersSent.LoadAndDelete(id); has {
				bus.Publish(Event{
					Type: EventInbox, To: "test",
					Text: "[late-result] Tool mock_tool (call id=" + id + ") completed: ok-" + id,
				})
			} else {
				bus.Publish(Event{
					Type: EventInbox, To: "test",
					ToolResult: &ToolResult{CallID: id, Content: "ok-" + id},
				})
			}
			thinker.pendingTools.Delete(id)
		}()
	}
	publishResult("call-A", 50*time.Millisecond)
	publishResult("call-B", 100*time.Millisecond)
	publishResult("call-C", 150*time.Millisecond)
	publishResult("call-SLOW", 2500*time.Millisecond)

	var toolResults []ToolResult
	var consumed []string
	var mediaParts []ContentPart
	// Short deadline so SLOW doesn't land in time — forces placeholder.
	thinker.waitForPendingTools(&toolResults, &consumed, &mediaParts, 800*time.Millisecond)

	if len(toolResults) != 3 {
		t.Fatalf("expected 3 real tool results (A/B/C) before deadline, got %d", len(toolResults))
	}
	if thinker.pendingToolCount() != 1 {
		t.Fatalf("expected 1 tool still pending after deadline, got %d", thinker.pendingToolCount())
	}

	// Inject placeholder for the laggard.
	thinker.injectPlaceholdersForPending(&toolResults)

	if len(toolResults) != 4 {
		t.Fatalf("expected 4 tool results after placeholder injection, got %d", len(toolResults))
	}
	// The placeholder must carry the SLOW call id and the in-progress marker.
	var found bool
	for _, tr := range toolResults {
		if tr.CallID == "call-SLOW" && strings.Contains(tr.Content, "In progress") {
			found = true
		}
	}
	if !found {
		t.Error("placeholder for call-SLOW not found in toolResults")
	}
	if _, ok := thinker.placeholdersSent.Load("call-SLOW"); !ok {
		t.Error("placeholdersSent missing call-SLOW entry")
	}

	// Now wait for SLOW to actually finish. Its late result should be
	// routed through the text-event path (prefix [late-result]) and
	// the placeholdersSent entry cleared.
	deadline := time.After(5 * time.Second)
	var lateResultSeen bool
	for !lateResultSeen {
		select {
		case ev := <-thinker.sub.C:
			if ev.Type == EventInbox && strings.HasPrefix(ev.Text, "[late-result]") && strings.Contains(ev.Text, "call-SLOW") {
				lateResultSeen = true
				if ev.ToolResult != nil {
					t.Error("late-result event must NOT carry a ToolResult — would create duplicate pair")
				}
			}
		case <-deadline:
			t.Fatal("timed out waiting for late-result event from slow tool")
		}
	}

	if _, ok := thinker.placeholdersSent.Load("call-SLOW"); ok {
		t.Error("placeholdersSent still contains call-SLOW after late-result delivery")
	}
}

// TestPlaceholder_NoDuplicateOnSecondIteration asserts that if the
// barrier runs again on a subsequent iteration while the same slow
// tool is STILL pending, it does NOT re-inject another placeholder —
// the original placeholder is already baked into message history.
func TestPlaceholder_NoDuplicateOnSecondIteration(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 100),
		threadID: "test",
		quit:     make(chan struct{}),
	}

	thinker.pendingTools.Store("call-HANG", "mock_tool")

	var results []ToolResult
	thinker.injectPlaceholdersForPending(&results)
	if len(results) != 1 {
		t.Fatalf("first inject: expected 1 placeholder, got %d", len(results))
	}

	// Second inject on the same iteration — same id already marked,
	// placeholder must NOT be re-added.
	thinker.injectPlaceholdersForPending(&results)
	if len(results) != 1 {
		t.Errorf("second inject: expected 1 placeholder total, got %d", len(results))
	}
}
