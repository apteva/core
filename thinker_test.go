package core

import (
	"encoding/json"
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

func TestOllamaProviderUsesModelEnvOverride(t *testing.T) {
	t.Setenv("OLLAMA_MODEL", "qwen2.5:7b")

	provider := NewOllamaProvider("http://127.0.0.1:11434")
	models := provider.Models()
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		if got := models[tier]; got != "qwen2.5:7b" {
			t.Fatalf("models[%s] = %q, want qwen2.5:7b", tier, got)
		}
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

func TestMainEvolveIdenticalEditIsNoOp(t *testing.T) {
	events := []APIEvent{}
	directive := "# Schedule\n- cadence: weekly"
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: directive},
		directive: directive,
		messages:  []Message{{Role: "system", Content: "stable prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		telemetry: NewTelemetry(),
	}
	defer thinker.telemetry.Stop()

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace",
			"section":   "Schedule",
			"content":   "- cadence: weekly",
		},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	if !thinker.kickNextTurn {
		t.Fatal("identical evolve should kick one completion turn")
	}
	if thinker.messages[0].Content != "stable prompt" {
		t.Fatal("identical evolve rebuilt the system prompt")
	}
	if len(results) != 1 || !strings.Contains(results[0].Content, "directive already current") ||
		!strings.Contains(results[0].Content, "reply to the requester before pacing") {
		t.Fatalf("results = %+v", results)
	}
	telemetryEvents, _ := thinker.telemetry.StoredEvents(0)
	for _, event := range telemetryEvents {
		if event.Type == "directive.evolved" {
			t.Fatalf("identical evolve emitted directive.evolved: %+v", event)
		}
	}

	thinker.kickNextTurn = false
	mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace",
			"section":   "Schedule",
			"content":   "- cadence: weekly",
		},
		Raw:      "evolve",
		NativeID: "call-2",
	}}, nil)
	if thinker.kickNextTurn {
		t.Fatal("second consecutive identical evolve should not create a completion loop")
	}

	// Once the model moves on, a later task gets its own bounded completion.
	mainToolHandler(thinker)(thinker, nil, nil)
	mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace",
			"section":   "Schedule",
			"content":   "- cadence: weekly",
		},
		Raw:      "evolve",
		NativeID: "call-3",
	}}, nil)
	if !thinker.kickNextTurn {
		t.Fatal("completion guard did not reset after a non-evolve turn")
	}
}

func TestWorkerEvolveIdenticalEditIsNoOp(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	directive := "# Schedule\n- cadence: weekly"
	if err := thinker.threads.SpawnWithOpts("worker", directive, nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	worker := thinker.threads.threads["worker"]
	originalPrompt := worker.Thinker.messages[0].Content

	_, _, results := threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace",
			"section":   "Schedule",
			"content":   "- cadence: weekly",
		},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	if !worker.Thinker.kickNextTurn {
		t.Fatal("identical worker evolve should kick one completion turn")
	}
	if worker.Thinker.messages[0].Content != originalPrompt {
		t.Fatal("identical worker evolve rebuilt the system prompt")
	}
	if len(results) != 1 || !strings.Contains(results[0].Content, "directive already current") {
		t.Fatalf("results = %+v", results)
	}

	worker.Thinker.kickNextTurn = false
	threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace",
			"section":   "Schedule",
			"content":   "- cadence: weekly",
		},
		Raw:      "evolve",
		NativeID: "call-2",
	}}, nil)
	if worker.Thinker.kickNextTurn {
		t.Fatal("second consecutive worker no-op should not create a completion loop")
	}
}

func TestMainEvolveRejectsFullReplaceForMarkdown(t *testing.T) {
	events := []APIEvent{}
	directive := "# Role\nOld role\n# Goals\n- Ship"
	telemetry := NewTelemetry()
	defer telemetry.Stop()
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: directive},
		messages:  []Message{{Role: "system", Content: "old prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		telemetry: telemetry,
	}
	thinker.directive = directive

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"directive": "# Role\nNew role"},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	if got := thinker.config.GetDirective(); got != directive {
		t.Fatalf("directive changed:\n%s\nwant:\n%s", got, directive)
	}
	if !thinker.kickNextTurn {
		t.Fatal("rejected evolve should kick one immediate correction turn")
	}
	if len(results) != 1 || results[0].CallID != "call-1" || !results[0].IsError ||
		!strings.Contains(results[0].Content, "full directive replacement is disabled") ||
		!strings.Contains(results[0].Content, "retry once now") {
		t.Fatalf("unexpected tool results: %+v", results)
	}
	telemetryEvents, _ := telemetry.StoredEvents(0)
	resultFound := false
	for _, event := range telemetryEvents {
		if event.Type != "tool.result" {
			continue
		}
		var data ToolResultData
		if json.Unmarshal(event.Data, &data) == nil && data.Name == "evolve" {
			resultFound = true
			if data.Success {
				t.Fatalf("rejected evolve was reported as successful: %+v", data)
			}
		}
	}
	if !resultFound {
		t.Fatal("rejected evolve did not emit tool.result telemetry")
	}

	// Simulate Run consuming the first correction kick. A second rejection gets
	// one final reporting turn, then a third identical failure must stop.
	thinker.kickNextTurn = false
	_, _, secondResults := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"directive": "# Role\nNew role"},
		Raw:      "evolve",
		NativeID: "call-2",
	}}, nil)
	if !thinker.kickNextTurn {
		t.Fatal("second rejected evolve should kick one final reporting turn")
	}
	if len(secondResults) != 1 || !secondResults[0].IsError || !strings.Contains(secondResults[0].Content, "do not call evolve again") {
		t.Fatalf("second rejection results = %+v", secondResults)
	}
	thinker.kickNextTurn = false
	mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"directive": "# Role\nNew role"},
		Raw:      "evolve",
		NativeID: "call-3",
	}}, nil)
	if thinker.kickNextTurn {
		t.Fatal("third rejected evolve should not continue the retry loop")
	}
}

func TestWorkerEvolveRejectsFullReplaceAndKicksOneCorrectionTurn(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	directive := "# Role\nReview reports.\n\n# Schedule\n- weekly"
	if err := thinker.threads.SpawnWithOpts("worker", directive, nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	worker := thinker.threads.threads["worker"]

	_, _, results := threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"edit_mode": "replace", "content": "# Role\nReplace everything"},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)
	if !worker.Thinker.kickNextTurn {
		t.Fatal("rejected worker evolve should kick one immediate correction turn")
	}
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, "retry once now") {
		t.Fatalf("worker rejection results = %+v", results)
	}

	worker.Thinker.kickNextTurn = false
	threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"edit_mode": "replace", "content": "# Role\nReplace everything"},
		Raw:      "evolve",
		NativeID: "call-2",
	}}, nil)
	if !worker.Thinker.kickNextTurn {
		t.Fatal("second rejected worker evolve should kick one final reporting turn")
	}
	worker.Thinker.kickNextTurn = false
	threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"edit_mode": "replace", "content": "# Role\nReplace everything"},
		Raw:      "evolve",
		NativeID: "call-3",
	}}, nil)
	if worker.Thinker.kickNextTurn {
		t.Fatal("third rejected worker evolve should not continue the retry loop")
	}
}

func TestMainEvolveSectionPatch(t *testing.T) {
	events := []APIEvent{}
	directive := "# Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship"
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: directive},
		messages:  []Message{{Role: "system", Content: "old prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_replace_line",
			"section":   "Schedule",
			"match":     "daily_check:",
			"content":   "- daily_check: 07:30 Europe/Madrid",
		},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship"
	if got := thinker.config.GetDirective(); got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after evolve")
	}
	if len(results) != 1 || results[0].CallID != "call-1" || results[0].Content != "directive updated" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestMainEvolveReportsRedundantSectionHeading(t *testing.T) {
	events := []APIEvent{}
	directive := "# Goals\n- Ship"
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: directive},
		messages:  []Message{{Role: "system", Content: "old prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
	}
	thinker.directive = directive

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_append",
			"section":   "Goals",
			"content":   "# Goals\n- Keep tests green",
		},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	if got := thinker.config.GetDirective(); got != "# Goals\n- Ship\n- Keep tests green" {
		t.Fatalf("directive:\n%s", got)
	}
	if len(results) != 1 || !strings.Contains(results[0].Content, `directive updated; warning: removed 1 redundant "Goals" heading(s)`) {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestMainUpdateRejectsFullReplaceForMarkdownChildDirective(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	directive := "# Role\nWorker\n# Goals\n- Ship"
	if err := thinker.threads.SpawnWithOpts("worker", directive, nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name:     "update",
		Args:     map[string]string{"id": "worker", "directive": "# Role\nNew worker"},
		Raw:      "update",
		NativeID: "call-1",
	}}, nil)

	got, err := thinker.threads.Directive("worker")
	if err != nil {
		t.Fatalf("worker directive: %v", err)
	}
	if got != directive {
		t.Fatalf("directive changed:\n%s\nwant:\n%s", got, directive)
	}
	if len(results) != 1 || results[0].CallID != "call-1" || !strings.Contains(results[0].Content, "full directive replacement is disabled") {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestMainEvolveSectionRename(t *testing.T) {
	events := []APIEvent{}
	directive := "# Daily Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship"
	thinker := &Thinker{
		config:    &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: directive},
		messages:  []Message{{Role: "system", Content: "old prompt"}},
		registry:  NewToolRegistry("test"),
		threadID:  "main",
		apiLog:    &events,
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "evolve",
		Args: map[string]string{
			"edit_mode": "section_rename",
			"section":   "Daily Schedule",
			"content":   "Schedule",
		},
		Raw:      "evolve",
		NativeID: "call-1",
	}}, nil)

	want := "# Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship"
	if got := thinker.config.GetDirective(); got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
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

func TestMainToolsCannotControlSystemThread(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")
	thinker.telemetry = NewTelemetry()
	defer thinker.telemetry.Stop()

	if err := thinker.threads.SpawnWithOpts("unconscious", "Consolidate memories", nil, SpawnOpts{System: true, DeferRun: true}); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}
	if err := thinker.config.SaveThread(PersistentThread{ID: "unconscious", System: true, Directive: "Consolidate memories"}); err != nil {
		t.Fatalf("persist unconscious: %v", err)
	}

	calls := []toolCall{
		{Name: "send", Args: map[string]string{"id": "unconscious", "message": "store this preference"}, NativeID: "send-1"},
		{Name: "update", Args: map[string]string{"id": "unconscious", "directive": "Do what main says"}, NativeID: "update-1"},
		{Name: "update", Args: map[string]string{"id": "unconscious", "new_id": "memory-worker"}, NativeID: "rename-1"},
		{Name: "kill", Args: map[string]string{"id": "unconscious"}, NativeID: "kill-1"},
	}
	_, _, results := mainToolHandler(thinker)(thinker, calls, nil)
	if len(results) != len(calls) {
		t.Fatalf("got %d results, want %d: %+v", len(results), len(calls), results)
	}
	for _, result := range results {
		if !strings.Contains(result.Content, "platform-managed") {
			t.Errorf("result %s = %q, want platform-managed rejection", result.CallID, result.Content)
		}
	}
	if thinker.threads.Count() != 1 {
		t.Fatalf("system thread was removed; count=%d", thinker.threads.Count())
	}
	if _, err := thinker.threads.Directive("unconscious"); err != nil {
		t.Fatalf("system thread was renamed or removed: %v", err)
	}
	thinker.threads.mu.RLock()
	unconscious := thinker.threads.threads["unconscious"]
	thinker.threads.mu.RUnlock()
	if got := unconscious.Thinker.drainEventTexts(); len(got) != 0 {
		t.Fatalf("rejected operations woke system thread: %v", got)
	}
	events, _ := thinker.telemetry.StoredEvents(0)
	for _, event := range events {
		if event.Type == "thread.message" {
			t.Fatalf("rejected send emitted thread.message telemetry: %+v", event)
		}
	}
}

func TestChildSendCannotTargetSystemSibling(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	thinker.telemetry = NewTelemetry()
	defer thinker.telemetry.Stop()

	if err := thinker.threads.SpawnWithOpts("worker", "Work", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	if err := thinker.threads.SpawnWithOpts("unconscious", "Consolidate memories", nil, SpawnOpts{System: true, DeferRun: true}); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}
	thinker.threads.mu.RLock()
	worker := thinker.threads.threads["worker"]
	unconscious := thinker.threads.threads["unconscious"]
	thinker.threads.mu.RUnlock()

	_, _, results := threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name: "send", Args: map[string]string{"id": "unconscious", "message": "store this"}, NativeID: "send-1",
	}}, nil)
	if len(results) != 1 || !strings.Contains(results[0].Content, "platform-managed") {
		t.Fatalf("child send results = %+v", results)
	}
	if got := unconscious.Thinker.drainEventTexts(); len(got) != 0 {
		t.Fatalf("rejected child send reached system thread: %v", got)
	}
	events, _ := thinker.telemetry.StoredEvents(0)
	for _, event := range events {
		if event.Type == "thread.message" {
			t.Fatalf("rejected child send emitted thread.message telemetry: %+v", event)
		}
	}
}

func TestMainSendReceiptKicksOnceWithoutHotLoop(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("worker", "Receive work.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	worker := thinker.threads.threads["worker"]
	call := toolCall{
		Name:     "send",
		Args:     map[string]string{"id": "worker", "message": "inspect account 42"},
		Raw:      "send",
		NativeID: "send-1",
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{call}, nil)
	if !thinker.kickNextTurn {
		t.Fatal("successful main send should kick one receipt-processing turn")
	}
	if len(results) != 1 || results[0].IsError ||
		!strings.Contains(results[0].Content, "Message delivered to worker") ||
		!strings.Contains(results[0].Content, "not completion by the recipient") ||
		!strings.Contains(results[0].Content, "Do not resend") {
		t.Fatalf("send receipt = %+v", results)
	}
	if got := worker.Thinker.drainEventTexts(); len(got) != 1 || got[0] != "[from:main] inspect account 42" {
		t.Fatalf("worker inbox = %v", got)
	}

	thinker.kickNextTurn = false
	call.NativeID = "send-2"
	mainToolHandler(thinker)(thinker, []toolCall{call}, nil)
	if thinker.kickNextTurn {
		t.Fatal("second consecutive send should not rearm the completion loop")
	}

	mainToolHandler(thinker)(thinker, nil, nil)
	call.NativeID = "send-3"
	mainToolHandler(thinker)(thinker, []toolCall{call}, nil)
	if !thinker.kickNextTurn {
		t.Fatal("send completion guard did not reset after a non-send turn")
	}
}

func TestWorkerSendReceiptKicksAndDeliversToMain(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("worker", "Report to main.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	thinker.drainEventTexts() // discard the worker-started event
	worker := thinker.threads.threads["worker"]

	_, _, results := threadToolHandler(worker, thinker.threads)(worker.Thinker, []toolCall{{
		Name:     "send",
		Args:     map[string]string{"id": "main", "message": "analysis ready"},
		Raw:      "send",
		NativeID: "send-1",
	}}, nil)
	if !worker.Thinker.kickNextTurn {
		t.Fatal("successful worker send should kick one receipt-processing turn")
	}
	if len(results) != 1 || results[0].IsError || !strings.Contains(results[0].Content, "Message delivered to main") {
		t.Fatalf("worker send receipt = %+v", results)
	}
	if got := thinker.drainEventTexts(); len(got) != 1 || got[0] != "[from:worker] analysis ready" {
		t.Fatalf("main inbox = %v", got)
	}
}

func TestConversationSendToMainUsesAuthoritativeRelayEnvelope(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("chat-test", "Relay owner requests.", nil, SpawnOpts{
		DeferRun:     true,
		Conversation: true,
	}); err != nil {
		t.Fatalf("spawn conversation: %v", err)
	}
	thinker.drainEventTexts()
	conversation := thinker.threads.threads["chat-test"]

	_, _, results := threadToolHandler(conversation, thinker.threads)(conversation.Thinker, []toolCall{{
		Name:     "send",
		Args:     map[string]string{"id": "main", "message": "From now on, run the daily check-in at 10:00 UTC."},
		Raw:      "send",
		NativeID: "send-conversation",
	}}, nil)
	if len(results) != 1 || results[0].IsError {
		t.Fatalf("conversation send result = %+v", results)
	}
	if got := thinker.drainEventTexts(); len(got) != 1 || got[0] != "[from-conversation:chat-test] From now on, run the daily check-in at 10:00 UTC." {
		t.Fatalf("main inbox = %v", got)
	}
}

func TestSendFailureCorrectionAndReportingTurnsAreBounded(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	handler := mainToolHandler(thinker)
	call := toolCall{Name: "send", Args: map[string]string{"id": "main"}, Raw: "send", NativeID: "send-1"}

	_, _, first := handler(thinker, []toolCall{call}, nil)
	if !thinker.kickNextTurn || len(first) != 1 || !first[0].IsError || !strings.Contains(first[0].Content, "retry once now") {
		t.Fatalf("first send failure = kick:%v results:%+v", thinker.kickNextTurn, first)
	}

	thinker.kickNextTurn = false
	call.NativeID = "send-2"
	_, _, second := handler(thinker, []toolCall{call}, nil)
	if !thinker.kickNextTurn || len(second) != 1 || !strings.Contains(second[0].Content, "do not call send again") {
		t.Fatalf("second send failure = kick:%v results:%+v", thinker.kickNextTurn, second)
	}

	thinker.kickNextTurn = false
	call.NativeID = "send-3"
	handler(thinker, []toolCall{call}, nil)
	if thinker.kickNextTurn {
		t.Fatal("third consecutive send failure should not continue the retry loop")
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

func TestMainUpdateEmptyDirectiveSectionPayloadCreatesMarkdown(t *testing.T) {
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
	if err := thinker.threads.SpawnWithOpts("worker", "", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "update",
		Args: map[string]string{
			"id":      "worker",
			"section": "Goals",
			"content": "- Ship",
		},
		Raw:      "update",
		NativeID: "call-1",
	}}, nil)

	if len(results) != 1 || results[0].CallID != "call-1" || results[0].Content != "thread worker updated" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
	got, err := thinker.threads.Directive("worker")
	if err != nil {
		t.Fatalf("worker directive: %v", err)
	}
	want := "# Goals\n- Ship"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after update")
	}
}

func TestMainUpdateDirectiveBatchPatch(t *testing.T) {
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
	if err := thinker.threads.SpawnWithOpts("worker", "# Schedule\n- daily_check: 09:00 Europe/Madrid\n# Goals\n- Ship", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "update",
		Args: map[string]string{
			"id": "worker",
			"edits": `[
				{"mode":"section_replace_line","section":"Schedule","match":"daily_check:","content":"- daily_check: 07:30 Europe/Madrid"},
				{"mode":"section_append","section":"Goals","content":"- Report release readiness"}
			]`,
		},
		Raw:      "update",
		NativeID: "call-1",
	}}, nil)

	if len(results) != 1 || results[0].CallID != "call-1" || results[0].Content != "thread worker updated" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
	got, err := thinker.threads.Directive("worker")
	if err != nil {
		t.Fatalf("worker directive: %v", err)
	}
	want := "# Schedule\n- daily_check: 07:30 Europe/Madrid\n# Goals\n- Ship\n- Report release readiness"
	if got != want {
		t.Fatalf("directive:\n%s\nwant:\n%s", got, want)
	}
	if !thinker.kickNextTurn {
		t.Fatal("kickNextTurn should be true after update")
	}
}

func TestMainUpdateReportsDirectiveStructureWarning(t *testing.T) {
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
	if err := thinker.threads.SpawnWithOpts("worker", "# Schedule\n- daily_check: 09:00 Europe/Madrid", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}

	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "update",
		Args: map[string]string{
			"id":        "worker",
			"edit_mode": "section_append",
			"section":   "Schedule",
			"content":   "- daily_check: 07:30 Europe/Madrid",
		},
		Raw:      "update",
		NativeID: "call-1",
	}}, nil)

	if len(results) != 1 || !strings.Contains(results[0].Content, `thread worker updated; warning: conflicting "daily_check" rules in section "Schedule"`) {
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
