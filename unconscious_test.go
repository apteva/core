package core

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type scriptedUnconsciousProvider struct {
	mu    sync.Mutex
	calls int
}

func (p *scriptedUnconsciousProvider) Chat(_ context.Context, _ []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++

	hasTool := func(name string) bool {
		for _, tool := range tools {
			if tool.Name == name {
				return true
			}
		}
		return false
	}
	switch p.calls {
	case 1:
		if !hasTool("review_history") {
			return ChatResponse{Text: "missing review_history"}, nil
		}
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "review-1", Name: "review_history", Args: map[string]string{"limit": "50"}}}}, nil
	case 2:
		if !hasTool("memory_remember") {
			return ChatResponse{Text: "missing memory_remember"}, nil
		}
		return ChatResponse{ToolCalls: []NativeToolCall{{
			ID:   "remember-1",
			Name: "memory_remember",
			Args: map[string]string{
				"content": "The deterministic startup-memory sentinel is cobalt-913.",
				"tags":    "test,startup",
				"weight":  "0.95",
			},
		}}}, nil
	default:
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "pace", Name: "pace", Args: map[string]string{"sleep": "1h"}}}}, nil
	}
}

func (p *scriptedUnconsciousProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "test", ModelMedium: "test", ModelSmall: "test"}
}
func (p *scriptedUnconsciousProvider) Name() string                           { return "scripted-unconscious" }
func (p *scriptedUnconsciousProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *scriptedUnconsciousProvider) SupportsNativeTools() bool              { return true }
func (p *scriptedUnconsciousProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *scriptedUnconsciousProvider) SetBuiltinTools([]string)               {}
func (p *scriptedUnconsciousProvider) WithBuiltins([]string) LLMProvider      { return p }

type thresholdLifecycleProvider struct {
	mu              sync.Mutex
	initialPaceOnce sync.Once
	initialPaced    chan struct{}
	wakeSeen        chan struct{}
	remembered      bool
}

func (p *thresholdLifecycleProvider) Chat(_ context.Context, messages []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	hasTool := func(name string) bool {
		for _, tool := range tools {
			if tool.Name == name {
				return true
			}
		}
		return false
	}
	if !hasTool("review_history") {
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "main-pace", Name: "pace", Args: map[string]string{"sleep": "1h"}}}}, nil
	}

	woke := false
	hasReviewResult := false
	for _, message := range messages {
		if strings.Contains(message.Content, "[wake] history grew") {
			woke = true
		}
		for _, result := range message.ToolResults {
			if result.CallID == "threshold-review" {
				hasReviewResult = true
			}
		}
	}
	if !woke {
		p.initialPaceOnce.Do(func() { close(p.initialPaced) })
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "initial-pace", Name: "pace", Args: map[string]string{"sleep": "24h"}}}}, nil
	}
	select {
	case p.wakeSeen <- struct{}{}:
	default:
	}
	if !hasReviewResult {
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "threshold-review", Name: "review_history", Args: map[string]string{"limit": "50"}}}}, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.remembered {
		p.remembered = true
		return ChatResponse{ToolCalls: []NativeToolCall{{
			ID: "threshold-remember", Name: "memory_remember",
			Args: map[string]string{
				"content": "The threshold lifecycle deployment token is heliotrope-threshold-582.",
				"tags":    "test,threshold-lifecycle,deployment",
				"weight":  "0.95",
			},
		}}}, nil
	}
	return ChatResponse{ToolCalls: []NativeToolCall{{ID: "final-pace", Name: "pace", Args: map[string]string{"sleep": "24h"}}}}, nil
}

func (p *thresholdLifecycleProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "test", ModelMedium: "test", ModelSmall: "test"}
}
func (p *thresholdLifecycleProvider) Name() string { return "threshold-lifecycle" }
func (p *thresholdLifecycleProvider) CostPer1M() (float64, float64, float64) {
	return 0, 0, 0
}
func (p *thresholdLifecycleProvider) SupportsNativeTools() bool            { return true }
func (p *thresholdLifecycleProvider) AvailableBuiltinTools() []BuiltinTool { return nil }
func (p *thresholdLifecycleProvider) SetBuiltinTools([]string)             {}
func (p *thresholdLifecycleProvider) WithBuiltins([]string) LLMProvider    { return p }

func TestUnconsciousAutoSpawnRunsAndCreatesMemory(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())
	if err := os.MkdirAll("history", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join("history", "main.jsonl"), []byte("{\"role\":\"user\",\"content\":\"Remember cobalt-913.\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := NewConfig()
	cfg.Directive = "Test parent."
	cfg.Unconscious = true
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	parent := NewThinker("", &scriptedUnconsciousProvider{}, cfg)
	t.Cleanup(parent.Stop)

	deadline := time.After(15 * time.Second)
	for {
		for _, rec := range parent.memory.Active() {
			if strings.Contains(rec.Content, "cobalt-913") {
				goto created
			}
		}
		select {
		case <-deadline:
			t.Fatalf("auto-spawned unconscious created no sentinel memory; active=%v", memoryContents(parent.memory.Active()))
		case <-time.After(20 * time.Millisecond):
		}
	}

created:
	parent.threads.mu.RLock()
	thread := parent.threads.threads["unconscious"]
	parent.threads.mu.RUnlock()
	if thread == nil {
		t.Fatal("unconscious thread was not auto-spawned")
	}
	if !thread.Thinker.systemThread {
		t.Fatal("auto-spawned unconscious thread is not marked system-only")
	}
	if parent.unconsciousSafety == nil {
		t.Fatal("unconscious safety tracker was not initialized")
	}
	if _, err := os.Stat("memory.jsonl"); err != nil {
		t.Fatalf("memory journal was not created: %v", err)
	}
}

func TestUnconsciousHistoryGrowthWakeCreatesPersistentMemory(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := &thresholdLifecycleProvider{
		initialPaced: make(chan struct{}),
		wakeSeen:     make(chan struct{}, 1),
	}
	cfg := NewConfig()
	cfg.Directive = "Persist incoming test facts through the unconscious thread."
	cfg.Unconscious = true
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	parent := NewThinker("", provider, cfg)
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.Stop()
	})
	select {
	case <-provider.initialPaced:
	case <-time.After(5 * time.Second):
		t.Fatal("unconscious thread did not complete its initial idle call")
	}
	// The iteration records the safety baseline immediately after Chat returns.
	time.Sleep(100 * time.Millisecond)
	baseline := fileSize(filepath.Join("history", "main.jsonl"))

	go parent.Run()
	padding := strings.Repeat("non-memory transport padding for threshold coverage. ", 700)
	parent.InjectConsole("User explicitly said this durable fact: the deployment token is heliotrope-threshold-582. " + padding)
	parent.InjectConsole("User confirmed heliotrope-threshold-582 must be available in future deployment checks. " + padding)
	waitForHistoryGrowth(t, baseline+unconsciousByteThreshold, 5*time.Second)

	ticks := make(chan time.Time, 1)
	floorDone := make(chan struct{})
	go func() {
		parent.runUnconsciousSafetyFloors(ticks, func() int64 { return fileSize(filepath.Join("history", "main.jsonl")) })
		close(floorDone)
	}()
	ticks <- time.Now().Add(time.Minute)
	select {
	case <-provider.wakeSeen:
	case <-time.After(5 * time.Second):
		t.Fatal("50KB history growth did not wake the unconscious thread")
	}

	waitForMemory(t, parent.memory, 5*time.Second, func(record MemoryRecord) bool {
		return strings.Contains(record.Content, "heliotrope-threshold-582")
	})
	if fileSize("memory.jsonl") == 0 {
		t.Fatal("threshold-triggered memory was not persisted")
	}
	reloaded := NewMemoryStore("")
	results := reloaded.Recall("Which deployment token is used for threshold lifecycle checks?", 5)
	if !memoryResultsContain(results, "heliotrope-threshold-582") {
		t.Fatalf("restarted store could not recall threshold memory: %v", memoryContents(results))
	}

	parent.Stop()
	select {
	case <-floorDone:
	case <-time.After(time.Second):
		t.Fatal("test safety-floor loop did not stop")
	}
}

func waitForHistoryGrowth(t *testing.T, minimum int64, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fileSize(filepath.Join("history", "main.jsonl")) >= minimum {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("history size = %d, want at least %d", fileSize(filepath.Join("history", "main.jsonl")), minimum)
}

func TestUnconsciousSafetyStateThresholdsAndCycleReset(t *testing.T) {
	base := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	state := newUnconsciousSafetyState(base, 1_000)

	if reason, wake := state.claimWake(base.Add(time.Hour), 1_000+unconsciousByteThreshold-1); wake {
		t.Fatalf("woke below byte threshold: %q", reason)
	}
	reason, wake := state.claimWake(base.Add(time.Hour), 1_000+unconsciousByteThreshold)
	if !wake || !strings.Contains(reason, "history grew 50KB") {
		t.Fatalf("byte-threshold wake = (%q, %v)", reason, wake)
	}
	if reason, wake := state.claimWake(base.Add(2*time.Hour), 1_000+2*unconsciousByteThreshold); wake {
		t.Fatalf("duplicate pending wake = (%q, %v)", reason, wake)
	}

	cycleAt := base.Add(3 * time.Hour)
	state.recordCycle(cycleAt, 10_000)
	if reason, wake := state.claimWake(cycleAt.Add(unconsciousMaxQuietInterval-time.Minute), 10_000); wake {
		t.Fatalf("woke before quiet threshold: %q", reason)
	}
	reason, wake = state.claimWake(cycleAt.Add(unconsciousMaxQuietInterval), 10_000)
	if !wake || !strings.Contains(reason, "no cycle in 8h0m") {
		t.Fatalf("quiet-threshold wake = (%q, %v)", reason, wake)
	}
}

func TestUnconsciousSafetyFloorsPublishTargetedWake(t *testing.T) {
	base := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	bus := NewEventBus()
	sub := bus.Subscribe("unconscious", 2)
	thinker := &Thinker{
		bus:               bus,
		quit:              make(chan struct{}),
		unconsciousSafety: newUnconsciousSafetyState(base, 100),
	}
	ticks := make(chan time.Time, 1)
	done := make(chan struct{})
	go func() {
		thinker.runUnconsciousSafetyFloors(ticks, func() int64 { return 100 })
		close(done)
	}()

	ticks <- base.Add(unconsciousMaxQuietInterval)
	select {
	case ev := <-sub.C:
		if ev.To != "unconscious" || ev.Type != EventInbox || !strings.Contains(ev.Text, "[wake] no cycle in 8h0m") {
			t.Fatalf("unexpected wake event: %+v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("safety floor did not publish a targeted wake")
	}

	close(thinker.quit)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("safety floor loop did not stop")
	}
}
