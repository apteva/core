package main

import (
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
		{"48h", 24 * time.Hour, true},             // clamped to max
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
	items := thinker.drainEvents()
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

	items := thinker.drainEvents()
	if len(items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(items))
	}
	if items[0] != "msg1" || items[1] != "msg2" || items[2] != "msg3" {
		t.Errorf("unexpected items: %v", items)
	}

	// Should be empty now
	items2 := thinker.drainEvents()
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
	items := thinker.drainEvents()
	if len(items) != 1 || items[0] != "test event" {
		t.Errorf("unexpected items: %v", items)
	}
}

func TestInjectUserMessage(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("test", 10),
		threadID: "test",
	}
	thinker.InjectUserMessage("marco", "Hello")
	time.Sleep(10 * time.Millisecond)
	items := thinker.drainEvents()
	if len(items) != 1 || items[0] != "[user:marco] Hello" {
		t.Errorf("expected '[user:marco] Hello', got %v", items)
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
