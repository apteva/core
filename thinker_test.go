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

func TestModelTier_String(t *testing.T) {
	if ModelLarge.String() != "large" {
		t.Errorf("expected 'large', got %q", ModelLarge.String())
	}
	if ModelSmall.String() != "small" {
		t.Errorf("expected 'small', got %q", ModelSmall.String())
	}
}

func TestModelTier_ID(t *testing.T) {
	if ModelLarge.ID() == "" {
		t.Error("large model ID should not be empty")
	}
	if ModelSmall.ID() == "" {
		t.Error("small model ID should not be empty")
	}
	// Both may use the same model ID temporarily
	_ = ModelLarge.ID()
	_ = ModelSmall.ID()
}

func TestModelNames(t *testing.T) {
	for name, tier := range modelNames {
		if tier.String() != name {
			t.Errorf("modelNames[%q] = %d, String() = %q", name, tier, tier.String())
		}
	}
}

func TestDrainInbox_Empty(t *testing.T) {
	thinker := &Thinker{
		inbox: make(chan string, 10),
	}
	items := thinker.drainInbox()
	if len(items) != 0 {
		t.Errorf("expected empty, got %d items", len(items))
	}
}

func TestDrainInbox_WithMessages(t *testing.T) {
	thinker := &Thinker{
		inbox: make(chan string, 10),
	}
	thinker.inbox <- "msg1"
	thinker.inbox <- "msg2"
	thinker.inbox <- "msg3"

	items := thinker.drainInbox()
	if len(items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(items))
	}
	if items[0] != "msg1" || items[1] != "msg2" || items[2] != "msg3" {
		t.Errorf("unexpected items: %v", items)
	}

	// Should be empty now
	items2 := thinker.drainInbox()
	if len(items2) != 0 {
		t.Errorf("expected empty after drain, got %d", len(items2))
	}
}

func TestInject(t *testing.T) {
	thinker := &Thinker{
		inbox:  make(chan string, 10),
		wakeup: make(chan struct{}, 1),
	}
	thinker.Inject("test event")
	items := thinker.drainInbox()
	if len(items) != 1 || items[0] != "test event" {
		t.Errorf("unexpected items: %v", items)
	}
}

func TestInjectUserMessage(t *testing.T) {
	thinker := &Thinker{
		inbox:  make(chan string, 10),
		wakeup: make(chan struct{}, 1),
	}
	thinker.InjectUserMessage("marco", "Hello")
	items := thinker.drainInbox()
	if len(items) != 1 || items[0] != "[user:marco] Hello" {
		t.Errorf("expected '[user:marco] Hello', got %v", items)
	}
}

func TestWakeup_NonBlocking(t *testing.T) {
	thinker := &Thinker{
		inbox:  make(chan string, 10),
		wakeup: make(chan struct{}, 1),
	}
	// Should not block even when called multiple times
	thinker.wake()
	thinker.wake()
	thinker.wake()
	// Drain wakeup
	select {
	case <-thinker.wakeup:
	default:
		t.Error("expected wakeup signal")
	}
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
