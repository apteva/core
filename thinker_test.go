package main

import (
	"strings"
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
	thinker.InjectUserMessage("Hello")
	items := thinker.drainInbox()
	if len(items) != 1 || items[0] != "[user] Hello" {
		t.Errorf("expected '[user] Hello', got %v", items)
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

func TestBuildMemorySummary(t *testing.T) {
	thinker := &Thinker{}

	tests := []struct {
		name     string
		consumed []string
		thought  string
		replies  []string
		tools    []string
		contains []string
		empty    bool
	}{
		{
			name:     "user message and reply",
			consumed: []string{"[user] Hello"},
			thought:  "The user said hello, I should greet them back",
			replies:  []string{"Hi there!"},
			contains: []string{"User: Hello", "Replied: Hi there!", "Thought:"},
		},
		{
			name:     "tool call",
			consumed: []string{},
			thought:  "fetching data",
			tools:    []string{"[[web url=\"https://example.com\"]]"},
			contains: []string{"Called:", "example.com"},
		},
		{
			name:     "tool result event",
			consumed: []string{"[tool:web] Some content from the web"},
			thought:  "analyzing the content",
			contains: []string{"[tool:web]", "Thought:"},
		},
		{
			name:  "empty iteration",
			empty: true,
		},
		{
			name:     "strips tool calls from thought",
			thought:  `Thinking [[reply message="hi"]] more thought`,
			replies:  []string{"hi"},
			contains: []string{"Thought: Thinking", "more thought"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := thinker.buildMemorySummary(tt.consumed, tt.thought, tt.replies, tt.tools)
			if tt.empty {
				if result != "" {
					t.Errorf("expected empty, got %q", result)
				}
				return
			}
			for _, c := range tt.contains {
				if !strings.Contains(result, c) {
					t.Errorf("expected result to contain %q, got %q", c, result)
				}
			}
		})
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
