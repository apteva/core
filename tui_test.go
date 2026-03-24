package main

import (
	"strings"
	"testing"
)

func TestWrapText(t *testing.T) {
	tests := []struct {
		name  string
		input string
		width int
		check func(string) bool
		desc  string
	}{
		{
			name: "wraps long line",
			input: "the quick brown fox jumps over the lazy dog",
			width: 20,
			check: func(s string) bool {
				for _, line := range strings.Split(s, "\n") {
					if len(line) > 20 {
						return false
					}
				}
				return true
			},
			desc: "all lines <= 20 chars",
		},
		{
			name:  "preserves short line",
			input: "short",
			width: 20,
			check: func(s string) bool { return s == "short" },
			desc:  "unchanged",
		},
		{
			name:  "zero width",
			input: "anything",
			width: 0,
			check: func(s string) bool { return s == "anything" },
			desc:  "unchanged",
		},
		{
			name:  "preserves newlines",
			input: "line1\nline2\nline3",
			width: 80,
			check: func(s string) bool { return strings.Count(s, "\n") == 2 },
			desc:  "2 newlines",
		},
		{
			name:  "single long word",
			input: "superlongwordthatcannotbreak",
			width: 10,
			check: func(s string) bool { return s == "superlongwordthatcannotbreak" },
			desc:  "unbroken since no space",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := wrapText(tt.input, tt.width)
			if !tt.check(result) {
				t.Errorf("wrapText(%q, %d) = %q — expected %s", tt.input, tt.width, result, tt.desc)
			}
		})
	}
}

func TestWrapText_PreservesAllWords(t *testing.T) {
	input := "the quick brown fox jumps over the lazy dog"
	result := wrapText(input, 15)
	// All words should be present
	for _, word := range strings.Fields(input) {
		if !strings.Contains(result, word) {
			t.Errorf("word %q missing from wrapped result: %q", word, result)
		}
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		input  string
		max    int
		expect string
	}{
		{"hello", 10, "hello"},
		{"hello world", 8, "hello..."},
		{"hi", 2, "hi"},
		{"hello", 3, "hel"},
		{"", 5, ""},
		{"anything", 0, ""},
	}
	for _, tt := range tests {
		result := truncate(tt.input, tt.max)
		if result != tt.expect {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.input, tt.max, result, tt.expect)
		}
	}
}

func TestPanelWidths(t *testing.T) {
	thinker := &Thinker{
		inbox:  make(chan string, 1),
		wakeup: make(chan struct{}, 1),
		events: make(chan ThinkEvent, 1),
		memory: &MemoryStore{path: "/dev/null"},
		config: &Config{Directive: "test"},
	}
	m := newModel(thinker)

	// Narrow terminal — no left panel
	m.width = 60
	if m.leftPanelWidth() != 0 {
		t.Errorf("expected 0 left panel width at 60, got %d", m.leftPanelWidth())
	}
	if m.thoughtsPanelWidth() != 60 {
		t.Errorf("expected full width thoughts at 60, got %d", m.thoughtsPanelWidth())
	}

	// Wide terminal
	m.width = 120
	lw := m.leftPanelWidth()
	tw := m.thoughtsPanelWidth()
	if lw <= 0 {
		t.Errorf("expected positive left panel width, got %d", lw)
	}
	if tw <= 0 {
		t.Errorf("expected positive thoughts width, got %d", tw)
	}
	if lw+tw+1 != 120 {
		t.Errorf("widths should sum to terminal width: %d + %d + 1 = %d, want 120", lw, tw, lw+tw+1)
	}

	// Always 1/3
	m.width = 200
	if m.leftPanelWidth() != 66 {
		t.Errorf("left panel should be 1/3 of 200 = 66, got %d", m.leftPanelWidth())
	}
}

func TestPanelToggle(t *testing.T) {
	m := model{panel: panelChat}
	if m.panel != panelChat {
		t.Error("should start in chat mode")
	}
	// Simulate 'm' press would toggle to memory
	m.panel = panelMemory
	if m.panel != panelMemory {
		t.Error("should be in memory mode")
	}
	m.panel = panelChat
	if m.panel != panelChat {
		t.Error("should be back in chat mode")
	}
}
