package main

import (
	"strings"
	"testing"
	"time"
)

func TestParseToolCalls_Single(t *testing.T) {
	text := `Some thought [[web url="https://example.com"]] more text`
	calls := parseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "web" {
		t.Errorf("expected name 'web', got %q", calls[0].Name)
	}
	if calls[0].Args["url"] != "https://example.com" {
		t.Errorf("expected url, got %q", calls[0].Args["url"])
	}
}

func TestParseToolCalls_Multiple(t *testing.T) {
	text := `[[reply message="Hello!"]] and then [[web url="https://x.com"]]`
	calls := parseToolCalls(text)
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
	if calls[0].Name != "reply" {
		t.Errorf("expected 'reply', got %q", calls[0].Name)
	}
	if calls[0].Args["message"] != "Hello!" {
		t.Errorf("expected message 'Hello!', got %q", calls[0].Args["message"])
	}
	if calls[1].Name != "web" {
		t.Errorf("expected 'web', got %q", calls[1].Name)
	}
}

func TestParseToolCalls_MultilineValue(t *testing.T) {
	text := "[[reply message=\"Hello\nWorld\nMultiple lines\"]]"
	calls := parseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Args["message"] != "Hello\nWorld\nMultiple lines" {
		t.Errorf("unexpected message: %q", calls[0].Args["message"])
	}
}

func TestParseToolCalls_NoMatch(t *testing.T) {
	calls := parseToolCalls("no tool calls here")
	if len(calls) != 0 {
		t.Errorf("expected 0 calls, got %d", len(calls))
	}
}

func TestParseToolCalls_Pace(t *testing.T) {
	text := `thinking... [[pace rate="slow"]]`
	calls := parseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "pace" {
		t.Errorf("expected 'pace', got %q", calls[0].Name)
	}
	if calls[0].Args["rate"] != "slow" {
		t.Errorf("expected 'slow', got %q", calls[0].Args["rate"])
	}
}

func TestParseToolCalls_EscapedQuotes(t *testing.T) {
	text := `[[spawn id="timer" directive="Send a notification saying \"hello world\" to the user" tools="pushover_send_notification"]]`
	calls := parseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "spawn" {
		t.Errorf("expected 'spawn', got %q", calls[0].Name)
	}
	if !strings.Contains(calls[0].Args["directive"], `"hello world"`) {
		t.Errorf("expected unescaped quotes in directive, got %q", calls[0].Args["directive"])
	}
	if calls[0].Args["tools"] != "pushover_send_notification" {
		t.Errorf("expected tools, got %q", calls[0].Args["tools"])
	}
}

func TestParseToolCalls_MultipleArgs(t *testing.T) {
	text := `[[foo key1="val1" key2="val2"]]`
	calls := parseToolCalls(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Args["key1"] != "val1" || calls[0].Args["key2"] != "val2" {
		t.Errorf("unexpected args: %v", calls[0].Args)
	}
}

func TestStripToolCalls(t *testing.T) {
	text := "Before [[reply message=\"Hi\"]] After [[web url=\"https://x.com\"]] End"
	result := stripToolCalls(text)
	if result != "Before  After  End" {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestStripToolCalls_NoTools(t *testing.T) {
	text := "No tools here"
	result := stripToolCalls(text)
	if result != "No tools here" {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestStripHTML(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		contains string
		excludes string
	}{
		{"removes tags", "<p>Hello</p>", "Hello", "<p>"},
		{"removes script", "<script>alert('x')</script>Text", "Text", "alert"},
		{"removes style", "<style>body{}</style>Text", "Text", "body"},
		{"decodes entities", "&amp; &lt; &gt; &quot; &#39;", "& < > \" '", "&amp;"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripHTML(tt.input)
			if tt.contains != "" && !containsStr(result, tt.contains) {
				t.Errorf("expected result to contain %q, got %q", tt.contains, result)
			}
			if tt.excludes != "" && containsStr(result, tt.excludes) {
				t.Errorf("expected result to NOT contain %q, got %q", tt.excludes, result)
			}
		})
	}
}

func TestCollapseWhitespace(t *testing.T) {
	input := "line1\n\n\n\nline2\n\nline3"
	result := collapseWhitespace(input)
	expected := "line1\n\nline2\n\nline3"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestCollapseWhitespace_Empty(t *testing.T) {
	result := collapseWhitespace("")
	if result != "" {
		t.Errorf("expected empty, got %q", result)
	}
}

// --- Config / Supervised Mode Tests ---

func TestConfig_DefaultMode(t *testing.T) {
	c := &Config{}
	if c.GetMode() != ModeAutonomous {
		t.Errorf("expected autonomous, got %s", c.GetMode())
	}
}

func TestConfig_SetMode(t *testing.T) {
	c := &Config{path: "/dev/null"}
	c.SetMode(ModeCautious)
	if c.GetMode() != ModeCautious {
		t.Errorf("expected cautious, got %s", c.GetMode())
	}
	c.SetMode(ModeAutonomous)
	if c.GetMode() != ModeAutonomous {
		t.Errorf("expected autonomous, got %s", c.GetMode())
	}
	c.SetMode(ModeLearn)
	if c.GetMode() != ModeLearn {
		t.Errorf("expected learn, got %s", c.GetMode())
	}
}

func TestToolArgsSummary(t *testing.T) {
	call := toolCall{Name: "web", Args: map[string]string{"url": "https://example.com"}}
	summary := toolArgsSummary(call)
	if summary != "url=https://example.com" {
		t.Errorf("unexpected summary: %q", summary)
	}
}

func TestToolArgsSummary_Truncated(t *testing.T) {
	longVal := strings.Repeat("x", 100)
	call := toolCall{Name: "reply", Args: map[string]string{"message": longVal}}
	summary := toolArgsSummary(call)
	if !strings.Contains(summary, "...") {
		t.Error("expected truncation in summary")
	}
}

// --- Multimodal Tests ---

func TestMessage_TextContent_Plain(t *testing.T) {
	m := Message{Role: "user", Content: "hello"}
	if m.TextContent() != "hello" {
		t.Errorf("expected 'hello', got %q", m.TextContent())
	}
	if m.HasParts() {
		t.Error("expected no parts")
	}
}

func TestMessage_TextContent_WithParts(t *testing.T) {
	m := Message{
		Role:    "user",
		Content: "fallback",
		Parts: []ContentPart{
			{Type: "text", Text: "describe this"},
			{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/cat.jpg"}},
		},
	}
	if m.TextContent() != "describe this" {
		t.Errorf("expected 'describe this', got %q", m.TextContent())
	}
	if !m.HasParts() {
		t.Error("expected HasParts to be true")
	}
}

func TestMessage_TextContent_PartsNoText(t *testing.T) {
	m := Message{
		Role:    "user",
		Content: "fallback",
		Parts:   []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/cat.jpg"}}},
	}
	// Falls back to Content when no text part
	if m.TextContent() != "fallback" {
		t.Errorf("expected 'fallback', got %q", m.TextContent())
	}
}

func TestDetectImageParts(t *testing.T) {
	parts := detectImageParts("describe this https://example.com/photo.jpg please")
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "describe this  please" {
		t.Errorf("unexpected text part: %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL.URL != "https://example.com/photo.jpg" {
		t.Errorf("unexpected image part: %+v", parts[1])
	}
}

func TestDetectImageParts_NoImages(t *testing.T) {
	parts := detectImageParts("just normal text here")
	if parts != nil {
		t.Errorf("expected nil, got %d parts", len(parts))
	}
}

func TestDetectImageParts_MultipleImages(t *testing.T) {
	parts := detectImageParts("compare https://a.com/1.png and https://b.com/2.webp")
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text + 2 images), got %d", len(parts))
	}
	if parts[0].Type != "text" {
		t.Error("first part should be text")
	}
	if parts[1].Type != "image_url" || parts[2].Type != "image_url" {
		t.Error("second and third parts should be image_url")
	}
}

func TestDetectImageParts_OnlyImage(t *testing.T) {
	parts := detectImageParts("https://example.com/photo.png")
	if len(parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(parts))
	}
	if parts[0].Type != "image_url" {
		t.Errorf("expected image_url, got %s", parts[0].Type)
	}
}

func TestExecuteTool_NoBlock(t *testing.T) {
	bus := NewEventBus()
	cfg := &Config{Mode: ModeAutonomous, path: "/dev/null"}
	thinker := &Thinker{
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		config:    cfg,
		quit:      make(chan struct{}),
		telemetry: NewTelemetry(),
		threadID:  "main",
	}

	call := toolCall{Name: "web", Args: map[string]string{"url": "https://example.com"}}

	done := make(chan struct{})
	go func() {
		executeTool(thinker, call)
		close(done)
	}()

	select {
	case <-done:
		// good — should not block
	case <-time.After(500 * time.Millisecond):
		t.Error("executeTool blocked")
	}
}

func containsStr(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && searchStr(s, substr)))
}

func searchStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
