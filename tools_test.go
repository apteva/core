package main

import (
	"testing"
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
