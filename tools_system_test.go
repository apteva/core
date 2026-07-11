package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSkillWriteIsNotACoreToolAndSkillFilesAreNotPromptLoaded(t *testing.T) {
	t.Chdir(t.TempDir())
	registry := NewToolRegistry("")
	registerSystemTools(registry, &MemoryStore{path: memoryFile, byID: map[string]int{}})
	if tool := registry.Get("skill_write"); tool != nil {
		t.Fatalf("skill_write remains registered in core: %+v", tool)
	}
	if err := os.MkdirAll("skills", 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join("skills", "server-owned.md"), []byte("SERVER_SKILL_SENTINEL"), 0600); err != nil {
		t.Fatal(err)
	}
	prompt := buildSystemPrompt("Idle.", ModeAutonomous, registry, "", nil, nil, nil, nil)
	if strings.Contains(prompt, "SERVER_SKILL_SENTINEL") || strings.Contains(prompt, "[LEARNED SKILLS") {
		t.Fatalf("server-owned skill file leaked into core prompt: %q", prompt)
	}
}

func TestReadTailLinesBoundsLargeHistoryAndKeepsOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	largeOldPrefix := strings.Repeat("old-data", 1<<18) + "\n"
	data := largeOldPrefix + "latest-one\nlatest-two\n"
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	lines, err := readTailLines(path, 2, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 2 || lines[0] != "latest-one" || lines[1] != "latest-two" {
		t.Fatalf("tail lines = %#v", lines)
	}
}

// TestReviewHistory_ReturnsAllPlantedLines verifies the tool actually
// hands the unconscious every line we planted, not a truncated preview.
// The previous integration-test debugging suggested the model was
// looping on review_history; one possible reason was that the tool
// returned only the first line — this test rules that out.
func TestReviewHistory_ReturnsAllPlantedLines(t *testing.T) {
	dir := t.TempDir()
	prev, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	if err := os.MkdirAll("history", 0755); err != nil {
		t.Fatal(err)
	}
	planted := []string{
		`{"ts":"2026-04-26T10:00:00Z","role":"user","content":"[chat] Hi — I always prefer terse replies."}`,
		`{"ts":"2026-04-26T10:00:05Z","role":"assistant","content":"Got it."}`,
		`{"ts":"2026-04-26T10:01:00Z","role":"user","content":"[chat] My Postgres is on port 6543."}`,
		`{"ts":"2026-04-26T10:01:08Z","role":"assistant","content":"Noted."}`,
		`{"ts":"2026-04-26T10:30:00Z","role":"user","content":"[chat] Dashboard pulse strip task done."}`,
		`{"ts":"2026-04-26T10:30:05Z","role":"assistant","content":"Closed out."}`,
	}
	if err := os.WriteFile(filepath.Join("history", "main.jsonl"),
		[]byte(strings.Join(planted, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}

	registry := NewToolRegistry("")
	mem := &MemoryStore{path: memoryFile, byID: map[string]int{}}
	registerSystemTools(registry, mem)

	tool := registry.Get("review_history")
	if tool == nil {
		t.Fatal("review_history tool not registered")
	}
	if tool.Handler == nil {
		t.Fatal("review_history Handler is nil")
	}

	resp := tool.Handler(map[string]string{})
	t.Logf("review_history result: %d bytes", len(resp.Text))

	// Each planted line — full content — must be present in the
	// tool's response text.
	for i, ln := range planted {
		// Compare on the content field rather than the whole JSON
		// (whitespace-insensitive).
		var probe map[string]any
		_ = json.Unmarshal([]byte(ln), &probe)
		needle, _ := probe["content"].(string)
		if needle == "" {
			continue
		}
		if !strings.Contains(resp.Text, needle) {
			t.Errorf("planted line %d ('%s') missing from review_history output", i, needle)
		}
	}

	// Verify the limit knob clamps. With limit=2 we should see only
	// the last 2 lines (most recent first by tail order).
	respLim := tool.Handler(map[string]string{"limit": "2"})
	hits := 0
	for _, ln := range planted {
		var probe map[string]any
		_ = json.Unmarshal([]byte(ln), &probe)
		if needle, _ := probe["content"].(string); needle != "" && strings.Contains(respLim.Text, needle) {
			hits++
		}
	}
	if hits != 2 {
		t.Errorf("limit=2 returned %d lines, want 2", hits)
	}
	t.Logf("limit=2 result: %d bytes, %d planted lines visible", len(respLim.Text), hits)
}
