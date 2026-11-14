package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCompactionPreservesJournalOnlyResultsByOriginalCallID(t *testing.T) {
	for _, recovery := range []bool{true, false} {
		t.Run(map[bool]string{true: "recovery-checkpoint", false: "semantic-compaction"}[recovery], func(t *testing.T) {
			memory, h := newHistoryTestStore(t)
			session := NewSession(filepath.Dir(memory.path), "worker")
			text := "screenshot metadata: " + strings.Repeat("exact observation 🔎 ", 600)
			if len(text) >= largeToolResultThresholdBytes {
				t.Fatal("fixture must be below the normal archive threshold")
			}
			originals := []ToolResult{
				{CallID: "original-screenshot", ToolName: "computer_use", Content: text},
				{CallID: "same-payload-different-call", ToolName: "computer_use", Content: text},
				{CallID: "small-error", Content: "exact historical error", IsError: true},
			}
			for _, result := range originals {
				if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{result}}, 0, TokenUsage{}); err != nil {
					t.Fatal(err)
				}
				if _, ok := session.archive.RefForCall(result.CallID); ok {
					t.Fatal("result was archived before compaction; fixture misses the regression")
				}
			}
			if err := session.AppendMessage(Message{Role: "user", Content: "Current instruction stays."}, 0, TokenUsage{}); err != nil {
				t.Fatal(err)
			}
			// Only the current instruction is in the replacement live context.
			if recovery {
				if _, err := session.checkpointRecoveredContext([]Message{{Role: "user", Content: "Current instruction stays."}}); err != nil {
					t.Fatal(err)
				}
			} else {
				session.ForceCompact(1, func(string) string { return "Earlier observations compacted." })
			}
			journal, err := os.ReadFile(session.path)
			if err != nil || strings.Contains(string(journal), "original-screenshot") {
				t.Fatal("fixture did not displace the original result entries", err)
			}
			h = newMemoryHistory(memory, h.config) // Read through a fresh call ledger after restart.
			archive := NewToolResultArchive(filepath.Dir(memory.path), "worker")
			for _, original := range originals {
				result := h.search(withMemoryCaller(context.Background(), "worker"), map[string]string{"call_id": original.CallID, "limit": "64000"})
				var page struct {
					Text     string `json:"text"`
					Complete bool   `json:"complete"`
					IsError  bool   `json:"is_error"`
				}
				if result.IsError || json.Unmarshal([]byte(result.Text), &page) != nil || page.Text != original.Content || !page.Complete || page.IsError != original.IsError {
					t.Fatalf("lookup %s lost the exact original result: %s", original.CallID, result.Text)
				}
				if _, ok := archive.RefForCall(original.CallID); !ok {
					t.Fatalf("original call %s missing from durable ledger", original.CallID)
				}
			}
			first, _ := archive.RefForCall(originals[0].CallID)
			second, _ := archive.RefForCall(originals[1].CallID)
			if first != second {
				t.Fatal("identical payloads should share an object while keeping both call IDs")
			}
			if !h.search(withMemoryCaller(context.Background(), "other"), map[string]string{"call_id": originals[0].CallID}).IsError {
				t.Fatal("original-call lookup escaped the caller's thread")
			}
		})
	}
}

func TestJournalRewriteLeavesHistoryOnCallIndexFailure(t *testing.T) {
	session := NewSession(t.TempDir(), "worker")
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "journal-only", Content: "irreplaceable original"}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	// A directory cannot be opened as an append-only call ledger.
	if err := os.Mkdir(session.archive.callsPath, 0700); err != nil {
		t.Fatal(err)
	}
	session.mu.Lock()
	err = session.rewriteEntriesLocked([]SessionEntry{{Role: "_compacted", Summary: "replacement summary"}})
	session.mu.Unlock()
	if err == nil {
		t.Fatal("rewrite succeeded without a durable original-call mapping")
	}
	after, err := os.ReadFile(session.path)
	if err != nil || string(after) != string(before) {
		t.Fatal("failed preservation changed the journal", err)
	}
}

func TestRecoveryCheckpointDoesNotArchivePreviewAsOriginal(t *testing.T) {
	session := NewSession(t.TempDir(), "worker")
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "missing-original", Content: "incomplete excerpt", ContentIsPreview: true}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.checkpointRecoveredContext([]Message{{Role: "user", Content: "summary"}}); err == nil {
		t.Fatal("checkpoint indexed an incomplete preview as the original")
	}
	after, err := os.ReadFile(session.path)
	if err != nil || string(after) != string(before) {
		t.Fatal("failed checkpoint replaced the journal", err)
	}
	if _, ok := session.archive.RefForCall("missing-original"); ok {
		t.Fatal("incomplete preview was indexed as an original result")
	}
}
