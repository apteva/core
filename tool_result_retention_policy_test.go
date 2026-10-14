package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

func retainedResultModel(t *testing.T, window int) *Thinker {
	t.Helper()
	id := fmt.Sprintf("retained-result-%d", window)
	registerModelCapabilities(map[string]ModelCapabilities{id: {ContextWindow: window}})
	t.Cleanup(func() {
		runtimeModelCapabilities.Lock()
		delete(runtimeModelCapabilities.models, id)
		runtimeModelCapabilities.Unlock()
	})
	return &Thinker{provider: &efficiencyBudgetProvider{scriptedRetryProvider: &scriptedRetryProvider{name: "test"}, model: id}}
}

func TestToolResultRetentionAdaptiveCaps(t *testing.T) {
	for _, tc := range []struct{ window, cap int }{{64000, 16000}, {100000, 32000}, {200000, 64000}, {1000000, 64000}} {
		t.Run(strconv.Itoa(tc.window), func(t *testing.T) {
			th := retainedResultModel(t, tc.window)
			per, total := th.historicalToolResultBudgets()
			if per != tc.cap || total < per {
				t.Fatalf("window=%d per=%d total=%d", tc.window, per, total)
			}
		})
	}
	th := retainedResultModel(t, 20000)
	per, total := th.historicalToolResultBudgets()
	budget := estimatePreparedRequest("test", th.modelID(), nil, nil)
	if per > budget.InputBudget*4*3/10 || total > budget.InputBudget*4*3/10 {
		t.Fatal("small window caps ignored output reservation")
	}
}

func TestToolResultRetentionKeepsLongJSONFieldsVerbatimWithinCap(t *testing.T) {
	text := strings.Repeat("Exact procedure instructions. ", 100)
	raw, _ := json.Marshal(map[string]any{"instructions": text, "data": strings.Repeat("Evidence ", 600), "url": strings.Repeat("r", 700)})
	full := ToolResult{CallID: "read", Content: string(raw)}
	result := projectArchivedToolResult(full, 16000)
	if result.Content != full.Content || result.ContentIsPreview || strings.Contains(result.Content, "omitted_bytes") {
		t.Fatal("long fields were replaced despite fitting the whole-result budget")
	}
}

func TestToolResultRetentionAgeAloneDoesNotShortenHistory(t *testing.T) {
	th := &Thinker{threadID: "worker", messages: []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "Inspect."}, {Role: "assistant", ToolCalls: []NativeToolCall{{ID: "old", Name: "read"}}}, {Role: "user", ToolResults: []ToolResult{{CallID: "old", Content: strings.Repeat("instructions ", 6000)}}}, {Role: "assistant", Content: "one"}, {Role: "assistant", Content: "two"}, {Role: "assistant", Content: "three"}}}
	for i := 0; i < 30; i++ {
		th.markToolResultsConsumed(th.messages)
	}
	result := th.prepareToolResultRequest(th.messages)[3].ToolResults[0]
	if result.Content != th.messages[3].ToolResults[0].Content || th.promptCacheEpoch != 0 {
		t.Fatal("age without pressure shortened history")
	}
}

func TestToolResultRetentionPressureProtectsRecentBootstrapAndPending(t *testing.T) {
	th := &Thinker{threadID: "worker", session: NewSession(t.TempDir(), "worker")}
	th.messages = []Message{{Role: "system", Content: "stable"}, {Role: "assistant", ToolCalls: []NativeToolCall{{ID: "bootstrap", Name: "read"}}}, {Role: "user", ToolResults: []ToolResult{{CallID: "bootstrap", Content: strings.Repeat("bootstrap ", 4000)}}}, {Role: "user", Content: "Proceed."}}
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("call-%d", i)
		th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "read"}}})
		th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: strings.Repeat("current instructions ", 5000)}}}))
	}
	for i := 0; i < 3; i++ {
		th.markToolResultsConsumed(th.messages)
	}
	th.pendingTools.Store("call-1", "read")
	next := th.prepareToolResultRequest(th.messages)
	protected := th.protectedHistoricalToolResults(th.messages)
	shortened := 0
	for i, msg := range th.messages {
		for j, result := range msg.ToolResults {
			got := next[i].ToolResults[j]
			if protected[result.CallID] {
				if got.Content != result.Content {
					t.Fatalf("protected result %s changed", result.CallID)
				}
			} else {
				if !got.ContentIsPreview || !strings.Contains(got.Content, "history_search") {
					t.Fatalf("older result %s was not recoverably shortened", result.CallID)
				}
				shortened++
			}
		}
	}
	if shortened != 2 {
		t.Fatalf("shortened %d results; recent, bootstrap and pending protections missing", shortened)
	}
	if th.promptCacheEpoch != 1 {
		t.Fatal("expected one pressure boundary")
	}
	th.prepareToolResultRequest(th.messages)
	if th.promptCacheEpoch != 1 {
		t.Fatal("unchanged projected history churned cache")
	}
}

func TestToolResultRetentionExcerptKeepsHeadTailAndUTF8(t *testing.T) {
	payload := "HEAD " + strings.Repeat("証拠🔎", 30000) + " TAIL"
	result := projectArchivedToolResult(ToolResult{CallID: "call", Content: payload}, 16000)
	if len(result.Content) > 16000 || !utf8.ValidString(result.Content) || !strings.Contains(result.Content, "HEAD") || !strings.Contains(result.Content, "TAIL") || strings.Contains(result.Content, "omitted_bytes") {
		t.Fatal("invalid bounded whole-result excerpt")
	}
}

func TestHistorySearchRecoversExactArchivedResultAfterCompaction(t *testing.T) {
	memory, h := newHistoryTestStore(t)
	session := NewSession(filepath.Dir(memory.path), "worker")
	text := "SOURCE-HEAD " + strings.Repeat("長い exact instructions 🔎\n", 4000) + " SOURCE-TAIL"
	archived, err := session.archive.Archive(ToolResult{CallID: "original-call", Content: text})
	if err != nil {
		t.Fatal(err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{archived}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if _, err := session.checkpointRecoveredContext([]Message{{Role: "system", Content: "summary"}}); err != nil {
		t.Fatal(err)
	}
	h = newMemoryHistory(memory, h.config) // restart drops no archive references
	ctx := withMemoryCaller(context.Background(), "worker")
	var recovered strings.Builder
	offset := 0
	for {
		response := h.search(ctx, map[string]string{"call_id": "original-call", "offset": strconv.Itoa(offset), "limit": "16000"})
		if response.IsError {
			t.Fatal(response.Text)
		}
		var page struct {
			Text     string `json:"text"`
			Complete bool   `json:"complete"`
			Next     int    `json:"next_offset"`
		}
		if err := json.Unmarshal([]byte(response.Text), &page); err != nil {
			t.Fatal(err)
		}
		if !utf8.ValidString(page.Text) {
			t.Fatal("page split UTF-8")
		}
		recovered.WriteString(page.Text)
		if page.Complete {
			break
		}
		if page.Next <= offset {
			t.Fatal("pagination made no progress")
		}
		offset = page.Next
	}
	if recovered.String() != text {
		t.Fatal("archive recovery lost exact text")
	}
	if other := h.search(withMemoryCaller(context.Background(), "other"), map[string]string{"call_id": "original-call"}); !other.IsError {
		t.Fatal("another thread read the caller's archive")
	}
	for _, args := range []map[string]string{{"call_id": "original-call", "offset": "-1"}, {"call_id": "original-call", "limit": "64001"}, {"call_id": "original-call", "offset": "16"}, {"call_id": "../objects/foreign"}} {
		if !h.search(ctx, args).IsError {
			t.Fatalf("invalid archive request accepted: %v", args)
		}
	}
	if response := h.search(context.Background(), map[string]string{"call_id": "original-call"}); !response.IsError {
		t.Fatal("untrusted caller read archive")
	}
	if _, err := memory.ForgetSource("worker", "test withdrawal"); err != nil {
		t.Fatal(err)
	}
	if response := h.search(ctx, map[string]string{"call_id": "original-call"}); !response.IsError {
		t.Fatal("forgotten source remained readable")
	}
	// The original durable archive remains intact even when retrieval is denied.
	if _, err := os.Stat(session.archive.objectPath(archived.SHA256)); err != nil {
		t.Fatal(err)
	}
}

func TestHistorySearchRecoversSmallJournalResultAndRejectsSymlink(t *testing.T) {
	memory, h := newHistoryTestStore(t)
	session := NewSession(filepath.Dir(memory.path), "worker")
	original := strings.Repeat("exact data ", 100)
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "small", Content: original}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	ctx := withMemoryCaller(context.Background(), "worker")
	result := h.search(ctx, map[string]string{"call_id": "small"})
	var page struct {
		Text     string `json:"text"`
		Complete bool   `json:"complete"`
	}
	if result.IsError || json.Unmarshal([]byte(result.Text), &page) != nil || page.Text != original || !page.Complete {
		t.Fatalf("small result recovery: %s", result.Text)
	}
	other := NewSession(t.TempDir(), "foreign")
	if err := other.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "foreign-call", Content: "private"}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(session.path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(other.path, session.path); err != nil {
		t.Fatal(err)
	}
	if !h.search(ctx, map[string]string{"call_id": "foreign-call"}).IsError {
		t.Fatal("reader followed another journal's symlink")
	}
}

func TestToolResultRetentionRestoresUsingModelBudget(t *testing.T) {
	base := t.TempDir()
	session := NewSession(base, "worker")
	raw, _ := json.Marshal(map[string]any{"instructions": strings.Repeat("Exact original instructions. ", 1000), "id": "source"})
	payload := string(raw)
	if err := session.AppendMessage(Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "restored", Name: "read"}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "restored", Content: payload, Image: []byte("historical pixels")}}}, 0, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	for _, window := range []int{64000, 100000, 200000} {
		t.Run(strconv.Itoa(window), func(t *testing.T) {
			thinker := retainedResultModel(t, window)
			per, _ := thinker.historicalToolResultBudgets()
			restarted := NewSession(base, "worker")
			messages, _ := restarted.LoadTail(10, per)
			if len(messages) != 2 || len(messages[1].ToolResults) != 1 {
				t.Fatalf("restored messages: %+v", messages)
			}
			result := messages[1].ToolResults[0]
			if len(result.Image) != 0 || len(result.Content) > per {
				t.Fatal("restore exceeded its model budget or reloaded pixels")
			}
			if window >= 100000 {
				if result.Content != payload || result.ContentIsPreview {
					t.Fatal("restart shortened instructions that fit this model's history cap")
				}
			} else if !result.ContentIsPreview || !strings.Contains(result.Content, "history_search") {
				t.Fatal("smaller model did not receive a recoverable preview")
			}
			object, err := restarted.archive.Read(result.ArchiveRef)
			if err != nil || object.Content != payload {
				t.Fatal("model budget altered durable original", err)
			}
		})
	}
}
