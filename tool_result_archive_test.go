package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func largeArchivedPayload() string {
	return strings.Repeat("HEAD-", 14_000) + "MIDDLE_ONLY_ARCHIVE_SENTINEL" + strings.Repeat("-TAIL", 14_000)
}

func TestToolResultArchiveDeduplicatesPayloadAndPreservesCalls(t *testing.T) {
	baseDir := t.TempDir()
	archive := NewToolResultArchive(baseDir, "main")
	payload := largeArchivedPayload()
	first, err := archive.Archive(ToolResult{CallID: "call-1", ToolName: "media_search", Content: payload})
	if err != nil {
		t.Fatal(err)
	}
	second, err := archive.Archive(ToolResult{CallID: "call-2", ToolName: "media_search", Content: payload})
	if err != nil {
		t.Fatal(err)
	}
	if first.SHA256 == "" || first.SHA256 != second.SHA256 || first.ArchiveRef != second.ArchiveRef {
		t.Fatalf("identical payloads were not deduplicated: first=%+v second=%+v", first, second)
	}
	objects, err := filepath.Glob(filepath.Join(baseDir, historyDir, toolResultArchiveDir, toolResultArchiveObjectDir, "sha256", "*", "*.json"))
	if err != nil || len(objects) != 1 {
		t.Fatalf("archive objects = %v, err=%v", objects, err)
	}
	f, err := os.Open(archive.callsPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	calls := map[string]bool{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var record toolResultAuditRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			t.Fatal(err)
		}
		calls[record.CallID] = true
	}
	if len(calls) != 2 || !calls["call-1"] || !calls["call-2"] {
		t.Fatalf("call ledger did not preserve both call IDs: %v", calls)
	}
}

func TestSessionPersistsOnlyToolResultMetadataAndReloadsPreview(t *testing.T) {
	baseDir := t.TempDir()
	session := NewSession(baseDir, "main")
	payload := largeArchivedPayload()
	if err := session.AppendMessage(Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", ToolName: "media_search", Content: payload, Image: []byte("pixels")}}}, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}

	history, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(history), "MIDDLE_ONLY_ARCHIVE_SENTINEL") || strings.Contains(string(history), "pixels") {
		t.Fatalf("durable conversation leaked full tool payload: %s", history)
	}
	entries := readSessionEntriesForTest(t, session.path)
	stored := entries[len(entries)-1].ToolResults[0]
	if stored.Content != "" || len(stored.Image) != 0 || stored.SHA256 == "" || stored.ArchiveRef == "" || stored.OriginalBytes == 0 {
		t.Fatalf("durable result is not metadata-only: %+v", stored)
	}

	restarted := NewSession(baseDir, "main")
	messages, _ := restarted.LoadTail(10)
	if len(messages) != 2 || len(messages[1].ToolResults) != 1 {
		t.Fatalf("reloaded messages = %+v", messages)
	}
	preview := messages[1].ToolResults[0]
	if !preview.ContentIsPreview || len(preview.Content) > historicalToolResultPerResultChars || !strings.Contains(preview.Content, "OLDER TOOL RESULT") || strings.Contains(preview.Content, "archive_ref") || len(preview.Image) != 0 {
		t.Fatalf("restart did not load bounded preview: %+v", preview)
	}
	object, err := restarted.archive.Read(preview.ArchiveRef)
	if err != nil {
		t.Fatal(err)
	}
	if object.Content != payload || string(object.Image) != "pixels" {
		t.Fatal("immutable archive did not preserve the full payload")
	}
}

func TestRestartMigratesLegacyLargeResultOutOfConversationJSONL(t *testing.T) {
	baseDir := t.TempDir()
	session := NewSession(baseDir, "legacy")
	payload := largeArchivedPayload()
	legacyEntries := []SessionEntry{
		{Sequence: 1, Role: "assistant", ToolCalls: []NativeToolCall{{ID: "legacy-call", Name: "media_search"}}},
		{Sequence: 2, Role: "user", ToolResults: []ToolResult{{CallID: "legacy-call", ToolName: "media_search", Content: payload}}},
	}
	var raw bytes.Buffer
	for _, entry := range legacyEntries {
		if err := json.NewEncoder(&raw).Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(session.path, raw.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}

	restarted := NewSession(baseDir, "legacy")
	messages, _ := restarted.LoadTail(10)
	if len(messages) != 2 || !strings.Contains(messages[1].ToolResults[0].Content, "OLDER TOOL RESULT") {
		t.Fatalf("legacy restart did not load a bounded preview: %+v", messages)
	}
	history, err := os.ReadFile(restarted.path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(history), "MIDDLE_ONLY_ARCHIVE_SENTINEL") {
		t.Fatal("legacy full payload remained in durable conversation JSONL after successful archival")
	}
	ref, ok := restarted.archive.RefForCall("legacy-call")
	if !ok {
		t.Fatal("legacy migration did not preserve the call audit record")
	}
	object, err := restarted.archive.Read(ref)
	if err != nil || object.Content != payload {
		t.Fatalf("legacy migration did not preserve full audit content: err=%v", err)
	}
}

func TestLargeToolResultStaysFullUntilRetentionExpires(t *testing.T) {
	baseDir := t.TempDir()
	thinker := &Thinker{
		threadID:      "main",
		session:       NewSession(baseDir, "main"),
		toolResultAge: map[string]int{},
	}
	payload := largeArchivedPayload()
	thinker.messages = []Message{
		{Role: "system", Content: "test"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}},
	}
	resultMessage := thinker.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: payload}}})
	thinker.messages = append(thinker.messages, resultMessage)

	first := thinker.prepareToolResultRequest(thinker.messages)
	protocolResult := first[2].ToolResults[0]
	if protocolResult.Content != payload || protocolResult.ContentIsPreview || len(first) != len(thinker.messages) {
		t.Fatal("new large result was not kept in place and fully visible")
	}

	initialEpoch := thinker.promptCacheEpoch
	for call := 1; call < toolResultFullRetentionCalls; call++ {
		thinker.markToolResultsConsumed(thinker.messages)
		request := thinker.prepareToolResultRequest(thinker.messages)
		if request[2].ToolResults[0].Content != payload {
			t.Fatalf("result truncated too early after %d successful calls", call)
		}
		if thinker.promptCacheEpoch != initialEpoch {
			t.Fatalf("cache epoch changed before retention expired: %d", thinker.promptCacheEpoch)
		}
	}
	thinker.markToolResultsConsumed(thinker.messages)
	aged := thinker.prepareToolResultRequest(thinker.messages)
	if strings.Contains(aged[2].ToolResults[0].Content, "MIDDLE_ONLY_ARCHIVE_SENTINEL") || len(aged[2].ToolResults[0].Content) > historicalToolResultPerResultChars || !aged[2].ToolResults[0].ContentIsPreview {
		t.Fatalf("old large result was not bounded: %+v", aged[2].ToolResults[0])
	}
	if thinker.promptCacheEpoch != initialEpoch+1 || thinker.promptCacheResetReason != "tool_result_retention_expired" {
		t.Fatalf("retention pruning did not record an intentional cache epoch: epoch=%d reason=%q", thinker.promptCacheEpoch, thinker.promptCacheResetReason)
	}
}

func TestHistoricalToolResultTotalLimit(t *testing.T) {
	baseDir := t.TempDir()
	thinker := &Thinker{threadID: "main", session: NewSession(baseDir, "main"), toolResultAge: map[string]int{}}
	thinker.messages = []Message{{Role: "system", Content: "test"}}
	for i := 0; i < 50; i++ {
		callID := "call-" + strconv.Itoa(i)
		thinker.messages = append(thinker.messages,
			Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: callID, Name: "media_search"}}},
		)
		result := thinker.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: callID, ToolName: "media_search", Content: largeArchivedPayload() + callID}}})
		thinker.messages = append(thinker.messages, result)
	}
	for i := 0; i < toolResultFullRetentionCalls; i++ {
		thinker.markToolResultsConsumed(thinker.messages)
	}
	request := thinker.prepareToolResultRequest(thinker.messages)
	total := 0
	for _, message := range request {
		for _, result := range message.ToolResults {
			total += len(result.Content)
			if len(result.Content) > historicalToolResultPerResultChars {
				t.Fatalf("per-result limit exceeded: %d", len(result.Content))
			}
		}
	}
	if total > historicalToolResultTotalChars {
		t.Fatalf("historical total = %d, limit = %d", total, historicalToolResultTotalChars)
	}
}

func TestSmallToolResultsDoNotChangeOrCreateArchiveObjects(t *testing.T) {
	baseDir := t.TempDir()
	session := NewSession(baseDir, "main")
	result := ToolResult{CallID: "small-call", ToolName: "pace", Content: "set sleep=30s"}
	if err := session.Append(SessionEntry{Role: "user", ToolResults: []ToolResult{result}}); err != nil {
		t.Fatal(err)
	}
	entries := readSessionEntriesForTest(t, session.path)
	if got := entries[0].ToolResults[0]; got.Content != result.Content || got.SHA256 != "" || got.ContentIsPreview {
		t.Fatalf("small result changed: %+v", got)
	}
	objects, err := filepath.Glob(filepath.Join(baseDir, historyDir, toolResultArchiveDir, toolResultArchiveObjectDir, "sha256", "*", "*.json"))
	if err != nil || len(objects) != 0 {
		t.Fatalf("small result created archive objects: %v err=%v", objects, err)
	}
}

func TestSessionCompactionKeepsArchiveAndMetadataOnlyRecentResult(t *testing.T) {
	baseDir := t.TempDir()
	session := NewSession(baseDir, "main")
	for i := 0; i < 8; i++ {
		if err := session.Append(SessionEntry{Role: "user", Content: "old"}); err != nil {
			t.Fatal(err)
		}
	}
	payload := largeArchivedPayload()
	if err := session.Append(SessionEntry{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-last", Name: "media_search"}}}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "user", ToolResults: []ToolResult{{CallID: "call-last", ToolName: "media_search", Content: payload}}}); err != nil {
		t.Fatal(err)
	}
	ref, ok := session.archive.RefForCall("call-last")
	if !ok {
		t.Fatal("missing audit ledger reference before compaction")
	}
	session.ForceCompact(2, func(string) string { return "summary" })
	entries := readSessionEntriesForTest(t, session.path)
	stored := entries[len(entries)-1].ToolResults[0]
	if stored.Content != "" || stored.ArchiveRef != ref {
		t.Fatalf("compaction did not retain metadata-only result: %+v", stored)
	}
	object, err := session.archive.Read(ref)
	if err != nil || object.Content != payload {
		t.Fatalf("compaction damaged archive: err=%v", err)
	}
	restarted := NewSession(baseDir, "main")
	messages, _ := restarted.LoadTail(10)
	if len(messages) != 2 || !strings.Contains(messages[1].ToolResults[0].Content, "OLDER TOOL RESULT") {
		t.Fatalf("compacted restart did not load preview: %+v", messages)
	}
}

func TestSessionRenameKeepsArchivedCallRetrievableAfterRestart(t *testing.T) {
	baseDir := t.TempDir()
	session := NewSession(baseDir, "before")
	if err := session.Append(SessionEntry{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "rename-call", Name: "media_search"}}}); err != nil {
		t.Fatal(err)
	}
	payload := largeArchivedPayload()
	if err := session.Append(SessionEntry{Role: "user", ToolResults: []ToolResult{{CallID: "rename-call", ToolName: "media_search", Content: payload}}}); err != nil {
		t.Fatal(err)
	}
	if err := session.Rename("after"); err != nil {
		t.Fatal(err)
	}
	restarted := NewSession(baseDir, "after")
	ref, ok := restarted.archive.RefForCall("rename-call")
	if !ok {
		t.Fatal("renamed session lost internal call audit reference")
	}
	object, err := restarted.archive.Read(ref)
	if err != nil || object.Content != payload {
		t.Fatalf("renamed session lost archived payload: err=%v", err)
	}
}

func TestNoAgentVisibleToolResultRetrievalTool(t *testing.T) {
	for _, tool := range NewToolRegistry("").NativeTools(nil, nil) {
		if tool.Name == "tool_result_read" {
			t.Fatal("tool_result_read must not be exposed to agents")
		}
	}
}
