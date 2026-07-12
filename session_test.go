package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Pins the regression where sanitizeToolPairs preserved orphan tool
// results carrying an image. The "screenshots must survive" exception
// produced wire payloads with unpaired tool_call_ids that strict
// providers (Moonshot via opencode-go, Anthropic) rejected with HTTP
// 400. Old screenshots are already evicted to text placeholders by
// thinker.go's image-cap, so dropping orphan-with-image entirely is
// safe and matches the protocol.
//
// If anyone reintroduces the `tr.Image != nil` exception in
// sanitizeToolPairs, these tests fail and the Patreon login flow
// (and any other long-running multi-turn image-tool flow) starts
// 400ing on opencode-go again.

func TestSanitize_OrphanToolResult_NoImage_Dropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{{CallID: "screenshot_tool:1", Content: "stale"}}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected orphan tool_result to be dropped, got %d msgs", len(out))
	}
}

// The regression. Before the fix this kept the orphan because
// `tr.Image != nil`, leaving a tool_call_id with no matching tool_use
// in the wire payload. Moonshot 400s on this; Fireworks accepts it.
func TestSanitize_OrphanToolResult_WithImage_Dropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:8", Content: "stale screenshot", Image: []byte("fake-png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected orphan tool_result with image to be dropped, got %d msgs (regression: image-bearing orphan being preserved again)", len(out))
	}
}

func TestSanitize_OrphanToolUse_NoResult_Dropped(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:1", Name: "screenshot_tool"}}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 1 {
		t.Fatalf("expected assistant message to be kept, got %d", len(out))
	}
	if len(out[0].ToolCalls) != 0 {
		t.Fatalf("expected orphan tool_call to be dropped, got %d", len(out[0].ToolCalls))
	}
}

func TestSanitize_OrphanToolUse_DropsProviderState(t *testing.T) {
	in := []Message{{
		Role:      "assistant",
		Content:   "working",
		ToolCalls: []NativeToolCall{{ID: "orphan", Name: "lookup"}},
		ProviderState: &ProviderResponseState{
			Provider: openAIResponsesStateProvider,
			Items: []json.RawMessage{
				json.RawMessage(`{"id":"rs_123","type":"reasoning","encrypted_content":"opaque"}`),
				json.RawMessage(`{"id":"fc_123","type":"function_call","call_id":"orphan","name":"lookup","arguments":"{}"}`),
			},
		},
	}}

	out := sanitizeToolPairs(in)
	if len(out) != 1 {
		t.Fatalf("expected assistant message to remain, got %d", len(out))
	}
	if len(out[0].ToolCalls) != 0 {
		t.Fatalf("ToolCalls = %#v", out[0].ToolCalls)
	}
	if out[0].ProviderState != nil {
		t.Fatalf("ProviderState survived changed calls: %#v", out[0].ProviderState)
	}
}

// Symmetric regression. Before the fix the assistant tool_use was
// kept when its only matching ToolResult had an image — but if that
// image-bearing result was itself dropped (which it now is), there
// was no real match. Now both must be dropped together.
func TestSanitize_OrphanToolUse_KeptByImageException_Dropped(t *testing.T) {
	in := []Message{
		// Assistant emits a tool_call.
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:8", Name: "screenshot_tool"}}},
		// Its result is an orphan-with-image (the matching tool_use is
		// elsewhere in the original conversation but rolled off the
		// LoadTail window). The previous code path: image-exception
		// kept the result → toolResultIDs included :8 → assistant
		// tool_call also kept → both wrong-shape on wire.
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:99", Image: []byte("fake-png")}, // orphan
		}},
	}
	out := sanitizeToolPairs(in)
	// Both orphans should be dropped: assistant turn loses its tool_call,
	// user turn loses its only tool_result and is therefore omitted.
	for _, m := range out {
		for _, tr := range m.ToolResults {
			if tr.CallID == "screenshot_tool:99" {
				t.Fatalf("orphan image tool_result :99 leaked through")
			}
		}
		for _, tc := range m.ToolCalls {
			if tc.ID == "screenshot_tool:8" && len(out) > 1 {
				// It's only valid to keep :8 if its result is also kept.
				found := false
				for _, m2 := range out {
					for _, tr := range m2.ToolResults {
						if tr.CallID == "screenshot_tool:8" {
							found = true
						}
					}
				}
				if !found {
					t.Fatalf("orphan tool_call :8 kept without matching result")
				}
			}
		}
	}
}

func TestSanitize_PairedToolsWithImage_Kept(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:1", Name: "screenshot_tool"}}},
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:1", Content: "ok", Image: []byte("png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 2 {
		t.Fatalf("expected both paired messages kept, got %d", len(out))
	}
	if len(out[0].ToolCalls) != 1 || out[0].ToolCalls[0].ID != "screenshot_tool:1" {
		t.Fatalf("assistant tool_call dropped from a valid pair")
	}
	if len(out[1].ToolResults) != 1 || out[1].ToolResults[0].CallID != "screenshot_tool:1" {
		t.Fatalf("user tool_result dropped from a valid pair")
	}
	if out[1].ToolResults[0].Image == nil {
		t.Fatalf("image stripped from valid tool_result")
	}
}

// In-flight async tool_calls dispatched as goroutines
// are tracked in pendingTools. The sanitizer must keep their
// assistant turn so the iteration-barrier placeholder can pair with
// them later — without this guard, sanitize would strip the call
// before its result lands.
func TestSanitize_InFlightToolUse_KeptViaPending(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:7", Name: "screenshot_tool"}}},
	}
	pending := map[string]bool{"screenshot_tool:7": true}
	out := sanitizeToolPairs(in, pending)
	if len(out) != 1 || len(out[0].ToolCalls) != 1 {
		t.Fatalf("in-flight tool_use stripped despite pending: out=%+v", out)
	}
}

func TestSanitize_AllOrphanResults_MessageDropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "a:1", Content: "x"},
			{CallID: "b:2", Content: "y", Image: []byte("png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected message with all-orphan results to be dropped, got %d", len(out))
	}
}

// Mixed message: one tool_result paired, one orphan. The orphan is
// dropped; the message survives with just the paired result.
func TestSanitize_MixedToolResults_OnlyOrphanDropped(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "a:1", Name: "x"}}},
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "a:1", Content: "ok"},                               // paired → keep
			{CallID: "ghost:99", Content: "stale", Image: []byte("png")}, // orphan → drop
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages kept, got %d", len(out))
	}
	if len(out[1].ToolResults) != 1 || out[1].ToolResults[0].CallID != "a:1" {
		t.Fatalf("expected only paired tool_result kept, got %+v", out[1].ToolResults)
	}
}

// End-to-end referential-integrity check: post-sanitize, every
// tool_result.CallID points to an assistant tool_calls[].id earlier
// in the slice, and every assistant tool_call has a tool_result later
// (or is in pending). This is the property Moonshot enforces and the
// reason the orphan-with-image rule had to die.
func TestSanitize_PostconditionReferentialIntegrity(t *testing.T) {
	// Simulates a LoadTail window where a screenshot tool_result for
	// screenshot_tool:8 made the cut but its assistant tool_call did not.
	in := []Message{
		// Window starts mid-conversation; the assistant turn for :8 is
		// outside the slice (rolled off by LoadTail). This is exactly
		// the shape that triggered the Patreon-login 400.
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:8", Content: "stale", Image: []byte("png")},
		}},
		// Healthy pair inside the window.
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:9", Name: "screenshot_tool"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "screenshot_tool:9", Content: "ok"}}},
		// Orphan tool_call with no result.
		// not in pendingTools because the run died).
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:10", Name: "screenshot_tool"}}},
	}
	out := sanitizeToolPairs(in)

	// Build id sets from the sanitized output.
	uses := map[string]int{} // id → index of assistant turn
	for i, m := range out {
		for _, tc := range m.ToolCalls {
			uses[tc.ID] = i
		}
	}
	for i, m := range out {
		for _, tr := range m.ToolResults {
			useIdx, ok := uses[tr.CallID]
			if !ok {
				t.Errorf("post-sanitize: tool_result %s has no preceding tool_use (Moonshot would 400)", tr.CallID)
			} else if useIdx >= i {
				t.Errorf("post-sanitize: tool_result %s appears at msg[%d] but its tool_use is at msg[%d] (must be earlier)", tr.CallID, i, useIdx)
			}
		}
	}
	for _, m := range out {
		for _, tc := range m.ToolCalls {
			matched := false
			for _, m2 := range out {
				for _, tr := range m2.ToolResults {
					if tr.CallID == tc.ID {
						matched = true
					}
				}
			}
			if !matched {
				t.Errorf("post-sanitize: tool_use %s has no matching tool_result (Moonshot would 400)", tc.ID)
			}
		}
	}
}

func TestSessionCompact_RewritesToSummaryAndRecentTail(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	for i := 0; i < compactThreshold+1; i++ {
		session.Append(SessionEntry{
			Role:    "user",
			Content: fmt.Sprintf("message-%03d", i),
		})
	}
	if !session.NeedsCompaction() {
		t.Fatal("session should need compaction after threshold+1 appends")
	}

	session.Compact(func(text string) string {
		if !strings.Contains(text, "message-000") {
			t.Fatalf("summary input missing early history: %q", text)
		}
		return "custom compact summary"
	})

	entries := readSessionEntriesForTest(t, session.path)
	wantCount := compactKeepRecent + 1
	if len(entries) != wantCount {
		t.Fatalf("compacted entry count = %d, want %d", len(entries), wantCount)
	}
	if session.Count() != wantCount {
		t.Fatalf("session count = %d, want %d", session.Count(), wantCount)
	}
	if entries[0].Role != "_compacted" {
		t.Fatalf("first entry role = %q, want _compacted", entries[0].Role)
	}
	if entries[0].Summary != "custom compact summary" {
		t.Fatalf("summary = %q", entries[0].Summary)
	}
	wantOrig := compactThreshold + 1 - compactKeepRecent
	if entries[0].OrigCount != wantOrig {
		t.Fatalf("original_count = %d, want %d", entries[0].OrigCount, wantOrig)
	}
	if entries[1].Content != fmt.Sprintf("message-%03d", wantOrig) {
		t.Fatalf("first retained content = %q, want message-%03d", entries[1].Content, wantOrig)
	}
	if entries[len(entries)-1].Content != fmt.Sprintf("message-%03d", compactThreshold) {
		t.Fatalf("last retained content = %q, want message-%03d", entries[len(entries)-1].Content, compactThreshold)
	}
	if session.NeedsCompaction() {
		t.Fatal("session should not still need compaction after rewrite")
	}
}

func TestSessionCompact_CallbackCanReadSessionStateWithoutDeadlock(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	for i := 0; i < compactThreshold+1; i++ {
		session.Append(SessionEntry{
			Role:    "assistant",
			Content: fmt.Sprintf("message-%03d", i),
		})
	}

	done := make(chan int, 1)
	go func() {
		callbackCount := 0
		session.Compact(func(text string) string {
			callbackCount = session.Count()
			return fmt.Sprintf("summary saw count %d", callbackCount)
		})
		done <- callbackCount
	}()

	select {
	case callbackCount := <-done:
		if callbackCount != compactThreshold+1 {
			t.Fatalf("callback session count = %d, want %d", callbackCount, compactThreshold+1)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Session.Compact deadlocked when callback read session state")
	}

	if session.Count() != compactKeepRecent+1 {
		t.Fatalf("session count after compaction = %d, want %d", session.Count(), compactKeepRecent+1)
	}
}

func TestThinkerCompactSessionIfNeeded_DoesNotDeadlock(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	for i := 0; i < compactThreshold+1; i++ {
		session.Append(SessionEntry{
			Role:    "assistant",
			Content: fmt.Sprintf("message-%03d", i),
		})
	}
	thinker := &Thinker{
		session:  session,
		threadID: "main",
	}

	done := make(chan struct{}, 1)
	go func() {
		thinker.compactSessionIfNeeded()
		done <- struct{}{}
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Thinker.compactSessionIfNeeded deadlocked")
	}

	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) != compactKeepRecent+1 {
		t.Fatalf("compacted entry count = %d, want %d", len(entries), compactKeepRecent+1)
	}
	if entries[0].Role != "_compacted" {
		t.Fatalf("first entry role = %q, want _compacted", entries[0].Role)
	}
	if !strings.Contains(entries[0].Summary, fmt.Sprintf("Summary of %d earlier messages", compactThreshold+1)) {
		t.Fatalf("thinker summary missing pre-compaction count: %q", entries[0].Summary)
	}
	if !strings.Contains(entries[0].Summary, "message-000") {
		t.Fatalf("thinker summary missing early history: %q", entries[0].Summary)
	}
}

func TestThinkerPersistentCompactionUsesLLMAndRemovesOldToolPayloads(t *testing.T) {
	session := NewSession(t.TempDir(), "semantic-history")
	if err := session.Append(SessionEntry{
		Role:      "assistant",
		ToolCalls: []NativeToolCall{{ID: "old-call", Name: "media_search", Args: map[string]string{"query": "critical lead"}}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{
		Role:        "user",
		ToolResults: []ToolResult{{CallID: "old-call", Content: "IMPORTANT_RESULT_SENTINEL"}},
	}); err != nil {
		t.Fatal(err)
	}
	for i := 2; i <= compactThreshold; i++ {
		if err := session.Append(SessionEntry{Role: "user", Content: fmt.Sprintf("message-%03d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	provider := &testCompactionProvider{response: "# Objective\n- Preserve IMPORTANT_RESULT_SENTINEL.\n## Open Tasks\n- Continue."}
	thinker := &Thinker{provider: provider, session: session, threadID: "main", quit: make(chan struct{})}
	thinker.compactSessionIfNeeded()

	if provider.calls != 1 {
		t.Fatalf("compaction provider calls = %d, want 1", provider.calls)
	}
	if len(provider.prompts) != 1 || !strings.Contains(provider.prompts[0][1].Content, "IMPORTANT_RESULT_SENTINEL") {
		t.Fatal("LLM compaction input omitted old tool result")
	}
	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) != compactKeepRecent+1 || entries[0].Role != "_compacted" {
		t.Fatalf("compacted entries = %+v", entries)
	}
	if !strings.Contains(entries[0].Summary, "IMPORTANT_RESULT_SENTINEL") {
		t.Fatalf("semantic summary = %q", entries[0].Summary)
	}
	for _, entry := range entries[1:] {
		if len(entry.ToolCalls) != 0 || len(entry.ToolResults) != 0 {
			t.Fatal("old raw tool call/result survived outside compacted summary")
		}
	}
}

func readSessionEntriesForTest(t *testing.T, path string) []SessionEntry {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read session file: %v", err)
	}
	var entries []SessionEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var entry SessionEntry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("decode session entry %q: %v", line, err)
		}
		entries = append(entries, entry)
	}
	return entries
}

func TestNewSessionKeepsUnsafeIDInsideHistoryDirectory(t *testing.T) {
	base := t.TempDir()
	s := NewSession(base, "../../outside")
	history := filepath.Join(base, historyDir)
	rel, err := filepath.Rel(history, s.path)
	if err != nil {
		t.Fatal(err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		t.Fatalf("session escaped history directory: %s", s.path)
	}
}

func TestSessionLoadAfterUsesMonotonicCursor(t *testing.T) {
	session := NewSession(t.TempDir(), "benchmark-history")
	if err := session.Append(SessionEntry{Role: "user", Content: "brief"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "assistant", Content: "working", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "crm_contacts_get"}}}); err != nil {
		t.Fatal(err)
	}
	first, cursor, err := session.LoadAfter(0, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || cursor != 2 || first[0].Sequence != 1 || first[1].Sequence != 2 {
		t.Fatalf("first history page = %#v, cursor=%d", first, cursor)
	}
	if err := session.Append(SessionEntry{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: `{"found":true}`}}}); err != nil {
		t.Fatal(err)
	}
	second, next, err := session.LoadAfter(cursor, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(second) != 1 || second[0].Sequence != 3 || next != 3 {
		t.Fatalf("second history page = %#v, cursor=%d", second, next)
	}
}

func TestSessionPersistsProviderResponseState(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "provider-state")
	reasoning := json.RawMessage(`{"id":"rs_123","type":"reasoning","encrypted_content":"opaque-state"}`)
	functionCall := json.RawMessage(`{"id":"fc_123","type":"function_call","status":"completed","call_id":"call_123","name":"lookup","arguments":"{}"}`)
	want := Message{
		Role:      "assistant",
		ToolCalls: []NativeToolCall{{ID: "call_123", OutputItemID: "fc_123", Status: "completed", Name: "lookup"}},
		ProviderState: &ProviderResponseState{
			Provider: openAIResponsesStateProvider,
			Items:    []json.RawMessage{reasoning, functionCall},
		},
	}
	if err := session.AppendMessage(want, 7, TokenUsage{}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "call_123", Content: "ok"}}}, 8, TokenUsage{}); err != nil {
		t.Fatalf("AppendMessage result: %v", err)
	}

	loaded, _ := NewSession(dir, "provider-state").LoadTail(10)
	if len(loaded) != 2 {
		t.Fatalf("loaded len = %d, want 2", len(loaded))
	}
	got := loaded[0]
	if got.ProviderState == nil || got.ProviderState.Provider != openAIResponsesStateProvider {
		t.Fatalf("ProviderState = %#v", got.ProviderState)
	}
	if len(got.ProviderState.Items) != 2 || !jsonEqual(got.ProviderState.Items[0], reasoning) || !jsonEqual(got.ProviderState.Items[1], functionCall) {
		t.Fatalf("provider items = %#v", got.ProviderState.Items)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].OutputItemID != "fc_123" || got.ToolCalls[0].Status != "completed" {
		t.Fatalf("ToolCalls = %#v", got.ToolCalls)
	}
}

func TestSessionLoadTailDoesNotReloadComputerScreenshots(t *testing.T) {
	session := NewSession(t.TempDir(), "computer-history")
	if err := session.AppendMessage(Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-screen", Name: "computer_computer_use"}}}, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-screen", Content: "screen metadata", Image: []byte("stale pixels")}}}, 2, TokenUsage{}); err != nil {
		t.Fatal(err)
	}

	messages, _ := session.LoadTail(10)
	if len(messages) != 2 || len(messages[1].ToolResults) != 1 {
		t.Fatalf("loaded messages = %+v", messages)
	}
	if messages[1].ToolResults[0].Image != nil {
		t.Fatal("stale computer screenshot was reloaded")
	}
	if messages[1].ToolResults[0].Content != evictedScreenshotPlaceholder {
		t.Fatalf("loaded result = %q, want screenshot placeholder", messages[1].ToolResults[0].Content)
	}
}

func TestSessionForceCompactStripsRecentComputerScreenshots(t *testing.T) {
	session := NewSession(t.TempDir(), "computer-compaction")
	for i := 0; i < 5; i++ {
		if err := session.Append(SessionEntry{Role: "user", Content: fmt.Sprintf("old-%d", i)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := session.AppendMessage(Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-screen", Name: "computer_computer_use"}}}, 6, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := session.AppendMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-screen", Content: "screen metadata", Image: []byte("pixels")}}}, 7, TokenUsage{}); err != nil {
		t.Fatal(err)
	}

	session.ForceCompact(2, func(string) string { return "summary" })
	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) != 3 || len(entries[2].ToolResults) != 1 {
		t.Fatalf("compacted entries = %+v", entries)
	}
	if entries[2].ToolResults[0].Image != nil {
		t.Fatal("recent computer screenshot remained in compacted session")
	}
}
