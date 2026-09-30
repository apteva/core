package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func newHistoryTestStore(t *testing.T) (*MemoryStore, *memoryHistory) {
	t.Helper()
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "history"), 0700); err != nil {
		t.Fatal(err)
	}
	m := &MemoryStore{path: filepath.Join(dir, "memory.jsonl"), byID: map[string]int{}}
	h := newMemoryHistory(m, &Config{})
	m.history = h
	return m, h
}

func writeMemoryHistory(t *testing.T, h *memoryHistory, thread string, entries ...SessionEntry) {
	t.Helper()
	var data strings.Builder
	for _, e := range entries {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		data.Write(b)
		data.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(h.dir, thread+".jsonl"), []byte(data.String()), 0600); err != nil {
		t.Fatal(err)
	}
}

func readMemoryBatch(t *testing.T, h *memoryHistory, limit string) *memoryReviewBatch {
	t.Helper()
	r := h.review(context.Background(), map[string]string{"limit": limit})
	if r.IsError {
		t.Fatal(r.Text)
	}
	var body struct {
		Batch *memoryReviewBatch `json:"batch"`
	}
	if err := json.Unmarshal([]byte(r.Text), &body); err != nil {
		t.Fatal(err)
	}
	return body.Batch
}

func commitMemoryBatch(t *testing.T, h *memoryHistory, b *memoryReviewBatch) {
	t.Helper()
	if r := h.review(context.Background(), map[string]string{"action": "commit", "batch_id": b.ID}); r.IsError {
		t.Fatal(r.Text)
	}
}

func TestMemoryHistoryCheckpointReplayFairnessAndRewrite(t *testing.T) {
	m, h := newHistoryTestStore(t)
	var main []SessionEntry
	for i := 1; i <= 123; i++ {
		main = append(main, SessionEntry{Sequence: int64(i), Role: "user", Content: fmt.Sprintf("decision %d", i)})
	}
	writeMemoryHistory(t, h, "main", main...)
	writeMemoryHistory(t, h, "worker", SessionEntry{Role: "user", Content: "worker decision"})
	first := readMemoryBatch(t, h, "50")
	if first.ThreadID != "main" || len(first.Entries) != 50 || first.Entries[0].Source.Sequence != 1 {
		t.Fatalf("first batch: %+v", first)
	}
	if replay := readMemoryBatch(t, h, "2"); replay.ID != first.ID || len(replay.Entries) != 50 {
		t.Fatal("pending batch changed")
	}
	restarted := newMemoryHistory(m, h.config)
	if replay := readMemoryBatch(t, restarted, "2"); replay.ID != first.ID {
		t.Fatal("restart lost pending batch")
	}
	if r := restarted.review(context.Background(), map[string]string{"action": "commit", "batch_id": "wrong"}); !r.IsError {
		t.Fatal("accepted wrong batch")
	}
	commitMemoryBatch(t, restarted, first)
	commitMemoryBatch(t, restarted, first)
	second := readMemoryBatch(t, restarted, "50")
	if second.ThreadID != "worker" {
		t.Fatalf("starved worker: %+v", second)
	}
	commitMemoryBatch(t, restarted, second)
	third := readMemoryBatch(t, restarted, "50")
	if third.ThreadID != "main" || third.Entries[0].Source.Sequence != 51 {
		t.Fatalf("skipped history: %+v", third)
	}
	commitMemoryBatch(t, restarted, third)
	// Compaction/rewrite retains recent entries with the same stable references.
	writeMemoryHistory(t, h, "main", main[90:]...)
	fourth := readMemoryBatch(t, restarted, "50")
	if len(fourth.Entries) != 23 || fourth.Entries[0].Source.Sequence != 101 {
		t.Fatalf("rewrite replayed seen history: %+v", fourth)
	}
	commitMemoryBatch(t, restarted, fourth)
	if b := readMemoryBatch(t, restarted, "50"); b != nil {
		t.Fatalf("unexpected history: %+v", b)
	}
}

func TestMemoryHistoryFailuresDoNotAdvance(t *testing.T) {
	_, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "main", SessionEntry{Role: "user", Content: "original"})
	b := readMemoryBatch(t, h, "50")
	checkpoint := h.path
	// atomicWriteFile creates parents, so use an existing directory as target.
	h.path = t.TempDir()
	if r := h.review(context.Background(), map[string]string{"action": "commit", "batch_id": b.ID}); !r.IsError {
		t.Fatal("expected persistence failure")
	}
	if h.state.Pending == nil || len(h.state.Seen) != 0 {
		t.Fatal("failed commit advanced checkpoint")
	}
	h.path = checkpoint
	commitMemoryBatch(t, h, b)
	if err := os.WriteFile(h.path, []byte("{"), 0600); err != nil {
		t.Fatal(err)
	}
	restarted := newMemoryHistory(h.store, h.config)
	if r := restarted.review(context.Background(), nil); !r.IsError {
		t.Fatal("silently ignored corrupt checkpoint")
	}
}

func TestMemoryHistoryIncompleteAndCorruptLines(t *testing.T) {
	_, h := newHistoryTestStore(t)
	p := filepath.Join(h.dir, "main.jsonl")
	full := "{\"role\":\"user\",\"content\":\"first\"}\n"
	partial := "{\"role\":\"user\",\"content\":\"second\"}"
	if err := os.WriteFile(p, []byte(full+partial), 0600); err != nil {
		t.Fatal(err)
	}
	b := readMemoryBatch(t, h, "50")
	if len(b.Entries) != 1 {
		t.Fatal("consumed unfinished JSONL entry")
	}
	commitMemoryBatch(t, h, b)
	if err := os.WriteFile(p, []byte(full+partial+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	b = readMemoryBatch(t, h, "50")
	if len(b.Entries) != 1 || b.Entries[0].Content != "second" {
		t.Fatalf("lost completed tail: %+v", b)
	}
	commitMemoryBatch(t, h, b)
	if err := os.WriteFile(p, []byte(full+"not json\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if r := h.review(context.Background(), nil); !r.IsError {
		t.Fatal("silently skipped corrupt record")
	}
}

func TestMemoryHistoryScopesSourcesAndIdempotentWrites(t *testing.T) {
	m, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "chat-a", SessionEntry{Role: "user", Content: "Prefers cobalt"})
	b := readMemoryBatch(t, h, "50")
	args := map[string]string{"content": "Prefers cobalt", "tags": "preference", "source_ids": b.Entries[0].Source.EntryID, "scope": ""}
	if r := h.remember(args); r.IsError {
		t.Fatal(r.Text)
	}
	restarted := newMemoryHistory(m, h.config)
	if r := restarted.remember(args); r.IsError {
		t.Fatal(r.Text)
	}
	a := m.Active()
	if len(a) != 1 || a[0].Scope != "chat-a" || len(a[0].Sources) != 1 || a[0].Sources[0].Role != "user" {
		t.Fatalf("bad provenance: %+v", a)
	}
	args["source_ids"] = "fake"
	if r := h.remember(args); !r.IsError {
		t.Fatal("forged evidence accepted")
	}
	shared, err := m.Remember("Shared cobalt", nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if r := h.mutate(map[string]string{"id": shared, "reason": "remove"}, false); !r.IsError {
		t.Fatal("background changed shared memory")
	}
	edit := map[string]string{"old_id": a[0].ID, "content": "Prefers blue", "reason": "correction", "tags": "preference"}
	if r := h.mutate(edit, true); r.IsError {
		t.Fatal(r.Text)
	}
	if r := restarted.mutate(edit, true); r.IsError {
		t.Fatal(r.Text)
	}
	if m.Count() != 2 {
		t.Fatalf("supersede replay duplicated memory: %d", m.Count())
	}
	var updated MemoryRecord
	for _, r := range m.Active() {
		if r.Scope == "chat-a" {
			updated = r
		}
	}
	if len(updated.Sources) != 1 {
		t.Fatal("supersede lost sources")
	}
	drop := map[string]string{"id": updated.ID, "reason": "obsolete"}
	if r := h.mutate(drop, false); r.IsError {
		t.Fatal(r.Text)
	}
	if r := restarted.mutate(drop, false); r.IsError {
		t.Fatal(r.Text)
	}
}

func TestMemoryScopedRecallFiltersBeforeTopK(t *testing.T) {
	m, _ := newHistoryTestStore(t)
	for i := 0; i < 10; i++ {
		if _, err := m.Remember(fmt.Sprintf("cobalt private %d", i), nil, 1, MemoryMetadata{Scope: "chat-b"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := m.Remember("cobalt allowed", nil, .5, MemoryMetadata{Scope: "chat-a"}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Remember("cobalt shared", nil, .5); err != nil {
		t.Fatal(err)
	}
	got := m.recallIndexed([]string{"cobalt"}, 2, "chat-a")
	if len(got) != 2 {
		t.Fatalf("private results crowded out allowed results: %+v", got)
	}
	for _, r := range got {
		if !memoryVisible(r.Record, "chat-a") {
			t.Fatal("cross-thread leak")
		}
	}
	for _, r := range m.searchForThread("cobalt", "chat-a", 10) {
		if r.Scope == "chat-b" {
			t.Fatal("search leaked")
		}
	}
}

func TestHistorySearchTrustedCallerAndSafeProjection(t *testing.T) {
	_, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "chat-a",
		SessionEntry{Role: "system", Content: "cobalt SYSTEM_SECRET"},
		SessionEntry{Role: "user", Content: "The cobalt deployment uses port 6543", Reasoning: "PRIVATE_REASONING"},
		SessionEntry{Role: "_compacted", Summary: "cobalt was selected last month"})
	writeMemoryHistory(t, h, "chat-b", SessionEntry{Role: "user", Content: "cobalt OTHER_CHAT"})
	args := map[string]string{"query": "cobalt"}
	if r := h.search(context.Background(), args); !r.IsError {
		t.Fatal("accepted untrusted identity")
	}
	ctx := withMemoryCaller(context.Background(), "chat-a")
	r := h.search(ctx, args)
	if r.IsError || !strings.Contains(r.Text, "6543") || !strings.Contains(r.Text, "entry_id") {
		t.Fatalf("missing fallback evidence: %s", r.Text)
	}
	for _, secret := range []string{"SYSTEM_SECRET", "PRIVATE_REASONING", "OTHER_CHAT"} {
		if strings.Contains(r.Text, secret) {
			t.Fatalf("leaked %s", secret)
		}
	}
	args["thread_id"] = "chat-b"
	if r := h.search(ctx, args); !r.IsError {
		t.Fatal("accepted spoofed thread")
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	delete(args, "thread_id")
	if r := h.search(ctx, args); !r.IsError {
		t.Fatal("ignored cancellation")
	}
}

func TestMemoryHistoryExclusionsAndSymlinks(t *testing.T) {
	_, h := newHistoryTestStore(t)
	h.config.Threads = []PersistentThread{{ID: "system-worker", System: true}}
	h.config.MemoryPolicy = MemoryPolicy{ExcludeThreads: []string{"private-chat"}}
	for _, id := range []string{"system-worker", "private-chat", "unconscious", "allowed"} {
		writeMemoryHistory(t, h, id, SessionEntry{Role: "user", Content: "cobalt " + id})
	}
	if err := os.Symlink(filepath.Join(h.dir, "allowed.jsonl"), filepath.Join(h.dir, "linked.jsonl")); err != nil {
		t.Fatal(err)
	}
	b := readMemoryBatch(t, h, "50")
	if b.ThreadID != "allowed" {
		t.Fatalf("reviewed excluded thread: %+v", b)
	}
	commitMemoryBatch(t, h, b)
	if b := readMemoryBatch(t, h, "50"); b != nil {
		t.Fatalf("reviewed excluded thread: %+v", b)
	}
	r := h.search(withMemoryCaller(context.Background(), "private-chat"), map[string]string{"query": "cobalt"})
	if strings.Contains(r.Text, "private-chat") {
		t.Fatal("searched excluded history")
	}
	if err := h.scan(context.Background(), "../allowed", func(historyMemoryEntry) bool { return true }); err == nil {
		t.Fatal("accepted traversal")
	}
	link := filepath.Join(t.TempDir(), "history")
	if err := os.Symlink(h.dir, link); err != nil {
		t.Fatal(err)
	}
	h.dir = link
	if r := h.review(context.Background(), nil); !r.IsError {
		t.Fatal("followed history directory symlink")
	}
}

func TestMemoryForgetSourceDurableAndDeniesRelearning(t *testing.T) {
	m, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "chat-a", SessionEntry{Role: "user", Content: "cobalt secret"})
	readMemoryBatch(t, h, "50")
	if r := h.remember(map[string]string{"content": "cobalt secret"}); r.IsError {
		t.Fatal(r.Text)
	}
	// Even an explicitly shared derivative keeps lineage and must be forgotten.
	old := m.Active()[0]
	if _, err := m.Supersede(old.ID, "cobalt derivative", nil, 1, "publish", MemoryMetadata{}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Remember("unrelated shared", nil, 1); err != nil {
		t.Fatal(err)
	}
	if n, err := m.ForgetSource("chat-a", "user request"); err != nil || n != 1 {
		t.Fatalf("forget: %d %v", n, err)
	}
	if m.Count() != 1 {
		t.Fatal("wrong memories removed")
	}
	reloaded := &MemoryStore{path: m.path, byID: map[string]int{}}
	reloaded.load()
	if !reloaded.SourceForgotten("chat-a") || reloaded.Count() != 1 {
		t.Fatal("forget not durable")
	}
	if _, err := reloaded.Remember("relearn", nil, 1, MemoryMetadata{Scope: "chat-a"}); err == nil {
		t.Fatal("relearned forgotten source")
	}
	if r := h.remember(map[string]string{"content": "cobalt again"}); !r.IsError {
		t.Fatal("pending batch relearned forgotten source")
	}
	if b := readMemoryBatch(t, h, "50"); b != nil {
		t.Fatal("forgotten pending batch replayed")
	}
	r := h.search(withMemoryCaller(context.Background(), "chat-a"), map[string]string{"query": "cobalt"})
	if strings.Contains(r.Text, "secret") {
		t.Fatal("forgotten source searched")
	}
	// Recovery from a marker-only journal tail also removes attributable facts.
	m2, _ := newHistoryTestStore(t)
	if _, err := m2.Remember("private", nil, 1, MemoryMetadata{Scope: "chat-a"}); err != nil {
		t.Fatal(err)
	}
	if err := m2.append(MemoryRecord{ID: newULID(), TS: time.Now(), ForgottenThread: "chat-a"}); err != nil {
		t.Fatal(err)
	}
	if m2.Count() != 0 {
		t.Fatal("marker alone did not enforce forgetting")
	}
	if err := m2.Reset(); err != nil {
		t.Fatal(err)
	}
	if !m2.SourceForgotten("chat-a") {
		t.Fatal("memory reset revoked forgetting")
	}
}

func TestMemoryHistoryConcurrentReadsSharePendingBatch(t *testing.T) {
	_, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "main", SessionEntry{Role: "user", Content: "cobalt"})
	var wg sync.WaitGroup
	results := make(chan string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); results <- h.review(context.Background(), nil).Text }()
	}
	wg.Wait()
	close(results)
	first := ""
	for r := range results {
		if first == "" {
			first = r
		}
		if r != first {
			t.Fatal("concurrent readers diverged")
		}
	}
}

func TestHistorySearchRuntimeDispatchAndToolBoundaries(t *testing.T) {
	m, h := newHistoryTestStore(t)
	writeMemoryHistory(t, h, "chat-a", SessionEntry{Role: "user", Content: "cobalt own"})
	writeMemoryHistory(t, h, "chat-b", SessionEntry{Role: "user", Content: "cobalt hidden"})
	registry := NewToolRegistry("")
	registerSystemTools(registry, m, h.config)
	allow := map[string]bool{}
	addManagedThreadBuiltins(allow, false, false)
	if !allow["history_search"] {
		t.Fatal("child missing history fallback")
	}
	for _, tool := range registry.NativeTools(allow, nil, false) {
		if strings.HasPrefix(tool.Name, "memory_") || tool.Name == "review_history" {
			t.Fatal("ordinary child received system memory tools")
		}
	}
	bus := NewEventBus()
	th := &Thinker{bus: bus, sub: bus.Subscribe("chat-a", 100), registry: registry, quit: make(chan struct{}), threadID: "chat-a", toolAllowlist: allow}
	executeTool(th, toolCall{Name: "history_search", NativeID: "search", Args: map[string]string{"query": "cobalt", "thread_id": "chat-a"}})
	select {
	case ev := <-th.sub.C:
		if ev.ToolResult == nil || !strings.Contains(ev.ToolResult.Content, "cobalt own") || strings.Contains(ev.ToolResult.Content, "cobalt hidden") {
			t.Fatalf("incorrect runtime caller identity: %+v", ev.ToolResult)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("tool did not complete")
	}
}

func TestHistorySearchFindsMiddleOfLongMessagesWithBoundedResults(t *testing.T) {
	_, h := newHistoryTestStore(t)
	var entries []SessionEntry
	for i := 0; i < 12; i++ {
		entries = append(entries, SessionEntry{Sequence: int64(i + 1), Role: "user", Content: strings.Repeat("prefix ", 2000) + "needle-port-6543 " + strings.Repeat("suffix ", 2000)})
	}
	writeMemoryHistory(t, h, "main", entries...)
	r := h.search(withMemoryCaller(context.Background(), "main"), map[string]string{"query": "needle-port-6543"})
	if r.IsError {
		t.Fatal(r.Text)
	}
	var body struct {
		Matches []historyMemoryEntry `json:"matches"`
	}
	if err := json.Unmarshal([]byte(r.Text), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Matches) != 5 {
		t.Fatalf("expected top five matches, got %d", len(body.Matches))
	}
	for _, e := range body.Matches {
		if !strings.Contains(e.Content, "needle-port-6543") || len(e.Content) > 8192 || !e.Truncated {
			t.Fatal("unbounded or missing search snippet")
		}
	}
}

func TestMemoryPublishedScopeOnlyRecordRetainsForgetLineage(t *testing.T) {
	m, _ := newHistoryTestStore(t)
	id, err := m.Remember("scoped operator fact", nil, 1, MemoryMetadata{Scope: "chat-a"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = m.Supersede(id, "published operator fact", nil, 1, "publish", MemoryMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	if n, err := m.ForgetSource("chat-a", "requested"); n != 1 || err != nil {
		t.Fatalf("published scope lineage lost: %d %v", n, err)
	}
	if _, err := m.Supersede(id, "revived", nil, 1, "republish", MemoryMetadata{}); err == nil {
		t.Fatal("revived a forgotten source by clearing scope")
	}
}
