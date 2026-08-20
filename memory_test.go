package core

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ---- cosine + format helpers (unchanged from v1) ---------------------

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float64{1, 2, 3}
	if sim := cosineSimilarity(a, a); math.Abs(sim-1.0) > 1e-9 {
		t.Errorf("identical vectors should have similarity 1.0, got %f", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}
	if sim := cosineSimilarity(a, b); math.Abs(sim) > 1e-9 {
		t.Errorf("orthogonal vectors should have similarity 0, got %f", sim)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{-1, -2, -3}
	if sim := cosineSimilarity(a, b); math.Abs(sim-(-1.0)) > 1e-9 {
		t.Errorf("opposite vectors should have similarity -1.0, got %f", sim)
	}
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	if sim := cosineSimilarity(a, b); sim != 0 {
		t.Errorf("different length vectors should give 0, got %f", sim)
	}
}

func TestFormatAge(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{30 * time.Second, "30s"},
		{5 * time.Minute, "5m"},
		{2 * time.Hour, "2h"},
		{72 * time.Hour, "3d"},
	}
	for _, c := range cases {
		if got := formatAge(c.d); got != c.want {
			t.Errorf("formatAge(%v) = %q, want %q", c.d, got, c.want)
		}
	}
}

// ---- ULID-shape sanity check ------------------------------------------

func TestNewULID_UniqueAndOrdered(t *testing.T) {
	a := newULID()
	b := newULID()
	if a == b {
		t.Errorf("two consecutive newULID calls returned same id: %q", a)
	}
	if len(a) != 32 {
		t.Errorf("ulid length = %d, want 32", len(a))
	}
	// Hex-encoded timestamps mean lexicographic order matches time order.
	if a >= b {
		t.Errorf("expected b > a (later timestamp), got a=%s b=%s", a, b)
	}
}

func TestDetectEmbeddingBackend_OllamaDefaults(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	t.Setenv("OLLAMA_EMBED_MODEL", "")
	t.Setenv("OLLAMA_EMBED_DIM", "")

	backend := detectEmbeddingBackend()
	if backend == nil {
		t.Fatal("expected ollama embedding backend")
	}
	if backend.Source != "ollama" {
		t.Fatalf("source = %q, want ollama", backend.Source)
	}
	if backend.URL != "http://127.0.0.1:11434/api/embeddings" {
		t.Fatalf("url = %q", backend.URL)
	}
	if backend.Model != "nomic-embed-text" {
		t.Fatalf("model = %q, want nomic-embed-text", backend.Model)
	}
	if backend.Dim != 768 {
		t.Fatalf("dim = %d, want 768", backend.Dim)
	}
}

func TestDetectEmbeddingBackend_OllamaEmbedOverrides(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434/")
	t.Setenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
	t.Setenv("OLLAMA_EMBED_DIM", "1024")

	backend := detectEmbeddingBackend()
	if backend == nil {
		t.Fatal("expected ollama embedding backend")
	}
	if backend.URL != "http://127.0.0.1:11434/api/embeddings" {
		t.Fatalf("url = %q", backend.URL)
	}
	if backend.Model != "qwen3-embedding:0.6b" {
		t.Fatalf("model = %q, want qwen3-embedding:0.6b", backend.Model)
	}
	if backend.Dim != 1024 {
		t.Fatalf("dim = %d, want 1024", backend.Dim)
	}
}

func TestDetectEmbeddingBackend_OllamaInvalidDimFallsBack(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	t.Setenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
	t.Setenv("OLLAMA_EMBED_DIM", "not-a-number")

	backend := detectEmbeddingBackend()
	if backend == nil {
		t.Fatal("expected ollama embedding backend")
	}
	if backend.Dim != 768 {
		t.Fatalf("dim = %d, want fallback 768", backend.Dim)
	}
}

// ---- write path: remember / supersede / drop --------------------------

// newOfflineStore returns a MemoryStore with no embedding backend, a
// temp memory.jsonl, and an empty in-memory state. All writes still
// hit disk via Remember/Supersede/Drop; recall falls back to lexical.
func newOfflineStore(t *testing.T) *MemoryStore {
	t.Helper()
	dir := t.TempDir()
	prev, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	return &MemoryStore{
		path: memoryFile,
		byID: map[string]int{},
	}
}

func TestRemember_AppendsToDisk(t *testing.T) {
	ms := newOfflineStore(t)

	id, err := ms.Remember("user prefers terse replies", []string{"preference"}, 0.9)
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	if id == "" {
		t.Fatal("Remember returned empty id")
	}

	if ms.Count() != 1 {
		t.Errorf("active count = %d, want 1", ms.Count())
	}

	// Verify the line is on disk and parses.
	data, err := os.ReadFile(memoryFile)
	if err != nil {
		t.Fatal(err)
	}
	var rec MemoryRecord
	if err := json.Unmarshal(data[:strings.Index(string(data), "\n")], &rec); err != nil {
		t.Fatal(err)
	}
	if rec.ID != id || rec.Content != "user prefers terse replies" || rec.Weight != 0.9 {
		t.Errorf("disk record mismatch: %+v", rec)
	}
}

func TestMemoryActiveIndexTracksRememberDropAndSupersede(t *testing.T) {
	ms := &MemoryStore{path: filepath.Join(t.TempDir(), "memory.jsonl"), byID: map[string]int{}, active: map[string]MemoryRecord{}}
	ids := make([]string, 100)
	for i := range ids {
		id, err := ms.Remember(fmt.Sprintf("memory %d", i), []string{"test"}, 0.7)
		if err != nil {
			t.Fatal(err)
		}
		ids[i] = id
	}
	for _, id := range ids[:50] {
		if err := ms.Drop(id, "obsolete"); err != nil {
			t.Fatal(err)
		}
	}
	if got := ms.Count(); got != 50 {
		t.Fatalf("active count after drops = %d, want 50", got)
	}
	newID, err := ms.Supersede(ids[50], "replacement", []string{"test"}, 0.8, "updated")
	if err != nil {
		t.Fatal(err)
	}
	if got := ms.Count(); got != 50 {
		t.Fatalf("active count after supersede = %d, want 50", got)
	}
	active := ms.Active()
	foundReplacement := false
	for _, record := range active {
		if record.ID == ids[50] {
			t.Fatal("superseded record remained active")
		}
		if record.ID == newID {
			foundReplacement = true
		}
	}
	if !foundReplacement {
		t.Fatal("replacement missing from active index")
	}
	info, err := os.Stat(ms.path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("memory journal mode = %o, want 600", got)
	}
}

func TestMemoryBatchAppendFailureDoesNotMutateIndexes(t *testing.T) {
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	ms := &MemoryStore{path: filepath.Join(blocker, "memory.jsonl"), byID: map[string]int{}, active: map[string]MemoryRecord{}}
	err := ms.appendRecords(
		MemoryRecord{ID: "one", Content: "first"},
		MemoryRecord{ID: "two", Tombstone: true, IDTarget: "one"},
	)
	if err == nil {
		t.Fatal("expected batch append failure")
	}
	if len(ms.records) != 0 || len(ms.byID) != 0 || ms.Count() != 0 {
		t.Fatal("failed batch append mutated in-memory indexes")
	}
}

func TestSupersede_OldHidden_NewActive(t *testing.T) {
	ms := newOfflineStore(t)

	oldID, _ := ms.Remember("user prefers verbose replies", []string{"preference"}, 0.8)
	newID, err := ms.Supersede(oldID, "user prefers terse replies", []string{"preference"}, 0.9, "explicit correction in chat")
	if err != nil {
		t.Fatalf("Supersede: %v", err)
	}

	active := ms.Active()
	if len(active) != 1 {
		t.Fatalf("active count = %d, want 1 (only the new memory)", len(active))
	}
	if active[0].ID != newID {
		t.Errorf("active record id = %q, want new id %q", active[0].ID, newID)
	}
	if active[0].Supersedes != oldID {
		t.Errorf("Supersedes link = %q, want %q", active[0].Supersedes, oldID)
	}

	// All records are still on disk (3: old, new, tombstone).
	all := ms.All()
	if len(all) != 3 {
		t.Errorf("All() = %d, want 3 (old + new + tombstone)", len(all))
	}
}

func TestDrop_TombstonesRecord_HiddenFromActive(t *testing.T) {
	ms := newOfflineStore(t)
	id, _ := ms.Remember("currently typing a long message", []string{"ephemeral"}, 0.3)
	if ms.Count() != 1 {
		t.Fatalf("setup count = %d, want 1", ms.Count())
	}
	if err := ms.Drop(id, "single-session ephemera"); err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if ms.Count() != 0 {
		t.Errorf("active count after drop = %d, want 0", ms.Count())
	}
	// Tombstone record is on disk for audit.
	all := ms.All()
	if len(all) != 2 {
		t.Errorf("All() = %d, want 2 (memory + tombstone)", len(all))
	}
	var sawTomb bool
	for _, r := range all {
		if r.Tombstone && r.IDTarget == id && r.Reason == "single-session ephemera" {
			sawTomb = true
		}
	}
	if !sawTomb {
		t.Error("tombstone record not found on disk")
	}
}

func TestDrop_RequiresReason(t *testing.T) {
	ms := newOfflineStore(t)
	id, _ := ms.Remember("x", nil, 0.5)
	if err := ms.Drop(id, ""); err == nil {
		t.Error("Drop with empty reason should error")
	}
	if err := ms.Drop("", "foo"); err == nil {
		t.Error("Drop with empty id should error")
	}
}

func TestSupersede_RejectsUnknownID(t *testing.T) {
	ms := newOfflineStore(t)
	if _, err := ms.Supersede("not-an-id", "new", nil, 0.5, "test"); err == nil {
		t.Error("Supersede with unknown id should error")
	}
}

// ---- RememberWithID + HasID (deterministic-id path) -------------------

func TestRememberWithID_InsertsWithSuppliedID(t *testing.T) {
	ms := newOfflineStore(t)
	id, err := ms.RememberWithID("external_abc_0", "playbook body", []string{"procedure", "storage:upload"}, 0.9)
	if err != nil {
		t.Fatalf("RememberWithID: %v", err)
	}
	if id != "external_abc_0" {
		t.Errorf("returned id = %q, want external_abc_0", id)
	}
	if !ms.HasID("external_abc_0") {
		t.Error("HasID should be true after RememberWithID")
	}
	active := ms.Active()
	if len(active) != 1 || active[0].ID != "external_abc_0" {
		t.Errorf("Active record mismatch: %+v", active)
	}
}

func TestUpsertTargetID_FollowsActiveReplacementChain(t *testing.T) {
	ms := newOfflineStore(t)
	original, err := ms.RememberWithID("external_abc_0", "first playbook", []string{"procedure"}, 0.8)
	if err != nil {
		t.Fatalf("RememberWithID: %v", err)
	}
	second, err := ms.Supersede(original, "second playbook", []string{"procedure"}, 0.8, "changed")
	if err != nil {
		t.Fatalf("Supersede second: %v", err)
	}
	target, ok := ms.UpsertTargetID(original)
	if !ok {
		t.Fatal("expected deterministic id to resolve")
	}
	if target != second {
		t.Fatalf("upsert target = %q, want active replacement %q", target, second)
	}
	third, err := ms.Supersede(target, "third playbook", []string{"procedure"}, 0.8, "changed again")
	if err != nil {
		t.Fatalf("Supersede third: %v", err)
	}
	active := ms.Active()
	if len(active) != 1 || active[0].ID != third || active[0].Content != "third playbook" {
		t.Fatalf("active records = %+v, want only latest replacement %s", active, third)
	}
}

func TestRememberWithID_RejectsEmptyID(t *testing.T) {
	ms := newOfflineStore(t)
	if _, err := ms.RememberWithID("", "x", nil, 0.5); err == nil {
		t.Error("empty id should error")
	}
	if _, err := ms.RememberWithID("   ", "x", nil, 0.5); err == nil {
		t.Error("whitespace-only id should error")
	}
}

func TestRememberWithID_RejectsDuplicateID(t *testing.T) {
	ms := newOfflineStore(t)
	if _, err := ms.RememberWithID("dup", "first", nil, 0.5); err != nil {
		t.Fatal(err)
	}
	if _, err := ms.RememberWithID("dup", "second", nil, 0.5); err == nil {
		t.Error("duplicate id should error (caller must use Supersede)")
	}
	// Original record is still active and untouched.
	active := ms.Active()
	if len(active) != 1 || active[0].Content != "first" {
		t.Errorf("expected original 'first' to remain, got %+v", active)
	}
}

func TestRememberWithID_RequiresContent(t *testing.T) {
	ms := newOfflineStore(t)
	if _, err := ms.RememberWithID("id", "", nil, 0.5); err == nil {
		t.Error("empty content should error")
	}
}

func TestHasID_FalseForUnknown(t *testing.T) {
	ms := newOfflineStore(t)
	if ms.HasID("nope") {
		t.Error("HasID should be false for never-seen id")
	}
}

func TestHasID_TrueAfterDrop(t *testing.T) {
	// Tombstoned ids still register as "seen" — HasID is about the
	// journal, not the active set. Callers route to Supersede when
	// HasID is true; the supersede path would then fail on a dropped
	// record (its way of saying "you can't bring back the dead").
	ms := newOfflineStore(t)
	id, _ := ms.RememberWithID("zombie", "body", nil, 0.5)
	_ = ms.Drop(id, "test")
	if !ms.HasID(id) {
		t.Error("HasID should remain true after Drop (the id was seen)")
	}
}

// ---- load + supersede chain reconstruction ----------------------------

func TestLoad_ReconstructsActiveSet(t *testing.T) {
	ms := newOfflineStore(t)

	a, _ := ms.Remember("memory A", []string{"a"}, 0.5)
	b, _ := ms.Remember("memory B", []string{"b"}, 0.5)
	c, _ := ms.Remember("memory C", []string{"c"}, 0.5)
	_, _ = ms.Supersede(b, "memory B' (refined)", []string{"b"}, 0.6, "refined wording")
	_ = ms.Drop(c, "no longer relevant")

	// Reload from disk via a fresh store.
	ms2 := &MemoryStore{path: memoryFile, byID: map[string]int{}}
	ms2.load()

	active := ms2.Active()
	if len(active) != 2 {
		t.Fatalf("reloaded active = %d, want 2 (A and B'). got %v", len(active), active)
	}
	got := map[string]bool{}
	for _, r := range active {
		got[r.ID] = true
	}
	if !got[a] {
		t.Error("expected A in active set")
	}
	if got[b] {
		t.Error("old B should NOT be active (superseded)")
	}
	if got[c] {
		t.Error("C should NOT be active (dropped)")
	}
}

// ---- recall scoring ---------------------------------------------------

func TestRecall_RanksByLexicalScore_NoEmbedding(t *testing.T) {
	ms := newOfflineStore(t)
	ms.Remember("user prefers terse replies on technical topics", []string{"preference"}, 0.9)
	ms.Remember("the weather in Paris was warm yesterday", []string{"chitchat"}, 0.5)
	ms.Remember("Postgres runs on a custom port for this user", []string{"fact"}, 0.9)

	results := ms.Recall("user preference for short replies", 2)
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	// Top hit must be the "terse replies" memory because of the
	// "user" + "replies" overlap; weather should rank lowest.
	if !strings.Contains(results[0].Content, "terse replies") {
		t.Errorf("expected 'terse replies' to rank first, got: %q", results[0].Content)
	}
}

func TestRecall_RejectsZeroOverlapAcrossMemoryTags(t *testing.T) {
	ms := newOfflineStore(t)
	ms.Remember("Use signed URLs for storage uploads", []string{"procedure", "storage"}, 0.95)
	ms.Remember("The billing queue processes invoices", []string{"fact", "billing"}, 0.9)

	results := ms.Recall("quietly wait for future requests", 5)
	if len(results) != 0 {
		t.Fatalf("zero-overlap recall returned %d records: %+v", len(results), results)
	}
}

func TestSearch_RanksProceduralMemoryLexically_NoEmbedding(t *testing.T) {
	ms := newOfflineStore(t)
	if _, err := ms.RememberWithID(
		"procedure_storage_upload_0",
		"When uploading files to storage, validate MIME type, create a signed URL, then confirm checksum.",
		[]string{"procedure", "storage", "upload", "signed-url"},
		0.95,
	); err != nil {
		t.Fatalf("RememberWithID: %v", err)
	}
	if _, err := ms.Remember("Billing imports use the invoices queue.", []string{"billing", "invoices"}, 0.8); err != nil {
		t.Fatalf("Remember distractor: %v", err)
	}
	results := ms.Search("upload storage signed url", 1)
	if len(results) != 1 {
		t.Fatalf("Search returned %d results, want 1", len(results))
	}
	if results[0].ID != "procedure_storage_upload_0" {
		t.Fatalf("top lexical procedural memory = %q (%q), want procedure_storage_upload_0", results[0].ID, results[0].Content)
	}
	if len(results[0].Embedding) != 0 {
		t.Fatalf("offline procedural memory unexpectedly has embedding len=%d", len(results[0].Embedding))
	}
}

func TestDynamicTurnContext_IncludesLexicalMemoryWithoutEmbedding(t *testing.T) {
	ms := newOfflineStore(t)
	const sentinel = "ultramarine-blue-742"
	if _, err := ms.Remember(
		"For lexical memory smoke tests, the deployment color is "+sentinel+".",
		[]string{"deployment", "color", "lexical-memory"},
		0.95,
	); err != nil {
		t.Fatalf("Remember: %v", err)
	}
	recalled := ms.Recall("What deployment color should the lexical memory test use?", 1)
	ctx := buildDynamicTurnContext(nil, ms.BuildContext(recalled))
	if !strings.Contains(ctx, sentinel) {
		t.Fatalf("dynamic context did not include recalled lexical memory:\n%s", ctx)
	}
	if !strings.Contains(ctx, "[memories") {
		t.Fatalf("dynamic context missing memories block:\n%s", ctx)
	}
}

func TestRecall_SkipsTombstonedAndSuperseded(t *testing.T) {
	ms := newOfflineStore(t)
	stale, _ := ms.Remember("user prefers verbose replies", []string{"preference"}, 0.9)
	gone, _ := ms.Remember("currently typing", []string{"ephemeral"}, 0.5)
	keep, _ := ms.Remember("user runs Postgres on port 6543", []string{"fact"}, 0.9)

	_, _ = ms.Supersede(stale, "user prefers terse replies", []string{"preference"}, 0.9, "correction")
	_ = ms.Drop(gone, "ephemeral")

	// Recall over a query that COULD match the superseded entry's old wording —
	// we want to verify it's NOT returned because it's superseded.
	results := ms.Recall("verbose replies preference", 5)
	for _, r := range results {
		if r.ID == stale {
			t.Errorf("recall returned superseded memory id %s", stale)
		}
		if r.ID == gone {
			t.Errorf("recall returned tombstoned memory id %s", gone)
		}
	}
	// The new (terse) memory should be reachable.
	results2 := ms.Recall("user prefers replies", 5)
	var seenKeep bool
	var seenNewVersion bool
	for _, r := range results2 {
		if r.ID == keep {
			seenKeep = true
		}
		if strings.Contains(r.Content, "terse replies") {
			seenNewVersion = true
		}
	}
	if !seenKeep {
		t.Error("expected the never-touched memory to surface")
	}
	if !seenNewVersion {
		t.Error("expected the superseder (new wording) to surface")
	}
}

func TestRecall_DecayPenalizesOldMemories(t *testing.T) {
	ms := newOfflineStore(t)

	// Two memories with identical content shape, one old, one fresh.
	// Both have weight 0.9. The fresh one should rank above the old.
	old := MemoryRecord{
		ID:      newULID(),
		TS:      time.Now().Add(-365 * 24 * time.Hour), // 1 year old
		Content: "old deployment runs on Kubernetes",
		Tags:    []string{"fact"},
		Weight:  0.9,
	}
	fresh := MemoryRecord{
		ID:      newULID(),
		TS:      time.Now().Add(-1 * time.Hour),
		Content: "current deployment runs on Kubernetes",
		Tags:    []string{"fact"},
		Weight:  0.9,
	}
	ms.records = []MemoryRecord{old, fresh}
	ms.byID[old.ID] = 0
	ms.byID[fresh.ID] = 1

	results := ms.Recall("deployment Kubernetes", 2)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].ID != fresh.ID {
		t.Errorf("expected fresh memory first; got %s", results[0].ID)
	}
}

func TestRecall_DeduplicatesIdenticalContentAcrossTags(t *testing.T) {
	ms := newOfflineStore(t)
	first, _ := ms.Remember("Use signed URLs for storage uploads", []string{"procedure", "storage"}, 0.9)
	second, _ := ms.Remember("  use SIGNED urls for storage uploads  ", []string{"fact", "storage"}, 0.8)

	results := ms.Recall("storage upload signed URL", 5)
	if len(results) != 1 {
		t.Fatalf("Recall returned %d duplicate records, want 1: %+v", len(results), results)
	}
	if results[0].ID != first && results[0].ID != second {
		t.Fatalf("unexpected recalled id %q", results[0].ID)
	}
}

// ---- migration --------------------------------------------------------

func TestMigrateLegacy_RewritesToNewFormat(t *testing.T) {
	dir := t.TempDir()
	prev, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	// Write a legacy memory.jsonl: each line is the old { text, time, embedding } shape.
	f, _ := os.Create(memoryFile)
	enc := json.NewEncoder(f)
	enc.Encode(map[string]any{
		"text":      "old memory one",
		"time":      time.Now().Add(-72 * time.Hour),
		"embedding": []float64{0.1, 0.2, 0.3},
	})
	enc.Encode(map[string]any{
		"text": "old memory two",
		"time": time.Now().Add(-24 * time.Hour),
	})
	f.Close()

	// Constructing a store should run the migration in-place.
	ms := &MemoryStore{path: memoryFile, byID: map[string]int{}}
	ms.migrateLegacyIfNeeded()
	ms.load()

	if ms.Count() != 2 {
		t.Errorf("post-migration active count = %d, want 2", ms.Count())
	}

	// Migrated entries should have ids, tags, weight=0.5.
	for _, r := range ms.Active() {
		if r.ID == "" {
			t.Error("migrated record has empty id")
		}
		if r.Weight != 0.5 {
			t.Errorf("migrated record weight = %v, want 0.5", r.Weight)
		}
		if !contains(r.Tags, "legacy") || !contains(r.Tags, "migrated") {
			t.Errorf("migrated record missing tags: %v", r.Tags)
		}
	}

	// Legacy backup should exist on disk.
	if _, err := os.Stat(filepath.Join(dir, legacyMemoryBak)); err != nil {
		t.Errorf("expected legacy backup at %s: %v", legacyMemoryBak, err)
	}
}

func TestMigrateLegacy_SkipsAlreadyMigrated(t *testing.T) {
	ms := newOfflineStore(t)
	// Write a fresh-format record first.
	_, _ = ms.Remember("already-migrated content", []string{"fact"}, 0.7)
	// Now simulate a fresh boot — migration should NOT trigger.
	ms2 := &MemoryStore{path: memoryFile, byID: map[string]int{}}
	ms2.migrateLegacyIfNeeded()
	ms2.load()
	if _, err := os.Stat(legacyMemoryBak); err == nil {
		t.Error("legacy backup file created on already-new-format file")
	}
	if ms2.Count() != 1 {
		t.Errorf("post-load active count = %d, want 1", ms2.Count())
	}
}

// ---- BuildContext rendering ------------------------------------------

func TestBuildContext_FramingHeaderIncludesGuard(t *testing.T) {
	ms := newOfflineStore(t)
	ms.Remember("user prefers async", []string{"preference"}, 0.85)
	out := ms.BuildContext(ms.Active())
	if !strings.Contains(out, "[memories") {
		t.Error("BuildContext output missing [memories header")
	}
	// The defense against the fabrication bug — the rendered header
	// MUST tell the model these are memories, not current statements.
	if !strings.Contains(out, "do not treat as the user's current input") {
		t.Errorf("BuildContext output missing fabrication-guard framing: %q", out)
	}
}

func TestBuildContext_IsDeterministicForCacheReuse(t *testing.T) {
	ms := newOfflineStore(t)
	ms.Remember("Use signed URLs for storage uploads", []string{"storage", "procedure"}, 0.85)
	records := ms.Active()
	first := ms.BuildContext(records)
	time.Sleep(10 * time.Millisecond)
	second := ms.BuildContext(records)
	if first != second {
		t.Fatalf("BuildContext changed without memory changes:\nfirst=%q\nsecond=%q", first, second)
	}
	if strings.Contains(first, " ago,") {
		t.Fatalf("BuildContext includes volatile relative age: %q", first)
	}
}

func TestAutomaticRecallContextUsesSignalAndTotalBudget(t *testing.T) {
	ms := &MemoryStore{}
	now := time.Date(2026, 8, 13, 10, 0, 0, 0, time.UTC)
	matches := []MemoryRecallMatch{
		{Record: MemoryRecord{ID: "primary", TS: now, Content: "PRIMARY\n" + strings.Repeat("p", 12_000), Tags: []string{"patreon"}, Weight: 0.95}, Signal: 0.9, Score: 0.85},
		{Record: MemoryRecord{ID: "computer", TS: now, Content: "COMPUTER\n" + strings.Repeat("c", 8_000), Tags: []string{"computer"}, Weight: 0.9}, Signal: 0.8, Score: 0.72},
		{Record: MemoryRecord{ID: "large-third", TS: now, Content: "THIRD\n" + strings.Repeat("x", 8_000), Tags: []string{"channels"}, Weight: 0.9}, Signal: 0.7, Score: 0.63},
		{Record: MemoryRecord{ID: "weak", TS: now, Content: "WEAK", Tags: []string{"tasks"}, Weight: 0.9}, Signal: 0.19, Score: 0.17},
	}
	selected, context := ms.BuildAutomaticRecallContext(matches)
	if len(context) > automaticMemoryRecallMaxChars {
		t.Fatalf("automatic context chars = %d, limit = %d", len(context), automaticMemoryRecallMaxChars)
	}
	if len(selected) != 2 || selected[0].Record.ID != "primary" || selected[1].Record.ID != "computer" {
		t.Fatalf("selected matches = %+v", selected)
	}
	for _, want := range []string{"PRIMARY", "COMPUTER"} {
		if !strings.Contains(context, want) {
			t.Fatalf("context missing %q", want)
		}
	}
	for _, unwanted := range []string{"THIRD", "WEAK"} {
		if strings.Contains(context, unwanted) {
			t.Fatalf("context contains %q despite budget/signal filtering", unwanted)
		}
	}
}

func TestAutomaticRecallContextBoundsOneOversizedRecord(t *testing.T) {
	ms := &MemoryStore{}
	record := MemoryRecord{
		ID: "oversized", TS: time.Now().UTC(), Weight: 1,
		Content: "HEAD\n" + strings.Repeat("middle", 10_000) + "\nTAIL",
	}
	selected, context := ms.BuildAutomaticRecallContext([]MemoryRecallMatch{{Record: record, Signal: 1, Score: 1}})
	if len(selected) != 1 || len(context) > automaticMemoryRecallMaxChars {
		t.Fatalf("selected=%d context_chars=%d limit=%d", len(selected), len(context), automaticMemoryRecallMaxChars)
	}
	for _, want := range []string{"HEAD", "TAIL", "memory content bounded by core"} {
		if !strings.Contains(context, want) {
			t.Fatalf("bounded context missing %q", want)
		}
	}
}

func TestAutomaticRecallUsesAllRelevantPartsOfLongStandingDirective(t *testing.T) {
	ms := newOfflineStore(t)
	const patreonID = "skill_patreon_validation"
	const computerID = "skill_computer_observation"
	if _, err := ms.RememberWithID(
		patreonID,
		"PATREON_VALIDATION_SKILL\nThe Patreon browser validation code is VIOLET-SEAL-927. Supply this exact code to every browser_validation_step call.\n"+
			strings.Repeat("Patreon browser validation procedure for an existing draft. ", 125),
		[]string{"skill", "patreon", "browser", "validation"}, 0.95,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := ms.RememberWithID(
		computerID,
		"COMPUTER_OBSERVATION_SKILL\nUse each structured browser observation result before selecting the next step.\n"+
			strings.Repeat("Computer browser observation procedure and structured tool result guidance. ", 110),
		[]string{"skill", "computer", "browser", "observation"}, 0.95,
	); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if _, err := ms.RememberWithID(
			fmt.Sprintf("skill_unrelated_%02d", i),
			fmt.Sprintf("UNRELATED_SKILL_%02d\n", i)+strings.Repeat("Payroll tax kitchen inventory warehouse archive. ", 130),
			[]string{"skill", "unrelated", "archive"}, 0.9,
		); err != nil {
			t.Fatal(err)
		}
	}

	directive := strings.Join([]string{
		"# Role",
		"Complete this bounded validation directly. Do not spawn, send, evolve, or delegate.",
		"# Workflow",
		"On the external request, read the automatically recalled Patreon and Computer guidance.",
		"Call browser_validation_step for step 1 using the exact recalled validation code.",
		"After each successful result, call it exactly once for next_step. Do not skip or repeat steps.",
		"When the tool reports complete=true, reply exactly CODEX_MEMORY_CACHE_OK with no other text and wait for events.",
	}, "\n")
	matches := ms.RecallMatchesForContexts([]string{
		"[console] Begin now and use the shared operating guidance.",
		directive,
	}, automaticMemoryRecallMaxRecords)
	selected, context := ms.BuildAutomaticRecallContext(matches)
	selectedIDs := make(map[string]bool, len(selected))
	for _, match := range selected {
		selectedIDs[match.Record.ID] = true
	}
	if !selectedIDs[patreonID] || !selectedIDs[computerID] {
		t.Fatalf("selected=%v, want both relevant skills; ranked=%+v", selectedIDs, matches)
	}
	if strings.Contains(context, "UNRELATED_SKILL_") {
		t.Fatalf("bounded context contains unrelated skill: %s", context)
	}
}

func TestMemoryGenerationChangesOnlyAfterSuccessfulAppend(t *testing.T) {
	t.Chdir(t.TempDir())
	ms := NewMemoryStore("")
	initial := ms.Generation()
	if _, err := ms.Remember("generation sentinel", []string{"test"}, 0.8); err != nil {
		t.Fatal(err)
	}
	if got := ms.Generation(); got != initial+1 {
		t.Fatalf("generation after remember = %d, want %d", got, initial+1)
	}
	if err := ms.Drop("missing", "must fail"); err == nil {
		t.Fatal("missing memory drop unexpectedly succeeded")
	}
	if got := ms.Generation(); got != initial+1 {
		t.Fatalf("failed mutation changed generation to %d", got)
	}
}

// ---- helpers ----------------------------------------------------------

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
