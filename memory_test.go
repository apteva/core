package core

import (
	"encoding/json"
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
	id, err := ms.RememberWithID("skill_abc_0", "playbook body", []string{"skill", "skill:storage:upload"}, 0.9)
	if err != nil {
		t.Fatalf("RememberWithID: %v", err)
	}
	if id != "skill_abc_0" {
		t.Errorf("returned id = %q, want skill_abc_0", id)
	}
	if !ms.HasID("skill_abc_0") {
		t.Error("HasID should be true after RememberWithID")
	}
	active := ms.Active()
	if len(active) != 1 || active[0].ID != "skill_abc_0" {
		t.Errorf("Active record mismatch: %+v", active)
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
		Content: "deployment runs on Kubernetes",
		Tags:    []string{"fact"},
		Weight:  0.9,
	}
	fresh := MemoryRecord{
		ID:      newULID(),
		TS:      time.Now().Add(-1 * time.Hour),
		Content: "deployment runs on Kubernetes",
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

// ---- helpers ----------------------------------------------------------

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
