package core

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Memory v2 — append-only journal, unconscious-only writer, automatic
// relevance injection into main's per-turn context.
//
// Each line in memory.jsonl is a MemoryRecord. There are two kinds of
// records:
//
//   1. Memory records — a typed memory the unconscious decided to keep.
//      Carry Content / Tags / Weight / Embedding. Optional Supersedes
//      pointing at the id of an older memory this one replaces.
//
//   2. Tombstone records — sparse marker that a memory was dropped or
//      explicitly superseded. Carry Tombstone=true + IDTarget + Reason.
//      Tombstone records never carry content; the only data they store
//      is "id X is no longer active, here is why".
//
// At read time the store reconstructs the "active" set: every memory
// record whose id is NOT the IDTarget of a tombstone AND is NOT pointed
// at by another record's Supersedes. Old records stay on disk forever
// (audit trail); recall just skips them.
//
// The runtime owns id, ts, embedding, supersede linkage, and tombstone
// shape. The LLM (unconscious thread) owns content, tags, and weight —
// no fixed `subject` or `type` schema. Tags are free-form; the LLM
// decides what dimensions matter.

const (
	memoryFile      = "memory.jsonl"
	legacyMemoryBak = "memory.jsonl.legacy.bak"

	// Automatic recall is request context, not a bulk memory export. Keep the
	// selected records useful and bounded; full records remain in the journal
	// and explicit memory search remains unchanged.
	automaticMemoryRecallMaxRecords = 5
	automaticMemoryRecallMaxChars   = 24 * 1024
	automaticMemoryRecallMinSignal  = 0.20

	// Default decay half-life: a memory's effective weight halves
	// every this many days unless reinforced. 90 days = a memory
	// from 6 months ago contributes 1/4 of its original weight.
	memoryHalfLifeDays = 90.0

	// Below this cosine score, embedding similarity is usually background
	// correlation rather than useful recall evidence.
	minEmbeddingRecallSimilarity = 0.20

	// Soft target the unconscious is told about each cycle. The
	// directive uses this to decide when to be more aggressive on
	// drops. Not a hard cap — exceeding it doesn't lose data.
	memorySoftTarget = 1000
)

// errMemoryDisabled — no embedding backend configured. Callers that
// produce per-iteration noise (RAG indexing, recall) short-circuit on
// this rather than logging.
var errMemoryDisabled = errors.New("memory disabled — no embedding backend configured")

// embeddingBackend captures everything embed() needs to call out to a
// concrete embeddings provider. Picked once at MemoryStore creation
// based on which env vars are set; never changes for the lifetime of
// the store. nil means "memory is disabled" — embed() short-circuits.
type embeddingBackend struct {
	URL    string
	Model  string
	APIKey string
	Header string
	Dim    int
	Source string
}

// detectEmbeddingBackend picks the embedding provider based on env.
// Order: Fireworks → OpenAI → Ollama. Returns nil when nothing is
// available — memory then runs in lexical-only mode (FTS-style scoring
// over content + tags), no embedding API calls made.
func detectEmbeddingBackend() *embeddingBackend {
	if k := os.Getenv("FIREWORKS_API_KEY"); k != "" {
		return &embeddingBackend{
			URL:    "https://api.fireworks.ai/inference/v1/embeddings",
			Model:  "nomic-ai/nomic-embed-text-v1.5",
			APIKey: k, Header: "Bearer", Dim: 768, Source: "fireworks",
		}
	}
	if k := os.Getenv("OPENAI_API_KEY"); k != "" {
		return &embeddingBackend{
			URL:    "https://api.openai.com/v1/embeddings",
			Model:  "text-embedding-3-small",
			APIKey: k, Header: "Bearer", Dim: 1536, Source: "openai",
		}
	}
	if h := os.Getenv("OLLAMA_HOST"); h != "" {
		model := strings.TrimSpace(os.Getenv("OLLAMA_EMBED_MODEL"))
		if model == "" {
			model = "nomic-embed-text"
		}
		dim := 768
		if raw := strings.TrimSpace(os.Getenv("OLLAMA_EMBED_DIM")); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
				dim = parsed
			}
		}
		return &embeddingBackend{
			URL:    strings.TrimRight(h, "/") + "/api/embeddings",
			Model:  model,
			APIKey: "", Header: "", Dim: dim, Source: "ollama",
		}
	}
	return nil
}

// MemoryRecord is one line in memory.jsonl. Either a memory or a
// tombstone — never both.
type MemoryRecord struct {
	ID         string    `json:"id"`
	TS         time.Time `json:"ts"`
	Content    string    `json:"content,omitempty"`
	Tags       []string  `json:"tags,omitempty"`
	Weight     float64   `json:"weight,omitempty"`
	Supersedes string    `json:"supersedes,omitempty"`
	Embedding  []float64 `json:"embedding,omitempty"`

	// Tombstone bits.
	Tombstone bool   `json:"tombstone,omitempty"`
	IDTarget  string `json:"id_target,omitempty"`
	Reason    string `json:"reason,omitempty"`
}

// IsTombstone reports whether this record is a tombstone marker.
func (r MemoryRecord) IsTombstone() bool { return r.Tombstone }

// MemoryStore is the in-process journal owner. Append-only on disk,
// rebuilds the active-set on load.
type MemoryStore struct {
	mu          sync.RWMutex
	records     []MemoryRecord // full sequence, in insertion order
	byID        map[string]int // id → index in records
	active      map[string]MemoryRecord
	activeOrder []string
	generation  uint64            // increments whenever records are appended
	backend     *embeddingBackend // nil → no embeddings, lexical-only
	path        string
}

// NewMemoryStore opens (or creates) memory.jsonl at the cwd, picks
// an embedding backend from env, runs the legacy-format migration if
// needed, and returns a ready store. apiKey is kept for backward-
// compat as a forced Fireworks key when env-based detection finds
// nothing — passing "" defers entirely to env.
func NewMemoryStore(apiKey string) *MemoryStore {
	backend := detectEmbeddingBackend()
	if backend == nil && apiKey != "" {
		backend = &embeddingBackend{
			URL:    "https://api.fireworks.ai/inference/v1/embeddings",
			Model:  "nomic-ai/nomic-embed-text-v1.5",
			APIKey: apiKey, Header: "Bearer", Dim: 768, Source: "fireworks (param)",
		}
	}
	ms := &MemoryStore{
		backend: backend,
		path:    memoryFile,
		byID:    map[string]int{},
		active:  map[string]MemoryRecord{},
	}
	if backend == nil {
		logMsg("MEMORY", "embeddings disabled — lexical-only retrieval (set FIREWORKS_API_KEY / OPENAI_API_KEY / OLLAMA_HOST to enable embeddings)")
	} else {
		logMsg("MEMORY", fmt.Sprintf("embeddings via %s (model=%s dim=%d)", backend.Source, backend.Model, backend.Dim))
	}
	ms.migrateLegacyIfNeeded()
	ms.load()
	return ms
}

// Enabled reports whether embeddings are available. Lexical scoring
// still works either way; callers that ONLY need embeddings (RAG
// tool indexing in api.go / thinker.go) check this to short-circuit.
func (ms *MemoryStore) Enabled() bool { return ms.backend != nil }

// migrateLegacyIfNeeded looks at the first record on disk; if it has
// the old shape (`text` field, no `id` field), the whole file is
// treated as legacy and migrated in one shot:
//  1. Rename memory.jsonl → memory.jsonl.legacy.bak
//  2. Re-read each legacy entry, write a fresh new-format record
//     with id=ULID, ts=original time, content=text, tags=["legacy",
//     "migrated"], weight=0.5, supersedes="".
//  3. Embeddings on legacy entries are preserved when their dim
//     matches the current backend; otherwise dropped (will recompute
//     on first recall if/when backend changes).
//
// One-shot, idempotent: after migration the file has only new-format
// records, the legacy bak stays for forensic reference.
func (ms *MemoryStore) migrateLegacyIfNeeded() {
	data, err := os.ReadFile(ms.path)
	if err != nil || len(data) == 0 {
		return
	}
	// Peek first non-empty line.
	lines := bytes.Split(data, []byte("\n"))
	var first []byte
	for _, l := range lines {
		if len(bytes.TrimSpace(l)) > 0 {
			first = l
			break
		}
	}
	if first == nil {
		return
	}
	var probe map[string]any
	if err := json.Unmarshal(first, &probe); err != nil {
		return
	}
	// Heuristic: legacy records have `text`, lack `id`. Tombstones
	// or new memories always have `id`.
	if _, hasText := probe["text"]; !hasText {
		return
	}
	if _, hasID := probe["id"]; hasID {
		return
	}

	// It's the legacy format. Migrate.
	if err := os.Rename(ms.path, legacyMemoryBak); err != nil {
		logMsg("MEMORY", fmt.Sprintf("legacy rename failed: %v — leaving file as-is", err))
		return
	}
	logMsg("MEMORY", "migrating legacy memory.jsonl → new journal format (backup at memory.jsonl.legacy.bak)")

	migrated := 0
	out, err := os.OpenFile(ms.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		logMsg("MEMORY", fmt.Sprintf("migration: failed to open new file: %v", err))
		return
	}
	_ = out.Chmod(0600)
	defer out.Close()
	enc := json.NewEncoder(out)

	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var legacy struct {
			Text      string    `json:"text"`
			Time      time.Time `json:"time"`
			Embedding []float64 `json:"embedding"`
		}
		if err := dec.Decode(&legacy); err != nil {
			continue
		}
		if legacy.Text == "" {
			continue
		}
		rec := MemoryRecord{
			ID:      newULID(),
			TS:      legacy.Time,
			Content: legacy.Text,
			Tags:    []string{"legacy", "migrated"},
			Weight:  0.5,
		}
		if rec.TS.IsZero() {
			rec.TS = time.Now().UTC()
		}
		// Preserve embedding if the dimension matches the current backend.
		if ms.backend != nil && len(legacy.Embedding) == ms.backend.Dim {
			rec.Embedding = legacy.Embedding
		}
		if err := enc.Encode(&rec); err != nil {
			continue
		}
		migrated++
	}
	logMsg("MEMORY", fmt.Sprintf("migration: %d legacy entries → new format", migrated))
}

// load reads memory.jsonl into ms.records in insertion order and
// builds the byID index. Records with mismatched embedding dim against
// the active backend keep their embedding (recall just won't use it
// for cosine — falls back to lexical match for those entries).
func (ms *MemoryStore) load() {
	data, err := os.ReadFile(ms.path)
	if err != nil {
		return
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var rec MemoryRecord
		if err := dec.Decode(&rec); err != nil {
			continue
		}
		if rec.ID == "" {
			continue
		}
		ms.records = append(ms.records, rec)
		ms.byID[rec.ID] = len(ms.records) - 1
	}
	ms.rebuildActiveLocked()
	logMsg("MEMORY", fmt.Sprintf("loaded %d records (%d active)", len(ms.records), ms.activeCount()))
}

// activeCount counts active (non-tombstoned, non-superseded) memories.
// Caller holds ms.mu.RLock() OR is within a write-locked section.
func (ms *MemoryStore) activeCount() int {
	ms.ensureActiveLocked()
	return len(ms.active)
}

func (ms *MemoryStore) ensureActiveLocked() {
	if ms.active == nil {
		ms.rebuildActiveLocked()
	}
}

func (ms *MemoryStore) rebuildActiveLocked() {
	ms.active = make(map[string]MemoryRecord)
	ms.activeOrder = nil
	for _, rec := range ms.records {
		ms.applyActiveRecordLocked(rec)
	}
}

func (ms *MemoryStore) applyActiveRecordLocked(rec MemoryRecord) {
	if rec.Tombstone {
		ms.removeActiveLocked(rec.IDTarget)
		return
	}
	if rec.Supersedes != "" {
		ms.removeActiveLocked(rec.Supersedes)
	}
	if _, exists := ms.active[rec.ID]; !exists {
		ms.activeOrder = append(ms.activeOrder, rec.ID)
	}
	ms.active[rec.ID] = rec
}

func (ms *MemoryStore) removeActiveLocked(id string) {
	if id == "" {
		return
	}
	if _, exists := ms.active[id]; !exists {
		return
	}
	delete(ms.active, id)
	for i, activeID := range ms.activeOrder {
		if activeID == id {
			ms.activeOrder = append(ms.activeOrder[:i], ms.activeOrder[i+1:]...)
			break
		}
	}
}

// deadIDs returns the sets of ids that are tombstoned (explicitly
// dropped or replaced via supersede) and superseded (pointed at by a
// newer record's Supersedes field). Caller must hold the lock.
func (ms *MemoryStore) deadIDs() (tombstoned, superseded map[string]bool) {
	tombstoned = map[string]bool{}
	superseded = map[string]bool{}
	for _, r := range ms.records {
		if r.Tombstone && r.IDTarget != "" {
			tombstoned[r.IDTarget] = true
		}
		if r.Supersedes != "" {
			superseded[r.Supersedes] = true
		}
	}
	return
}

// Active returns the current active memories — everything not
// tombstoned and not superseded by a newer record. Returned slice is
// a copy; callers can mutate / sort freely.
func (ms *MemoryStore) Active() []MemoryRecord {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	// Stores loaded/created through public constructors always have this
	// index. The nil case is only possible in legacy test literals.
	if ms.active == nil {
		ms.mu.RUnlock()
		ms.mu.Lock()
		ms.ensureActiveLocked()
		ms.mu.Unlock()
		ms.mu.RLock()
	}
	out := make([]MemoryRecord, 0, len(ms.active))
	for _, id := range ms.activeOrder {
		if rec, ok := ms.active[id]; ok {
			out = append(out, rec)
		}
	}
	return out
}

// Count returns the number of currently-active memories. Used by
// telemetry and the unconscious's directive ("you have N memories").
func (ms *MemoryStore) Count() int {
	ms.mu.RLock()
	if ms.active != nil {
		n := len(ms.active)
		ms.mu.RUnlock()
		return n
	}
	ms.mu.RUnlock()
	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.ensureActiveLocked()
	return len(ms.active)
}

// Generation identifies the current in-process memory view. Threads use it
// to keep one retrieval snapshot across internal continuations while still
// noticing memories written by another thread.
func (ms *MemoryStore) Generation() uint64 {
	if ms == nil {
		return 0
	}
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.generation
}

// All returns every record in insertion order, including tombstones
// and superseded entries. Used by the dashboard memory panel for
// debugging / audit.
func (ms *MemoryStore) All() []MemoryRecord {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	out := make([]MemoryRecord, len(ms.records))
	copy(out, ms.records)
	return out
}

// append writes a record to disk and updates in-memory state. Caller
// must NOT hold ms.mu — append takes it.
func (ms *MemoryStore) append(rec MemoryRecord) error {
	return ms.appendRecords(rec)
}

func (ms *MemoryStore) appendRecords(records ...MemoryRecord) error {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	var payload bytes.Buffer
	enc := json.NewEncoder(&payload)
	for i := range records {
		if err := enc.Encode(&records[i]); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(ms.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_ = f.Chmod(0600)
	defer f.Close()
	if _, err := f.Write(payload.Bytes()); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	ms.ensureActiveLocked()
	for _, rec := range records {
		ms.records = append(ms.records, rec)
		if rec.ID != "" {
			ms.byID[rec.ID] = len(ms.records) - 1
		}
		ms.applyActiveRecordLocked(rec)
	}
	ms.generation++
	return nil
}

// Remember writes a fresh memory and returns its id. weight defaults to
// 0.7 if zero. tags may be nil. Embedding is computed when a backend is
// configured; on failure the record is still written without an
// embedding (lexical recall continues to work).
func (ms *MemoryStore) Remember(content string, tags []string, weight float64) (string, error) {
	if strings.TrimSpace(content) == "" {
		return "", errors.New("memory_remember: content required")
	}
	if weight <= 0 {
		weight = 0.7
	}
	if weight > 1 {
		weight = 1
	}
	rec := MemoryRecord{
		ID:      newULID(),
		TS:      time.Now().UTC(),
		Content: content,
		Tags:    tags,
		Weight:  weight,
	}
	if ms.backend != nil {
		if emb, err := ms.embed(content); err == nil {
			rec.Embedding = emb
		}
	}
	if err := ms.append(rec); err != nil {
		return "", err
	}
	logMsg("MEMORY", fmt.Sprintf("remember: id=%s w=%.2f tags=%v len=%d", rec.ID, rec.Weight, rec.Tags, len(content)))
	return rec.ID, nil
}

// RememberWithID is Remember with a caller-supplied id instead of a
// freshly-minted ULID. Required for deterministic platform-managed
// records, where re-pushing the same source upserts via Supersede rather
// than creating a duplicate row.
//
// Errors if id is empty (use Remember) or if the id already exists
// (caller should call HasID first and route to Supersede). Refusing
// silent overwrite keeps the journal append-only semantics intact.
func (ms *MemoryStore) RememberWithID(id, content string, tags []string, weight float64) (string, error) {
	if strings.TrimSpace(id) == "" {
		return "", errors.New("memory_remember: id required (use Remember for autogen)")
	}
	if strings.TrimSpace(content) == "" {
		return "", errors.New("memory_remember: content required")
	}
	ms.mu.RLock()
	_, exists := ms.byID[id]
	ms.mu.RUnlock()
	if exists {
		return "", fmt.Errorf("memory_remember: id %q already exists (use Supersede to replace)", id)
	}
	if weight <= 0 {
		weight = 0.7
	}
	if weight > 1 {
		weight = 1
	}
	rec := MemoryRecord{
		ID:      id,
		TS:      time.Now().UTC(),
		Content: content,
		Tags:    tags,
		Weight:  weight,
	}
	if ms.backend != nil {
		if emb, err := ms.embed(content); err == nil {
			rec.Embedding = emb
		}
	}
	if err := ms.append(rec); err != nil {
		return "", err
	}
	logMsg("MEMORY", fmt.Sprintf("remember-with-id: id=%s w=%.2f tags=%v len=%d", rec.ID, rec.Weight, rec.Tags, len(content)))
	return rec.ID, nil
}

// HasID reports whether a record with this id exists in the journal
// (active, tombstoned, or superseded — anything that was ever written).
// Used by callers that want to decide between RememberWithID (insert)
// and Supersede (update) without racing on the not-found error.
func (ms *MemoryStore) HasID(id string) bool {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	_, ok := ms.byID[id]
	return ok
}

// UpsertTargetID returns the active record that should be superseded
// when a caller upserts a deterministic id. The first push writes the
// deterministic id itself. Later pushes create replacement records with
// generated ids, so a repeated push of the original id must supersede the
// latest active replacement, not the already-dead original.
func (ms *MemoryStore) UpsertTargetID(id string) (string, bool) {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	if _, ok := ms.byID[id]; !ok {
		return "", false
	}
	ms.ensureActiveLocked()
	for _, activeID := range ms.activeOrder {
		r, ok := ms.active[activeID]
		if !ok {
			continue
		}
		if r.ID == id || ms.supersedesLocked(r.ID, id) {
			return r.ID, true
		}
	}
	return id, true
}

func (ms *MemoryStore) supersedesLocked(candidateID, targetID string) bool {
	seen := map[string]bool{}
	for candidateID != "" && !seen[candidateID] {
		seen[candidateID] = true
		idx, ok := ms.byID[candidateID]
		if !ok {
			return false
		}
		parent := ms.records[idx].Supersedes
		if parent == targetID {
			return true
		}
		candidateID = parent
	}
	return false
}

// Supersede writes a NEW memory and a tombstone for oldID, linking
// them via the new record's Supersedes field. Both records are
// appended atomically (one after the other, no other writer in
// between because we hold the lock for both). Returns the new id.
func (ms *MemoryStore) Supersede(oldID, content string, tags []string, weight float64, reason string) (string, error) {
	if oldID == "" {
		return "", errors.New("memory_supersede: old_id required")
	}
	if strings.TrimSpace(content) == "" {
		return "", errors.New("memory_supersede: content required")
	}
	if reason == "" {
		return "", errors.New("memory_supersede: reason required")
	}

	ms.mu.RLock()
	if _, ok := ms.byID[oldID]; !ok {
		ms.mu.RUnlock()
		return "", fmt.Errorf("memory_supersede: id %q not found", oldID)
	}
	ms.mu.RUnlock()

	if weight <= 0 {
		weight = 0.7
	}
	if weight > 1 {
		weight = 1
	}
	newRec := MemoryRecord{
		ID:         newULID(),
		TS:         time.Now().UTC(),
		Content:    content,
		Tags:       tags,
		Weight:     weight,
		Supersedes: oldID,
	}
	if ms.backend != nil {
		if emb, err := ms.embed(content); err == nil {
			newRec.Embedding = emb
		}
	}
	tomb := MemoryRecord{
		ID:        newULID(),
		TS:        time.Now().UTC(),
		Tombstone: true,
		IDTarget:  oldID,
		Reason:    "superseded by " + newRec.ID + ": " + reason,
	}
	if err := ms.appendRecords(newRec, tomb); err != nil {
		return "", err
	}
	logMsg("MEMORY", fmt.Sprintf("supersede: %s → %s (%s)", oldID, newRec.ID, reason))
	return newRec.ID, nil
}

// Drop tombstones a memory by id. reason is required.
func (ms *MemoryStore) Drop(id, reason string) error {
	if id == "" {
		return errors.New("memory_drop: id required")
	}
	if reason == "" {
		return errors.New("memory_drop: reason required")
	}
	ms.mu.RLock()
	if _, ok := ms.byID[id]; !ok {
		ms.mu.RUnlock()
		return fmt.Errorf("memory_drop: id %q not found", id)
	}
	ms.mu.RUnlock()
	tomb := MemoryRecord{
		ID:        newULID(),
		TS:        time.Now().UTC(),
		Tombstone: true,
		IDTarget:  id,
		Reason:    reason,
	}
	if err := ms.append(tomb); err != nil {
		return err
	}
	logMsg("MEMORY", fmt.Sprintf("drop: %s (%s)", id, reason))
	return nil
}

// Search returns active memories matching the query. Embedding-based
// when a backend is configured, lexical (BM25-ish over content + tags)
// otherwise. Used by the unconscious's memory_search tool to look up
// existing memories before deciding remember vs supersede.
func (ms *MemoryStore) Search(query string, limit int) []MemoryRecord {
	if limit <= 0 {
		limit = 10
	}
	scored := ms.scoreActive(query, scoreOpts{useEmbedding: ms.backend != nil})
	if len(scored) > limit {
		scored = scored[:limit]
	}
	out := make([]MemoryRecord, len(scored))
	for i, s := range scored {
		out[i] = s.rec
	}
	return out
}

// Recall returns the top-N active memories scored by relevance to the
// given query context — multi-factor: cosine × weight × decay, with a
// lexical fallback when no embedding backend.
//
// Used by buildDynamicTurnContext for auto-injection at every turn. N
// is typically 3–5 with a token-budget cap applied by the caller.
func (ms *MemoryStore) Recall(query string, n int) []MemoryRecord {
	matches := ms.RecallMatches(query, n)
	out := make([]MemoryRecord, len(matches))
	for i, match := range matches {
		out[i] = match.Record
	}
	return out
}

// MemoryRecallMatch retains scoring metadata for request telemetry. All
// records intentionally share one path regardless of their free-form tags.
type MemoryRecallMatch struct {
	Record MemoryRecord
	Score  float64
	Signal float64
}

// RecallMatches is Recall with score metadata and content deduplication.
func (ms *MemoryStore) RecallMatches(query string, n int) []MemoryRecallMatch {
	if n <= 0 {
		n = 5
	}
	scored := ms.scoreActive(query, scoreOpts{useEmbedding: ms.backend != nil, applyDecay: true})
	out := make([]MemoryRecallMatch, 0, min(n, len(scored)))
	seenContent := make(map[string]struct{}, len(scored))
	for _, item := range scored {
		contentKey := normalizeMemoryContent(item.rec.Content)
		if _, duplicate := seenContent[contentKey]; duplicate {
			continue
		}
		seenContent[contentKey] = struct{}{}
		out = append(out, MemoryRecallMatch{Record: item.rec, Score: item.score, Signal: item.signal})
		if len(out) == n {
			break
		}
	}
	return out
}

type scoreOpts struct {
	useEmbedding bool
	applyDecay   bool
}

type scoredRec struct {
	rec    MemoryRecord
	score  float64
	signal float64
}

// scoreActive ranks all currently-active memories by relevance to a
// query. Returns sorted descending. Caller-defined options control
// whether embeddings are used and whether decay is applied.
//
// Scoring formula:
//
//	score = signal(query, content+tags) * weight * decay(age)
//
// where signal is cosine similarity if we have an embedding backend
// AND the record has an embedding of the matching dim, otherwise
// lexical token-overlap similarity. Weight and decay are no-ops if
// not applicable (weight < 0 → treated as 0; no embedding → fall back
// to lexical for that record alone).
func (ms *MemoryStore) scoreActive(query string, opts scoreOpts) []scoredRec {
	if strings.TrimSpace(query) == "" {
		return nil
	}
	active := ms.Active()
	if len(active) == 0 {
		return nil
	}

	var queryEmb []float64
	if opts.useEmbedding && ms.backend != nil {
		if emb, err := ms.embed(query); err == nil {
			queryEmb = emb
		}
	}
	queryTokens := meaningfulMemoryTokens(tokenize(query))
	now := time.Now().UTC()

	out := make([]scoredRec, 0, len(active))
	for _, r := range active {
		// Signal: prefer embedding cosine when both sides have one
		// of matching dim; fall back to lexical otherwise.
		var signal float64
		usedEmbedding := queryEmb != nil && len(r.Embedding) == len(queryEmb)
		if usedEmbedding {
			signal = cosineSimilarity(queryEmb, r.Embedding)
			if signal < 0 {
				signal = 0
			}
		} else {
			signal = lexicalScore(queryTokens, r)
		}
		if signal <= 0 || (usedEmbedding && signal < minEmbeddingRecallSimilarity) {
			continue
		}

		// Weight floor at 0.05 so a memory with weight=0 doesn't
		// disappear entirely — the LLM's weight=0 might mean "low
		// importance" rather than "delete". Tombstone is the
		// disappearance mechanism.
		w := r.Weight
		if w <= 0 {
			w = 0.05
		}

		var decay float64 = 1.0
		if opts.applyDecay {
			ageDays := now.Sub(r.TS).Hours() / 24.0
			decay = math.Pow(0.5, ageDays/memoryHalfLifeDays)
		}

		out = append(out, scoredRec{
			rec:    r,
			score:  signal * w * decay,
			signal: signal,
		})
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].score == out[j].score {
			return out[i].rec.ID < out[j].rec.ID
		}
		return out[i].score > out[j].score
	})
	return out
}

// BuildContext renders a slice of memories as the dynamic-context
// [memories] block. Header is explicit about provenance so the LLM
// reads these as memories, not current statements — that's the
// structural defense against the fabricated-approvals failure mode.
func (ms *MemoryStore) BuildContext(records []MemoryRecord) string {
	if len(records) == 0 {
		return ""
	}
	var buf bytes.Buffer
	buf.WriteString("[memories — surfaced because they may be relevant; check the dates, do not treat as the user's current input]\n")
	for _, r := range records {
		buf.WriteString(renderMemoryContextEntry(r))
	}
	return buf.String()
}

// BuildAutomaticRecallContext filters and renders the ranked matches used by
// the automatic per-cycle memory injection. It keeps whole records whenever
// possible so procedural guidance is not silently cut in half. Only an
// individually oversized first record is deterministically excerpted.
func (ms *MemoryStore) BuildAutomaticRecallContext(matches []MemoryRecallMatch) ([]MemoryRecallMatch, string) {
	if len(matches) == 0 {
		return nil, ""
	}
	const header = "[memories — surfaced because they may be relevant; check the dates, do not treat as the user's current input]\n"
	selected := make([]MemoryRecallMatch, 0, len(matches))
	used := len(header)
	for _, match := range matches {
		if match.Signal < automaticMemoryRecallMinSignal {
			continue
		}
		entry := renderMemoryContextEntry(match.Record)
		if used+len(entry) <= automaticMemoryRecallMaxChars {
			selected = append(selected, match)
			used += len(entry)
			continue
		}
		if len(selected) > 0 {
			continue
		}

		// A single record can exceed the whole automatic-recall budget. Keep
		// a deterministic head/tail excerpt rather than letting one memory
		// make every model request unbounded.
		bounded := match
		overhead := len(renderMemoryContextEntry(MemoryRecord{
			TS: match.Record.TS, Tags: match.Record.Tags, Weight: match.Record.Weight,
		}))
		available := automaticMemoryRecallMaxChars - len(header) - overhead
		bounded.Record.Content = boundedMemoryContent(match.Record.Content, available)
		entry = renderMemoryContextEntry(bounded.Record)
		selected = append(selected, bounded)
		used += len(entry)
	}
	if len(selected) == 0 {
		return nil, ""
	}
	records := make([]MemoryRecord, len(selected))
	for i, match := range selected {
		records[i] = match.Record
	}
	return selected, ms.BuildContext(records)
}

func renderMemoryContextEntry(r MemoryRecord) string {
	tagStr := ""
	if len(r.Tags) > 0 {
		tags := append([]string(nil), r.Tags...)
		sort.Strings(tags)
		tagStr = " [" + strings.Join(tags, ",") + "]"
	}
	return fmt.Sprintf("- (remembered %s, w=%.2f)%s %s\n",
		r.TS.UTC().Format("2006-01-02"), r.Weight, tagStr, r.Content)
}

func boundedMemoryContent(content string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(content) <= maxBytes {
		return content
	}
	marker := "\n... [memory content bounded by core] ...\n"
	if len(marker) >= maxBytes {
		return validToolResultPrefix(content, maxBytes)
	}
	available := maxBytes - len(marker)
	head := available * 3 / 4
	tail := available - head
	return validToolResultPrefix(content, head) + marker + validToolResultSuffix(content, tail)
}

// embed calls the active embedding backend. Public so the tool-RAG
// indexer in api.go / thinker.go can reuse the same path. Returns
// errMemoryDisabled when no backend is configured.
type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

type ollamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (ms *MemoryStore) embed(text string) ([]float64, error) {
	if ms.backend == nil {
		return nil, errMemoryDisabled
	}
	b := ms.backend
	var reqBody []byte
	if b.Source == "ollama" {
		reqBody, _ = json.Marshal(ollamaEmbeddingRequest{Model: b.Model, Prompt: text})
	} else {
		reqBody, _ = json.Marshal(embeddingRequest{Model: b.Model, Input: text})
	}

	req, err := http.NewRequest("POST", b.URL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if b.Header != "" && b.APIKey != "" {
		req.Header.Set("Authorization", b.Header+" "+b.APIKey)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API error %d (backend=%s): %s", resp.StatusCode, b.Source, string(body))
	}
	if b.Source == "ollama" {
		var ollama ollamaEmbeddingResponse
		if err := json.NewDecoder(resp.Body).Decode(&ollama); err != nil {
			return nil, err
		}
		if len(ollama.Embedding) == 0 {
			return nil, errors.New("no embedding returned (ollama)")
		}
		return ollama.Embedding, nil
	}
	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Data) == 0 {
		return nil, errors.New("no embedding returned")
	}
	return result.Data[0].Embedding, nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// tokenize lowercases + splits on non-alphanumerics. Used by the
// lexical-fallback scorer when no embedding backend is configured.
// Cheap; we don't need linguistic correctness, just consistent
// matching between query and content.
func tokenize(s string) map[string]int {
	out := map[string]int{}
	cur := strings.Builder{}
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		w := strings.ToLower(cur.String())
		out[w]++
		cur.Reset()
	}
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// lexicalScore is the no-embeddings fallback. Counts the share of
// query tokens that appear in the record's content + tags. Range
// [0, 1]. Not BM25 — we don't have a corpus statistic — but it's
// cheap, deterministic, and correlates well enough with relevance
// for the small per-instance memory sizes we care about.
func lexicalScore(queryTokens map[string]int, r MemoryRecord) float64 {
	if len(queryTokens) == 0 {
		return 0
	}
	doc := r.Content
	if len(r.Tags) > 0 {
		doc += " " + strings.Join(r.Tags, " ")
	}
	docTokens := tokenize(doc)
	hits := 0
	total := 0
	for q, qcount := range queryTokens {
		total += qcount
		if dc, ok := docTokens[q]; ok {
			if dc < qcount {
				hits += dc
			} else {
				hits += qcount
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

var memoryRecallStopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "been": {},
	"but": {}, "by": {}, "do": {}, "for": {}, "from": {}, "had": {}, "has": {}, "have": {},
	"he": {}, "her": {}, "his": {}, "i": {}, "if": {}, "in": {}, "is": {}, "it": {}, "its": {},
	"me": {}, "my": {}, "of": {}, "on": {}, "or": {}, "our": {}, "she": {}, "that": {}, "the": {},
	"their": {}, "them": {}, "they": {}, "this": {}, "to": {}, "was": {}, "we": {}, "were": {},
	"what": {}, "when": {}, "where": {}, "which": {}, "who": {}, "will": {}, "with": {}, "you": {},
	"your": {},
}

func meaningfulMemoryTokens(tokens map[string]int) map[string]int {
	filtered := make(map[string]int, len(tokens))
	for token, count := range tokens {
		if len(token) < 3 {
			continue
		}
		if _, stopword := memoryRecallStopwords[token]; stopword {
			continue
		}
		filtered[token] = count
	}
	return filtered
}

func normalizeMemoryContent(content string) string {
	return strings.ToLower(strings.Join(strings.Fields(content), " "))
}

// newULID returns a sortable, globally-unique id. We don't pull in a
// ULID library — a hex-encoded 64-bit timestamp + 64-bit random gives
// us the same monotonic-by-time + collision-free properties for our
// volume. Format: <hex_ts><hex_rand> = 32 chars total.
func newULID() string {
	ts := time.Now().UTC().UnixNano()
	var rnd [8]byte
	_, _ = rand.Read(rnd[:])
	return fmt.Sprintf("%016x%s", ts, hex.EncodeToString(rnd[:]))
}

func formatAge(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd", int(d.Hours()/24))
	}
}
