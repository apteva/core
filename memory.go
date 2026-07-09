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

	// Default decay half-life: a memory's effective weight halves
	// every this many days unless reinforced. 90 days = a memory
	// from 6 months ago contributes 1/4 of its original weight.
	memoryHalfLifeDays = 90.0

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
	mu      sync.RWMutex
	records []MemoryRecord    // full sequence, in insertion order
	byID    map[string]int    // id → index in records
	backend *embeddingBackend // nil → no embeddings, lexical-only
	path    string
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
	out, err := os.OpenFile(ms.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logMsg("MEMORY", fmt.Sprintf("migration: failed to open new file: %v", err))
		return
	}
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
	logMsg("MEMORY", fmt.Sprintf("loaded %d records (%d active)", len(ms.records), ms.activeCount()))
}

// activeCount counts active (non-tombstoned, non-superseded) memories.
// Caller holds ms.mu.RLock() OR is within a write-locked section.
func (ms *MemoryStore) activeCount() int {
	tombstoned, superseded := ms.deadIDs()
	n := 0
	for _, r := range ms.records {
		if r.Tombstone {
			continue
		}
		if tombstoned[r.ID] || superseded[r.ID] {
			continue
		}
		n++
	}
	return n
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
	tombstoned, superseded := ms.deadIDs()
	out := make([]MemoryRecord, 0, len(ms.records))
	for _, r := range ms.records {
		if r.Tombstone {
			continue
		}
		if tombstoned[r.ID] || superseded[r.ID] {
			continue
		}
		out = append(out, r)
	}
	return out
}

// Count returns the number of currently-active memories. Used by
// telemetry and the unconscious's directive ("you have N memories").
func (ms *MemoryStore) Count() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.activeCount()
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
	ms.mu.Lock()
	defer ms.mu.Unlock()
	f, err := os.OpenFile(ms.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(&rec); err != nil {
		return err
	}
	ms.records = append(ms.records, rec)
	if rec.ID != "" {
		ms.byID[rec.ID] = len(ms.records) - 1
	}
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
// freshly-minted ULID. Required for deterministic ids — the platform
// uses this to push skill-as-memory records keyed by the skill's
// primary key, so re-pushing the same skill upserts via Supersede
// rather than creating a duplicate row.
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
	tombstoned, superseded := ms.deadIDs()
	for _, r := range ms.records {
		if r.Tombstone || tombstoned[r.ID] || superseded[r.ID] {
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
	if err := ms.append(newRec); err != nil {
		return "", err
	}
	if err := ms.append(tomb); err != nil {
		// New memory landed; tombstone failed. The supersede field
		// on the new memory still flags the old as superseded at
		// read time (deadIDs collects both signals), so we're
		// not in an inconsistent state — just missing the
		// audit-trail line. Log and move on.
		logMsg("MEMORY", fmt.Sprintf("supersede: tombstone write failed for %s: %v (supersede field still applies)", oldID, err))
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
	if n <= 0 {
		n = 5
	}
	scored := ms.scoreActive(query, scoreOpts{useEmbedding: ms.backend != nil, applyDecay: true})
	if len(scored) > n {
		scored = scored[:n]
	}
	out := make([]MemoryRecord, len(scored))
	for i, s := range scored {
		out[i] = s.rec
	}
	return out
}

type scoreOpts struct {
	useEmbedding bool
	applyDecay   bool
}

type scoredRec struct {
	rec   MemoryRecord
	score float64
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
	queryTokens := tokenize(query)
	now := time.Now().UTC()

	out := make([]scoredRec, 0, len(active))
	for _, r := range active {
		// Signal: prefer embedding cosine when both sides have one
		// of matching dim; fall back to lexical otherwise.
		var signal float64
		if queryEmb != nil && len(r.Embedding) == len(queryEmb) {
			signal = cosineSimilarity(queryEmb, r.Embedding)
			if signal < 0 {
				signal = 0
			}
		} else {
			signal = lexicalScore(queryTokens, r)
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
			rec:   r,
			score: signal * w * decay,
		})
	}

	sort.Slice(out, func(i, j int) bool { return out[i].score > out[j].score })
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
		age := time.Since(r.TS)
		tagStr := ""
		if len(r.Tags) > 0 {
			tagStr = " [" + strings.Join(r.Tags, ",") + "]"
		}
		buf.WriteString(fmt.Sprintf("- (%s ago, w=%.2f)%s %s\n",
			formatAge(age), r.Weight, tagStr, r.Content))
	}
	return buf.String()
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
