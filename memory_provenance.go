package core

import (
	"fmt"
	"strings"
	"time"
)

// Sources are runtime-generated references, not claims of user authority.
// A hash identifies a retained entry even after its JSONL file is rewritten.
type MemorySource struct {
	ThreadID string `json:"thread_id"`
	EntryID  string `json:"entry_id"`
	Sequence int64  `json:"sequence,omitempty"`
	Role     string `json:"role"`
}

type MemoryMetadata struct {
	Scope   string         `json:"scope,omitempty"`
	Sources []MemorySource `json:"sources,omitempty"`
}

func memoryVisible(r MemoryRecord, thread string) bool {
	return r.Scope == "" || r.Scope == thread
}

func memoryFromSource(r MemoryRecord, thread string) bool {
	if r.Scope == thread {
		return true
	}
	for _, source := range r.Sources {
		if source.ThreadID == thread {
			return true
		}
	}
	return false
}

func mergeMemorySources(a, b []MemorySource) []MemorySource {
	out := append([]MemorySource(nil), a...)
	for _, source := range b {
		found := false
		for _, existing := range out {
			if existing == source {
				found = true
				break
			}
		}
		if !found {
			out = append(out, source)
		}
	}
	return out
}

func (ms *MemoryStore) forgottenLocked(thread string) bool {
	if thread == "" {
		return false
	}
	for _, record := range ms.records {
		if record.ForgottenThread == thread {
			return true
		}
	}
	return false
}

func (ms *MemoryStore) SourceForgotten(thread string) bool {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.forgottenLocked(thread)
}

// ForgetSource is logical deletion, not erasure of the journal or transcripts.
// The deny marker and tombstones commit together and prevent later ingestion.
func (ms *MemoryStore) ForgetSource(thread, reason string) (int, error) {
	if strings.TrimSpace(thread) == "" || strings.TrimSpace(reason) == "" {
		return 0, fmt.Errorf("thread_id and reason required")
	}
	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.ensureActiveLocked()
	var records []MemoryRecord
	for _, rec := range ms.active {
		match := rec.Scope == thread
		for _, source := range rec.Sources {
			match = match || source.ThreadID == thread
		}
		if match {
			records = append(records, MemoryRecord{ID: newULID(), TS: time.Now().UTC(), Tombstone: true, IDTarget: rec.ID, Reason: reason})
		}
	}
	count := len(records)
	if !ms.forgottenLocked(thread) {
		records = append([]MemoryRecord{{ID: newULID(), TS: time.Now().UTC(), ForgottenThread: thread, Reason: reason}}, records...)
	}
	if len(records) == 0 {
		return 0, nil
	}
	if err := ms.appendRecordsLocked(records...); err != nil {
		return 0, err
	}
	return count, nil
}

func (ms *MemoryStore) searchForThread(query, thread string, limit int) []MemoryRecord {
	if limit <= 0 {
		limit = 10
	}
	ranked := ms.scoreActive(query, scoreOpts{useEmbedding: ms.backend != nil, thread: &thread})
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}
	out := make([]MemoryRecord, len(ranked))
	for i, r := range ranked {
		out[i] = r.rec
	}
	return out
}
