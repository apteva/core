package main

import (
	"math"
	"os"
	"testing"
	"time"
)

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float64{1, 2, 3}
	sim := cosineSimilarity(a, a)
	if math.Abs(sim-1.0) > 1e-9 {
		t.Errorf("identical vectors should have similarity 1.0, got %f", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{0, 1, 0}
	sim := cosineSimilarity(a, b)
	if math.Abs(sim) > 1e-9 {
		t.Errorf("orthogonal vectors should have similarity 0, got %f", sim)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{-1, -2, -3}
	sim := cosineSimilarity(a, b)
	if math.Abs(sim-(-1.0)) > 1e-9 {
		t.Errorf("opposite vectors should have similarity -1.0, got %f", sim)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float64{0, 0, 0}
	b := []float64{1, 2, 3}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("zero vector should give similarity 0, got %f", sim)
	}
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("different length vectors should give 0, got %f", sim)
	}
}

func TestCosineSimilarity_Empty(t *testing.T) {
	sim := cosineSimilarity(nil, nil)
	if sim != 0 {
		t.Errorf("empty vectors should give 0, got %f", sim)
	}
}

func TestFormatAge(t *testing.T) {
	tests := []struct {
		d        time.Duration
		expected string
	}{
		{5 * time.Second, "5s"},
		{30 * time.Second, "30s"},
		{5 * time.Minute, "5m"},
		{90 * time.Minute, "1h"},
		{3 * time.Hour, "3h"},
		{25 * time.Hour, "1d"},
		{72 * time.Hour, "3d"},
	}
	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			result := formatAge(tt.d)
			if result != tt.expected {
				t.Errorf("formatAge(%v) = %q, want %q", tt.d, result, tt.expected)
			}
		})
	}
}

func TestMemoryStore_StoreAndRetrieve_WithMockEmbeddings(t *testing.T) {
	// Create a temp file for the memory store
	tmp, err := os.CreateTemp("", "memory_test_*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())
	tmp.Close()

	ms := &MemoryStore{
		apiKey:  "unused",
		session: "test-session",
		path:    tmp.Name(),
	}

	// Manually add entries with known embeddings (skip API call)
	ms.mu.Lock()
	ms.entries = []MemoryEntry{
		{Text: "User asked about Go concurrency", Time: time.Now().Add(-10 * time.Minute), Session: "s1", Embedding: []float64{1, 0, 0}},
		{Text: "User asked about Python async", Time: time.Now().Add(-5 * time.Minute), Session: "s1", Embedding: []float64{0.9, 0.1, 0}},
		{Text: "User asked about cooking recipes", Time: time.Now().Add(-1 * time.Minute), Session: "s1", Embedding: []float64{0, 0, 1}},
	}
	ms.mu.Unlock()

	// Test Count
	if ms.Count() != 3 {
		t.Errorf("expected count 3, got %d", ms.Count())
	}

	// Test Recent
	recent := ms.Recent(2)
	if len(recent) != 2 {
		t.Fatalf("expected 2 recent, got %d", len(recent))
	}
	if recent[0].Text != "User asked about Python async" {
		t.Errorf("unexpected recent[0]: %q", recent[0].Text)
	}
	if recent[1].Text != "User asked about cooking recipes" {
		t.Errorf("unexpected recent[1]: %q", recent[1].Text)
	}

	// Test Recent with n > count
	all := ms.Recent(100)
	if len(all) != 3 {
		t.Errorf("expected 3, got %d", len(all))
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	tmp, err := os.CreateTemp("", "memory_del_*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())
	tmp.Close()

	ms := &MemoryStore{
		apiKey:  "unused",
		session: "test",
		path:    tmp.Name(),
	}

	ms.mu.Lock()
	ms.entries = []MemoryEntry{
		{Text: "first", Time: time.Now(), Session: "s", Embedding: []float64{1, 0}},
		{Text: "second", Time: time.Now(), Session: "s", Embedding: []float64{0, 1}},
		{Text: "third", Time: time.Now(), Session: "s", Embedding: []float64{1, 1}},
	}
	ms.mu.Unlock()

	ms.Delete(1) // delete "second"
	if ms.Count() != 2 {
		t.Fatalf("expected 2 after delete, got %d", ms.Count())
	}
	remaining := ms.Recent(2)
	if remaining[0].Text != "first" || remaining[1].Text != "third" {
		t.Errorf("unexpected entries after delete: %q, %q", remaining[0].Text, remaining[1].Text)
	}
}

func TestMemoryStore_Delete_OutOfBounds(t *testing.T) {
	ms := &MemoryStore{path: "/dev/null"}
	ms.entries = []MemoryEntry{{Text: "only"}}
	ms.Delete(-1)
	ms.Delete(5)
	if ms.Count() != 1 {
		t.Error("out of bounds delete should not remove anything")
	}
}

func TestMemoryStore_BuildContext(t *testing.T) {
	ms := &MemoryStore{}
	entries := []MemoryEntry{
		{Text: "Some memory", Time: time.Now().Add(-5 * time.Minute)},
		{Text: "Another one", Time: time.Now().Add(-2 * time.Hour)},
	}
	ctx := ms.BuildContext(entries)
	if ctx == "" {
		t.Fatal("expected non-empty context")
	}
	if !searchStr(ctx, "[memories]") {
		t.Error("expected [memories] header")
	}
	if !searchStr(ctx, "Some memory") {
		t.Error("expected memory text in context")
	}
	if !searchStr(ctx, "5m ago") {
		t.Error("expected age in context")
	}
}

func TestMemoryStore_BuildContext_Empty(t *testing.T) {
	ms := &MemoryStore{}
	ctx := ms.BuildContext(nil)
	if ctx != "" {
		t.Error("expected empty context for nil entries")
	}
}

func TestMemoryStore_LoadSave_Roundtrip(t *testing.T) {
	tmp, err := os.CreateTemp("", "memory_rt_*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())
	tmp.Close()

	ms := &MemoryStore{
		apiKey:  "unused",
		session: "test",
		path:    tmp.Name(),
	}

	entry := MemoryEntry{
		Text:      "Test memory",
		Time:      time.Now().Truncate(time.Millisecond),
		Session:   "test",
		Embedding: []float64{0.1, 0.2, 0.3},
	}
	if err := ms.save(entry); err != nil {
		t.Fatal(err)
	}

	// Load into a new store
	ms2 := &MemoryStore{path: tmp.Name()}
	ms2.load()
	if len(ms2.entries) != 1 {
		t.Fatalf("expected 1 entry after load, got %d", len(ms2.entries))
	}
	if ms2.entries[0].Text != "Test memory" {
		t.Errorf("unexpected text: %q", ms2.entries[0].Text)
	}
	if len(ms2.entries[0].Embedding) != 3 {
		t.Errorf("expected 3-dim embedding, got %d", len(ms2.entries[0].Embedding))
	}
}
