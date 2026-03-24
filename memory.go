package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sync"
	"time"
)

const (
	embeddingURL   = "https://api.fireworks.ai/inference/v1/embeddings"
	embeddingModel = "nomic-ai/nomic-embed-text-v1.5"
	embeddingDim   = 768
	memoryFile     = "memory.jsonl"
	maxMemories    = 5000
	recallTopN     = 5
)

type MemoryEntry struct {
	Text      string    `json:"text"`
	Time      time.Time `json:"time"`
	Session   string    `json:"session"`
	Embedding []float64 `json:"embedding"`
}

type MemoryStore struct {
	mu      sync.RWMutex
	entries []MemoryEntry
	apiKey  string
	session string
	path    string
}

func NewMemoryStore(apiKey string) *MemoryStore {
	ms := &MemoryStore{
		apiKey:  apiKey,
		session: fmt.Sprintf("%d", time.Now().UnixNano()),
		path:    memoryFile,
	}
	ms.load()
	return ms
}

func (ms *MemoryStore) load() {
	data, err := os.ReadFile(ms.path)
	if err != nil {
		return
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		var entry MemoryEntry
		if err := dec.Decode(&entry); err != nil {
			continue
		}
		ms.entries = append(ms.entries, entry)
	}
}

func (ms *MemoryStore) save(entry MemoryEntry) error {
	f, err := os.OpenFile(ms.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(entry)
}

func (ms *MemoryStore) Store(text string) error {
	embedding, err := ms.embed(text)
	if err != nil {
		return fmt.Errorf("embedding failed: %w", err)
	}

	entry := MemoryEntry{
		Text:      text,
		Time:      time.Now(),
		Session:   ms.session,
		Embedding: embedding,
	}

	ms.mu.Lock()
	ms.entries = append(ms.entries, entry)
	// Trim old entries if over limit
	if len(ms.entries) > maxMemories {
		ms.entries = ms.entries[len(ms.entries)-maxMemories:]
	}
	ms.mu.Unlock()

	return ms.save(entry)
}

func (ms *MemoryStore) Retrieve(query string, n int) []MemoryEntry {
	if len(ms.entries) == 0 {
		return nil
	}

	queryEmb, err := ms.embed(query)
	if err != nil {
		return nil
	}

	ms.mu.RLock()
	defer ms.mu.RUnlock()

	type scored struct {
		entry MemoryEntry
		score float64
	}

	var results []scored
	for _, e := range ms.entries {
		sim := cosineSimilarity(queryEmb, e.Embedding)
		results = append(results, scored{entry: e, score: sim})
	}

	// Sort by score descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if n > len(results) {
		n = len(results)
	}

	out := make([]MemoryEntry, n)
	for i := 0; i < n; i++ {
		out[i] = results[i].entry
	}
	return out
}

func (ms *MemoryStore) Delete(index int) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if index < 0 || index >= len(ms.entries) {
		return
	}
	ms.entries = append(ms.entries[:index], ms.entries[index+1:]...)
	ms.rewrite()
}

func (ms *MemoryStore) Count() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return len(ms.entries)
}

func (ms *MemoryStore) Recent(n int) []MemoryEntry {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	if n > len(ms.entries) {
		n = len(ms.entries)
	}
	if n == 0 {
		return nil
	}
	out := make([]MemoryEntry, n)
	copy(out, ms.entries[len(ms.entries)-n:])
	return out
}

func (ms *MemoryStore) rewrite() {
	f, err := os.Create(ms.path)
	if err != nil {
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, e := range ms.entries {
		enc.Encode(e)
	}
}

func (ms *MemoryStore) BuildContext(entries []MemoryEntry) string {
	if len(entries) == 0 {
		return ""
	}
	var buf bytes.Buffer
	buf.WriteString("[memories]\n")
	for _, e := range entries {
		age := time.Since(e.Time)
		buf.WriteString(fmt.Sprintf("- (%s ago) %s\n", formatAge(age), e.Text))
	}
	return buf.String()
}

// embed calls the Fireworks embedding API
type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (ms *MemoryStore) embed(text string) ([]float64, error) {
	reqBody, _ := json.Marshal(embeddingRequest{
		Model: embeddingModel,
		Input: text,
	})

	req, err := http.NewRequest("POST", embeddingURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+ms.apiKey)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API error %d: %s", resp.StatusCode, string(body))
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
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
