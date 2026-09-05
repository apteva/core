package core

import (
	"crypto/sha256"
	"fmt"
	"time"
)

type embeddingCacheEntry struct {
	embedding []float64
	expires   time.Time
}

// A bounded content-addressed cache avoids repeated standing-directive calls.
// Failed embedding services cool down briefly; lexical retrieval stays usable.
func (ms *MemoryStore) cachedQueryEmbedding(text string) ([]float64, error) {
	key := sha256.Sum256([]byte(text))
	now := time.Now()
	ms.embeddingMu.Lock()
	if entry, ok := ms.embeddingCache[key]; ok && now.Before(entry.expires) {
		out := append([]float64(nil), entry.embedding...)
		ms.embeddingMu.Unlock()
		return out, nil
	}
	if now.Before(ms.embeddingRetryAt) {
		ms.embeddingMu.Unlock()
		return nil, fmt.Errorf("embedding backend cooling down")
	}
	ms.embeddingMu.Unlock()
	embedding, err := ms.embed(text)
	ms.embeddingMu.Lock()
	defer ms.embeddingMu.Unlock()
	if err != nil {
		ms.embeddingRetryAt = now.Add(30 * time.Second)
		return nil, err
	}
	if ms.embeddingCache == nil || len(ms.embeddingCache) >= 128 {
		ms.embeddingCache = map[[32]byte]embeddingCacheEntry{}
	}
	ms.embeddingCache[key] = embeddingCacheEntry{append([]float64(nil), embedding...), now.Add(10 * time.Minute)}
	return embedding, nil
}
