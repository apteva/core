package core

import (
	"container/heap"
	"math"
	"sort"
	"strings"
	"time"
)

type recallCandidate struct {
	scoredRec
	key string
}
type recallHeap struct {
	values    []recallCandidate
	positions map[string]int
}

func (h recallHeap) Len() int { return len(h.values) }
func (h recallHeap) Less(i, j int) bool {
	return betterMemory(h.values[j].scoredRec, h.values[i].scoredRec)
}
func (h recallHeap) Swap(i, j int) {
	h.values[i], h.values[j] = h.values[j], h.values[i]
	h.positions[h.values[i].key] = i
	h.positions[h.values[j].key] = j
}
func (h *recallHeap) Push(v any) {
	item := v.(recallCandidate)
	h.positions[item.key] = len(h.values)
	h.values = append(h.values, item)
}
func (h *recallHeap) Pop() any {
	item := h.values[len(h.values)-1]
	delete(h.positions, item.key)
	h.values = h.values[:len(h.values)-1]
	return item
}

func (ms *MemoryStore) recallIndexed(queries []string, n int) []MemoryRecallMatch {
	if n <= 0 {
		n = 5
	}
	type queryPlan struct {
		tokens    []map[string]int
		embedding []float64
	}
	plans := make([]queryPlan, 0, len(queries))
	seen := map[string]bool{}
	all := false
	for _, query := range queries {
		query = strings.TrimSpace(query)
		if query == "" || seen[query] {
			continue
		}
		seen[query] = true
		plan := queryPlan{tokens: recallQueryTokenSets(query)}
		if ms.backend != nil {
			plan.embedding, _ = ms.cachedQueryEmbedding(query)
			if plan.embedding != nil {
				all = true
			}
		}
		plans = append(plans, plan)
	}
	if len(plans) == 0 {
		return nil
	}
	ms.mu.Lock()
	ms.ensureActiveLocked()
	if ms.lexical == nil || ms.normalized == nil {
		ms.lexical = map[string]map[string]int{}
		ms.postings = map[string]map[string]bool{}
		ms.normalized = map[string]string{}
		for _, r := range ms.active {
			ms.indexLocked(r)
		}
	}
	ms.mu.Unlock()
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	candidates := map[string]bool{}
	if !all {
		for _, plan := range plans {
			for _, tokens := range plan.tokens {
				for token := range tokens {
					for id := range ms.postings[token] {
						candidates[id] = true
					}
				}
			}
		}
	}
	selected := &recallHeap{positions: map[string]int{}}
	now := time.Now().UTC()
	score := func(id string) {
		r := ms.active[id]
		signal := 0.0
		for _, plan := range plans {
			var s float64
			if plan.embedding != nil && len(plan.embedding) == len(r.Embedding) {
				s = cosineSimilarity(plan.embedding, r.Embedding)
				if s < minEmbeddingRecallSimilarity {
					continue
				}
			} else {
				s = lexicalTokenScore(plan.tokens, ms.lexical[id])
			}
			if s > 0 {
				signal = 1 - (1-signal)*(1-s)
			}
		}
		if signal <= 0 {
			return
		}
		w := r.Weight
		if w <= 0 {
			w = .05
		}
		candidate := recallCandidate{scoredRec: scoredRec{rec: r, signal: signal, score: signal * w * math.Pow(.5, now.Sub(r.TS).Hours()/24/memoryHalfLifeDays)}, key: ms.normalized[id]}
		if i, exists := selected.positions[candidate.key]; exists {
			if betterMemory(candidate.scoredRec, selected.values[i].scoredRec) {
				selected.values[i] = candidate
				heap.Fix(selected, i)
			}
			return
		}
		if selected.Len() < n {
			heap.Push(selected, candidate)
		} else if betterMemory(candidate.scoredRec, selected.values[0].scoredRec) {
			delete(selected.positions, selected.values[0].key)
			selected.values[0] = candidate
			selected.positions[candidate.key] = 0
			heap.Fix(selected, 0)
		}
	}
	if all {
		for id := range ms.active {
			score(id)
		}
	} else {
		for id := range candidates {
			score(id)
		}
	}
	sort.Slice(selected.values, func(i, j int) bool { return betterMemory(selected.values[i].scoredRec, selected.values[j].scoredRec) })
	out := make([]MemoryRecallMatch, len(selected.values))
	for i, item := range selected.values {
		out[i] = MemoryRecallMatch{Record: item.rec, Score: item.score, Signal: item.signal}
	}
	return out
}
