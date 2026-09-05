package core

import (
	"container/heap"
	"sort"
)

type memoryHeap []scoredRec

func (h memoryHeap) Len() int { return len(h) }
func betterMemory(a, b scoredRec) bool {
	if a.score == b.score {
		return a.rec.ID < b.rec.ID
	}
	return a.score > b.score
}
func (h memoryHeap) Less(i, j int) bool { return betterMemory(h[j], h[i]) }
func (h memoryHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *memoryHeap) Push(v any)        { *h = append(*h, v.(scoredRec)) }
func (h *memoryHeap) Pop() any          { v := (*h)[len(*h)-1]; *h = (*h)[:len(*h)-1]; return v }
func selectMemoryTopK(items []scoredRec, n int) []scoredRec {
	// Deduplicate before selection; keeping only N IDs first could discard a
	// distinct record when several high-ranked IDs contain identical content.
	best := map[string]scoredRec{}
	for _, item := range items {
		key := normalizeMemoryContent(item.rec.Content)
		old, ok := best[key]
		if !ok || betterMemory(item, old) {
			best[key] = item
		}
	}
	h := memoryHeap{}
	for _, item := range best {
		if len(h) < n {
			heap.Push(&h, item)
		} else if betterMemory(item, h[0]) {
			h[0] = item
			heap.Fix(&h, 0)
		}
	}
	sort.Slice(h, func(i, j int) bool { return betterMemory(h[i], h[j]) })
	return []scoredRec(h)
}
