package core

import (
	"sort"
	"strings"
	"sync"
)

// ToolIndex catalogs every MCP tool known to the process by metadata
// (server, name, description, schema, no_spawn) and supports cheap
// keyword search. It is the search surface for `search_tools` and the
// per-turn preload, replacing the old main-access/catalog split.
//
// The index is a metadata mirror of what's in ToolRegistry. The
// registry remains the source of truth for handlers and dispatch;
// the index is just a fast read-side view optimised for ranking.
//
// Why a separate structure instead of querying the registry directly:
//   - The registry stores tools by name, not by server; per-server
//     enumeration (for spawn-time preload) would mean an O(N) scan
//     every time.
//   - Search wants tokenized text; precomputing it once at add-time
//     keeps the per-query cost to a sort.
//   - The registry will eventually want to evict tools (uninstall an
//     app); a separate index keeps that bookkeeping local.
type ToolIndex struct {
	mu      sync.RWMutex
	entries []IndexEntry
}

// IndexEntry is one tool's worth of searchable metadata.
type IndexEntry struct {
	Server      string
	Name        string // fully-qualified (e.g. "storage_files_upload")
	Description string
	NoSpawn     bool // sub-threads cannot see this tool in search
	// tokens is the lowercased, deduplicated set of search terms
	// (name segments + description words). Precomputed at Add() time.
	tokens map[string]int
}

// NewToolIndex returns an empty index.
func NewToolIndex() *ToolIndex {
	return &ToolIndex{}
}

// Add registers a server's tools. Replaces any prior entries for the
// same server name so reconnect or hot-reload semantics work cleanly.
func (ix *ToolIndex) Add(server string, tools []mcpToolDef, noSpawn bool) {
	if ix == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	// Drop existing entries from this server first
	filtered := ix.entries[:0]
	for _, e := range ix.entries {
		if e.Server != server {
			filtered = append(filtered, e)
		}
	}
	ix.entries = filtered
	for _, t := range tools {
		full := server + "_" + t.Name
		e := IndexEntry{
			Server:      server,
			Name:        full,
			Description: t.Description,
			NoSpawn:     noSpawn,
			tokens:      indexTokens(full + " " + t.Description),
		}
		ix.entries = append(ix.entries, e)
	}
}

// Remove drops every entry for the named server. Used when an app
// uninstalls or an MCP disconnects.
func (ix *ToolIndex) Remove(server string) {
	if ix == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	filtered := ix.entries[:0]
	for _, e := range ix.entries {
		if e.Server != server {
			filtered = append(filtered, e)
		}
	}
	ix.entries = filtered
}

// Get returns the entry for a fully-qualified tool name, if present.
func (ix *ToolIndex) Get(name string) (IndexEntry, bool) {
	if ix == nil {
		return IndexEntry{}, false
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	for _, e := range ix.entries {
		if e.Name == name {
			return e, true
		}
	}
	return IndexEntry{}, false
}

// ToolsForServer returns every tool name a server contributes. Used by
// spawn-time preload (SpawnOpts.MCPNames) to seed a child thread's
// activeTools with the full surface of the listed servers.
func (ix *ToolIndex) ToolsForServer(server string) []string {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	var out []string
	for _, e := range ix.entries {
		if e.Server == server {
			out = append(out, e.Name)
		}
	}
	sort.Strings(out)
	return out
}

// ToolCountByServer returns name → count of indexed tools. Used by
// the system prompt's [AVAILABLE MCP SERVERS] block so the LLM sees
// which servers exist and how many tools each contributes without
// the full schemas appearing in context.
func (ix *ToolIndex) ToolCountByServer() map[string]int {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := map[string]int{}
	for _, e := range ix.entries {
		out[e.Server]++
	}
	return out
}

// computeMCPCatalog projects the index into the legacy
// MCPServerInfo shape consumed by buildSystemPrompt. Kept as a free
// function so the prompt builder doesn't need to learn about the
// index type yet.
func computeMCPCatalog(ix *ToolIndex) []MCPServerInfo {
	if ix == nil {
		return nil
	}
	counts := ix.ToolCountByServer()
	out := make([]MCPServerInfo, 0, len(counts))
	for _, name := range ix.Servers() {
		out = append(out, MCPServerInfo{Name: name, ToolCount: counts[name]})
	}
	return out
}

// Servers returns every server name currently indexed.
func (ix *ToolIndex) Servers() []string {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	seen := map[string]struct{}{}
	for _, e := range ix.entries {
		seen[e.Server] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

// Search returns up to k entries matching the query, ranked. When
// allowNoSpawn is false, no_spawn entries are filtered out — that
// path is used from sub-threads, which must not discover gateway or
// channels tools they have no business calling.
func (ix *ToolIndex) Search(query string, k int, allowNoSpawn bool) []IndexEntry {
	if ix == nil || k <= 0 {
		return nil
	}
	terms := indexQueryTokens(query)
	if len(terms) == 0 {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()

	type scored struct {
		entry IndexEntry
		score float64
	}
	var hits []scored
	for _, e := range ix.entries {
		if !allowNoSpawn && e.NoSpawn {
			continue
		}
		s := scoreEntry(terms, e)
		if s > 0 {
			hits = append(hits, scored{e, s})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score == hits[j].score {
			return hits[i].entry.Name < hits[j].entry.Name
		}
		return hits[i].score > hits[j].score
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	out := make([]IndexEntry, len(hits))
	for i, h := range hits {
		out[i] = h.entry
	}
	return out
}

// scoreEntry ranks a single entry against the query terms. The
// scoring is intentionally simple — keyword presence with a small
// boost for name hits over description hits — because the index sits
// at ~100-500 tools where BM25 vs naive TF makes no observable recall
// difference. If that changes (10k+ tools, ambiguous queries), swap
// in a real BM25 here without touching callers.
func scoreEntry(terms []string, e IndexEntry) float64 {
	name := strings.ToLower(e.Name)
	score := 0.0
	for _, t := range terms {
		if cnt, ok := e.tokens[t]; ok {
			weight := 1.0
			if strings.Contains(name, t) {
				weight = 2.5 // name hits are stronger signal
			}
			score += weight * float64(cnt)
		}
	}
	return score
}

// indexTokens lowercases s, splits on non-alphanumeric, and returns a
// term-frequency map. Tokens shorter than 2 chars are dropped (noise:
// "a", "I", isolated punctuation). Named distinctly from memory.go's
// tokenize() to avoid collision; the rules differ (we keep underscores
// out, memory.go keeps them in) so a shared helper would obscure intent.
func indexTokens(s string) map[string]int {
	out := map[string]int{}
	var cur strings.Builder
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		tok := strings.ToLower(cur.String())
		cur.Reset()
		if len(tok) < 2 {
			return
		}
		if stopWord(tok) {
			return
		}
		out[tok]++
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

// indexQueryTokens returns the deduplicated token list for a query
// string. Order doesn't matter for scoring; uniqueness avoids
// over-weighting a single repeated term in a verbose user message.
func indexQueryTokens(s string) []string {
	m := indexTokens(s)
	out := make([]string, 0, len(m))
	for t := range m {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// stopWord trims very common English words so they don't dominate
// short queries ("the tool that sends X" → ["tool","sends","X"]).
// Kept tiny on purpose — over-aggressive stopword lists hurt recall
// for tool descriptions, which are themselves terse.
func stopWord(t string) bool {
	switch t {
	case "the", "and", "for", "with", "that", "this", "from", "into",
		"can", "will", "you", "your", "are", "was", "but", "not", "all",
		"any", "use", "uses", "used", "using":
		return true
	}
	return false
}
