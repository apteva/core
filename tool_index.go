package core

import (
	"fmt"
	"math"
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
	mu                sync.RWMutex
	entries           []IndexEntry
	byName            map[string]int
	serverPrefixes    map[string]bool
	aliases           map[string]string
	revision          uint64
	diagnosedRevision uint64
	changed           chan struct{}
}

// IndexEntry is one tool's worth of searchable metadata.
type IndexEntry struct {
	Server      string
	LocalName   string // raw MCP name (for example "send")
	Name        string // fully-qualified (e.g. "storage_files_upload")
	Description string
	NoSpawn     bool // sub-threads cannot see this tool in search
	LoadMode    ToolLoadMode
	Match       string
	Score       float64
	// Full-description term frequencies and separate name terms are computed
	// at registration. Descriptive ranking uses BM25 with a distinct name boost.
	tokens     map[string]int
	nameTokens map[string]int
	tokenCount int
}

// NewToolIndex returns an empty index.
func NewToolIndex() *ToolIndex {
	return &ToolIndex{}
}

// Add registers a server's tools. Replaces any prior entries for the
// same server name so reconnect or hot-reload semantics work cleanly.
func (ix *ToolIndex) Add(server string, tools []mcpToolDef, noSpawn bool, loading ...*MCPToolLoadingConfig) {
	if ix == nil {
		return
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.changedLocked()
	// Drop existing entries from this server first
	filtered := ix.entries[:0]
	for _, e := range ix.entries {
		if e.Server != server {
			filtered = append(filtered, e)
		}
	}
	ix.entries = filtered
	for alias, target := range ix.aliases {
		found := false
		for _, e := range ix.entries {
			if e.Name == target {
				found = true
				break
			}
		}
		if !found {
			delete(ix.aliases, alias)
		}
	}
	cfg := MCPServerConfig{Name: server}
	if len(loading) > 0 {
		cfg.ToolLoading = loading[0]
	}
	for _, t := range tools {
		full := server + "_" + t.Name
		e := IndexEntry{
			Server:      server,
			LocalName:   t.Name,
			Name:        full,
			Description: t.Description,
			NoSpawn:     noSpawn,
			LoadMode:    cfg.toolLoadMode(t.Name),
			tokens:      indexTokens(t.Description),
			nameTokens:  indexTokens(full),
		}
		for _, count := range e.tokens {
			e.tokenCount += count
		}
		ix.entries = append(ix.entries, e)
	}
	ix.rebuildNamesLocked()
}

// UpdatePolicy changes only prompt visibility metadata. MCP connections and
// registered handlers remain untouched, which is required for host-owned
// no_spawn servers such as Channels that cannot safely reconnect while their
// management request is in flight.
func (ix *ToolIndex) UpdatePolicy(server string, noSpawn bool, loading *MCPToolLoadingConfig) {
	if ix == nil {
		return
	}
	cfg := MCPServerConfig{Name: server, ToolLoading: loading}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.changedLocked()
	for i := range ix.entries {
		if ix.entries[i].Server != server {
			continue
		}
		ix.entries[i].NoSpawn = noSpawn
		ix.entries[i].LoadMode = cfg.toolLoadMode(ix.entries[i].LocalName)
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
	ix.changedLocked()
	filtered := ix.entries[:0]
	for _, e := range ix.entries {
		if e.Server != server {
			filtered = append(filtered, e)
		}
	}
	ix.entries = filtered
	for alias, target := range ix.aliases {
		found := false
		for _, e := range ix.entries {
			if e.Name == target {
				found = true
				break
			}
		}
		if !found {
			delete(ix.aliases, alias)
		}
	}
	ix.rebuildNamesLocked()
}

// Get returns the entry for a fully-qualified tool name, if present.
func (ix *ToolIndex) Get(name string) (IndexEntry, bool) {
	if ix == nil {
		return IndexEntry{}, false
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	if index, ok := ix.byName[strings.ToLower(name)]; ok && ix.entries[index].Name == name {
		return ix.entries[index], true
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

// ToolPolicyCountsByServer returns per-server counts keyed by normalized load
// mode. It feeds the prompt's compact catalog without exposing index internals.
func (ix *ToolIndex) ToolPolicyCountsByServer() map[string]map[ToolLoadMode]int {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := map[string]map[ToolLoadMode]int{}
	for _, e := range ix.entries {
		if out[e.Server] == nil {
			out[e.Server] = map[ToolLoadMode]int{}
		}
		out[e.Server][normalizeToolLoadMode(e.LoadMode)]++
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
	policyCounts := ix.ToolPolicyCountsByServer()
	out := make([]MCPServerInfo, 0, len(counts))
	for _, name := range ix.Servers() {
		out = append(out, MCPServerInfo{
			Name:          name,
			ToolCount:     counts[name],
			AlwaysCount:   policyCounts[name][ToolLoadAlways],
			AutoCount:     policyCounts[name][ToolLoadAuto],
			DeferredCount: policyCounts[name][ToolLoadDeferred],
		})
	}
	return out
}

// BaselineNames returns schemas that are visible without activation for the
// current global mode. Explicit policy wins over the global threshold:
// always is always present, deferred always requires activation, and auto
// follows eagerMode. no_spawn remains an authorization boundary.
func (ix *ToolIndex) BaselineNames(eagerMode, allowNoSpawn bool) []string {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := make([]string, 0, len(ix.entries))
	for _, e := range ix.entries {
		if !allowNoSpawn && e.NoSpawn {
			continue
		}
		mode := normalizeToolLoadMode(e.LoadMode)
		if mode == ToolLoadAlways || (mode == ToolLoadAuto && eagerMode) {
			out = append(out, e.Name)
		}
	}
	sort.Strings(out)
	return out
}

// AlwaysCount reports pinned schemas in the caller's authorized scope.
func (ix *ToolIndex) AlwaysCount(allowNoSpawn bool) int {
	if ix == nil {
		return 0
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	n := 0
	for _, e := range ix.entries {
		if (!allowNoSpawn && e.NoSpawn) || normalizeToolLoadMode(e.LoadMode) != ToolLoadAlways {
			continue
		}
		n++
	}
	return n
}

// DeferredCount reports schemas excluded from the baseline in the current
// global mode. It includes explicit deferred tools plus auto tools while the
// process is in discovery mode.
func (ix *ToolIndex) DeferredCount(eagerMode, allowNoSpawn bool) int {
	if ix == nil {
		return 0
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	n := 0
	for _, e := range ix.entries {
		if !allowNoSpawn && e.NoSpawn {
			continue
		}
		mode := normalizeToolLoadMode(e.LoadMode)
		if mode == ToolLoadDeferred || (mode == ToolLoadAuto && !eagerMode) {
			n++
		}
	}
	return n
}

// AllNames returns every indexed tool's fully-qualified name. Backs
// the APTEVA_EAGER_TOOLS escape hatch, which makes the whole attached
// surface visible every turn (the pre-discovery behaviour). When
// allowNoSpawn is false, no_spawn entries are excluded — a sub-thread
// in eager mode still must not see gateway/channels tools.
func (ix *ToolIndex) AllNames(allowNoSpawn bool) []string {
	if ix == nil {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	out := make([]string, 0, len(ix.entries))
	for _, e := range ix.entries {
		if !allowNoSpawn && e.NoSpawn {
			continue
		}
		out = append(out, e.Name)
	}
	sort.Strings(out)
	return out
}

// Count returns the number of tools currently indexed. Backs the
// APTEVA_TOOL_SEARCH=auto decision: below a threshold the surface is
// small enough that eager-loading beats the search round-trip.
func (ix *ToolIndex) Count() int {
	if ix == nil {
		return 0
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.entries)
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
	return ix.search(query, k, allowNoSpawn, nil)
}

// RegisterAlias registers a discovery alias for an existing canonical name.
// Aliases never become dispatch names and cannot shadow canonical identities.
func (ix *ToolIndex) RegisterAlias(alias, canonical string) error {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.changedLocked()
	alias = strings.ToLower(strings.TrimSpace(alias))
	if alias == "" || len(toolNameTokens(alias)) != 1 || toolNameTokens(alias)[0] != alias {
		return fmt.Errorf("invalid tool alias %q", alias)
	}
	found := false
	for _, e := range ix.entries {
		if strings.EqualFold(e.Name, alias) && e.Name != canonical {
			return fmt.Errorf("alias shadows tool %q", alias)
		}
		if e.Name == canonical {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("unknown alias target %q", canonical)
	}
	if target, exists := ix.aliases[alias]; exists && target != canonical {
		return fmt.Errorf("ambiguous tool alias %q", alias)
	}
	if ix.aliases == nil {
		ix.aliases = map[string]string{}
	}
	ix.aliases[alias] = canonical
	return nil
}

// toolNameTokens preserves qualified names, unlike descriptive tokenization.
func toolNameTokens(query string) []string {
	return strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '-')
	})
}

func (ix *ToolIndex) search(query string, k int, allowNoSpawn bool, authorized func(string) bool) []IndexEntry {
	if ix == nil || k <= 0 {
		return nil
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	allowed := func(e IndexEntry) bool {
		return (allowNoSpawn || !e.NoSpawn) && (authorized == nil || authorized(e.Name))
	}
	requested := map[string]int{}
	explicit := false
	for pos, token := range toolNameTokens(query) {
		canonical := token
		if _, exists := ix.byName[canonical]; !exists {
			if target, ok := ix.aliases[token]; ok {
				canonical = strings.ToLower(target)
				explicit = true
			}
		}
		if _, exists := ix.byName[canonical]; exists {
			if _, seen := requested[canonical]; !seen {
				requested[canonical] = pos
			}
			explicit = true
		}
		for prefix := range ix.serverPrefixes {
			if strings.HasPrefix(token, prefix) {
				explicit = true
				break
			}
		}
	}
	if strings.Contains(strings.TrimSpace(query), "_") && len(strings.Fields(query)) == 1 {
		explicit = true
	}
	if explicit {
		var out []IndexEntry
		for canonical := range requested {
			e := ix.entries[ix.byName[canonical]]
			if allowed(e) {
				e.Match = "exact_or_alias"
				out = append(out, e)
			}
		}
		sort.Slice(out, func(i, j int) bool {
			a, b := requested[strings.ToLower(out[i].Name)], requested[strings.ToLower(out[j].Name)]
			if a != b {
				return a < b
			}
			return out[i].Name < out[j].Name
		})
		if len(out) > k {
			out = out[:k]
		}
		return out
	}
	terms := indexQueryTokens(query)
	if len(terms) == 0 {
		return nil
	}
	var candidates []IndexEntry
	totalLength := 0
	df := map[string]int{}
	for _, e := range ix.entries {
		if !allowed(e) || normalizeToolLoadMode(e.LoadMode) == ToolLoadAlways {
			continue
		}
		candidates = append(candidates, e)
		totalLength += e.tokenCount
		for _, term := range terms {
			if e.tokens[term] > 0 || e.nameTokens[term] > 0 {
				df[term]++
			}
		}
	}
	avgLength := 1.0
	if len(candidates) > 0 {
		avgLength = math.Max(1, float64(totalLength)/float64(len(candidates)))
	}
	type scored struct {
		entry IndexEntry
		score float64
	}
	var hits []scored
	for _, e := range candidates {
		score := 0.0
		for _, term := range terms {
			idf := math.Log1p((float64(len(candidates)-df[term]) + 0.5) / (float64(df[term]) + 0.5))
			if e.nameTokens[term] > 0 {
				score += 8 * idf
			}
			tf := float64(e.tokens[term])
			if tf > 0 {
				score += idf * tf * 2.2 / (tf + 1.2*(0.25+0.75*float64(e.tokenCount)/avgLength))
			}
		}
		if score > 0 {
			hits = append(hits, scored{e, score})
		}
	}
	sort.Slice(hits, func(i, j int) bool {
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
		out[i].Match = "bm25"
		out[i].Score = h.score
	}
	return out
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

func (ix *ToolIndex) replaceServerAliases(server string, aliases map[string]string) {
	ix.mu.Lock()
	for alias, target := range ix.aliases {
		for _, e := range ix.entries {
			if e.Name == target && e.Server == server {
				delete(ix.aliases, alias)
				break
			}
		}
	}
	ix.mu.Unlock()
	for alias, target := range aliases {
		if err := ix.RegisterAlias(alias, server+"_"+target); err != nil {
			logMsg("MCP", err.Error())
		}
	}
}

// Rebuilt only at mount/reconnect. Exact lookup avoids corpus scans and ranking.
func (ix *ToolIndex) rebuildNamesLocked() {
	ix.byName = make(map[string]int, len(ix.entries))
	ix.serverPrefixes = map[string]bool{}
	for i, e := range ix.entries {
		ix.byName[strings.ToLower(e.Name)] = i
		ix.serverPrefixes[strings.ToLower(e.Server)+"_"] = true
	}
}

// Changes broadcasts catalog revisions without one goroutine or queue per
// subscriber. Subscribe before resolving schemas to avoid missing a change.
func (ix *ToolIndex) Changes() <-chan struct{} {
	if ix == nil {
		return nil
	}
	ix.mu.Lock()
	defer ix.mu.Unlock()
	if ix.changed == nil {
		ix.changed = make(chan struct{})
	}
	return ix.changed
}

func (ix *ToolIndex) changedLocked() {
	ix.revision++
	if ix.changed != nil {
		close(ix.changed)
	}
	ix.changed = make(chan struct{})
}
