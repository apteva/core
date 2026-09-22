package core

import (
	"encoding/json"
	"os"
	"sort"
	"strconv"
	"strings"
)

// defaultToolSearchAutoThreshold: at or below this many indexed
// tools, APTEVA_TOOL_SEARCH=auto loads everything eagerly. The
// original A/B runs put the crossover above ~21 tools (discovery
// lost there) and well below 113 (discovery won big). We ship 60 as
// the default so typical agents stay on the cheaper eager path, while
// large app surfaces (60+ MCP tools, common for media/channel managers)
// use discovery instead of sending every schema on each scheduled wake.
//
// Override at runtime via APTEVA_EAGER_TOOL_LIMIT=<int>. The two
// env vars compose: APTEVA_TOOL_SEARCH=off forces eager regardless
// of count; APTEVA_TOOL_SEARCH=on forces discovery regardless;
// APTEVA_TOOL_SEARCH=auto (default) consults APTEVA_EAGER_TOOL_LIMIT.
const defaultToolSearchAutoThreshold = 60

func eagerToolLimit() int {
	if raw := strings.TrimSpace(os.Getenv("APTEVA_EAGER_TOOL_LIMIT")); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}
	return defaultToolSearchAutoThreshold
}

// activeToolsCap bounds the sticky active-tool set in discovery mode.
// Sticky preload makes the per-turn tool list grow then STABILISE,
// which is what lets the prompt prefix cache. The cap stops an
// unbounded set on a long, topic-hopping conversation from creeping
// toward "everything"; evictActiveToolsLRU trims to ~70% when hit.
const activeToolsCap = 40

// toolSearchMode resolves APTEVA_TOOL_SEARCH: "auto" (default) | "on"
// | "off". "on" forces the discovery model (search + preload); "off"
// makes auto-policy tools eager; "auto" decides by the size of the
// attached surface. Explicit always/deferred policy still wins over this
// process-wide default. Mirrors Anthropic's own ENABLE_TOOL_SEARCH knob.
func toolSearchMode() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("APTEVA_TOOL_SEARCH"))) {
	case "on":
		return "on"
	case "off":
		return "off"
	default:
		return "auto"
	}
}

// isEagerMode is the pure-function form of useEagerTools — takes the
// raw indexed-tool count and returns whether eager applies, honoring
// the APTEVA_TOOL_SEARCH / APTEVA_EAGER_TOOL_LIMIT env vars.
// Pulled out so buildSystemPrompt can decide which [AVAILABLE MCP
// SERVERS] wording to emit without needing a Thinker pointer.
func isEagerMode(toolCount int) bool {
	switch toolSearchMode() {
	case "off":
		return true
	case "on":
		return false
	default:
		return toolCount <= eagerToolLimit()
	}
}

// useEagerTools reports whether this thinker should load every
// attached tool every turn (eager) rather than the discovery model.
// Resolved per-turn so a runtime connect / app-install that pushes the
// surface past the auto threshold flips the mode without a restart.
func (t *Thinker) useEagerTools() bool {
	n := 0
	if t.toolIndex != nil {
		n = t.toolIndex.Count()
	}
	return isEagerMode(n)
}

func poolUsesEagerTools(pool *ProviderPool, toolCount int) bool {
	return isEagerMode(toolCount)
}

// search_tools is the meta-tool every thread has in its scaffolding.
// It searches the process-wide ToolIndex for MCP tools matching the
// query, activates the top k matches for this thread (their full
// schemas appear in the next provider call's tool list), and returns
// a compact JSON summary the LLM can read.
//
// Activation is sticky for the rest of the thread's life (subject to
// context-window compaction): once the LLM has called a tool we keep
// its schema visible so a follow-up call has no search overhead. The
// per-turn BM25 preload (see thinker.go iteration loop) handles the
// transient case — tools the user's current message is about that
// the LLM hasn't yet searched for.

// registerSearchTool adds search_tools to a registry as a Core tool
// so it appears in every thread's tool list regardless of allowlist.
// Called from registerDefaults so main and every sub-thread share
// the same wired-in meta-tool.
func registerSearchTool(r *ToolRegistry) {
	r.Register(&ToolDef{
		Name: "search_tools",
		Description: "Search the authorized MCP catalog by capability and load matching tool schemas. Use one structured query per capability; name servers when known and set access=read_only for inspection-only work. " +
			"Compound legacy queries are split by named server so one large integration cannot crowd out another. Returns bounded matches " +
			"with name + summary; their full schemas become available for you to call on the " +
			"next turn. It cannot expand tools= exact grants or reach ungranted servers. Use search_tools only in a discovery-only turn: you may call multiple " +
			"search_tools in parallel, but do not call any other tool until you have received " +
			"the search results on the next turn.",
		Syntax: `[[search_tools queries='[{"query":"list repository files","servers":["code"],"access":"read_only","limit":5},{"query":"top traffic","servers":["analytics"],"access":"read_only","limit":5}]']]`,
		Rules:  `Provide query or queries (at most 16). access is any, prefer_read, or read_only. k defaults to 5 and total loading caps at 20. Search never widens this thread's spawn capabilities. A lexical miss is not proof that a capability is unavailable. Schemas appear on the next thinking turn — call only search_tools during discovery, then wait for that turn before calling any execution, reporting, messaging, pacing, or completion tool.`,
		Core:   true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "An exact tool name (preferred when known), registered alias, or capability description. Unambiguous local names resolve to canonical tools. Unresolved names are reported separately from descriptive suggestions; suggestions are not equivalent replacements.",
				},
				"queries": map[string]any{
					"type":        "array",
					"description": "Independent capability searches. Each item may be a string or an object with query, servers, access, and limit.",
					"maxItems":    16,
					"items": map[string]any{"oneOf": []any{
						map[string]any{"type": "string"},
						map[string]any{"type": "object", "properties": map[string]any{
							"query":   map[string]any{"type": "string"},
							"servers": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
							"access":  map[string]any{"type": "string", "enum": []string{"any", "prefer_read", "read_only"}},
							"limit":   map[string]any{"type": "integer", "minimum": 1, "maximum": 20},
						}, "required": []string{"query"}},
					}},
				},
				"server_names": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional hard server scope for the scalar query."},
				"access":       map[string]any{"type": "string", "enum": []string{"any", "prefer_read", "read_only"}},
				"k": map[string]any{
					"type":        "integer",
					"description": "Maximum number of tools to load (default 5, max 20).",
				},
				"max_schema_tokens": map[string]any{"type": "integer", "minimum": 1, "maximum": 16000},
			},
			"anyOf": []any{map[string]any{"required": []string{"query"}}, map[string]any{"required": []string{"queries"}}},
		},
		// Handler is intentionally nil — search_tools runs as an inline
		// tool through main/threadToolHandler because it mutates the
		// Thinker's activeTools set, which a registry Handler closure
		// can't reach without smuggling state through globals.
	})
}

// searchToolsResult is the JSON shape returned to the LLM. Compact
// on purpose: each hit's full schema lands in the tool list next
// turn, so the LLM doesn't need a paragraph of description here —
// just enough to confirm it found what it was looking for.
type searchToolsResult struct {
	Unresolved       []string                `json:"unresolved_names,omitempty"`
	Ambiguous        map[string][]string     `json:"ambiguous_names,omitempty"`
	Suggestions      []searchToolHit         `json:"suggestions,omitempty"`
	Error            string                  `json:"error,omitempty"`
	Query            string                  `json:"query"`
	Hits             []searchToolHit         `json:"hits"`
	Loaded           []string                `json:"loaded"` // names whose schemas are now in context
	Note             string                  `json:"note,omitempty"`
	CatalogRevision  uint64                  `json:"catalog_revision,omitempty"`
	AvailableServers []string                `json:"available_servers,omitempty"`
	Results          []DiscoveryIntentResult `json:"results,omitempty"`
	SchemaTokens     int                     `json:"schema_tokens_est,omitempty"`
	SkippedBudget    int                     `json:"skipped_budget,omitempty"`
}

type searchToolHit struct {
	Match   string  `json:"match,omitempty"`
	Score   float64 `json:"score,omitempty"`
	Name    string  `json:"name"`
	Server  string  `json:"server"`
	Summary string  `json:"summary"`
}

// applyPreload searches the directive and events drained this iteration, then
// stickily activates a bounded set. Quiet turns use only the directive. Schema
// changes can invalidate the provider's cacheable prefix even on user turns;
// keeping existing selections avoids needless reshuffling. Unresolved exact
// names never automatically activate descriptive suggestions here.
func (t *Thinker) applyPreload(k int, extraQuery string) {
	if t == nil || t.toolIndex == nil {
		return
	}
	query := strings.TrimSpace(t.directive)
	if extra := strings.TrimSpace(extraQuery); extra != "" {
		if query == "" {
			query = extra
		} else {
			query = query + "\n" + extra
		}
	}
	if query == "" {
		return
	}
	allowNoSpawn := t.threadID == "main" || t.allowNoSpawn
	for _, h := range t.searchAuthorizedTools(query, k, allowNoSpawn) {
		t.touchActiveTool(h.Name)
	}
}

// toolAuthorized is the common hard capability predicate for model schema
// exposure, discovery/preloading, and execution. Main has the complete
// attached surface. Workers get Core/exact tools from toolAllowlist plus MCP
// tools belonging to servers explicitly granted through mcp=.
func (t *Thinker) toolAuthorized(name string) bool {
	if t == nil {
		return false
	}
	return t.toolAuthorizedFor(name, t.toolAllowlist, t.toolMCPScopes)
}

func (t *Thinker) searchAuthorizedTools(query string, k int, allowNoSpawn bool) []IndexEntry {
	return t.searchAuthorizedToolsDetailed(query, k, allowNoSpawn).Hits
}

func (t *Thinker) searchAuthorizedToolsDetailed(query string, k int, allowNoSpawn bool) indexSearchResult {
	if t == nil || t.toolIndex == nil || k <= 0 {
		return indexSearchResult{}
	}
	if t.toolAllowlist == nil {
		return t.toolIndex.searchDetailed(query, k, allowNoSpawn, nil)
	}
	// Snapshot grants outside the index lock: toolAuthorized also reads it.
	allowed := make(map[string]bool, len(t.toolAllowlist))
	for name, enabled := range t.toolAllowlist {
		if enabled && t.toolAuthorized(name) {
			allowed[name] = true
		}
	}
	for server, enabled := range t.toolMCPScopes {
		if enabled {
			for _, name := range t.toolIndex.ToolsForServer(server) {
				if t.toolAuthorized(name) {
					allowed[name] = true
				}
			}
		}
	}
	return t.toolIndex.searchDetailed(query, k, allowNoSpawn, func(name string) bool { return allowed[name] })
}

func (t *Thinker) authorizedActiveTools(active map[string]bool) map[string]bool {
	if t == nil || t.toolAllowlist == nil {
		return active
	}
	filtered := make(map[string]bool, len(active))
	for name, enabled := range active {
		if enabled && t.toolAuthorized(name) {
			filtered[name] = true
		}
	}
	return filtered
}

func (t *Thinker) authorizedToolAllowlist(allowlist map[string]bool) map[string]bool {
	if t == nil || allowlist == nil {
		return allowlist
	}
	filtered := make(map[string]bool, len(allowlist))
	for name, enabled := range allowlist {
		if enabled && t.toolAuthorized(name) {
			filtered[name] = true
		}
	}
	return filtered
}

// touchActiveTool marks a tool active and refreshes its recency to
// the current iteration. Every path that surfaces a tool — spawn-time
// preload, search_tools, per-turn applyPreload — goes through here so
// evictActiveToolsLRU has a consistent recency key. Lazily inits both
// maps so bare test thinkers don't need to.
func (t *Thinker) touchActiveTool(name string) {
	if !t.toolAuthorized(name) {
		return
	}
	if t.activeTools == nil {
		t.activeTools = map[string]bool{}
	}
	if t.activeToolAge == nil {
		t.activeToolAge = map[string]int{}
	}
	t.activeTools[name] = true
	t.activeToolAge[name] = t.iteration
}

func countActiveMCPTools(active map[string]bool) int {
	n := 0
	for _, ok := range active {
		if ok {
			n++
		}
	}
	return n
}

// recordPresentedTools snapshots the exact provider-visible tool names for
// the current model request (or realtime session configuration). Dispatch
// consults this snapshot rather than trying to reconstruct visibility from a
// different subset of tool state.
func (t *Thinker) recordPresentedTools(tools []NativeTool, snapshots ...map[string]*ToolDef) {
	if t == nil {
		return
	}
	presented := make(map[string]bool, len(tools))
	identities := make(map[string]ToolIdentity, len(tools))
	manifest := make([]ToolIdentity, 0, len(tools))
	definitions := map[string]*ToolDef{}
	for _, tool := range tools {
		presented[tool.Name] = true
		identity := ToolIdentity{Name: tool.Name}
		var def *ToolDef
		if len(snapshots) > 0 {
			def = snapshots[0][tool.Name]
		} else if t.registry != nil {
			def = t.registry.Get(tool.Name)
		}
		if def != nil {
			identity = toolIdentity(def)
			definitions[tool.Name] = def
		}
		identities[tool.Name] = identity
		manifest = append(manifest, identity)
	}
	t.presentedToolsMu.Lock()
	t.presentedTools = presented
	t.presentedIdentities = identities
	t.presentedDefinitions = definitions
	digest := manifestDigest(manifest)
	t.toolManifestHash = digest
	t.presentedToolsMu.Unlock()
	if t.telemetry != nil && len(tools) > 0 {
		t.telemetry.Emit("tool.manifest", t.threadID, map[string]any{"iteration": t.iteration, "hash": digest, "tools": manifest})
	}
}

// modelToolCallable applies the thread's effective tool set: durable/static
// spawn grants remain callable, and dynamic tools are callable only when their
// schemas were actually presented for the current request/session.
func (t *Thinker) modelToolCallable(name string, fallback map[string]bool) bool {
	if t == nil {
		return false
	}
	if !t.toolAuthorized(name) {
		return false
	}
	if fallback[name] {
		return true
	}
	t.presentedToolsMu.RLock()
	allowed := t.presentedTools[name]
	t.presentedToolsMu.RUnlock()
	return allowed
}

// prepareNativeTools resolves the provider-neutral MCP loading policy into
// the exact schema list for one model request. Always-loaded tools are merged
// transiently, never inserted into activeTools, so search activation remains
// LRU-bounded while pinned schemas cannot be evicted.
func (t *Thinker) prepareNativeTools(providerName string) []NativeTool {
	if t == nil || t.registry == nil {
		return nil
	}
	t.recordToolCatalog()
	eager := t.useEagerTools()
	automatic := t.config.GetAutomaticToolLoading()
	var overlay map[string]bool
	if eager {
		t.automaticTools = automaticToolSelection{}
		t.lastToolMode = "eager"
	} else {
		t.lastToolMode = "discovery"
		preloadK := 5
		if providerName == "openai-codex" {
			preloadK = 3
		}
		if automatic.Enabled {
			overlay = t.prepareAutomaticTools(automatic)
		} else {
			t.automaticTools = automaticToolSelection{}
			t.applyPreload(preloadK, t.lastInboundForPreload)
		}
		t.evictActiveToolsLRU(activeToolsCap)
	}

	allowNoSpawn := t.threadID == "main" || t.allowNoSpawn
	// Expiration is per-turn bookkeeping, not part of visibility resolution.
	for name, until := range t.discoveredToolUntil {
		if t.iteration > until {
			delete(t.discoveredToolUntil, name)
		}
	}
	tools, definitions, active := t.visibleNativeToolSnapshot(t.toolAllowlist, t.toolMCPScopes, overlay)
	t.recordPresentedTools(tools, definitions)
	t.lastNativeToolCount = len(tools)
	t.lastActiveMCPCount = countActiveMCPTools(active)
	t.lastAlwaysMCPCount = t.toolIndex.AlwaysCount(allowNoSpawn)
	t.lastDeferredMCPCount = t.toolIndex.DeferredCount(eager, allowNoSpawn)
	return tools
}

// evictActiveToolsLRU bounds the sticky active-tool set. Sticky preload
// is what makes the tool list cache-friendly (grow then stabilise),
// but an unbounded set on a long, topic-hopping conversation would
// creep toward "everything" and lose the token win. When the set
// exceeds limit, drop the least-recently-surfaced tools down to ~70%
// of limit — batched so the next turn doesn't immediately re-evict
// (each eviction is itself a one-turn cache bust).
func (t *Thinker) evictActiveToolsLRU(limit int) {
	if limit <= 0 || len(t.activeTools) <= limit {
		return
	}
	type aged struct {
		name string
		age  int
	}
	all := make([]aged, 0, len(t.activeTools))
	for n := range t.activeTools {
		all = append(all, aged{n, t.activeToolAge[n]})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].age == all[j].age {
			return all[i].name < all[j].name
		}
		return all[i].age < all[j].age
	})
	keep := limit * 7 / 10
	for i := 0; i < len(all)-keep; i++ {
		delete(t.activeTools, all[i].name)
		delete(t.activeToolAge, all[i].name)
	}
}

// runSearchTools executes the search and mutates t.activeTools. Used
// by both mainToolHandler and threadToolHandler. allowNoSpawn is
// false for sub-threads (they cannot see no_spawn-flagged servers)
// and true for main.
func runSearchTools(t *Thinker, args map[string]string, allowNoSpawn bool) string {
	query, rawQueries := strings.TrimSpace(args["query"]), strings.TrimSpace(args["queries"])
	if query == "" && rawQueries == "" {
		return `{"error":"query or queries is required"}`
	}
	k := 5
	if raw, ok := args["k"]; ok && raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			if n > 20 {
				n = 20
			}
			k = n
		}
	}
	if t.toolIndex == nil {
		return `{"error":"tool index not initialised — no MCPs attached"}`
	}
	maxSchemaTokens := 16000
	if raw := args["max_schema_tokens"]; raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 && n <= 16000 {
			maxSchemaTokens = n
		}
	}
	available := t.authorizedDiscoveryServers(allowNoSpawn)
	defaultAccess := normalizeDiscoveryAccess(DiscoveryAccess(args["access"]))
	defaultServers := parseStringList(args["server_names"])
	var intents []DiscoveryIntent
	maxTools := k
	if rawQueries != "" {
		inputs, err := parseDiscoveryQueries(rawQueries)
		if err != nil {
			return `{"error":"queries must be a JSON array of strings or query objects"}`
		}
		if len(inputs) > 16 {
			return `{"error":"queries exceeds the 16-intent limit"}`
		}
		if args["k"] == "" {
			maxTools = 20
		}
		for _, input := range inputs {
			if strings.TrimSpace(input.Query) == "" {
				continue
			}
			access := input.Access
			if access == "" {
				access = defaultAccess
			}
			servers := input.Servers
			if len(servers) == 0 {
				servers = defaultServers
			}
			limit := input.Limit
			if limit <= 0 || limit > 20 {
				limit = k
			}
			if len(servers) > 0 {
				intents = append(intents, DiscoveryIntent{Query: strings.TrimSpace(input.Query), Servers: sortedSet(stringSetLower(servers)), Access: normalizeDiscoveryAccess(access), Limit: limit, Source: "explicit", Priority: 90})
			} else {
				intents = append(intents, compileDiscoveryIntent(input.Query, nil, access, limit, "explicit", 90, available)...)
			}
		}
	} else {
		if len(defaultServers) > 0 {
			intents = []DiscoveryIntent{{Query: query, Servers: sortedSet(stringSetLower(defaultServers)), Access: defaultAccess, Limit: k, Source: "explicit", Priority: 90}}
		} else {
			intents = compileDiscoveryIntent(query, nil, defaultAccess, k, "explicit", 90, available)
		}
	}
	if len(intents) == 0 {
		return `{"error":"at least one non-empty query is required"}`
	}
	discovery := t.discoverTools(DiscoveryRequest{Intents: intents, AllowNoSpawn: allowNoSpawn, AllowSuggestions: true, MaxTools: maxTools, MaxSchemaTokens: maxSchemaTokens, Activate: true})
	res := searchToolsResult{Query: query, Loaded: discovery.Loaded, CatalogRevision: discovery.CatalogRevision, AvailableServers: available, Results: discovery.Results, SchemaTokens: discovery.SchemaTokens, SkippedBudget: discovery.SkippedBudget}
	if t.discoveredToolUntil == nil {
		t.discoveredToolUntil = map[string]int{}
	}
	for _, name := range discovery.Loaded {
		t.discoveredToolUntil[name] = t.iteration + 1
	}
	seenHits, seenSuggestions := map[string]bool{}, map[string]bool{}
	coverageUnavailable := false
	for _, group := range discovery.Results {
		res.Unresolved = append(res.Unresolved, group.Unresolved...)
		if len(group.Ambiguous) > 0 {
			if res.Ambiguous == nil {
				res.Ambiguous = map[string][]string{}
			}
			for name, matches := range group.Ambiguous {
				res.Ambiguous[name] = matches
			}
		}
		coverageUnavailable = coverageUnavailable || group.Coverage == "unavailable_in_scope"
		for _, candidate := range group.Candidates {
			hit := searchToolHit{Name: candidate.Name, Server: candidate.Server, Summary: candidate.Summary, Match: candidate.Match, Score: candidate.Score}
			if candidate.Suggestion {
				if !seenSuggestions[candidate.Name] {
					res.Suggestions = append(res.Suggestions, hit)
					seenSuggestions[candidate.Name] = true
				}
			} else if !seenHits[candidate.Name] {
				res.Hits = append(res.Hits, hit)
				seenHits[candidate.Name] = true
			}
		}
	}
	if len(res.Hits) == 0 && len(res.Suggestions) == 0 {
		if coverageUnavailable {
			res.Error = "unavailable_in_scope: no matching tool is authorized in the requested server scope; do not substitute another operation."
		} else {
			res.Error = "lexical_miss: no indexed terms matched. This does not prove the capability is unavailable; refine the query or inspect available_servers."
		}
		if len(available) > 0 {
			res.Note = "no lexical matches; available_servers: " + strings.Join(available, ", ") + "; refine the query"
		} else {
			res.Note = "no MCP servers visible to this thread"
		}
	}
	if len(res.Unresolved) > 0 || len(res.Ambiguous) > 0 {
		res.Note = strings.TrimSpace(res.Note + " Requested names were unresolved or ambiguous. Descriptive suggestions, if present, have schemas loaded for inspection but are not exact matches or equivalent replacements. Confirm their documented operation before use; do not infer that a capability is unsupported from a failed name lookup.")
	}
	if t.telemetry != nil {
		t.telemetry.Emit("tool.discovery", t.threadID, map[string]any{"iteration": t.iteration, "result": res})
	}
	out, _ := json.Marshal(res)
	return string(out)
}
