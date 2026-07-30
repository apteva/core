package core

import (
	"encoding/json"
	"fmt"
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
		Description: "Search for MCP tools by keyword and load their schemas into your context. " +
			"Use when you need a capability you don't currently have visible — file upload, " +
			"posting to a channel, fetching from an integration, etc. Returns up to k matches " +
			"with name + summary; their full schemas become available for you to call on the " +
			"next turn. Use search_tools only in a discovery-only turn: you may call multiple " +
			"search_tools in parallel, but do not call any other tool until you have received " +
			"the search results on the next turn.",
		Syntax: `[[search_tools query="upload file" k="5"]]`,
		Rules:  `query is required; k defaults to 5 and caps at 20. Loaded tools persist for the rest of this thread's conversation (subject to compaction). Schemas appear on the next thinking turn — call only search_tools during discovery, then wait for that turn before calling any execution, reporting, messaging, pacing, or completion tool.`,
		Core:   true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "Keywords describing the capability you need.",
				},
				"k": map[string]any{
					"type":        "integer",
					"description": "Maximum number of tools to load (default 5, max 20).",
				},
			},
			"required": []string{"query"},
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
	Query  string          `json:"query"`
	Hits   []searchToolHit `json:"hits"`
	Loaded []string        `json:"loaded"` // names whose schemas are now in context
	Note   string          `json:"note,omitempty"`
}

type searchToolHit struct {
	Name    string `json:"name"`
	Server  string `json:"server"`
	Summary string `json:"summary"`
}

// applyPreload runs a BM25 search against the thread's DIRECTIVE and
// stickily activates up to k matching tools — they join t.activeTools
// and stay (subject to the LRU cap). Called every turn in discovery
// mode, but idempotent after the first.
//
// Two design choices, both for prompt caching:
//
//  1. STICKY, not transient. An early draft merged the preload
//     transiently each turn — that churned the `tools` array, which
//     sits in the cacheable prompt prefix, busting the cache wholesale
//     (~0.1% hit measured).
//  2. DIRECTIVE-ONLY query, not directive+lastUserText. The directive
//     is stable; the last user turn is not. Including the user turn
//     made preload surface different tools every turn, so the active
//     set never stopped growing and the array never stabilised — even
//     sticky, monotonic growth still busts the cache each turn it
//     grows. With a directive-only query the set seeds once and then
//     holds: same query → same hits → already-active → no-op. The
//     array goes stable as soon as explicit search_tools calls stop,
//     and from there every turn caches.
//
// The cost of (2): per-turn task adaptivity moves to search_tools — if
// the agent needs something the directive doesn't imply, it searches.
// A bounded one-time round-trip per new capability beats a permanent
// per-turn cache bust.
//
// Sub-threads run with allowNoSpawn=false (their thread-id is not
// "main"); main runs with allowNoSpawn=true.
//
// extraQuery is the text of any events drained THIS iteration (user
// message, peer send, etc.). Appending it lets BM25 surface tools
// the user asked for — "send a pushover test" preloads pushover_*
// the same turn the agent first sees the request, no `search_tools`
// round-trip. Cache cost: the turn after a fresh event already busts
// the prompt prefix (new user content in messages), so widening
// preload that turn is free. On subsequent quiet turns extraQuery
// is "", we revert to directive-only, and the active set stabilises
// so the prefix caches again.
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
	allowNoSpawn := t.threadID == "main"
	for _, h := range t.toolIndex.Search(query, k, allowNoSpawn) {
		t.touchActiveTool(h.Name)
	}
}

// touchActiveTool marks a tool active and refreshes its recency to
// the current iteration. Every path that surfaces a tool — spawn-time
// preload, search_tools, per-turn applyPreload — goes through here so
// evictActiveToolsLRU has a consistent recency key. Lazily inits both
// maps so bare test thinkers don't need to.
func (t *Thinker) touchActiveTool(name string) {
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
func (t *Thinker) recordPresentedTools(tools []NativeTool) {
	if t == nil {
		return
	}
	presented := make(map[string]bool, len(tools))
	for _, tool := range tools {
		presented[tool.Name] = true
	}
	t.presentedToolsMu.Lock()
	t.presentedTools = presented
	t.presentedToolsMu.Unlock()
}

// modelToolCallable applies the thread's effective tool set: durable/static
// spawn grants remain callable, and dynamic tools are callable only when their
// schemas were actually presented for the current request/session.
func (t *Thinker) modelToolCallable(name string, fallback map[string]bool) bool {
	if t == nil {
		return false
	}
	if t.threadID != "main" && !t.allowNoSpawn && t.toolIndex != nil {
		if entry, ok := t.toolIndex.Get(name); ok && entry.NoSpawn {
			return false
		}
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
	eager := t.useEagerTools()
	if eager {
		t.lastToolMode = "eager"
	} else {
		t.lastToolMode = "discovery"
		preloadK := 5
		if providerName == "openai-codex" {
			preloadK = 3
		}
		t.applyPreload(preloadK, t.lastInboundForPreload)
		t.evictActiveToolsLRU(activeToolsCap)
	}

	allowNoSpawn := t.threadID == "main"
	active := t.activeTools
	baseline := t.toolIndex.BaselineNames(eager, allowNoSpawn)
	if len(baseline) > 0 {
		merged := make(map[string]bool, len(t.activeTools)+len(baseline))
		for name, enabled := range t.activeTools {
			if enabled {
				merged[name] = true
			}
		}
		for _, name := range baseline {
			merged[name] = true
		}
		active = merged
	}

	tools := t.registry.NativeTools(t.toolAllowlist, active, t.systemThread)
	t.recordPresentedTools(tools)
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
	sort.Slice(all, func(i, j int) bool { return all[i].age < all[j].age })
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
	query := args["query"]
	if query == "" {
		return `{"error":"query is required"}`
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
	// A valid search always owes the model one continuation. With hits, that
	// turn exposes the newly activated schemas; without hits, it lets the
	// model process the bounded diagnostic note and choose another path.
	// This wake is independent of (and does not move) the pending pace timer.
	t.kickNextTurn = true
	hits := t.toolIndex.Search(query, k, allowNoSpawn)
	res := searchToolsResult{Query: query}
	for _, h := range hits {
		t.touchActiveTool(h.Name)
		summary := h.Description
		if len(summary) > 240 {
			summary = summary[:237] + "..."
		}
		res.Hits = append(res.Hits, searchToolHit{
			Name: h.Name, Server: h.Server, Summary: summary,
		})
		res.Loaded = append(res.Loaded, h.Name)
	}
	if len(res.Hits) == 0 {
		// Tell the LLM what *is* attached so it can refine the query or
		// reach for the gateway's install/list_apps tool to add what's
		// missing. Cheaper than another search round-trip.
		servers := t.toolIndex.Servers()
		var visible []string
		for _, s := range servers {
			// Reuse Search to honour the no_spawn filter — we don't want
			// a sub-thread learning that `apteva-server` exists.
			if hits := t.toolIndex.Search(s, 1, allowNoSpawn); len(hits) > 0 {
				visible = append(visible, s)
			}
		}
		if len(visible) > 0 {
			res.Note = fmt.Sprintf("no matches; attached servers: %v", visible)
		} else {
			res.Note = "no matches; no MCP servers visible to this thread"
		}
	}
	out, _ := json.Marshal(res)
	return string(out)
}
