package core

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// ToolResponse is the return value from a tool handler.
type ToolResponse struct {
	Text  string // text result (always present)
	Image []byte // optional image (screenshot etc.) — sent as part of tool result to LLM
}

// ToolDef defines a tool available to threads.
type ToolDef struct {
	Name        string
	Description string // human-readable, used for RAG embedding
	Syntax      string // example usage
	Rules       string // usage rules for the prompt
	Core        bool   // always in prompt (pace, send, done, evolve)
	MainOnly    bool   // only for main thread (spawn, kill)
	ThreadOnly  bool   // only for sub-threads, not main (reply)
	SystemOnly  bool   // only for system threads (unconscious)
	MCP         bool   // provided by an MCP server — not sent as native tools to main, only to sub-threads
	MCPServer   string // name of the MCP server that provides this tool
	Handler     func(args map[string]string) ToolResponse // nil = handled inline by tool handler
	Embedding   []float64
	InputSchema map[string]any // JSON Schema for native tool calling (nil = auto-generated from Syntax)
}

// ToolRegistry holds all tool definitions with embeddings for RAG retrieval.
type ToolRegistry struct {
	mu       sync.RWMutex
	tools    map[string]*ToolDef
	apiKey   string
	embedded bool
}

// sortedToolKeys returns tool names in deterministic sorted order.
// This is critical for LLM token caching — non-deterministic ordering
// breaks prefix caching on every call.
func (tr *ToolRegistry) sortedToolKeys() []string {
	keys := make([]string, 0, len(tr.tools))
	for k := range tr.tools {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func NewToolRegistry(apiKey string) *ToolRegistry {
	tr := &ToolRegistry{
		tools:  make(map[string]*ToolDef),
		apiKey: apiKey,
	}
	tr.registerDefaults()
	return tr
}

func (tr *ToolRegistry) registerDefaults() {
	// Scaffolding meta-tools — search + (eventually) load_tool. Listed
	// first so they sort to the top of the registry for readers and
	// so the LLM's tool-list always opens with the discovery surface.
	registerSearchTool(tr)
	// Core tools — always in prompt
	tr.Register(&ToolDef{
		Name:        "pace",
		Description: "Control sleep duration, model tier, and provider. Events always wake you immediately.",
		Syntax:      `[[pace sleep="5m" model="small" provider="anthropic"]]`,
		Rules:       `sleep accepts any duration: "2s", "30s", "5m", "1h", "6h". Named aliases also work: rate="fast" (2s), rate="normal" (10s), rate="slow" (30s), rate="sleep" (2m). Models: "large", "medium", "small". provider: switch to a different LLM provider by name (optional, only when multiple providers are configured). Sleep long when idle — you'll be woken by events.`,
		Core:        true,
		// All fields optional — pace() with no args continues current state.
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"sleep":    map[string]any{"type": "string", "description": "Duration like \"2s\", \"5m\", \"1h\". Mutually exclusive with rate."},
				"rate":     map[string]any{"type": "string", "description": "Named alias: \"fast\" (2s), \"normal\" (10s), \"slow\" (30s), \"sleep\" (2m)."},
				"model":    map[string]any{"type": "string", "description": "Model tier: \"large\", \"medium\", \"small\"."},
				"provider": map[string]any{"type": "string", "description": "LLM provider name (optional)."},
			},
		},
	})
	tr.Register(&ToolDef{
		Name:        "send",
		Description: "Send a message to another thread. Use id=\"parent\" to report to your parent thread. ALWAYS send results back after completing work.",
		Syntax:      `send(id="parent", message="...", media="url1 url2")`,
		Rules:       `Use id="parent" for your parent thread. Use id="main" for the top coordinator. media is optional — space-separated URLs (audio, images, video). Media URLs are sent natively to the target thread's LLM for analysis. You MUST send results back to your parent after completing any task.`,
		Core:        true,
		// Explicit schema with required fields so the LLM is forced to
		// include id and message. Without this, schemaFromSyntax would
		// produce properties but no "required" array — we saw Kimi
		// occasionally drop id, causing send to silently no-op and the
		// parent thread to never receive the reply.
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":      map[string]any{"type": "string", "description": "Target thread id — use \"parent\" for your spawner, \"main\" for the top coordinator, or a sibling / child id."},
				"message": map[string]any{"type": "string", "description": "Message body."},
				"media":   map[string]any{"type": "string", "description": "Optional space-separated media URLs (audio/image/video)."},
			},
			"required": []string{"id", "message"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "done",
		Description: "Permanently terminate this thread. Send a final message and shut down.",
		Syntax:      `[[done message="Final result"]]`,
		Rules:       `PERMANENTLY kills this thread. Only use when truly complete. Do NOT use after a single reply in a conversation.`,
		Core:        true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"message": map[string]any{"type": "string", "description": "Final message sent to parent before the thread shuts down."},
			},
			"required": []string{"message"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "evolve",
		Description: "Rewrite ONLY the mission portion of your directive — the prose between the [DIRECTIVE] markers in your system prompt. Do NOT include framework rules (THINKING/EVENTS/SPAWNING/PACING/TOOL CALLS sections, the opening 'You are the main coordinating thread…' sentence, [ACTIVE THREADS], [AVAILABLE MCP SERVERS]) — those are injected by the platform and are not part of your directive.",
		Syntax:      `[[evolve directive="Updated mission text"]]`,
		Rules:       `Persisted to config. Use sparingly — only when you've learned something worth remembering in your mission. The text you submit fully replaces the current directive; keep it focused on goals, constraints, and learned rules. Submitting platform boilerplate is rejected with an error.`,
		Core:        true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"directive": map[string]any{"type": "string", "description": "New mission text. JUST the mission — not the framework rules, not section headers like THINKING/EVENTS/SPAWNING."},
			},
			"required": []string{"directive"},
		},
	})

	// remember tool intentionally NOT registered for now. Memory writes
	// are off until we redesign that subsystem — the dispatch case in
	// thinker.go / thread.go and the MemoryStore code stay in place so
	// re-enabling is a one-line change (uncomment the Register block).
	/*
		tr.Register(&ToolDef{
			Name:        "remember",
			Description: "Store a fact for future turns. Prefix with a tag: [preference] [correction] [decision] [fact] [user]. Write memories that match FUTURE queries (include the tool, the target, the outcome). Remember liberally; skip transient in-flight state.",
			Syntax:      `[[remember text="[preference] exec: user OK with shell commands on their own server"]]`,
			Core:        true,
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string", "description": "Memory text, typically prefixed with a bracketed tag like [preference], [correction], [decision], [fact], [user]."},
				},
				"required": []string{"text"},
			},
		})
	*/

	// Main-only tools
	tr.Register(&ToolDef{
		Name:        "spawn",
		Description: "Create a new thread with its own directive, tools, and continuous thinking loop. By default the worker starts thinking immediately on its directive; pass paused=\"true\" to spawn it dormant — it'll wake on the first `send` you give it. Use media to pass audio/image/video URLs for the new thread's LLM to analyze natively.",
		Syntax:      `spawn(id="name", directive="What this thread does", tools="web,exec", mcp="store,stripe", paused="true")`,
		Rules:       `id: unique name. directive: what the thread does. tools: comma-separated local tools (web, exec, read_file, etc). mcp: comma-separated MCP server names — thread gets its own connection and only sees those tools. provider: LLM provider name (optional). paused: "true" to spawn dormant — useful when you want to spawn several workers atomically before any think, or when you want to attach a message to the worker's first turn rather than letting it act on the directive alone. media: space-separated URLs (audio/image/video) — sent directly to the thread's LLM as native content for analysis.`,
		Core:        true,
		MainOnly:    true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":        map[string]any{"type": "string", "description": "Unique thread id for the new worker."},
				"directive": map[string]any{"type": "string", "description": "What the thread does — its system prompt / role."},
				"tools":     map[string]any{"type": "string", "description": "Comma-separated local tool names (web, exec, read_file, ...). Optional."},
				"mcp":       map[string]any{"type": "string", "description": "Comma-separated MCP server names. Optional."},
				"provider":  map[string]any{"type": "string", "description": "LLM provider name. Optional."},
				"paused":    map[string]any{"type": "string", "description": "Set to \"true\" to spawn the worker in paused state; it will not think until you send it a message. Default: not paused (auto-starts on directive)."},
				"media":     map[string]any{"type": "string", "description": "Space-separated media URLs passed to the new thread's LLM. Optional."},
			},
			"required": []string{"id", "directive"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "kill",
		Description: "Stop a thread immediately and remove it from persistent config.",
		Syntax:      `[[kill id="name"]]`,
		Core:        true,
		MainOnly:    true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "string", "description": "Thread id to kill."},
			},
			"required": []string{"id"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "update",
		Description: "Update a running thread's id, display name, directive, and/or tools. Renaming the id cascades through children's parent_id and the on-disk session file.",
		Syntax:      `[[update id="thread-id" new_id="renamed" name="Friendly Label" directive="New directive" tools="tool1,tool2"]]`,
		Rules:       `Provide at least one of new_id, name, directive, or tools. The thread is notified of directive changes. Tools replace the full set (builtins are always included). new_id renames the immutable id (children + session storage follow); name is just a display label.`,
		Core:        true,
		MainOnly:    true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":        map[string]any{"type": "string", "description": "Target thread id."},
				"new_id":    map[string]any{"type": "string", "description": "Replacement id. Cascades through children's parent_id and the on-disk session file. Must be unique among siblings."},
				"name":      map[string]any{"type": "string", "description": "Human-readable label for display. Independent of id."},
				"directive": map[string]any{"type": "string", "description": "Replacement directive."},
				"tools":     map[string]any{"type": "string", "description": "Comma-separated tool names replacing the current set."},
			},
			"required": []string{"id"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "connect",
		Description: "Register a NEW MCP server at runtime that isn't already in the instance catalog. For MCP servers already listed under [AVAILABLE MCP SERVERS] in your prompt, do NOT use connect — use spawn(mcp=\"name\", ...) to give a worker access to those tools. Only reach for connect when you need to hook up a brand-new server by URL.",
		Syntax:      `[[connect name="server-name" url="http://host:port/mcp/1" transport="http"]]`,
		Rules:       `HTTP only here: pass url and transport="http". Tools become available immediately after connecting. Stdio connect is an advanced path — use command/args, not covered by this schema.`,
		Core:        true,
		MainOnly:    true,
		// Schema intentionally exposes ONLY the HTTP-connect path (the
		// common case). Previously we included command/args as
		// first-class fields; Kimi misread that as "pushover is a
		// command" for catalog MCPs and called connect(command=pushover)
		// instead of spawn(mcp=pushover). Stdio connect is still
		// possible via raw bracket syntax for advanced use, it just
		// isn't in the schema surface Kimi sees.
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":      map[string]any{"type": "string", "description": "Friendly name for the new MCP server."},
				"url":       map[string]any{"type": "string", "description": "HTTP endpoint for Streamable-HTTP transport."},
				"transport": map[string]any{"type": "string", "description": "\"http\" for Streamable-HTTP."},
			},
			"required": []string{"name", "url", "transport"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "disconnect",
		Description: "Disconnect from a running MCP server and unregister its tools.",
		Syntax:      `[[disconnect name="server-name"]]`,
		Core:        true,
		MainOnly:    true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string", "description": "Name of the MCP server to disconnect."},
			},
			"required": []string{"name"},
		},
	})
	tr.Register(&ToolDef{
		Name:        "list_connected",
		Description: "List all MCP servers currently connected to this instance.",
		Syntax:      `[[list_connected]]`,
		Core:        true,
		MainOnly:    true,
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	})

	// Capability tools (web, exec, files, code, pdf, …) live in the
	// sibling github.com/apteva/tools module and are wired in by the
	// instance config when enabled.
}

// NewScopedRegistry creates a minimal registry containing only the specified tools
// copied from the parent. Core tools are always included. Local tools (web, exec, etc.)
// are included if they appear in localTools. MCP tools are NOT copied — they should be
// registered separately from thread-local MCP connections.
func (tr *ToolRegistry) NewScopedRegistry(localTools map[string]bool) *ToolRegistry {
	scoped := &ToolRegistry{
		tools:  make(map[string]*ToolDef),
		apiKey: tr.apiKey,
	}

	tr.mu.RLock()
	defer tr.mu.RUnlock()

	for name, tool := range tr.tools {
		if tool.Core {
			// Always include core tools
			scoped.tools[name] = tool
		} else if !tool.MCP && localTools[name] {
			// Include allowed local tools (web, exec, read_file, etc.)
			scoped.tools[name] = tool
		}
	}
	scoped.embedded = tr.embedded // inherit embedding state
	return scoped
}

func (tr *ToolRegistry) Register(tool *ToolDef) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.tools[tool.Name] = tool
}

func (tr *ToolRegistry) Get(name string) *ToolDef {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	return tr.tools[name]
}

// RemoveByMCPServer removes all tools provided by the named MCP server.
func (tr *ToolRegistry) RemoveByMCPServer(serverName string) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	for name, tool := range tr.tools {
		if tool.MCPServer == serverName {
			delete(tr.tools, name)
		}
	}
}

// EmbedAll computes embeddings for all non-core tools. Call once on startup.
func (tr *ToolRegistry) EmbedAll(ms *MemoryStore) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	for _, tool := range tr.tools {
		if tool.Core {
			continue // core tools don't need embeddings — always in prompt
		}
		emb, err := ms.embed(tool.Name + ": " + tool.Description)
		if err == nil {
			tool.Embedding = emb
		}
	}
	tr.embedded = true
}

// CoreDocsSummary returns a one-line summary of core tool names,
// sized for providers that receive full schemas via NativeTools in
// their `tools[]` payload. Emitting the full prose here (see CoreDocs)
// duplicates every tool's Description + Rules in the system prompt —
// ~5k extra input chars per iteration on a typical main thread.
//
// Callers that target providers WITHOUT native-tool support should
// keep using CoreDocs: those providers only see the prose and need
// the rules in the system prompt to behave.
//
// Ordering matches CoreDocs so the two agree when comparing, and the
// block is prefixed with the same marker so the composition breakdown
// still identifies it as the "core_tools" segment.
func (tr *ToolRegistry) CoreDocsSummary(includeMainOnly bool, includeSystemOnly ...bool) string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	sysOnly := len(includeSystemOnly) > 0 && includeSystemOnly[0]

	var names []string
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if !tool.Core {
			continue
		}
		if tool.MainOnly && !includeMainOnly {
			continue
		}
		if tool.SystemOnly && !sysOnly {
			continue
		}
		names = append(names, tool.Name)
	}
	if len(names) == 0 {
		return ""
	}
	return "CORE TOOLS — always available: " + strings.Join(names, ", ") +
		"\n(full schemas appear in your tools[] payload; use exactly those names)\n"
}

// CoreDocs returns documentation for core tools, always included in prompts.
func (tr *ToolRegistry) CoreDocs(includeMainOnly bool, includeSystemOnly ...bool) string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	sysOnly := len(includeSystemOnly) > 0 && includeSystemOnly[0]

	var sb strings.Builder
	sb.WriteString("CORE TOOLS — always available:\n")
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if !tool.Core {
			continue
		}
		if tool.MainOnly && !includeMainOnly {
			continue
		}
		if tool.SystemOnly && !sysOnly {
			continue
		}
		sb.WriteString(fmt.Sprintf("  %s — %s\n", tool.Name, tool.Description))
		if tool.Rules != "" {
			sb.WriteString(fmt.Sprintf("    %s\n", tool.Rules))
		}
	}
	return sb.String()
}

// Retrieve finds the most relevant non-core tools for the given context.
// Returns up to n tools, filtered by the allowed set.
func (tr *ToolRegistry) Retrieve(query string, n int, allowed map[string]bool, ms *MemoryStore) []*ToolDef {
	if !tr.embedded || query == "" {
		// Fallback: return all allowed non-core tools
		return tr.getAllowed(allowed)
	}

	queryEmb, err := ms.embed(query)
	if err != nil {
		return tr.getAllowed(allowed)
	}

	tr.mu.RLock()
	defer tr.mu.RUnlock()

	type scored struct {
		tool  *ToolDef
		score float64
	}

	isMainThread := allowed == nil
	var results []scored
	for _, tool := range tr.tools {
		if tool.Core || len(tool.Embedding) == 0 {
			continue
		}
		if allowed != nil && !allowed[tool.Name] {
			continue
		}
		if isMainThread && tool.ThreadOnly {
			continue
		}
		if !isMainThread && tool.MainOnly {
			continue
		}
		sim := cosineSimilarity(queryEmb, tool.Embedding)
		results = append(results, scored{tool: tool, score: sim})
	}

	// Sort descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Be generous — low threshold, include anything remotely relevant
	const minScore = 0.1
	var out []*ToolDef
	for i := 0; i < len(results) && len(out) < n; i++ {
		if results[i].score >= minScore {
			out = append(out, results[i].tool)
		}
	}
	return out
}

func (tr *ToolRegistry) getAllowed(allowed map[string]bool) []*ToolDef {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	isMainThread := allowed == nil
	var out []*ToolDef
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if tool.Core {
			continue
		}
		if allowed != nil && !allowed[tool.Name] {
			continue
		}
		if isMainThread && tool.ThreadOnly {
			continue
		}
		if !isMainThread && tool.MainOnly {
			continue
		}
		out = append(out, tool)
	}
	return out
}

// BuildDocs generates tool documentation for a set of discovered tools.
func (tr *ToolRegistry) BuildDocs(tools []*ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n[available tools — matched to your current context]\n")
	for _, tool := range tools {
		sb.WriteString(fmt.Sprintf("  %s — %s\n", tool.Name, tool.Description))
		if tool.Rules != "" {
			sb.WriteString(fmt.Sprintf("    %s\n", tool.Rules))
		}
	}
	sb.WriteString("If you need a different tool, describe what you need and it may appear next thought.\n")
	return sb.String()
}

// Dispatch executes a tool by name if it has a Handler. Returns response and whether it was handled.
func (tr *ToolRegistry) Dispatch(name string, args map[string]string) (ToolResponse, bool) {
	tr.mu.RLock()
	tool, exists := tr.tools[name]
	tr.mu.RUnlock()
	if !exists || tool.Handler == nil {
		return ToolResponse{}, false
	}
	return tool.Handler(args), true
}

// MCPToolSummary returns a compact summary of MCP tools grouped by server.
// Used in the main thread's system prompt so it knows what's available
// without sending full tool definitions to the LLM.
func (tr *ToolRegistry) MCPToolSummary() string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	servers := make(map[string][]string) // server → ["tool_name — description", ...]
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if !tool.MCP {
			continue
		}
		srv := tool.MCPServer
		if srv == "" {
			srv = "unknown"
		}
		// Strip server prefix from display name
		displayName := tool.Name
		if len(srv) > 0 && len(tool.Name) > len(srv)+1 {
			displayName = tool.Name[len(srv)+1:]
		}
		servers[srv] = append(servers[srv], fmt.Sprintf("  - %s — %s", displayName, tool.Description))
	}
	if len(servers) == 0 {
		return ""
	}

	// Sort server names for deterministic output
	srvNames := make([]string, 0, len(servers))
	for k := range servers {
		srvNames = append(srvNames, k)
	}
	sort.Strings(srvNames)

	var sb strings.Builder
	sb.WriteString("\n[MCP TOOLS — available for sub-threads]\n")
	sb.WriteString("These tools are NOT available to you directly. To use them, spawn a thread with the tool in its tools list.\n")
	sb.WriteString("When spawning, use the FULL prefixed name (e.g. \"servername_toolname\").\n\n")
	for _, srv := range srvNames {
		tools := servers[srv]
		sb.WriteString(fmt.Sprintf("%s (%d tools):\n", srv, len(tools)))
		for _, t := range tools {
			sb.WriteString(t + "\n")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// AllToolNames returns all non-core tool names (for spawn docs).
func (tr *ToolRegistry) AllToolNames() []string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	var names []string
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if !tool.Core && !tool.MainOnly {
			names = append(names, tool.Name)
		}
	}
	return names
}

// AllTools returns all tool definitions for display.
func (tr *ToolRegistry) AllTools() []*ToolDef {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	var out []*ToolDef
	for _, tool := range tr.tools {
		out = append(out, tool)
	}
	return out
}

// Count returns the total number of registered tools.
func (tr *ToolRegistry) Count() int {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	return len(tr.tools)
}

// Counts returns core, discoverable (RAG), and total tool counts.
func (tr *ToolRegistry) Counts() (core, rag, total int) {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	for _, tool := range tr.tools {
		if tool.Core {
			core++
		} else {
			rag++
		}
	}
	total = core + rag
	return
}

// NativeTools returns tool definitions for the LLM provider API.
//
// Visibility model:
//   - Core / non-MCP tools: always visible unless excluded by an
//     allowlist (sub-thread restriction) or by role flags
//     (MainOnly / ThreadOnly / SystemOnly applied at caller via
//     RetrieveTools — this method trusts what's in the registry).
//   - MCP tools (ToolDef.MCP == true): hidden by default — they only
//     appear when their name is in `active`. That's how the
//     "agent-driven discovery" model works: spawn-time MCPNames
//     preload, search_tools, and per-turn BM25 preload all push
//     names into the active set; nothing else gets MCP tools on
//     the wire.
//
// allowlist is the boot-time per-thread allowlist (sub-threads pass
// their tool set; main passes nil). active is the live per-turn set
// of MCP tools the thread has surfaced for use. Either argument may
// be nil.
func (tr *ToolRegistry) NativeTools(allowlist, active map[string]bool) []NativeTool {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	var out []NativeTool
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		if !toolVisible(tool, allowlist, active) {
			continue
		}

		nt := NativeTool{
			Name:        tool.Name,
			Description: tool.Description,
		}
		if tool.Rules != "" {
			nt.Description += " " + tool.Rules
		}

		// Use explicit schema if provided, otherwise generate from syntax
		if tool.InputSchema != nil {
			nt.Parameters = copyAndInjectReason(tool.InputSchema)
		} else {
			nt.Parameters = copyAndInjectReason(schemaFromSyntax(tool.Syntax))
		}
		out = append(out, nt)
	}
	return out
}

// toolVisible centralises the per-turn visibility check so callers
// can apply the same rules without duplicating the logic. Order
// matters: allowlist gates first (it's a hard boundary set at spawn
// time), then non-MCP defaults, then MCP requires explicit activation.
func toolVisible(tool *ToolDef, allowlist, active map[string]bool) bool {
	if tool.SystemOnly {
		return false
	}
	if allowlist != nil {
		// Sub-thread / scoped path: only the allowed names. activeTools
		// can still surface a name that wasn't in the boot allowlist —
		// the LLM searched for it at runtime.
		if !allowlist[tool.Name] && !active[tool.Name] {
			return false
		}
		return true
	}
	// Main thread (no allowlist).
	if tool.ThreadOnly {
		return false
	}
	if tool.MCP && !active[tool.Name] {
		return false
	}
	return true
}

// copyAndInjectReason adds the _reason field to a tool's JSON Schema.
// Returns a shallow copy so the original schema is not modified.
func copyAndInjectReason(schema map[string]any) map[string]any {
	out := make(map[string]any, len(schema)+1)
	for k, v := range schema {
		out[k] = v
	}
	// Copy properties map and add _reason
	props := make(map[string]any)
	if existing, ok := schema["properties"].(map[string]any); ok {
		for k, v := range existing {
			props[k] = v
		}
	}
	// Short schema description — the full rule lives once in the system
	// prompt's "TOOL CALL LABELS" section (see baseSystemPrompt). Keeping
	// the per-schema description terse saves ~150 chars × N tools on
	// every Chat call.
	props["_reason"] = map[string]any{
		"type":        "string",
		"description": "Short action label (3-6 words, imperative).",
	}
	out["properties"] = props
	// Add _reason to required list
	var required []any
	if existing, ok := schema["required"].([]any); ok {
		required = append(required, existing...)
	}
	if existingStr, ok := schema["required"].([]string); ok {
		for _, s := range existingStr {
			required = append(required, s)
		}
	}
	required = append(required, "_reason")
	out["required"] = required
	return out
}

// schemaFromSyntax extracts a JSON Schema from tool syntax like: [[name key="val" key2="val2"]]
func schemaFromSyntax(syntax string) map[string]any {
	props := make(map[string]any)
	// Extract key="..." patterns
	for _, m := range argRe.FindAllStringSubmatch(syntax, -1) {
		if len(m) >= 2 {
			props[m[1]] = map[string]string{"type": "string", "description": m[1]}
		}
	}
	if len(props) == 0 {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return map[string]any{"type": "object", "properties": props}
}
