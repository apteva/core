package main

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
	// Core tools — always in prompt
	tr.Register(&ToolDef{
		Name:        "pace",
		Description: "Control sleep duration, model tier, and provider. Events always wake you immediately.",
		Syntax:      `[[pace sleep="5m" model="small" provider="anthropic"]]`,
		Rules:       `sleep accepts any duration: "2s", "30s", "5m", "1h", "6h". Named aliases also work: rate="fast" (2s), rate="normal" (10s), rate="slow" (30s), rate="sleep" (2m). Models: "large", "medium", "small". provider: switch to a different LLM provider by name (optional, only when multiple providers are configured). Sleep long when idle — you'll be woken by events.`,
		Core:        true,
	})
	tr.Register(&ToolDef{
		Name:        "send",
		Description: "Send a message to any thread by ID. Optionally attach media (images/audio) via space-separated URLs.",
		Syntax:      `[[send id="thread-name" message="..." media="url1 url2"]]`,
		Rules:       `Use id="main" for the coordinator thread. media is optional — space-separated URLs. Supported: .png .jpg .gif .webp .mp3 .wav .aac .ogg .flac .aiff .m4a`,
		Core:        true,
	})
	tr.Register(&ToolDef{
		Name:        "done",
		Description: "Permanently terminate this thread. Send a final message and shut down.",
		Syntax:      `[[done message="Final result"]]`,
		Rules:       `PERMANENTLY kills this thread. Only use when truly complete. Do NOT use after a single reply in a conversation.`,
		Core:        true,
	})
	tr.Register(&ToolDef{
		Name:        "evolve",
		Description: "Rewrite your own directive to self-improve based on experience. Adjust approach, add learned rules, refine role.",
		Syntax:      `[[evolve directive="Updated directive"]]`,
		Rules:       `Persisted to config. Use sparingly — only when you've learned something worth remembering in your directive.`,
		Core:        true,
	})

	tr.Register(&ToolDef{
		Name:        "remember",
		Description: "Store something in persistent memory. Use for important facts, user preferences, lessons learned, decisions made. Memories survive restarts and are auto-recalled by relevance.",
		Syntax:      `[[remember text="Important fact to remember"]]`,
		Rules:       `Be selective — only store what you'd want to recall in future sessions. Not for transient state.`,
		Core:        true,
	})

	// Main-only tools
	tr.Register(&ToolDef{
		Name:        "spawn",
		Description: "Create a new thread with its own directive, tools, and continuous thinking loop. Optionally select a provider, MCP servers, and forward media.",
		Syntax:      `[[spawn id="name" directive="What this thread does" tools="web,exec" mcp="store,stripe" builtins="" provider="openai" media="url1 url2"]]`,
		Rules:       `id: unique name. directive: what the thread does. tools: comma-separated local tools (web, exec, read_file, etc). mcp: comma-separated MCP server names — thread gets its own connection and only sees those tools (efficient, auto-cleanup). builtins: provider builtins like "code_execution" (omit to inherit, empty string "" to disable all). provider: LLM provider name (optional). media: optional space-separated URLs forwarded as the thread's first event.`,
		Core:        true,
		MainOnly:    true,
	})
	tr.Register(&ToolDef{
		Name:        "kill",
		Description: "Stop a thread immediately and remove it from persistent config.",
		Syntax:      `[[kill id="name"]]`,
		Core:        true,
		MainOnly:    true,
	})
	tr.Register(&ToolDef{
		Name:        "update",
		Description: "Update a running thread's directive and/or tools. The thread's system prompt is rebuilt immediately.",
		Syntax:      `[[update id="name" directive="New directive" tools="tool1,tool2"]]`,
		Rules:       `Provide directive, tools, or both. The thread is notified of directive changes. Tools replace the full set (builtins are always included).`,
		Core:        true,
		MainOnly:    true,
	})
	tr.Register(&ToolDef{
		Name:        "connect",
		Description: "Connect to an MCP server at runtime. Supports stdio (command) or Streamable HTTP (url) transport. Discovers and registers all tools from the server.",
		Syntax:      `[[connect name="server-name" url="http://host:port/mcp/1" transport="http"]]`,
		Rules:       `For stdio: use command="path" args="arg1,arg2". For HTTP: use url="..." transport="http". Tools become available immediately after connecting.`,
		Core:        true,
		MainOnly:    true,
	})
	tr.Register(&ToolDef{
		Name:        "disconnect",
		Description: "Disconnect from a running MCP server and unregister its tools.",
		Syntax:      `[[disconnect name="server-name"]]`,
		Core:        true,
		MainOnly:    true,
	})
	tr.Register(&ToolDef{
		Name:        "list_connected",
		Description: "List all MCP servers currently connected to this instance.",
		Syntax:      `[[list_connected]]`,
		Core:        true,
		MainOnly:    true,
	})

	// Discoverable tools — retrieved by RAG
	tr.Register(&ToolDef{
		Name:        "web",
		Description: "Fetch a URL from the internet and return its text content. Use for research, looking up information, checking websites.",
		Syntax:      `[[web url="https://example.com"]]`,
		Rules:       `Only parameter is url. Results arrive as events in your next thought.`,
		Handler:     func(args map[string]string) ToolResponse { return ToolResponse{Text: webTool(args)} },
	})
	tr.Register(&ToolDef{
		Name:        "exec",
		Description: "Execute a shell command on the host machine and return stdout+stderr. Use for system administration, checking logs, running scripts, managing containers, inspecting files, git operations, deployments.",
		Syntax:      `[[exec command="ls -la /app" timeout="30" dir="/home"]]`,
		Rules:       `command: the shell command to run. timeout: seconds (default 30, max 300). dir: optional working directory. No interactive commands (no vim, top, less). Output truncated to 4000 chars.`,
		Handler:     func(args map[string]string) ToolResponse { return ToolResponse{Text: execTool(args)} },
	})
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
		sb.WriteString(fmt.Sprintf("  %s — %s\n", tool.Syntax, tool.Description))
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
		sb.WriteString(fmt.Sprintf("  %s — %s\n", tool.Syntax, tool.Description))
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

// NativeTools returns all tools as NativeTool definitions for the provider API.
// NativeTools returns tool definitions for the LLM provider API.
// allowlist filters to specific tools (nil = main thread, which excludes MCP tools).
// Sub-threads pass their allowlist which includes MCP tools they need.
func (tr *ToolRegistry) NativeTools(allowlist map[string]bool) []NativeTool {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	var out []NativeTool
	for _, name := range tr.sortedToolKeys() {
		tool := tr.tools[name]
		// Filter by allowlist if set (sub-threads)
		if allowlist != nil {
			if !allowlist[tool.Name] {
				continue
			}
		} else {
			// Main thread (nil allowlist): skip MCP tools, thread-only, and system-only tools
			if tool.ThreadOnly || tool.MCP || tool.SystemOnly {
				continue
			}
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
	props["_reason"] = map[string]any{
		"type":        "string",
		"description": "Brief reason for this tool call — what you're doing and why (for observability)",
	}
	out["properties"] = props
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
