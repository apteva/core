package main

import (
	"fmt"
	"strings"
	"sync"
)

// ToolDef defines a tool available to threads.
type ToolDef struct {
	Name        string
	Description string // human-readable, used for RAG embedding
	Syntax      string // example usage
	Rules       string // usage rules for the prompt
	Core        bool   // always in prompt (pace, send, done, evolve)
	MainOnly    bool   // only for main thread (spawn, kill)
	ThreadOnly  bool   // only for sub-threads, not main (reply)
	Handler     func(args map[string]string) string // nil = handled inline by tool handler
	Embedding   []float64
}

// ToolRegistry holds all tool definitions with embeddings for RAG retrieval.
type ToolRegistry struct {
	mu       sync.RWMutex
	tools    map[string]*ToolDef
	apiKey   string
	embedded bool
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
		Description: "Control thinking speed and model size. Set rate to fast, normal, slow, or sleep. Set model to large or small.",
		Syntax:      `[[pace rate="slow" model="small"]]`,
		Rules:       `Rates: "fast" (2s), "normal" (10s), "slow" (30s), "sleep" (2min). Models: "large", "small". Pace down when idle.`,
		Core:        true,
	})
	tr.Register(&ToolDef{
		Name:        "send",
		Description: "Send a message to any thread by ID. Use to communicate between threads or report to the main coordinator.",
		Syntax:      `[[send id="thread-name" message="..."]]`,
		Rules:       `Use id="main" for the coordinator thread.`,
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
		Description: "Create a new thread with its own directive, tools, and continuous thinking loop. Threads are persistent across restarts.",
		Syntax:      `[[spawn id="name" directive="What this thread does" tools="reply,web"]]`,
		Rules:       `id: unique name. directive: what the thread does. tools: comma-separated. Thread runs continuously and calls [[done]] when finished.`,
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

	// Discoverable tools — retrieved by RAG
	tr.Register(&ToolDef{
		Name:        "reply",
		Description: "Send a visible message to the user. Users cannot see your thoughts — only reply messages. Use for conversations and responses.",
		Syntax:      `[[reply message="Your response"]]`,
		Rules:       `Users can ONLY see [[reply]] messages, not your thoughts.`,
		Handler:     nil, // handled inline by thread tool handler
		ThreadOnly:  true, // NOT available to main thread
	})
	tr.Register(&ToolDef{
		Name:        "web",
		Description: "Fetch a URL from the internet and return its text content. Use for research, looking up information, checking websites.",
		Syntax:      `[[web url="https://example.com"]]`,
		Rules:       `Only parameter is url. Results arrive as events in your next thought.`,
		Handler:     webTool,
	})
	tr.Register(&ToolDef{
		Name:        "write_file",
		Description: "Write content to a file in the workspace directory. Use for creating documents, saving drafts, producing output files.",
		Syntax:      `[[write_file path="drafts/doc.md" content="..."]]`,
		Rules:       `Paths are relative to workspace/ directory.`,
		Handler:     writeFileTool,
	})
	tr.Register(&ToolDef{
		Name:        "read_file",
		Description: "Read a file from the workspace directory. Use for reviewing documents, checking existing content, loading data.",
		Syntax:      `[[read_file path="drafts/doc.md"]]`,
		Rules:       `Paths are relative to workspace/ directory.`,
		Handler:     readFileTool,
	})
	tr.Register(&ToolDef{
		Name:        "list_files",
		Description: "List files and directories in the workspace. Use for checking what files exist, monitoring for changes.",
		Syntax:      `[[list_files path="drafts/"]]`,
		Rules:       `Paths are relative to workspace/ directory.`,
		Handler:     listFilesTool,
	})
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
func (tr *ToolRegistry) CoreDocs(includeMainOnly bool) string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()

	var sb strings.Builder
	sb.WriteString("CORE TOOLS — always available:\n")
	for _, tool := range tr.tools {
		if !tool.Core {
			continue
		}
		if tool.MainOnly && !includeMainOnly {
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
	for _, tool := range tr.tools {
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

// Dispatch executes a tool by name if it has a Handler. Returns result string and whether it was handled.
func (tr *ToolRegistry) Dispatch(name string, args map[string]string) (string, bool) {
	tr.mu.RLock()
	tool, exists := tr.tools[name]
	tr.mu.RUnlock()
	if !exists || tool.Handler == nil {
		return "", false
	}
	return tool.Handler(args), true
}

// AllToolNames returns all non-core tool names (for spawn docs).
func (tr *ToolRegistry) AllToolNames() []string {
	tr.mu.RLock()
	defer tr.mu.RUnlock()
	var names []string
	for _, tool := range tr.tools {
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
