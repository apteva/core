package main

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// MaxSpawnDepth is the maximum depth for sub-thread spawning.
// Main = depth -1 (conceptual), its children = 0, grandchildren = 1, etc.
const MaxSpawnDepth = 2

// baseThreadPromptTemplate is for leaf threads that cannot spawn.
const baseThreadPromptTemplate = `You are a SUB-THREAD (id="%s") in a continuous thinking engine. You were spawned by the %s thread.

IDENTITY:
- Your ID is "%s". You are NOT the main thread — you are a worker thread with a specific task.
- You cannot spawn other threads. You cannot restructure the system.
- You report results back to your parent via [[send id="parent" message="..."]].
- When done with current work, sleep until needed again: [[pace sleep="5m"]] or [[pace sleep="1h"]] etc.
- Only call [[done]] if you are certain this thread should never run again.

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Stay focused on YOUR directive. Do not try to take over coordination duties.
- Keep each thought concise — 1-2 short paragraphs max.

PACING — this is critical:
- Tool results (like [[list_files]] or [[web]]) will wake you up for the next thought. Do NOT set [[pace]] in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Example flow: Thought 1: call [[list_files]]. Thought 2: process results, [[send]] report, [[pace sleep="5m"]].
- Set sleep duration based on need: "2s" when actively working, "5m" when monitoring, "1h" for deep idle.
- Only use [[pace]] when you have NO pending tool calls and are ready to wait.

TIMING:
- You do NOT have precise timing control. Pace rates are approximate, not exact.
- For delayed tasks (like "do X in 5 minutes"), use [[pace rate="sleep"]] and act on the next wake-up. Do not overthink exact timing — approximate is fine.
- Never spiral trying to calculate exact seconds. Just set a pace close to the delay, wake up, do the action, done.

IMPORTANT — tool calls and [[done]]:
- NEVER call [[done]] in the same thought as a tool call. Tool results arrive in your NEXT thought.
- Always wait for tool results before calling [[done]] — you need to confirm the action succeeded.
- Example: Thought 1: [[pushover_send_notification ...]]. Thought 2: see result, confirm success, [[done]].`

// leaderThreadPromptTemplate is for threads that CAN spawn sub-threads (depth < MaxSpawnDepth).
const leaderThreadPromptTemplate = `You are a SUB-THREAD (id="%s") in a continuous thinking engine. You were spawned by the %s thread.

IDENTITY:
- Your ID is "%s". You are a team lead — you can spawn and manage your own sub-threads.
- You report results back to your parent via [[send id="parent" message="..."]].
- When done with current work, sleep until needed again: [[pace sleep="5m"]] or [[pace sleep="1h"]] etc.
- Only call [[done]] if you are certain this thread should never run again.

SPAWNING SUB-THREADS:
- Use [[spawn id="..." directive="..." tools="..."]] to create workers for parallel or long-running tasks.
- Use [[kill id="..."]] to stop a sub-thread.
- Use [[update id="..." directive="..." tools="..."]] to change a sub-thread's directive or tools.
- Your sub-threads report to YOU, not to main. You coordinate your team.
- The "directive" must be PLAIN NATURAL LANGUAGE. Never put tool call syntax in directives.

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Stay focused on YOUR directive. Delegate sub-tasks to your workers.
- Keep each thought concise — 1-2 short paragraphs max.

PACING — this is critical:
- Tool results (like [[list_files]] or [[web]]) will wake you up for the next thought. Do NOT set [[pace]] in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Set sleep duration based on need: "2s" when actively working, "5m" when monitoring, "1h" for deep idle.
- Only use [[pace]] when you have NO pending tool calls and are ready to wait.

TIMING:
- You do NOT have precise timing control. Pace rates are approximate, not exact.
- For delayed tasks, use [[pace rate="sleep"]] and act on the next wake-up.

IMPORTANT — tool calls and [[done]]:
- NEVER call [[done]] in the same thought as a tool call. Tool results arrive in your NEXT thought.
- Always wait for tool results before calling [[done]] — you need to confirm the action succeeded.`

type ThreadInfo struct {
	ID           string
	ParentID     string // "main" or parent thread ID
	Depth        int
	Directive    string
	Tools        []string
	Running      bool
	Iteration    int
	Rate         ThinkRate
	Model        ModelTier
	Provider     string // active provider name
	Started      time.Time
	ContextMsgs  int
	ContextChars int
	SubThreads   int // number of direct children
}

type Thread struct {
	ID           string
	ParentID     string // "main" or parent thread ID
	Depth        int    // 0 = child of main, 1 = grandchild, etc.
	Directive    string // original directive before tool docs
	Thinker      *Thinker
	Parent       *Thinker
	Children     *ThreadManager // non-nil if this thread can spawn (depth < MaxSpawnDepth)
	Tools        map[string]bool
	Started      time.Time
	initialParts []ContentPart // media to inject before first Run()
	doneForever  bool          // true if thread called [[done]] (permanent termination)
}

type ThreadManager struct {
	mu      sync.RWMutex
	threads map[string]*Thread
	parent  *Thinker
}

func NewThreadManager(parent *Thinker) *ThreadManager {
	return &ThreadManager{
		threads: make(map[string]*Thread),
		parent:  parent,
	}
}

// SpawnOpts holds optional parameters for spawning a thread.
type SpawnOpts struct {
	MediaParts      []ContentPart
	ProviderName    string   // override provider from pool (empty = inherit parent)
	InitialMessages []string
	ParentID        string   // "main" or parent thread ID (empty = "main")
	Depth           int      // depth in the spawn tree (0 = child of main)
	MCPNames        []string // MCP server names to connect (thread gets own connections)
	BuiltinTools    []string // provider builtin overrides (nil = inherit, empty = none)
}

// SpawnWithMedia creates a thread and injects media parts before it starts thinking.
func (tm *ThreadManager) SpawnWithMedia(id, directive string, tools []string, parts []ContentPart, initialMessages ...string) error {
	return tm.spawnInternal(id, directive, tools, SpawnOpts{MediaParts: parts, InitialMessages: initialMessages})
}

func (tm *ThreadManager) Spawn(id, directive string, tools []string, initialMessages ...string) error {
	return tm.spawnInternal(id, directive, tools, SpawnOpts{InitialMessages: initialMessages})
}

// SpawnWithOpts creates a thread with full options (provider, media, etc).
func (tm *ThreadManager) SpawnWithOpts(id, directive string, tools []string, opts SpawnOpts) error {
	return tm.spawnInternal(id, directive, tools, opts)
}

func (tm *ThreadManager) spawnInternal(id, directive string, tools []string, opts SpawnOpts) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.threads[id]; exists {
		return fmt.Errorf("thread %q already exists", id)
	}

	depth := opts.Depth
	parentID := opts.ParentID
	if parentID == "" {
		parentID = tm.parent.threadID // inherit from the owning thinker
	}

	// Enforce max spawn depth
	if depth > MaxSpawnDepth {
		return fmt.Errorf("max spawn depth (%d) exceeded", MaxSpawnDepth)
	}

	canSpawn := depth < MaxSpawnDepth

	// Build tool set
	toolSet := make(map[string]bool)
	for _, t := range tools {
		toolSet[strings.TrimSpace(t)] = true
	}
	toolSet["send"] = true
	toolSet["done"] = true
	toolSet["pace"] = true
	toolSet["evolve"] = true
	// Leaders get spawn/kill/update
	if canSpawn {
		toolSet["spawn"] = true
		toolSet["kill"] = true
		toolSet["update"] = true
	}

	// Build system prompt: use leader or worker template based on depth
	parentLabel := parentID
	if parentID == "main" {
		parentLabel = "main coordinator"
	}
	var basePrompt string
	if canSpawn {
		basePrompt = fmt.Sprintf(leaderThreadPromptTemplate, id, parentLabel, id)
	} else {
		basePrompt = fmt.Sprintf(baseThreadPromptTemplate, id, parentLabel, id)
	}
	coreDocs := ""
	if tm.parent.registry != nil {
		coreDocs = "\n" + tm.parent.registry.CoreDocs(false)
	}
	// Inject safety mode from parent config
	mode := tm.parent.config.GetMode()
	modeBlock := ""
	switch mode {
	case ModeCautious:
		modeBlock = "\n\n[SAFETY MODE: cautious]\nBefore executing any tool that modifies state, send a message to your parent explaining what you plan to do. Wait for confirmation before proceeding. Read-only tools can be used freely."
	case ModeLearn:
		modeBlock = "\n\n[SAFETY MODE: learn]\nFor every new type of tool call, send a message to your parent asking if the user is comfortable with it. Remember their preferences."
	}
	threadSystemPrompt := basePrompt + coreDocs + modeBlock + "\n\n[DIRECTIVE]\n" + directive

	thread := &Thread{
		ID:           id,
		ParentID:     parentID,
		Depth:        depth,
		Directive:    directive,
		Parent:       tm.parent,
		Tools:        toolSet,
		Started:      time.Now(),
		initialParts: opts.MediaParts,
	}

	// Create a Thinker — same struct as main, shares the bus and provider pool
	// Default: inherit parent's provider. Override via opts.ProviderName.
	threadProvider := tm.parent.provider
	if opts.ProviderName != "" && tm.parent.pool != nil {
		if p := tm.parent.pool.Get(opts.ProviderName); p != nil {
			threadProvider = p
		}
	}

	// Scope provider builtins if overridden (nil = inherit all, empty = none)
	if opts.BuiltinTools != nil && threadProvider != nil {
		threadProvider = threadProvider.WithBuiltins(opts.BuiltinTools)
	}

	// Build thread-local registry: core tools + allowed local tools + MCP tools
	// Auto-detect MCP server names from tool prefixes if not explicitly set.
	// e.g. tools="store_get_inventory,web" → auto-detects "store" as MCP server needed.
	mcpNames := opts.MCPNames
	if len(mcpNames) == 0 && tm.parent.config != nil {
		knownServers := map[string]bool{}
		for _, sc := range tm.parent.config.GetMCPServers() {
			knownServers[sc.Name] = true
		}
		// Also check parent's mcpCatalog
		for _, info := range tm.parent.mcpCatalog {
			knownServers[info.Name] = true
		}
		detected := map[string]bool{}
		for toolName := range toolSet {
			// Check if tool name has a known MCP server prefix (e.g. "store_get_inventory" → "store")
			for srv := range knownServers {
				if strings.HasPrefix(toolName, srv+"_") {
					detected[srv] = true
					break
				}
			}
		}
		for srv := range detected {
			mcpNames = append(mcpNames, srv)
		}
	}

	threadRegistry := tm.parent.registry
	threadAllowlist := toolSet
	var threadMCPServers []MCPConn

	if len(mcpNames) > 0 && tm.parent.registry != nil {
		// Create scoped registry with only core + allowed local tools
		threadRegistry = tm.parent.registry.NewScopedRegistry(toolSet)
		threadAllowlist = nil // not needed — registry IS the scope

		// Connect to each specified MCP server
		for _, mcpName := range mcpNames {
			// Look up MCP config from parent's config
			var cfg *MCPServerConfig
			for _, sc := range tm.parent.config.GetMCPServers() {
				if sc.Name == mcpName {
					c := sc
					cfg = &c
					break
				}
			}
			// Also check parent's live MCP connections for runtime-connected servers
			if cfg == nil {
				for _, srv := range tm.parent.mcpServers {
					if srv.GetName() == mcpName {
						// Already connected on parent — look up config
						for _, sc := range tm.parent.config.MCPServers {
							if sc.Name == mcpName {
								c := sc
								cfg = &c
								break
							}
						}
						break
					}
				}
			}
			if cfg == nil {
				logMsg("THREAD-MCP", fmt.Sprintf("%s: MCP server %q not found in config", id, mcpName))
				continue
			}

			srv, err := connectAnyMCP(*cfg)
			if err != nil {
				logMsg("THREAD-MCP", fmt.Sprintf("%s: connect %q: %v", id, mcpName, err))
				continue
			}
			mcpTools, err := srv.ListTools()
			if err != nil {
				logMsg("THREAD-MCP", fmt.Sprintf("%s: list tools %q: %v", id, mcpName, err))
				srv.Close()
				continue
			}
			// Register MCP tools into the thread's scoped registry
			for _, tool := range mcpTools {
				fullName := mcpName + "_" + tool.Name
				threadRegistry.Register(&ToolDef{
					Name:        fullName,
					Description: fmt.Sprintf("[%s] %s", mcpName, tool.Description),
					Syntax:      buildMCPSyntax(fullName, tool.InputSchema),
					Rules:       fmt.Sprintf("Provided by MCP server '%s'.", mcpName),
					Handler:     mcpProxyHandler(srv, tool.Name),
					InputSchema: tool.InputSchema,
					MCP:         false, // not filtered — this IS the thread's registry
					MCPServer:   mcpName,
				})
				// Add to tool set so spawn/allowlist tracking sees it
				toolSet[fullName] = true
			}
			threadMCPServers = append(threadMCPServers, srv)
			logMsg("THREAD-MCP", fmt.Sprintf("%s: connected %q (%d tools)", id, mcpName, len(mcpTools)))
		}
	}

	// Context window size based on role
	historyLimit := maxHistoryWorker
	if canSpawn {
		historyLimit = maxHistoryLead
	}

	thinker := &Thinker{
		apiKey:   tm.parent.apiKey,
		pool:     tm.parent.pool,
		provider: threadProvider,
		messages: []Message{
			{Role: "system", Content: threadSystemPrompt},
		},
		bus:       tm.parent.bus,
		sub:       tm.parent.bus.Subscribe(id, 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:       RateReactive,
		agentRate:  RateNormal,
		agentSleep: 10 * time.Second,
		maxHistory: historyLimit,
		memory:    tm.parent.memory,
		session:   NewSession(".", id),
		onStop:    func() { tm.cleanupThread(id) },
		threadID:  id,
		apiLog:        tm.parent.apiLog,
		apiMu:         tm.parent.apiMu,
		apiNotify:     tm.parent.apiNotify,
		registry:      threadRegistry,
		toolAllowlist: threadAllowlist,
		config:        tm.parent.config,
		mcpServers:    threadMCPServers,
		rebuildPrompt: func(toolDocs string) string {
			cd := ""
			if threadRegistry != nil {
				cd = "\n" + threadRegistry.CoreDocs(false)
			}
			var bp string
			if canSpawn {
				bp = fmt.Sprintf(leaderThreadPromptTemplate, id, parentLabel, id)
			} else {
				bp = fmt.Sprintf(baseThreadPromptTemplate, id, parentLabel, id)
			}
			prompt := bp + cd
			if toolDocs != "" {
				prompt += "\n" + toolDocs
			}
			// Inject active sub-threads for leaders
			if thread.Children != nil {
				children := thread.Children.List()
				if len(children) > 0 {
					prompt += "\n\n[ACTIVE SUB-THREADS]\n"
					for _, c := range children {
						age := time.Since(c.Started).Truncate(time.Second)
						prompt += fmt.Sprintf("- %s (running %s, iter #%d, pace %s)\n  directive: %s\n",
							c.ID, age, c.Iteration, c.Rate.String(), truncateStr(c.Directive, 150))
					}
				}
			}
			prompt += "\n\n[DIRECTIVE]\n" + thread.Directive
			return prompt
		},
	}
	thread.Thinker = thinker
	thinker.telemetry = tm.parent.telemetry // share telemetry

	// Set up Children ThreadManager for leaders (depth < MaxSpawnDepth)
	if canSpawn {
		thread.Children = NewThreadManager(thinker)
		thinker.threads = thread.Children
	}

	// Set tool handler AFTER Children is set up (handler references thread.Children)
	thinker.handleTools = threadToolHandler(thread, tm)

	// Load conversation history from persistent session (for respawned threads)
	if saved, summaries := thinker.session.LoadTail(defaultLoadTail); len(saved) > 0 {
		if len(summaries) > 0 {
			contextBlock := "\n\n[PREVIOUS CONTEXT]\n"
			for _, s := range summaries {
				contextBlock += s + "\n"
			}
			thinker.messages[0].Content += contextBlock
		}
		thinker.messages = append(thinker.messages, saved...)
		logMsg("THREAD", fmt.Sprintf("%s loaded %d messages from history (%d compacted summaries)", id, len(saved), len(summaries)))
	}

	tm.threads[id] = thread

	// Inject initial messages before starting so first thought picks them up
	for _, msg := range opts.InitialMessages {
		tm.parent.bus.Publish(Event{Type: EventInbox, To: id, Text: msg})
	}

	// Inject initial media parts if provided (before Run starts)
	if thread.initialParts != nil {
		tm.parent.bus.Publish(Event{
			Type:  EventInbox,
			To:    id,
			Text:  "[media] attached",
			Parts: thread.initialParts,
		})
		thread.initialParts = nil
	}

	// Same Run() as the main thinker — no duplicated loop
	go thinker.Run()

	provName := "unknown"
	if threadProvider != nil {
		provName = threadProvider.Name()
	}
	role := "worker"
	if canSpawn {
		role = "leader"
	}
	tm.parent.bus.Publish(Event{Type: EventThreadStart, From: id, Text: fmt.Sprintf("Thread %q spawned (provider: %s, role: %s, depth: %d)", id, provName, role, depth)})
	toolList := toolSetToSlice(thread.Tools)
	tm.parent.Inject(fmt.Sprintf("[thread:%s] started (provider: %s, role: %s, tools: %s)", id, provName, role, strings.Join(toolList, ", ")))
	tm.parent.logAPI(APIEvent{Type: "thread_started", ThreadID: id})

	// Telemetry: thread.spawn
	if tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.spawn", id, ThreadSpawnData{
			ParentID:  parentID,
			Directive: directive,
			Tools:     tools,
		})
	}

	return nil
}

// resolveTarget resolves "parent" to the actual parent ID, and routes the message.
// Returns the resolved target ID and whether the send succeeded.
func (thread *Thread) resolveSend(tm *ThreadManager, tagged string, targetID string) bool {
	// "parent" alias → route to parent thinker
	if targetID == "parent" || targetID == thread.ParentID {
		thread.Parent.Inject(tagged)
		return true
	}
	// "main" always goes to main (even from grandchildren)
	if targetID == "main" {
		// Walk up to the root thinker
		t := thread.Parent
		for t.threadID != "main" {
			// Find parent's parent — but we don't have a direct ref, so just use the bus
			break
		}
		// Use the bus to deliver to main directly
		thread.Parent.bus.Publish(Event{Type: EventInbox, To: "main", Text: tagged})
		return true
	}
	// Try own children first
	if thread.Children != nil {
		if thread.Children.Send(targetID, tagged) {
			return true
		}
	}
	// Try sibling threads (same ThreadManager)
	if tm.Send(targetID, tagged) {
		return true
	}
	return false
}

// threadToolHandler returns a ToolHandler scoped to a thread's allowed tools.
func threadToolHandler(thread *Thread, tm *ThreadManager) ToolHandler {
	return func(t *Thinker, calls []toolCall, _ []string) ([]string, []string, []ToolResult) {
		var replies []string
		var toolNames []string
		var results []ToolResult
		var doneMsg *string
		var doneCallID string

		addResult := func(callID, content string) {
			if callID != "" {
				results = append(results, ToolResult{CallID: callID, Content: content})
			}
		}

		for _, call := range calls {
			delete(call.Args, "_reason")
			if !thread.Tools[call.Name] {
				continue
			}
			switch call.Name {
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				if id != "" && msg != "" {
					tagged := fmt.Sprintf("[from:%s] %s", thread.ID, msg)
					logMsg("THREAD", fmt.Sprintf("%s send to=%s msg=%q", thread.ID, id, msg))
					if !thread.resolveSend(tm, tagged, id) {
						t.Inject(fmt.Sprintf("[error] thread %q not found", id))
					}
					if t.telemetry != nil {
						resolvedID := id
						if id == "parent" {
							resolvedID = thread.ParentID
						}
						t.telemetry.Emit("thread.message", thread.ID, ThreadMessageData{From: thread.ID, To: resolvedID, Message: msg})
					}
					addResult(call.NativeID, fmt.Sprintf("sent to %s", id))
				}
			case "spawn":
				// Leaders only (depth < MaxSpawnDepth) — enforced by tool allowlist
				if thread.Children == nil {
					addResult(call.NativeID, "error: cannot spawn (not a leader thread)")
					break
				}
				sid := call.Args["id"]
				directive := call.Args["directive"]
				if directive == "" {
					directive = call.Args["prompt"]
				}
				toolsStr := call.Args["tools"]
				var spawnTools []string
				if toolsStr != "" {
					spawnTools = strings.Split(toolsStr, ",")
				}
				providerName := call.Args["provider"]
				// MCP scoping
				var mcpNames []string
				if mcpStr := call.Args["mcp"]; mcpStr != "" {
					for _, name := range strings.Split(mcpStr, ",") {
						if n := strings.TrimSpace(name); n != "" {
							mcpNames = append(mcpNames, n)
						}
					}
				}
				// Provider builtin scoping
				var builtinTools []string
				if btStr, hasBuiltins := call.Args["builtins"]; hasBuiltins {
					if btStr == "" {
						builtinTools = []string{}
					} else {
						for _, bt := range strings.Split(btStr, ",") {
							if b := strings.TrimSpace(bt); b != "" {
								builtinTools = append(builtinTools, b)
							}
						}
					}
				}
				if sid != "" && directive != "" {
					err := thread.Children.SpawnWithOpts(sid, directive, spawnTools, SpawnOpts{
						ProviderName: providerName,
						ParentID:     thread.ID,
						Depth:        thread.Depth + 1,
						MCPNames:     mcpNames,
						BuiltinTools: builtinTools,
					})
					if err != nil {
						addResult(call.NativeID, fmt.Sprintf("error: %v", err))
					} else {
						t.config.SaveThread(PersistentThread{
							ID: sid, ParentID: thread.ID, Depth: thread.Depth + 1,
							Directive: directive, Tools: spawnTools,
						})
						addResult(call.NativeID, fmt.Sprintf("thread %s spawned (depth %d)", sid, thread.Depth+1))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "kill":
				sid := call.Args["id"]
				if sid != "" && thread.Children != nil {
					thread.Children.Kill(sid)
					t.config.RemoveThread(sid)
					addResult(call.NativeID, fmt.Sprintf("thread %s killed", sid))
				}
				toolNames = append(toolNames, call.Raw)
			case "update":
				sid := call.Args["id"]
				if sid != "" && thread.Children != nil {
					directive := call.Args["directive"]
					toolsStr := call.Args["tools"]
					var updateTools []string
					if toolsStr != "" {
						updateTools = strings.Split(toolsStr, ",")
					}
					if err := thread.Children.Update(sid, directive, updateTools); err != nil {
						addResult(call.NativeID, fmt.Sprintf("error: %v", err))
					} else {
						if directive != "" {
							thread.Children.Send(sid, fmt.Sprintf("[directive updated] %s", directive))
						}
						addResult(call.NativeID, fmt.Sprintf("thread %s updated", sid))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "done":
				msg := call.Args["message"]
				doneMsg = &msg
				doneCallID = call.NativeID
			case "pace":
				var parts []string
				if s := call.Args["sleep"]; s != "" {
					if d, ok := parseSleepDuration(s); ok {
						t.agentSleep = d
						t.agentRate = RateSleep
						parts = append(parts, "sleep="+s)
					}
				} else if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
					if d, ok2 := rateAliases[call.Args["rate"]]; ok2 {
						t.agentSleep = d
					}
					parts = append(parts, "rate="+call.Args["rate"])
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
					parts = append(parts, "model="+call.Args["model"])
				}
				if pn := call.Args["provider"]; pn != "" && t.pool != nil {
					if p := t.pool.Get(pn); p != nil {
						t.provider = p
						parts = append(parts, "provider="+pn)
					}
				}
				if len(parts) > 0 {
					addResult(call.NativeID, "set "+strings.Join(parts, " "))
				} else {
					addResult(call.NativeID, "ok")
				}
			case "evolve":
				if d := call.Args["directive"]; d != "" {
					thread.Directive = d
					if t.rebuildPrompt != nil {
						t.messages[0] = Message{Role: "system", Content: t.rebuildPrompt("")}
					}
					tm.parent.config.SaveThread(PersistentThread{
						ID: thread.ID, ParentID: thread.ParentID, Depth: thread.Depth,
						Directive: d, Tools: toolSetToSlice(thread.Tools),
					})
					t.logAPI(APIEvent{Type: "evolved", ThreadID: thread.ID, Message: d})
					addResult(call.NativeID, "directive updated")
				}
			case "remember":
				if text := call.Args["text"]; text != "" && t.memory != nil {
					ns := thread.ID // namespace = thread ID
					go func(txt, namespace string) {
						if err := t.memory.StoreWithNamespace(txt, namespace); err != nil {
							t.Inject(fmt.Sprintf("[remember] error: %v", err))
						}
					}(text, ns)
					addResult(call.NativeID, "stored")
				}
			default:
				executeTool(t, call)
				toolNames = append(toolNames, call.Raw)
			}
		}

		if doneMsg != nil {
			addResult(doneCallID, "stopping")
			logMsg("THREAD", fmt.Sprintf("%s calling done, msg=%q", thread.ID, *doneMsg))
			thread.doneForever = true // mark for permanent cleanup (deletes session)
			if *doneMsg != "" {
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done] %s", thread.ID, *doneMsg))
			} else {
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done]", thread.ID))
			}
			t.Stop()
		}

		return replies, toolNames, results
	}
}

func (tm *ThreadManager) Kill(id string) {
	tm.mu.RLock()
	thread, exists := tm.threads[id]
	tm.mu.RUnlock()
	if !exists {
		return
	}
	thread.Thinker.Stop()
	// Wait briefly for cleanup
	for i := 0; i < 20; i++ {
		time.Sleep(50 * time.Millisecond)
		tm.mu.RLock()
		_, still := tm.threads[id]
		tm.mu.RUnlock()
		if !still {
			return
		}
	}
	// Force cleanup if still lingering
	tm.cleanupThread(id)
}

func (tm *ThreadManager) KillAll() {
	tm.mu.RLock()
	ids := make([]string, 0, len(tm.threads))
	for id := range tm.threads {
		ids = append(ids, id)
	}
	tm.mu.RUnlock()
	for _, id := range ids {
		tm.Kill(id)
	}
}

func (tm *ThreadManager) Send(id, message string) bool {
	return tm.SendWithParts(id, message, nil)
}

func (tm *ThreadManager) SendWithParts(id, message string, parts []ContentPart) bool {
	tm.mu.RLock()
	_, exists := tm.threads[id]
	tm.mu.RUnlock()
	if !exists {
		return false
	}
	tm.parent.bus.Publish(Event{Type: EventInbox, To: id, Text: message, Parts: parts})
	return true
}

func (tm *ThreadManager) List() []ThreadInfo {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	var infos []ThreadInfo
	for _, t := range tm.threads {
		providerName := ""
		if t.Thinker.provider != nil {
			providerName = t.Thinker.provider.Name()
		}
		subCount := 0
		if t.Children != nil {
			subCount = t.Children.Count()
		}
		infos = append(infos, ThreadInfo{
			ID:        t.ID,
			ParentID:  t.ParentID,
			Depth:     t.Depth,
			Directive: t.Directive,
			Tools:     toolSetToSlice(t.Tools),
			Running:   true,
			Iteration: t.Thinker.iteration,
			Rate:         t.Thinker.rate,
			Model:        t.Thinker.model,
			Provider:     providerName,
			Started:      t.Started,
			ContextMsgs:  len(t.Thinker.messages),
			ContextChars: func() int { n := 0; for _, m := range t.Thinker.messages { n += len(m.Content) }; return n }(),
			SubThreads:   subCount,
		})
	}
	return infos
}

func (tm *ThreadManager) Count() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.threads)
}

// Update changes a thread's directive and/or tools. Rebuilds the system prompt immediately.
func (tm *ThreadManager) Update(id, directive string, tools []string) error {
	tm.mu.RLock()
	thread, exists := tm.threads[id]
	tm.mu.RUnlock()
	if !exists {
		return fmt.Errorf("thread %q not found", id)
	}

	if directive != "" {
		thread.Directive = directive
	}
	if len(tools) > 0 {
		toolSet := make(map[string]bool)
		for _, t := range tools {
			toolSet[strings.TrimSpace(t)] = true
		}
		// Always include builtins
		for _, b := range []string{"send", "done", "pace", "evolve", "remember"} {
			toolSet[b] = true
		}
		thread.Tools = toolSet
		thread.Thinker.toolAllowlist = toolSet
	}

	// Rebuild system prompt
	if thread.Thinker.rebuildPrompt != nil {
		thread.Thinker.messages[0] = Message{Role: "system", Content: thread.Thinker.rebuildPrompt("")}
	}

	// Persist
	tm.parent.config.SaveThread(PersistentThread{
		ID: id, ParentID: thread.ParentID, Depth: thread.Depth,
		Directive: thread.Directive, Tools: toolSetToSlice(thread.Tools),
	})

	return nil
}

// PauseAll pauses or resumes all child threads.
func (tm *ThreadManager) PauseAll(paused bool) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for _, thread := range tm.threads {
		t := thread.Thinker
		if t.paused != paused {
			select {
			case <-t.pause:
			default:
			}
			t.pause <- paused
			t.paused = paused
		}
	}
}

func (tm *ThreadManager) cleanupThread(id string) {
	logMsg("THREAD", fmt.Sprintf("%s cleanupThread start", id))
	tm.mu.Lock()
	thread := tm.threads[id]
	delete(tm.threads, id)
	tm.mu.Unlock()

	// Cascade: kill all children first
	if thread != nil && thread.Children != nil {
		logMsg("THREAD", fmt.Sprintf("%s killing %d children", id, thread.Children.Count()))
		thread.Children.KillAll()
	}

	// Close thread-local MCP connections
	if thread != nil && thread.Thinker != nil {
		for _, srv := range thread.Thinker.mcpServers {
			logMsg("THREAD-MCP", fmt.Sprintf("%s closing MCP %s", id, srv.GetName()))
			srv.Close()
		}
		thread.Thinker.mcpServers = nil
	}

	// Only delete session history if thread called [[done]] (permanent termination).
	// For kills/restarts, keep the session so the thread can restore context on respawn.
	if thread != nil && thread.doneForever && thread.Thinker.session != nil {
		thread.Thinker.session.Delete()
	}

	parentID := "main"
	if thread != nil {
		parentID = thread.ParentID
	}

	tm.parent.config.RemoveThread(id)
	logMsg("THREAD", fmt.Sprintf("%s publishing EventThreadDone from cleanup", id))
	tm.parent.bus.Publish(Event{Type: EventThreadDone, From: id})
	logMsg("THREAD", fmt.Sprintf("%s unsubscribing from bus", id))
	tm.parent.bus.Unsubscribe(id)
	tm.parent.logAPI(APIEvent{Type: "thread_done", ThreadID: id})

	// Telemetry: thread.done
	if tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.done", id, ThreadDoneData{
			ParentID: parentID,
		})
	}
}

func toolSetToSlice(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
