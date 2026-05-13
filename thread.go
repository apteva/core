package core

import (
	"context"
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
- You MUST report results to your parent via send(id="parent", message="...") BEFORE pacing to a long sleep. Your parent is waiting for that reply — silent completion leaves them blocked. A final send is mandatory at the end of every task, even if the work was trivial or everything was already done.
- When done with current work AND after sending your result, sleep until needed again: pace(sleep="5m") or pace(sleep="1h") etc.
- Only call done if you are certain this thread should never run again.

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Stay focused on YOUR directive. Do not try to take over coordination duties.
- Keep each thought concise — 1-2 short paragraphs max.
- If you have no events to process, just sleep. Silence is normal — do not invent emergencies or report false failures.

PACING — this is critical:
- Tool results (like list_files or web) will wake you up for the next thought. Do NOT set pace in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Example flow: Thought 1: call list_files. Thought 2: process results, send report, pace(sleep="5m").
- Set sleep duration based on need: "2s" when actively working, "5m" when monitoring, "1h" for deep idle.
- Only use pace when you have NO pending tool calls and are ready to wait.

TIMING:
- You do NOT have precise timing control. Pace rates are approximate, not exact.
- For delayed tasks (like "do X in 5 minutes"), use pace(rate="sleep") and act on the next wake-up. Do not overthink exact timing — approximate is fine.
- Never spiral trying to calculate exact seconds. Just set a pace close to the delay, wake up, do the action, done.

IMPORTANT — tool calls and done:
- NEVER call done in the same thought as a tool call. Tool results arrive in your NEXT thought.
- Always wait for tool results before calling done — you need to confirm the action succeeded.
- Example: Thought 1: pushover_send_notification(...). Thought 2: see result, confirm success, done.`

// leaderThreadPromptTemplate is for threads that CAN spawn sub-threads (depth < MaxSpawnDepth).
const leaderThreadPromptTemplate = `You are a SUB-THREAD (id="%s") in a continuous thinking engine. You were spawned by the %s thread.

IDENTITY:
- Your ID is "%s". You are a team lead — you can spawn and manage your own sub-threads.
- You MUST report results to your parent via send(id="parent", message="...") BEFORE pacing to a long sleep. Your parent is waiting for that reply — silent completion leaves them blocked. A final send is mandatory at the end of every task, even if the work was trivial or everything was already done.
- When done with current work AND after sending your result, sleep until needed again: pace(sleep="5m") or pace(sleep="1h") etc.
- Only call done if you are certain this thread should never run again.

SPAWNING SUB-THREADS:
- Use spawn(id="..." directive="..." tools="...") to create workers for parallel or long-running tasks.
- Use kill(id="...") to stop a sub-thread.
- Use update(id="..." directive="..." tools="...") to change a sub-thread's directive or tools.
- Your sub-threads report to YOU, not to main. You coordinate your team.
- The "directive" must be PLAIN NATURAL LANGUAGE. Never put tool call syntax in directives.
- NEVER spawn a replacement for a thread that already exists. Threads sleep — silence is normal, not a crash.
- NEVER spawn threads with new IDs to "work around" a slow thread. Wait patiently or send it a message.
- Only spawn threads that are defined in your team. Do not invent new thread IDs.

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Stay focused on YOUR directive. Delegate sub-tasks to your workers.
- Keep each thought concise — 1-2 short paragraphs max.

PACING — this is critical:
- Tool results (like list_files or web) will wake you up for the next thought. Do NOT set pace in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Set sleep duration based on need: "2s" when actively working, "5m" when monitoring, "1h" for deep idle.
- Only use pace when you have NO pending tool calls and are ready to wait.

TIMING:
- You do NOT have precise timing control. Pace rates are approximate, not exact.
- For delayed tasks, use pace(rate="sleep") and act on the next wake-up.

IMPORTANT — tool calls and done:
- NEVER call done in the same thought as a tool call. Tool results arrive in your NEXT thought.
- Always wait for tool results before calling done — you need to confirm the action succeeded.`

type ThreadInfo struct {
	ID           string
	Name         string // human-readable display label; empty = render id
	ParentID     string // "main" or parent thread ID
	Depth        int
	Directive    string
	Tools        []string
	MCPNames     []string
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
	Name         string // human-readable label, separate from ID. ID is immutable;
	                    // Name can be edited via update without touching parent_id
	                    // references or session storage. Empty means "use ID for display".
	ParentID     string // "main" or parent thread ID
	Depth        int    // 0 = child of main, 1 = grandchild, etc.
	Directive    string // original directive before tool docs
	MCPNames     []string // MCP server names this thread connected to
	Thinker      *Thinker
	Realtime     *RealtimeThinker // non-nil for realtime (voice/audio) threads; runs in place of Thinker.Run
	Parent       *Thinker
	Children     *ThreadManager // non-nil if this thread can spawn (depth < MaxSpawnDepth)
	Tools        map[string]bool
	Started      time.Time
	initialParts []ContentPart // media to inject before first Run()
	doneForever  bool          // true if thread called done (permanent termination)
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
	DeferRun        bool     // if true, don't start Run() — call StartAll() later
	// Paused: if true, the thread spawns in paused state. Run() loop
	// blocks at the top of its first iteration until either an inbox
	// event arrives (an explicit `send` from the leader) OR the
	// thinker is unpaused via PauseAll(false). Useful for
	//   - "configure-then-launch" patterns where the leader spawns
	//     several workers atomically before any of them think
	//   - cautious/learn modes that want children to wait for explicit
	//     instruction rather than acting on the directive alone
	//   - debugging — inspect the worker before it does anything
	Paused bool
	// BypassNoSpawn skips the no_spawn MCP filter. Set by the
	// authenticated HTTP spawn endpoint (POST /threads/{id}) where
	// the caller has the core API key — the system itself is asking
	// for a privileged sub-thread (e.g. channelchat's chat thread
	// needs the `channels` MCP to reply to users). The LLM-driven
	// spawn tool path never sets this, so an in-agent worker still
	// can't escalate by attaching a no_spawn MCP.
	BypassNoSpawn bool
	// Realtime: if true, construct a realtime (voice/audio) thread
	// driven by a RealtimeProvider session instead of the standard
	// request/response Thinker. ProviderName must resolve to a
	// registered RealtimeProvider (e.g. "openai-realtime"); when
	// empty, the pool's RealtimeDefault is used. SpawnWithOpts will
	// refuse with a clear error if no realtime provider is available.
	Realtime bool
	// Voice: realtime voice id (e.g. "alloy"). Empty = provider's
	// default. Ignored when Realtime is false.
	Voice string
	// AudioIn: PCM audio chunks pushed by the caller (telephony
	// bridge, browser WebRTC, mic source). The realtime thread reads
	// these and forwards to session.SendAudio. nil = no inbound audio
	// (text-only realtime — useful for tests). Ignored when
	// Realtime is false.
	AudioIn <-chan []byte
	// AudioOut: PCM audio chunks the realtime thread writes when the
	// model speaks. Caller plays/streams them. nil = audio output
	// silently dropped. Ignored when Realtime is false.
	AudioOut chan<- []byte
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
	logMsg("SPAWN", fmt.Sprintf("enter id=%q parent=%q depth=%d tools=%v mcps=%v", id, opts.ParentID, opts.Depth, tools, opts.MCPNames))
	tm.mu.Lock()
	defer tm.mu.Unlock()
	logMsg("SPAWN", fmt.Sprintf("acquired tm.mu id=%q", id))

	if _, exists := tm.threads[id]; exists {
		logMsg("SPAWN", fmt.Sprintf("reject id=%q: already exists in this manager", id))
		return fmt.Errorf("thread %q already exists", id)
	}
	// Also check the entire tree — prevent duplicates across hierarchy levels
	if threadExistsInTree(tm, id) {
		logMsg("SPAWN", fmt.Sprintf("reject id=%q: already exists elsewhere in tree", id))
		return fmt.Errorf("thread %q already exists in tree", id)
	}
	logMsg("SPAWN", fmt.Sprintf("passed existence checks id=%q", id))

	// Realtime: pre-validate provider availability. Actual session
	// opening + RealtimeThinker construction happens after the
	// regular Thinker is built (we reuse its registry, bus
	// subscription, telemetry, etc.). Resolving the provider here
	// means we fail fast before doing the expensive Thinker setup if
	// the request is misconfigured.
	var realtimeProvider RealtimeProvider
	if opts.Realtime {
		pool := tm.parent.pool
		if pool == nil || !pool.HasRealtimeProvider() {
			return fmt.Errorf("realtime spawn refused: no realtime provider registered in pool")
		}
		if opts.ProviderName != "" {
			realtimeProvider = pool.RealtimeByName(opts.ProviderName)
			if realtimeProvider == nil {
				return fmt.Errorf("realtime spawn refused: provider %q not registered as a realtime provider", opts.ProviderName)
			}
		} else {
			realtimeProvider = pool.RealtimeDefault()
			if realtimeProvider == nil {
				return fmt.Errorf("realtime spawn refused: no default realtime provider")
			}
		}
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

	// canSpawn only if depth allows AND spawn was explicitly in the tools list
	wantsSpawn := false
	for _, t := range tools {
		if strings.TrimSpace(t) == "spawn" {
			wantsSpawn = true
			break
		}
	}
	canSpawn := depth < MaxSpawnDepth && wantsSpawn

	// Check if this is a system thread (e.g. unconscious) and recover the
	// persistent display Name if one was set on a previous run.
	isSystem := false
	persistedName := ""
	for _, pt := range tm.parent.config.GetThreads() {
		if pt.ID == id {
			isSystem = pt.System
			persistedName = pt.Name
			break
		}
	}

	// Build tool set
	toolSet := make(map[string]bool)
	for _, t := range tools {
		toolSet[strings.TrimSpace(t)] = true
	}
	if !isSystem {
		toolSet["send"] = true
		toolSet["done"] = true
	}
	toolSet["pace"] = true
	if !isSystem {
		toolSet["evolve"] = true
	}
	// Leaders get kill/update only if spawn was explicitly requested
	if canSpawn && toolSet["spawn"] {
		toolSet["kill"] = true
		toolSet["update"] = true
	} else {
		// Not a leader — remove spawn even if canSpawn by depth
		delete(toolSet, "spawn")
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
		// Same compact-vs-full trade-off as buildSystemPrompt: if the
		// sub-thread's provider supports native tools, the schemas are
		// already in tools[] and we skip duplicating them in prose.
		if poolSupportsNativeTools(tm.parent.pool) {
			coreDocs = "\n" + tm.parent.registry.CoreDocsSummary(false)
		} else {
			coreDocs = "\n" + tm.parent.registry.CoreDocs(false)
		}
	}
	// Inject safety mode from parent config. Child-thread wording is a
	// tighter version of the main-thread prompt: the child escalates to
	// its PARENT (not the user directly).
	mode := tm.parent.config.GetMode()
	modeBlock := ""
	switch mode {
	case ModeCautious:
		modeBlock = "\n\n[SAFETY MODE: cautious]\nRead-only tools are free. Before any state-changing tool (exec, write, delete, deploy, restart, external send), send one concise `send` to your parent with action + target + why, and wait for their next message. If unsure whether an action is state-changing, ask."
	case ModeLearn:
		modeBlock = "\n\n[SAFETY MODE: learn]\nSoft gate — no runtime block, the discipline is on you. DEFAULT: before any action you haven't taken before this session, `send` a one-line check to your parent — \"About to <verb> <target>. Reason: <one sentence>. OK?\" — and wait. This applies to EVERY tool — reads, file IO, exec, browser, thread spawning, channel sends — except `pace` (loop control, never gated). Once approved on a scope, reuse freely on the same scope without re-asking."
	default: // ModeAutonomous
		modeBlock = "\n\n[SAFETY MODE: autonomous]\nDecide yourself. For irreversible or high-blast-radius actions, inform your parent briefly before acting. Stop and adjust the moment a correction comes back. ACT, DON'T NARRATE — your parent only sees what you `send` or `done` with; prose between tool calls is not observed by anyone, so skip it. Take the next tool call, let the result guide the next."
	}
	threadSystemPrompt := basePrompt + coreDocs + modeBlock + "\n\n[DIRECTIVE]\n" + directive

	thread := &Thread{
		ID:           id,
		Name:         persistedName,
		ParentID:     parentID,
		Depth:        depth,
		Directive:    directive,
		MCPNames:     opts.MCPNames,
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
	//
	// Strip any MCP whose config carries no_spawn=true. The host marks
	// infrastructure-level servers (gateways, outbound bridges) with
	// that flag so a worker spawned by the LLM can't escalate by
	// attaching them via spawn(mcp="..."). Core has no opinion about
	// which names are privileged — it only honors the flag.
	mcpNames := opts.MCPNames
	if !opts.BypassNoSpawn && len(mcpNames) > 0 && tm.parent.config != nil {
		noSpawn := map[string]bool{}
		for _, sc := range tm.parent.config.GetMCPServers() {
			if sc.NoSpawn {
				noSpawn[sc.Name] = true
			}
		}
		filtered := mcpNames[:0]
		for _, n := range mcpNames {
			if noSpawn[n] {
				logMsg("SPAWN", fmt.Sprintf("%s: refusing no-spawn MCP %q on sub-thread", id, n))
				continue
			}
			filtered = append(filtered, n)
		}
		mcpNames = filtered
	}
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
					Handler:     mcpProxyHandler(srv, tool.Name, tm.parent.blobs),
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
		rebuildPrompt: func(_ string) string {
			cd := ""
			if threadRegistry != nil {
				if poolSupportsNativeTools(tm.parent.pool) {
					cd = "\n" + threadRegistry.CoreDocsSummary(false)
				} else {
					cd = "\n" + threadRegistry.CoreDocs(false)
				}
			}
			var bp string
			if canSpawn {
				bp = fmt.Sprintf(leaderThreadPromptTemplate, id, parentLabel, id)
			} else {
				bp = fmt.Sprintf(baseThreadPromptTemplate, id, parentLabel, id)
			}
			prompt := bp + cd
			// Active sub-threads + RAG-retrieved toolDocs USED to render
			// here — they busted the cache every iteration. Both now ride
			// in the per-turn user message via buildDynamicTurnContext
			// (same path as main thread's Run loop), keeping leader
			// messages[0] cache-stable.
			// Show available MCP servers so leaders know what to assign when spawning
			if canSpawn {
				var mcpList []string
				for _, cfg := range tm.parent.config.GetMCPServers() {
					if !cfg.MainAccess {
						mcpList = append(mcpList, cfg.Name)
					}
				}
				if len(mcpList) > 0 {
					prompt += "\n\n[AVAILABLE MCP SERVERS — use mcp=\"name\" when spawning]\n"
					for _, name := range mcpList {
						prompt += "- " + name + "\n"
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

	// Start the thinking loop (unless deferred for batch respawn).
	// Paused workers start their goroutine but block at the top of
	// Run() until an inbox event arrives or PauseAll(false) wakes them.
	if !opts.DeferRun {
		if opts.Paused {
			thinker.paused = true
			logMsg("SPAWN", fmt.Sprintf("starting Run() PAUSED for id=%q tools=%d mcps=%d", id, len(toolSet), len(threadMCPServers)))
		} else {
			logMsg("SPAWN", fmt.Sprintf("starting Run() for id=%q tools=%d mcps=%d", id, len(toolSet), len(threadMCPServers)))
		}
		// Realtime threads run an event-driven loop against a live
		// WebSocket session instead of the standard request/response
		// iteration. The Thinker remains constructed (shared state:
		// registry, bus, telemetry) so all the late-result and
		// tool-dispatch machinery works identically.
		if opts.Realtime && realtimeProvider != nil {
			rt, err := startRealtimeThinker(
				context.Background(),
				thinker,
				realtimeProvider,
				directive,
				opts.Voice,
				opts.AudioIn,
				opts.AudioOut,
			)
			if err != nil {
				logMsg("REALTIME", fmt.Sprintf("[%s] open failed: %v", id, err))
				// Tear down the Thinker shell we built — its session
				// goroutines never started, but we registered the
				// thread in tm.threads above and that lookup needs
				// cleanup so a retry with the same id succeeds.
				delete(tm.threads, id)
				return fmt.Errorf("realtime open: %w", err)
			}
			thread.Realtime = rt
			go rt.Run()
		} else {
			go thinker.Run()
		}
	} else {
		logMsg("SPAWN", fmt.Sprintf("deferred Run() for id=%q (batch respawn)", id))
	}

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
func (thread *Thread) resolveSend(tm *ThreadManager, tagged string, targetID string, parts ...[]ContentPart) bool {
	var mediaParts []ContentPart
	if len(parts) > 0 {
		mediaParts = parts[0]
	}
	// "parent" alias → route to parent thinker
	if targetID == "parent" || targetID == thread.ParentID {
		if len(mediaParts) > 0 {
			thread.Parent.InjectWithParts(tagged, mediaParts)
		} else {
			thread.Parent.Inject(tagged)
		}
		return true
	}
	// "main" always goes to main (even from grandchildren)
	if targetID == "main" {
		thread.Parent.bus.Publish(Event{Type: EventInbox, To: "main", Text: tagged, Parts: mediaParts})
		return true
	}
	// Try own children first
	if thread.Children != nil {
		if thread.Children.SendWithParts(targetID, tagged, mediaParts) {
			return true
		}
	}
	// Try sibling threads (same ThreadManager)
	if tm.SendWithParts(targetID, tagged, mediaParts) {
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
		// Emit tool.result telemetry for inline tools
		emitResult := func(call toolCall, content string) {
			addResult(call.NativeID, content)
			if t.telemetry != nil {
				t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
					ID: call.NativeID, Name: call.Name, Success: true, Result: content,
				})
			}
		}

		for _, call := range calls {
			if !thread.Tools[call.Name] {
				continue
			}
			// Check if inline or registry tool
			isInline := true
			switch call.Name {
			case "send", "spawn", "kill", "update", "evolve", "remember", "pace", "done":
				// inline
			default:
				isInline = false // executeTool handles _reason and telemetry
			}

			reason := ""
			if isInline {
				reason = call.Args["_reason"]
				delete(call.Args, "_reason")
			}
			if isInline && t.telemetry != nil {
				t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
					ID: call.NativeID, Name: call.Name, Args: call.Args, Reason: reason,
				})
			}
			switch call.Name {
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				mediaStr := call.Args["media"]
				if id == "" || msg == "" {
					// Silent no-op on missing args would leave the LLM
					// believing it sent — and the parent thread waiting
					// forever for a reply. Surface the mistake so the
					// LLM retries next iteration.
					emitResult(call, fmt.Sprintf("error: send requires both id and message (got id=%q, message_len=%d)", id, len(msg)))
				} else {
					tagged := fmt.Sprintf("[from:%s] %s", thread.ID, msg)
					mediaParts := parseMediaURLs(mediaStr)
					logMsg("THREAD", fmt.Sprintf("%s send to=%s msg=%q media=%d", thread.ID, id, msg, len(mediaParts)))
					if !thread.resolveSend(tm, tagged, id, mediaParts) {
						t.Inject(fmt.Sprintf("[error] thread %q not found", id))
					}
					if t.telemetry != nil {
						resolvedID := id
						if id == "parent" {
							resolvedID = thread.ParentID
						}
						t.telemetry.Emit("thread.message", thread.ID, ThreadMessageData{From: thread.ID, To: resolvedID, Message: msg})
					}
					emitResult(call, fmt.Sprintf("sent to %s", id))
				}
			case "spawn":
				// Leaders only (depth < MaxSpawnDepth) — enforced by tool allowlist
				if thread.Children == nil {
					emitResult(call, "error: cannot spawn (not a leader thread)")
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
				paused := parseTruthy(call.Args["paused"])
				if sid == "" || directive == "" {
					emitResult(call, fmt.Sprintf("error: spawn requires both id and directive (got id=%q, directive_len=%d)", sid, len(directive)))
				} else {
					err := thread.Children.SpawnWithOpts(sid, directive, spawnTools, SpawnOpts{
						ProviderName: providerName,
						ParentID:     thread.ID,
						Depth:        thread.Depth + 1,
						MCPNames:     mcpNames,
						BuiltinTools: builtinTools,
						Paused:       paused,
					})
					if err != nil {
						emitResult(call, fmt.Sprintf("error: %v", err))
					} else {
						t.config.SaveThread(PersistentThread{
							ID: sid, ParentID: thread.ID, Depth: thread.Depth + 1,
							Directive: directive, Tools: spawnTools, MCPNames: mcpNames,
						})
						if paused {
							emitResult(call, fmt.Sprintf("thread %s spawned (depth %d, paused — send a message to wake)", sid, thread.Depth+1))
						} else {
							emitResult(call, fmt.Sprintf("thread %s spawned (depth %d)", sid, thread.Depth+1))
						}
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "kill":
				sid := call.Args["id"]
				if sid == "" {
					emitResult(call, "error: kill requires id")
				} else if thread.Children == nil {
					emitResult(call, "error: cannot kill (not a leader thread)")
				} else {
					thread.Children.Kill(sid)
					t.config.RemoveThread(sid)
					emitResult(call, fmt.Sprintf("thread %s killed", sid))
				}
				toolNames = append(toolNames, call.Raw)
			case "update":
				sid := call.Args["id"]
				newID := call.Args["new_id"]
				name := call.Args["name"]
				directive := call.Args["directive"]
				toolsStr := call.Args["tools"]
				if sid == "" {
					emitResult(call, "error: update requires id")
				} else if thread.Children == nil {
					emitResult(call, "error: cannot update (not a leader thread)")
				} else if newID == "" && name == "" && directive == "" && toolsStr == "" {
					emitResult(call, "error: update requires at least one of new_id, name, directive, tools")
				} else {
					var updateTools []string
					if toolsStr != "" {
						updateTools = strings.Split(toolsStr, ",")
					}
					applyErr := error(nil)
					if name != "" || directive != "" || len(updateTools) > 0 {
						applyErr = thread.Children.Update(sid, name, directive, updateTools)
						if applyErr == nil && directive != "" {
							thread.Children.Send(sid, fmt.Sprintf("[directive updated] %s", directive))
						}
					}
					if applyErr != nil {
						emitResult(call, fmt.Sprintf("error: %v", applyErr))
					} else if newID != "" {
						if err := thread.Children.Rename(sid, newID); err != nil {
							emitResult(call, fmt.Sprintf("error: %v", err))
						} else {
							emitResult(call, fmt.Sprintf("thread renamed %s → %s", sid, newID))
						}
					} else {
						emitResult(call, fmt.Sprintf("thread %s updated", sid))
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
					emitResult(call, "set "+strings.Join(parts, " "))
				} else {
					emitResult(call, "ok")
				}
			case "evolve":
				d := call.Args["directive"]
				if d == "" {
					emitResult(call, "error: evolve requires directive")
				} else {
					thread.Directive = d
					if t.rebuildPrompt != nil {
						t.messages[0] = Message{Role: "system", Content: t.rebuildPrompt("")}
					}
					tm.parent.config.SaveThread(PersistentThread{
						ID: thread.ID, Name: thread.Name, ParentID: thread.ParentID, Depth: thread.Depth,
						Directive: d, Tools: toolSetToSlice(thread.Tools), MCPNames: thread.MCPNames,
					})
					t.logAPI(APIEvent{Type: "evolved", ThreadID: thread.ID, Message: d})
					emitResult(call, "directive updated")
				}
			case "remember":
				// Memory v2: sub-threads can't write either. The unconscious
				// is the only writer and it observes sub-thread activity
				// the same way it observes main. Surface a clear error so
				// the LLM stops calling this if a legacy directive still
				// instructs it to.
				emitResult(call, "error: remember is not available — memory writes are owned by the unconscious thread")
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
			Name:      t.Name,
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
			MCPNames:     t.MCPNames,
			SubThreads:   subCount,
		})
	}
	// Sort by ID for deterministic order
	sort.Slice(infos, func(i, j int) bool {
		return infos[i].ID < infos[j].ID
	})
	return infos
}

func (tm *ThreadManager) Count() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.threads)
}

// StartAll starts Run() on all threads (and their children) that were spawned with DeferRun.
// Used after batch-respawning persisted threads so parents see their children before thinking.
func (tm *ThreadManager) StartAll() {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for _, thread := range tm.threads {
		go thread.Thinker.Run()
		if thread.Children != nil {
			thread.Children.StartAll()
		}
	}
}

// Update changes a thread's directive and/or tools. Rebuilds the system prompt immediately.
func (tm *ThreadManager) Update(id, name, directive string, tools []string) error {
	tm.mu.RLock()
	thread, exists := tm.threads[id]
	tm.mu.RUnlock()
	if !exists {
		return fmt.Errorf("thread %q not found", id)
	}

	nameChanged := false
	if name != "" && name != thread.Name {
		thread.Name = name
		nameChanged = true
	}
	if directive != "" {
		thread.Directive = directive
	}
	if len(tools) > 0 {
		toolSet := make(map[string]bool)
		for _, t := range tools {
			toolSet[strings.TrimSpace(t)] = true
		}
		// Always include builtins. (remember is intentionally absent — the
		// tool is unregistered for now while memory is being redesigned.)
		for _, b := range []string{"send", "done", "pace", "evolve"} {
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
		ID: id, Name: thread.Name, ParentID: thread.ParentID, Depth: thread.Depth,
		Directive: thread.Directive, Tools: toolSetToSlice(thread.Tools), MCPNames: thread.MCPNames,
	})

	// Telemetry: name-only changes fire thread.renamed (id stays the
	// same on both sides) so the dashboard can update its label without
	// thinking the thread got recreated.
	if nameChanged && tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.renamed", id, ThreadRenamedData{
			OldID: id, NewID: id, Name: thread.Name, ParentID: thread.ParentID,
		})
	}

	return nil
}

// Rename changes a thread's immutable id. Touches every reference:
//   - the threads map key
//   - children's ParentID
//   - the persistent record (delete old, save new)
//   - the on-disk session file
//   - emits thread.renamed telemetry so the dashboard can swap its
//     record over to the new identity
//
// Returns an error if the new id is empty, equals the old, collides
// with an existing sibling, or any of the persistence steps fail.
func (tm *ThreadManager) Rename(oldID, newID string) error {
	if newID == "" {
		return fmt.Errorf("new id required")
	}
	if newID == oldID {
		return nil
	}
	tm.mu.Lock()
	thread, exists := tm.threads[oldID]
	if !exists {
		tm.mu.Unlock()
		return fmt.Errorf("thread %q not found", oldID)
	}
	if _, taken := tm.threads[newID]; taken {
		tm.mu.Unlock()
		return fmt.Errorf("thread id %q already in use", newID)
	}
	delete(tm.threads, oldID)
	thread.ID = newID
	tm.threads[newID] = thread
	tm.mu.Unlock()

	// Rename the thinker's session file so future history loads find
	// it under the new id. Best-effort: a brand-new thread with no
	// entries yet has no file on disk, which is treated as success.
	if thread.Thinker != nil && thread.Thinker.session != nil {
		if err := thread.Thinker.session.Rename(newID); err != nil {
			// Roll back the in-memory rename so we don't end up with a
			// thread whose id and session disagree.
			tm.mu.Lock()
			thread.ID = oldID
			delete(tm.threads, newID)
			tm.threads[oldID] = thread
			tm.mu.Unlock()
			return fmt.Errorf("rename session: %w", err)
		}
	}

	// Cascade ParentID on direct children of the renamed thread.
	if thread.Children != nil {
		thread.Children.mu.Lock()
		for _, child := range thread.Children.threads {
			child.ParentID = newID
		}
		thread.Children.mu.Unlock()
	}

	// Persist: drop the old record + save under the new id.
	tm.parent.config.RemoveThread(oldID)
	tm.parent.config.SaveThread(PersistentThread{
		ID: newID, Name: thread.Name, ParentID: thread.ParentID, Depth: thread.Depth,
		Directive: thread.Directive, Tools: toolSetToSlice(thread.Tools), MCPNames: thread.MCPNames,
	})

	if tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.renamed", newID, ThreadRenamedData{
			OldID: oldID, NewID: newID, Name: thread.Name, ParentID: thread.ParentID,
		})
	}
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

	// Only delete session history if thread called done (permanent termination).
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

// threadExistsInTree checks if a thread ID exists anywhere below root
// (caller has already checked root's direct children).
//
// Locking contract: the caller of the TOP-level call holds root.mu
// (spawnInternal does). We read `root.threads` without re-locking
// (re-locking the same RWMutex that the caller's write-lock holds would
// deadlock — Go's RWMutex is not re-entrant). For each child we take
// that child's RLock strictly to inspect its map — never held across
// the recursive call, so lock order stays top-down and there's no risk
// of a concurrent cleanup that holds a child lock deadlocking us.
func threadExistsInTree(root *ThreadManager, id string) bool {
	// Snapshot child-TM pointers under whatever lock the caller already
	// holds. Do NOT re-acquire root.mu here (spawnInternal owns the
	// write lock and a nested RLock on the same mutex would deadlock).
	children := make([]*ThreadManager, 0, len(root.threads))
	for _, t := range root.threads {
		if t.Children != nil {
			children = append(children, t.Children)
		}
	}

	for _, childTM := range children {
		// Short critical section: check presence, then release.
		childTM.mu.RLock()
		_, found := childTM.threads[id]
		// Snapshot this child's grandchildren under the same lock so
		// the recursion below can read them without re-locking.
		grandchildren := make([]*ThreadManager, 0, len(childTM.threads))
		for _, t := range childTM.threads {
			if t.Children != nil {
				grandchildren = append(grandchildren, t.Children)
			}
		}
		childTM.mu.RUnlock()
		if found {
			return true
		}
		// Recurse outside the child's lock — each recursion acquires its
		// own short RLock for its level.
		for _, grand := range grandchildren {
			if threadExistsInTreeLocked(grand, id) {
				return true
			}
		}
	}
	return false
}

// threadExistsInTreeLocked is the helper used when recursing from a
// caller that does NOT hold any lock. It acquires root.mu.RLock to
// read root.threads, then delegates to the shared walking logic.
func threadExistsInTreeLocked(root *ThreadManager, id string) bool {
	root.mu.RLock()
	if _, found := root.threads[id]; found {
		root.mu.RUnlock()
		return true
	}
	grandchildren := make([]*ThreadManager, 0, len(root.threads))
	for _, t := range root.threads {
		if t.Children != nil {
			grandchildren = append(grandchildren, t.Children)
		}
	}
	root.mu.RUnlock()
	for _, grand := range grandchildren {
		if threadExistsInTreeLocked(grand, id) {
			return true
		}
	}
	return false
}

func toolSetToSlice(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
