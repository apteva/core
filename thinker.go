package main

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

const maxHistory = 20

type ModelTier int

const (
	ModelLarge ModelTier = iota
	ModelSmall
)

var modelNames = map[string]ModelTier{
	"large": ModelLarge,
	"small": ModelSmall,
}

func (m ModelTier) String() string {
	switch m {
	case ModelLarge:
		return "large"
	case ModelSmall:
		return "small"
	default:
		return "large"
	}
}

// modelID returns the model ID from the provider for a given tier.
func (t *Thinker) modelID() string {
	if t.provider != nil {
		return t.provider.Models()[t.model]
	}
	return "unknown"
}

// baseSystemPrompt contains the fixed rules/tools. The editable directive is prepended at runtime.
const baseSystemPrompt = `You are the main coordinating thread of a continuous thinking engine (Cogito). You observe all events, manage threads, and coordinate work. You do NOT talk to users directly — you spawn threads for that.

THINKING — every thought must contain meaningful text:
- Always explain what you observe, what you're doing, and why — even briefly.
- NEVER output only tool calls. Always include at least one sentence of reasoning.
- When idle: briefly state your current status and what you're waiting for.
- When busy: explain what you're working on and next steps.
- Keep each thought concise — 1-2 short paragraphs max.

EVENT FORMAT:
- [user:name] message — a user sent a message. Spawn or route to a thread for them.
- [from:id] message — a thread sent you a message via send.
- [thread:id done] message — a thread finished and terminated.
- [console] message — a direct system command. Do NOT reply — just incorporate into your thinking.

BEHAVIOR:
- When you see [user:X], spawn a thread with id="X" so future messages auto-route. The triggering message is auto-forwarded.
- If the thread already exists, events are auto-routed — you won't see them.
- Spawn threads for any task — conversations, research, monitoring, timed actions.
- Additional tools may appear in [available tools] blocks based on context. If you need a tool you don't see, describe what you need.

SPAWNING THREADS — critical rules:
- The "tools" parameter lists which tools the thread can use. ALWAYS include ALL tools the thread needs.
- Tool names MUST match EXACTLY as shown in [available tools]. They include a prefix (e.g. "schedule_get_schedule", NOT "get_schedule"). Copy the exact name.
- Example: tools="pushover_send_notification,schedule_get_schedule" — use the full prefixed name.
- The "directive" parameter must be PLAIN NATURAL LANGUAGE describing the thread's goal and behavior.
  NEVER put tool call syntax like [[ ]] in the directive. NEVER put tool names in the directive.
  The thread already receives its own tool documentation — it knows what tools it has.
  BAD:  directive="Call [[helpdesk_list_tickets]] to check for tickets"
  GOOD: directive="Check for new support tickets periodically. When you find tickets, report them to main."

PACING — critical:
- Events ALWAYS wake you instantly, no matter how long your sleep is. There is ZERO cost to sleeping long.
- Be aggressive about saving power: if you have no pending work, go straight to [[pace sleep="1h" model="small"]]. Do NOT gradually increase — jump to long sleep immediately.
- Only use short sleep (2-10s) when you are actively waiting for a tool result in the NEXT iteration.
- Your pace persists until you change it. Do NOT call [[pace]] every thought — only when transitioning between active work and idle.
- When an event wakes you, you automatically switch to large model and fast pace for that iteration. You do NOT need to manually set pace when events arrive.

CRITICAL — never hallucinate events:
- You ONLY receive events in [Events:] blocks. If there is no [Events:] block, NOTHING happened.
- NEVER invent, imagine, or assume events that are not explicitly shown to you.
- NEVER pretend a user sent a message. NEVER fabricate [user:...] events.
- If no events arrived, your ONLY job is to set your pace and wait. Do not take any action.
- Violating this rule causes real damage — spawning threads or sending notifications based on imagined events wastes resources and confuses users.

You have persistent memory across restarts. Relevant memories appear as [memories] blocks.`

func buildSystemPrompt(directive string, registry *ToolRegistry, extraToolDocs string, servers []MCPConn) string {
	coreDocs := ""
	if registry != nil {
		coreDocs = "\n" + registry.CoreDocs(true)
	}
	prompt := baseSystemPrompt + coreDocs
	if extraToolDocs != "" {
		prompt += "\n" + extraToolDocs
	}

	// Inject connected servers
	if len(servers) > 0 {
		prompt += "\n\n[CONNECTED SERVERS]\n"
		for _, srv := range servers {
			prompt += "- " + srv.GetName() + "\n"
		}
	}

	prompt += "\n\n[DIRECTIVE — EXECUTE ON STARTUP]\nThe following is your mission. On your FIRST thought, take any actions needed to fulfill it (spawn threads, etc). This overrides default idle behavior.\n\n" + directive
	return prompt
}

type TokenUsage struct {
	PromptTokens     int
	CachedTokens     int
	CompletionTokens int
}

type ThinkRate int

const (
	RateReactive ThinkRate = iota // 500ms — event just arrived
	RateFast                      // 2s — actively working
	RateNormal                    // 10s — thinking, no urgency
	RateSlow                      // 30s — not much to do
	RateSleep                     // 120s — deep idle
)

// rateAliases maps named rates to durations (backwards compat + convenience)
var rateAliases = map[string]time.Duration{
	"reactive": 500 * time.Millisecond,
	"fast":     2 * time.Second,
	"normal":   10 * time.Second,
	"slow":     30 * time.Second,
	"sleep":    2 * time.Minute,
}

// rateNames kept for ThinkRate enum mapping (used by eventbus, TUI, etc.)
var rateNames = map[string]ThinkRate{
	"reactive": RateReactive,
	"fast":     RateFast,
	"normal":   RateNormal,
	"slow":     RateSlow,
	"sleep":    RateSleep,
}

const (
	minSleep = 500 * time.Millisecond
	maxSleep = 24 * time.Hour
)

// parseSleepDuration parses a sleep duration from agent input.
// Accepts Go duration strings ("30s", "5m", "2h") or named aliases ("slow", "sleep").
func parseSleepDuration(s string) (time.Duration, bool) {
	// Check named aliases first
	if d, ok := rateAliases[s]; ok {
		return d, true
	}
	// Try Go duration string
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, false
	}
	// Clamp to bounds
	if d < minSleep {
		d = minSleep
	}
	if d > maxSleep {
		d = maxSleep
	}
	return d, true
}

// formatSleep returns a human-readable sleep duration string.
func formatSleep(d time.Duration) string {
	if d >= time.Hour {
		return fmt.Sprintf("%.1fh", d.Hours())
	}
	if d >= time.Minute {
		return fmt.Sprintf("%.1fm", d.Minutes())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func (r ThinkRate) String() string {
	switch r {
	case RateReactive:
		return "reactive"
	case RateFast:
		return "fast"
	case RateNormal:
		return "normal"
	case RateSlow:
		return "slow"
	case RateSleep:
		return "sleep"
	default:
		return "normal"
	}
}

func (r ThinkRate) Delay() time.Duration {
	switch r {
	case RateReactive:
		return 500 * time.Millisecond
	case RateFast:
		return 2 * time.Second
	case RateNormal:
		return 10 * time.Second
	case RateSlow:
		return 30 * time.Second
	case RateSleep:
		return 120 * time.Second
	default:
		return 10 * time.Second
	}
}

type APIEvent struct {
	Time      time.Time `json:"time"`
	Type      string    `json:"type"`                 // "thought", "chunk", "reply", "thread_started", "thread_done", "error"
	ThreadID  string    `json:"thread_id"`
	Message   string    `json:"message,omitempty"`
	Iteration int       `json:"iteration,omitempty"`
	Duration  string    `json:"duration,omitempty"`
}

// ToolHandler processes parsed tool calls from a thought. Returns replies and tool names logged.
// consumed contains the events that were consumed this iteration (for context).
type ToolHandler func(t *Thinker, calls []toolCall, consumed []string) (replies []string, toolNames []string)

// EventFilter preprocesses drained bus events. Can route/drop events.
type EventFilter func(events []string) []string

type Thinker struct {
	apiKey    string
	provider  LLMProvider
	messages  []Message
	bus       *EventBus
	sub       *Subscription
	pause     chan bool
	quit      chan struct{}
	iteration int
	paused    bool
	rate       ThinkRate
	agentRate  ThinkRate
	agentSleep time.Duration // freeform sleep duration (takes priority over agentRate when > 0)
	model      ModelTier
	agentModel ModelTier
	memory     *MemoryStore
	threads    *ThreadManager
	config     *Config
	registry   *ToolRegistry

	// Hooks — set these to customize behavior. nil = defaults.
	handleTools    ToolHandler
	filterEvents   EventFilter
	rebuildPrompt  func(toolDocs string) string // rebuild system prompt with current tool docs
	onStop         func()
	toolAllowlist  map[string]bool // nil = all tools allowed (main thread)

	// API event log — shared across all threads, owned by main thinker
	apiLog    *[]APIEvent
	apiMu     *sync.RWMutex
	apiNotify chan struct{}
	threadID  string // "main" for main thinker, thread ID for sub-threads

	// Telemetry — shared across all threads, owned by main thinker
	telemetry *Telemetry

	// Live MCP connections — servers connected at runtime
	mcpServers []MCPConn
}

func NewThinker(apiKey string, provider LLMProvider) *Thinker {
	cfg := NewConfig()
	bus := NewEventBus()
	t := &Thinker{
		apiKey:   apiKey,
		provider: provider,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(cfg.GetDirective(), nil, "", nil)},
		},
		config:    cfg,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool, 1),
		quit:      make(chan struct{}),
		rate:       RateSlow,
		agentRate:  RateSlow,
		agentSleep: 30 * time.Second,
		memory:    NewMemoryStore(apiKey),
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
		telemetry: NewTelemetry(),
	}
	t.threads = NewThreadManager(t)
	t.registry = NewToolRegistry(apiKey)

	// Rebuild system prompt now that registry exists (with core tool docs)
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(cfg.GetDirective(), t.registry, "", nil)}

	// Embed tool descriptions in background (non-blocking)
	go t.registry.EmbedAll(t.memory)

	// Main thread hooks
	t.filterEvents = func(events []string) []string {
		var kept []string
		for _, ev := range events {
			if !t.threads.Route(ev) {
				kept = append(kept, ev)
			}
		}
		return kept
	}
	t.handleTools = mainToolHandler(t)
	t.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(t.config.GetDirective(), t.registry, toolDocs, t.mcpServers)
	}

	// Connect MCP servers and register their tools
	if len(cfg.MCPServers) > 0 {
		t.mcpServers = connectAndRegisterMCP(cfg.MCPServers, t.registry, t.memory)
		// Rebuild prompt now that servers are connected
		t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(cfg.GetDirective(), t.registry, "", t.mcpServers)}
	}

	// Respawn persistent threads from config
	for _, pt := range cfg.GetThreads() {
		t.threads.Spawn(pt.ID, pt.Directive, pt.Tools)
	}

	return t
}

// mainToolHandler returns the tool handler for the main coordinating thread.
func mainToolHandler(t *Thinker) ToolHandler {
	return func(_ *Thinker, calls []toolCall, consumed []string) ([]string, []string) {
		var replies []string
		var toolNames []string
		for _, call := range calls {
			switch call.Name {
			case "spawn":
				id := call.Args["id"]
				directive := call.Args["directive"]
				if directive == "" {
					directive = call.Args["prompt"] // backwards compat
				}
				toolsStr := call.Args["tools"]
				var tools []string
				if toolsStr != "" {
					tools = strings.Split(toolsStr, ",")
				}
				if id != "" && directive != "" {
					if err := t.threads.Spawn(id, directive, tools, consumed...); err != nil {
						t.Inject(fmt.Sprintf("[error] spawn %q: %v", id, err))
					} else {
						t.config.SaveThread(PersistentThread{
							ID: id, Directive: directive, Tools: tools,
						})
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "kill":
				if id := call.Args["id"]; id != "" {
					t.threads.Kill(id)
					t.config.RemoveThread(id)
				}
				toolNames = append(toolNames, call.Raw)
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				if id != "" && msg != "" {
					if !t.threads.Send(id, msg) {
						t.Inject(fmt.Sprintf("[error] thread %q not found", id))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "evolve":
				if d := call.Args["directive"]; d != "" {
					t.config.SetDirective(d)
					t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(d, t.registry, "", t.mcpServers)}
					t.logAPI(APIEvent{Type: "evolved", ThreadID: "main", Message: d})
				}
			case "remember":
				if text := call.Args["text"]; text != "" && t.memory != nil {
					go t.memory.Store(text)
				}
			case "pace":
				// Freeform sleep duration takes priority
				if s := call.Args["sleep"]; s != "" {
					if d, ok := parseSleepDuration(s); ok {
						t.agentSleep = d
						t.agentRate = RateSleep // fallback enum for display
					}
				} else if r, ok := rateNames[call.Args["rate"]]; ok {
					// Named rate alias — also set agentSleep from alias
					t.agentRate = r
					if d, ok2 := rateAliases[call.Args["rate"]]; ok2 {
						t.agentSleep = d
					}
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
				}
			case "connect":
				name := call.Args["name"]
				command := call.Args["command"]
				argsStr := call.Args["args"]
				url := call.Args["url"]
				transport := call.Args["transport"]
				if name != "" && (command != "" || url != "") {
					var mcpArgs []string
					if argsStr != "" {
						mcpArgs = strings.Split(argsStr, ",")
					}
					cfg := MCPServerConfig{Name: name, Command: command, Args: mcpArgs, URL: url, Transport: transport}
					srv, err := connectAnyMCP(cfg)
					if err != nil {
						t.Inject(fmt.Sprintf("[connect] error: %v", err))
					} else {
						tools, err := srv.ListTools()
						if err != nil {
							t.Inject(fmt.Sprintf("[connect] tool discovery error: %v", err))
							srv.Close()
						} else {
							t.mcpServers = append(t.mcpServers, srv)
							for _, tool := range tools {
								fullName := name + "_" + tool.Name
								syntax := buildMCPSyntax(fullName, tool.InputSchema)
								t.registry.Register(&ToolDef{
									Name:        fullName,
									Description: fmt.Sprintf("[%s] %s", name, tool.Description),
									Syntax:      syntax,
									Rules:       fmt.Sprintf("Provided by MCP server '%s'.", name),
									Handler:     mcpProxyHandler(srv, tool.Name),
								})
							}
							if t.memory != nil {
								go func(srvName string, srvTools []mcpToolDef) {
									for _, tl := range srvTools {
										fullName := srvName + "_" + tl.Name
										emb, err := t.memory.embed(fullName + ": " + tl.Description)
										if err == nil {
											td := t.registry.Get(fullName)
											if td != nil {
												td.Embedding = emb
											}
										}
									}
								}(name, tools)
							}
							t.Inject(fmt.Sprintf("[connect] connected to %s: %d tools registered", name, len(tools)))
							// Persist to config for restart survival
							t.config.SaveMCPServer(cfg)
						}
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "disconnect":
				name := call.Args["name"]
				if name != "" {
					found := false
					for i, srv := range t.mcpServers {
						if srv.GetName() == name {
							srv.Close()
							t.mcpServers = append(t.mcpServers[:i], t.mcpServers[i+1:]...)
							t.Inject(fmt.Sprintf("[disconnect] disconnected from %s", name))
							t.config.RemoveMCPServer(name)
							found = true
							break
						}
					}
					if !found {
						t.Inject(fmt.Sprintf("[disconnect] server %q not found", name))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "list_connected":
				var names []string
				for _, srv := range t.mcpServers {
					names = append(names, srv.GetName())
				}
				t.Inject(fmt.Sprintf("[connected] %d servers: %s", len(names), strings.Join(names, ", ")))
			default:
				// Dispatch to registry (MCP tools, etc)
				executeTool(t, call)
				toolNames = append(toolNames, call.Raw)
			}
		}
		return replies, toolNames
	}
}

func (t *Thinker) Run() {
	defer func() {
		if t.onStop != nil {
			t.onStop()
		}
	}()

	for {
		// Check pause/quit
		select {
		case <-t.quit:
			return
		case p := <-t.pause:
			t.paused = p
			if t.paused {
				select {
				case p = <-t.pause:
					t.paused = p
				case <-t.quit:
					return
				}
			}
		default:
		}

		t.iteration++
		logMsg("RUN", fmt.Sprintf("[%s] iteration #%d start, rate=%s", t.threadID, t.iteration, t.rate.String()))

		// Drain events from bus, optionally filter/route
		consumed := t.drainEvents()
		if len(consumed) > 0 {
			logMsg("RUN", fmt.Sprintf("[%s] drained %d events", t.threadID, len(consumed)))
			for i, ev := range consumed {
				preview := ev
				if len(preview) > 100 {
					preview = preview[:100] + "..."
				}
				logMsg("RUN", fmt.Sprintf("[%s]   event[%d]: %s", t.threadID, i, preview))
			}
		}
		if t.filterEvents != nil {
			consumed = t.filterEvents(consumed)
		}

		// Only go reactive for non-tool events (user messages, console, thread sends)
		hasExternalEvent := false
		for _, ev := range consumed {
			if !strings.HasPrefix(ev, "[tool:") {
				hasExternalEvent = true
				break
			}
		}

		hadEvents := len(consumed) > 0
		if hasExternalEvent {
			t.rate = RateReactive
			t.model = ModelLarge
		} else if hadEvents {
			// Tool results — wake but less aggressive than external events
			t.rate = RateFast
		}

		now := time.Now().Format("2006-01-02 15:04:05")
		if hadEvents {
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
			for _, ev := range consumed {
				sb.WriteString("• " + ev + "\n")
			}
			t.messages = append(t.messages, Message{Role: "user", Content: sb.String()})
		} else {
			t.messages = append(t.messages, Message{Role: "user", Content: fmt.Sprintf("[%s] (no events)", now)})
		}

		// Memory recall
		if t.memory != nil && t.memory.Count() > 0 {
			var memQuery string
			if hadEvents {
				memQuery = strings.Join(consumed, " ")
			} else {
				for i := len(t.messages) - 1; i >= 0; i-- {
					if t.messages[i].Role == "assistant" {
						memQuery = t.messages[i].Content
						break
					}
				}
			}
			if memQuery != "" {
				recalled := t.memory.Retrieve(memQuery, recallTopN)
				if ctx := t.memory.BuildContext(recalled); ctx != "" {
					t.messages = append(t.messages, Message{Role: "system", Content: ctx})
				}
			}
		}

		// Tool discovery via RAG — update system prompt with discovered tools
		if t.registry != nil && t.rebuildPrompt != nil {
			var toolQuery string
			if hadEvents {
				toolQuery = strings.Join(consumed, " ")
			} else {
				for i := len(t.messages) - 1; i >= 0; i-- {
					if t.messages[i].Role == "assistant" {
						toolQuery = t.messages[i].Content
						break
					}
				}
			}
			tools := t.registry.Retrieve(toolQuery, 5, t.allowedTools(), t.memory)
			toolDocs := t.registry.BuildDocs(tools)
			t.messages[0] = Message{Role: "system", Content: t.rebuildPrompt(toolDocs)}
		}

		start := time.Now()
		reply, usage, err := t.think()
		duration := time.Since(start)

		if err != nil {
			t.bus.Publish(Event{Type: EventThinkError, From: t.threadID, Error: err, Iteration: t.iteration})
			if t.telemetry != nil {
				t.telemetry.Emit("llm.error", t.threadID, LLMErrorData{
					Model: t.modelID(), Error: err.Error(), Iteration: t.iteration,
				})
			}
			select {
			case <-time.After(5 * time.Second):
			case <-t.quit:
				return
			}
			continue
		}

		t.messages = append(t.messages, Message{Role: "assistant", Content: reply})

		// Dispatch tool calls via handler
		calls := parseToolCalls(reply)
		var replies []string
		var toolNames []string
		if t.handleTools != nil {
			replies, toolNames = t.handleTools(t, calls, consumed)
		}

		// Sliding window
		if len(t.messages) > maxHistory+1 {
			t.messages = append(t.messages[:1], t.messages[len(t.messages)-maxHistory:]...)
		}

		// After processing, fall back to agent's chosen rate/sleep
		// (external events already set reactive above for this iteration)
		t.rate = t.agentRate
		t.model = t.agentModel

		// Compute actual sleep duration: agentSleep takes priority, else rate enum
		sleepDur := t.agentSleep
		if sleepDur <= 0 {
			sleepDur = t.rate.Delay()
		}

		// Thread count (0 if no thread manager)
		threadCount := 0
		if t.threads != nil {
			threadCount = t.threads.Count()
		}

		// Context size
		ctxChars := 0
		for _, msg := range t.messages {
			ctxChars += len(msg.Content)
		}

		t.bus.Publish(Event{
			Type: EventThinkDone, From: t.threadID,
			Iteration: t.iteration, Duration: duration,
			ConsumedEvents: consumed, Usage: usage,
			ToolCalls: toolNames, Replies: replies,
			Rate: t.rate, Model: t.model,
			MemoryCount: t.memory.Count(), ThreadCount: threadCount,
			ContextMsgs: len(t.messages), ContextChars: ctxChars,
		})

		// Log to API — include full reply so tool calls are visible too
		thoughtLog := strings.TrimSpace(reply)
		if len(thoughtLog) > 1000 {
			thoughtLog = thoughtLog[:1000] + "..."
		}
		t.logAPI(APIEvent{Type: "thought", Iteration: t.iteration, Message: thoughtLog, Duration: duration.Round(time.Millisecond).String()})
		for _, r := range replies {
			t.logAPI(APIEvent{Type: "reply", Message: r})
		}

		// Telemetry: llm.done with full data
		if t.telemetry != nil {
			t.telemetry.Emit("llm.done", t.threadID, LLMDoneData{
				Model:        t.modelID(),
				TokensIn:     usage.PromptTokens,
				TokensCached: usage.CachedTokens,
				TokensOut:    usage.CompletionTokens,
				DurationMs:   duration.Milliseconds(),
				CostUSD:      calculateCostForProvider(t.provider, usage),
				Iteration:    t.iteration,
				Rate:         formatSleep(sleepDur),
				ContextMsgs:  len(t.messages),
				ContextChars: ctxChars,
				MemoryCount:  t.memory.Count(),
				ThreadCount:  threadCount,
				Message:      thoughtLog,
			})
		}

		// Interruptible sleep — wakes on new event, quit, or pause
		logMsg("RUN", fmt.Sprintf("[%s] sleeping %s", t.threadID, formatSleep(sleepDur)))
		select {
		case <-time.After(sleepDur):
			logMsg("RUN", fmt.Sprintf("[%s] woke: timer expired", t.threadID))
		case <-t.sub.Wake:
			logMsg("RUN", fmt.Sprintf("[%s] woke: event received", t.threadID))
		case p := <-t.pause:
			t.paused = p
			logMsg("RUN", fmt.Sprintf("[%s] paused=%v during sleep", t.threadID, t.paused))
			if t.paused {
				// Block until unpaused or quit
				select {
				case p = <-t.pause:
					t.paused = p
					logMsg("RUN", fmt.Sprintf("[%s] resumed", t.threadID))
				case <-t.quit:
					return
				}
			}
		case <-t.quit:
			logMsg("RUN", fmt.Sprintf("[%s] woke: quit signal", t.threadID))
			return
		}
	}
}

func (t *Thinker) think() (string, TokenUsage, error) {
	onChunk := func(chunk string) {
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: chunk, Iteration: t.iteration})
		if t.telemetry != nil && chunk != "" {
			t.telemetry.EmitLive("llm.chunk", t.threadID, LLMChunkData{
				Text: chunk, Iteration: t.iteration,
			})
		}
	}
	return t.provider.Chat(t.messages, t.modelID(), onChunk)
}

// drainEvents reads all pending events and wake signals from this thinker's bus subscription.
func (t *Thinker) drainEvents() []string {
	var items []string
	for {
		select {
		case ev := <-t.sub.C:
			if ev.Type == EventInbox {
				items = append(items, ev.Text)
			}
		case <-t.sub.Wake:
			// Consume wake signals alongside their events
			continue
		default:
			return items
		}
	}
}

func (t *Thinker) logAPI(ev APIEvent) {
	if t.apiNotify == nil || t.apiLog == nil {
		return
	}
	ev.Time = time.Now()
	if ev.ThreadID == "" {
		ev.ThreadID = t.threadID
	}
	t.apiMu.Lock()
	*t.apiLog = append(*t.apiLog, ev)
	if len(*t.apiLog) > 1000 {
		*t.apiLog = (*t.apiLog)[len(*t.apiLog)-500:]
	}
	t.apiMu.Unlock()
	select {
	case t.apiNotify <- struct{}{}:
	default:
	}
}

func (t *Thinker) APIEvents(since int) ([]APIEvent, int) {
	t.apiMu.RLock()
	defer t.apiMu.RUnlock()
	if since >= len(*t.apiLog) {
		return nil, len(*t.apiLog)
	}
	events := make([]APIEvent, len(*t.apiLog)-since)
	copy(events, (*t.apiLog)[since:])
	return events, len(*t.apiLog)
}

// allowedTools returns the tool allowlist for this thinker. nil = all tools allowed.
func (t *Thinker) allowedTools() map[string]bool {
	return t.toolAllowlist
}

func (t *Thinker) ReloadDirective() {
	directive := t.config.GetDirective()
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, t.registry, "", t.mcpServers)}
	t.InjectConsole("Directive updated to: " + directive + "\n\nAdjust the system accordingly — spawn, kill, or reconfigure threads as needed.")
}

// Inject sends a message event to this thinker's bus subscription.
func (t *Thinker) Inject(msg string) {
	logMsg("INJECT", fmt.Sprintf("to=%s msg=%s", t.threadID, msg))
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: msg})
}

// InjectConsole sends a console event to this thinker.
func (t *Thinker) InjectConsole(msg string) {
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: "[console] " + msg})
}

// InjectUserMessage sends a user message event to this thinker.
func (t *Thinker) InjectUserMessage(userID, msg string) {
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: fmt.Sprintf("[user:%s] %s", userID, msg)})
}

func (t *Thinker) TogglePause() {
	newState := !t.paused
	// Non-blocking send — channel is buffered(1), drain any stale value first
	select {
	case <-t.pause:
	default:
	}
	t.pause <- newState
	t.paused = newState
}

func (t *Thinker) Stop() {
	select {
	case <-t.quit:
	default:
		close(t.quit)
	}
}
