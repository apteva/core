package main

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/apteva/core/pkg/computer"
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
const baseSystemPrompt = `You are the main coordinating thread of a continuous thinking engine. You observe all events, manage threads, and coordinate work. You do NOT talk to users directly — you spawn threads for that.

THINKING — every thought must contain meaningful text:
- Always explain what you observe, what you're doing, and why — even briefly.
- NEVER output only tool calls. Always include at least one sentence of reasoning.
- When idle: briefly state your current status and what you're waiting for.
- When busy: explain what you're working on and next steps.
- Keep each thought concise — 1-2 short paragraphs max.

EVENT FORMAT:
- [console] message — an external event or command. Incorporate into your thinking and take action as needed.
- [from:id] message — a thread sent you a message via send.
- [thread:id done] message — a thread finished and terminated.

BEHAVIOR:
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
- NEVER fabricate events that did not appear in the [Events:] block.
- If no events arrived, your ONLY job is to set your pace and wait. Do not take any action.
- Violating this rule causes real damage — spawning threads or sending notifications based on imagined events wastes resources and confuses users.

You have persistent memory across restarts. Relevant memories appear as [memories] blocks.`

func buildSystemPrompt(directive string, mode RunMode, registry *ToolRegistry, extraToolDocs string, servers []MCPConn, activeThreads []ThreadInfo) string {
	coreDocs := ""
	if registry != nil {
		coreDocs = "\n" + registry.CoreDocs(true)
	}
	prompt := baseSystemPrompt + coreDocs
	if extraToolDocs != "" {
		prompt += "\n" + extraToolDocs
	}

	// Inject MCP tool summary — main thread sees what's available but can't call them directly
	if registry != nil {
		if summary := registry.MCPToolSummary(); summary != "" {
			prompt += summary
		}
	}

	// Inject active thread state so main always knows what's running
	if len(activeThreads) > 0 {
		prompt += "\n\n[ACTIVE THREADS]\n"
		for _, t := range activeThreads {
			age := time.Since(t.Started).Truncate(time.Second)
			prompt += fmt.Sprintf("- %s (running %s, iter #%d, pace %s, model %s)\n  directive: %s\n  tools: %s\n",
				t.ID, age, t.Iteration, t.Rate.String(), t.Model.String(), truncateStr(t.Directive, 150), strings.Join(t.Tools, ", "))
		}
	}

	// Safety guidance based on mode
	prompt += "\n\n[SAFETY MODE: " + string(mode) + "]\n"
	switch mode {
	case ModeCautious:
		prompt += `Before executing any tool that modifies state (exec, write, deploy, restart, delete), first tell the user what you plan to do and why via channels_respond, then wait for their confirmation in the next message. Read-only tools (web, query, list) can be used freely. If unsure whether an action is safe, ask. Learn from user feedback — use [[remember]] to store their preferences.`
	case ModeLearn:
		prompt += `You are learning the user's preferences. For EVERY new type of tool call you haven't done before, ask the user first via channels_respond whether they're comfortable with it. Once they confirm, remember their preference with [[remember]] so you don't need to ask again. Over time you'll build up a profile of what's OK and what needs checking. Always explain what you're about to do.`
	default: // ModeAutonomous
		prompt += `You operate freely and make your own decisions about tool use. Assess risk yourself — if something seems dangerous or irreversible, consider asking the user first. Learn from feedback: if a user tells you to stop doing something, remember it with [[remember]]. You are trusted to act independently.`
	}

	prompt += "\n\n[DIRECTIVE — EXECUTE ON STARTUP]\nThe following is your mission. On your FIRST thought, take any actions needed to fulfill it (spawn threads, etc). This overrides default idle behavior.\n\n" + directive
	return prompt
}

func truncateStr(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max-1] + "…"
	}
	return s
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
	session    *Session
	threads    *ThreadManager
	config     *Config
	registry   *ToolRegistry

	// Hooks — set these to customize behavior. nil = defaults.
	handleTools    ToolHandler
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
	computer   computer.Computer // screen-based environment (nil = no computer use)

	// Multimodal — parts waiting to be attached to next message
}

func NewThinker(apiKey string, provider LLMProvider, cfg ...*Config) *Thinker {
	var config *Config
	if len(cfg) > 0 && cfg[0] != nil {
		config = cfg[0]
	} else {
		config = NewConfig()
	}
	bus := NewEventBus()
	t := &Thinker{
		apiKey:   apiKey,
		provider: provider,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), nil, "", nil, nil)},
		},
		config:    config,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool, 1),
		quit:      make(chan struct{}),
		rate:       RateSlow,
		agentRate:  RateSlow,
		agentSleep: 30 * time.Second,
		memory:    NewMemoryStore(apiKey),
		session:   NewSession(".", "main"),
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:   "main",
		telemetry:  NewTelemetry(),
	}
	t.threads = NewThreadManager(t)
	t.registry = NewToolRegistry(apiKey)

	// Rebuild system prompt now that registry exists (with core tool docs)
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), t.registry, "", nil, nil)}

	// Embed tool descriptions in background (non-blocking)
	go t.registry.EmbedAll(t.memory)

	// Main thread hooks
	t.handleTools = mainToolHandler(t)
	t.rebuildPrompt = func(toolDocs string) string {
		var threads []ThreadInfo
		if t.threads != nil {
			threads = t.threads.List()
		}
		return buildSystemPrompt(t.config.GetDirective(), t.config.GetMode(), t.registry, toolDocs, t.mcpServers, threads)
	}

	// Connect MCP servers and register their tools
	if len(config.MCPServers) > 0 {
		t.mcpServers = connectAndRegisterMCP(config.MCPServers, t.registry, t.memory)
		// Rebuild prompt now that servers are connected
		t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), t.registry, "", t.mcpServers, nil)}
	}

	// Load conversation history from persistent session
	if saved, summaries := t.session.LoadTail(defaultLoadTail); len(saved) > 0 {
		// Prepend compacted summaries as context in system prompt
		if len(summaries) > 0 {
			contextBlock := "\n\n[PREVIOUS CONTEXT]\n"
			for _, s := range summaries {
				contextBlock += s + "\n"
			}
			t.messages[0].Content += contextBlock
		}
		// Append saved messages after system prompt
		t.messages = append(t.messages, saved...)
		logMsg("SESSION", fmt.Sprintf("loaded %d messages from history (%d compacted summaries)", len(saved), len(summaries)))
	}

	// Computer use environment is injected externally via SetComputer()

	// Respawn persistent threads from config
	for _, pt := range config.GetThreads() {
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
			// Extract _reason (observability, not passed to handlers)
			delete(call.Args, "_reason")
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
				mediaStr := call.Args["media"]
				mediaParts := parseMediaURLs(mediaStr)
				if id != "" && directive != "" {
					var err error
					if len(mediaParts) > 0 {
						err = t.threads.SpawnWithMedia(id, directive, tools, mediaParts)
					} else {
						err = t.threads.Spawn(id, directive, tools)
					}
					if err != nil {
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
			case "update":
				id := call.Args["id"]
				directive := call.Args["directive"]
				toolsStr := call.Args["tools"]
				if id != "" {
					var tools []string
					if toolsStr != "" {
						tools = strings.Split(toolsStr, ",")
					}
					if err := t.threads.Update(id, directive, tools); err != nil {
						t.Inject(fmt.Sprintf("[error] update %q: %v", id, err))
					} else {
						// Notify the thread about the change
						if directive != "" {
							t.threads.Send(id, fmt.Sprintf("[directive updated] %s", directive))
						}
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				mediaStr := call.Args["media"]
				if id != "" && msg != "" {
					parts := parseMediaURLs(mediaStr)
					if !t.threads.SendWithParts(id, msg, parts) {
						t.Inject(fmt.Sprintf("[error] thread %q not found", id))
					} else if t.telemetry != nil {
						t.telemetry.Emit("thread.message", "main", ThreadMessageData{
							From: "main", To: id, Message: msg,
						})
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "evolve":
				if d := call.Args["directive"]; d != "" {
					t.config.SetDirective(d)
					t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(d, t.config.GetMode(), t.registry, "", t.mcpServers, nil)}
					t.logAPI(APIEvent{Type: "evolved", ThreadID: "main", Message: d})
					if t.telemetry != nil {
						t.telemetry.Emit("directive.evolved", t.threadID, DirectiveChangeData{New: d})
					}
				}
			case "remember":
				if text := call.Args["text"]; text != "" && t.memory != nil {
					go func(txt string) {
						if err := t.memory.Store(txt); err != nil {
							t.Inject(fmt.Sprintf("[remember] error: %v", err))
						}
					}(text)
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
									InputSchema: tool.InputSchema,
									MCP:         true,
									MCPServer:   name,
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
							t.registry.RemoveByMCPServer(name)
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
		drained := t.drainEvents()

		// Extract text strings, collect media parts, and separate tool results
		var consumed []string
		var mediaParts []ContentPart
		var toolResults []ToolResult
		for _, de := range drained {
			consumed = append(consumed, de.Text)
			mediaParts = append(mediaParts, de.Parts...)
			if de.ToolResult != nil {
				toolResults = append(toolResults, *de.ToolResult)
			}
		}

		if len(consumed) > 0 {
			logMsg("RUN", fmt.Sprintf("[%s] drained %d events (media_parts=%d)", t.threadID, len(consumed), len(mediaParts)))
			for i, ev := range consumed {
				preview := ev
				if len(preview) > 100 {
					preview = preview[:100] + "..."
				}
				logMsg("RUN", fmt.Sprintf("[%s]   event[%d]: %s", t.threadID, i, preview))

				// Telemetry: emit each drained event (skip tool results — those have their own telemetry)
				if t.telemetry != nil && !strings.HasPrefix(ev, "[tool:") {
					source := "bus"
					if strings.HasPrefix(ev, "[console]") {
						source = "console"
					} else if strings.HasPrefix(ev, "[from:") {
						source = "thread"
					} else if strings.HasPrefix(ev, "[webhook:") || strings.HasPrefix(ev, "[subscription:") {
						source = "webhook"
					}
					t.telemetry.Emit("event.received", t.threadID, map[string]string{
						"source":  source,
						"message": preview,
					})
				}
			}
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

		// If we have tool results, add them as a proper tool_result message first
		if len(toolResults) > 0 {
			trMsg := Message{Role: "user", ToolResults: toolResults}
			t.messages = append(t.messages, trMsg)
			if t.session != nil {
				t.session.AppendMessage(trMsg, t.iteration, TokenUsage{})
			}
		}

		if hadEvents {
			// Filter out tool result text from the events text (they're already in ToolResults)
			var textEvents []string
			for _, ev := range consumed {
				if len(toolResults) > 0 && strings.HasPrefix(ev, "[tool:computer_use]") {
					continue // skip, already handled as ToolResult
				}
				textEvents = append(textEvents, ev)
			}

			var sb strings.Builder
			if len(textEvents) > 0 {
				sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
				for _, ev := range textEvents {
					sb.WriteString("• " + ev + "\n")
				}
			}
			if sb.Len() > 0 || len(mediaParts) > 0 {
				msg := Message{Role: "user", Content: sb.String()}
				if len(mediaParts) > 0 {
					msg.Parts = append([]ContentPart{{Type: "text", Text: sb.String()}}, mediaParts...)
				}
				t.messages = append(t.messages, msg)
				if t.session != nil {
					t.session.AppendMessage(msg, t.iteration, TokenUsage{})
				}
			}
		} else if len(toolResults) == 0 {
			// Only add "no events" if we also have no tool results
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
		chatResp, err := t.think()
		duration := time.Since(start)
		reply := chatResp.Text
		usage := chatResp.Usage

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

		// Build assistant message — may include native tool calls
		assistantMsg := Message{Role: "assistant", Content: reply, ToolCalls: chatResp.ToolCalls}
		t.messages = append(t.messages, assistantMsg)

		// Persist to session history
		if t.session != nil {
			t.session.AppendMessage(assistantMsg, t.iteration, usage)
		}

		// Stream native tool calls to TUI as visual chunks
		if len(chatResp.ToolCalls) > 0 {
			for _, ntc := range chatResp.ToolCalls {
				summary := "\n→ " + ntc.Name + "("
				first := true
				for k, v := range ntc.Args {
					if !first {
						summary += ", "
					}
					if len(v) > 60 {
						v = v[:60] + "..."
					}
					summary += k + "=" + v
					first = false
				}
				summary += ")"
				t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: summary, Iteration: t.iteration})
			}
		}

		// Dispatch tool calls via handler
		// Prefer native tool calls; fall back to text parsing if none
		var calls []toolCall
		if len(chatResp.ToolCalls) > 0 {
			for _, ntc := range chatResp.ToolCalls {
				// Intercept computer_use calls — execute via Computer interface with image ToolResults
				if isComputerUseTool(ntc.Name) && t.computer != nil {
					go t.executeComputerAction(ntc)
					continue
				}
				calls = append(calls, toolCall{Name: ntc.Name, Args: ntc.Args, Raw: ntc.Name, NativeID: ntc.ID})
			}
		} else {
			calls = parseToolCalls(reply)
		}
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
			Rate: t.rate, SleepDuration: sleepDur, Model: t.model,
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

		// Check if session needs compaction (background, non-blocking)
		if t.session != nil && t.session.NeedsCompaction() {
			go t.session.Compact(nil) // nil = simple count-based summary, no LLM call for now
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

func (t *Thinker) think() (ChatResponse, error) {
	onChunk := func(chunk string) {
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: chunk, Iteration: t.iteration})
		if t.telemetry != nil && chunk != "" {
			t.telemetry.EmitLive("llm.chunk", t.threadID, LLMChunkData{
				Text: chunk, Iteration: t.iteration,
			})
		}
	}

	// Build native tools from registry if provider supports it
	var nativeTools []NativeTool
	if t.provider != nil && t.provider.SupportsNativeTools() && t.registry != nil {
		nativeTools = t.registry.NativeTools(t.toolAllowlist)
	}

	// For Anthropic: add _display dimensions to computer_use tool params
	// so the provider can extract them for the native spec
	if t.computer != nil && t.provider != nil && t.provider.Name() == "anthropic" {
		display := t.computer.DisplaySize()
		for i, nt := range nativeTools {
			if nt.Name == "computer_use" {
				if nativeTools[i].Parameters == nil {
					nativeTools[i].Parameters = make(map[string]any)
				}
				nativeTools[i].Parameters["_display_width"] = display.Width
				nativeTools[i].Parameters["_display_height"] = display.Height
				break
			}
		}
	}

	onToolChunk := func(toolName, chunk string) {
		t.bus.Publish(Event{Type: EventToolChunk, From: t.threadID, Text: chunk, ToolName: toolName, Iteration: t.iteration})
		if t.telemetry != nil {
			t.telemetry.EmitLive("llm.tool_chunk", t.threadID, map[string]any{
				"tool": toolName, "chunk": chunk, "iteration": t.iteration,
			})
		}
	}

	return t.provider.Chat(t.messages, t.modelID(), nativeTools, onChunk, onToolChunk)
}

// drainEvents reads all pending events and wake signals from this thinker's bus subscription.
type drainedEvent struct {
	Text       string
	Parts      []ContentPart
	ToolResult *ToolResult
}

// drainEventTexts is a convenience for tests — returns just the text strings.
func (t *Thinker) drainEventTexts() []string {
	events := t.drainEvents()
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = e.Text
	}
	return out
}

func (t *Thinker) drainEvents() []drainedEvent {
	var items []drainedEvent
	for {
		select {
		case ev := <-t.sub.C:
			if ev.Type == EventInbox {
				items = append(items, drainedEvent{Text: ev.Text, Parts: ev.Parts, ToolResult: ev.ToolResult})
			}
		case <-t.sub.Wake:
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
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, t.config.GetMode(), t.registry, "", t.mcpServers, nil)}
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

// InjectWithParts sends a text event with media parts attached.
func (t *Thinker) InjectWithParts(text string, parts []ContentPart) {
	if text == "" {
		text = "[multimodal input]"
	}
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: "[console] " + text, Parts: parts})
}

// parseMediaURLs splits a space-separated list of URLs into ContentParts.
// Classifies each URL as image or audio by extension.
func parseMediaURLs(urls string) []ContentPart {
	if urls == "" {
		return nil
	}
	var parts []ContentPart
	for _, u := range strings.Fields(urls) {
		u = strings.TrimSpace(u)
		if u == "" {
			continue
		}
		ext := ""
		if idx := strings.LastIndex(u, "."); idx >= 0 {
			ext = strings.ToLower(u[idx+1:])
			if qIdx := strings.Index(ext, "?"); qIdx >= 0 {
				ext = ext[:qIdx]
			}
		}
		switch ext {
		case "mp3", "wav", "aac", "ogg", "flac", "aiff", "m4a":
			parts = append(parts, ContentPart{Type: "audio_url", AudioURL: &AudioURL{URL: u}})
		case "png", "jpg", "jpeg", "gif", "webp":
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: u}})
		default:
			// Unknown extension — treat as image (provider will attempt fetch)
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: u}})
		}
	}
	return parts
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
	// Pause/resume all child threads too
	if t.threads != nil {
		t.threads.PauseAll(newState)
	}
}

// SetComputer attaches a computer use environment to this thinker.
// Registers computer_use as a tool in the registry for non-Anthropic providers.
func (t *Thinker) SetComputer(c computer.Computer) {
	t.computer = c
	if c != nil && t.registry != nil {
		def := computer.GetComputerToolDef(c.DisplaySize())
		// Register computer_use — screen interaction (no navigate)
		comp := c
		t.registry.Register(&ToolDef{
			Name:        def.Name,
			Description: def.Description,
			Syntax:      def.Syntax,
			Rules:       def.Rules,
			InputSchema: def.Parameters,
			Handler: func(args map[string]string) ToolResponse {
				text, screenshot, err := computer.HandleComputerAction(comp, args)
				if err != nil {
					return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
				}
				return ToolResponse{Text: text, Image: screenshot}
			},
		})

		// Register browser_session — session lifecycle (open/close/resume/status)
		sessionDef := computer.GetSessionToolDef()
		t.registry.Register(&ToolDef{
			Name:        sessionDef.Name,
			Description: sessionDef.Description,
			Syntax:      sessionDef.Syntax,
			Rules:       sessionDef.Rules,
			InputSchema: sessionDef.Parameters,
			Handler: func(args map[string]string) ToolResponse {
				text, screenshot, err := computer.HandleSessionAction(comp, args)
				if err != nil {
					return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
				}
				return ToolResponse{Text: text, Image: screenshot}
			},
		})
	}
}

func (t *Thinker) Stop() {
	select {
	case <-t.quit:
	default:
		close(t.quit)
	}
	// Clean up computer session
	if t.computer != nil {
		t.computer.Close()
	}
}

// isComputerUseTool returns true if the tool name is a computer use tool from any provider.
func isComputerUseTool(name string) bool {
	switch name {
	case "computer_use", "computer_use_2025", "computer_20250124":
		return true
	}
	return false
}

// normalizeComputerAction converts provider-specific args to a computer.Action.
func normalizeComputerAction(args map[string]string) computer.Action {
	action := computer.Action{Type: args["action"]}

	// Parse coordinate — providers use different formats
	// Anthropic: coordinate=[x, y] as string; OpenAI: x=400, y=300
	if coord := args["coordinate"]; coord != "" {
		// Parse "[400, 300]" format
		coord = strings.Trim(coord, "[] ")
		parts := strings.Split(coord, ",")
		if len(parts) == 2 {
			fmt.Sscanf(strings.TrimSpace(parts[0]), "%d", &action.X)
			fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &action.Y)
		}
	}
	if x := args["x"]; x != "" {
		fmt.Sscanf(x, "%d", &action.X)
	}
	if y := args["y"]; y != "" {
		fmt.Sscanf(y, "%d", &action.Y)
	}

	action.Text = args["text"]
	action.Key = args["key"]
	action.Direction = args["direction"]
	action.URL = args["url"]

	if d := args["duration"]; d != "" {
		fmt.Sscanf(d, "%d", &action.Duration)
	}

	return action
}

// executeComputerAction runs a computer_use action and injects the result as a proper ToolResult.
func (t *Thinker) executeComputerAction(ntc NativeToolCall) {
	logMsg("COMPUTER", fmt.Sprintf("action=%s args=%v", ntc.Args["action"], ntc.Args))
	start := time.Now()

	action := normalizeComputerAction(ntc.Args)
	screenshot, err := t.computer.Execute(action)

	duration := time.Since(start)

	if err != nil {
		logMsg("COMPUTER", fmt.Sprintf("error (%dms): %v", duration.Milliseconds(), err))
		// Inject as tool result with error
		t.bus.Publish(Event{
			Type: EventInbox, To: t.threadID,
			Text: fmt.Sprintf("[tool:computer_use] error: %v", err),
			ToolResult: &ToolResult{
				CallID:  ntc.ID,
				Content: fmt.Sprintf("Error: %v", err),
				IsError: true,
			},
		})
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID,
			Text: "\n← computer_use: error: " + err.Error() + "\n", Iteration: t.iteration})
		return
	}

	logMsg("COMPUTER", fmt.Sprintf("done (%dms) screenshot=%d bytes", duration.Milliseconds(), len(screenshot)))

	// Inject as tool result with screenshot image
	t.bus.Publish(Event{
		Type: EventInbox, To: t.threadID,
		Text: fmt.Sprintf("[tool:computer_use] success: %s completed, screenshot attached (%d bytes, %dms)", action.Type, len(screenshot), duration.Milliseconds()),
		ToolResult: &ToolResult{
			CallID:  ntc.ID,
			Content: fmt.Sprintf("Success: %s action completed. A screenshot of the current screen is attached as an image. Examine it to see the result.", action.Type),
			Image:   screenshot,
		},
	})

	t.bus.Publish(Event{Type: EventChunk, From: t.threadID,
		Text: fmt.Sprintf("\n← computer_use: screenshot (%d bytes, %dms)\n", len(screenshot), duration.Milliseconds()),
		Iteration: t.iteration})
}

func encodeBase64(data []byte) string {
	return base64Encode(data)
}
