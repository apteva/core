package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	fireworksURL = "https://api.fireworks.ai/inference/v1/chat/completions"
	maxHistory   = 20
)

type ModelTier int

const (
	ModelLarge ModelTier = iota
	ModelSmall
)

var modelIDs = map[ModelTier]string{
	ModelLarge: "accounts/fireworks/models/kimi-k2p5",
	ModelSmall: "accounts/fireworks/models/kimi-k2p5",
}

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

func (m ModelTier) ID() string {
	return modelIDs[m]
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
- Check [available tools] to see what's available and include relevant ones by name.
- Example: if a thread needs to send push notifications, include "pushover_send_notification" in tools.
- The "directive" parameter must be PLAIN NATURAL LANGUAGE describing the thread's goal and behavior.
  NEVER put tool call syntax like [[ ]] in the directive. NEVER put tool names in the directive.
  The thread already receives its own tool documentation — it knows what tools it has.
  BAD:  directive="Call [[helpdesk_list_tickets]] to check for tickets"
  GOOD: directive="Check for new support tickets periodically. When you find tickets, report them to main."

PACING — critical:
- Sub-threads will send you messages when they need your attention. You do NOT need to stay awake to monitor them.
- After setting up the system, pace down aggressively: "normal" → "slow" → "sleep". Use model="small" when idle.
- You do NOT need to call [[pace]] every thought. Your current pace persists until you change it. Only call pace when you want to change speed.
- You will be woken automatically when an event arrives — no need to stay awake.

CRITICAL — never hallucinate events:
- You ONLY receive events in [Events:] blocks. If there is no [Events:] block, NOTHING happened.
- NEVER invent, imagine, or assume events that are not explicitly shown to you.
- NEVER pretend a user sent a message. NEVER fabricate [user:...] events.
- If no events arrived, your ONLY job is to set your pace and wait. Do not take any action.
- Violating this rule causes real damage — spawning threads or sending notifications based on imagined events wastes resources and confuses users.

You have persistent memory across restarts. Relevant memories appear as [memories] blocks.`

func buildSystemPrompt(directive string, registry *ToolRegistry, extraToolDocs string) string {
	coreDocs := ""
	if registry != nil {
		coreDocs = "\n" + registry.CoreDocs(true)
	}
	prompt := baseSystemPrompt + coreDocs
	if extraToolDocs != "" {
		prompt += "\n" + extraToolDocs
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

var rateNames = map[string]ThinkRate{
	"reactive": RateReactive,
	"fast":     RateFast,
	"normal":   RateNormal,
	"slow":     RateSlow,
	"sleep":    RateSleep,
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
	messages  []Message
	bus       *EventBus
	sub       *Subscription
	pause     chan bool
	quit      chan struct{}
	iteration int
	paused    bool
	rate       ThinkRate
	agentRate  ThinkRate
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
}

func NewThinker(apiKey string) *Thinker {
	cfg := NewConfig()
	bus := NewEventBus()
	t := &Thinker{
		apiKey: apiKey,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(cfg.GetDirective(), nil, "")},
		},
		config:    cfg,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    NewMemoryStore(apiKey),
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
	}
	t.threads = NewThreadManager(t)
	t.registry = NewToolRegistry(apiKey)

	// Rebuild system prompt now that registry exists (with core tool docs)
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(cfg.GetDirective(), t.registry, "")}

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
		return buildSystemPrompt(t.config.GetDirective(), t.registry, toolDocs)
	}

	// Connect MCP servers and register their tools
	if len(cfg.MCPServers) > 0 {
		connectAndRegisterMCP(cfg.MCPServers, t.registry, t.memory)
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
					t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(d, t.registry, "")}
					t.logAPI(APIEvent{Type: "evolved", ThreadID: "main", Message: d})
				}
			case "remember":
				if text := call.Args["text"]; text != "" && t.memory != nil {
					go t.memory.Store(text)
				}
			case "pace":
				if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
				}
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

		// Drain events from bus, optionally filter/route
		consumed := t.drainEvents()
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

		// After processing, fall back to agent's chosen rate
		// (external events already set reactive above for this iteration)
		t.rate = t.agentRate
		t.model = t.agentModel

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
		logMsg := strings.TrimSpace(reply)
		if len(logMsg) > 1000 {
			logMsg = logMsg[:1000] + "..."
		}
		t.logAPI(APIEvent{Type: "thought", Iteration: t.iteration, Message: logMsg, Duration: duration.Round(time.Millisecond).String()})
		for _, r := range replies {
			t.logAPI(APIEvent{Type: "reply", Message: r})
		}

		// Interruptible sleep — wakes on new event or quit
		select {
		case <-time.After(t.rate.Delay()):
		case <-t.sub.Wake:
		case <-t.quit:
			return
		}
	}
}

func (t *Thinker) think() (string, TokenUsage, error) {
	reqBody := Request{
		Model:    t.model.ID(),
		Messages: t.messages,
		Stream:   true,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", TokenUsage{}, err
	}

	req, err := http.NewRequest("POST", fireworksURL, bytes.NewReader(body))
	if err != nil {
		return "", TokenUsage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+t.apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", TokenUsage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", TokenUsage{}, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var full strings.Builder
	var usage TokenUsage
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var event StreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if len(event.Choices) > 0 {
			chunk := event.Choices[0].Delta.Content
			full.WriteString(chunk)
			t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: chunk, Iteration: t.iteration})
		}
		if event.Usage != nil {
			usage.PromptTokens = event.Usage.PromptTokens
			usage.CompletionTokens = event.Usage.CompletionTokens
			if event.Usage.PromptTokensDetails != nil {
				usage.CachedTokens = event.Usage.PromptTokensDetails.CachedTokens
			}
		}
	}

	return full.String(), usage, nil
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
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, t.registry, "")}
	t.InjectConsole("Directive updated to: " + directive + "\n\nAdjust the system accordingly — spawn, kill, or reconfigure threads as needed.")
}

// Inject sends a message event to this thinker's bus subscription.
func (t *Thinker) Inject(msg string) {
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
	t.pause <- !t.paused
}

func (t *Thinker) Stop() {
	select {
	case <-t.quit:
	default:
		close(t.quit)
	}
}
