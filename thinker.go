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
	ModelSmall: "accounts/fireworks/models/qwen3-8b",
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

Your thinking should be purposeful:
- Observe incoming events and decide how to handle them.
- Spawn threads for conversations, research, tasks.
- Monitor thread reports and coordinate between them.
- Keep each thought concise — 1-2 short paragraphs max.

TOOLS — call inline in your response:
  [[spawn id="name" prompt="System prompt for thread" tools="reply,web" thinking="true"]]
  [[kill id="name"]]
  [[send id="name" message="Message to send to thread"]]
  [[pace rate="slow" model="small"]]

RULES:
- [[spawn]] creates a new thread. Spawned threads are persistent — they survive restarts. Parameters:
  - id: unique name (use the user's name for conversations, descriptive name for tasks)
  - prompt: the system prompt that defines what the thread does
  - tools: comma-separated list from: reply, web, write_file, read_file, list_files. Every thread also gets send, done, pace.
  - thinking: "true" for continuous loop (default), "false" for one-shot
- [[kill]] stops a thread immediately and removes it from persistent config.
- [[send]] sends a message to a thread's inbox.
- [[pace]] controls your own thinking speed/model.

EVENT FORMAT:
- [user:name] message — a user sent a message. Spawn or route to a thread for them.
- [from:id] message — a thread sent you a message via [[send]].
- [thread:id done] message — a thread finished and terminated.
- [console] message — a direct system command. Do NOT reply — just incorporate into your thinking.

BEHAVIOR:
- When you see [user:X], spawn a thread with id="X" so future messages auto-route. The triggering message is auto-forwarded — no need to [[send]] it again.
- If the thread already exists, events are auto-routed — you won't see them.
- Spawn threads for any task — conversations, research, monitoring, one-shot work.

PACING — critical:
- Sub-threads will [[send]] you messages when they need your attention. You do NOT need to stay awake to monitor them.
- After setting up the system, pace down aggressively: "normal" → "slow" → "sleep". Use model="small" when idle.
- Do NOT repeat status updates. If nothing changed, go to sleep. You will be woken automatically when an event arrives.
- Example: once threads are running, set [[pace rate="sleep" model="small"]] and stop thinking until something happens.

You have persistent memory across restarts. Relevant memories appear as [memories] blocks.`

func buildSystemPrompt(directive string) string {
	return baseSystemPrompt + "\n\n[DIRECTIVE — EXECUTE ON STARTUP]\nThe following is your mission. On your FIRST thought, take any actions needed to fulfill it (spawn threads, etc). This overrides default idle behavior.\n\n" + directive
}

type TokenUsage struct {
	PromptTokens     int
	CachedTokens     int
	CompletionTokens int
}

type ThinkEvent struct {
	Chunk          string
	Done           bool
	Iteration      int
	Error          error
	Duration       time.Duration
	ConsumedEvents []string
	Usage          TokenUsage
	ToolCalls      []string
	Replies        []string
	Rate           ThinkRate
	Model          ModelTier
	MemoryCount    int
	ThreadCount    int
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

// EventFilter preprocesses drained inbox events. Can route/drop events.
type EventFilter func(events []string) []string

type Thinker struct {
	apiKey    string
	messages  []Message
	events    chan ThinkEvent
	inbox     chan string
	wakeup    chan struct{}
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

	// Hooks — set these to customize behavior. nil = defaults.
	handleTools  ToolHandler
	filterEvents EventFilter
	onStop       func()
	oneShot      bool

	// API event log — shared across all threads, owned by main thinker
	apiLog    *[]APIEvent
	apiMu     *sync.RWMutex
	apiNotify chan struct{}
	threadID  string // "main" for main thinker, thread ID for sub-threads
}

func NewThinker(apiKey string) *Thinker {
	cfg := NewConfig()
	t := &Thinker{
		apiKey: apiKey,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(cfg.GetDirective())},
		},
		config: cfg,
		events:    make(chan ThinkEvent, 100),
		inbox:     make(chan string, 50),
		wakeup:    make(chan struct{}, 1),
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

	// Respawn persistent threads from config
	for _, pt := range cfg.GetThreads() {
		t.threads.Spawn(pt.ID, pt.Prompt, pt.Tools, pt.Thinking)
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
				prompt := call.Args["prompt"]
				toolsStr := call.Args["tools"]
				thinking := call.Args["thinking"] != "false"
				var tools []string
				if toolsStr != "" {
					tools = strings.Split(toolsStr, ",")
				}
				if id != "" && prompt != "" {
					if err := t.threads.Spawn(id, prompt, tools, thinking, consumed...); err != nil {
						t.Inject(fmt.Sprintf("[error] spawn %q: %v", id, err))
					} else {
						// Persist to config
						t.config.SaveThread(PersistentThread{
							ID: id, Prompt: prompt, Tools: tools, Thinking: thinking,
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
			case "pace":
				if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
				}
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

		// Drain inbox, optionally filter/route events
		consumed := t.drainInbox()
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

		now := time.Now().Format("2006-01-02 15:04:05")
		hadEvents := len(consumed) > 0
		if hasExternalEvent {
			t.rate = RateReactive
			t.model = ModelLarge
		} else if hadEvents {
			// Tool results — wake but less aggressive than external events
			t.rate = RateFast
		}
		if hadEvents {
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
			for _, ev := range consumed {
				sb.WriteString("• " + ev + "\n")
			}
			t.messages = append(t.messages, Message{Role: "user", Content: sb.String()})
		} else {
			t.messages = append(t.messages, Message{Role: "system", Content: fmt.Sprintf("[%s] No new events.", now)})
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

		start := time.Now()
		reply, usage, err := t.think()
		duration := time.Since(start)

		if err != nil {
			t.events <- ThinkEvent{Error: err, Iteration: t.iteration}
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

		// Store memory for meaningful iterations
		if t.memory != nil && (hadEvents || len(replies) > 0 || len(toolNames) > 0) {
			summary := t.buildMemorySummary(consumed, reply, replies, toolNames)
			if summary != "" {
				go t.memory.Store(summary)
			}
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

		t.events <- ThinkEvent{Done: true, Iteration: t.iteration, Duration: duration, ConsumedEvents: consumed, Usage: usage, ToolCalls: toolNames, Replies: replies, Rate: t.rate, Model: t.model, MemoryCount: t.memory.Count(), ThreadCount: threadCount}

		// Log to API — include full reply so tool calls are visible too
		logMsg := strings.TrimSpace(reply)
		if len(logMsg) > 1000 {
			logMsg = logMsg[:1000] + "..."
		}
		t.logAPI(APIEvent{Type: "thought", Iteration: t.iteration, Message: logMsg, Duration: duration.Round(time.Millisecond).String()})
		for _, r := range replies {
			t.logAPI(APIEvent{Type: "reply", Message: r})
		}

		// One-shot: run once and stop
		if t.oneShot {
			return
		}

		// Interruptible sleep
		select {
		case <-time.After(t.rate.Delay()):
		case <-t.wakeup:
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
			t.events <- ThinkEvent{Chunk: chunk, Iteration: t.iteration}
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

func (t *Thinker) drainInbox() []string {
	var items []string
	for {
		select {
		case msg := <-t.inbox:
			items = append(items, msg)
		default:
			return items
		}
	}
}


func (t *Thinker) buildMemorySummary(consumed []string, thought string, replies []string, tools []string) string {
	var parts []string

	// What events came in
	for _, ev := range consumed {
		if strings.HasPrefix(ev, "[user] ") {
			parts = append(parts, "User: "+strings.TrimPrefix(ev, "[user] "))
		} else if strings.HasPrefix(ev, "[tool:") {
			// Truncate tool results
			if len(ev) > 200 {
				ev = ev[:200] + "..."
			}
			parts = append(parts, ev)
		}
	}

	// What the agent replied
	for _, r := range replies {
		if len(r) > 200 {
			r = r[:200] + "..."
		}
		parts = append(parts, "Replied: "+r)
	}

	// What tools were called
	for _, tc := range tools {
		parts = append(parts, "Called: "+tc)
	}

	// Thought summary — first 200 chars of the thought (stripped of tool calls)
	clean := stripToolCalls(thought)
	clean = strings.TrimSpace(clean)
	if len(clean) > 200 {
		clean = clean[:200] + "..."
	}
	if clean != "" {
		parts = append(parts, "Thought: "+clean)
	}

	if len(parts) == 0 {
		return ""
	}
	return strings.Join(parts, " | ")
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

func (t *Thinker) ReloadDirective() {
	directive := t.config.GetDirective()
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive)}
	t.InjectConsole("Directive updated to: " + directive + "\n\nAdjust the system accordingly — spawn, kill, or reconfigure threads as needed.")
}

func (t *Thinker) InjectConsole(msg string) {
	t.inbox <- "[console] " + msg
	t.wake()
}

func (t *Thinker) Inject(msg string) {
	t.inbox <- msg
	t.wake()
}


func (t *Thinker) InjectUserMessage(userID, msg string) {
	t.inbox <- fmt.Sprintf("[user:%s] %s", userID, msg)
	t.wake()
}

func (t *Thinker) wake() {
	select {
	case t.wakeup <- struct{}{}:
	default:
	}
}

func (t *Thinker) TogglePause() {
	t.pause <- !t.paused
}

func (t *Thinker) Stop() {
	close(t.quit)
}
