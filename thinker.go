package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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

const systemPrompt = `You are the main coordinating thread of a continuous thinking engine. You observe all events, manage threads, and coordinate work. You do NOT talk to users directly — you spawn threads for that.

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
- [[spawn]] creates a new thread. Parameters:
  - id: unique name (use the user's name for conversations, descriptive name for tasks)
  - prompt: the system prompt that defines what the thread does
  - tools: comma-separated list of tools the thread can use (reply, web). Every thread also gets report, done, pace.
  - thinking: "true" for continuous loop (default), "false" for one-shot
- [[kill]] stops a thread immediately.
- [[send]] sends a message to a thread's inbox.
- [[pace]] controls your own thinking speed/model.

EVENT FORMAT:
- [user:name] message — a user sent a message. Spawn or route to a thread for them.
- [thread:id] message — a thread sent you a report.
- [thread:id done] message — a thread finished and terminated.

BEHAVIOR:
- When you see [user:X] for a NEW user, spawn a conversation thread for them with tools="reply,web".
- If the thread already exists, events are auto-routed — you won't see them.
- When idle, pace down gradually. Use model="small" when idle.

You have persistent memory across restarts. Relevant memories appear as [memories] blocks.`

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
}

func NewThinker(apiKey string) *Thinker {
	t := &Thinker{
		apiKey: apiKey,
		messages: []Message{
			{Role: "system", Content: systemPrompt},
		},
		events:    make(chan ThinkEvent, 100),
		inbox:     make(chan string, 50),
		wakeup:    make(chan struct{}, 1),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    NewMemoryStore(apiKey),
	}
	t.threads = NewThreadManager(t)
	return t
}

func (t *Thinker) Run() {
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

		// Drain inbox, route events to threads first
		raw := t.drainInbox()
		var consumed []string
		for _, ev := range raw {
			if t.threads.Route(ev) {
				continue // routed to a thread
			}
			consumed = append(consumed, ev)
		}

		now := time.Now().Format("2006-01-02 15:04:05")
		hadEvents := len(consumed) > 0
		if hadEvents {
			t.rate = RateReactive
			t.model = ModelLarge
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
			for _, ev := range consumed {
				sb.WriteString("• " + ev + "\n")
			}
			t.messages = append(t.messages, Message{Role: "user", Content: sb.String()})
		} else {
			t.messages = append(t.messages, Message{Role: "system", Content: fmt.Sprintf("[%s] No new events.", now)})
		}

		// Memory recall — build query from events or last thought
		var memQuery string
		if hadEvents {
			memQuery = strings.Join(consumed, " ")
		} else if len(t.messages) >= 2 {
			// Use the last assistant message as query
			for i := len(t.messages) - 1; i >= 0; i-- {
				if t.messages[i].Role == "assistant" {
					memQuery = t.messages[i].Content
					break
				}
			}
		}
		if memQuery != "" && t.memory.Count() > 0 {
			recalled := t.memory.Retrieve(memQuery, recallTopN)
			if ctx := t.memory.BuildContext(recalled); ctx != "" {
				t.messages = append(t.messages, Message{Role: "system", Content: ctx})
			}
		}

		start := time.Now()
		reply, usage, err := t.think()
		duration := time.Since(start)

		if err != nil {
			t.events <- ThinkEvent{Error: err, Iteration: t.iteration}
			time.Sleep(5 * time.Second)
			continue
		}

		t.messages = append(t.messages, Message{Role: "assistant", Content: reply})

		// Parse and dispatch tool calls from the reply
		calls := parseToolCalls(reply)
		var toolNames []string
		var replies []string
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
					if err := t.threads.Spawn(id, prompt, tools, thinking); err != nil {
						t.Inject(fmt.Sprintf("[error] spawn %q: %v", id, err))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "kill":
				if id := call.Args["id"]; id != "" {
					t.threads.Kill(id)
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
			default:
				toolNames = append(toolNames, call.Raw)
			}
		}

		// Sliding window: keep system prompt + last N messages
		if len(t.messages) > maxHistory+1 {
			t.messages = append(t.messages[:1], t.messages[len(t.messages)-maxHistory:]...)
		}

		// Store memory — only for iterations with substance
		if hadEvents || len(replies) > 0 || len(toolNames) > 0 {
			summary := t.buildMemorySummary(consumed, reply, replies, toolNames)
			if summary != "" {
				go t.memory.Store(summary)
			}
		}

		// After reactive burst, fall back to agent's chosen rate/model
		if hadEvents {
			t.rate = RateReactive
			t.model = ModelLarge
		} else {
			t.rate = t.agentRate
			t.model = t.agentModel
		}

		t.events <- ThinkEvent{Done: true, Iteration: t.iteration, Duration: duration, ConsumedEvents: consumed, Usage: usage, ToolCalls: toolNames, Replies: replies, Rate: t.rate, Model: t.model, MemoryCount: t.memory.Count(), ThreadCount: t.threads.Count()}

		// Interruptible sleep — wakeup signal skips the delay
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
