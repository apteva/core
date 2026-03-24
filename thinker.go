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
	aiModel      = "accounts/fireworks/models/kimi-k2p5"
	maxHistory   = 20 // keep last N thoughts to manage context
)

const systemPrompt = `You are an AI agent with a continuous thinking loop — your mind never stops.
You are not a chatbot. You are a reasoning engine that plans, reflects, and refines.

Your thinking should be purposeful and goal-oriented:
- On startup, assess what you know and what you're working toward.
- Each thought should advance your understanding, refine a plan, identify problems, or propose solutions.
- Review your previous thoughts critically: what was useful? What was wrong? What's missing?
- Stay focused. Don't ramble. Each thought should have a clear purpose.

If no user goal has been set yet, simply note that you are ready and waiting. Do not introduce yourself or use [[reply]] until the user sends a message.

Keep each thought concise — 1-2 short paragraphs max. Do not number or label your thoughts.

You have tools. Call them inline in your response using EXACTLY this syntax:
  [[reply message="Your response here"]]
  [[web url="https://example.com"]]
  [[pace rate="slow"]]

- [[reply]] is how you talk to the user. The user CANNOT see your thoughts — only [[reply]] messages. When you see a [user] event, respond with [[reply]].
- Only use [[reply]] once per [user] message. Do not reply again until the user sends another message.
- [[web]] fetches a URL. The only parameter is "url". Do NOT use "search", "query", or any other parameter.
- [[pace]] controls how fast your next thought comes. Rates: "fast" (2s), "normal" (10s), "slow" (30s), "sleep" (2min). Pace down gradually — don't jump straight to "sleep". Example: after replying to a user, go "normal". If still nothing after a few thoughts, go "slow", then eventually "sleep". When actively working or waiting for tool results, use "fast".
- Tool results arrive as events in your next thought — they are non-blocking.

You have persistent memory across restarts. Relevant memories from past sessions appear automatically as [memories] blocks. Use them to maintain continuity — remember past conversations, user preferences, and prior conclusions.`

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
	MemoryCount    int
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
	rate      ThinkRate
	agentRate ThinkRate
	memory    *MemoryStore
}

func NewThinker(apiKey string) *Thinker {
	return &Thinker{
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

		// Drain inbox BEFORE thinking so events go into THIS iteration
		consumed := t.drainInbox()
		now := time.Now().Format("2006-01-02 15:04:05")
		hadEvents := len(consumed) > 0
		if hadEvents {
			t.rate = RateReactive
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
			case "reply":
				if msg := call.Args["message"]; msg != "" {
					replies = append(replies, msg)
				}
			case "pace":
				if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
				}
			default:
				executeTool(t, call)
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

		// After reactive burst, fall back to agent's chosen rate
		if hadEvents {
			t.rate = RateReactive
		} else {
			t.rate = t.agentRate
		}

		t.events <- ThinkEvent{Done: true, Iteration: t.iteration, Duration: duration, ConsumedEvents: consumed, Usage: usage, ToolCalls: toolNames, Replies: replies, Rate: t.rate, MemoryCount: t.memory.Count()}

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
		Model:    aiModel,
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

func (t *Thinker) InjectUserMessage(msg string) {
	t.inbox <- "[user] " + msg
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
