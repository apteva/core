package main

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

type ThreadInfo struct {
	ID        string
	Prompt    string
	Tools     []string
	Thinking  bool
	Running   bool
	Iteration int
	Rate      ThinkRate
	Model     ModelTier
	Started   time.Time
}

type Thread struct {
	ID       string
	Thinker  *Thinker
	Parent   *Thinker
	Tools    map[string]bool // allowed tool names
	Thinking bool
	Started  time.Time
}

type ThreadManager struct {
	mu      sync.RWMutex
	threads map[string]*Thread
	parent  *Thinker
	events  chan ThreadEvent // thread lifecycle events for the TUI
}

type ThreadEvent struct {
	ThreadID string
	Type     string // "started", "done", "reply", "report"
	Message  string
}

func NewThreadManager(parent *Thinker) *ThreadManager {
	return &ThreadManager{
		threads: make(map[string]*Thread),
		parent:  parent,
		events:  make(chan ThreadEvent, 100),
	}
}

func (tm *ThreadManager) Spawn(id, prompt string, tools []string, thinking bool) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.threads[id]; exists {
		return fmt.Errorf("thread %q already exists", id)
	}

	// Build tool set
	toolSet := make(map[string]bool)
	for _, t := range tools {
		toolSet[strings.TrimSpace(t)] = true
	}
	// Every thread always gets report and done
	toolSet["report"] = true
	toolSet["done"] = true
	toolSet["pace"] = true

	// Build system prompt for the thread
	threadSystemPrompt := prompt + "\n\n" + buildThreadToolDocs(toolSet)

	// Create a Thinker for this thread with its own channels and state
	thinker := &Thinker{
		apiKey: tm.parent.apiKey,
		messages: []Message{
			{Role: "system", Content: threadSystemPrompt},
		},
		events:    make(chan ThinkEvent, 100),
		inbox:     make(chan string, 50),
		wakeup:    make(chan struct{}, 1),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateNormal,
		agentRate: RateNormal,
		memory:    tm.parent.memory, // share memory with parent
	}

	thread := &Thread{
		ID:       id,
		Thinker:  thinker,
		Parent:   tm.parent,
		Tools:    toolSet,
		Thinking: thinking,
		Started:  time.Now(),
	}

	tm.threads[id] = thread

	// Start the thread
	if thinking {
		go tm.runThinkingThread(thread)
	} else {
		go tm.runOneShotThread(thread)
	}

	tm.events <- ThreadEvent{ThreadID: id, Type: "started", Message: fmt.Sprintf("Thread %q spawned", id)}
	tm.parent.Inject(fmt.Sprintf("[thread:%s] started", id))

	return nil
}

func (tm *ThreadManager) runThinkingThread(thread *Thread) {
	t := thread.Thinker
	for {
		select {
		case <-t.quit:
			tm.cleanupThread(thread.ID)
			return
		default:
		}

		t.iteration++

		consumed := t.drainInbox()
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

		start := time.Now()
		reply, _, err := t.think()
		duration := time.Since(start)

		if err != nil {
			t.events <- ThinkEvent{Error: err, Iteration: t.iteration}
			select {
			case <-time.After(5 * time.Second):
			case <-t.quit:
				tm.cleanupThread(thread.ID)
				return
			}
			continue
		}

		t.messages = append(t.messages, Message{Role: "assistant", Content: reply})

		// Process tool calls with allowlist
		tm.processToolCalls(thread, reply, consumed)

		// Sliding window
		if len(t.messages) > maxHistory+1 {
			t.messages = append(t.messages[:1], t.messages[len(t.messages)-maxHistory:]...)
		}

		if hadEvents {
			t.rate = RateReactive
			t.model = ModelLarge
		} else {
			t.rate = t.agentRate
			t.model = t.agentModel
		}

		t.events <- ThinkEvent{Done: true, Iteration: t.iteration, Duration: duration, Rate: t.rate, Model: t.model}

		select {
		case <-time.After(t.rate.Delay()):
		case <-t.wakeup:
		case <-t.quit:
			tm.cleanupThread(thread.ID)
			return
		}
	}
}

func (tm *ThreadManager) runOneShotThread(thread *Thread) {
	t := thread.Thinker

	// Wait for the first event (the initial task should be in inbox already)
	consumed := t.drainInbox()
	now := time.Now().Format("2006-01-02 15:04:05")
	if len(consumed) > 0 {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
		for _, ev := range consumed {
			sb.WriteString("• " + ev + "\n")
		}
		t.messages = append(t.messages, Message{Role: "user", Content: sb.String()})
	}

	t.iteration = 1
	reply, _, err := t.think()
	if err != nil {
		tm.parent.Inject(fmt.Sprintf("[thread:%s] error: %v", thread.ID, err))
		tm.cleanupThread(thread.ID)
		return
	}

	t.messages = append(t.messages, Message{Role: "assistant", Content: reply})
	tm.processToolCalls(thread, reply, consumed)

	// If the thread didn't call [[done]], auto-done with the reply
	clean := stripToolCalls(reply)
	clean = strings.TrimSpace(clean)
	if clean != "" {
		tm.parent.Inject(fmt.Sprintf("[thread:%s] %s", thread.ID, truncate(clean, 500)))
	}

	// Drain events channel
	for {
		select {
		case <-t.events:
		default:
			goto done
		}
	}
done:
	tm.cleanupThread(thread.ID)
}

func (tm *ThreadManager) processToolCalls(thread *Thread, reply string, consumed []string) {
	calls := parseToolCalls(reply)
	for _, call := range calls {
		// Check allowlist
		if !thread.Tools[call.Name] {
			continue
		}

		switch call.Name {
		case "reply":
			if msg := call.Args["message"]; msg != "" {
				tm.events <- ThreadEvent{ThreadID: thread.ID, Type: "reply", Message: msg}
			}
		case "report":
			if msg := call.Args["message"]; msg != "" {
				thread.Parent.Inject(fmt.Sprintf("[thread:%s] %s", thread.ID, msg))
				tm.events <- ThreadEvent{ThreadID: thread.ID, Type: "report", Message: msg}
			}
		case "done":
			msg := call.Args["message"]
			if msg != "" {
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done] %s", thread.ID, msg))
			} else {
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done]", thread.ID))
			}
			tm.events <- ThreadEvent{ThreadID: thread.ID, Type: "done", Message: msg}
			thread.Thinker.Stop()
		case "pace":
			if r, ok := rateNames[call.Args["rate"]]; ok {
				thread.Thinker.agentRate = r
			}
			if m, ok := modelNames[call.Args["model"]]; ok {
				thread.Thinker.agentModel = m
			}
		case "web":
			executeTool(thread.Thinker, call)
		}
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
}

func (tm *ThreadManager) Send(id, message string) bool {
	tm.mu.RLock()
	thread, exists := tm.threads[id]
	tm.mu.RUnlock()

	if !exists {
		return false
	}
	thread.Thinker.Inject(message)
	return true
}

// Route tries to route an event to a matching thread. Returns true if routed.
func (tm *ThreadManager) Route(event string) bool {
	// Extract user ID from "[user:xxx] message" format
	if strings.HasPrefix(event, "[user:") {
		idx := strings.Index(event, "]")
		if idx > 0 {
			userID := event[6:idx] // extract "xxx" from "[user:xxx]"
			msg := strings.TrimSpace(event[idx+1:])

			tm.mu.RLock()
			thread, exists := tm.threads[userID]
			tm.mu.RUnlock()

			if exists {
				thread.Thinker.Inject(fmt.Sprintf("[user] %s", msg))
				return true
			}
		}
	}
	return false
}

func (tm *ThreadManager) List() []ThreadInfo {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	var infos []ThreadInfo
	for _, t := range tm.threads {
		infos = append(infos, ThreadInfo{
			ID:        t.ID,
			Tools:     toolSetToSlice(t.Tools),
			Thinking:  t.Thinking,
			Running:   true,
			Iteration: t.Thinker.iteration,
			Rate:      t.Thinker.rate,
			Model:     t.Thinker.model,
			Started:   t.Started,
		})
	}
	return infos
}

func (tm *ThreadManager) Count() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.threads)
}

func (tm *ThreadManager) cleanupThread(id string) {
	tm.mu.Lock()
	delete(tm.threads, id)
	tm.mu.Unlock()
	tm.events <- ThreadEvent{ThreadID: id, Type: "done"}
}

func toolSetToSlice(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

func buildThreadToolDocs(tools map[string]bool) string {
	var sb strings.Builder
	sb.WriteString("You have tools. Call them inline:\n")

	if tools["reply"] {
		sb.WriteString("  [[reply message=\"Your response to the user\"]]\n")
	}
	if tools["web"] {
		sb.WriteString("  [[web url=\"https://example.com\"]]\n")
	}
	sb.WriteString("  [[report message=\"Send observation to main thread\"]]\n")
	sb.WriteString("  [[done message=\"Final result, then terminate\"]]\n")
	sb.WriteString("  [[pace rate=\"fast\" model=\"large\"]]\n")
	sb.WriteString("\nRULES:\n")
	if tools["reply"] {
		sb.WriteString("- [[reply]] talks to the user. They can only see [[reply]] messages, not your thoughts.\n")
	}
	if tools["web"] {
		sb.WriteString("- [[web]] fetches a URL. Only param is url.\n")
	}
	sb.WriteString("- [[report]] sends info to the main coordinating thread.\n")
	sb.WriteString("- [[done]] sends a final summary and terminates this thread.\n")
	sb.WriteString("- [[pace]] controls thinking speed and model.\n")
	return sb.String()
}
