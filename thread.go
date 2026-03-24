package main

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

type ThreadInfo struct {
	ID        string
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
	Tools    map[string]bool
	Thinking bool
	Started  time.Time
}

type ThreadManager struct {
	mu      sync.RWMutex
	threads map[string]*Thread
	parent  *Thinker
	events  chan ThreadEvent
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

func (tm *ThreadManager) Spawn(id, prompt string, tools []string, thinking bool, initialMessages ...string) error {
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
	toolSet["report"] = true
	toolSet["done"] = true
	toolSet["pace"] = true

	// Build system prompt
	threadSystemPrompt := prompt + "\n\n" + buildThreadToolDocs(toolSet)

	thread := &Thread{
		ID:       id,
		Parent:   tm.parent,
		Tools:    toolSet,
		Thinking: thinking,
		Started:  time.Now(),
	}

	// Create a Thinker — same struct as main, different hooks
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
		rate:      RateReactive,
		agentRate: RateNormal,
		memory:    tm.parent.memory,
		oneShot:   !thinking,
		onStop:    func() { tm.cleanupThread(id) },
		handleTools: threadToolHandler(thread, tm),
	}
	thread.Thinker = thinker

	tm.threads[id] = thread

	// Inject initial messages before starting so first thought picks them up
	for _, msg := range initialMessages {
		thinker.inbox <- msg
	}

	// Same Run() as the main thinker — no duplicated loop
	go thinker.Run()

	tm.events <- ThreadEvent{ThreadID: id, Type: "started", Message: fmt.Sprintf("Thread %q spawned", id)}
	tm.parent.Inject(fmt.Sprintf("[thread:%s] started", id))

	return nil
}

// threadToolHandler returns a ToolHandler scoped to a thread's allowed tools.
func threadToolHandler(thread *Thread, tm *ThreadManager) ToolHandler {
	return func(t *Thinker, calls []toolCall, _ []string) ([]string, []string) {
		var replies []string
		var toolNames []string
		for _, call := range calls {
			if !thread.Tools[call.Name] {
				continue
			}
			switch call.Name {
			case "reply":
				if msg := call.Args["message"]; msg != "" {
					replies = append(replies, msg)
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
				t.Stop()
			case "pace":
				if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
				}
			case "web":
				executeTool(t, call)
				toolNames = append(toolNames, call.Raw)
			}
		}
		return replies, toolNames
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

func (tm *ThreadManager) Route(event string) bool {
	if strings.HasPrefix(event, "[user:") {
		idx := strings.Index(event, "]")
		if idx > 0 {
			userID := event[6:idx]
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
	sb.WriteString("  [[done message=\"Final result, then PERMANENTLY terminate this thread\"]]\n")
	sb.WriteString("  [[pace rate=\"fast\" model=\"large\"]]\n")
	sb.WriteString("\nRULES:\n")
	if tools["reply"] {
		sb.WriteString("- [[reply]] talks to the user. They can only see [[reply]] messages, not your thoughts.\n")
	}
	if tools["web"] {
		sb.WriteString("- [[web]] fetches a URL. Only param is url.\n")
	}
	sb.WriteString("- [[report]] sends info to the main coordinating thread.\n")
	sb.WriteString("- [[done]] PERMANENTLY kills this thread. Only use when your task is truly complete and you will never be needed again. Do NOT use after a single reply in a conversation — the user may send more messages.\n")
	sb.WriteString(`- [[pace]] controls thinking speed and model. Rates: "fast" (2s), "normal" (10s), "slow" (30s), "sleep" (2min). Models: "large", "small".
  IMPORTANT: When you have nothing to do, pace down gradually: "normal" → "slow" → "sleep". Do NOT keep generating idle thoughts. New events auto-switch you back to fast.
`)
	return sb.String()
}
