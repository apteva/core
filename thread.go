package main

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

const baseThreadPrompt = `You are a thread in a continuous thinking engine. Your loop runs forever.

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Use [[send id="main" message="..."]] to report results to the coordinator.
- Keep each thought concise — 1-2 short paragraphs max.

PACING — this is critical:
- Tool results (like [[list_files]] or [[web]]) will wake you up for the next thought. Do NOT set [[pace]] in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Example flow: Thought 1: call [[list_files]]. Thought 2: process results, [[send]] report, [[pace rate="sleep"]].
- Only use [[pace]] when you have NO pending tool calls and are ready to wait.`

type ThreadInfo struct {
	ID        string
	Directive string
	Tools     []string
	Running   bool
	Iteration int
	Rate      ThinkRate
	Model     ModelTier
	Started   time.Time
}

type Thread struct {
	ID        string
	Directive string // original directive before tool docs
	Thinker  *Thinker
	Parent   *Thinker
	Tools    map[string]bool
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

func (tm *ThreadManager) Spawn(id, directive string, tools []string, initialMessages ...string) error {
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
	toolSet["send"] = true
	toolSet["done"] = true
	toolSet["pace"] = true
	toolSet["evolve"] = true

	// Build system prompt: core behavior + directive + core tool docs
	coreDocs := ""
	if tm.parent.registry != nil {
		coreDocs = "\n" + tm.parent.registry.CoreDocs(false)
	}
	threadSystemPrompt := baseThreadPrompt + coreDocs + "\n\n[DIRECTIVE]\n" + directive

	thread := &Thread{
		ID:        id,
		Directive: directive,
		Parent:    tm.parent,
		Tools:     toolSet,
		Started:   time.Now(),
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
		onStop:    func() { tm.cleanupThread(id) },
		handleTools: threadToolHandler(thread, tm),
		threadID:  id,
		apiLog:        tm.parent.apiLog,
		apiMu:         tm.parent.apiMu,
		apiNotify:     tm.parent.apiNotify,
		registry:      tm.parent.registry,
		toolAllowlist: toolSet,
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
	tm.parent.logAPI(APIEvent{Type: "thread_started", ThreadID: id})

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
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				if id != "" && msg != "" {
					tagged := fmt.Sprintf("[from:%s] %s", thread.ID, msg)
					if id == "main" {
						thread.Parent.Inject(tagged)
					} else {
						if !tm.Send(id, tagged) {
							t.Inject(fmt.Sprintf("[error] thread %q not found", id))
						}
					}
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
			case "evolve":
				if d := call.Args["directive"]; d != "" {
					thread.Directive = d
					coreDocs := ""
					if tm.parent.registry != nil {
						coreDocs = "\n" + tm.parent.registry.CoreDocs(false)
					}
					t.messages[0] = Message{Role: "system", Content: baseThreadPrompt + coreDocs + "\n\n[DIRECTIVE]\n" + d}
					// Persist
					tm.parent.config.SaveThread(PersistentThread{
						ID: thread.ID, Directive: d, Tools: toolSetToSlice(thread.Tools),
					})
					t.logAPI(APIEvent{Type: "evolved", ThreadID: thread.ID, Message: d})
				}
			case "remember":
				if text := call.Args["text"]; text != "" && t.memory != nil {
					go t.memory.Store(text)
				}
			case "web", "write_file", "read_file", "list_files":
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
			Directive: t.Directive,
			Tools:     toolSetToSlice(t.Tools),
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
	tm.parent.logAPI(APIEvent{Type: "thread_done", ThreadID: id})
}

func toolSetToSlice(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

