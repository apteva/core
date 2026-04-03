package main

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// baseThreadPrompt is a template — %s is replaced with the thread ID at spawn time.
const baseThreadPromptTemplate = `You are a SUB-THREAD (id="%s") in a continuous thinking engine. You were spawned by the main coordinator thread.

IDENTITY:
- Your ID is "%s". You are NOT the main thread — you are a worker thread with a specific task.
- You cannot spawn other threads. You cannot restructure the system.
- You report results back to main via [[send id="main" message="..."]].
- When your task is complete, call [[done message="..."]].

BEHAVIOR:
- Think out loud — explain what you're doing and why. Never output empty thoughts.
- Process events when they arrive. Use your tools to accomplish tasks.
- Stay focused on YOUR directive. Do not try to take over coordination duties.
- Keep each thought concise — 1-2 short paragraphs max.

PACING — this is critical:
- Tool results (like [[list_files]] or [[web]]) will wake you up for the next thought. Do NOT set [[pace]] in the same thought as a tool call — you'll be woken immediately.
- Instead: call tools first, THEN in the next thought (after seeing results), set your pace.
- Example flow: Thought 1: call [[list_files]]. Thought 2: process results, [[send]] report, [[pace sleep="5m"]].
- Set sleep duration based on need: "2s" when actively working, "5m" when monitoring, "1h" for deep idle.
- Only use [[pace]] when you have NO pending tool calls and are ready to wait.

TIMING:
- You do NOT have precise timing control. Pace rates are approximate, not exact.
- For delayed tasks (like "do X in 5 minutes"), use [[pace rate="sleep"]] and act on the next wake-up. Do not overthink exact timing — approximate is fine.
- Never spiral trying to calculate exact seconds. Just set a pace close to the delay, wake up, do the action, done.

IMPORTANT — tool calls and [[done]]:
- NEVER call [[done]] in the same thought as a tool call. Tool results arrive in your NEXT thought.
- Always wait for tool results before calling [[done]] — you need to confirm the action succeeded.
- Example: Thought 1: [[pushover_send_notification ...]]. Thought 2: see result, confirm success, [[done]].`

type ThreadInfo struct {
	ID           string
	Directive    string
	Tools        []string
	Running      bool
	Iteration    int
	Rate         ThinkRate
	Model        ModelTier
	Started      time.Time
	ContextMsgs  int
	ContextChars int
}

type Thread struct {
	ID           string
	Directive    string // original directive before tool docs
	Thinker      *Thinker
	Parent       *Thinker
	Tools        map[string]bool
	Started      time.Time
	initialParts []ContentPart // media to inject before first Run()
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

// SpawnWithMedia creates a thread and injects media parts before it starts thinking.
func (tm *ThreadManager) SpawnWithMedia(id, directive string, tools []string, parts []ContentPart, initialMessages ...string) error {
	return tm.spawnInternal(id, directive, tools, parts, initialMessages...)
}

func (tm *ThreadManager) Spawn(id, directive string, tools []string, initialMessages ...string) error {
	return tm.spawnInternal(id, directive, tools, nil, initialMessages...)
}

func (tm *ThreadManager) spawnInternal(id, directive string, tools []string, mediaParts []ContentPart, initialMessages ...string) error {
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
	basePrompt := fmt.Sprintf(baseThreadPromptTemplate, id, id)
	coreDocs := ""
	if tm.parent.registry != nil {
		coreDocs = "\n" + tm.parent.registry.CoreDocs(false)
	}
	threadSystemPrompt := basePrompt + coreDocs + "\n\n[DIRECTIVE]\n" + directive

	thread := &Thread{
		ID:           id,
		Directive:    directive,
		Parent:       tm.parent,
		Tools:        toolSet,
		Started:      time.Now(),
		initialParts: mediaParts,
	}

	// Create a Thinker — same struct as main, shares the bus
	thinker := &Thinker{
		apiKey:   tm.parent.apiKey,
		provider: tm.parent.provider,
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
		memory:    tm.parent.memory,
		onStop:    func() { tm.cleanupThread(id) },
		handleTools: threadToolHandler(thread, tm),
		threadID:  id,
		apiLog:        tm.parent.apiLog,
		apiMu:         tm.parent.apiMu,
		apiNotify:     tm.parent.apiNotify,
		registry:      tm.parent.registry,
		toolAllowlist: toolSet,
		approvalCh:    tm.parent.approvalCh,
		config:        tm.parent.config,
		rebuildPrompt: func(toolDocs string) string {
			cd := ""
			if tm.parent.registry != nil {
				cd = "\n" + tm.parent.registry.CoreDocs(false)
			}
			prompt := fmt.Sprintf(baseThreadPromptTemplate, id, id) + cd
			if toolDocs != "" {
				prompt += "\n" + toolDocs
			}
			prompt += "\n\n[DIRECTIVE]\n" + thread.Directive
			return prompt
		},
	}
	thread.Thinker = thinker
	thinker.telemetry = tm.parent.telemetry // share telemetry

	tm.threads[id] = thread

	// Inject initial messages before starting so first thought picks them up
	for _, msg := range initialMessages {
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

	// Same Run() as the main thinker — no duplicated loop
	go thinker.Run()

	tm.parent.bus.Publish(Event{Type: EventThreadStart, From: id, Text: fmt.Sprintf("Thread %q spawned", id)})
	toolList := toolSetToSlice(thread.Tools)
	tm.parent.Inject(fmt.Sprintf("[thread:%s] started (tools: %s)", id, strings.Join(toolList, ", ")))
	tm.parent.logAPI(APIEvent{Type: "thread_started", ThreadID: id})

	// Telemetry: thread.spawn
	if tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.spawn", id, ThreadSpawnData{
			ParentID:  "main",
			Directive: directive,
			Tools:     tools,
		})
	}

	return nil
}

// threadToolHandler returns a ToolHandler scoped to a thread's allowed tools.
func threadToolHandler(thread *Thread, tm *ThreadManager) ToolHandler {
	return func(t *Thinker, calls []toolCall, _ []string) ([]string, []string) {
		var replies []string
		var toolNames []string
		var doneMsg *string // defer done until all tools processed

		for _, call := range calls {
			if !thread.Tools[call.Name] {
				continue
			}
			// Supervised mode gate — applies to all tools
			if !waitForApproval(t, call) {
				t.Inject(fmt.Sprintf("[rejected] %s was rejected by the user. Do NOT retry the same call. Ask the user what they want instead, or try a different approach.", call.Name))
				continue
			}
			switch call.Name {
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				if id != "" && msg != "" {
					tagged := fmt.Sprintf("[from:%s] %s", thread.ID, msg)
					logMsg("THREAD", fmt.Sprintf("%s send to=%s msg=%q", thread.ID, id, msg))
					if id == "main" {
						thread.Parent.Inject(tagged)
					} else {
						if !tm.Send(id, tagged) {
							t.Inject(fmt.Sprintf("[error] thread %q not found", id))
						}
					}
					// Telemetry: thread.message
					if t.telemetry != nil {
						t.telemetry.Emit("thread.message", thread.ID, ThreadMessageData{
							From: thread.ID, To: id, Message: msg,
						})
					}
				}
			case "done":
				msg := call.Args["message"]
				doneMsg = &msg // defer — process after all other tools
			case "pace":
				if s := call.Args["sleep"]; s != "" {
					if d, ok := parseSleepDuration(s); ok {
						t.agentSleep = d
						t.agentRate = RateSleep
					}
				} else if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
					if d, ok2 := rateAliases[call.Args["rate"]]; ok2 {
						t.agentSleep = d
					}
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
				}
			case "evolve":
				if d := call.Args["directive"]; d != "" {
					thread.Directive = d
					if t.rebuildPrompt != nil {
						t.messages[0] = Message{Role: "system", Content: t.rebuildPrompt("")}
					}
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
			default:
				// Dispatch to registry (MCP tools, file tools, web, etc)
				executeTool(t, call)
				toolNames = append(toolNames, call.Raw)
			}
		}

		// Process done LAST — after all other tools have been dispatched
		// Stop() triggers cleanupThread which publishes EventThreadDone + logAPI
		if doneMsg != nil {
			logMsg("THREAD", fmt.Sprintf("%s calling done, msg=%q", thread.ID, *doneMsg))
			if *doneMsg != "" {
				logMsg("THREAD", fmt.Sprintf("%s injecting done message to parent (main)", thread.ID))
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done] %s", thread.ID, *doneMsg))
			} else {
				logMsg("THREAD", fmt.Sprintf("%s injecting done (no message) to parent", thread.ID))
				thread.Parent.Inject(fmt.Sprintf("[thread:%s done]", thread.ID))
			}
			logMsg("THREAD", fmt.Sprintf("%s calling Stop()", thread.ID))
			t.Stop()
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
		infos = append(infos, ThreadInfo{
			ID:        t.ID,
			Directive: t.Directive,
			Tools:     toolSetToSlice(t.Tools),
			Running:   true,
			Iteration: t.Thinker.iteration,
			Rate:         t.Thinker.rate,
			Model:        t.Thinker.model,
			Started:      t.Started,
			ContextMsgs:  len(t.Thinker.messages),
			ContextChars: func() int { n := 0; for _, m := range t.Thinker.messages { n += len(m.Content) }; return n }(),
		})
	}
	return infos
}

func (tm *ThreadManager) Count() int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return len(tm.threads)
}

// Update changes a thread's directive and/or tools. Rebuilds the system prompt immediately.
func (tm *ThreadManager) Update(id, directive string, tools []string) error {
	tm.mu.RLock()
	thread, exists := tm.threads[id]
	tm.mu.RUnlock()
	if !exists {
		return fmt.Errorf("thread %q not found", id)
	}

	if directive != "" {
		thread.Directive = directive
	}
	if len(tools) > 0 {
		toolSet := make(map[string]bool)
		for _, t := range tools {
			toolSet[strings.TrimSpace(t)] = true
		}
		// Always include builtins
		for _, b := range []string{"send", "done", "pace", "evolve", "remember"} {
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
		ID: id, Directive: thread.Directive, Tools: toolSetToSlice(thread.Tools),
	})

	return nil
}

// FindPendingApproval checks all child threads for a pending tool call.
func (tm *ThreadManager) FindPendingApproval() *toolCall {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for _, thread := range tm.threads {
		thread.Thinker.pendingMu.Lock()
		if thread.Thinker.pendingTool != nil {
			tc := thread.Thinker.pendingTool
			thread.Thinker.pendingMu.Unlock()
			return tc
		}
		thread.Thinker.pendingMu.Unlock()
	}
	return nil
}

// FindPendingThinker returns the child thinker that has a pending tool call, or nil.
func (tm *ThreadManager) FindPendingThinker() *Thinker {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for _, thread := range tm.threads {
		thread.Thinker.pendingMu.Lock()
		if thread.Thinker.pendingTool != nil {
			thread.Thinker.pendingMu.Unlock()
			return thread.Thinker
		}
		thread.Thinker.pendingMu.Unlock()
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
	delete(tm.threads, id)
	tm.mu.Unlock()
	tm.parent.config.RemoveThread(id)
	logMsg("THREAD", fmt.Sprintf("%s publishing EventThreadDone from cleanup", id))
	tm.parent.bus.Publish(Event{Type: EventThreadDone, From: id})
	logMsg("THREAD", fmt.Sprintf("%s unsubscribing from bus", id))
	tm.parent.bus.Unsubscribe(id)
	tm.parent.logAPI(APIEvent{Type: "thread_done", ThreadID: id})

	// Telemetry: thread.done
	if tm.parent.telemetry != nil {
		tm.parent.telemetry.Emit("thread.done", id, ThreadDoneData{
			ParentID: "main",
		})
	}
}

func toolSetToSlice(m map[string]bool) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
