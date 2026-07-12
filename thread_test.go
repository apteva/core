package core

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestThinker() *Thinker {
	bus := NewEventBus()
	t := &Thinker{
		apiKey:      "test-key",
		provider:    NewFireworksProvider("test-key"),
		messages:    []Message{{Role: "system", Content: "test"}},
		bus:         bus,
		sub:         bus.Subscribe("main", 100),
		pause:       make(chan bool),
		quit:        make(chan struct{}),
		rate:        RateSlow,
		agentRate:   RateSlow,
		memory:      &MemoryStore{path: "/dev/null"},
		config:      &Config{Directive: "test"},
		threadID:    "main",
		telemetry:   &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
		toolIndex:   NewToolIndex(),
		activeTools: map[string]bool{},
	}
	t.threads = NewThreadManager(t)
	return t
}

func TestThreadManager_SpawnAndList(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	err := thinker.threads.Spawn("test-thread", "Test prompt", []string{"web"})
	if err != nil {
		t.Fatalf("Spawn error: %v", err)
	}

	if thinker.threads.Count() != 1 {
		t.Errorf("expected 1 thread, got %d", thinker.threads.Count())
	}

	threads := thinker.threads.List()
	if len(threads) != 1 {
		t.Fatalf("expected 1 thread in list, got %d", len(threads))
	}
	if threads[0].ID != "test-thread" {
		t.Errorf("expected id 'test-thread', got %q", threads[0].ID)
	}
	if !threads[0].Running {
		t.Error("expected running=true")
	}
}

func TestThreadManager_SpawnDuplicate(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("dup", "test", nil)
	err := thinker.threads.Spawn("dup", "test2", nil)
	if err == nil {
		t.Error("expected error on duplicate spawn")
	}
}

func TestThreadManager_SpawnWithModelAndReasoning(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	err := thinker.threads.SpawnWithOpts("profiled", "test", nil, SpawnOpts{
		Model:     "small",
		Reasoning: ReasoningHigh,
		DeferRun:  true,
	})
	if err != nil {
		t.Fatalf("SpawnWithOpts error: %v", err)
	}

	threads := thinker.threads.List()
	if len(threads) != 1 {
		t.Fatalf("expected 1 thread, got %d", len(threads))
	}
	if threads[0].Model != ModelSmall {
		t.Fatalf("model = %s, want small", threads[0].Model)
	}
	if threads[0].Reasoning != ReasoningHigh {
		t.Fatalf("reasoning = %s, want high", threads[0].Reasoning)
	}
}

func TestThreadManager_Kill(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("killme", "test", nil)
	if thinker.threads.Count() != 1 {
		t.Fatal("expected 1 thread")
	}

	thinker.threads.Kill("killme")
	// Give goroutine time to clean up
	time.Sleep(100 * time.Millisecond)

	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 threads after kill, got %d", thinker.threads.Count())
	}
}

func TestThreadManager_Send(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("sendto", "test", nil)

	ok := thinker.threads.Send("sendto", "hello from parent")
	if !ok {
		t.Error("expected Send to succeed")
	}

	ok = thinker.threads.Send("nonexistent", "should fail")
	if ok {
		t.Error("expected Send to fail for nonexistent thread")
	}
}

func TestThreadManager_SystemThreadsAreOperationalButNotAgentVisible(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	if err := thinker.threads.SpawnWithOpts("worker", "Handle operator work", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	workerEvents := thinker.drainEventTexts()
	if len(workerEvents) != 1 || !strings.Contains(workerEvents[0], "[thread:worker] started") {
		t.Fatalf("agent-visible worker startup = %v", workerEvents)
	}
	if err := thinker.threads.SpawnWithOpts("unconscious", "Consolidate memories", nil, SpawnOpts{System: true, DeferRun: true}); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}
	if systemEvents := thinker.drainEventTexts(); len(systemEvents) != 0 {
		t.Fatalf("system thread leaked into parent inbox: %v", systemEvents)
	}

	all := thinker.threads.List()
	if len(all) != 2 {
		t.Fatalf("List() returned %d threads, want 2", len(all))
	}
	if !all[0].System || all[0].ID != "unconscious" {
		t.Fatalf("system metadata missing from complete list: %+v", all)
	}
	visible := thinker.threads.ListAgentVisible()
	if len(visible) != 1 || visible[0].ID != "worker" {
		t.Fatalf("ListAgentVisible() = %+v, want only worker", visible)
	}

	if err := thinker.threads.SendAgentWithParts("unconscious", "store this", nil); err == nil || !strings.Contains(err.Error(), "platform-managed") {
		t.Fatalf("agent send error = %v, want platform-managed rejection", err)
	}

	thinker.threads.mu.RLock()
	unconscious := thinker.threads.threads["unconscious"]
	thinker.threads.mu.RUnlock()
	if got := unconscious.Thinker.drainEventTexts(); len(got) != 0 {
		t.Fatalf("rejected agent send reached system thread: %v", got)
	}

	if !thinker.threads.Send("unconscious", "runtime wake") {
		t.Fatal("privileged runtime send should reach system thread")
	}
	got := drainEventTextsWait(t, unconscious.Thinker, 1, time.Second)
	if len(got) != 1 || got[0] != "runtime wake" {
		t.Fatalf("privileged runtime delivery = %v", got)
	}
}

func TestThreadManager_SystemFlagSurvivesPrivilegedUpdateAndRename(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")

	if err := thinker.threads.SpawnWithOpts("unconscious", "old", nil, SpawnOpts{System: true, DeferRun: true}); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}
	if err := thinker.config.SaveThread(PersistentThread{ID: "unconscious", System: true, Directive: "old"}); err != nil {
		t.Fatalf("persist unconscious: %v", err)
	}
	if err := thinker.threads.Update("unconscious", "Memory", "new", nil); err != nil {
		t.Fatalf("privileged update: %v", err)
	}
	if threads := thinker.config.GetThreads(); len(threads) != 1 || !threads[0].System {
		t.Fatalf("system flag lost after update: %+v", threads)
	}
	if err := thinker.threads.Rename("unconscious", "memory-system"); err != nil {
		t.Fatalf("privileged rename: %v", err)
	}
	if threads := thinker.config.GetThreads(); len(threads) != 1 || !threads[0].System || threads[0].ID != "memory-system" {
		t.Fatalf("system flag lost after rename: %+v", threads)
	}
}

func TestThreadManager_ToolSetAlwaysIncludesBuiltins(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("minimal", "test", nil)

	threads := thinker.threads.List()
	if len(threads) != 1 {
		t.Fatal("expected 1 thread")
	}

	tools := threads[0].Tools
	hasSend := false
	hasDone := false
	hasPace := false
	for _, tool := range tools {
		switch tool {
		case "send":
			hasSend = true
		case "done":
			hasDone = true
		case "pace":
			hasPace = true
		}
	}
	if !hasSend || !hasDone || !hasPace {
		t.Errorf("expected report, done, pace in tools; got %v", tools)
	}
}

func TestToolRegistry_CoreDocs(t *testing.T) {
	reg := NewToolRegistry("test")
	docs := reg.CoreDocs(true)

	// CoreDocs lists each tool as `  <name> — <description>` (native tool
	// calling — no `[[...]]` syntax). Check for the formatted prefix so we
	// don't match tool names that appear inside other descriptions.
	if !strings.Contains(docs, "  spawn —") {
		t.Error("expected spawn in main core docs")
	}
	if !strings.Contains(docs, "  send —") {
		t.Error("expected send in core docs")
	}
	if !strings.Contains(docs, "  pace —") {
		t.Error("expected pace in core docs")
	}
	if !strings.Contains(docs, "  connect —") {
		t.Error("expected connect in main core docs")
	}
	if !strings.Contains(docs, "  disconnect —") {
		t.Error("expected disconnect in main core docs")
	}
	if !strings.Contains(docs, "  list_connected —") {
		t.Error("expected list_connected in main core docs")
	}

	// Without main-only
	docs = reg.CoreDocs(false)
	if strings.Contains(docs, "  spawn —") {
		t.Error("spawn should not be in non-main core docs")
	}
	if strings.Contains(docs, "  connect —") {
		t.Error("connect should not be in non-main core docs")
	}
	if !strings.Contains(docs, "  send —") {
		t.Error("expected send in non-main core docs")
	}
}

func TestThread_DoneInjectsToMain(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	// Simulate a thread done message — inject directly to main
	thinker.Inject("[thread:worker done] task complete")

	// Main should receive it
	select {
	case ev := <-thinker.sub.C:
		if ev.Type != EventInbox {
			t.Errorf("expected EventInbox, got %s", ev.Type)
		}
		if !strings.Contains(ev.Text, "[thread:worker done]") {
			t.Errorf("expected done message, got %q", ev.Text)
		}
	case <-time.After(1 * time.Second):
		t.Error("main did not receive thread done event within 1s")
	}
}

func TestThreadDone_WakesMainSleep(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	// Drain any existing events
	for {
		select {
		case <-thinker.sub.C:
		case <-thinker.sub.Wake:
		default:
			goto ready
		}
	}
ready:

	// Start sleeping on main's wake channel
	woke := make(chan string, 1)
	go func() {
		select {
		case <-thinker.sub.Wake:
			woke <- "wake"
		case <-time.After(2 * time.Second):
			woke <- "timeout"
		}
	}()

	// Inject to main (simulating thread done)
	time.Sleep(50 * time.Millisecond)
	thinker.Inject("[thread:worker done] finished")

	result := <-woke
	if result != "wake" {
		t.Errorf("expected main to wake on inject, got %s", result)
	}
}

func TestThreadKill_Cleanup(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("killtest", "test", nil)
	// Thread's Run() will crash on API call — that's fine, we just test kill
	time.Sleep(100 * time.Millisecond)

	thinker.threads.Kill("killtest")
	time.Sleep(200 * time.Millisecond)

	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 threads after kill, got %d", thinker.threads.Count())
	}
}

func TestToolRegistry_Dispatch(t *testing.T) {
	reg := NewToolRegistry("test")

	// Stub a tool with a handler to verify dispatch works
	reg.Register(&ToolDef{
		Name:    "stub",
		Handler: func(args map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
	})
	_, ok := reg.Dispatch("stub", nil)
	if !ok {
		t.Error("expected stub to dispatch")
	}

	// Core tool (no handler)
	_, ok = reg.Dispatch("pace", nil)
	if ok {
		t.Error("pace should not dispatch (no handler)")
	}

	// Unknown tool
	_, ok = reg.Dispatch("nonexistent", nil)
	if ok {
		t.Error("nonexistent should not dispatch")
	}
}

func TestThreadManagerRejectsUnsafeID(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("../../outside", "test", nil, SpawnOpts{DeferRun: true}); err == nil {
		t.Fatal("unsafe thread id was accepted")
	}
	if thinker.bus.HasSubscriber("../../outside") {
		t.Fatal("unsafe thread id reserved an event subscription")
	}
}

func TestThreadManagerRejectsDuplicateIDAcrossBranches(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	for _, id := range []string{"leader-a", "leader-b"} {
		if err := thinker.threads.SpawnWithOpts(id, "lead", []string{"spawn"}, SpawnOpts{DeferRun: true}); err != nil {
			t.Fatalf("spawn %s: %v", id, err)
		}
	}
	a := thinker.threads.threads["leader-a"]
	b := thinker.threads.threads["leader-b"]
	if a.Children == nil || b.Children == nil {
		t.Fatal("leaders did not receive child managers")
	}
	if err := a.Children.SpawnWithOpts("shared-worker", "first", nil, SpawnOpts{Depth: 1, DeferRun: true}); err != nil {
		t.Fatalf("first branch spawn: %v", err)
	}
	if err := b.Children.SpawnWithOpts("shared-worker", "second", nil, SpawnOpts{Depth: 1, DeferRun: true}); err == nil {
		t.Fatal("duplicate id in a separate branch was accepted")
	}
}

func TestThreadManagerRenameMovesIdentityRoutingAndCleanup(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("worker-old", "test", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	thread := thinker.threads.threads["worker-old"]
	if err := thinker.threads.Rename("worker-old", "worker-new"); err != nil {
		t.Fatalf("rename: %v", err)
	}
	if thread.ID != "worker-new" || thread.Thinker.threadID != "worker-new" {
		t.Fatalf("identity not moved: thread=%q thinker=%q", thread.ID, thread.Thinker.threadID)
	}
	if thinker.bus.HasSubscriber("worker-old") || !thinker.bus.HasSubscriber("worker-new") {
		t.Fatal("event route not moved")
	}
	if !strings.Contains(thread.Thinker.messages[0].Content, `id="worker-new"`) {
		t.Fatalf("prompt kept old identity: %q", thread.Thinker.messages[0].Content)
	}
	if got := filepath.Base(thread.Thinker.session.path); got != "worker-new.jsonl" {
		t.Fatalf("session filename = %q", got)
	}
	if !thinker.threads.Send("worker-new", "hello after rename") {
		t.Fatal("send to renamed thread failed")
	}
	select {
	case ev := <-thread.Thinker.sub.C:
		if ev.Text != "hello after rename" {
			t.Fatalf("delivered text = %q", ev.Text)
		}
	case <-time.After(time.Second):
		t.Fatal("renamed thread did not receive message")
	}
	thinker.threads.cleanupThread("worker-new")
	if thinker.threads.Count() != 0 || thinker.bus.HasSubscriber("worker-new") {
		t.Fatal("cleanup left renamed thread or subscription behind")
	}
}
