package main

import (
	"strings"
	"testing"
	"time"
)

func newTestThinker() *Thinker {
	bus := NewEventBus()
	t := &Thinker{
		apiKey:    "test-key",
		provider:  NewFireworksProvider("test-key"),
		messages:  []Message{{Role: "system", Content: "test"}},
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    &MemoryStore{path: "/dev/null"},
		config:    &Config{Directive: "test"},
		threadID:  "main",
		telemetry: &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
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

func TestThreadManager_Route(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("marco", "Handle Marco", []string{"web"})

	// Should route to thread
	routed := thinker.threads.Route("[user:marco] Hello there")
	if !routed {
		t.Error("expected event to be routed to thread 'marco'")
	}

	// Should NOT route (no matching thread)
	routed = thinker.threads.Route("[user:alice] Hi")
	if routed {
		t.Error("expected event NOT to be routed (no thread for alice)")
	}

	// Should NOT route (not a user event)
	routed = thinker.threads.Route("[tool:web] some result")
	if routed {
		t.Error("expected non-user event NOT to be routed")
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

	if !strings.Contains(docs, "[[spawn") {
		t.Error("expected spawn in main core docs")
	}
	if !strings.Contains(docs, "[[send") {
		t.Error("expected send in core docs")
	}
	if !strings.Contains(docs, "[[pace") {
		t.Error("expected pace in core docs")
	}
	if !strings.Contains(docs, "[[connect") {
		t.Error("expected connect in main core docs")
	}
	if !strings.Contains(docs, "[[disconnect") {
		t.Error("expected disconnect in main core docs")
	}
	if !strings.Contains(docs, "[[list_connected") {
		t.Error("expected list_connected in main core docs")
	}

	// Without main-only
	docs = reg.CoreDocs(false)
	if strings.Contains(docs, "[[spawn") {
		t.Error("spawn should not be in non-main core docs")
	}
	if strings.Contains(docs, "[[connect") {
		t.Error("connect should not be in non-main core docs")
	}
	if !strings.Contains(docs, "[[send") {
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

	// Known tool with handler — web needs a URL but we just check dispatch works
	_, ok := reg.Dispatch("web", map[string]string{"url": ""})
	if !ok {
		t.Error("expected web to dispatch")
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
