package main

import (
	"strings"
	"testing"
	"time"
)

func newTestThinker() *Thinker {
	t := &Thinker{
		apiKey:    "test-key",
		messages:  []Message{{Role: "system", Content: "test"}},
		events:    make(chan ThinkEvent, 100),
		inbox:     make(chan string, 50),
		wakeup:    make(chan struct{}, 1),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    &MemoryStore{path: "/dev/null"},
		config:    &Config{Directive: "test"},
	}
	t.threads = NewThreadManager(t)
	return t
}

func TestThreadManager_SpawnAndList(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	err := thinker.threads.Spawn("test-thread", "Test prompt", []string{"reply", "web"})
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

	thinker.threads.Spawn("marco", "Handle Marco", []string{"reply"})

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

	// Without main-only
	docs = reg.CoreDocs(false)
	if strings.Contains(docs, "[[spawn") {
		t.Error("spawn should not be in non-main core docs")
	}
	if !strings.Contains(docs, "[[send") {
		t.Error("expected send in non-main core docs")
	}
}

func TestToolRegistry_Dispatch(t *testing.T) {
	reg := NewToolRegistry("test")

	// Known tool with handler
	result, ok := reg.Dispatch("list_files", map[string]string{"path": "."})
	if !ok {
		t.Error("expected list_files to dispatch")
	}
	if result == "" {
		t.Error("expected non-empty result")
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
