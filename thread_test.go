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
	}
	t.threads = NewThreadManager(t)
	return t
}

func TestThreadManager_SpawnAndList(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	err := thinker.threads.Spawn("test-thread", "Test prompt", []string{"reply", "web"}, true)
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
	if !threads[0].Thinking {
		t.Error("expected thinking=true")
	}
}

func TestThreadManager_SpawnDuplicate(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("dup", "test", nil, true)
	err := thinker.threads.Spawn("dup", "test2", nil, true)
	if err == nil {
		t.Error("expected error on duplicate spawn")
	}
}

func TestThreadManager_Kill(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()

	thinker.threads.Spawn("killme", "test", nil, true)
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

	thinker.threads.Spawn("sendto", "test", nil, true)

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

	thinker.threads.Spawn("marco", "Handle Marco", []string{"reply"}, true)

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

	thinker.threads.Spawn("minimal", "test", nil, true)

	threads := thinker.threads.List()
	if len(threads) != 1 {
		t.Fatal("expected 1 thread")
	}

	tools := threads[0].Tools
	hasReport := false
	hasDone := false
	hasPace := false
	for _, tool := range tools {
		switch tool {
		case "report":
			hasReport = true
		case "done":
			hasDone = true
		case "pace":
			hasPace = true
		}
	}
	if !hasReport || !hasDone || !hasPace {
		t.Errorf("expected report, done, pace in tools; got %v", tools)
	}
}

func TestBuildThreadToolDocs(t *testing.T) {
	tools := map[string]bool{"reply": true, "web": true, "report": true, "done": true, "pace": true}
	docs := buildThreadToolDocs(tools)

	if !strings.Contains(docs, "[[reply") {
		t.Error("expected reply in docs")
	}
	if !strings.Contains(docs, "[[web") {
		t.Error("expected web in docs")
	}
	if !strings.Contains(docs, "[[report") {
		t.Error("expected report in docs")
	}
	if !strings.Contains(docs, "[[done") {
		t.Error("expected done in docs")
	}
}

func TestBuildThreadToolDocs_NoReply(t *testing.T) {
	tools := map[string]bool{"web": true, "report": true, "done": true, "pace": true}
	docs := buildThreadToolDocs(tools)

	if strings.Contains(docs, "[[reply") {
		t.Error("should not include reply when not in tools")
	}
	if !strings.Contains(docs, "[[web") {
		t.Error("expected web in docs")
	}
}
