package main

import (
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// newTestThinkerFull creates a thinker with all fields initialized for testing Run()
func newTestThinkerFull() *Thinker {
	t := &Thinker{
		apiKey:    "test",
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
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
	}
	t.threads = NewThreadManager(t)
	return t
}

func TestEventFilter_RoutesToThreads(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	// Spawn a thread
	thinker.threads.Spawn("alice", "test", nil)
	defer thinker.threads.Kill("alice")

	// Set up the filter (same as NewThinker does)
	thinker.filterEvents = func(events []string) []string {
		var kept []string
		for _, ev := range events {
			if !thinker.threads.Route(ev) {
				kept = append(kept, ev)
			}
		}
		return kept
	}

	// Simulate events
	events := []string{
		"[user:alice] Hello",
		"[user:bob] Hi",
		"[console] do something",
	}

	filtered := thinker.filterEvents(events)

	// alice's message should be routed, bob's and console should remain
	if len(filtered) != 2 {
		t.Fatalf("expected 2 events after filtering, got %d: %v", len(filtered), filtered)
	}
	if filtered[0] != "[user:bob] Hi" {
		t.Errorf("expected bob's message, got %q", filtered[0])
	}
	if filtered[1] != "[console] do something" {
		t.Errorf("expected console message, got %q", filtered[1])
	}
}

func TestExternalEventDetection(t *testing.T) {
	tests := []struct {
		event      string
		isExternal bool
	}{
		{"[user:marco] Hello", true},
		{"[console] do something", true},
		{"[from:writer] report", true},
		{"[tool:list_files] (empty)", false},
		{"[tool:web] some content", false},
	}

	for _, tt := range tests {
		isExternal := !strings.HasPrefix(tt.event, "[tool:")
		if isExternal != tt.isExternal {
			t.Errorf("event %q: expected external=%v, got %v", tt.event, tt.isExternal, isExternal)
		}
	}
}

func TestSubThread_ReceivesEvents(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("worker", "test worker", []string{"web"})
	defer thinker.threads.Kill("worker")

	// Send to the thread
	ok := thinker.threads.Send("worker", "do work")
	if !ok {
		t.Fatal("expected send to succeed")
	}

	// Verify it's in the thread's inbox
	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	items := thread.Thinker.drainInbox()
	if len(items) != 1 || items[0] != "do work" {
		t.Errorf("expected 'do work' in thread inbox, got %v", items)
	}
}

func TestSubThread_InitialMessages(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("greeter", "test", nil, "[user] Hello", "[user] How are you?")
	defer thinker.threads.Kill("greeter")

	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["greeter"]
	thinker.threads.mu.RUnlock()

	items := thread.Thinker.drainInbox()
	if len(items) != 2 {
		t.Fatalf("expected 2 initial messages, got %d", len(items))
	}
	if items[0] != "[user] Hello" || items[1] != "[user] How are you?" {
		t.Errorf("unexpected messages: %v", items)
	}
}

func TestSubThread_HasBasePrompt(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("worker", "You monitor files", []string{"list_files"})
	defer thinker.threads.Kill("worker")

	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	sysPrompt := thread.Thinker.messages[0].Content

	// Should have base thread prompt
	if !strings.Contains(sysPrompt, "continuous thinking engine") {
		t.Error("missing base thread prompt")
	}
	// Should have the role
	if !strings.Contains(sysPrompt, "You monitor files") {
		t.Error("missing role prompt")
	}
	// Should have tool docs
	if !strings.Contains(sysPrompt, "[[list_files") {
		t.Error("missing list_files tool docs")
	}
	// Should have pacing instructions
	if !strings.Contains(sysPrompt, "PACING") {
		t.Error("missing pacing instructions")
	}
}

func TestSubThread_SharedMemory(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("worker", "test", nil)
	defer thinker.threads.Kill("worker")

	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	// Should share the same memory store
	if thread.Thinker.memory != thinker.memory {
		t.Error("sub-thread should share parent's memory store")
	}
}

func TestSubThread_SharedAPILog(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("worker", "test", nil)
	defer thinker.threads.Kill("worker")

	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	// Should share the same API log
	if thread.Thinker.apiLog != thinker.apiLog {
		t.Error("sub-thread should share parent's API log")
	}
	if thread.Thinker.apiMu != thinker.apiMu {
		t.Error("sub-thread should share parent's API mutex")
	}
}

func TestSubThread_APILogTagsThreadID(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("worker", "test", nil)
	defer thinker.threads.Kill("worker")

	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	thread.Thinker.logAPI(APIEvent{Type: "thought", Message: "test thought"})

	events, _ := thinker.APIEvents(0)
	// Find the thought event (skip thread_started)
	found := false
	for _, ev := range events {
		if ev.Type == "thought" && ev.ThreadID == "worker" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected thought event tagged with 'worker', got %v", events)
	}
}

func TestSubThread_ToolSet(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("writer", "test", []string{"write_file", "read_file"})
	defer thinker.threads.Kill("writer")

	threads := thinker.threads.List()
	if len(threads) != 1 {
		t.Fatalf("expected 1 thread, got %d", len(threads))
	}

	tools := make(map[string]bool)
	for _, tool := range threads[0].Tools {
		tools[tool] = true
	}

	// Requested tools
	if !tools["write_file"] || !tools["read_file"] {
		t.Error("missing requested tools")
	}
	// Built-in tools
	if !tools["send"] || !tools["done"] || !tools["pace"] {
		t.Error("missing built-in tools (send, done, pace)")
	}
}

func TestSubThread_KillRemovesFromList(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("temp", "test", nil)

	if thinker.threads.Count() != 1 {
		t.Fatal("expected 1 thread")
	}

	thinker.threads.Kill("temp")

	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 threads after kill, got %d", thinker.threads.Count())
	}
}

func TestSubThread_KillAll(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("a", "test", nil)
	thinker.threads.Spawn("b", "test", nil)
	thinker.threads.Spawn("c", "test", nil)

	if thinker.threads.Count() != 3 {
		t.Fatalf("expected 3 threads, got %d", thinker.threads.Count())
	}

	thinker.threads.KillAll()

	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 after KillAll, got %d", thinker.threads.Count())
	}
}

func TestConfig_PersistentThreads(t *testing.T) {
	cfg := &Config{path: "/dev/null"}

	cfg.SaveThread(PersistentThread{ID: "writer", Directive: "write stuff", Tools: []string{"write_file"}})
	cfg.SaveThread(PersistentThread{ID: "reader", Directive: "read stuff", Tools: []string{"read_file"}})

	threads := cfg.GetThreads()
	if len(threads) != 2 {
		t.Fatalf("expected 2 threads, got %d", len(threads))
	}

	// Update existing
	cfg.SaveThread(PersistentThread{ID: "writer", Directive: "updated", Tools: []string{"write_file", "web"}})
	threads = cfg.GetThreads()
	if len(threads) != 2 {
		t.Fatalf("expected still 2 threads after update, got %d", len(threads))
	}
	if threads[0].Directive != "updated" {
		t.Errorf("expected updated directive, got %q", threads[0].Directive)
	}

	// Remove
	cfg.RemoveThread("writer")
	threads = cfg.GetThreads()
	if len(threads) != 1 {
		t.Fatalf("expected 1 thread after remove, got %d", len(threads))
	}
	if threads[0].ID != "reader" {
		t.Errorf("expected reader, got %q", threads[0].ID)
	}

	// Clear all
	cfg.ClearThreads()
	if len(cfg.GetThreads()) != 0 {
		t.Error("expected 0 after clear")
	}
}

func TestFileTools(t *testing.T) {
	// Write
	result := writeFileTool(map[string]string{"path": "test_output.txt", "content": "hello world"})
	if strings.Contains(result, "error") {
		t.Fatalf("write_file error: %s", result)
	}
	t.Cleanup(func() {
		os.Remove("workspace/test_output.txt")
	})

	// Read
	result = readFileTool(map[string]string{"path": "test_output.txt"})
	if result != "hello world" {
		t.Errorf("expected 'hello world', got %q", result)
	}

	// List
	result = listFilesTool(map[string]string{"path": "."})
	if !strings.Contains(result, "test_output.txt") {
		t.Errorf("expected test_output.txt in listing, got %q", result)
	}

	// Nonexistent dir
	result = listFilesTool(map[string]string{"path": "nonexistent/"})
	if result != "(directory does not exist)" {
		t.Errorf("expected does not exist, got %q", result)
	}

	// Missing path
	result = readFileTool(map[string]string{})
	if result != "error: missing path" {
		t.Errorf("expected missing path error, got %q", result)
	}

	// Path escape
	result = writeFileTool(map[string]string{"path": "../../../etc/passwd", "content": "hack"})
	if !strings.Contains(result, "error") {
		t.Error("expected error for path escape")
	}
}

func TestAPIEvents_Ordering(t *testing.T) {
	thinker := newTestThinkerFull()

	thinker.logAPI(APIEvent{Type: "thought", Message: "first"})
	thinker.logAPI(APIEvent{Type: "thought", Message: "second"})
	thinker.logAPI(APIEvent{Type: "reply", Message: "third"})

	events, cursor := thinker.APIEvents(0)
	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}
	if events[0].Message != "first" || events[2].Message != "third" {
		t.Error("events not in order")
	}

	// Read from cursor
	thinker.logAPI(APIEvent{Type: "thought", Message: "fourth"})
	events, _ = thinker.APIEvents(cursor)
	if len(events) != 1 || events[0].Message != "fourth" {
		t.Errorf("expected 1 new event 'fourth', got %v", events)
	}
}

func TestAPIEvents_ThreadLifecycle(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("lifecycle-test", "test", nil)

	// Should have thread_started event
	events, _ := thinker.APIEvents(0)
	found := false
	for _, ev := range events {
		if ev.Type == "thread_started" && ev.ThreadID == "lifecycle-test" {
			found = true
		}
	}
	if !found {
		t.Error("expected thread_started event")
	}

	cursor := len(events)
	thinker.threads.Kill("lifecycle-test")

	// Wait for cleanup
	time.Sleep(200 * time.Millisecond)

	events, _ = thinker.APIEvents(cursor)
	found = false
	for _, ev := range events {
		if ev.Type == "thread_done" && ev.ThreadID == "lifecycle-test" {
			found = true
		}
	}
	if !found {
		t.Error("expected thread_done event")
	}
}
