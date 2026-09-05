package core

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestPerfAuditLostWake(t *testing.T) {
	for attempt := 0; attempt < 5; attempt++ {
		th := retryTestThinker(nil)
		th.sub = th.bus.Subscribe("main", 4096)
		ids := make([]string, 1024)
		for i := 0; i < 4096; i++ {
			th.bus.Publish(Event{Type: EventInbox, To: "main", Text: "initial", ExecutionIDs: ids})
		}
		done := make(chan []drainedEvent, 1)
		go func() { done <- th.drainEvents() }()
		for len(th.sub.C) > 0 {
			runtime.Gosched()
		}
		th.sub.queueMu.Lock()
		th.sub.queueMu.Unlock()
		th.bus.Publish(Event{Type: EventInbox, To: "main", Text: "arrived during drain"})
		got := <-done
		found := false
		for _, e := range got {
			if e.Text == "arrived during drain" {
				found = true
			}
		}
		if !found && len(th.sub.C) > 0 && len(th.sub.Wake) == 0 {
			t.Fatal("new message remains queued but drainEvents consumed its wake; an event-only thinker may sleep indefinitely")
		}
	}
}

func TestPerfAuditBarrierStaleWake(t *testing.T) {
	th := retryTestThinker(nil)
	th.pendingTools.Store("slow", "slow")
	go func() {
		time.Sleep(5 * time.Millisecond)
		th.bus.Publish(Event{Type: EventInbox, To: "main", ToolResult: &ToolResult{CallID: "slow", Content: "done"}})
		th.pendingTools.Delete("slow")
	}()
	var results []ToolResult
	var texts []string
	var parts []ContentPart
	th.waitForPendingTools(&results, &texts, &parts, time.Second)
	if len(results) != 1 {
		t.Fatalf("results=%d", len(results))
	}
	if len(th.sub.C) == 0 && len(th.sub.Wake) > 0 {
		t.Fatal("barrier consumed the result but left its wake; the next event wait can trigger a redundant LLM turn")
	}
}

func TestPerfAuditToolSlotsBlockSend(t *testing.T) {
	th := newTestThinkerFull()
	th.maxConcurrentTools = 1
	th.registry = NewToolRegistry("")
	release := make(chan struct{})
	started := make(chan struct{}, 2)
	th.registry.Register(&ToolDef{Name: "slow", Handler: func(map[string]string) ToolResponse {
		started <- struct{}{}
		<-release
		return ToolResponse{Text: "ok"}
	}})
	worker := retryTestThinker(nil)
	worker.threadID = "worker"
	th.threads.threads["worker"] = &Thread{ID: "worker", Thinker: worker}
	inbox := th.bus.Subscribe("worker", 10)
	done := make(chan struct{})
	go func() {
		defer close(done)
		mainToolHandler(th)(th, []toolCall{
			{Name: "slow", NativeID: "one", Args: map[string]string{}},
			{Name: "slow", NativeID: "two", Args: map[string]string{}},
			{Name: "send", NativeID: "three", Args: map[string]string{"id": "worker", "message": "coordinate now"}},
		}, nil)
	}()
	<-started
	blocked := false
	select {
	case <-inbox.C:
	case <-time.After(100 * time.Millisecond):
		blocked = true
	}
	close(release)
	<-done
	deadline := time.Now().Add(time.Second)
	for th.asyncToolsActive.Load() > 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if blocked {
		t.Fatal("coordination send behind a queued tool was blocked by the per-thread tool semaphore")
	}
}

type auditSlowRealtimeProvider struct {
	fakeRealtimeProvider
	entered chan struct{}
	release chan struct{}
}

func (p *auditSlowRealtimeProvider) Open(context.Context, RealtimeSessionOpts) (RealtimeSession, error) {
	close(p.entered)
	<-p.release
	return nil, errors.New("mock connection failed")
}

func TestPerfAuditRealtimeSpawnBlocksSiblingSend(t *testing.T) {
	t.Chdir(t.TempDir())
	th := newTestThinkerFull()
	th.provider = &scriptedRetryProvider{name: "mock"}
	p := &auditSlowRealtimeProvider{entered: make(chan struct{}), release: make(chan struct{})}
	th.pool = &ProviderPool{providers: map[string]LLMProvider{"mock": th.provider}, order: []string{"mock"}, default_: "mock", realtimeProviders: map[string]RealtimeProvider{"rt": p}, realtimeDefault: "rt"}
	worker := retryTestThinker(nil)
	worker.threadID = "worker"
	th.threads.threads["worker"] = &Thread{ID: "worker", Thinker: worker}
	th.bus.Subscribe("worker", 10)
	spawnDone := make(chan error, 1)
	go func() {
		spawnDone <- th.threads.SpawnWithOpts("voice", "test", nil, SpawnOpts{Realtime: true, ProviderName: "rt", Ephemeral: true})
	}()
	select {
	case <-p.entered:
	case <-time.After(time.Second):
		t.Fatal("mock connection never reached")
	}
	sendDone := make(chan bool, 1)
	go func() { sendDone <- th.threads.Send("worker", "hello") }()
	blocked := false
	select {
	case <-sendDone:
	case <-time.After(100 * time.Millisecond):
		blocked = true
	}
	close(p.release)
	<-spawnDone
	if blocked {
		<-sendDone
		t.Fatal("an unrelated sibling send waited for realtime network startup under ThreadManager.mu")
	}
}

func TestPerfAuditParallelToolsActuallyOverlap(t *testing.T) {
	th := newTestThinkerFull()
	th.registry = NewToolRegistry("")
	th.maxConcurrentTools = 16
	release := make(chan struct{})
	started := make(chan struct{}, 16)
	th.registry.Register(&ToolDef{Name: "parallel", Handler: func(map[string]string) ToolResponse {
		started <- struct{}{}
		<-release
		return ToolResponse{Text: "ok"}
	}})
	for i := 0; i < 16; i++ {
		executeTool(th, toolCall{Name: "parallel", NativeID: fmt.Sprint(i), Args: map[string]string{}})
	}
	n := 0
	deadline := time.After(time.Second)
loop:
	for n < 16 {
		select {
		case <-started:
			n++
		case <-deadline:
			break loop
		}
	}
	close(release)
	end := time.Now().Add(time.Second)
	for th.asyncToolsActive.Load() > 0 && time.Now().Before(end) {
		time.Sleep(time.Millisecond)
	}
	if n != 16 {
		t.Fatalf("only %d/16 independent tool handlers overlapped", n)
	}
}

func BenchmarkPerfAuditThreadSend(b *testing.B) {
	for _, n := range []int{1, 100, 1000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			th := newTestThinkerFull()
			var target *Subscription
			for i := 0; i < n; i++ {
				id := fmt.Sprint(i)
				th.threads.threads[id] = &Thread{ID: id}
				sub := th.bus.Subscribe(id, 1)
				if i == 0 {
					target = sub
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !th.threads.Send("0", "work") {
					b.Fatal("send")
				}
				<-target.C
			}
		})
	}
}

func BenchmarkPerfAuditTrackedSend(b *testing.B) {
	for _, n := range []int{1, 100, 1000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			cfg := &Config{path: filepath.Join(b.TempDir(), "config.json")}
			for i := 0; i < n; i++ {
				cfg.Threads = append(cfg.Threads, PersistentThread{ID: fmt.Sprint(i), Directive: strings.Repeat("d", 1024)})
			}
			cfg.EventExecutions = []PersistentEventExecution{{ExecutionID: "tracked", Status: eventActive, Participants: map[string]bool{"main": true, "worker": true}}}
			th := newTestThinkerFull()
			th.config = cfg
			th.eventLifecycle = NewEventLifecycle(cfg, nil)
			th.threads.threads["worker"] = &Thread{ID: "worker"}
			sub := th.bus.Subscribe("worker", 1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !th.threads.SendWithPartsExecution("worker", "work", nil, []string{"tracked"}) {
					b.Fatal("send")
				}
				<-sub.C
			}
		})
	}
}

func BenchmarkPerfAuditRoster(b *testing.B) {
	for _, n := range []int{10, 100, 1000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			th := newTestThinkerFull()
			for i := 0; i < n; i++ {
				id := fmt.Sprint(i)
				th.threads.threads[id] = &Thread{ID: id, Thinker: retryTestThinker(nil), Tools: map[string]bool{"send": true, "done": true}}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				th.threads.List()
			}
		})
	}
}

func BenchmarkPerfAuditContextStatus(b *testing.B) {
	for _, n := range []int{20, 100, 300} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			th := newTestThinkerFull()
			for i := 0; i < n; i++ {
				th.messages = append(th.messages, Message{Role: "user", Content: strings.Repeat("context ", 512)})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				th.publishContextStatus()
			}
		})
	}
}

func TestPerfAuditConcurrentFanInNoLoss(t *testing.T) {
	bus := NewEventBus()
	sub := bus.Subscribe("main", 16)
	const workers = 32
	const messages = 500
	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for m := 0; m < messages; m++ {
				bus.Publish(Event{Type: EventInbox, To: "main", From: fmt.Sprint(w), ID: fmt.Sprintf("%d/%d", w, m)})
			}
		}(w)
	}
	wg.Wait()
	events := sub.DrainTargeted()
	seen := map[string]bool{}
	for _, e := range events {
		if seen[e.ID] {
			t.Fatalf("duplicate %s", e.ID)
		}
		seen[e.ID] = true
	}
	if len(seen) != workers*messages {
		t.Fatalf("received=%d", len(seen))
	}
	t.Logf("32 concurrent publishers, 16000 unique targeted events, publish+drain+verify=%s", time.Since(start))
}

type auditBatchProvider struct {
	scriptedRetryProvider
	count atomic.Int32
}

func (p *auditBatchProvider) Chat(context.Context, []Message, string, []NativeTool, func(string), func(string), func(string, string, string)) (ChatResponse, error) {
	if p.count.Add(1) == 1 {
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "fast", Name: "fast", Args: map[string]string{}}, {ID: "slow", Name: "slow", Args: map[string]string{}}}}, nil
	}
	return ChatResponse{Text: "All tool results handled."}, nil
}
func TestPerfAuditBarrierCausesExtraModelCall(t *testing.T) {
	th := newTestThinkerFull()
	p := &auditBatchProvider{}
	th.provider = p
	th.registry = NewToolRegistry("")
	th.handleTools = mainToolHandler(th)
	th.registry.Register(&ToolDef{Name: "fast", Handler: func(map[string]string) ToolResponse {
		time.Sleep(20 * time.Millisecond)
		return ToolResponse{Text: "fast result"}
	}})
	th.registry.Register(&ToolDef{Name: "slow", Handler: func(map[string]string) ToolResponse {
		time.Sleep(80 * time.Millisecond)
		return ToolResponse{Text: "slow result"}
	}})
	done := make(chan struct{})
	go func() { defer close(done); th.Run() }()
	time.Sleep(250 * time.Millisecond)
	th.Stop()
	<-done
	if got := p.count.Load(); got != 2 {
		t.Fatalf("one tool batch plus one result-processing call should require 2 model calls; got %d", got)
	}
}

func TestPerfAuditDoneWithPendingTool(t *testing.T) {
	parent := newTestThinkerFull()
	worker := newTestThinkerFull()
	worker.threadID = "worker"
	worker.telemetry = nil
	worker.registry = NewToolRegistry("")
	worker.recordPresentedTools([]NativeTool{{Name: "done"}})
	worker.pendingTools.Store("unfinished", "slow")
	thread := &Thread{ID: "worker", ParentID: "main", Parent: parent, Thinker: worker, Tools: map[string]bool{"done": true}}
	threadToolHandler(thread, parent.threads)(worker, []toolCall{{Name: "done", NativeID: "terminal", Args: map[string]string{"message": "finished"}}}, nil)
	if thread.doneForever {
		t.Fatal("done terminated a worker with an earlier tool still pending")
	}
}

func TestPerfAuditCompletionNotInParentDurableInbox(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinkerFull()
	parent.config.path = "config.json"
	worker := newTestThinkerFull()
	worker.threadID = "worker"
	worker.telemetry = nil
	worker.session = NewSession(".", "worker")
	worker.registry = NewToolRegistry("")
	worker.recordPresentedTools([]NativeTool{{Name: "done"}})
	thread := &Thread{ID: "worker", ParentID: "main", Parent: parent, Thinker: worker, Tools: map[string]bool{"done": true}}
	parent.threads.threads["worker"] = thread
	final := "unique completed work payload"
	threadToolHandler(thread, parent.threads)(worker, []toolCall{{Name: "done", NativeID: "terminal", Args: map[string]string{"message": final}}}, nil)
	if !thread.doneForever {
		t.Fatal("test did not reach completion")
	}
	parent.threads.cleanupThread("worker")
	found := false
	for _, e := range parent.config.MainEvents {
		if strings.Contains(e.Text, final) {
			found = true
		}
	}
	if !found {
		t.Fatal("worker completion exists only on parent's in-memory bus; worker history was deleted without persisting the parent inbox payload")
	}
}

func TestPerfAuditBarrierDelaysNewInstruction(t *testing.T) {
	th := retryTestThinker(nil)
	th.pendingTools.Store("slow", "slow")
	th.bus.Publish(Event{Type: EventInbox, To: "main", Text: "new instruction"})
	var results []ToolResult
	var texts []string
	var parts []ContentPart
	start := time.Now()
	th.waitForPendingTools(&results, &texts, &parts, 150*time.Millisecond)
	elapsed := time.Since(start)
	if elapsed >= 140*time.Millisecond {
		t.Fatalf("fresh instruction waited %s for unrelated pending tool; production barrier is 3 seconds", elapsed)
	}
}

func BenchmarkPerfAuditTrackedSendParallel(b *testing.B) {
	cfg := &Config{path: filepath.Join(b.TempDir(), "config.json")}
	for i := 0; i < 100; i++ {
		cfg.Threads = append(cfg.Threads, PersistentThread{ID: fmt.Sprint(i), Directive: strings.Repeat("d", 1024)})
	}
	cfg.EventExecutions = []PersistentEventExecution{{ExecutionID: "tracked", Status: eventActive, Participants: map[string]bool{"main": true, "worker": true}}}
	th := newTestThinkerFull()
	th.config = cfg
	th.eventLifecycle = NewEventLifecycle(cfg, nil)
	th.threads.threads["worker"] = &Thread{ID: "worker"}
	sub := th.bus.Subscribe("worker", 1024)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if !th.threads.SendWithPartsExecution("worker", "work", nil, []string{"tracked"}) {
				b.Error("send")
			}
			<-sub.C
		}
	})
}

func BenchmarkPerfAuditNativeStatus(b *testing.B) {
	for _, n := range []int{10, 100, 500} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			th := newTestThinkerFull()
			th.registry = &ToolRegistry{tools: map[string]*ToolDef{}}
			for i := 0; i < n; i++ {
				name := fmt.Sprint(i)
				th.registry.Register(&ToolDef{Name: name, Core: true, Description: strings.Repeat("description ", 20), InputSchema: map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string", "description": strings.Repeat("query ", 50)}}}})
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				th.publishContextStatus()
			}
		})
	}
}

type auditBlockingWorkerProvider struct {
	scriptedRetryProvider
	entered chan struct{}
}

func (p *auditBlockingWorkerProvider) Chat(ctx context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.entered <- struct{}{}
	<-ctx.Done()
	return ChatResponse{}, ctx.Err()
}
func TestPerfAuditWorkerModelCallsOverlap(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinkerFull()
	p := &auditBlockingWorkerProvider{entered: make(chan struct{}, 8)}
	parent.provider = p
	var workers []*Thinker
	var stopped sync.WaitGroup
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("worker-%d", i)
		if err := parent.threads.SpawnWithOpts(id, "work", nil, SpawnOpts{DeferRun: true}); err != nil {
			t.Fatal(err)
		}
		worker := parent.threads.threads[id].Thinker
		workers = append(workers, worker)
		original := worker.onStop
		stopped.Add(1)
		worker.onStop = func() { defer stopped.Done(); original() }
	}
	parent.threads.StartAll()
	received := 0
	timeout := time.After(time.Second)
loop:
	for received < 8 {
		select {
		case <-p.entered:
			received++
		case <-timeout:
			break loop
		}
	}
	for _, worker := range workers {
		worker.Stop()
	}
	stopped.Wait()
	if received != 8 {
		t.Fatalf("only %d/8 workers reached simultaneous in-flight model requests", received)
	}
}
