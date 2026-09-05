package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAuditConcurrentMailboxLatency(t *testing.T) {
	const workers, senders, perSender = 32, 16, 100
	bus := NewEventBus()
	stop := make(chan struct{})
	defer close(stop)
	type receipt struct {
		id      string
		latency time.Duration
	}
	received := make(chan receipt, senders*perSender)
	for i := 0; i < workers; i++ {
		sub := bus.Subscribe(fmt.Sprint(i), 4)
		go func() {
			for {
				select {
				case <-stop:
					return
				case <-sub.Wake:
					for _, event := range sub.DrainTargeted() {
						sent, _ := strconv.ParseInt(event.Text, 10, 64)
						received <- receipt{event.ID, time.Since(time.Unix(0, sent))}
					}
				}
			}
		}()
	}
	var wg sync.WaitGroup
	for sender := 0; sender < senders; sender++ {
		wg.Go(func() {
			for seq := 0; seq < perSender; seq++ {
				bus.Publish(Event{ID: fmt.Sprintf("%d/%d", sender, seq), Type: EventInbox, To: fmt.Sprint((sender + seq) % workers), Text: strconv.FormatInt(time.Now().UnixNano(), 10)})
			}
		})
	}
	wg.Wait()
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	seen := map[string]bool{}
	latencies := make([]time.Duration, 0, senders*perSender)
	for len(seen) < senders*perSender {
		select {
		case r := <-received:
			if seen[r.id] {
				t.Fatalf("duplicate delivery %s", r.id)
			}
			seen[r.id] = true
			latencies = append(latencies, r.latency)
		case <-deadline.C:
			t.Fatalf("delivered %d/%d", len(seen), senders*perSender)
		}
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	t.Logf("MAILBOX_LATENCY workers=%d senders=%d messages=%d p50=%s p95=%s p99=%s", workers, senders, len(seen), latencies[len(latencies)*50/100], latencies[len(latencies)*95/100], latencies[len(latencies)*99/100])
}

func TestAuditRuntimeJournalRecoveryAndRollback(t *testing.T) {
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "preserve static configuration", EventExecutions: []PersistentEventExecution{{ExecutionID: "e", Status: eventActive, Participants: map[string]bool{"main": true}}}}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	original, _ := os.ReadFile(cfg.path)
	lifecycle := &EventLifecycle{config: cfg}
	if err := lifecycle.Propagate([]string{"e"}, "worker"); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(cfg.path)
	if string(after) != string(original) {
		t.Fatal("propagation rewrote static configuration")
	}
	journal, _ := os.ReadFile(cfg.path + ".runtime.jsonl")
	if len(journal) == 0 || strings.Contains(string(journal), cfg.Directive) {
		t.Fatalf("not a compact runtime delta: %s", journal)
	}
	restored := &Config{path: cfg.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	if !restored.EventExecutions[0].Participants["worker"] {
		t.Fatal("lost participant after restart")
	}
	if err := lifecycle.Propagate([]string{"e"}, "worker"); err != nil {
		t.Fatal(err)
	}
	same, _ := os.ReadFile(cfg.path + ".runtime.jsonl")
	if string(same) != string(journal) {
		t.Fatal("redundant propagation appended journal")
	}
	if err := os.WriteFile(cfg.path+".runtime.jsonl", bytes.TrimSuffix(journal, []byte("\n")), 0600); err != nil {
		t.Fatal(err)
	}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	repaired, _ := os.ReadFile(cfg.path + ".runtime.jsonl")
	if !bytes.Equal(repaired, journal) {
		t.Fatal("complete final record lost its append separator")
	}
	f, err := os.OpenFile(cfg.path+".runtime.jsonl", os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString(`{"seq":`)
	_ = f.Close()
	recovered := &Config{path: cfg.path}
	if err := recovered.load(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(cfg.path + ".runtime.jsonl"); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(cfg.path+".runtime.jsonl", 0700); err != nil {
		t.Fatal(err)
	}
	if err := lifecycle.Propagate([]string{"e"}, "rejected"); err == nil {
		t.Fatal("expected storage error")
	}
	if cfg.EventExecutions[0].Participants["rejected"] {
		t.Fatal("failed write mutated runtime")
	}
}

func TestAuditSessionResetRejectsInFlightCompaction(t *testing.T) {
	s := NewSession(t.TempDir(), "worker")
	for i := 0; i < 5; i++ {
		if err := s.AppendMessage(Message{Role: "user", Content: fmt.Sprint("old", i)}, 0, TokenUsage{}); err != nil {
			t.Fatal(err)
		}
	}
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		s.ForceCompact(1, func(string) string { close(started); <-release; return "OLD SUMMARY" })
		close(done)
	}()
	<-started
	if err := s.Reset(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if err := s.AppendMessage(Message{Role: "user", Content: fmt.Sprint("new", i)}, 0, TokenUsage{}); err != nil {
			t.Fatal(err)
		}
	}
	close(release)
	<-done
	raw, err := os.ReadFile(s.path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "OLD SUMMARY") || !strings.Contains(string(raw), "new0") {
		t.Fatalf("compaction overwrote reset: %s", raw)
	}
}

func TestAuditNativeAndGeminiRejectMissingCompletion(t *testing.T) {
	_, err := (&OpenAINativeProvider{}).streamResponse(strings.NewReader("data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n"), nil, nil, nil)
	if err == nil {
		t.Fatal("native accepted missing terminal frame")
	}
	_, err = parseGeminiStream(strings.NewReader("data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"write\",\"args\":{}}}]}}]}\n"), nil, nil)
	if err == nil {
		t.Fatal("Gemini accepted missing terminal frame")
	}
}

func TestAuditGeminiTypedArgumentsReplay(t *testing.T) {
	old := llmHTTPClient
	defer func() { llmHTTPClient = old }()
	var request geminiRequest
	llmHTTPClient = &http.Client{Transport: auditRoundTripper(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: http.NoBody}, nil
	})}
	p := NewGoogleProvider("fixture")
	_, _ = p.Chat(context.Background(), []Message{{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "id", Name: "lookup", Args: map[string]string{"s": "123", "n": "123", "b": "true"}, CanonicalArgs: json.RawMessage(`{"s":"123","n":123,"b":true}`)}}}}, "fixture", nil, nil, nil, nil)
	var args map[string]any
	for _, content := range request.Contents {
		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				args = part.FunctionCall.Args
			}
		}
	}
	if args == nil {
		t.Fatal("missing function replay")
	}
	if args["s"] != "123" || args["n"] != float64(123) || args["b"] != true {
		t.Fatalf("typed replay=%#v", args)
	}
}

func TestAuditMCPHTTPCancellationAndClose(t *testing.T) {
	entered := make(chan struct{}, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		entered <- struct{}{}
		<-r.Context().Done()
	}))
	defer server.Close()
	mcp := &MCPHTTPServer{Name: "slow", url: server.URL, client: server.Client()}
	done := make(chan error, 1)
	go func() { _, _, _, err := mcp.doPOSTContext(context.Background(), []byte(`{"id":1}`), true); done <- err }()
	<-entered
	mcp.Close()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("close did not cancel request")
		}
	case <-time.After(time.Second):
		t.Fatal("MCP Close blocked")
	}
}

func TestAuditResetCancelsToolsAndSuppressesLateResult(t *testing.T) {
	th := newTestThinkerFull()
	th.registry = NewToolRegistry("")
	defer th.Stop()
	started, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	th.registry.Register(&ToolDef{Name: "slow_reset", HandlerContext: func(ctx context.Context, _ map[string]string) ToolResponse {
		close(started)
		<-ctx.Done()
		close(cancelled)
		<-release
		return ToolResponse{Text: "STALE"}
	}})
	queueTool(th, toolCall{Name: "slow_reset", NativeID: "old", Args: map[string]string{}})
	<-started
	if _, err := resetThinkerContext(th); err != nil {
		t.Fatal(err)
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("reset did not cancel tool")
	}
	close(release)
	deadline := time.Now().Add(time.Second)
	for th.asyncToolsActive.Load() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	for _, ev := range th.sub.DrainTargeted() {
		if ev.ToolResult != nil && ev.ToolResult.CallID == "old" {
			t.Fatal("stale result survived reset")
		}
	}
}

func TestAuditMailboxAdmissionPreservesAcceptedEvents(t *testing.T) {
	bus := NewEventBus()
	sub := bus.Subscribe("worker", 1)
	payload := strings.Repeat("x", 1<<20)
	for i := 0; i < 64; i++ {
		if !bus.TryPublish(Event{Type: EventInbox, To: "worker", Text: payload}) {
			t.Fatalf("rejected event %d early", i)
		}
	}
	if bus.TryPublish(Event{Type: EventInbox, To: "worker", Text: "overflow"}) {
		t.Fatal("unbounded mailbox admission")
	}
	if got := len(sub.DrainTargeted()); got != 64 {
		t.Fatalf("lost accepted events: %d", got)
	}
	if !bus.TryPublish(Event{Type: EventInbox, To: "worker", Text: "after drain"}) {
		t.Fatal("capacity did not recover")
	}
}

func TestAuditBoundedProviderRetries(t *testing.T) {
	p := &scriptedRetryProvider{name: "fixture", failures: 100}
	th := retryTestThinker(p)
	if _, err := th.callLLMWithRetry(context.Background()); err == nil {
		t.Fatal("expected exhausted error")
	}
	if p.calls != 4 {
		t.Fatalf("attempts=%d want 4", p.calls)
	}
}

func TestAuditShutdownPreservesWorkers(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: "config.json", Directive: "main"}
	main := NewThinker("", newParkedAPIProvider(), cfg)
	if err := main.threads.SpawnWithOpts("worker", "wait", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	state, err := main.threads.PersistentState("worker")
	if err != nil {
		t.Fatal(err)
	}
	if err = cfg.SaveThread(state); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err = main.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if len(cfg.GetThreads()) != 1 {
		t.Fatal("shutdown erased worker definition")
	}
}

func TestAuditArchiveBudgetRefusesGrowthWithoutDeletingHistory(t *testing.T) {
	t.Setenv("APTEVA_TOOL_ARCHIVE_MAX_BYTES", "32")
	archive := NewToolResultArchive(t.TempDir(), "worker")
	if _, err := archive.Archive(ToolResult{CallID: "c", Content: strings.Repeat("payload", 100)}); err == nil {
		t.Fatal("archive ignored storage budget")
	}
}

func TestAuditWorkerInboxUsesRuntimeJournal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	cfg := &Config{path: path, Threads: []PersistentThread{{ID: "worker", Directive: "keep static"}}}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	event := PersistentThreadEvent{ID: "input", Text: "work", Hash: threadEventHash("work", nil)}
	if err := cfg.saveThreadAndRegisterEventExecutions(PersistentThread{ID: "worker", Directive: "keep static", Events: []PersistentThreadEvent{event}}, nil); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("worker inbox rewrote config")
	}
	restored := &Config{path: path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	if len(restored.Threads[0].Events) != 1 || restored.Threads[0].Events[0].Text != "work" {
		t.Fatal("worker inbox not recovered")
	}
}
