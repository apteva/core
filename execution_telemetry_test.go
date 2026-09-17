package core

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func traceEvents(t *testing.T, tel *Telemetry, kind string) []map[string]any {
	t.Helper()
	events, _ := tel.Events(0)
	var out []map[string]any
	for _, ev := range events {
		if ev.Type != kind {
			continue
		}
		var d map[string]any
		if err := json.Unmarshal(ev.Data, &d); err != nil {
			t.Fatal(err)
		}
		d["event_thread_id"] = ev.ThreadID
		out = append(out, d)
	}
	return out
}

func attachTraceTelemetry(th *Thinker) {
	th.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
}

func TestExecutionTelemetryRetryFallbackAttribution(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 1, failureErr: errors.New("temporary outage"), response: ChatResponse{Text: "ok", Usage: TokenUsage{PromptTokens: 20, CompletionTokens: 3}}}
	fallback := &scriptedRetryProvider{name: "fallback", failures: 1, failureErr: errors.New("temporary outage")}
	th := retryTestThinker(primary)
	attachTraceTelemetry(th)
	th.addEventExecutions([]string{"execution-A"})
	th.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "fallback": fallback}, order: []string{"primary", "fallback"}, default_: "primary"}
	if _, err := th.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	finished := traceEvents(t, th.telemetry, "llm.request.finished")
	if len(finished) != 3 {
		t.Fatalf("attempts: %+v", finished)
	}
	seen := map[any]bool{}
	for i, e := range finished {
		if seen[e["request_id"]] || e["request_id"] == "" {
			t.Fatalf("request identity reused: %+v", e)
		}
		seen[e["request_id"]] = true
		if e["turn_id"] != finished[0]["turn_id"] || e["worker_run_id"] != finished[0]["worker_run_id"] {
			t.Fatal("turn/run changed during retry")
		}
		if e["attempt"] != float64(i+1) {
			t.Fatalf("attempt order: %+v", e)
		}
		if e["execution_ids"].([]any)[0] != "execution-A" {
			t.Fatal("execution lost")
		}
	}
	if finished[0]["provider"] != "primary" || finished[1]["role"] != "fallback" || finished[2]["provider"] != "primary" || finished[2]["outcome"] != "success" {
		t.Fatalf("chain: %+v", finished)
	}
	if finished[2]["tokens_in"] != float64(20) || finished[2]["tokens_out"] != float64(3) {
		t.Fatal("usage lost")
	}
	backoffs := traceEvents(t, th.telemetry, "llm.retry.finished")
	if len(backoffs) != 1 || backoffs[0]["planned_backoff_ms"] != float64(1) || backoffs[0]["actual_backoff_ms"] == nil {
		t.Fatalf("backoff: %+v", backoffs)
	}
	if _, err := th.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	last := traceEvents(t, th.telemetry, "llm.request.finished")[3]
	if last["turn_id"] == finished[0]["turn_id"] || last["worker_run_id"] != finished[0]["worker_run_id"] {
		t.Fatal("next turn identity incorrect")
	}
}

type outputTraceProvider struct{ scriptedRetryProvider }

func (p *outputTraceProvider) Chat(_ context.Context, _ []Message, _ string, _ []NativeTool, text func(string), reasoning func(string), tool func(string, string, string)) (ChatResponse, error) {
	text("")
	reasoning("thinking")
	reasoning("more")
	tool("probe", "call_0", "{}")
	tool("probe", "call_0", "")
	return ChatResponse{ToolCalls: []NativeToolCall{{ID: "call_0", Name: "probe"}}}, nil
}

func TestExecutionTelemetryToolOnlyFirstOutput(t *testing.T) {
	th := retryTestThinker(&outputTraceProvider{scriptedRetryProvider: scriptedRetryProvider{name: "codex-fixture"}})
	attachTraceTelemetry(th)
	if _, err := th.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	outputs := traceEvents(t, th.telemetry, "llm.request.first_output")
	if len(outputs) != 2 || outputs[0]["output_kind"] != "reasoning" || outputs[1]["output_kind"] != "tool" {
		t.Fatalf("outputs: %+v", outputs)
	}
	finished := traceEvents(t, th.telemetry, "llm.request.finished")
	if finished[0]["request_id"] != outputs[0]["request_id"] {
		t.Fatal("output attribution differs")
	}
}

func TestExecutionTelemetryCancelledBeforeDispatch(t *testing.T) {
	p := &scriptedRetryProvider{name: "test"}
	th := retryTestThinker(p)
	attachTraceTelemetry(th)
	slots := th.bus.limits().mainLLM
	for i := 0; i < cap(slots); i++ {
		slots <- struct{}{}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := th.callLLMWithRetry(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error: %v", err)
	}
	if p.calls != 0 || len(traceEvents(t, th.telemetry, "llm.request.started")) != 0 {
		t.Fatal("dispatched despite unavailable slot")
	}
	finished := traceEvents(t, th.telemetry, "llm.request.finished")
	if len(finished) != 1 || finished[0]["outcome"] != "cancelled" || finished[0]["duration_ms"] != nil {
		t.Fatalf("terminal: %+v", finished)
	}
}

func TestExecutionTelemetryCancelledDuringRequest(t *testing.T) {
	p := &scriptedRetryProvider{name: "test", block: true, started: make(chan struct{})}
	th := retryTestThinker(p)
	attachTraceTelemetry(th)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := th.callLLMWithRetry(ctx); done <- err }()
	<-p.started
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	finished := traceEvents(t, th.telemetry, "llm.request.finished")
	if len(finished) != 1 || finished[0]["outcome"] != "cancelled" {
		t.Fatalf("terminal: %+v", finished)
	}
	if len(traceEvents(t, th.telemetry, "llm.request.first_output")) != 0 {
		t.Fatal("invented first output")
	}
}

func TestExecutionTelemetryAsyncOriginAndReusedCallIDs(t *testing.T) {
	th := retryTestThinker(&scriptedRetryProvider{name: "test"})
	attachTraceTelemetry(th)
	th.registry = NewToolRegistry("")
	started, release := make(chan struct{}, 2), make(chan struct{})
	th.registry.Register(&ToolDef{Name: "probe", HandlerContext: func(ctx context.Context, _ map[string]string) ToolResponse {
		started <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
		}
		return ToolResponse{Text: "ok"}
	}})
	th.beginTraceTurn()
	firstRequest := th.queueRequestTrace("test", "test")
	queueTool(th, toolCall{NativeID: "call_0", Name: "probe", Args: map[string]string{}})
	<-started
	th.beginTraceTurn()
	secondRequest := th.queueRequestTrace("test", "test")
	queueTool(th, toolCall{NativeID: "call_0", Name: "probe", Args: map[string]string{}})
	<-started
	close(release)
	defer th.Stop()
	deadline := time.Now().Add(2 * time.Second)
	for len(traceEvents(t, th.telemetry, "tool.result")) < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	results := traceEvents(t, th.telemetry, "tool.result")
	if len(results) != 2 {
		t.Fatalf("results: %+v", results)
	}
	ids := map[any]bool{results[0]["request_id"]: true, results[1]["request_id"]: true}
	if !ids[firstRequest.ids.RequestID] || !ids[secondRequest.ids.RequestID] || results[0]["tool_span_id"] == results[1]["tool_span_id"] {
		t.Fatalf("mixed origins: %+v", results)
	}
}

func TestExecutionTelemetryInlineTimingAndDeferredWorker(t *testing.T) {
	t.Chdir(t.TempDir())
	th := newTestThinker()
	defer th.Stop()
	th.provider = &scriptedRetryProvider{name: "test"}
	th.beginTraceTurn()
	req := th.queueRequestTrace("test", "test")
	call := toolCall{Name: "spawn", NativeID: "spawn-1"}
	th.queueToolTrace(&call)
	call.trace.start()
	if err := th.threads.SpawnWithOpts("worker", "test", nil, SpawnOpts{DeferRun: true, ParentTrace: call.trace}); err != nil {
		t.Fatal(err)
	}
	child := th.threads.threads["worker"].Thinker
	created := traceEvents(t, th.telemetry, "worker.created")
	if len(created) != 1 || created[0]["parent_request_id"] != req.ids.RequestID || created[0]["parent_tool_span_id"] != call.trace.id {
		t.Fatalf("creation: %+v", created)
	}
	if len(traceEvents(t, th.telemetry, "worker.ready")) != 0 {
		t.Fatal("deferred worker reported ready")
	}
	child.workerMilestone("ready", "")
	child.workerMilestone("ready", "")
	if len(traceEvents(t, th.telemetry, "worker.ready")) != 1 {
		t.Fatal("duplicate readiness")
	}
	// Deliberately delayed inline work must no longer report zero duration.
	time.Sleep(5 * time.Millisecond)
	if ms := call.trace.result("ok"); ms < 1 {
		t.Fatalf("duration=%d", ms)
	}
	child.workerMilestone("finished", "completed")
	child.workerMilestone("finished", "cancelled")
	finished := traceEvents(t, th.telemetry, "worker.finished")
	if len(finished) != 1 || finished[0]["outcome"] != "completed" {
		t.Fatalf("completion: %+v", finished)
	}
}

func TestExecutionTelemetryPhysicalHTTPAttempts(t *testing.T) {
	th := retryTestThinker(&scriptedRetryProvider{name: "test"})
	attachTraceTelemetry(th)
	r := th.queueRequestTrace("test", "test")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("X-Request-ID", "upstream-id")
		w.WriteHeader(400)
		_, _ = w.Write([]byte("invalid"))
	}))
	defer server.Close()
	for i := 0; i < 2; i++ {
		req, _ := http.NewRequestWithContext(context.WithValue(context.Background(), requestTraceKey{}, r), "POST", server.URL+"?key=secret", strings.NewReader("secret"))
		resp, err := tracedProviderHTTP(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	finished := traceEvents(t, th.telemetry, "llm.http.finished")
	if len(finished) != 2 || finished[0]["http_attempt_id"] == finished[1]["http_attempt_id"] || finished[0]["request_id"] != finished[1]["request_id"] {
		t.Fatalf("attempts: %+v", finished)
	}
	raw, _ := json.Marshal(finished)
	if strings.Contains(string(raw), "secret") {
		t.Fatal("trace leaked request data")
	}
}

func TestExecutionTelemetryActualInlineSpawnDuration(t *testing.T) {
	t.Chdir(t.TempDir())
	th := newTestThinker()
	th.provider = &scriptedRetryProvider{name: "test"}
	th.config.path = "config.json"
	defer th.Stop()
	th.threads.spawnMu.Lock()
	done := make(chan []ToolResult, 1)
	go func() {
		_, _, result := mainToolHandler(th)(th, []toolCall{{Name: "spawn", NativeID: "inline-spawn", Args: map[string]string{"id": "delayed-worker", "directive": "Wait for instructions", "paused": "true"}}}, nil)
		done <- result
	}()
	deadline := time.Now().Add(time.Second)
	for len(traceEvents(t, th.telemetry, "tool.call")) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(5 * time.Millisecond)
	th.threads.spawnMu.Unlock()
	select {
	case results := <-done:
		if len(results) != 1 || results[0].IsError {
			t.Fatalf("spawn: %+v", results)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("spawn did not finish")
	}
	results := traceEvents(t, th.telemetry, "tool.result")
	if len(results) == 0 || results[0]["duration_ms"].(float64) < 5 {
		t.Fatalf("inline duration: %+v", results)
	}
	created := traceEvents(t, th.telemetry, "worker.created")
	if len(created) != 1 || created[0]["parent_tool_span_id"] != results[0]["tool_span_id"] {
		t.Fatalf("spawn origin: %+v", created)
	}
}

func TestExecutionTelemetryCancelledBackoff(t *testing.T) {
	p := &scriptedRetryProvider{name: "test", failures: 10, failureErr: errors.New("temporary outage")}
	th := retryTestThinker(p)
	attachTraceTelemetry(th)
	th.retryDelay = func(error, int) time.Duration { return time.Minute }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := th.callLLMWithRetry(ctx); done <- err }()
	deadline := time.Now().Add(time.Second)
	for len(traceEvents(t, th.telemetry, "llm.retry.scheduled")) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	backoff := traceEvents(t, th.telemetry, "llm.retry.finished")
	if len(backoff) != 1 || backoff[0]["outcome"] != "cancelled" || backoff[0]["actual_backoff_ms"].(float64) >= 60000 {
		t.Fatalf("backoff: %+v", backoff)
	}
	if p.calls != 1 {
		t.Fatalf("unexpected retry after cancel: %d", p.calls)
	}
}

func TestExecutionTelemetryQueuedToolCancelled(t *testing.T) {
	th := newTestThinker()
	th.registry = NewToolRegistry("")
	dispatched := make(chan struct{}, 1)
	th.registry.Register(&ToolDef{Name: "probe", Handler: func(map[string]string) ToolResponse { dispatched <- struct{}{}; return ToolResponse{Text: "ok"} }})
	slots := th.bus.limits().mainTools
	for i := 0; i < cap(slots); i++ {
		slots <- struct{}{}
	}
	queueTool(th, toolCall{Name: "probe", NativeID: "waiting-tool", Args: map[string]string{}})
	deadline := time.Now().Add(time.Second)
	for len(traceEvents(t, th.telemetry, "tool.call")) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	th.Stop()
	deadline = time.Now().Add(time.Second)
	for len(traceEvents(t, th.telemetry, "tool.execution.finished")) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	finished := traceEvents(t, th.telemetry, "tool.execution.finished")
	if len(finished) != 1 || finished[0]["outcome"] != "cancelled" {
		t.Fatalf("queued cancellation: %+v", finished)
	}
	select {
	case <-dispatched:
		t.Fatal("handler ran despite unavailable slot")
	default:
	}
}
