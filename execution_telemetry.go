package core

// Detailed timing is additive. Legacy llm.done/tool.call/thread.spawn events
// remain the accounting source; these events must not increment usage totals.
import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

var traceSequence atomic.Uint64

func newTraceID(kind string) string {
	return fmt.Sprintf("%s-%d-%d-%d", kind, os.Getpid(), time.Now().UnixNano(), traceSequence.Add(1))
}

type traceIDs struct {
	WorkerRunID       string   `json:"worker_run_id,omitempty"`
	ParentWorkerRunID string   `json:"parent_worker_run_id,omitempty"`
	ParentRequestID   string   `json:"parent_request_id,omitempty"`
	ParentToolSpanID  string   `json:"parent_tool_span_id,omitempty"`
	TurnID            string   `json:"turn_id"`
	RequestID         string   `json:"request_id"`
	ExecutionIDs      []string `json:"execution_ids"`
}

type executionTrace struct {
	mu                                           sync.Mutex
	ids                                          traceIDs
	created                                      time.Time
	ready, firstInference, firstAction, finished bool
	attempt                                      int
}

func (t *Thinker) tracing() *executionTrace {
	t.traceOnce.Do(func() {
		if t.trace == nil {
			t.trace = &executionTrace{ids: traceIDs{WorkerRunID: newTraceID("run")}, created: time.Now()}
		}
		if t.telemetry != nil {
			t.telemetry.traces.Store(t.threadID, t.trace)
		}
	})
	return t.trace
}

func (tr *executionTrace) snapshot() traceIDs {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	ids := tr.ids
	ids.ExecutionIDs = append([]string(nil), ids.ExecutionIDs...)
	return ids
}

// IDs live in data because older servers only preserve the original envelope.
// Explicit snapshots win over the thread's current context (async results).
func withTrace(data any, ids traceIDs) map[string]any {
	raw, _ := json.Marshal(data)
	var out map[string]any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	_ = decoder.Decode(&out)
	if out == nil {
		out = map[string]any{}
	}
	if run, _ := out["worker_run_id"].(string); run != "" && run != ids.WorkerRunID {
		return out
	}
	fields, _ := json.Marshal(ids)
	var extra map[string]any
	_ = json.Unmarshal(fields, &extra)
	for k, v := range extra {
		if _, exists := out[k]; !exists {
			out[k] = v
		}
	}
	return out
}

func (t *Thinker) emitTrace(kind string, data any, ids traceIDs) {
	if t.telemetry != nil {
		t.telemetry.Emit(kind, t.threadID, withTrace(data, ids))
	}
}

func (t *Thinker) workerMilestone(kind, outcome string) {
	t.workerMilestoneTrace(kind, outcome, nil, "")
}

func (t *Thinker) workerMilestoneTrace(kind, outcome string, origin *traceIDs, toolSpan string) {
	tr := t.tracing()
	tr.mu.Lock()
	var flag *bool
	switch kind {
	case "ready":
		flag = &tr.ready
	case "first_inference":
		flag = &tr.firstInference
	case "first_action":
		flag = &tr.firstAction
	case "finished":
		flag = &tr.finished
	}
	if flag != nil && *flag {
		tr.mu.Unlock()
		return
	}
	if flag != nil {
		*flag = true
	}
	ids, created := tr.ids, tr.created
	tr.mu.Unlock()
	if origin != nil {
		ids = *origin
	}
	data := map[string]any{"outcome": outcome, "elapsed_ms": time.Since(created).Milliseconds()}
	if toolSpan != "" {
		data["tool_span_id"] = toolSpan
	}
	t.emitTrace("worker."+kind, data, ids)
}

func (t *Thinker) beginTraceTurn() {
	tr := t.tracing()
	ids := t.currentEventExecutions()
	tr.mu.Lock()
	tr.ids.TurnID, tr.ids.RequestID = newTraceID("turn"), ""
	tr.ids.ExecutionIDs = ids
	tr.attempt = 0
	tr.mu.Unlock()
}

type requestTrace struct {
	mu                       sync.Mutex
	t                        *Thinker
	ids                      traceIDs
	provider, model, role    string
	attempt                  int
	queued, waiting, started time.Time
	first                    map[string]int64
}

func (t *Thinker) queueRequestTrace(provider, model string) *requestTrace {
	tr := t.tracing()
	tr.mu.Lock()
	if tr.ids.TurnID == "" {
		tr.ids.TurnID = newTraceID("turn")
	}
	tr.ids.RequestID = newTraceID("request")
	tr.attempt++
	r := &requestTrace{t: t, ids: tr.ids, provider: provider, model: model, role: "primary", attempt: tr.attempt, queued: time.Now(), first: map[string]int64{}}
	tr.mu.Unlock()
	if t.provider != nil && provider != t.provider.Name() {
		r.role = "fallback"
	}
	r.emit("queued", nil)
	return r
}

func (r *requestTrace) emit(phase string, values map[string]any) {
	if values == nil {
		values = map[string]any{}
	}
	values["provider"], values["model"], values["role"], values["attempt"] = r.provider, r.model, r.role, r.attempt
	r.t.emitTrace("llm.request."+phase, values, r.ids)
}

func (r *requestTrace) wait() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.waiting = time.Now()
}

func (r *requestTrace) start() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.started = time.Now()
	if r.waiting.IsZero() {
		r.waiting = r.queued
	}
	r.emit("started", map[string]any{"queue_ms": r.started.Sub(r.waiting).Milliseconds(), "preparation_ms": r.waiting.Sub(r.queued).Milliseconds()})
	r.t.workerMilestoneTrace("first_inference", "", &r.ids, "")
}

func (r *requestTrace) output(kind string, present bool) {
	if !present {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.first[kind]; exists || r.started.IsZero() {
		return
	}
	ms := time.Since(r.started).Milliseconds()
	r.first[kind] = ms
	r.emit("first_output", map[string]any{"output_kind": kind, "elapsed_ms": ms})
}

func (r *requestTrace) finish(resp ChatResponse, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	outcome := "success"
	if err != nil {
		outcome = "failed"
	}
	if errors.Is(err, context.Canceled) {
		outcome = "cancelled"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		outcome = "timeout"
	}
	values := map[string]any{"outcome": outcome, "total_ms": time.Since(r.queued).Milliseconds(), "tokens_in": resp.Usage.PromptTokens, "tokens_out": resp.Usage.CompletionTokens, "tokens_cached": resp.Usage.CachedTokens, "cache_write_tokens": resp.Usage.CacheWriteTokens, "usage_reported": resp.Usage != (TokenUsage{}), "first_output_ms": r.first, "provider_timing": resp.ProviderTiming}
	if !r.started.IsZero() {
		values["duration_ms"] = time.Since(r.started).Milliseconds()
		values["queue_ms"] = r.started.Sub(r.waiting).Milliseconds()
		values["preparation_ms"] = r.waiting.Sub(r.queued).Milliseconds()
	} else {
		if r.waiting.IsZero() {
			values["preparation_ms"] = time.Since(r.queued).Milliseconds()
		} else {
			values["queue_ms"] = time.Since(r.waiting).Milliseconds()
			values["preparation_ms"] = r.waiting.Sub(r.queued).Milliseconds()
		}
	}
	if err != nil {
		values["error"] = err.Error()
	}
	r.emit("finished", values)
}

func (t *Thinker) traceRetry(attempt int, delay time.Duration, err error) func(string) {
	ids := t.tracing().snapshot()
	start := time.Now()
	data := map[string]any{"retry_id": newTraceID("retry"), "attempt": attempt, "planned_backoff_ms": delay.Milliseconds(), "reason": err.Error()}
	t.emitTrace("llm.retry.scheduled", data, ids)
	return func(outcome string) {
		data["actual_backoff_ms"], data["outcome"] = time.Since(start).Milliseconds(), outcome
		t.emitTrace("llm.retry.finished", data, ids)
	}
}

type toolTrace struct {
	mu               sync.Mutex
	t                *Thinker
	ids              traceIDs
	id, callID, name string
	queued, started  time.Time
	finished         bool
}

func (t *Thinker) queueToolTrace(call *toolCall) {
	if call.trace != nil {
		return
	}
	ids := t.tracing().snapshot()
	ids.ExecutionIDs = append([]string(nil), call.executionIDs...)
	if ids.ExecutionIDs == nil {
		ids.ExecutionIDs = t.currentEventExecutions()
	}
	call.trace = &toolTrace{t: t, ids: ids, id: newTraceID("tool"), callID: call.NativeID, name: call.Name, queued: time.Now()}
	call.trace.emit("queued", nil)
}

func (s *toolTrace) data(values any) map[string]any {
	d := withTrace(values, s.ids)
	d["tool_span_id"] = s.id
	// Empty origin IDs must also override a newer thread context.
	d["request_id"], d["turn_id"], d["execution_ids"] = s.ids.RequestID, s.ids.TurnID, s.ids.ExecutionIDs
	return d
}

func (s *toolTrace) emit(phase string, data map[string]any) {
	if data == nil {
		data = map[string]any{}
	}
	data["id"], data["name"] = s.callID, s.name
	s.t.emitTrace("tool.execution."+phase, s.data(data), s.ids)
}

func (s *toolTrace) start() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started.IsZero() || s.finished {
		return
	}
	s.started = time.Now()
	s.emit("started", map[string]any{"queue_ms": s.started.Sub(s.queued).Milliseconds()})
	s.t.workerMilestoneTrace("first_action", "", &s.ids, s.id)
}

func (s *toolTrace) finish(outcome string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return
	}
	s.finished = true
	data := map[string]any{"outcome": outcome, "total_ms": time.Since(s.queued).Milliseconds()}
	if !s.started.IsZero() {
		data["duration_ms"] = time.Since(s.started).Milliseconds()
		data["queue_ms"] = s.started.Sub(s.queued).Milliseconds()
	}
	s.emit("finished", data)
}

func (s *toolTrace) result(content string) int64 {
	if s == nil {
		return 0
	}
	outcome := "success"
	if inlineToolResultIsError(content) {
		outcome = "failed"
	}
	s.finish(outcome)
	return s.duration()
}

func (s *toolTrace) duration() int64 {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started.IsZero() {
		return 0
	}
	return time.Since(s.started).Milliseconds()
}
