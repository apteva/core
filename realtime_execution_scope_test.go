package core

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Script the model only; exercise the ordinary worker, tool authorization,
// dispatch, durable assignment delivery and realtime result forwarding.
type scopedBookingProvider struct {
	scriptedRetryProvider
	calls atomic.Int32
}

func (p *scopedBookingProvider) SupportsNativeTools() bool { return true }
func (p *scopedBookingProvider) Chat(_ context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	switch p.calls.Add(1) {
	case 1:
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "availability-1", Name: "appointments_availability", Args: map[string]string{"_reason": "Check tomorrow at 15:00"}}}}, nil
	case 2:
		return ChatResponse{Text: "Tomorrow at 15:00 is available. Please confirm."}, nil
	case 3:
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "booking-1", Name: "appointments_book", Args: map[string]string{"_reason": "Book the confirmed appointment"}}}}, nil
	default:
		return ChatResponse{Text: "Your appointment is confirmed, reference R42."}, nil
	}
}

func TestRealtimeExecutionScopedPauseBookingAndCancellation(t *testing.T) {
	for _, global := range []bool{false, true} {
		name := "root_paused"
		if global {
			name = "instance_paused"
		}
		t.Run(name, func(t *testing.T) {
			rt, session := completionFixture(t)
			cfg := rootPausedExecutionConfig()
			if global {
				cfg.Mode = ExecutionPaused
			}
			rt.execution = NewExecutionController(cfg)
			root, _ := startScopedGate(t, rt.execution, "main")
			waitForActiveThread(t, rt.execution, "main")
			rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, CoreExecution: true, TranscriptFragments: true}
			p := &scopedBookingProvider{scriptedRetryProvider: scriptedRetryProvider{name: "fixture"}}
			rt.Thinker.provider = p
			rt.pool = &ProviderPool{providers: map[string]LLMProvider{"fixture": p}, default_: "fixture"}
			rt.configureExecution()
			rt.registry = NewToolRegistry("")
			rt.toolAllowlist = map[string]bool{"appointments_availability": true, "appointments_book": true}
			catalog := []mcpToolDef{mkTool("availability", "Check appointment availability"), mkTool("book", "Book a confirmed appointment")}
			rt.toolIndex.Add("appointments", catalog, false)
			var availability, bookings atomic.Int32
			for _, name := range []string{"appointments_availability", "appointments_book"} {
				rt.registry.Register(&ToolDef{Name: name, Description: name, InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}, Handler: func(_ map[string]string) ToolResponse {
					if name == "appointments_availability" {
						availability.Add(1)
						return ToolResponse{Text: `{"available":true,"time":"tomorrow at 15:00"}`}
					}
					bookings.Add(1)
					return ToolResponse{Text: `{"status":"confirmed","reference":"R42"}`}
				}})
			}
			defer rt.stopExecutionThread()
			delegate := func(id, text string, end int) {
				rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: id, Transcript: text, Final: true, TranscriptEndMS: end})
				e := RealtimeEvent{Type: RealtimeEventDelegation, DelegationID: id, DelegationOffsetMS: end}
				rt.handleSessionEvent(e)
				rt.handleSessionEvent(e) // duplicate notifications must not repeat work
				rt.backend.lastInput = time.Now().Add(-time.Second)
				rt.maybeStartExecution()
			}
			awaitResult := func() {
				deadline := time.After(3 * time.Second)
				for rt.backend.active {
					select {
					case <-deadline:
						t.Fatal("delegated worker did not finish")
					case <-rt.sub.Wake:
						for _, e := range rt.sub.DrainTargeted() {
							rt.handleBusEvent(e)
						}
					}
				}
			}
			delegate("availability", "Can you call me tomorrow at 15:00?", 800)
			if rt.backend.worker == nil || rt.backend.worker.Thinker.execution != rt.execution {
				t.Fatal("worker did not share execution controller")
			}
			workerID := rt.backend.worker.ID
			if global {
				waitForWaitingCount(t, rt.execution, 2)
				if p.calls.Load() != 0 || availability.Load() != 0 {
					t.Fatal("global pause allowed backend execution")
				}
				rt.stopExecutionThread()
				waitForWaitingCount(t, rt.execution, 1)
			} else {
				awaitResult()
				if availability.Load() != 1 || bookings.Load() != 0 {
					t.Fatal("availability must not book without confirmation")
				}
				delegate("confirmation", "Yes, I confirm tomorrow at 15:00.", 1800)
				awaitResult()
				if availability.Load() != 1 || bookings.Load() != 1 {
					t.Fatalf("availability=%d bookings=%d", availability.Load(), bookings.Load())
				}
				if len(session.texts) != 2 || !strings.Contains(session.texts[1].Content, "R42") {
					t.Fatalf("missing confirmed result: %#v", session.texts)
				}
				if session.responses != 0 {
					t.Fatal("execution controls forced a realtime response")
				}
			}
			// A queued assignment must not be misreported as actual running work.
			rt.telemetry.mu.Lock()
			var queued, blocked, running bool
			for _, event := range rt.telemetry.log {
				var data map[string]any
				_ = json.Unmarshal(event.Data, &data)
				queued = queued || event.Type == "realtime.execution_queued" && data["state"] == "queued"
				blocked = blocked || event.Type == "execution.waiting" && data["thread_id"] == workerID && data["scope"] == "instance" && data["state"] == "blocked"
				running = running || event.Type == "execution.running" && data["thread_id"] == workerID && data["request_id"] != "" && data["delegation_id"] != ""
			}
			rt.telemetry.mu.Unlock()
			if !queued || global && (!blocked || running) || !global && (!running || blocked) {
				t.Fatalf("queued=%v blocked=%v running=%v global=%v", queued, blocked, running, global)
			}
			if !rt.execution.CancelThread("main") {
				t.Fatal("worker execution released root pause")
			}
			requireGateResult(t, root, false)
		})
	}
}
