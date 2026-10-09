package core

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newRealtimeToolLifecycleFixture(t *testing.T) (*RealtimeThinker, *fakeRealtimeSession) {
	t.Helper()
	thinker := newTestThinker()
	t.Cleanup(thinker.Stop)
	thinker.registry = NewToolRegistry("test")
	thinker.toolAllowlist = map[string]bool{"probe": true}
	thinker.recordPresentedTools([]NativeTool{{Name: "probe"}})
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, nil, nil)
	t.Cleanup(rt.cancel)
	session := newFakeRealtimeSession()
	rt.replaceSession(session)
	return rt, session
}

func TestRealtimeToolReplayDoesNotExecuteOrStartAnotherResponse(t *testing.T) {
	rt, session := newRealtimeToolLifecycleFixture(t)
	calls := 0
	rt.handleTools = func(_ *Thinker, call []toolCall, _ []string) ([]string, []string, []ToolResult) {
		calls++
		return nil, nil, []ToolResult{{CallID: call[0].NativeID, ToolName: "probe", Content: "done"}}
	}
	event := RealtimeEvent{Type: RealtimeEventToolCall, ResponseID: "r", ToolCallID: "a", ToolName: "probe", ToolArgs: `{}`}
	rt.handleSessionEvent(event)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "r"})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "final"})
	rt.handleSessionEvent(event)
	if calls != 1 || len(session.toolResults) != 1 || rt.responseInProgress() {
		t.Fatalf("replay executed or changed state: calls=%d results=%d active=%v", calls, len(session.toolResults), rt.responseInProgress())
	}
	// A new ID remains a distinct operation, even with identical arguments.
	event.ToolCallID = "b"
	rt.handleSessionEvent(event)
	if calls != 2 {
		t.Fatal("legitimate repeated operation was suppressed")
	}
	// Provider IDs may be reused by another socket.
	rt.replaceSession(newFakeRealtimeSession())
	event.ToolCallID = "a"
	rt.handleSessionEvent(event)
	if calls != 3 {
		t.Fatal("terminal replay protection leaked across sessions")
	}
}

func TestRealtimeToolCancellationStopsContextAndPreservesLateOutcome(t *testing.T) {
	rt, session := newRealtimeToolLifecycleFixture(t)
	rt.beginToolCall(RealtimeEvent{ResponseID: "r", ToolCallID: "a", ToolName: "probe"}, session)
	ctx := rt.realtimeToolContext(session, "a")
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventToolCallCancelled, ToolCallID: "a"})
	if ctx.Err() != context.Canceled || rt.pendingToolWork() || session.responses != 0 {
		t.Fatal("cancellation did not stop only the pending operation")
	}
	rt.submitToolResult(session, "a", "completed just before cancellation", false)
	if len(session.toolResults) != 0 || rt.recovery.want {
		t.Fatal("cancelled result reached provider or caused recovery")
	}
	event := Event{Type: EventInbox, ToolResult: &ToolResult{CallID: "a", ToolName: "probe", Content: "actual outcome"}}
	before := len(rt.messages)
	rt.handleBusEvent(event)
	rt.handleBusEvent(event)
	if len(rt.messages) != before+1 || rt.messages[len(rt.messages)-1].ToolResults[0].Content != "actual outcome" {
		t.Fatal("late outcome lost or duplicated")
	}
	if len(session.toolResults) != 0 || rt.recovery.want {
		t.Fatal("late completion woke provider or triggered recovery")
	}
}

func TestRealtimeToolCancellationDoesNotAffectOtherCalls(t *testing.T) {
	rt, session := newRealtimeToolLifecycleFixture(t)
	for _, id := range []string{"a", "b"} {
		rt.beginToolCall(RealtimeEvent{ResponseID: "r", ToolCallID: id, ToolName: "probe"}, session)
	}
	b := rt.realtimeToolContext(session, "b")
	rt.cancelRealtimeToolCall(session, "a")
	if b.Err() != nil || !rt.pendingToolWork() {
		t.Fatal("one cancellation affected sibling work")
	}
	rt.submitToolResult(session, "b", "ok", false)
	if rt.pendingToolWork() || len(session.toolResults) != 1 || session.toolResults[0].callID != "b" {
		t.Fatal("uncancelled result was lost")
	}
}

func TestRealtimeToolCancellationReachesExecutionAndSkipsQueuedCalls(t *testing.T) {
	for _, queued := range []bool{false, true} {
		t.Run(fmt.Sprint(queued), func(t *testing.T) {
			rt, session := newRealtimeToolLifecycleFixture(t)
			rt.beginToolCall(RealtimeEvent{ResponseID: "r", ToolCallID: "a", ToolName: "probe"}, session)
			ctx := rt.realtimeToolContext(session, "a")
			started, finished := make(chan struct{}), make(chan struct{})
			var calls atomic.Int32
			call := toolCall{Name: "probe", NativeID: "a", context: ctx, definition: &ToolDef{
				Name: "probe", HandlerContext: func(ctx context.Context, _ map[string]string) ToolResponse {
					calls.Add(1)
					close(started)
					<-ctx.Done()
					close(finished)
					return ToolResponse{Text: "cancelled", IsError: true}
				},
			}}
			if queued {
				rt.Thinker.pendingTools.Store("a", "probe")
				rt.cancelRealtimeToolCall(session, "a")
				executeTool(rt.Thinker, call)
				if calls.Load() != 0 {
					t.Fatal("cancelled queued call executed")
				}
				if _, pending := rt.Thinker.pendingTools.Load("a"); pending {
					t.Fatal("cancelled queued call remained pending")
				}
			} else {
				executeTool(rt.Thinker, call)
				select {
				case <-started:
				case <-time.After(2 * time.Second):
					t.Fatal("tool did not start")
				}
				rt.cancelRealtimeToolCall(session, "a")
				select {
				case <-finished:
				case <-time.After(2 * time.Second):
					t.Fatal("tool context not cancelled")
				}
			}
			if rt.Thinker.toolContext().Err() != nil {
				t.Fatal("call cancellation stopped the owner")
			}
		})
	}
}

func TestRealtimeTerminalToolRecordsBoundedWithoutEvictingPendingCalls(t *testing.T) {
	rt, session := newRealtimeToolLifecycleFixture(t)
	rt.beginToolCall(RealtimeEvent{ToolCallID: "pending", ToolName: "probe"}, session)
	for i := 0; i < realtimeTerminalToolLimit+12; i++ {
		id := fmt.Sprint(i)
		rt.beginToolCall(RealtimeEvent{ToolCallID: id, ToolName: "probe"}, session)
		rt.completeToolCall(id)
	}
	if len(rt.toolCallRecords) != realtimeTerminalToolLimit+1 || !rt.seenToolCall(session, "pending") || rt.seenToolCall(session, "0") {
		t.Fatal("replay ledger is unbounded or evicted pending work")
	}
}

func TestRealtimeCancelledCallDoesNotWaitForSaturatedToolPool(t *testing.T) {
	rt, session := newRealtimeToolLifecycleFixture(t)
	rt.maxConcurrentTools = 1
	rt.initializeToolSlots()
	rt.acquireToolSlot()
	defer rt.releaseToolSlot()
	rt.beginToolCall(RealtimeEvent{ToolCallID: "a", ToolName: "probe"}, session)
	ctx := rt.realtimeToolContext(session, "a")
	finished := make(chan struct{})
	go func() {
		executeTool(rt.Thinker, toolCall{Name: "probe", NativeID: "a", context: ctx, definition: &ToolDef{Name: "probe"}})
		close(finished)
	}()
	rt.cancelRealtimeToolCall(session, "a")
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled call waited for an execution slot")
	}
}

func TestGoogleRealtimeCancellationEmitsGenericEventAndDropsQueuedResponse(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	s.callNames["a"], s.callNames["b"] = "probe", "probe"
	s.pendingResponses = []googleLiveFunctionResponse{{ID: "a"}, {ID: "b"}}
	s.translate(&googleLiveServerMessage{ToolCallCancellation: &struct {
		IDs []string `json:"ids"`
	}{IDs: []string{"a"}}})
	event := <-s.events
	if event.Type != RealtimeEventToolCallCancelled || event.ToolCallID != "a" || len(s.pendingResponses) != 1 || s.pendingResponses[0].ID != "b" {
		t.Fatal("cancellation did not normalize or retained a cancelled response")
	}
	if !errors.Is(s.SendToolResult("a", "late", false), ErrRealtimeToolCallCancelled) {
		t.Fatal("late cancelled result treated as unknown call")
	}
}

func TestRealtimeCancellationBeforeCoreEventDoesNotTriggerRecovery(t *testing.T) {
	rt, _ := newRealtimeToolLifecycleFixture(t)
	s := newGoogleRealtimeTestSession()
	rt.replaceSession(s)
	s.callNames["a"] = "probe"
	rt.beginToolCall(RealtimeEvent{ResponseID: "r", ToolCallID: "a", ToolName: "probe"}, s)
	// The adapter received cancellation, but Core's event loop handles the
	// bus completion before it consumes the corresponding provider event.
	s.translate(&googleLiveServerMessage{ToolCallCancellation: &struct {
		IDs []string `json:"ids"`
	}{IDs: []string{"a"}}})
	event := Event{Type: EventInbox, ToolResult: &ToolResult{CallID: "a", ToolName: "probe", Content: "actual outcome"}}
	before := len(rt.messages)
	rt.handleBusEvent(event)
	rt.handleSessionEvent(<-s.events)
	rt.handleBusEvent(event)
	if rt.pendingToolWork() || rt.recovery.want || len(s.outbox) != 0 || len(rt.messages) != before+1 {
		t.Fatal("cancellation/delivery race caused recovery, stale response, or duplicate history")
	}
}

func TestRealtimeAsyncInstructionsAreProviderNeutralAndModeScoped(t *testing.T) {
	for _, tc := range []struct {
		model, mode string
		async       bool
	}{
		{"gemini-3.8-live", "", false}, {"gemini-3.8-live", "async", true}, {"gemini-3.8-live-extended-thinking", "", true},
	} {
		wire, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: tc.model, Instructions: "Talk naturally.", OutputConfig: RealtimeOutputConfig{ToolMode: tc.mode}}, "Kore")
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(wire), "BACKGROUND TOOL RESULTS") != tc.async {
			t.Fatalf("async contract incorrectly scoped: %s", wire)
		}
	}
	if realtimeToolInstructions("unchanged", false) != "unchanged" || strings.Contains(realtimeAsyncToolPrompt, "Gemini") {
		t.Fatal("generic instructions changed blocking or contain provider rules")
	}
}

func TestLiveProbeEvaluationRejectsPrematureAndCorrectedClaims(t *testing.T) {
	for _, text := range []string{"Both markers have returned: ALPHA and BETA.", "BRAVO", "Both probes have completed."} {
		if liveProbeTranscriptError(text, map[string]bool{"probe_fast": true}) == nil {
			t.Fatalf("accepted premature/wrong output %q", text)
		}
	}
	if liveProbeTranscriptError("Both probe results have returned: ALPHA and OMEGA.", map[string]bool{"probe_fast": true, "probe_slow": true}) == nil {
		t.Fatal("accepted an incorrect marker pair after both deliveries")
	}
	if err := liveProbeTranscriptError("Both probes completed: ALPHA and BRAVO.", map[string]bool{"probe_fast": true, "probe_slow": true}); err != nil {
		t.Fatal(err)
	}
	if err := liveProbeTranscriptError("Checking now.", nil); err != nil {
		t.Fatal(err)
	}
}
