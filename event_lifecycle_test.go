package core

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func waitForLifecycleTypes(t *testing.T, lifecycle *EventLifecycle, want ...string) []EventLifecycleTransition {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		transitions := lifecycle.PendingTransitions()
		seen := map[string]bool{}
		for _, transition := range transitions {
			seen[transition.Type] = true
		}
		matched := true
		for _, eventType := range want {
			if !seen[eventType] {
				matched = false
				break
			}
		}
		if matched {
			return transitions
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("lifecycle never reached %v; got %#v", want, lifecycle.PendingTransitions())
	return nil
}

func TestTrackedThreadEventLifecycleDoesNotAddModelTurns(t *testing.T) {
	api, thinker, parked := newPersistentThreadTestAPI(t)
	parked.Release()
	provider := newRecordingThreadEventProvider()
	useThreadEventProvider(thinker, provider)
	thinker.eventLifecycle = NewEventLifecycle(thinker.config, thinker.telemetry)
	thinker.eventExecutionIDs = map[string]bool{}

	w := postThreadForTest(t, api, "tracked-worker", map[string]any{
		"directive": "Handle one event and then wait.",
		"events": []any{map[string]any{
			"id": "crm:message:42", "message": "Look up customer 42.", "track_lifecycle": true,
		}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status=%d body=%s", w.Code, w.Body.String())
	}
	response := decodeThreadEventResponse(t, w.Body.Bytes())
	events := response["events"].(map[string]any)
	executions := events["executions"].(map[string]any)
	executionID, _ := executions["crm:message:42"].(string)
	if executionID == "" {
		t.Fatalf("missing execution id: %#v", response)
	}

	request := waitThreadEventRequest(t, provider)
	if _, ok := messageContaining(request, "Look up customer 42."); !ok {
		t.Fatalf("tracked event missing from request: %#v", request)
	}
	for _, message := range request {
		if strings.Contains(message.TextContent(), executionID) {
			t.Fatal("execution bookkeeping leaked into model context")
		}
	}
	transitions := waitForLifecycleTypes(t, thinker.eventLifecycle, eventClaimed, eventActive, eventSettled)
	if provider.calls.Load() != 1 {
		t.Fatalf("tracking added model calls: got %d want 1", provider.calls.Load())
	}
	if len(transitions) != 3 {
		t.Fatalf("transitions=%#v", transitions)
	}
	for index, transition := range transitions {
		if transition.ExecutionID != executionID || transition.EventID != "crm:message:42" || transition.Sequence != uint64(index+1) {
			t.Fatalf("transition[%d]=%#v", index, transition)
		}
	}
	if transitions[2].Reason != "event_wait" {
		t.Fatalf("settled reason=%q", transitions[2].Reason)
	}
}

func newClaimedLifecycleForTest(t *testing.T, executionID string) (*Config, *EventLifecycle) {
	t.Helper()
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json")}
	event := PersistentThreadEvent{
		ID: "event-" + executionID, Text: "work", Hash: threadEventHash("work", nil),
		TrackLifecycle: true, ExecutionID: executionID,
	}
	if err := cfg.saveMainEventsAndRegisterExecutions([]PersistentThreadEvent{event}, []PersistentThreadEvent{event}); err != nil {
		t.Fatal(err)
	}
	lifecycle := NewEventLifecycle(cfg, nil)
	if err := lifecycle.Claim("main", []string{executionID}); err != nil {
		t.Fatal(err)
	}
	return cfg, lifecycle
}

func TestPendingToolPreventsEventSettlement(t *testing.T) {
	_, lifecycle := newClaimedLifecycleForTest(t, "exe-tool")
	thinker := &Thinker{
		threadID: "main", eventLifecycle: lifecycle,
		eventExecutionIDs: map[string]bool{"exe-tool": true},
	}
	thinker.pendingTools.Store("call-1", "slow_lookup")
	thinker.settleEventExecutions("event_wait")
	for _, transition := range lifecycle.PendingTransitions() {
		if transition.Type == eventSettled {
			t.Fatal("pending asynchronous tool allowed event settlement")
		}
	}
	if len(thinker.currentEventExecutions()) != 1 {
		t.Fatal("pending execution correlation was cleared")
	}
	thinker.pendingTools.Delete("call-1")
	thinker.asyncToolsActive.Store(1)
	thinker.settleEventExecutions("event_wait")
	for _, transition := range lifecycle.PendingTransitions() {
		if transition.Type == eventSettled {
			t.Fatal("anonymous asynchronous tool allowed event settlement")
		}
	}
	thinker.asyncToolsActive.Store(0)
	thinker.settleEventExecutions("event_wait")
	transitions := lifecycle.PendingTransitions()
	if transitions[len(transitions)-1].Type != eventSettled {
		t.Fatalf("execution did not settle after tool completion: %#v", transitions)
	}
}

func TestSendAndSpawnCarryEventExecutionCorrelation(t *testing.T) {
	_, lifecycle := newClaimedLifecycleForTest(t, "exe-handoff")
	parent := newTestThinkerFull()
	defer parent.Stop()
	parent.eventLifecycle = lifecycle
	parent.eventExecutionIDs = map[string]bool{"exe-handoff": true}
	if err := parent.threads.SpawnWithOpts("spawned-worker", "Do the delegated work.", nil, SpawnOpts{
		DeferRun: true, ExecutionIDs: []string{"exe-handoff"},
	}); err != nil {
		t.Fatal(err)
	}
	defer parent.threads.KillAll()
	worker := parent.threads.threads["spawned-worker"].Thinker
	if got := worker.currentEventExecutions(); len(got) != 1 || got[0] != "exe-handoff" {
		t.Fatalf("spawn correlation=%v", got)
	}

	// Drain Core's initial thread notification, then verify an ordinary direct
	// message carries the same opaque execution metadata without changing text.
	_ = parent.drainEvents()
	if !parent.threads.SendWithPartsExecution("spawned-worker", "continue", nil, []string{"exe-handoff"}) {
		t.Fatal("correlated send failed")
	}
	events := worker.drainEvents()
	if len(events) != 1 || events[0].Text != "continue" || len(events[0].ExecutionIDs) != 1 || events[0].ExecutionIDs[0] != "exe-handoff" {
		t.Fatalf("correlated event=%#v", events)
	}
	if err := lifecycle.SettleThread("main", []string{"exe-handoff"}, "event_wait"); err != nil {
		t.Fatal(err)
	}
	for _, transition := range lifecycle.PendingTransitions() {
		if transition.Type == eventSettled {
			t.Fatal("parent settled execution before spawned worker")
		}
	}
	worker.settleEventExecutions("event_wait")
	transitions := lifecycle.PendingTransitions()
	if transitions[len(transitions)-1].Type != eventSettled {
		t.Fatalf("worker did not settle causal execution: %#v", transitions)
	}
}

func TestUntrackedEventCreatesNoLifecycleState(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	thinker.eventLifecycle = NewEventLifecycle(thinker.config, thinker.telemetry)
	w := postThreadForTest(t, api, "untracked-worker", map[string]any{
		"events": []any{map[string]any{"id": "ordinary-event", "message": "ordinary work"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	thinker.config.mu.RLock()
	executions := len(thinker.config.EventExecutions)
	outbox := len(thinker.config.EventLifecycleOutbox)
	thinker.config.mu.RUnlock()
	if executions != 0 || outbox != 0 {
		t.Fatalf("untracked event created lifecycle state: executions=%d outbox=%d", executions, outbox)
	}
}

func TestEventLifecycleWaitsForCausalWorker(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: configFile}
	event := PersistentThreadEvent{ID: "evt-1", Text: "coordinate", TrackLifecycle: true}
	event.Hash = threadEventHash(event.Text, nil)
	event.ExecutionID = "exe-causal"
	if err := cfg.saveMainEventsAndRegisterExecutions([]PersistentThreadEvent{event}, []PersistentThreadEvent{event}); err != nil {
		t.Fatal(err)
	}
	lifecycle := NewEventLifecycle(cfg, nil)
	if err := lifecycle.Claim("main", []string{event.ExecutionID}); err != nil {
		t.Fatal(err)
	}
	if err := lifecycle.Propagate([]string{event.ExecutionID}, "worker"); err != nil {
		t.Fatal(err)
	}
	if err := lifecycle.SettleThread("main", []string{event.ExecutionID}, "event_wait"); err != nil {
		t.Fatal(err)
	}
	for _, transition := range lifecycle.PendingTransitions() {
		if transition.Type == eventSettled {
			t.Fatal("execution settled while causal worker was still active")
		}
	}
	if err := lifecycle.SettleThread("worker", []string{event.ExecutionID}, "event_wait"); err != nil {
		t.Fatal(err)
	}
	transitions := lifecycle.PendingTransitions()
	if got := transitions[len(transitions)-1]; got.Type != eventSettled || got.ExecutionID != event.ExecutionID {
		t.Fatalf("final transition=%#v", got)
	}
}

func TestPostEventTrackedMainIsDurableIdempotentAndAckable(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := newRecordingThreadEventProvider()
	cfg := &Config{path: configFile, Directive: "Handle incoming events.", Mode: ModeAutonomous}
	thinker := NewThinker("", provider, cfg)
	api := &APIServer{thinker: thinker}
	t.Cleanup(func() {
		thinker.threads.KillAll()
		thinker.Stop()
	})

	post := func(message string) *httptest.ResponseRecorder {
		body, _ := json.Marshal(map[string]any{
			"message": message, "event_id": "tasks:occurrence:9", "track_lifecycle": true,
		})
		req := httptest.NewRequest(http.MethodPost, "/event", bytes.NewReader(body))
		rec := httptest.NewRecorder()
		api.postEvent(rec, req)
		return rec
	}
	first := post("Process occurrence 9.")
	if first.Code != http.StatusOK {
		t.Fatalf("first status=%d body=%s", first.Code, first.Body.String())
	}
	firstResponse := decodeThreadEventResponse(t, first.Body.Bytes())
	firstEvents := firstResponse["events"].(map[string]any)
	executionID := firstEvents["executions"].(map[string]any)["tasks:occurrence:9"].(string)
	if executionID == "" {
		t.Fatal("missing main execution id")
	}

	duplicate := post("Process occurrence 9.")
	if duplicate.Code != http.StatusOK {
		t.Fatalf("duplicate status=%d body=%s", duplicate.Code, duplicate.Body.String())
	}
	duplicateEvents := decodeThreadEventResponse(t, duplicate.Body.Bytes())["events"].(map[string]any)
	if got := duplicateEvents["executions"].(map[string]any)["tasks:occurrence:9"].(string); got != executionID {
		t.Fatalf("duplicate execution=%q want %q", got, executionID)
	}

	go thinker.Run()
	request := waitThreadEventRequest(t, provider)
	if _, ok := messageContaining(request, "Process occurrence 9."); !ok {
		t.Fatalf("main request lost event: %#v", request)
	}
	transitions := waitForLifecycleTypes(t, thinker.eventLifecycle, eventSettled)
	ids := make([]string, 0, len(transitions))
	for _, transition := range transitions {
		ids = append(ids, transition.ID)
	}
	body, _ := json.Marshal(map[string]any{"ack_ids": ids})
	req := httptest.NewRequest(http.MethodPost, "/event-lifecycle", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	api.eventLifecycle(rec, req)
	if rec.Code != http.StatusOK || len(thinker.eventLifecycle.PendingTransitions()) != 0 {
		t.Fatalf("ack status=%d body=%s remaining=%#v", rec.Code, rec.Body.String(), thinker.eventLifecycle.PendingTransitions())
	}
	// A repeated acknowledgement is harmless and cannot recreate delivery.
	body, _ = json.Marshal(map[string]any{"ack_ids": ids})
	req = httptest.NewRequest(http.MethodPost, "/event-lifecycle", bytes.NewReader(body))
	rec = httptest.NewRecorder()
	api.eventLifecycle(rec, req)
	if rec.Code != http.StatusOK || len(thinker.eventLifecycle.PendingTransitions()) != 0 {
		t.Fatalf("repeated ack status=%d remaining=%#v", rec.Code, thinker.eventLifecycle.PendingTransitions())
	}
	thinker.config.mu.RLock()
	retainedExecutions := len(thinker.config.EventExecutions)
	thinker.config.mu.RUnlock()
	if retainedExecutions != 0 {
		t.Fatalf("acknowledged terminal execution was retained: %d", retainedExecutions)
	}
	afterAckDuplicate := post("Process occurrence 9.")
	afterAckEvents := decodeThreadEventResponse(t, afterAckDuplicate.Body.Bytes())["events"].(map[string]any)
	if got := afterAckEvents["executions"].(map[string]any)["tasks:occurrence:9"].(string); got != executionID {
		t.Fatalf("post-ack duplicate execution=%q want %q", got, executionID)
	}
	if len(thinker.eventLifecycle.PendingTransitions()) != 0 {
		t.Fatal("post-ack duplicate recreated lifecycle transitions")
	}
}

func TestEventLifecycleOutboxAndActiveExecutionSurviveRestart(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: configFile}
	event := PersistentThreadEvent{ID: "restart-event", Text: "continue", Hash: threadEventHash("continue", nil), TrackLifecycle: true, ExecutionID: "exe-restart"}
	if err := cfg.saveMainEventsAndRegisterExecutions([]PersistentThreadEvent{event}, []PersistentThreadEvent{event}); err != nil {
		t.Fatal(err)
	}
	lifecycle := NewEventLifecycle(cfg, nil)
	if err := lifecycle.Claim("main", []string{event.ExecutionID}); err != nil {
		t.Fatal(err)
	}

	reloaded := NewConfig()
	restarted := NewEventLifecycle(reloaded, nil)
	active := restarted.ActiveForThread("main")
	if len(active) != 1 || active[0] != event.ExecutionID {
		t.Fatalf("active after restart=%v", active)
	}
	transitions := restarted.PendingTransitions()
	if len(transitions) != 2 || transitions[0].Type != eventClaimed || transitions[1].Type != eventActive {
		t.Fatalf("outbox after restart=%#v", transitions)
	}
}

func TestEventLifecycleRestartErrorsWhenEphemeralParticipantDisappears(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: configFile, Directive: "Handle tracked work.", Mode: ModeAutonomous}
	event := PersistentThreadEvent{
		ID: "ephemeral-restart-event", Text: "coordinate temporary work",
		Hash:           threadEventHash("coordinate temporary work", nil),
		TrackLifecycle: true, ExecutionID: "exe-ephemeral-restart",
	}
	if err := cfg.saveMainEventsAndRegisterExecutions([]PersistentThreadEvent{event}, []PersistentThreadEvent{event}); err != nil {
		t.Fatal(err)
	}
	provider := newRecordingThreadEventProvider()
	parent := NewThinker("", provider, cfg)
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.Stop()
	})
	if err := parent.eventLifecycle.Claim("main", []string{event.ExecutionID}); err != nil {
		t.Fatal(err)
	}
	if err := parent.threads.SpawnWithOpts("temporary-worker", "Handle the temporary step.", nil, SpawnOpts{
		DeferRun: true, Ephemeral: true,
	}); err != nil {
		t.Fatal(err)
	}
	if !parent.threads.SendWithPartsExecution("temporary-worker", "continue", nil, []string{event.ExecutionID}) {
		t.Fatal("tracked send to ephemeral worker failed")
	}

	cfg.mu.RLock()
	if len(cfg.Threads) != 0 {
		cfg.mu.RUnlock()
		t.Fatalf("ephemeral worker was persisted: %#v", cfg.Threads)
	}
	if got := cfg.EventExecutions[0].Participants["temporary-worker"]; !got {
		cfg.mu.RUnlock()
		t.Fatal("ephemeral worker was not recorded as an active participant")
	}
	cfg.mu.RUnlock()

	// Constructing a new thinker from the same configuration models Core
	// restart. The ephemeral worker is absent from the restored runtime tree.
	reloaded := NewConfig()
	restarted := NewThinker("", newRecordingThreadEventProvider(), reloaded)
	t.Cleanup(func() {
		restarted.threads.KillAll()
		restarted.Stop()
	})
	transitions := restarted.eventLifecycle.PendingTransitions()
	last := transitions[len(transitions)-1]
	if last.Type != eventError || last.ExecutionID != event.ExecutionID || last.Reason != "participant_not_restored" {
		t.Fatalf("restart transition=%#v", last)
	}
	if active := restarted.eventLifecycle.ActiveForThread("main"); len(active) != 0 {
		t.Fatalf("terminal execution remained active after restart: %v", active)
	}
}

func TestEventLifecycleRestartKeepsRestoredParticipantActive(t *testing.T) {
	_, lifecycle := newClaimedLifecycleForTest(t, "exe-restored-worker")
	if err := lifecycle.Propagate([]string{"exe-restored-worker"}, "durable-worker"); err != nil {
		t.Fatal(err)
	}
	failed, err := lifecycle.ReconcileRestoredParticipants(map[string]bool{
		"main": true, "durable-worker": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(failed) != 0 {
		t.Fatalf("restored participant failed reconciliation: %v", failed)
	}
	for _, transition := range lifecycle.PendingTransitions() {
		if transition.Type == eventError {
			t.Fatalf("restored participant emitted error: %#v", transition)
		}
	}
}

func TestEventLifecycleTerminalErrorIsDurable(t *testing.T) {
	_, lifecycle := newClaimedLifecycleForTest(t, "exe-cancelled")
	if err := lifecycle.Fail([]string{"exe-cancelled"}, "main", "cancelled"); err != nil {
		t.Fatal(err)
	}
	transitions := lifecycle.PendingTransitions()
	last := transitions[len(transitions)-1]
	if last.Type != eventError || last.Reason != "cancelled" || last.ExecutionID != "exe-cancelled" {
		t.Fatalf("terminal transition=%#v", last)
	}
}
