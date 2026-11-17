package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func parentReportFixture(t *testing.T) (*Thinker, *Thread, *fakeRealtimeSession) {
	t.Helper()
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.config.path = filepath.Join(t.TempDir(), "config.json")
	parent.registry = NewToolRegistry("test")
	parent.config.RealtimeEnabled = true
	session := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{session}}
	parent.pool = &ProviderPool{providers: map[string]LLMProvider{"fireworks": parent.provider}, default_: "fireworks", realtimeProviders: map[string]RealtimeProvider{"fake-realtime": provider}, realtimeDefault: "fake-realtime"}
	if err := parent.threads.SpawnWithOpts("voice", "Answer politely", nil, SpawnOpts{Realtime: true, ProviderName: "fake-realtime", DeferRun: true, Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["voice"]
	thread.Realtime.replaceSession(session)
	thread.Realtime.sessionGeneration = 1
	t.Cleanup(func() { parent.Stop(); thread.Thinker.Stop(); thread.Realtime.cancel() })
	return parent, thread, session
}
func readParentReport(t *testing.T, parent *Thinker) (PersistentThreadEvent, realtimeParentReport) {
	t.Helper()
	var found []PersistentThreadEvent
	for _, ev := range parent.config.getMainEvents() {
		if strings.HasPrefix(ev.ID, "realtime-completion/") {
			found = append(found, ev)
		}
	}
	if len(found) != 1 {
		t.Fatalf("parent final reports=%d want 1", len(found))
	}
	var report realtimeParentReport
	_, body, ok := strings.Cut(found[0].Text, "\n")
	if !ok {
		t.Fatal("missing report body")
	}
	if err := json.Unmarshal([]byte(body), &report); err != nil {
		t.Fatal(err)
	}
	return found[0], report
}
func TestRealtimeParentReportTerminalPaths(t *testing.T) {
	for _, reason := range []string{"done_tool", "caller_done", "audio_disconnected", "transport_failure", "stopped", "server_shutdown"} {
		t.Run(reason, func(t *testing.T) {
			parent, thread, _ := parentReportFixture(t)
			rt := thread.Realtime
			rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Please call me tomorrow.", Final: false})
			if reason == "done_tool" {
				thread.Thinker.handleTools(thread.Thinker, []toolCall{{Name: "done", NativeID: "done-1", Args: map[string]string{"message": "Caller requested a callback; it is not booked."}}}, nil)
				if len(parent.config.getMainEvents()) != 0 {
					t.Fatal("done reported before terminal shutdown")
				}
			} else {
				rt.setTerminalReason(reason)
				thread.Thinker.Stop()
			}
			done := make(chan struct{})
			go func() { defer close(done); rt.Run() }()
			select {
			case <-done:
			case <-time.After(4 * time.Second):
				t.Fatal("realtime cleanup blocked")
			}
			ev, report := readParentReport(t, parent)
			if report.Reason != reason || report.TaskOutcome != "not_inferred_from_thread_termination" || report.ParentID != "main" {
				t.Fatalf("report=%+v", report)
			}
			if len(report.Recent) != 1 || report.Recent[0].Status != "interrupted" {
				t.Fatalf("partial caller evidence lost: %+v", report.Recent)
			}
			if reason == "done_tool" && !strings.Contains(report.ModelSummary, "not booked") {
				t.Fatal("lost done summary")
			}
			if parent.threads.Count() != 0 {
				t.Fatal("report prevented cleanup")
			}
			thread.reportRealtimeTermination()
			rt.finishShutdown()
			parent.threads.cleanupThreadInstance(thread.ID, thread.Thinker)
			again, _ := readParentReport(t, parent)
			if again.ID != ev.ID {
				t.Fatal("duplicate cleanup changed identity")
			}
			restored := &Config{path: parent.config.path}
			if err := restored.load(); err != nil {
				t.Fatal(err)
			}
			if len(restored.getMainEvents()) != 1 {
				t.Fatal("report not durable")
			}
		})
	}
}
func TestRealtimeParentReportRetryAfterRestart(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	// Saturate the parent's durable inbox. Reporting must not keep voice alive.
	blocked := make([]PersistentThreadEvent, 4096)
	for i := range blocked {
		blocked[i] = PersistentThreadEvent{ID: newULID(), Hash: "fixture", Text: "pending"}
	}
	parent.mainInboxEvents = blocked
	thread.Realtime.setTerminalReason("caller_done")
	thread.Thinker.Stop()
	thread.Realtime.Run()
	parent.Stop() // stop its retry loop before inspecting the crash snapshot
	parent.parentReportMu.Lock()
	parent.config.mu.RLock()
	pending := append([]PendingParentReport(nil), parent.config.ParentReports...)
	parent.config.mu.RUnlock()
	parent.parentReportMu.Unlock()
	if len(pending) != 1 {
		t.Fatalf("failed delivery not preserved: %d", len(pending))
	}
	restored := &Config{path: parent.config.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	next := newTestThinker()
	defer next.Stop()
	next.config = restored
	next.parentReportMu.Lock()
	err := next.flushParentReportsLocked()
	next.parentReportMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	ev, _ := readParentReport(t, next)
	if ev.ID != pending[0].Event.ID {
		t.Fatal("restart invented a second identity")
	}
	// Simulate a crash after parent acceptance but before outbox acknowledgement.
	if err := next.config.updateRuntime(func() { next.config.ParentReports = pending }); err != nil {
		t.Fatal(err)
	}
	next.parentReportMu.Lock()
	err = next.flushParentReportsLocked()
	next.parentReportMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	readParentReport(t, next)
	if len(next.config.ParentReports) != 0 {
		t.Fatal("outbox not acknowledged")
	}
}
func TestRealtimeParentReportPersistenceFailureRetried(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	path := parent.config.path
	if err := os.Mkdir(path+".runtime.jsonl", 0700); err != nil {
		t.Fatal(err)
	}
	thread.Realtime.setTerminalReason("transport_failure")
	thread.Thinker.Stop()
	thread.Realtime.Run()
	if parent.threads.Count() != 0 {
		t.Fatal("disk failure blocked voice cleanup")
	}
	parent.parentReportMu.Lock()
	if len(parent.parentReportPending) != 1 {
		t.Fatal("failed journal write discarded report")
	}
	if err := os.Remove(path + ".runtime.jsonl"); err != nil {
		t.Fatal(err)
	}
	err := parent.flushParentReportsLocked()
	parent.parentReportMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	readParentReport(t, parent)
}
func TestRealtimeParentReportNestedAndSpeechGuard(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	if err := parent.threads.SpawnWithOpts("leader", "Wait", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	leader := parent.threads.threads["leader"].Thinker
	thread.Parent = leader
	thread.ParentID = "leader"
	thread.Realtime.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ItemID: "leak", Transcript: `callto:private_tool{"secret":"not-for-parent"}`, Final: true})
	thread.Realtime.setTerminalReason("caller_done")
	thread.Thinker.Stop()
	thread.Realtime.Run()
	events := parent.threads.threads["leader"].pendingInboxEvents()
	if len(events) != 1 || !strings.HasPrefix(events[0].ID, "realtime-completion/") {
		t.Fatalf("nested report missing: %#v", events)
	}
	if strings.Contains(events[0].Text, "not-for-parent") {
		t.Fatal("speech guard bypassed")
	}
	if len(parent.config.getMainEvents()) != 0 {
		t.Fatal("nested report was sent to main")
	}
	leader.Stop()
}

func TestRealtimeParentReportNoFinalOnTemporaryDisconnect(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	thread.Realtime.audioBridgeDisconnected()
	thread.Realtime.handleSessionEvent(RealtimeEvent{Type: RealtimeEventSessionEnded, CloseReason: "renewal", Terminal: false})
	if len(parent.config.getMainEvents()) != 0 || len(parent.config.ParentReports) != 0 {
		t.Fatal("temporary interruption reported terminal completion")
	}
	thread.Realtime.audioBridgeConnected()
	if len(parent.config.getMainEvents()) != 0 {
		t.Fatal("reconnect reported completion")
	}
}

func TestRealtimeParentReportKeepsPendingAssignmentsAndRejectsOldTranscript(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	rt := thread.Realtime
	rt.messages = append(rt.messages, Message{Role: "assistant", Content: "An old call booked R-OLD", RealtimeTranscript: &RealtimeTranscriptMetadata{ID: "old-incarnation/output", Status: "provider_final"}})
	// A managed child has accepted work but is still blocked at its execution gate.
	rt.threads = NewThreadManager(rt.Thinker)
	state := &threadWorkState{}
	work := ThreadWork{ID: "pending-booking", Revision: 2, DelegationID: "delegation", Kind: "request"}
	state.advance(work)
	rt.threads.threads["backend"] = &Thread{ID: "backend", Thinker: &Thinker{work: state}}
	rt.persistExecutionMessage(Message{Role: "assistant", RealtimeBackend: true, Content: "Booking R42 was confirmed by the backend.", Work: &ThreadWork{ID: "completed", Kind: "result", Status: "completed"}}, TokenUsage{})
	rt.captureTerminalWork()
	delete(rt.threads.threads, "backend") // as when the execution child is cleaned up
	rt.setTerminalReason("caller_done")
	rt.finishShutdown()
	thread.reportRealtimeTermination()
	_, report := readParentReport(t, parent)
	if len(report.BackendReports) != 1 || !strings.Contains(report.BackendReports[0].Text, "R42") {
		t.Fatal("available backend outcome lost")
	}
	if len(report.Recent) != 0 {
		t.Fatal("previous incarnation leaked into final report")
	}
	pending, ok := report.Shutdown["pending_assignments"].([]any)
	if !ok || len(pending) != 1 || pending[0].(map[string]any)["id"] != "pending-booking" {
		t.Fatalf("pending assignment lost: %v", report.Shutdown)
	}
}

func TestRealtimeParentReportEphemeralParentRetainsDurableEvidence(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	if err := parent.threads.SpawnWithOpts("leader", "Wait", nil, SpawnOpts{DeferRun: true, Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	leaderThread := parent.threads.threads["leader"]
	leader := leaderThread.Thinker
	thread.Parent = leader
	thread.ParentID = "leader"
	thread.Realtime.setTerminalReason("caller_done")
	thread.Thinker.Stop()
	thread.Realtime.Run()
	parent.parentReportMu.Lock()
	if len(parent.config.ParentReports) != 1 {
		t.Fatal("memory-only parent acceptance discarded durable report")
	}
	report := parent.config.ParentReports[0]
	if err := leader.session.AppendMessage(Message{Role: "user", Content: report.Event.Text, EventIDs: []string{report.Event.ID}}, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if err := parent.flushParentReportsLocked(); err != nil {
		t.Fatal(err)
	}
	if len(parent.config.ParentReports) != 0 {
		t.Fatal("durably consumed report was not acknowledged")
	}
	parent.parentReportMu.Unlock()
	if len(leaderThread.pendingInboxEvents()) != 1 {
		t.Fatal("ephemeral retry duplicated report")
	}
	leader.Stop()
}

func TestRealtimeParentReportRecoveryRunsWhileRootPaused(t *testing.T) {
	parent := newTestThinker()
	t.Cleanup(parent.Stop)
	parent.config.path = filepath.Join(t.TempDir(), "config.json")
	text := "[thread:voice ended] recovered report\n" + `{"reason":"caller_done","parent_id":"main"}`
	report := PendingParentReport{ParentID: "main", Event: PersistentThreadEvent{ID: "realtime-completion/recovered", Text: text, Hash: threadEventHash(text, nil)}}
	if err := parent.config.updateRuntime(func() { parent.config.ParentReports = []PendingParentReport{report} }); err != nil {
		t.Fatal(err)
	}
	parent.threadID = "main"
	parent.execution = NewExecutionController(rootPausedExecutionConfig())
	gate, _ := startScopedGate(t, parent.execution, "main")
	waitForActiveThread(t, parent.execution, "main")
	parent.beginRuntime()
	defer parent.endRuntime()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("paused root did not retry its pending report")
		case <-parent.sub.Wake:
			if len(parent.config.getMainEvents()) == 0 {
				continue
			}
			readParentReport(t, parent)
			if parent.execution.Status().WaitingCount != 1 {
				t.Fatal("delivery released root gate")
			}
			parent.execution.CancelThread("main")
			requireGateResult(t, gate, false)
			return
		}
	}
}

func TestRealtimeParentReportAtomicInboxHandoff(t *testing.T) {
	parent := newTestThinker()
	defer parent.Stop()
	parent.config.path = filepath.Join(t.TempDir(), "config.json")
	event := PersistentThreadEvent{ID: "realtime-completion/atomic", Text: "terminal receipt", Hash: threadEventHash("terminal receipt", nil)}
	if err := parent.config.updateRuntime(func() { parent.config.ParentReports = []PendingParentReport{{ParentID: "main", Event: event}} }); err != nil {
		t.Fatal(err)
	}
	if _, err := parent.QueueMainEvents([]PersistentThreadEvent{event}); err != nil {
		t.Fatal(err)
	}
	restored := &Config{path: parent.config.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	if len(restored.ParentReports) != 0 || len(restored.getMainEvents()) != 1 {
		t.Fatal("inbox acceptance and outbox removal were not committed together")
	}
}

func TestRealtimeParentReportNewIncarnationUsesNewIdentity(t *testing.T) {
	parent, thread, _ := parentReportFixture(t)
	thread.Realtime.setTerminalReason("caller_done")
	thread.Thinker.Stop()
	thread.Realtime.Run()
	first, _ := readParentReport(t, parent)
	if err := parent.threads.SpawnWithOpts("voice", "Another call", nil, SpawnOpts{Realtime: true, ProviderName: "fake-realtime", DeferRun: true, Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	next := parent.threads.threads["voice"]
	next.Realtime.setTerminalReason("caller_done")
	next.Thinker.Stop()
	next.Realtime.Run()
	reports := parent.config.getMainEvents()
	if len(reports) != 2 || reports[0].ID != first.ID || reports[0].ID == reports[1].ID {
		t.Fatal("new call reused the previous terminal identity")
	}
}
