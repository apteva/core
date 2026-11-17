package core

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

func completionFixture(t *testing.T) (*RealtimeThinker, *fakeRealtimeSession) {
	t.Helper()
	thinker := newTestThinker()
	thinker.session = NewSession(t.TempDir(), "live")
	session := newFakeRealtimeSession()
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, make(chan RealtimeAudioFrame, 8), nil)
	rt.sessionGeneration = 1
	rt.timing = newRealtimeTiming(1, rt.emit)
	rt.bridgeConnected = true
	rt.replaceSession(session)
	t.Cleanup(func() { thinker.Stop(); rt.cancel() })
	return rt, session
}

func runCompletionFixture(rt *RealtimeThinker) <-chan struct{} {
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	return done
}

func waitCompletionFixture(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("realtime worker did not finish")
	}
}

func assertCompletionStillRunning(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
		t.Fatal("worker stopped before output was drained/acknowledged")
	case <-time.After(150 * time.Millisecond):
	}
}

func shutdownReceipt(t *testing.T, rt *RealtimeThinker) map[string]any {
	t.Helper()
	rt.telemetry.mu.Lock()
	defer rt.telemetry.mu.Unlock()
	var result map[string]any
	count := 0
	for _, event := range rt.telemetry.log {
		if event.Type == "realtime.thread_ended" {
			count++
			if err := json.Unmarshal(event.Data, &result); err != nil {
				t.Fatal(err)
			}
		}
	}
	if count != 1 {
		t.Fatalf("shutdown receipts = %d", count)
	}
	return result
}

func TestRealtimeCompletionDrainsQueuedAndInFlightAudioAndPlayback(t *testing.T) {
	rt, session := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "reply", ItemID: "speech", Audio: make([]byte, 4800)})
	rt.acknowledgePlayback("speech", 0) // renderer advertises acknowledgement support
	rt.requestCompletion("done-1")
	done := runCompletionFixture(rt)
	session.events <- RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "Goodbye.", Final: true}
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "reply"}
	assertCompletionStillRunning(t, done)
	frame := <-rt.audioOut
	assertCompletionStillRunning(t, done) // dequeue is not proof that the socket write finished
	frame.written(true)
	assertCompletionStillRunning(t, done) // socket write is not proof of playback
	rt.acknowledgePlayback("speech", 100)
	waitCompletionFixture(t, done)
	receipt := shutdownReceipt(t, rt)
	if receipt["graceful_completion_outcome"] != "completed" || receipt["trigger"] != "done_tool" || receipt["tool_call_id"] != "done-1" || receipt["acknowledged_audio_ms"] != float64(100) {
		t.Fatalf("receipt: %+v", receipt)
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.responses != 0 {
		t.Fatalf("finishing requested %d new responses", session.responses)
	}
}

func TestRealtimeCompletionFinalTranscriptUpdatesInterruptedTurn(t *testing.T) {
	rt, session := completionFixture(t)
	partial := RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "Good"}
	rt.handleSessionEvent(partial)
	rt.flushPendingTranscripts("bridge_disconnected")
	before := readSessionEntriesForTest(t, rt.Thinker.session.path)
	rt.requestCompletion("done-2")
	done := runCompletionFixture(rt)
	partial.Transcript = "Goodbye."
	partial.Final = true
	session.events <- partial
	session.events <- partial // duplicate provider final
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "reply"}
	waitCompletionFixture(t, done)
	rt.finishShutdown() // repeated cleanup is idempotent
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || len(before) != 1 || entries[0].RealtimeTranscript.ID != before[0].RealtimeTranscript.ID || entries[0].Content != "Goodbye." || entries[0].RealtimeTranscript.Status != "provider_final" {
		t.Fatalf("before=%+v after=%+v", before, entries)
	}
	updates, cursor, err := rt.Thinker.session.LoadAfter(before[0].Sequence, 1)
	if err != nil || len(updates) != 1 || cursor <= before[0].Sequence {
		t.Fatalf("revised transcript was not visible to cursor reader: %v %+v", err, updates)
	}
	shutdownReceipt(t, rt)
}

func TestRealtimeShutdownPreservesPartialCallerAndDeltas(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Please ", TranscriptDelta: true})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "help", TranscriptDelta: true})
	rt.audioBridgeDisconnected()
	rt.setTerminalReason("audio_disconnected")
	rt.Stop()
	done := runCompletionFixture(rt)
	waitCompletionFixture(t, done)
	rt.finishShutdown()
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || entries[0].Content != "Please help" || entries[0].RealtimeTranscript.Status != "interrupted" || entries[0].RealtimeTranscript.FinalizationReason != "audio_disconnected" {
		t.Fatalf("interrupted caller: %+v", entries)
	}
	receipt := shutdownReceipt(t, rt)
	if receipt["trigger"] != "audio_disconnected" || receipt["pending_transcripts"] != float64(1) {
		t.Fatalf("receipt: %+v", receipt)
	}
}

func TestRealtimeForcedStopDoesNotWaitForQueuedAudio(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "reply", ItemID: "speech", Audio: make([]byte, 4800)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "Safe unfinished speech"})
	rt.requestCompletion("done-3")
	done := runCompletionFixture(rt)
	rt.setTerminalReason("explicit_stop")
	rt.Stop()
	select {
	case <-done:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("forced stop waited for finishing deadline")
	}
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || entries[0].RealtimeTranscript.Status != "interrupted" {
		t.Fatalf("partial transcript: %+v", entries)
	}
	receipt := shutdownReceipt(t, rt)
	if receipt["trigger"] != "explicit_stop" || receipt["graceful_completion_outcome"] != "forced_stop" || receipt["pending_audio_ms"] != float64(100) {
		t.Fatalf("receipt: %+v", receipt)
	}
}

func TestRealtimeCompletionTimesOutAndRejectsNewTools(t *testing.T) {
	rt, session := completionFixture(t)
	rt.responseStarted()
	rt.completion.timeout = 40 * time.Millisecond
	rt.requestCompletion("done-4")
	rt.handleTools = func(*Thinker, []toolCall, []string) ([]string, []string, []ToolResult) {
		t.Error("new tool executed while finishing")
		return nil, nil, nil
	}
	session.events <- RealtimeEvent{Type: RealtimeEventToolCall, ToolCallID: "new", ToolName: "pace"}
	done := runCompletionFixture(rt)
	waitCompletionFixture(t, done)
	if shutdownReceipt(t, rt)["graceful_completion_outcome"] != "timed_out" {
		t.Fatal("missing timeout outcome")
	}
}

func TestRealtimePartialTranscriptCannotBypassSpeechGuard(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "call"})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "to:secret_tool{}", TranscriptDelta: true})
	rt.flushPendingTranscripts("explicit_stop")
	if _, err := os.Stat(rt.Thinker.session.path); !os.IsNotExist(err) {
		t.Fatal("blocked assistant transcript entered durable history")
	}
}

func TestRealtimeTranscriptGenerationSeparatesReusedProviderIDs(t *testing.T) {
	rt, _ := completionFixture(t)
	event := RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "reused", Transcript: "First", Final: true}
	rt.handleSessionEvent(event)
	rt.sessionGeneration = 2
	event.Transcript = "Second"
	rt.handleSessionEvent(event)
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 2 || entries[0].RealtimeTranscript.ID == entries[1].RealtimeTranscript.ID {
		t.Fatalf("generation collision: %+v", entries)
	}
}

func TestRealtimeDoneToolFinishesBeforeCleanupAndRetainsEphemeralHistory(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinkerFull()
	defer parent.Stop()
	parent.config.RealtimeEnabled = true
	provider := &fakeRealtimeProvider{}
	parent.pool = &ProviderPool{
		providers: map[string]LLMProvider{"fireworks": parent.provider}, order: []string{"fireworks"}, default_: "fireworks",
		realtimeProviders: map[string]RealtimeProvider{provider.Name(): provider}, realtimeOrder: []string{provider.Name()}, realtimeDefault: provider.Name(),
	}
	if err := parent.threads.SpawnWithOpts("voice", "Finish politely.", nil, SpawnOpts{Realtime: true, Ephemeral: true, ProviderName: provider.Name(), DeferRun: true, AudioOut: make(chan RealtimeAudioFrame, 8)}); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["voice"]
	rt := worker.Realtime
	if err := rt.openSession(false); err != nil {
		t.Fatal(err)
	}
	parent.threads.realtimeBridgeConnected("voice")
	session := rt.currentSession().(*fakeRealtimeSession)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "last", ItemID: "last", Audio: make([]byte, 4800)})
	_, _, results := threadToolHandler(worker, parent.threads)(worker.Thinker, []toolCall{{Name: "done", NativeID: "actual-done", Args: map[string]string{"message": "Call complete"}}}, nil)
	if len(results) != 1 || results[0].IsError || !rt.completionRequested() {
		t.Fatalf("done did not request finishing: %+v", results)
	}
	select {
	case <-rt.quit:
		t.Fatal("done stopped worker immediately")
	default:
	}
	done := runCompletionFixture(rt)
	session.events <- RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "last", ItemID: "last", Transcript: "Goodbye.", Final: true}
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "last"}
	assertCompletionStillRunning(t, done)
	frame := <-worker.audioOut
	frame.written(true)
	waitCompletionFixture(t, done)
	if parent.threads.Count() != 0 {
		t.Fatal("completed live worker was not cleaned up")
	}
	entries := readSessionEntriesForTest(t, worker.Thinker.session.path)
	found := false
	for _, entry := range entries {
		if entry.Content == "Goodbye." && entry.RealtimeTranscript != nil && entry.RealtimeTranscript.Status == "provider_final" {
			found = true
		}
	}
	if !found {
		t.Fatalf("cleanup lost ephemeral realtime transcript: %+v", entries)
	}
	receipt := shutdownReceipt(t, rt)
	if receipt["graceful_completion_outcome"] != "completed" {
		t.Fatalf("shutdown receipt: %+v", receipt)
	}
}

func TestRealtimeRevisedTranscriptPaginationDoesNotSkipOtherTurns(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "first", Transcript: "First partial"})
	rt.flushPendingTranscripts("disconnect")
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ItemID: "second", Transcript: "Second", Final: true})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "first", Transcript: "First final", Final: true})
	updates, cursor, err := rt.Thinker.session.LoadAfter(1, 1)
	if err != nil || len(updates) != 1 || updates[0].Content != "Second" {
		t.Fatalf("first page: %v %+v", err, updates)
	}
	updates, _, err = rt.Thinker.session.LoadAfter(cursor, 1)
	if err != nil || len(updates) != 1 || updates[0].Content != "First final" {
		t.Fatalf("revision page: %v %+v", err, updates)
	}
	loaded, _ := rt.Thinker.session.LoadTail(10)
	if len(loaded) != 2 || loaded[0].Content != "First final" || loaded[0].RealtimeTranscript == nil {
		t.Fatalf("history chronology: %+v", loaded)
	}
	raw, err := json.Marshal(loaded[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err = json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	if _, ok := wire["realtime_transcript"]; ok {
		t.Fatal("provider message exposes internal history metadata")
	}
}

func TestRealtimeLateGuardBlockRetractsSavedPartialText(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "call"})
	rt.flushPendingTranscripts("bridge_disconnected")
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "reply", ItemID: "speech", Transcript: "to:secret_tool{}", TranscriptDelta: true})
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || entries[0].Content != "" || entries[0].RealtimeTranscript.Status != "blocked" {
		t.Fatalf("saved partial was not retracted: %+v", entries)
	}
}

func TestRealtimeFinalTranscriptPersistenceFailureRetriesOnShutdown(t *testing.T) {
	rt, _ := completionFixture(t)
	original := rt.Thinker.session.path
	rt.Thinker.session.path = t.TempDir()
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Keep this evidence", Final: true})
	rt.Thinker.session.path = original
	rt.finishShutdown()
	entries := readSessionEntriesForTest(t, original)
	if len(entries) != 1 || entries[0].RealtimeTranscript.Status != "provider_final" || entries[0].Content != "Keep this evidence" {
		t.Fatalf("final was lost after persistence failure: %+v", entries)
	}
}

func TestRealtimeCompletionWithoutPlaybackSupportUsesWriteReceipts(t *testing.T) {
	rt, session := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "reply", ItemID: "speech", Audio: make([]byte, 4800)})
	rt.requestCompletion("done-no-ack")
	done := runCompletionFixture(rt)
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "reply"}
	frame := <-rt.audioOut
	frame.written(true)
	waitCompletionFixture(t, done)
	receipt := shutdownReceipt(t, rt)
	if receipt["graceful_completion_outcome"] != "completed" || receipt["playback_ack_supported"] != false || receipt["acknowledged_audio_ms"] != float64(0) || receipt["bridge_written_audio_ms"] != float64(100) {
		t.Fatalf("write receipt invents playback: %+v", receipt)
	}
}

func TestRealtimeCompletionDisconnectedDoesNotDrainToMissingCaller(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "reply", ItemID: "speech", Audio: make([]byte, 4800)})
	rt.audioBridgeDisconnected()
	rt.requestCompletion("done-disconnected")
	done := runCompletionFixture(rt)
	select {
	case <-done:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("disconnected completion waited for audio")
	}
	receipt := shutdownReceipt(t, rt)
	if receipt["graceful_completion_outcome"] != "disconnected" || receipt["acknowledged_audio_ms"] != float64(0) {
		t.Fatalf("receipt: %+v", receipt)
	}
}

func TestRealtimeInterruptedPlaybackDoesNotBlockLaterCompletion(t *testing.T) {
	rt, session := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "old", ItemID: "old", Audio: make([]byte, 4800)})
	rt.acknowledgePlayback("old", 0)
	frame := <-rt.audioOut
	frame.written(true)
	rt.interruptPlayback("caller_barge_in", "old", false, false)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "new", ItemID: "new", Audio: make([]byte, 4800)})
	rt.requestCompletion("done-new")
	done := runCompletionFixture(rt)
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "new"}
	frame = <-rt.audioOut
	frame.written(true)
	rt.acknowledgePlayback("new", 100)
	waitCompletionFixture(t, done)
	if shutdownReceipt(t, rt)["graceful_completion_outcome"] != "completed" {
		t.Fatal("old discarded speech prevented completion")
	}
}

func TestRealtimePlaybackReadyWaitsForFirstUtteranceAcknowledgement(t *testing.T) {
	rt, session := completionFixture(t)
	rt.delivery.playbackReady()
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "first", ItemID: "first", Audio: make([]byte, 4800)})
	rt.requestCompletion("done-first")
	done := runCompletionFixture(rt)
	session.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "first"}
	frame := <-rt.audioOut
	frame.written(true)
	assertCompletionStillRunning(t, done)
	rt.acknowledgePlayback("first", 100)
	waitCompletionFixture(t, done)
	if shutdownReceipt(t, rt)["acknowledged_audio_ms"] != float64(100) {
		t.Fatal("missing first utterance receipt")
	}
}

func TestRealtimeShutdownReceiptIsDurableBeforeCleanup(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.telemetry.durableDir = t.TempDir()
	called := false
	rt.onStop = func() {
		files, err := os.ReadDir(rt.telemetry.durableDir)
		if err != nil || len(files) != 1 {
			t.Errorf("cleanup ran without durable terminal receipt: files=%d err=%v", len(files), err)
		}
		called = true
	}
	rt.setTerminalReason("server_shutdown")
	rt.Stop()
	done := runCompletionFixture(rt)
	waitCompletionFixture(t, done)
	if !called {
		t.Fatal("cleanup callback did not run")
	}
	shutdownReceipt(t, rt)
}

func TestRealtimeExplicitSessionRestartPreservesPendingTranscript(t *testing.T) {
	rt, _ := completionFixture(t)
	instructions, tools := rt.configurationSnapshot()
	session := &immutableRealtimeSession{fakeRealtimeSession: newFakeRealtimeSession(), fingerprint: googleRealtimeConfigFingerprint(instructions, tools)}
	rt.replaceSession(session)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Pending caller words"})
	rt.transcriptMu.Lock()
	rt.messages[0].Content += " Updated configuration."
	rt.transcriptMu.Unlock()
	restarted, err := rt.applyExternalConfigurationChange(true, "external_configuration_update")
	if err != nil || !restarted {
		t.Fatalf("restart: %v %v", restarted, err)
	}
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || entries[0].RealtimeTranscript.Status != "interrupted" || entries[0].RealtimeTranscript.FinalizationReason != "external_configuration_update" {
		t.Fatalf("restart transcript: %+v", entries)
	}
	events, _ := rt.telemetry.StoredEvents(0)
	found := false
	for _, event := range events {
		if event.Type == "realtime.session_closed" {
			var data map[string]any
			if err := json.Unmarshal(event.Data, &data); err != nil {
				t.Fatal(err)
			}
			if data["trigger"] == "external_configuration_update" && data["pending_transcripts"] == float64(1) {
				found = true
			}
		}
	}
	if !found {
		t.Fatal("restart did not record a reason and pending transcription")
	}
}
