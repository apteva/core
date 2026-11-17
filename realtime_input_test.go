package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

type pressureRecoverySession struct {
	*recoveryTestSession
	pressure atomic.Bool
	calls    atomic.Int32
	failure  error
}

func (s *pressureRecoverySession) SendAudio(p []byte) error {
	s.calls.Add(1)
	if s.failure != nil {
		return s.failure
	}
	if s.pressure.Load() {
		return realtimeAdapterError(RealtimeErrorPressure, errors.New("test microphone capacity"))
	}
	return s.recoveryTestSession.SendAudio(p)
}

type delayedInputRecoveryProvider struct {
	fakeRealtimeProvider
	first, second    RealtimeSession
	entered, release chan struct{}
	count            atomic.Int32
	ignoreCancel     bool
}

func (p *delayedInputRecoveryProvider) Open(ctx context.Context, _ RealtimeSessionOpts) (RealtimeSession, error) {
	if p.count.Add(1) == 1 {
		return p.first, nil
	}
	close(p.entered)
	if p.ignoreCancel {
		<-p.release
		return p.second, nil
	}
	select {
	case <-p.release:
		return p.second, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
func recoveryEvents(rt *RealtimeThinker, kind string) []map[string]any {
	rt.telemetry.mu.Lock()
	defer rt.telemetry.mu.Unlock()
	var result []map[string]any
	for _, e := range rt.telemetry.log {
		if e.Type == kind {
			var d map[string]any
			_ = json.Unmarshal(e.Data, &d)
			result = append(result, d)
		}
	}
	return result
}
func TestRealtimeRecoveryIntakeContinuesDuringSlowOpen(t *testing.T) {
	first := &pressureRecoverySession{recoveryTestSession: newRecoveryTestSession(), failure: realtimeAdapterError(RealtimeErrorTransport, errors.New("original socket failure"))}
	second := newRecoveryTestSession()
	p := &delayedInputRecoveryProvider{first: first, second: second, entered: make(chan struct{}), release: make(chan struct{})}
	input := make(chan []byte, 6)
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	runRecoveryThinker(t, rt)
	input <- make([]byte, 960)
	select {
	case <-p.entered:
	case <-time.After(time.Second):
		t.Fatal("recovery not started")
	}
	var expected []byte
	for i := 1; i <= 80; i++ {
		frame := bytes.Repeat([]byte{byte(i), 0}, 480)
		expected = append(expected, frame...)
		select {
		case input <- frame:
		case <-time.After(100 * time.Millisecond):
			t.Fatal("reconnect blocked external audio intake")
		}
	}
	waitRecovery(t, func() bool { return rt.input.backlog() == 1600*time.Millisecond })
	// The actor still processes lifecycle controls while Open is blocked.
	rt.pause <- true
	waitRecovery(t, func() bool { return rt.inputPaused.Load() })
	rt.pause <- false
	waitRecovery(t, func() bool { return !rt.inputPaused.Load() })
	// Pausing explicitly discarded the first queue; send it again for ordering.
	for offset := 0; offset < len(expected); offset += 960 {
		input <- expected[offset : offset+960]
	}
	waitRecovery(t, func() bool { return rt.input.backlog() == 1600*time.Millisecond })
	close(p.release)
	waitRecovery(t, func() bool { second.mu.Lock(); defer second.mu.Unlock(); return len(second.audio) == len(expected) })
	second.mu.Lock()
	got := append([]byte(nil), second.audio...)
	second.mu.Unlock()
	if !bytes.Equal(got, expected) {
		t.Fatal("recovery changed ordering or duplicated admitted audio")
	}
	found := false
	for _, e := range recoveryEvents(rt, "realtime.error") {
		if e["error"] == "original socket failure" && e["error_class"] == "transport" {
			found = true
		}
	}
	if !found {
		t.Fatal("original adapter failure missing from telemetry")
	}
	for _, state := range []string{"recovering", "ready"} {
		found = false
		for _, e := range recoveryEvents(rt, "realtime.provider_state") {
			if e["state"] == state {
				found = true
			}
		}
		if !found {
			t.Fatalf("missing provider state %s", state)
		}
	}
}
func TestRealtimeInputTemporaryPressureRetriesWithoutReconnectOrDuplication(t *testing.T) {
	s := &pressureRecoverySession{recoveryTestSession: newRecoveryTestSession()}
	s.pressure.Store(true)
	p := &recoveryTestProvider{next: []RealtimeSession{s}}
	input := make(chan []byte, 6)
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	runRecoveryThinker(t, rt)
	var expected []byte
	for i := 0; i < 40; i++ {
		frame := bytes.Repeat([]byte{byte(i), 0}, 480)
		expected = append(expected, frame...)
		input <- frame
	}
	waitRecovery(t, func() bool { return s.calls.Load() >= 3 })
	if recoveryOpenCount(p) != 1 {
		t.Fatal("temporary pressure reconnected")
	}
	s.pressure.Store(false)
	waitRecovery(t, func() bool { s.mu.Lock(); defer s.mu.Unlock(); return len(s.audio) == len(expected) })
	s.mu.Lock()
	defer s.mu.Unlock()
	if !bytes.Equal(s.audio, expected) {
		t.Fatal("pressure retry duplicated or reordered audio")
	}
	if recoveryOpenCount(p) != 1 {
		t.Fatal("successful drain reconnected")
	}
}
func TestRealtimeInputDurationBoundsAndFreshness(t *testing.T) {
	for _, tc := range []struct {
		format    AudioFormat
		rate, bps int
	}{{AudioPCM16, 24000, 48000}, {AudioPCM16, 16000, 32000}, {AudioG711ULaw, 8000, 8000}, {AudioG711ALaw, 8000, 8000}} {
		t.Run(fmt.Sprint(tc.format, tc.rate), func(t *testing.T) {
			q := realtimeInputQueue{bytesPerSecond: tc.bps, policy: RealtimeInputPolicy{MaxBuffered: 100 * time.Millisecond, MaxAge: time.Second, FrameDuration: 20 * time.Millisecond}}
			now := time.Now()
			q.push(make([]byte, tc.bps/5), now)
			if q.backlog() != 100*time.Millisecond || q.drops["input_buffer_full"] != tc.bps/10 {
				t.Fatalf("duration bound: %v %+v", q.backlog(), q.drops)
			}
			q.push([]byte{1, 2}, now.Add(2*time.Second))
			if q.drops["stale_input"] != tc.bps/10 || q.bytes != 2 {
				t.Fatalf("stale input not replaced: %+v", q)
			}
		})
	}
}
func TestRealtimeInputContinuousRecoveryIsPaced(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.input.policy.Pace = true
	s := newRecoveryTestSession()
	rt.input.push(make([]byte, 3*960), time.Now())
	rt.drainRealtimeInput(s)
	rt.drainRealtimeInput(s)
	s.mu.Lock()
	n := len(s.audio)
	s.mu.Unlock()
	if n != 960 {
		t.Fatalf("burst drained %d bytes", n)
	}
	// Advance the scheduling deadline deterministically, retaining the frame duration.
	if time.Until(rt.input.nextSend) < 10*time.Millisecond {
		t.Fatal("missing real-time pacing deadline")
	}
	rt.input.nextSend = time.Time{}
	rt.drainRealtimeInput(s)
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.audio) != 1920 {
		t.Fatal("paced drain lost frame")
	}
}
func TestRealtimeInputSustainedPressureRecoversAndReports(t *testing.T) {
	rt, _ := completionFixture(t)
	s := &pressureRecoverySession{recoveryTestSession: newRecoveryTestSession()}
	s.pressure.Store(true)
	rt.replaceSession(s)
	rt.input.push(make([]byte, 960), time.Now())
	rt.input.pressureSince = time.Now().Add(-2 * time.Second)
	rt.drainRealtimeInput(s)
	if rt.currentSession() != nil || rt.input.backlog() != 20*time.Millisecond {
		t.Fatal("sustained pressure lost unadmitted input or did not recover")
	}
	found := false
	for _, e := range recoveryEvents(rt, "realtime.error") {
		if e["operation"] == "sustained_input_pressure" && e["error_class"] == "pressure" {
			found = true
		}
	}
	if !found {
		t.Fatal("no sustained pressure evidence")
	}
}
func TestRealtimeRecoveryLateCandidateClosedAfterStop(t *testing.T) {
	first, late := newRecoveryTestSession(), newRecoveryTestSession()
	p := &delayedInputRecoveryProvider{first: first, second: late, entered: make(chan struct{}), release: make(chan struct{}), ignoreCancel: true}
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	_ = first.Close()
	select {
	case <-p.entered:
	case <-time.After(time.Second):
		t.Fatal("no attempt")
	}
	rt.Thinker.Stop()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("stop waited for provider ignoring cancellation")
	}
	close(p.release)
	select {
	case <-late.events:
	case <-time.After(time.Second):
		t.Fatal("late candidate leaked")
	}
	if rt.currentSession() != nil {
		t.Fatal("late session installed after stop")
	}
}
func TestRealtimeRecoverySupersededAttemptCannotInstall(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.replaceSession(nil)
	a, err := rt.prepareSession(true)
	if err != nil {
		t.Fatal(err)
	}
	rt.opening = a
	rt.cancelOpening()
	late := newRecoveryTestSession()
	rt.finishOpening(realtimeOpenResult{attempt: a, session: late})
	if rt.currentSession() != nil {
		t.Fatal("superseded session installed")
	}
	select {
	case <-late.events:
	case <-time.After(time.Second):
		t.Fatal("superseded candidate not closed")
	}
}

type slowCloseInputSession struct {
	*recoveryTestSession
	release chan struct{}
}

func (s *slowCloseInputSession) Close() error { <-s.release; return s.recoveryTestSession.Close() }
func TestRealtimeRecoveryCloseDoesNotBlockRuntime(t *testing.T) {
	rt, _ := completionFixture(t)
	s := &slowCloseInputSession{recoveryTestSession: newRecoveryTestSession(), release: make(chan struct{})}
	rt.replaceSession(s)
	done := make(chan struct{})
	go func() { rt.closeForRecovery(s, false, "test_close"); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		close(s.release)
		t.Fatal("provider Close blocked runtime")
	}
	close(s.release)
}
func TestRealtimeAdapterErrorClassesPreserveCause(t *testing.T) {
	cause := errors.New("underlying cause")
	for _, class := range []RealtimeErrorClass{RealtimeErrorPressure, RealtimeErrorClosed, RealtimeErrorTransport, RealtimeErrorProtocol} {
		err := fmt.Errorf("wrapper: %w", realtimeAdapterError(class, cause))
		if realtimeErrorClass(err) != class || !errors.Is(err, cause) {
			t.Fatal("classification lost original cause")
		}
	}
}

func TestRealtimeAdapterQueuesClassifyPressureAndClosed(t *testing.T) {
	openai := &openaiRealtimeSession{outbox: make(chan realtimeOutboundFrame, 1), audioOutbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{})}
	google := &googleRealtimeSession{outbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{})}
	for name, send := range map[string]func() error{"openai_control": func() error { return openai.enqueue(map[string]any{"type": "test"}) }, "openai_audio": func() error { return openai.enqueueAudio(map[string]any{"type": "audio"}) }, "google": func() error { return google.SendAudio([]byte{0, 0}) }} {
		t.Run(name, func(t *testing.T) {
			if err := send(); err != nil {
				t.Fatal(err)
			}
			if err := send(); realtimeErrorClass(err) != RealtimeErrorPressure {
				t.Fatalf("not retry-safe pressure: %v", err)
			}
		})
	}
	close(openai.done)
	close(google.done)
	for _, send := range []func() error{func() error { return openai.enqueueAudio(map[string]any{}) }, func() error { return google.SendAudio([]byte{0, 0}) }} {
		if err := send(); realtimeErrorClass(err) != RealtimeErrorClosed {
			t.Fatalf("not closed: %v", err)
		}
	}
}
func TestRealtimeInputDropsHaveDurationAndAreNotRecounted(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.input.push(make([]byte, 6*realtimePCMBytesPerSecond), time.Now())
	rt.reportInputDrops()
	drops := recoveryEvents(rt, "realtime.audio_drop")
	if len(drops) != 1 || drops[0]["duration_ms"] != float64(1000) || drops[0]["discontinuity"] != true {
		t.Fatalf("drop evidence: %+v", drops)
	}
	rt.input.discard("test_cleanup")
	rt.reportInputDrops()
	rt.recordRealtimeInput(960)
	rt.queueRecoveryAudio(make([]byte, 960))
	s := newRecoveryTestSession()
	rt.replaceSession(s)
	rt.flushRecoveryInput()
	rt.recovery.Lock()
	n := rt.recovery.inputBytes
	rt.recovery.Unlock()
	if n != 960 {
		t.Fatalf("replay counted received audio twice: %d", n)
	}
}
func TestRealtimeProviderStateLaneCoalescesWithoutBlockingPlayback(t *testing.T) {
	rt, _ := completionFixture(t)
	control := make(chan string, 1)
	rt.audioControl = control
	control <- "interrupt"
	rt.providerState("recovering", nil)
	rt.providerState("failed", errors.New("test failure"))
	rt.providerState("ready", nil)
	if len(rt.providerStateWake) != 1 {
		t.Fatal("state wake should coalesce")
	}
	if <-control != "interrupt" {
		t.Fatal("state disrupted playback controls")
	}
	<-rt.providerStateWake
	var m map[string]any
	if err := json.Unmarshal([]byte(rt.providerStateControl()), &m); err != nil {
		t.Fatal(err)
	}
	if m["type"] != "session.state" || m["state"] != "ready" || m["session_generation"] != float64(1) {
		t.Fatalf("latest wire state %+v", m)
	}
}

type stalledRestoreInputSession struct {
	*recoveryTestSession
	entered, release chan struct{}
}

func (s *stalledRestoreInputSession) RestoreConversation([]Message) error {
	close(s.entered)
	<-s.release
	return nil
}
func TestRealtimeRecoveryRestoreDoesNotBlockIntakeOrStop(t *testing.T) {
	first := newRecoveryTestSession()
	second := &stalledRestoreInputSession{recoveryTestSession: newRecoveryTestSession(), entered: make(chan struct{}), release: make(chan struct{})}
	p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
	input := make(chan []byte, 6)
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	defer close(second.release)
	defer rt.cancel()
	_ = first.Close()
	select {
	case <-second.entered:
	case <-time.After(time.Second):
		t.Fatal("restore did not start")
	}
	for i := 0; i < 100; i++ {
		select {
		case input <- make([]byte, 960):
		case <-time.After(100 * time.Millisecond):
			t.Fatal("restore stalled external intake")
		}
	}
	waitRecovery(t, func() bool { return rt.input.backlog() == 2*time.Second })
	rt.Thinker.Stop()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("stop blocked on restore")
	}
}
func TestRealtimeRecoveryRepeatedFailuresBoundInputAndBackOff(t *testing.T) {
	first := newRecoveryTestSession()
	p := &recoveryTestProvider{next: []RealtimeSession{first}}
	input := make(chan []byte, 6)
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	runRecoveryThinker(t, rt)
	_ = first.Close()
	waitRecovery(t, func() bool { return rt.providerStateSnapshot() == "failed" })
	for i := 0; i < 300; i++ {
		select {
		case input <- make([]byte, 960):
		case <-time.After(100 * time.Millisecond):
			t.Fatal("failure backoff blocked intake")
		}
	}
	waitRecovery(t, func() bool {
		n := float64(0)
		for _, e := range recoveryEvents(rt, "realtime.audio_drop") {
			if e["reason"] == "input_buffer_full" {
				n += e["duration_ms"].(float64)
			}
		}
		return n == 1000
	})
	if rt.input.backlog() > 5*time.Second {
		t.Fatal("unbounded failed-session buffer")
	}
	if recoveryOpenCount(p) != 2 {
		t.Fatal("failed reconnect busy-looped instead of backing off")
	}
}

func TestRealtimeInputPauseDiscardsRacingAdmission(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.paused = true
	// Simulate intake already past its atomic pause check when pause committed.
	rt.queueRecoveryAudio(make([]byte, 960))
	s := newRecoveryTestSession()
	rt.drainRealtimeInput(s)
	if rt.input.pending() {
		t.Fatal("paused input left a runnable drain queue")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.audio) != 0 {
		t.Fatal("paused input sent")
	}
}
