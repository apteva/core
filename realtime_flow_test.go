package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gobwas/ws/wsutil"
)

// The actor is deliberately blocked inside durable transcript persistence.
// Both directions must continue on the audio lane, including the first reply.
func TestRealtimeAudioFlowsWhileTranscriptPersistenceBlocked(t *testing.T) {
	rt, _ := completionFixture(t)
	session := newRecoveryTestSession()
	rt.replaceSession(session)
	input := make(chan []byte, 64)
	rt.audioIn = input
	done := runCompletionFixture(rt)
	defer func() { rt.cancel(); waitCompletionFixture(t, done) }()
	waitRecovery(t, func() bool { return rt.audioLane.active.Load() })
	rt.Thinker.session.mu.Lock()
	locked := true
	defer func() {
		if locked {
			rt.Thinker.session.mu.Unlock()
		}
	}()
	session.events <- RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Bonjour.", Final: true}
	waitRecovery(t, func() bool {
		if rt.transcriptMu.TryLock() {
			rt.transcriptMu.Unlock()
			return false
		}
		return true
	})
	var expected []byte
	for i := 0; i < 30; i++ {
		audio := bytes.Repeat([]byte{byte(i), 0}, 480)
		expected = append(expected, audio...)
		input <- audio
	}
	for _, item := range []string{"first", "later"} {
		session.events <- RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: item, Audio: make([]byte, 960)}
		select {
		case f := <-rt.audioOut:
			if f.ItemID != item {
				t.Fatal(f.ItemID)
			}
			f.written(true)
		case <-time.After(time.Second):
			t.Fatal("persistence blocked output", item)
		}
	}
	waitRecovery(t, func() bool { session.mu.Lock(); defer session.mu.Unlock(); return bytes.Equal(session.audio, expected) })
	rt.Thinker.session.mu.Unlock()
	locked = false
}

func TestRealtimeAudioFlowsWhileTelemetryBlocked(t *testing.T) {
	rt, _ := completionFixture(t)
	session := newRecoveryTestSession()
	rt.replaceSession(session)
	done := runCompletionFixture(rt)
	defer func() { rt.cancel(); waitCompletionFixture(t, done) }()
	waitRecovery(t, func() bool { return rt.audioLane.active.Load() })
	rt.telemetry.mu.Lock()
	locked := true
	defer func() {
		if locked {
			rt.telemetry.mu.Unlock()
		}
	}()
	for i := 0; i < 40; i++ {
		rt.emit("realtime.test", map[string]any{"i": i})
	}
	rt.admitRealtimeAudio(make([]byte, 960), time.Now())
	session.events <- RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "speech", Audio: make([]byte, 960)}
	select {
	case f := <-rt.audioOut:
		f.written(true)
	case <-time.After(time.Second):
		t.Fatal("telemetry blocked output")
	}
	waitRecovery(t, func() bool { session.mu.Lock(); defer session.mu.Unlock(); return len(session.audio) == 960 })
	rt.telemetry.mu.Unlock()
	locked = false
}

func TestRealtimeSharedInputBudgetIncludesAdapterQueue(t *testing.T) {
	for _, provider := range []string{"openai", "google", "live"} {
		t.Run(provider, func(t *testing.T) {
			rt, _ := completionFixture(t)
			base := &openaiRealtimeSession{outbox: make(chan realtimeOutboundFrame, 512), done: make(chan struct{})}
			google := &googleRealtimeSession{outbox: make(chan realtimeOutboundFrame, 512), done: make(chan struct{}), inputRate: 24000}
			var s RealtimeSession = base
			switch provider {
			case "google":
				s = google
			case "live":
				s = &openaiLiveSession{openaiRealtimeSession: base}
			}
			rt.replaceSession(s)
			for i := 0; i < 251; i++ {
				rt.input.push(make([]byte, 960), time.Now())
				rt.drainRealtimeInput(s)
			}
			if got := rt.input.backlog(); got != 5*time.Second {
				t.Fatalf("total backlog=%s", got)
			}
			rt.input.mu.Lock()
			dropped := rt.input.drops["input_buffer_full"]
			rt.input.mu.Unlock()
			if dropped != 960 {
				t.Fatalf("excess admission=%d", dropped)
			}
			_ = s.Close()
			if got := rt.input.backlog(); got != 0 {
				t.Fatalf("close leaked reservation: %s", got)
			}
		})
	}
}

func TestRealtimeInputReservationReleasedAfterTransportWrite(t *testing.T) {
	conn, peer := net.Pipe()
	defer peer.Close()
	s := &openaiRealtimeSession{conn: conn, outbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}), events: make(chan RealtimeEvent, 8), writeTimeout: time.Second}
	defer s.Close()
	var completed atomic.Int32
	err := s.SendAudioFrame(RealtimeInputAudio{Audio: make([]byte, 960), ReceivedAt: time.Now(), MaxAge: time.Second, Complete: func(wait, write time.Duration, success bool) {
		if !success {
			t.Error("write failed")
		}
		completed.Add(1)
	}})
	if err != nil {
		t.Fatal(err)
	}
	if completed.Load() != 0 {
		t.Fatal("admission claimed delivery")
	}
	go s.writeLoop()
	if _, _, err := wsutil.ReadClientData(peer); err != nil {
		t.Fatal(err)
	}
	waitRecovery(t, func() bool { return completed.Load() == 1 })
	_ = s.Close()
	if completed.Load() != 1 {
		t.Fatal("double settlement")
	}
}

func TestRealtimeStaleAdapterAudioNeverWritten(t *testing.T) {
	conn, peer := net.Pipe()
	defer peer.Close()
	s := &openaiRealtimeSession{conn: conn, outbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}), events: make(chan RealtimeEvent, 8)}
	defer s.Close()
	var dropped atomic.Bool
	if err := s.SendAudioFrame(RealtimeInputAudio{Audio: make([]byte, 960), ReceivedAt: time.Now().Add(-time.Second), MaxAge: time.Millisecond, Complete: func(_, _ time.Duration, success bool) { dropped.Store(!success) }}); err != nil {
		t.Fatal(err)
	}
	go s.writeLoop()
	select {
	case e := <-s.events:
		if e.Err == nil {
			t.Fatal("missing discontinuity")
		}
	case <-time.After(time.Second):
		t.Fatal("no stale-input failure")
	}
	if !dropped.Load() {
		t.Fatal("stale input reservation retained")
	}
}

func TestRealtimeOutputBurstsFirstAndLaterStayOrderedAndScoped(t *testing.T) {
	rt, _ := completionFixture(t)
	for phase, item := range []string{"first", "later"} {
		for i := 0; i < 80; i++ {
			rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: item, Audio: bytes.Repeat([]byte{byte(i + phase), 0}, 480)})
		}
		for i := 0; i < 80; i++ {
			rt.pumpRealtimeOutput()
			f := <-rt.audioOut
			if f.ItemID != item || f.Audio[0] != byte(i+phase) {
				t.Fatal("reordered output")
			}
			f.written(true)
		}
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "old", Audio: make([]byte, 960)})
	for i := 0; i < 16; i++ {
		rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "new", Audio: make([]byte, 960)})
	}
	rt.capabilities.ContinuousConversation = true
	rt.interruptPlayback("test", "old", false, true)
	for i := 0; i < 16; i++ {
		rt.pumpRealtimeOutput()
		f := <-rt.audioOut
		if f.ItemID != "new" {
			t.Fatal("unrelated playback lost")
		}
		f.written(true)
	}
}

func TestRealtimePlaybackCapacityCreditsRequireExactScope(t *testing.T) {
	var w realtimePlaybackWindow
	w.configure(40)
	if !w.reserve(1, "a", 960, 960) || !w.reserve(1, "a", 960, 1920) || w.reserve(1, "a", 960, 2880) {
		t.Fatal("capacity not enforced")
	}
	w.progress(2, "a", 1000)
	w.progress(1, "b", 1000)
	if w.reserve(1, "a", 960, 2880) {
		t.Fatal("stale progress released credits")
	}
	w.progress(1, "a", 20)
	if !w.reserve(1, "a", 960, 2880) {
		t.Fatal("progress failed to release credits")
	}
	w.clear(2, "a")
	if w.reserve(1, "a", 960, 3840) {
		t.Fatal("stale clear released credits")
	}
	w.clear(1, "a")
	if !w.reserve(2, "a", 960, 960) {
		t.Fatal("clear failed to release credits")
	}
}

func TestRealtimePlaybackWindowSplitsBurstsAndWaitsForProgress(t *testing.T) {
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	s := &realtimeBridgeSocket{conn: conn, writeTimeout: time.Second}
	s.window.configure(20)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := s.writeFlowControlledAudio(ctx, RealtimeAudioFrame{SessionGeneration: 7, ItemID: "speech", Audio: make([]byte, 2880), AudioEndMS: 60}, func() error { return nil })
		done <- err
	}()
	for i := 1; i <= 3; i++ {
		raw, _, err := wsutil.ReadServerData(peer)
		if err != nil {
			t.Fatal(err)
		}
		var meta map[string]any
		_ = json.Unmarshal(raw, &meta)
		if meta["audio_end_ms"] != float64(i*20) {
			t.Fatal(meta)
		}
		pcm, _, err := wsutil.ReadServerData(peer)
		if err != nil || len(pcm) != 960 {
			t.Fatal(len(pcm), err)
		}
		if i < 3 {
			select {
			case err := <-done:
				t.Fatal("window bypassed", err)
			case <-time.After(10 * time.Millisecond):
			}
		}
		s.window.progress(7, "speech", i*20)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("flow stalled")
	}
}

func TestRealtimeInterruptionDuringBlockedActorPreservesNewPlayback(t *testing.T) {
	rt, _ := completionFixture(t)
	session := newRecoveryTestSession()
	rt.replaceSession(session)
	done := runCompletionFixture(rt)
	defer func() { rt.cancel(); waitCompletionFixture(t, done) }()
	waitRecovery(t, func() bool { return rt.audioLane.active.Load() })
	rt.Thinker.session.mu.Lock()
	locked := true
	defer func() {
		if locked {
			rt.Thinker.session.mu.Unlock()
		}
	}()
	session.events <- RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "caller", Transcript: "Bonjour.", Final: true}
	waitRecovery(t, func() bool {
		if rt.transcriptMu.TryLock() {
			rt.transcriptMu.Unlock()
			return false
		}
		return true
	})
	session.events <- RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "old", Audio: make([]byte, 960)}
	session.events <- RealtimeEvent{Type: RealtimeEventSpeechStarted}
	session.events <- RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "new", Audio: make([]byte, 960)}
	waitRecovery(t, func() bool { rt.outputMu.Lock(); defer rt.outputMu.Unlock(); return rt.outputItemID == "new" })
	rt.Thinker.session.mu.Unlock()
	locked = false
	// Barrier behind speech-start on the actor lane.
	session.events <- RealtimeEvent{Type: RealtimeEventTranscriptInput, ItemID: "barrier", Transcript: "Continuez.", Final: true}
	waitRecovery(t, func() bool {
		rt.transcriptMu.Lock()
		defer rt.transcriptMu.Unlock()
		for _, m := range rt.messages {
			if m.Content == "Continuez." {
				return true
			}
		}
		return false
	})
	select {
	case f := <-rt.audioOut:
		if f.ItemID != "new" || f.cancelled() {
			t.Fatal("actor replay cancelled new playback")
		}
		f.written(true)
	case <-time.After(time.Second):
		t.Fatal("new playback missing")
	}
}

func TestRealtimeOutputDurationLimitIsExplicitAndBounded(t *testing.T) {
	rt, _ := completionFixture(t)
	// Initialize the production failure lane without starting its consumer.
	rt.audioLane.active.Store(true)
	defer rt.audioLane.active.Store(false)
	for i := 0; i < 1000; i++ {
		rt.forwardRealtimeAudio(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "long", Audio: make([]byte, 960)}, 1)
	}
	pending, _, _ := rt.delivery.snapshot()
	if pending > 15*realtimePCMBytesPerSecond {
		t.Fatal("unbounded output", pending)
	}
	select {
	case failure := <-rt.audioLane.failures:
		if failure.reason != "output_duration_limit" {
			t.Fatal(failure.reason)
		}
	default:
		t.Fatal("overflow was silent")
	}
}

func TestRealtimePartialWriteReceiptPreservesDeliveredBytes(t *testing.T) {
	var d realtimeDelivery
	settle := d.enqueueBytes("1/a", "a", 2880)
	settle(960)
	settle(2880)
	pending, _, data := d.snapshot()
	if pending != 0 || data["bridge_written_audio_ms"] != float64(20) || data["dropped_audio_ms"] != float64(40) {
		t.Fatal(data)
	}
}

func TestRealtimeAudioPreflightDoesNotWaitForPersistence(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.audioLane.tools.Store([]NativeTool{{Name: "bookings_create"}})
	rt.audioLane.blocked = map[string]bool{}
	rt.audioLane.tails = map[string]string{}
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	rt.preflightAudioTranscript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "r", ItemID: "speech", Transcript: `callto:bookings_create{"slot_id":"one"}`})
	if rt.forwardRealtimeAudio(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "r", ItemID: "speech", Audio: make([]byte, 960)}, 1) {
		t.Fatal("unchecked technical speech escaped")
	}
	if len(rt.audioOut) != 0 {
		t.Fatal("blocked audio reached output")
	}
}

func TestRealtimeInputRejectedAdmissionHasNoTransportReceipt(t *testing.T) {
	s := &openaiRealtimeSession{outbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{})}
	defer s.Close()
	var calls atomic.Int32
	frame := RealtimeInputAudio{Audio: make([]byte, 960), Complete: func(_, _ time.Duration, _ bool) { calls.Add(1) }}
	if err := s.SendAudioFrame(frame); err != nil {
		t.Fatal(err)
	}
	if err := s.SendAudioFrame(frame); realtimeErrorClass(err) != RealtimeErrorPressure {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatal("unadmitted frame produced receipt")
	}
	_ = s.Close()
	if calls.Load() != 1 {
		t.Fatal("accepted frame did not settle exactly once")
	}
}

func TestRealtimeNativeInterruptionClearsAllExistingSegments(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "one", Audio: make([]byte, 960)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "two", Audio: make([]byte, 960)})
	rt.clearCallerInterruptedPlayback("caller", false, false)
	if len(rt.audioOut) != 0 {
		t.Fatal("older native-model playback survived interruption")
	}
}

func TestRealtimeReconnectedGenerationCanReuseProviderItemID(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "reused", Audio: make([]byte, 960)})
	rt.interruptPlayback("caller", "reused", false, true)
	if err := rt.openSession(true); err != nil {
		t.Fatal(err)
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "reused", Audio: make([]byte, 960)})
	select {
	case f := <-rt.audioOut:
		if f.cancelled() {
			t.Fatal("prior generation cancelled current audio")
		}
		f.written(true)
	default:
		t.Fatal("new generation was suppressed")
	}
}

func TestRealtimeTrackedAdmissionPreservesInputUsage(t *testing.T) {
	s := &openaiRealtimeSession{outbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{})}
	defer s.Close()
	f := RealtimeInputAudio{Audio: make([]byte, 960)}
	if err := s.SendAudioFrame(f); err != nil {
		t.Fatal(err)
	}
	if err := s.SendAudioFrame(f); realtimeErrorClass(err) != RealtimeErrorPressure {
		t.Fatal(err)
	}
	if s.audioInBytes.Load() != 960 {
		t.Fatal("admission changed usage accounting", s.audioInBytes.Load())
	}
}
