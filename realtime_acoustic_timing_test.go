package core

import (
	"bytes"
	"encoding/binary"
	"sync"
	"testing"
	"time"
)

func acousticPCM(ms, amplitude int) []byte {
	b := make([]byte, ms*48)
	for i := 0; i < len(b); i += 2 {
		binary.LittleEndian.PutUint16(b[i:], uint16(int16(amplitude)))
	}
	return b
}

func TestAcousticTimingRejectsPeaksAndPreservesPacketization(t *testing.T) {
	spike := acousticPCM(100, 0)
	binary.LittleEndian.PutUint16(spike, 32767)
	var d realtimeAcousticDetector
	if d.observe(spike, AudioPCM16, 24000, 100) {
		t.Fatal("isolated peak counted as sustained activity")
	}
	for _, size := range []int{48, 240, 480, 960} {
		d = realtimeAcousticDetector{}
		pcm := acousticPCM(20, 120)
		seen := false
		for off := 0; off < len(pcm); off += size {
			seen = d.observe(pcm[off:min(off+size, len(pcm))], AudioPCM16, 24000, 100) || seen
		}
		if !seen {
			t.Fatalf("missed quiet activity with chunk size %d", size)
		}
	}
	for _, format := range []AudioFormat{AudioG711ULaw, AudioG711ALaw} {
		d = realtimeAcousticDetector{}
		if !d.observe(bytes.Repeat([]byte{0}, 160), format, 8000, 100) {
			t.Fatalf("G711 activity unavailable: %s", format)
		}
	}
}

func TestCallerTimingSurvivesSilentPlaybackSegments(t *testing.T) {
	var reports []map[string]any
	timing := newRealtimeTiming(5, func(k string, d map[string]any) {
		if k == "realtime.caller_timing" {
			reports = append(reports, d)
		}
	})
	base := time.Now()
	timing.speechStopped(base, "client_reported_signal_received")
	g := realtimeContinuousGuard{}
	silence, voice := acousticPCM(400, 0), acousticPCM(100, 1000)
	a := g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: base.Add(time.Second), Audio: silence}, timing)
	b := g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: base.Add(1400 * time.Millisecond), Audio: voice}, timing)
	var out RealtimeEvent
	for _, e := range b {
		if e.Type == RealtimeEventAudioOut {
			out = e
		}
	}
	if len(a) != 1 || !bytes.Equal(a[0].Audio, silence) || !bytes.Equal(out.Audio, voice) {
		t.Fatal("measurement changed PCM")
	}
	if a[0].ItemID == out.ItemID {
		t.Fatal("expected separate playback segments")
	}
	if timing.items[out.ItemID].speechEnd != base {
		t.Fatal("voiced segment lost caller endpoint")
	}
	timing.enqueued("", out.ItemID, base.Add(1401*time.Millisecond), 1, len(voice))
	timing.written("", out.ItemID, base.Add(1402*time.Millisecond), len(voice), true)
	timing.reportCallerTiming("session_ended")
	if len(reports) != 1 {
		t.Fatalf("reports=%v", reports)
	}
	d := reports[0]
	for key, want := range map[string]float64{"speech_end_to_first_audio_ms": 1000, "speech_end_to_acoustic_output_ms": 1400, "pre_acoustic_audio_ms": 400, "acoustic_output_to_bridge_write_ms": 2} {
		if d[key] != want {
			t.Errorf("%s=%v want %v", key, d[key], want)
		}
	}
	if d["useful_answer_confirmed"] != false || d["provider_receipt_confirmed"] != false {
		t.Fatal("invented remote evidence")
	}
}

func TestCallerTimingInputWriteAndQuietEndpoint(t *testing.T) {
	var result map[string]any
	timing := newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.caller_timing" {
			result = d
		}
	})
	base := time.Now()
	f := timing.inputObserved(acousticPCM(20, 80), base, AudioPCM16, 24000)
	f.submittedAt(base.Add(30 * time.Millisecond))
	// A fast reply may arrive before the observational quiet interval completes.
	timing.providerAudio("", "reply", base.Add(100*time.Millisecond), acousticPCM(40, 200), AudioPCM16, 24000)
	for i := 1; i <= 15; i++ {
		timing.inputObserved(acousticPCM(20, 0), base.Add(time.Duration(i)*20*time.Millisecond), AudioPCM16, 24000)
	}
	f.settled(base.Add(40*time.Millisecond), true, true)
	timing.reportCallerTiming("session_ended")
	if result["speech_end_source"] != "core_input_acoustic_estimate" || result["final_speech_frame_to_submission_ms"] != float64(30) || result["final_speech_frame_to_local_write_ms"] != float64(40) || result["speech_end_to_acoustic_output_ms"] != float64(100) {
		t.Fatalf("wrong boundaries: %v", result)
	}
	if result["provider_receipt_confirmed"] != false {
		t.Fatal("local write became remote receipt")
	}
}

func TestCallerTimingNoInventedWritesAndNoReply(t *testing.T) {
	for _, success := range []bool{false, true} {
		var result map[string]any
		timing := newRealtimeTiming(1, func(k string, d map[string]any) {
			if k == "realtime.caller_timing" {
				result = d
			}
		})
		f := timing.inputObserved(acousticPCM(20, 500), time.Now(), AudioPCM16, 24000)
		f.settled(time.Now(), success, false)
		timing.reportCallerTiming("disconnect")
		if result["final_speech_frame_to_local_write_ms"] != nil || result["speech_end_to_acoustic_output_ms"] != nil {
			t.Fatalf("invented evidence: %v", result)
		}
	}
}

func TestCallerTimingBridgeMustCoverAcousticPacket(t *testing.T) {
	timing := newRealtimeTiming(1, nil)
	base := time.Now()
	timing.speechStopped(base, "test")
	timing.providerAudio("", "same-item", base.Add(time.Second), acousticPCM(20, 0), AudioPCM16, 24000)
	timing.providerAudio("", "same-item", base.Add(1100*time.Millisecond), acousticPCM(20, 500), AudioPCM16, 24000)
	// The first (silent) packet may only leave Core after voiced audio arrives.
	timing.written("", "same-item", base.Add(1200*time.Millisecond), 960, true)
	if !timing.caller.bridge.IsZero() {
		t.Fatal("late silent packet write counted as acoustic transmission")
	}
	timing.written("", "same-item", base.Add(1220*time.Millisecond), 960, true)
	if timing.caller.bridge != base.Add(1220*time.Millisecond) {
		t.Fatal("acoustic packet write missing")
	}
}

func TestCallerTimingPublicationIsBoundedAndClosureIsIdempotent(t *testing.T) {
	count := 0
	timing := newRealtimeTiming(1, func(k string, _ map[string]any) {
		if k == "realtime.caller_timing" {
			count++
		}
	})
	base := time.Now()
	timing.speechStopped(base, "test")
	timing.reportCallerTiming("")
	for i := 0; i < 100; i++ {
		timing.providerAudio("", "i", base.Add(time.Duration(i)*time.Millisecond), acousticPCM(20, 0), AudioPCM16, 24000)
		timing.reportCallerTiming("")
	}
	if count != 1 {
		t.Fatalf("per-packet telemetry: %d", count)
	}
	timing.reportCallerTiming("disconnect")
	timing.reportCallerTiming("disconnect")
	if count != 2 {
		t.Fatalf("duplicate closure: %d", count)
	}
}

func TestCallerTimingOverlapAndLateCallbacksAreIsolated(t *testing.T) {
	timing := newRealtimeTiming(1, nil)
	base := time.Now()
	timing.providerAudio("", "greeting", base, acousticPCM(20, 500), AudioPCM16, 24000)
	old := timing.inputObserved(acousticPCM(20, 500), base.Add(20*time.Millisecond), AudioPCM16, 24000)
	if !timing.caller.overlap {
		t.Fatal("overlapping greeting not identified")
	}
	timing.inputObserved(acousticPCM(300, 0), base.Add(320*time.Millisecond), AudioPCM16, 24000)
	newFrame := timing.inputObserved(acousticPCM(20, 500), base.Add(time.Second), AudioPCM16, 24000)
	old.settled(base.Add(2*time.Second), true, true)
	if timing.caller.finalInput != newFrame || !newFrame.written.IsZero() {
		t.Fatal("late write contaminated next caller observation")
	}
	timing.reportCallerTiming("session_replaced")
	if timing.inputObserved(acousticPCM(20, 500), time.Now(), AudioPCM16, 24000) != nil {
		t.Fatal("closed tracker accepted new input")
	}
}

func TestCallerTimingBoundedConcurrentAndNoHotPathPublication(t *testing.T) {
	timing := newRealtimeTiming(1, func(string, map[string]any) { t.Error("hot path emitted telemetry") })
	var wg sync.WaitGroup
	for n := 0; n < 4; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				at := time.Now()
				f := timing.inputObserved(acousticPCM(20, 500), at, AudioPCM16, 24000)
				f.submittedAt(at)
				f.settled(at, true, true)
				timing.speechStopped(at, "test")
				timing.providerAudio("", "out", at, acousticPCM(20, 200), AudioPCM16, 24000)
			}
		}()
	}
	wg.Wait()
	if len(timing.callerReports) > 64 {
		t.Fatal("unbounded reports")
	}
	timing.emit = nil
	timing.reportCallerTiming("session_ended")
	timing.reportCallerTiming("session_ended")
}
