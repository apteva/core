package core

import (
	"testing"
	"time"
)

func TestRealtimeQuietPendingSilenceSettlesBeforeSummary(t *testing.T) {
	for _, written := range []int{4800, 2400, 0} {
		t.Run(string(rune('a'+written/2400)), func(t *testing.T) {
			rt, _ := completionFixture(t)
			rt.capabilities.ContinuousConversation = true
			var summaries []map[string]any
			rt.timing = newRealtimeTiming(1, func(kind string, d map[string]any) {
				if kind == "realtime.utterance_timing" {
					summaries = append(summaries, d)
				}
			})
			now := time.Now()
			rt.timing.audio("", "tail", now.Add(-time.Second))
			rt.timing.enqueued("", "tail", now, 1, 4800)
			settle := rt.delivery.enqueueBytes("1/tail", "tail", 4800, true)
			rt.continuous.lastRelease = now.Add(-time.Second)
			rt.observeContinuousIdle()
			if len(summaries) != 0 {
				t.Fatal("quiet finalized an outstanding write", summaries)
			}
			if rt.delivery.hasPendingPlayback() {
				t.Fatal("silence incorrectly holds speaking state")
			}
			rt.timing.writtenBytes("", "tail", now.Add(time.Millisecond), 4800, written)
			settle(written)
			if len(summaries) != 1 || summaries[0]["dropped_audio_bytes"] != 4800-written {
				t.Fatal(summaries)
			}
			_, _, delivery := rt.delivery.snapshot()
			if delivery["dropped_audio_ms"] != float64(4800-written)/48 {
				t.Fatal(delivery)
			}
			want := "application_observed_quiet"
			if written < 4800 {
				want = "output_dropped"
			}
			if summaries[0]["completion_status"] != want {
				t.Fatal(summaries)
			}
			rt.observeContinuousIdle()
			if len(summaries) != 1 {
				t.Fatal("duplicate finalization")
			}
		})
	}
}

func TestRealtimeQuietMultiplePendingWritesAndTermination(t *testing.T) {
	for _, terminal := range []string{"", "interrupted", "session_ended"} {
		t.Run("terminal="+terminal, func(t *testing.T) {
			var out []map[string]any
			timing := newRealtimeTiming(3, func(_ string, d map[string]any) { out = append(out, d) })
			now := time.Now()
			for range 2 {
				timing.enqueued("", "tail", now, 2, 960)
			}
			timing.finish("", "", "application_observed_quiet")
			timing.writtenBytes("", "tail", now, 960, 960)
			if len(out) != 0 {
				t.Fatal("second write still pending")
			}
			if terminal != "" {
				timing.finish("", "", terminal)
			}
			timing.writtenBytes("", "tail", now, 960, 960)
			if len(out) != 1 {
				t.Fatal(out)
			}
			want := 0
			if terminal != "" {
				want = 960
			}
			if out[0]["dropped_audio_bytes"] != want {
				t.Fatal(out)
			}
		})
	}
}

func TestRealtimeRepeatedQuietPreservesWriteFailure(t *testing.T) {
	var result map[string]any
	timing := newRealtimeTiming(1, func(_ string, d map[string]any) { result = d })
	for range 2 {
		timing.enqueued("", "tail", time.Now(), 2, 960)
	}
	timing.finish("", "", "application_observed_quiet")
	timing.writtenBytes("", "tail", time.Now(), 960, 480)
	timing.finish("", "", "application_observed_quiet")
	timing.writtenBytes("", "tail", time.Now(), 960, 960)
	if result == nil || result["completion_status"] != "output_dropped" || result["dropped_audio_bytes"] != 480 {
		t.Fatal(result)
	}
}
