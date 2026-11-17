package core

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestRealtimeRequestTimingStagesAndNoInventedPlayback(t *testing.T) {
	var events []map[string]any
	tracker := newRealtimeTiming(4, func(k string, d map[string]any) {
		if k == "realtime.request_timing" {
			events = append(events, d)
		}
	})
	base := time.Now()
	w := ThreadWork{ID: "booking", Revision: 2, DelegationID: "delegation"}
	tracker.speechStopped(base, "client_reported_signal_received")
	for _, step := range []struct {
		stage string
		ms    int
	}{{"requested", 100}, {"started", 150}, {"result_available", 450}, {"submitted", 460}} {
		tracker.requestStage(w, step.stage, base.Add(time.Duration(step.ms)*time.Millisecond))
	}
	tracker.registerContext("append", w)
	tracker.contextAcknowledged("append", base.Add(490*time.Millisecond))
	tracker.contextAcknowledged("append", base.Add(time.Second))
	last := events[len(events)-1]
	for field, want := range map[string]float64{"speech_end_to_delegation_ms": 100, "delegation_to_execution_ms": 50, "execution_to_result_ms": 300, "result_to_submission_ms": 10, "submission_to_context_ack_ms": 30} {
		if last[field] != want {
			t.Fatalf("%s=%v want=%v", field, last[field], want)
		}
	}
	if len(events) != 5 || last["speech_end_to_useful_playback_report_ms"] != nil || last["playback_confirmed"] != false {
		t.Fatalf("bad context acknowledgement: %v", events)
	}
	tracker.requestStage(w, "backend_completed", base.Add(time.Second))
	tracker.finish("", "", "thread_ended")
	if events[len(events)-1]["status"] != "backend_completed" {
		t.Fatal("backend outcome lost on shutdown")
	}
	tracker.finish("", "", "thread_ended")
	if len(events) != 7 {
		t.Fatal("duplicate shutdown")
	}
}

func TestRealtimeRequestTimingSilenceBackchannelsAndExplicitAnswers(t *testing.T) {
	for _, purpose := range []string{"", "backchannel", "answer"} {
		t.Run(purpose, func(t *testing.T) {
			var playback []map[string]any
			tracker := newRealtimeTiming(1, func(k string, d map[string]any) {
				if k == "realtime.playback_timing" {
					playback = append(playback, d)
				}
			})
			base := time.Now()
			w := ThreadWork{ID: "work", Revision: 1, DelegationID: "d"}
			tracker.speechStopped(base, "provider_vad_event_received")
			tracker.requestStage(w, "requested", base.Add(10*time.Millisecond))
			tracker.requestStage(w, "submitted", base.Add(20*time.Millisecond))
			tracker.audio("r", "i", base.Add(30*time.Millisecond))
			// First 20ms is silent. Its receipt must not count as useful speech.
			event := RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "r", ItemID: "i", OutputPurpose: purpose, AudioEndMS: 20}
			if purpose != "" {
				event.OutputWork = &w
			}
			tracker.outputObservation(event, false, base.Add(30*time.Millisecond))
			tracker.enqueued("r", "i", base.Add(31*time.Millisecond), 1, 960)
			tracker.written("r", "i", base.Add(32*time.Millisecond), 960, true)
			tracker.playback("i", 20, base.Add(52*time.Millisecond), "estimated")
			event.AudioEndMS = 40
			tracker.outputObservation(event, true, base.Add(50*time.Millisecond))
			tracker.enqueued("r", "i", base.Add(51*time.Millisecond), 1, 960)
			tracker.written("r", "i", base.Add(52*time.Millisecond), 960, true)
			tracker.finish("r", "i", "completed")
			tracker.playback("i", 9999, base.Add(72*time.Millisecond), "carrier_mark")
			want := 1
			if purpose == "answer" {
				want = 2
			}
			if len(playback) != want {
				t.Fatalf("receipts=%v", playback)
			}
			if playback[0]["speech_end_to_useful_playback_report_ms"] != nil {
				t.Fatal("silent audio became answer")
			}
			last := playback[len(playback)-1]
			if purpose == "answer" {
				if last["speech_end_to_useful_playback_report_ms"] != float64(72) || last["audio_end_ms"] != 40 || last["speech_end_to_useful_audio_ms"] != float64(50) {
					t.Fatalf("wrong answer timing: %v", last)
				}
			} else if last["speech_end_to_useful_playback_report_ms"] != nil {
				t.Fatal("invented useful-answer association")
			}
			if last["audible_playback_confirmed"] != false {
				t.Fatal("Core claimed the caller heard it")
			}
		})
	}
}

func TestRealtimeRequestTimingFailureBoundsAndGeneration(t *testing.T) {
	var events []map[string]any
	tracker := newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.request_timing" {
			events = append(events, d)
		}
	})
	base := time.Now()
	for i := 0; i < 100; i++ {
		w := ThreadWork{ID: fmt.Sprint(i), Revision: 1}
		tracker.requestStage(w, "requested", base)
		tracker.registerContext(fmt.Sprint(i), w)
	}
	if len(tracker.requests) != 64 || len(events) != 136 {
		t.Fatalf("unbounded tracker or missing evictions %d %d", len(tracker.requests), len(events))
	}
	for _, status := range []string{"timed_out", "failed", "superseded", "delivery_failed"} {
		tracker.requestStage(ThreadWork{ID: "99", Revision: 1}, status, base)
	}
	fresh := newRealtimeTiming(2, nil)
	fresh.contextAcknowledged("99", base)
	if len(fresh.requests) != 0 {
		t.Fatal("old ack created a new request")
	}
	tracker.finish("", "", "session_replaced")
	if len(tracker.requests) != 0 || len(tracker.contextRequests) != 0 {
		t.Fatal("tracking retained after shutdown")
	}
	for _, e := range events {
		if e["speech_end_to_delegation_ms"] != nil {
			t.Fatal("invented speech end")
		}
	}
}

func TestRealtimeTimingRejectsUnwrittenInterruptedAndStaleReceipts(t *testing.T) {
	for _, status := range []string{"unwritten", "interrupted", "blocked", "old_generation"} {
		t.Run(status, func(t *testing.T) {
			rt, _ := completionFixture(t)
			receipts := 0
			tracker := newRealtimeTiming(1, func(k string, _ map[string]any) {
				if k == "realtime.playback_timing" {
					receipts++
				}
			})
			rt.timing = tracker
			at := time.Now()
			tracker.audio("r", "i", at)
			tracker.enqueued("r", "i", at, 1, 960)
			if status != "unwritten" {
				tracker.written("r", "i", at, 960, true)
			}
			if status == "interrupted" || status == "blocked" {
				tracker.finish("r", "i", status)
			}
			if status == "old_generation" {
				rt.acknowledgePlaybackEvidence("i", 20, 99, "carrier_mark")
			} else {
				tracker.playback("i", 20, at, "carrier_mark")
			}
			if receipts != 0 {
				t.Fatal("invalid acknowledgement accepted")
			}
		})
	}
}

func TestRealtimeTimingLateItemReceiptAndEvidence(t *testing.T) {
	rt, _ := completionFixture(t)
	var data map[string]any
	rt.timing = newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.playback_timing" {
			data = d
		}
	})
	at := time.Now()
	rt.timing.audio("r", "old", at)
	rt.timing.enqueued("r", "old", at, 1, 960)
	rt.timing.written("r", "old", at, 960, true)
	rt.timing.finish("r", "old", "completed")
	rt.outputItemID = "new"
	rt.acknowledgePlaybackEvidence("old", 20, 0, "rtp_transmitted")
	if data == nil || data["evidence"] != "rtp_transmitted" || data["audible_playback_confirmed"] != false {
		t.Fatalf("lost late receipt semantics: %v", data)
	}
}

func TestRealtimeRequestTimingConcurrentSessions(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tracker := newRealtimeTiming(1, func(string, map[string]any) {})
			at := time.Now()
			w := ThreadWork{ID: "r", Revision: 1}
			var lanes sync.WaitGroup
			for lane := 0; lane < 3; lane++ {
				lanes.Add(1)
				go func() {
					defer lanes.Done()
					for j := 0; j < 100; j++ {
						tracker.requestStage(w, "started", at)
						tracker.speechStopped(at, "test")
						tracker.audio("r", "i", at)
						tracker.outputObservation(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "i"}, true, at)
						tracker.registerContext("ctx", w)
						tracker.contextAcknowledged("ctx", at)
					}
				}()
			}
			lanes.Wait()
			tracker.finish("", "", "thread_ended")
		}()
	}
	wg.Wait()
}

func TestRealtimeExecutionResultTimingFlowsThroughSharedPath(t *testing.T) {
	rt, _, _, w := streamLatencyFixture(t)
	var stages []string
	rt.timing = newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.request_timing" {
			stages = append(stages, d["stage"].(string))
		}
	})
	rt.timing.requestStage(w, "requested", time.Now())
	rt.timing.requestStage(w, "started", time.Now())
	w.Kind = "result"
	w.Status = "completed"
	w.Worker = "worker"
	rt.consumeThreadWork(Event{Work: &w, Text: "Verified result."})
	rt.consumeThreadWork(Event{Work: &w, Text: "Verified result."})
	if fmt.Sprint(stages) != "[requested started result_available submitted backend_completed]" {
		t.Fatalf("stages=%v", stages)
	}
}

func TestRealtimeTimingSupersededAnswerAndLateCaption(t *testing.T) {
	var captions, receipts []map[string]any
	tracker := newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.caption_timing" {
			captions = append(captions, d)
		}
		if k == "realtime.playback_timing" {
			receipts = append(receipts, d)
		}
	})
	at := time.Now()
	w := ThreadWork{ID: "r", Revision: 1}
	tracker.speechStopped(at, "test")
	tracker.requestStage(w, "requested", at)
	tracker.audio("response", "i", at)
	tracker.outputObservation(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "i", OutputWork: &w, OutputPurpose: "answer", AudioEndMS: 20}, true, at)
	tracker.enqueued("response", "i", at, 1, 960)
	tracker.written("response", "i", at, 960, true)
	tracker.finish("response", "i", "completed")
	tracker.requestStage(w, "superseded", at)
	tracker.outputObservation(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ItemID: "i"}, false, at.Add(50*time.Millisecond))
	tracker.outputObservation(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ItemID: "i"}, false, at.Add(time.Second))
	tracker.playback("i", 20, at.Add(100*time.Millisecond), "carrier_mark")
	if len(captions) != 1 || captions[0]["provider_audio_to_caption_ms"] != float64(50) {
		t.Fatalf("late caption=%v", captions)
	}
	if len(receipts) != 1 || receipts[0]["output_association"] != "superseded" || receipts[0]["speech_end_to_useful_playback_report_ms"] != nil {
		t.Fatal("stale answer counted as useful", receipts)
	}
	tracker.finish("", "", "thread_ended")
	tracker.requestStage(w, "backend_completed", at)
	if len(tracker.requests) != 0 {
		t.Fatal("late completion reopened closed tracking")
	}
}

func TestRealtimeTimingInterruptAfterSummaryRejectsReceipt(t *testing.T) {
	receipts := 0
	tracker := newRealtimeTiming(1, func(k string, _ map[string]any) {
		if k == "realtime.playback_timing" {
			receipts++
		}
	})
	at := time.Now()
	tracker.audio("r", "i", at)
	tracker.enqueued("r", "i", at, 1, 960)
	tracker.written("r", "i", at, 960, true)
	tracker.finish("r", "i", "completed")
	tracker.finish("", "i", "interrupted")
	tracker.playback("i", 20, at, "carrier_mark")
	if receipts != 0 {
		t.Fatal("cleared audio counted as played")
	}
}

func TestRealtimeRequestTimingNativeToolOutcomes(t *testing.T) {
	for _, outcome := range []string{"success", "failure", "cancelled"} {
		t.Run(outcome, func(t *testing.T) {
			rt, session := completionFixture(t)
			var stages []string
			rt.timing = newRealtimeTiming(1, func(k string, d map[string]any) {
				if k == "realtime.request_timing" {
					if d["request_id"] != "tool/call" {
						t.Error("lost call identity")
					}
					stages = append(stages, d["stage"].(string))
				}
			})
			rt.beginToolCall(RealtimeEvent{Type: RealtimeEventToolCall, ToolCallID: "call", ToolName: "book", ResponseID: "r"}, session)
			if outcome == "cancelled" {
				rt.cancelRealtimeToolCall(session, "call")
			} else {
				rt.submitToolResult(session, "call", "fixture", outcome == "failure")
			}
			want := "[requested started result_available submitted backend_completed]"
			if outcome == "failure" {
				want = "[requested started result_available submitted failed]"
			}
			if outcome == "cancelled" {
				want = "[requested started cancelled]"
			}
			if fmt.Sprint(stages) != want {
				t.Fatalf("stages=%v", stages)
			}
		})
	}
}

func TestRealtimeExecutionFailedResultTiming(t *testing.T) {
	rt, _, _, w := streamLatencyFixture(t)
	var last map[string]any
	rt.timing = newRealtimeTiming(1, func(k string, d map[string]any) {
		if k == "realtime.request_timing" {
			last = d
		}
	})
	rt.timing.requestStage(w, "requested", time.Now())
	w.Kind = "result"
	w.Status = "failed"
	w.Worker = "worker"
	rt.consumeThreadWork(Event{Work: &w, Text: "The backend failed."})
	if last == nil || last["status"] != "failed" {
		t.Fatal("backend error counted as successful", last)
	}
}
