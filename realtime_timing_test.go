package core

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRealtimeTimingSeparatesStagesAndWaitsForBridge(t *testing.T) {
	var summaries []map[string]any
	timing := newRealtimeTiming(7, func(kind string, data map[string]any) {
		if kind == "realtime.utterance_timing" {
			summaries = append(summaries, data)
		}
	})
	base := time.Now()
	timing.speechStopped(base, "client_reported_signal_received")
	timing.audio("r", "i", base.Add(time.Second))
	timing.audio("r", "i", base.Add(2*time.Second)) // Must retain the first arrival.
	timing.released("r", "i", base.Add(1100*time.Millisecond), 48000, "transcript_prefix")
	timing.enqueued("r", "i", base.Add(1105*time.Millisecond), 3)
	timing.finish("r", "i", "completed")
	if len(summaries) != 0 {
		t.Fatal("summary preceded pending bridge write")
	}
	timing.written("r", "i", base.Add(1120*time.Millisecond), 48000, true)
	timing.finish("r", "i", "completed")
	if len(summaries) != 1 {
		t.Fatalf("summaries=%d", len(summaries))
	}
	s := summaries[0]
	for name, want := range map[string]float64{"speech_end_to_provider_audio_ms": 1000, "provider_audio_to_guard_release_ms": 100, "guard_release_to_output_enqueue_ms": 5, "output_enqueue_to_bridge_write_ms": 15, "buffered_audio_ms": 1000} {
		if s[name] != want {
			t.Errorf("%s=%v want %v", name, s[name], want)
		}
	}
	if s["generation"] != uint64(7) || s["max_output_queue_depth"] != 3 {
		t.Fatalf("bad correlation/queue: %v", s)
	}
}

func TestRealtimeTimingUnavailableAndInterrupted(t *testing.T) {
	for _, status := range []string{"blocked", "interrupted", "session_ended"} {
		t.Run(status, func(t *testing.T) {
			var data map[string]any
			timing := newRealtimeTiming(1, func(_ string, d map[string]any) { data = d })
			timing.audio("r", "i", time.Now())
			timing.buffered("r", "i", 48000)
			timing.dropped("r", "i", 48000)
			timing.finish("r", "i", status)
			if data == nil || data["completion_status"] != status || data["dropped_audio_bytes"] != 48000 {
				t.Fatalf("missing terminal summary: %v", data)
			}
			for _, field := range []string{"speech_end_to_provider_audio_ms", "provider_audio_to_guard_release_ms", "output_enqueue_to_bridge_write_ms"} {
				if data[field] != nil {
					t.Errorf("unavailable %s=%v", field, data[field])
				}
			}
		})
	}
}

func TestRealtimeTimingConcurrentAudioDoesNotEmitAndIsBounded(t *testing.T) {
	timing := newRealtimeTiming(1, func(string, map[string]any) { t.Error("audio observations emitted telemetry") })
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 1000 {
				timing.audio("r", "i", time.Now())
				timing.released("r", "i", time.Now(), 0, "provider_stream")
				timing.dropped("r", "i", 2)
			}
		}()
	}
	wg.Wait()
	for n := 0; n < 200; n++ {
		timing.audio("", strings.Repeat("x", n+1), time.Now())
	}
	if len(timing.items) > 64 {
		t.Fatal("unbounded active state")
	}
	timing.emit = nil
	timing.finish("", "", "session_ended")
	if len(timing.closed) > 64 || len(timing.completed) > 64 {
		t.Fatal("unbounded completed state")
	}
}

func TestRealtimeTimingPlaybackReceiptAfterSummary(t *testing.T) {
	var kinds []string
	var receipt map[string]any
	timing := newRealtimeTiming(3, func(kind string, data map[string]any) { kinds = append(kinds, kind); receipt = data })
	at := time.Now()
	timing.audio("r", "i", at)
	timing.enqueued("r", "i", at, 1)
	timing.written("r", "i", at.Add(time.Millisecond), 2, true)
	timing.finish("r", "i", "completed")
	timing.playback("i", 0, at)
	timing.playback("i", 20, at.Add(51*time.Millisecond))
	timing.playback("i", 40, at.Add(time.Second))
	if len(kinds) != 2 || kinds[1] != "realtime.playback_timing" || receipt["bridge_write_to_playback_report_ms"] != float64(50) || receipt["source"] != "client_reported_progress_received" {
		t.Fatalf("bad playback receipts: %v %v", kinds, receipt)
	}
}

func TestRealtimeTimingRealBridgeWrite(t *testing.T) {
	fixture := newRealtimeAudioBridgeFixture(t, "timing-bridge")
	timing := newRealtimeTiming(2, func(kind string, data map[string]any) { fixture.telemetry.Emit(kind, fixture.thread, data) })
	at := time.Now()
	timing.audio("r", "i", at)
	timing.released("r", "i", at, 0, "provider_stream")
	timing.enqueued("r", "i", at, 1)
	fixture.audioOut <- RealtimeAudioFrame{Audio: []byte{1, 2}, ResponseID: "r", ItemID: "i", timing: timing}
	timing.finish("r", "i", "completed")
	readRealtimeBridgeTestFrame(t, fixture.conn)
	readRealtimeBridgeTestFrame(t, fixture.conn)
	event := waitForRealtimeBridgeEvent(t, fixture.telemetry, fixture.thread, "realtime.utterance_timing")
	var data map[string]any
	_ = json.Unmarshal(event.Data, &data)
	if data["output_enqueue_to_bridge_write_ms"] == nil || data["completion_status"] != "completed" {
		t.Fatalf("no bridge measurement: %s", event.Data)
	}
}

func TestRealtimeOutputConfigPersistsAndRestores(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	provider := &fakeRealtimeProvider{}
	thinker.config.RealtimeEnabled = true
	thinker.pool.realtimeProviders = map[string]RealtimeProvider{provider.Name(): provider}
	thinker.pool.realtimeOrder = []string{provider.Name()}
	thinker.pool.realtimeDefault = provider.Name()
	output := RealtimeOutputConfig{ToolMode: "async", SpeechGuard: RealtimeSpeechGuardConfig{Mode: "transcript_prefix", PrefixBytes: 48, MaxBufferedAudioMS: 2000}}
	w := postThreadForTest(t, api, "voice-output", map[string]any{"realtime": true, "directive": "Answer calls.", "provider": provider.Name(), "realtime_output": output})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn: %d %s", w.Code, w.Body.String())
	}
	provider.mu.Lock()
	actual := provider.opens[0].OutputConfig
	provider.mu.Unlock()
	if actual != output {
		t.Fatalf("live options=%v", actual)
	}
	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatal(err)
	}
	state, ok := persistentThreadByID(reloaded.GetThreads(), "voice-output")
	if !ok || state.RealtimeOutput == nil || *state.RealtimeOutput != output {
		t.Fatalf("lost durable intent: %+v", state)
	}
	copy := reloaded.GetThreads()
	for i := range copy {
		if copy[i].RealtimeOutput != nil {
			copy[i].RealtimeOutput.ToolMode = "blocking"
		}
	}
	state, _ = persistentThreadByID(reloaded.GetThreads(), "voice-output")
	if *state.RealtimeOutput != output {
		t.Fatal("GetThreads aliased config")
	}
	parent := newTestThinker()
	defer parent.Stop()
	defer parent.threads.KillAll()
	parent.pool = thinker.pool
	if err := parent.threads.SpawnWithOpts("restored-output", "Answer calls.", nil, SpawnOpts{Realtime: true, DeferRun: true, ProviderName: provider.Name(), RealtimeOutput: realtimeOutputConfigValue(state.RealtimeOutput)}); err != nil {
		t.Fatal(err)
	}
	if parent.threads.threads["restored-output"].Realtime.opts.OutputConfig != output {
		t.Fatal("deferred runtime lost output policy")
	}
	if err := parent.threads.SpawnWithOpts("invalid-output", "Normal", nil, SpawnOpts{RealtimeOutput: output}); err == nil {
		t.Fatal("normal thread accepted realtime options")
	}
}

func TestRealtimeOutputConfigProviderSupportAndSettings(t *testing.T) {
	opts := RealtimeSessionOpts{Model: "gemini-3.8-live", Reasoning: "high", TranscriptionModel: "not-selectable", TurnDetection: RealtimeTurnDetectionConfig{Profile: "telephony"}, OutputConfig: RealtimeOutputConfig{ToolMode: "async"}, Tools: []NativeTool{{Name: "probe", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}}
	wire, err := buildGoogleLiveSetup(opts, "Kore")
	if err != nil {
		t.Fatal(err)
	}
	profile, err := googleLiveProfileWithOptions(opts)
	if err != nil || !profile.asyncTools || !strings.Contains(string(wire), `"behavior":"NON_BLOCKING"`) {
		t.Fatalf("async wire/runtime mismatch: %s %v", wire, err)
	}
	settings := googleRealtimeSettings(opts, wire)
	if settings["tool_mode"] != "async" || len(settings["ignored"].(map[string]string)) != 2 {
		t.Fatalf("settings=%v", settings)
	}
	if strings.Contains(string(mustJSON(t, settings)), "systemInstruction") {
		t.Fatal("settings exposed prompt")
	}
	for _, model := range []string{"gemini-3.1-flash-live-preview", "gemini-3.8-live-extended-thinking"} {
		opts.Model = model
		opts.OutputConfig.ToolMode = "async"
		if strings.Contains(model, "extended") {
			opts.OutputConfig.ToolMode = "blocking"
		}
		if _, err := buildGoogleLiveSetup(opts, "Kore"); err == nil {
			t.Fatalf("accepted unsupported mode for %s", model)
		}
	}
	for _, build := range []func(RealtimeSessionOpts, string) ([]byte, error){buildSessionUpdate, buildXAISessionUpdate} {
		if _, err := build(RealtimeSessionOpts{OutputConfig: RealtimeOutputConfig{ToolMode: "async"}}, ""); err == nil {
			t.Fatal("unsupported async mode silently ignored")
		}
		if _, err := build(RealtimeSessionOpts{OutputConfig: RealtimeOutputConfig{SpeechGuard: RealtimeSpeechGuardConfig{Mode: "transcript_prefix"}}}, ""); err == nil {
			t.Fatal("unsupported prefix guard silently ignored")
		}
	}
	if _, err := (RealtimeOutputConfig{SpeechGuard: RealtimeSpeechGuardConfig{PrefixBytes: -1}}).normalized(); err == nil {
		t.Fatal("accepted negative prefix")
	}
}

func TestRealtimeTimingGoogleLateTranscriptAndGuardOverflow(t *testing.T) {
	for _, overflow := range []bool{false, true} {
		t.Run(map[bool]string{false: "late_transcript", true: "overflow"}[overflow], func(t *testing.T) {
			s := newGoogleRealtimeTestSession()
			var result map[string]any
			s.timing = newRealtimeTiming(1, func(_ string, d map[string]any) { result = d })
			s.guardConfig = RealtimeSpeechGuardConfig{MaxBufferedAudioMS: 100, PrefixBytes: 48}
			at := time.Now()
			s.receivedAt = at
			s.translate(speechTestContent(make([]byte, 4800), "", false))
			if len(s.events) != 0 {
				t.Fatal("unchecked audio escaped guard")
			}
			if overflow {
				s.translate(speechTestContent([]byte{1, 2}, "", false))
			} else {
				s.translate(speechTestContent(nil, "Short", false))
				if len(s.speech.pending) != 1 {
					t.Fatal("short fragment released audio")
				}
				s.translate(speechTestContent(nil, " clean answer.", true))
			}
			for len(s.events) > 0 {
				e := <-s.events
				if e.Type == RealtimeEventOutputBlocked {
					s.timing.finish(e.ResponseID, e.ItemID, "blocked")
				}
				if e.Type == RealtimeEventResponseDone {
					s.timing.finish(e.ResponseID, "", "completed")
				}
			}
			if overflow {
				if result == nil || result["completion_status"] != "blocked" || result["dropped_audio_bytes"] != 4802 || result["buffered_audio_ms"] != float64(100) {
					t.Fatalf("overflow summary=%v", result)
				}
			} else {
				if result == nil || result["provider_audio_to_guard_release_ms"] == nil || result["speech_end_to_provider_audio_ms"] != nil {
					t.Fatalf("late transcript timing=%v", result)
				}
			}
		})
	}
}

func TestRealtimeTimingOpenAIProviderSpeechEndAndReady(t *testing.T) {
	s := &openaiRealtimeSession{events: make(chan RealtimeEvent, 8), done: make(chan struct{}), timing: newRealtimeTiming(1, nil)}
	at := time.Now()
	s.receivedAt = at
	s.translate(&openaiRealtimeEvent{Type: "input_audio_buffer.speech_stopped", AudioEndMS: 900})
	if e := <-s.events; e.Type != RealtimeEventSpeechStopped || e.AudioEndMS != 900 {
		t.Fatalf("missing VAD mapping: %v", e)
	}
	s.receivedAt = at.Add(time.Second)
	s.translate(&openaiRealtimeEvent{Type: "response.audio.delta", ResponseID: "r", ItemID: "i", Delta: "AQI="})
	s.timing.mu.Lock()
	u := s.timing.items["i"]
	delay := timingMS(u.speechEnd, u.firstAudio)
	s.timing.mu.Unlock()
	if delay != float64(1000) || u.speechSource != "provider_vad_event_received" {
		t.Fatalf("bad VAD receipt timing %v", u)
	}
	<-s.events
	s.translate(&openaiRealtimeEvent{Type: "session.updated", Session: map[string]any{"model": "model", "instructions": "private"}})
	e := <-s.events
	if e.Type != RealtimeEventSessionReady || strings.Contains(string(mustJSON(t, e.Settings)), "private") {
		t.Fatalf("bad setup ack: %v", e)
	}
}

func TestGoogleStandardAsyncUsesTurnCompleteLifecycle(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	profile, err := googleLiveProfileWithOptions(RealtimeSessionOpts{Model: "gemini-3.8-live", OutputConfig: RealtimeOutputConfig{ToolMode: "async"}})
	if err != nil {
		t.Fatal(err)
	}
	s.profile = profile
	if !s.profile.asyncTools || s.profile.interactionStatus {
		t.Fatal("tool behavior was coupled to Extended Thinking lifecycle")
	}
	events := google38Translate(t, s, `{"toolCall":{"functionCalls":[{"id":"a","name":"probe","args":{}}]}}`)
	if len(events) != 2 || events[0].Type != RealtimeEventToolCall || events[1].Type != RealtimeEventResponseDone {
		t.Fatalf("function selection did not finish: %v", events)
	}
	if err := s.SendToolResult("a", "ORANGE", false); err != nil {
		t.Fatal(err)
	}
	if len(s.outbox) != 1 || len(s.pendingResponses) != 0 {
		t.Fatal("async result was queued for a synchronous flush")
	}
	frame := <-s.outbox
	if !strings.Contains(string(frame.data), `"scheduling":"WHEN_IDLE"`) {
		t.Fatalf("standard async omitted result scheduling: %s", frame.data)
	}
	events = google38Translate(t, s, `{"serverContent":{"outputTranscription":{"text":"Orange."},"turnComplete":true}}`)
	if len(events) == 0 || events[len(events)-1].Type != RealtimeEventResponseDone || s.activeResponse() {
		t.Fatalf("standard async waited for interactionStatus: %v", events)
	}
}

func TestRealtimeTimingSameEnvelopeLeakIsObservedBeforeGuard(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	var summary map[string]any
	s.timing = newRealtimeTiming(1, func(_ string, data map[string]any) { summary = data })
	s.translate(speechTestContent([]byte{1, 2}, "Private reasoning: I should respond now.", true))
	for len(s.events) > 0 {
		event := <-s.events
		if event.Type == RealtimeEventOutputBlocked {
			s.timing.finish(event.ResponseID, event.ItemID, "blocked")
		}
		if event.Type == RealtimeEventAudioOut {
			t.Fatal("leaked audio escaped")
		}
	}
	if summary == nil || summary["completion_status"] != "blocked" || summary["dropped_audio_bytes"] != 2 {
		t.Fatalf("guard preceded raw audio observation: %v", summary)
	}
}

func TestRealtimeTimingFailureAndGenerationIsolation(t *testing.T) {
	var old, fresh map[string]any
	first := newRealtimeTiming(1, func(_ string, d map[string]any) { old = d })
	second := newRealtimeTiming(2, func(_ string, d map[string]any) { fresh = d })
	at := time.Now()
	for _, tracker := range []*realtimeTiming{first, second} {
		tracker.audio("r", "i", at)
		tracker.enqueued("r", "i", at, 1, 480)
	}
	first.written("r", "i", time.Time{}, 480, false)
	first.finish("r", "i", "completed")
	if old == nil || old["completion_status"] != "output_dropped" || old["output_enqueue_to_bridge_write_ms"] != nil {
		t.Fatalf("failed write became playback: %v", old)
	}
	second.finish("r", "i", "session_closed")
	if fresh == nil || fresh["generation"] != uint64(2) || fresh["dropped_audio_bytes"] != 480 || fresh["completion_status"] != "session_closed" {
		t.Fatalf("lost queued bytes at close: %v", fresh)
	}
}

func TestRealtimeTimingTranscriptOnlyAndCoreRejectedUtterances(t *testing.T) {
	for _, blocked := range []bool{false, true} {
		t.Run(map[bool]string{false: "transcript_only", true: "core_rejection"}[blocked], func(t *testing.T) {
			thinker := newTestThinker()
			defer thinker.Stop()
			rt := newRealtimeThinker(nil, thinker, &fakeRealtimeProvider{}, "", nil, nil, nil)
			defer rt.cancel()
			var summaries []map[string]any
			rt.timing = newRealtimeTiming(1, func(kind string, d map[string]any) {
				if kind == "realtime.utterance_timing" {
					summaries = append(summaries, d)
				}
			})
			text := "A clean answer."
			if blocked {
				text = "Private reasoning: I should respond now."
			}
			rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ResponseID: "r", ItemID: "i", Transcript: text, Final: true})
			rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "r"})
			want := "completed"
			if blocked {
				want = "blocked"
			}
			if len(summaries) != 1 || summaries[0]["completion_status"] != want || summaries[0]["provider_audio_to_guard_release_ms"] != nil {
				t.Fatalf("missing/misleading no-audio summary: %v", summaries)
			}
		})
	}
}

func TestRealtimeTimingCancellationAfterGenerationBeforePlayback(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	output := make(chan RealtimeAudioFrame, 1)
	rt := newRealtimeThinker(nil, thinker, &fakeRealtimeProvider{}, "", nil, output, nil)
	defer rt.cancel()
	var summary map[string]any
	rt.timing = newRealtimeTiming(1, func(_ string, d map[string]any) { summary = d })
	rt.timing.audio("r", "i", time.Now())
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ResponseID: "r", ItemID: "i", Audio: make([]byte, 480)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "r"})
	if summary != nil {
		t.Fatal("summary preceded bridge write/cancellation")
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventSpeechStarted})
	if summary == nil || summary["completion_status"] != "interrupted" || summary["dropped_audio_bytes"] != 480 || len(output) != 0 {
		t.Fatalf("cancellation became bridge failure: %v", summary)
	}
}

func TestRealtimeStandardAsyncSpeechDoesNotFinishPendingToolWork(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	rt := newRealtimeThinker(nil, thinker, &fakeRealtimeProvider{}, "", nil, nil, nil)
	defer rt.cancel()
	s := newGoogleRealtimeTestSession()
	s.profile, _ = googleLiveProfileWithOptions(RealtimeSessionOpts{Model: "gemini-3.8-live", OutputConfig: RealtimeOutputConfig{ToolMode: "async"}})
	rt.replaceSession(s)
	defer s.Close()
	s.callNames["slow"] = "probe"
	rt.beginToolCall(RealtimeEvent{ResponseID: "tool-selection", ToolCallID: "slow", ToolName: "probe"}, s)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "progress-speech"})
	if !rt.pendingToolWork() || rt.state != "working" {
		t.Fatalf("progress speech falsely finished pending work: state=%s pending=%v", rt.state, rt.pendingToolWork())
	}
	rt.submitToolResult(s, "slow", "probe_unavailable", true)
	if rt.pendingToolWork() || len(s.outbox) != 1 || len(s.pendingResponses) != 0 {
		t.Fatal("async error result stalled completion")
	}
	frame := <-s.outbox
	if !strings.Contains(string(frame.data), `"error":"probe_unavailable"`) {
		t.Fatalf("error became success: %s", frame.data)
	}
	if !strings.Contains(string(frame.data), `"scheduling":"WHEN_IDLE"`) {
		t.Fatal("error result omitted async delivery policy")
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "final-speech"})
	if rt.state != "listening" {
		t.Fatalf("final speech did not finish work: %s", rt.state)
	}
}
