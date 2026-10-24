package core

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestRealtimeInputPostStallRecovery(t *testing.T) {
	for _, rate := range []float64{1, 1.25} {
		for session := 0; session < 4; session++ {
			t.Run(fmt.Sprintf("rate_%g/session_%d", rate, session), func(t *testing.T) {
				t.Parallel()
				rt, _ := completionFixture(t)
				rt.input.policy = defaultRealtimeInputPolicy(true)
				rt.input.policy.CatchUpRate = rate
				sink := newRecoveryTestSession()
				epoch := time.Now()
				var expected []byte
				var peaks [2]time.Duration
				var recovered [2]time.Duration
				var ages []string
				for ms := 0; ms < 9000; ms++ {
					now := epoch.Add(time.Duration(ms) * time.Millisecond)
					// Continuous ordered PCM with deterministic 0..6ms arrival jitter.
					nominal := ms - ms%20
					if ms%20 == (nominal/20+session)%7 {
						frame := bytes.Repeat([]byte{byte(nominal / 20), byte(session)}, 480)
						expected = append(expected, frame...)
						rt.input.push(frame, now)
					}
					stalled := (ms >= 1000 && ms < 1400) || (ms >= 5000 && ms < 5400)
					if !stalled {
						rt.drainRealtimeInputAt(sink, now)
					}
					age := time.Duration(0)
					if len(rt.input.frames) > 0 {
						age = now.Sub(rt.input.frames[0].received)
					}
					if ms%500 == 0 {
						ages = append(ages, fmt.Sprintf("%d:%d", ms, age.Milliseconds()))
					}
					for i, end := range []int{1400, 5400} {
						if ms >= end && ms < end+3000 {
							if age > peaks[i] {
								peaks[i] = age
							}
							if recovered[i] == 0 && age <= 60*time.Millisecond {
								recovered[i] = time.Duration(ms-end) * time.Millisecond
							}
						}
					}
				}
				for ms := 9000; rt.input.pending() && ms < 12000; ms++ {
					rt.drainRealtimeInputAt(sink, epoch.Add(time.Duration(ms)*time.Millisecond))
				}
				if !bytes.Equal(expected, sink.audio) || len(rt.input.drops) != 0 {
					t.Fatal("lost, reordered or duplicated PCM")
				}
				if rate > 1 {
					for i, r := range recovered {
						if r == 0 || r > 2500*time.Millisecond {
							t.Fatalf("stall %d failed to recover: %v, ages=%v", i, r, ages)
						}
					}
				} else if recovered[0] != 0 || recovered[1] != 0 {
					t.Fatal("strict pacing baseline unexpectedly recovered")
				}
				t.Logf("peak=%v recovery_after_stall=%v queue_age_ms=%v", peaks, recovered, ages)
			})
		}
	}
}
func TestRealtimeDelegationEvidenceAcrossSilence(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name  string
		b     realtimeBackend
		ready bool
	}{
		{"speech_then_silence", realtimeBackend{lastInput: now.Add(-time.Second), inputEndMS: 33000, offsetMS: 34400, requestedAt: now}, true},
		{"late_caption", realtimeBackend{lastInput: now, inputEndMS: 33000, offsetMS: 34400}, false},
		{"partial_settled", realtimeBackend{lastInput: now.Add(-300 * time.Millisecond)}, true},
		{"provider_final", realtimeBackend{lastInput: now, inputFinal: true}, true},
		{"no_evidence", realtimeBackend{}, false},
		{"new_speech_old_caption", realtimeBackend{lastInput: now.Add(-time.Second), speechPending: true}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.b.evidenceReady(now) != tc.ready {
				t.Fatal("incorrect evidence readiness")
			}
		})
	}
	rt, _ := completionFixture(t)
	rt.capabilities.CoreExecution = true
	rt.acceptDelegation(RealtimeEvent{DelegationID: "first"})
	rev := rt.backend.revision
	rt.acceptDelegation(RealtimeEvent{DelegationID: "first"})
	if rt.backend.revision != rev {
		t.Fatal("duplicate delegation revised work")
	}
	rt.noteBackendInput(RealtimeEvent{Transcript: "Actually, Tuesday.", Final: true})
	if rt.backend.revision != rev+1 || !rt.backend.want {
		t.Fatal("late correction not applied")
	}
}

func streamLatencyFixture(t *testing.T) (*RealtimeThinker, *fakeRealtimeSession, *Thinker, ThreadWork) {
	rt, voice := completionFixture(t)
	rt.threadID = rt.sub.ID
	rt.backend.mode = "thread"
	worker := newTestThinker()
	t.Cleanup(worker.Stop)
	worker.threadID = "worker"
	worker.work = &threadWorkState{}
	worker.owner = NewThreadManager(rt.Thinker)
	w := ThreadWork{ID: "request", Revision: 1, Origin: rt.threadID, Kind: "request", DelegationID: "delegation", StreamResults: true}
	worker.work.observe(&w)
	// Publish through the shared bus as production child work does.
	worker.bus = rt.bus
	rt.backend.worker = &Thread{ID: "worker", Thinker: worker}
	rt.backend.active = true
	rt.backend.requestID = w.ID
	rt.backend.revision = w.Revision
	rt.backend.delegation = w.DelegationID
	return rt, voice, worker, w
}
func TestRealtimeVerifiedSentenceBeforeCompletion(t *testing.T) {
	rt, voice, worker, w := streamLatencyFixture(t)
	sentence := "Le tarif confirmé est de 42 euros."
	stream := worker.newThreadWorkStream([]Message{{Role: "user", ToolResults: []ToolResult{{CallID: "price", Content: `{"answer":"` + sentence + `"}`}}}})
	stream.append("Le tarif confirmé est de ")
	if len(rt.sub.DrainTargeted()) != 0 {
		t.Fatal("partial sentence escaped")
	}
	stream.append("42 euros. Vous pouvez")
	events := rt.sub.DrainTargeted()
	if len(events) != 1 || events[0].Work.DelegationID != w.DelegationID {
		t.Fatalf("no correlated chunk: %+v", events)
	}
	rt.consumeThreadWork(events[0])
	rt.consumeThreadWork(events[0])
	if len(voice.texts) != 1 || !rt.backend.active || voice.texts[0].Content != sentence {
		t.Fatal("first verified sentence did not precede worker completion or duplicate escaped")
	}
	result := w
	result.Kind = "result"
	result.Worker = "worker"
	result.Status = "completed"
	final := Event{Work: &result, Text: sentence + " Vous pouvez payer sur place."}
	rt.consumeThreadWork(final)
	rt.consumeThreadWork(final)
	if len(voice.texts) != 2 || strings.Contains(voice.texts[1].Content, "42 euros") || rt.backend.active {
		t.Fatalf("final result not deduplicated: %+v", voice.texts)
	}
}
func TestRealtimeStreamExcludesUnverifiedFailedAndStaleText(t *testing.T) {
	for _, tc := range []struct {
		name, content, text    string
		failed, preview, stale bool
	}{
		{name: "failed", content: "La réservation est confirmée.", text: "La réservation est confirmée. ", failed: true},
		{name: "unconfirmed", content: `{"status":"pending"}`, text: "La réservation est confirmée. "},
		{name: "preview", content: "La réservation est confirmée.", text: "La réservation est confirmée. ", preview: true},
		{name: "stale", content: "Le tarif est de 42 euros.", text: "Le tarif est de 42 euros. ", stale: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt, voice, worker, w := streamLatencyFixture(t)
			stream := worker.newThreadWorkStream([]Message{{Role: "user", ToolResults: []ToolResult{{Content: tc.content, IsError: tc.failed, ContentIsPreview: tc.preview}}}})
			if tc.stale {
				w.Revision++
				worker.work.advance(w)
			}
			stream.append(tc.text)
			for _, e := range rt.sub.DrainTargeted() {
				rt.consumeThreadWork(e)
			}
			if len(voice.texts) != 0 {
				t.Fatal("unsafe stream escaped")
			}
		})
	}
}

type latencyStreamingProvider struct {
	scriptedRetryProvider
	release  chan struct{}
	sentence string
}

func (p *latencyStreamingProvider) SupportsNativeTools() bool { return true }
func (p *latencyStreamingProvider) Chat(ctx context.Context, _ []Message, _ string, _ []NativeTool, onText, onThinking func(string), onTool func(string, string, string)) (ChatResponse, error) {
	onThinking("private reasoning: " + p.sentence + " ")
	onTool("pending", "lookup", `{"incomplete":`)
	onText(p.sentence + " ")
	select {
	case <-ctx.Done():
		return ChatResponse{}, ctx.Err()
	case <-p.release:
	}
	return ChatResponse{Text: p.sentence}, nil
}
func TestRealtimeStreamThroughProviderCallbackBeforeCompletion(t *testing.T) {
	rt, voice, worker, _ := streamLatencyFixture(t)
	sentence := "Le tarif confirmé est de 42 euros."
	provider := &latencyStreamingProvider{scriptedRetryProvider: scriptedRetryProvider{name: "fixture"}, release: make(chan struct{}), sentence: sentence}
	messages := []Message{{Role: "system", Content: "Return verified result"}, {Role: "assistant", ToolCalls: []NativeToolCall{{ID: "price", Name: "lookup", Args: map[string]string{}}}}, {Role: "user", ToolResults: []ToolResult{{CallID: "price", Content: sentence}}}}
	done := make(chan error, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _, err := worker.thinkWithProviderMessages(ctx, provider, messages); done <- err }()
	deadline := time.After(time.Second)
	for len(voice.texts) == 0 {
		select {
		case err := <-done:
			t.Fatalf("worker completed before delivery: %v", err)
		case <-deadline:
			t.Fatal("first complete verified sentence did not reach voice")
		case <-rt.sub.Wake:
			for _, e := range rt.sub.DrainTargeted() {
				rt.consumeThreadWork(e)
			}
		}
	}
	if len(voice.texts) != 1 || voice.texts[0].Content != sentence {
		t.Fatal("reasoning or partial tool arguments escaped")
	}
	close(provider.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
func TestRealtimeChunksIgnoreLateAndSupersededDelivery(t *testing.T) {
	rt, voice, _, w := streamLatencyFixture(t)
	w.Kind = "result"
	w.Worker = "worker"
	w.Status = "completed"
	rt.consumeThreadWork(Event{Work: &w, Text: "Le tarif est de 42 euros."})
	w.Kind = "result_chunk"
	w.Status = "verified"
	w.ChunkID = "late"
	rt.consumeThreadWork(Event{Work: &w, Text: "Le tarif est de 42 euros."})
	w.Revision = 0
	rt.consumeThreadWork(Event{Work: &w, Text: "Le tarif est de 99 euros."})
	if len(voice.texts) != 1 {
		t.Fatal("late or superseded result escaped")
	}
}
func TestRealtimeFinalDedupKeepsChangedSentence(t *testing.T) {
	prior := "The booking is confirmed."
	changed := "It is not true that The booking is confirmed."
	if removeDeliveredWorkSentences(changed, prior) != changed {
		t.Fatal("dedup changed correction semantics")
	}
	if got := removeDeliveredWorkSentences(prior+" Bring your ID.", prior); got != "Bring your ID." {
		t.Fatalf("dedup: %q", got)
	}
}

func TestRealtimeChunkInterruptionKeepsBackendAndFinalEvidence(t *testing.T) {
	rt, voice, worker, w := streamLatencyFixture(t)
	sentence := "Le tarif confirmé est de 42 euros."
	rt.callerSpeechPending.Store(true)
	stream := worker.newThreadWorkStream([]Message{{Role: "user", ToolResults: []ToolResult{{Content: sentence}}}})
	stream.append(sentence + " ")
	for _, e := range rt.sub.DrainTargeted() {
		rt.consumeThreadWork(e)
	}
	if len(voice.texts) != 1 || voice.responses != 0 || !rt.backend.active {
		t.Fatal("interruption lost evidence, requested speech or cancelled backend")
	}
	rt.callerSpeechPending.Store(false)
	// Reconstruct delivery state from persisted messages as an inbox replay does.
	rt.backend.deliveredWork = nil
	final := w
	final.Kind = "result"
	final.Worker = "worker"
	rt.consumeThreadWork(Event{Work: &final, Text: sentence})
	if len(voice.texts) != 2 || voice.texts[1].Content != sentence {
		t.Fatal("quiet evidence incorrectly removed from spoken final")
	}
}
func TestRealtimeStreamReplayDedupFromHistory(t *testing.T) {
	rt, voice, worker, w := streamLatencyFixture(t)
	sentence := "Le tarif confirmé est de 42 euros."
	stream := worker.newThreadWorkStream([]Message{{Role: "user", ToolResults: []ToolResult{{Content: sentence}}}})
	stream.append(sentence + " ")
	events := rt.sub.DrainTargeted()
	rt.consumeThreadWork(events[0])
	rt.backend.deliveredWork = nil
	rt.consumeThreadWork(events[0])
	final := w
	final.Kind = "result"
	final.Worker = "worker"
	rt.consumeThreadWork(Event{Work: &final, Text: sentence})
	if len(voice.texts) != 1 {
		t.Fatal("history replay duplicated delivered sentence")
	}
}
