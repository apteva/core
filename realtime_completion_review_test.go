package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type completionReviewProvider struct {
	LLMProvider
	answer  string
	release chan struct{}
	calls   atomic.Int32
}

func (p *completionReviewProvider) Chat(ctx context.Context, m []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.calls.Add(1)
	if len(tools) != 0 || len(m) != 2 || !strings.Contains(m[0].Content, "untrusted transcript") {
		return ChatResponse{}, fmt.Errorf("unsafe review request")
	}
	if p.release != nil {
		select {
		case <-p.release:
		case <-ctx.Done():
			return ChatResponse{}, ctx.Err()
		}
	}
	return ChatResponse{Text: p.answer}, nil
}
func reviewFixture(t *testing.T) (*RealtimeThinker, *completionReviewProvider) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, CoreExecution: true, ProviderInterruption: true}
	rt.opts.TurnDetection.Profile = RealtimeTurnProfileTelephony
	rt.toolAllowlist = map[string]bool{"done": true}
	provider := &completionReviewProvider{LLMProvider: rt.Thinker.provider, answer: `{"completed":true}`}
	rt.backend.provider = provider
	rt.messages = []Message{{Role: "user", Content: "Merci, au revoir."}, {Role: "assistant", Content: "Au revoir."}}
	rt.completionReview = realtimeCompletionReview{revision: 2, lastTranscript: time.Now().Add(-2 * time.Second), caller: true, assistant: true}
	return rt, provider
}
func settleReview(t *testing.T, rt *RealtimeThinker) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for rt.completionReview.result != nil && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
		rt.pollCompletionReview(time.Now())
	}
	if rt.completionReview.result != nil {
		t.Fatal("review stuck")
	}
}
func TestCompletionReviewExactlyOnceNoSecondFarewell(t *testing.T) {
	rt, p := reviewFixture(t)
	rt.pollCompletionReview(time.Now())
	settleReview(t, rt)
	if !rt.completionRequested() || rt.completion.awaitOutput || rt.terminalReason != "completion_review" {
		t.Fatal("completion did not finish existing farewell", rt.completion)
	}
	for i := 0; i < 10; i++ {
		rt.pollCompletionReview(time.Now())
	}
	if p.calls.Load() != 1 {
		t.Fatal("repeated review")
	}
	rt.finishRealtimeCompletion(rt.currentSession())
	if rt.completion.outcome != "completed" {
		t.Fatal(rt.completion.outcome)
	}
}
func TestCompletionReviewRejectsStaleAndUnsafeDecisions(t *testing.T) {
	for _, change := range []string{"new_transcript", "new_voice", "generation", "directive", "backend", "permission", "paused", "disconnected", "playback", "cancelled_farewell", "bridge_closed", "input_backlog"} {
		t.Run(change, func(t *testing.T) {
			rt, p := reviewFixture(t)
			p.release = make(chan struct{})
			rt.pollCompletionReview(time.Now())
			switch change {
			case "new_transcript":
				rt.noteCompletionTranscript(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "Wait, another question"})
			case "new_voice":
				rt.lastCallerVoice.Store(time.Now().UnixNano())
			case "generation":
				rt.sessionGeneration++
			case "directive":
				rt.directive += "changed"
			case "backend":
				rt.backend.active = true
			case "permission":
				rt.toolAllowlist["done"] = false
			case "paused":
				rt.paused = true
			case "disconnected":
				rt.replaceSession(nil)
			case "cancelled_farewell":
				rt.completionReview.lastAssistantPlayback = "farewell"
				rt.cancelledPlaybacks = map[string]bool{"farewell": true}
			case "input_backlog":
				rt.configureInputQueue()
				rt.input.push(make([]byte, 48000), time.Now())
			case "bridge_closed":
				rt.bridgeConnected = false
			case "playback":
				rt.delivery.enqueue("1/farewell", "farewell", 960)
			}
			close(p.release)
			settleReview(t, rt)
			if rt.completionRequested() {
				t.Fatal("stale/unsafe result ended conversation")
			}
		})
	}
}
func TestCompletionReviewWaitsForPlaybackAndDoesNotBlockAudio(t *testing.T) {
	rt, p := reviewFixture(t)
	rt.delivery.playbackReady()
	written := rt.delivery.enqueue("1/farewell", "farewell", 4800)
	rt.pollCompletionReview(time.Now())
	if p.calls.Load() != 0 {
		t.Fatal("review before delivery")
	}
	written(true)
	rt.pollCompletionReview(time.Now())
	if p.calls.Load() != 0 {
		t.Fatal("socket write treated as playback")
	}
	rt.delivery.acknowledge("1/farewell", 100)
	p.release = make(chan struct{})
	rt.pollCompletionReview(time.Now())
	started := time.Now()
	rt.configureInputQueue()
	for i := 0; i < 10; i++ {
		rt.admitRealtimeAudio(make([]byte, 960), time.Now())
	}
	if time.Since(started) > 100*time.Millisecond || !rt.input.pending() {
		t.Fatal("review blocked microphone")
	}
	close(p.release)
	settleReview(t, rt)
	if !rt.completionRequested() {
		t.Fatal("acknowledged closing did not finish")
	}
}
func TestCompletionReviewInvalidAndNegativeResultsLeaveCallOpen(t *testing.T) {
	for _, answer := range []string{`{"completed":false}`, `{}`, `not json`, `{"completed":"true"}`} {
		t.Run(answer, func(t *testing.T) {
			rt, p := reviewFixture(t)
			p.answer = answer
			rt.pollCompletionReview(time.Now())
			settleReview(t, rt)
			if rt.completionRequested() {
				t.Fatal("invalid review ended call")
			}
		})
	}
}
func TestCompletionReviewOptInAndBoundedContext(t *testing.T) {
	rt, _ := reviewFixture(t)
	rt.opts.TurnDetection.Profile = ""
	if rt.completionReviewEnabled() {
		t.Fatal("non-phone opted in implicitly")
	}
	rt.opts.OutputConfig.CompletionMode = "semantic"
	if !rt.completionReviewEnabled() {
		t.Fatal("explicit opt in ignored")
	}
	rt.opts.OutputConfig.CompletionMode = "delegation"
	if rt.completionReviewEnabled() {
		t.Fatal("opt out ignored")
	}
	rt.messages = append(rt.messages, Message{Role: "user", Content: "secret-tool-data", RealtimeBackend: true}, Message{Role: "system", Content: "secret-system"})
	messages := rt.completionReviewMessages()
	if strings.Contains(messages[1].Content, "secret") {
		t.Fatal("review received operational context")
	}
	if _, err := (RealtimeOutputConfig{CompletionMode: "invalid"}).normalized(); err == nil {
		t.Fatal("invalid mode accepted")
	}
}
func TestEffectiveTurnDetectionTelemetryContinuousAndTraditional(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.opts.TurnDetection.Profile = RealtimeTurnProfileTelephony
	for _, continuous := range []bool{false, true} {
		rt.capabilities = RealtimeCapabilities{ContinuousConversation: continuous, ProviderInterruption: continuous}
		raw, _ := json.Marshal(rt.effectiveTurnDetectionTelemetry())
		if continuous && (strings.Contains(string(raw), "750") || strings.Contains(string(raw), "silence_duration_ms")) {
			t.Fatal("fabricated continuous VAD", string(raw))
		}
		if !continuous && !strings.Contains(string(raw), `"silence_duration_ms":750`) {
			t.Fatal("traditional VAD lost", string(raw))
		}
	}
}

func TestCompletionReviewRetriesAreBounded(t *testing.T) {
	rt, p := reviewFixture(t)
	p.answer = "invalid"
	for i := 0; i < 5; i++ {
		rt.completionReview.retryAt = time.Time{}
		rt.pollCompletionReview(time.Now())
		settleReview(t, rt)
	}
	if p.calls.Load() != 2 || rt.completionRequested() {
		t.Fatal("unbounded retries or unsafe completion", p.calls.Load())
	}
}

func TestCompletionCloseDistinguishesRecoveryFromHangup(t *testing.T) {
	for _, outcome := range []string{"completed", "timed_out", "disconnected", "forced_stop", ""} {
		rt, _ := completionFixture(t)
		rt.requestCompletionWithClosing("review", false)
		rt.setCompletionOutcome(outcome)
		reason := rt.completionCloseReason()
		want := outcome == "completed" || outcome == "timed_out"
		if (reason != "") != want {
			t.Fatal(outcome, reason)
		}
		ctx, cancel := context.WithCancelCause(context.Background())
		thread := "completion-close-" + outcome
		audioBridgeMu.Lock()
		audioBridgeLive[thread] = &audioBridgeConnection{cancel: cancel}
		audioBridgeMu.Unlock()
		unregisterAudioBridge(thread, reason)
		end := classifyRealtimeBridgeError("context", "close", context.Cause(ctx))
		if want && end.data.CloseReason != reason {
			t.Fatal("completion reason lost", end)
		}
		if !want && end.data.CloseReason != "audio bridge unregistered" {
			t.Fatal("ordinary shutdown became hangup", end)
		}
		if rt.shutdownData()["initiator"] != "application_review" {
			t.Fatal("incorrect initiator")
		}
	}
}
