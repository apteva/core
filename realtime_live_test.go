package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

func TestOpenAILiveUsesExistingAPIIntegration(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "fixture-key")
	cfg := &Config{RealtimeEnabled: true, Providers: []ProviderConfig{{Name: "openai", Default: true, RealtimeVoice: "brise", RealtimeModels: map[string]string{"large": "gpt-live-1"}}}}
	pool, err := buildProviderPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := pool.RealtimeByName("openai").(*OpenAIRealtimeProvider)
	if !ok || p.apiKey != "fixture-key" || p.Models()[ModelLarge] != "gpt-live-1" || p.DefaultVoice() != "brise" {
		t.Fatalf("integration not reused: %#v", p)
	}
	if !capabilitiesForRealtime(p, "gpt-live-1").CoreExecution || capabilitiesForRealtime(p, "gpt-realtime-2.1").CoreExecution {
		t.Fatal("model capabilities do not distinguish wire protocols")
	}
	copy := cfg.GetProviders()
	copy[0].RealtimeModels["large"] = "mutated"
	if cfg.GetProviders()[0].RealtimeModels["large"] != "gpt-live-1" {
		t.Fatal("provider config aliases mutable map")
	}
	cfg.Providers = append(cfg.Providers, ProviderConfig{Name: "openai-realtime", Models: map[string]string{"large": "gpt-realtime-2.1"}})
	pool, err = buildProviderPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if pool.RealtimeByName("openai").Models()[ModelLarge] != "gpt-live-1" {
		t.Fatal("legacy server companion masked explicit primary-integration voice selection")
	}
}

func TestOpenAILiveWebSocketStartupAudioContextAndFinalUsage(t *testing.T) {
	received := make(chan map[string]any, 16)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/live/sessions" || r.URL.RawQuery != "" || r.Header.Get("Authorization") != "Bearer same-key" {
			t.Errorf("incorrect Live routing: %s", r.URL.String())
			w.WriteHeader(400)
			return
		}
		conn, _, _, err := ws.UpgradeHTTP(r, w)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			data, _, err := wsutil.ReadClientData(conn)
			if err != nil {
				return
			}
			var m map[string]any
			if json.Unmarshal(data, &m) != nil {
				return
			}
			received <- m
			switch m["type"] {
			case "session.start":
				_ = wsutil.WriteServerMessage(conn, ws.OpText, []byte(`{"type":"session.started","session":{"id":"live_fixture"}}`))
			case "session.close":
				_ = wsutil.WriteServerMessage(conn, ws.OpText, []byte(`{"type":"session.closed","reason":"close_requested","usage":{"seconds":12}}`))
				return
			}
		}
	}))
	defer server.Close()
	p := NewOpenAIRealtimeProvider("same-key")
	p.endpoint = strings.Replace(server.URL, "http", "ws", 1) + "/v1/realtime"
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	s, err := p.Open(ctx, RealtimeSessionOpts{Model: "gpt-live-1", Voice: "sillage", Instructions: "private Core prompt", ConversationInstructions: "Parlez français.", History: []Message{{Role: "user", Content: "Bonjour"}}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	start := <-received
	session := start["session"].(map[string]any)
	if start["type"] != "session.start" || session["delegation"].(map[string]any)["type"] != "client" || strings.Contains(session["instructions"].(string), "private Core prompt") {
		t.Fatalf("bad startup: %#v", start)
	}
	if session["audio"].(map[string]any)["output"].(map[string]any)["voice"] != "sillage" {
		t.Fatal("French voice lost")
	}
	if err := s.SendAudio([]byte{1, 2}); err != nil {
		t.Fatal(err)
	}
	if m := <-received; m["type"] != "session.input_audio.append" || m["audio"] != base64.StdEncoding.EncodeToString([]byte{1, 2}) {
		t.Fatalf("audio=%#v", m)
	}
	if err := s.(RealtimeContextSession).AppendContext(RealtimeContextUpdate{Kind: "commentary", DelegationID: "work-1", Content: "Le résultat est confirmé."}); err != nil {
		t.Fatal(err)
	}
	if m := <-received; m["type"] != "session.commentary.append" || m["delegation_id"] != "work-1" {
		t.Fatalf("context=%#v", m)
	}
	if err := s.RequestResponse(); err != nil {
		t.Fatal(err)
	}
	if err := s.(RealtimeGracefulSession).BeginClose(); err != nil {
		t.Fatal(err)
	}
	if m := <-received; m["type"] != "session.close" {
		t.Fatalf("unexpected voice response trigger: %#v", m)
	}
	usage := float64(0)
	ended := false
	for e := range s.Events() {
		if e.Type == RealtimeEventSessionUsage {
			usage += e.Usage.SessionSeconds
		}
		if e.Type == RealtimeEventSessionEnded {
			ended = e.CloseReason == "close_requested"
		}
	}
	if usage != 12 || !ended {
		t.Fatalf("usage=%v ended=%v", usage, ended)
	}
}

func TestOpenAILiveUsageIsDeltaAndFragmentsAreNeverFinal(t *testing.T) {
	s := &openaiLiveSession{openaiRealtimeSession: newTranslationSession(), ready: make(chan error, 1)}
	for _, seconds := range []float64{12, 15, 15, 14} {
		e := openaiLiveEvent{Type: "session.usage.updated"}
		e.Usage.Seconds = seconds
		s.translateLive(e)
	}
	s.translateLive(openaiLiveEvent{Type: "session.input_transcript.delta", Delta: "Bonjour ", StartMS: 100, EndMS: 500})
	var usage float64
	var transcript RealtimeEvent
	for len(s.events) > 0 {
		e := <-s.events
		if e.Type == RealtimeEventSessionUsage {
			usage += e.Usage.SessionSeconds
		}
		if e.Type == RealtimeEventTranscriptInput {
			transcript = e
		}
	}
	if usage != 15 || transcript.Final || !transcript.TranscriptDelta || transcript.TranscriptEndMS != 500 {
		t.Fatalf("usage=%v transcript=%#v", usage, transcript)
	}
	if got := calculateCostForRealtimeProvider(NewOpenAIRealtimeProvider(""), "gpt-live-1", RealtimeUsage{SessionSeconds: 60}); got != 0.05 {
		t.Fatalf("session price=%v", got)
	}
}

func TestContinuousTranscriptCheckpointsOverlapAndInterruptions(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true, ProviderInterruption: true}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "Je souhaite ", TranscriptDelta: true, TranscriptStartMS: 10, TranscriptEndMS: 250})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptOutput, Transcript: "Oui.", TranscriptDelta: true, TranscriptStartMS: 100, TranscriptEndMS: 300})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "vendredi.", TranscriptDelta: true, TranscriptStartMS: 250, TranscriptEndMS: 700})
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 2 || entries[0].Content != "Je souhaite vendredi." || entries[0].RealtimeTranscript.Status != "application_grouped" || entries[0].RealtimeTranscript.EndMS != 700 {
		t.Fatalf("entries=%#v", entries)
	}
	rt.flushPendingTranscripts("connection_lost")
	rt.flushPendingTranscripts("connection_lost")
	if entries = readSessionEntriesForTest(t, rt.Thinker.session.path); len(entries) != 2 || entries[0].RealtimeTranscript.Status == "provider_final" {
		t.Fatalf("duplicate/fake final: %#v", entries)
	}
}

func TestContinuousSpeechGuardBuffersRejectsAndRecovers(t *testing.T) {
	g := realtimeContinuousGuard{}
	now := time.Now()
	audio := RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now, Audio: []byte{255, 127, 255, 127}}
	if got := g.audio(audio, nil); len(got) != 0 {
		t.Fatal("audio escaped prefix guard")
	}
	if got := g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now, Transcript: "Bonjour."}, nil, nil); len(got) != 2 {
		t.Fatalf("clean speech not released: %#v", got)
	}
	if got := g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now, Transcript: " callto:secret{}"}, nil, nil); len(got) != 1 || got[0].Type != RealtimeEventOutputBlocked {
		t.Fatalf("technical speech escaped: %#v", got)
	}
	audio.ObservedAt = now.Add(2 * time.Second)
	if got := g.audio(audio, nil); len(got) != 0 {
		t.Fatalf("new prefix not buffered: %#v", got)
	}
	if got := g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: audio.ObservedAt, Transcript: "Reprenons."}, nil, nil); len(got) != 2 {
		t.Fatalf("guard did not recover: %#v", got)
	}
	if got := g.audio(RealtimeEvent{ObservedAt: audio.ObservedAt, Audio: make([]byte, realtimePCMBytesPerSecond)}, nil); len(got) != 0 {
		t.Fatal("silence became an utterance")
	}
}

type liveExecutionFixtureProvider struct{ scriptedRetryProvider }

func (p *liveExecutionFixtureProvider) SupportsNativeTools() bool { return true }
func (p *liveExecutionFixtureProvider) Chat(_ context.Context, m []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.messages = append(p.messages, cloneMessages(m))
	for _, message := range m {
		for _, r := range message.ToolResults {
			if r.CallID == "balance-1" {
				return ChatResponse{Text: "Votre solde confirmé est de 42 euros."}, nil
			}
		}
	}
	return ChatResponse{ToolCalls: []NativeToolCall{{ID: "balance-1", Name: "account_balance", Args: map[string]string{"_reason": "Consulter le solde"}}}}, nil
}

func TestRealtimeCoreDelegationUsesExistingToolDispatchOnce(t *testing.T) {
	rt, session := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, CoreExecution: true, TranscriptFragments: true}
	p := &liveExecutionFixtureProvider{scriptedRetryProvider: scriptedRetryProvider{name: "fixture"}}
	rt.Thinker.provider = p
	rt.pool = &ProviderPool{providers: map[string]LLMProvider{"fixture": p}, default_: "fixture"}
	rt.configureExecution()
	rt.registry = NewToolRegistry("")
	rt.toolAllowlist = map[string]bool{"account_balance": true}
	catalog := []mcpToolDef{mkTool("balance", "Read the caller's account balance")}
	rt.toolIndex.Add("account", catalog, false)
	registerTestMCPTools(rt.registry, "account", catalog)
	var calls int
	rt.handleTools = func(owner *Thinker, tc []toolCall, _ []string) ([]string, []string, []ToolResult) {
		if len(tc) == 0 {
			return nil, nil, nil
		}
		calls++
		owner.kickNextTurn = true
		return nil, nil, []ToolResult{{CallID: tc[0].NativeID, ToolName: tc[0].Name, Content: `{"balance":42,"currency":"EUR"}`}}
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "Quel est mon solde ?", TranscriptDelta: true, TranscriptEndMS: 800})
	e := RealtimeEvent{Type: RealtimeEventDelegation, DelegationID: "request-1", DelegationOffsetMS: 700}
	rt.handleSessionEvent(e)
	revision := rt.backend.revision
	rt.handleSessionEvent(e)
	if rt.backend.revision != revision {
		t.Fatal("duplicate delegation changed work")
	}
	rt.backend.lastInput = time.Now().Add(-time.Second)
	worker, err := rt.ensureExecutionThread()
	if err != nil {
		t.Fatal(err)
	}
	worker.Thinker.handleTools = rt.handleTools
	defer rt.stopExecutionThread()
	rt.maybeStartExecution()
	deadline := time.After(3 * time.Second)
	for rt.backend.active {
		select {
		case <-deadline:
			t.Fatal("backend failed to progress")
		case <-rt.sub.Wake:
			for _, event := range rt.sub.DrainTargeted() {
				rt.handleBusEvent(event)
			}
		}
	}

	if calls != 1 || len(session.toolResults) != 0 || session.responses != 0 {
		t.Fatalf("calls=%d native results=%d voice triggers=%d", calls, len(session.toolResults), session.responses)
	}
	if len(session.texts) != 1 || !strings.Contains(session.texts[0].Content, "42 euros") {
		t.Fatalf("verified result not returned: %#v", session.texts)
	}
	if err := validateToolHistory(rt.executionSnapshot()); err != nil {
		t.Fatal(err)
	}
	if rt.backend.worker.Thinker.bus != rt.bus {
		t.Fatal("execution bypassed shared request budget")
	}
}

func TestRealtimeCoreDelegationDiscardsStaleToolDecisions(t *testing.T) {
	state := &threadWorkState{}
	first := ThreadWork{ID: "one", Revision: 1, Origin: "voice", Kind: "request"}
	state.observe(&first)
	second := first
	second.Revision = 2
	state.advance(second)
	if state.current(first) || !state.current(second) {
		t.Fatal("stale tool admission accepted")
	}
}

func TestOpenAILiveContextSplitPreservesUTF8AndLimits(t *testing.T) {
	text := strings.Repeat("Éléonore 日本語 ", 100)
	parts := splitRealtimeContext(text, 480)
	if strings.Join(parts, "") != text {
		t.Fatal("context was truncated")
	}
	for _, part := range parts {
		if len(part) > 480 || !json.Valid([]byte(`"`+part+`"`)) {
			t.Fatal("invalid context chunk")
		}
	}
}

type gracefulLiveFixture struct {
	*fakeRealtimeSession
	closed chan struct{}
}

func (s *gracefulLiveFixture) BeginClose() error {
	close(s.closed)
	s.events <- RealtimeEvent{Type: RealtimeEventSessionUsage, Usage: RealtimeUsage{SessionSeconds: 3}}
	s.events <- RealtimeEvent{Type: RealtimeEventSessionEnded, CloseReason: "close_requested"}
	return nil
}

func TestContinuousCompletionDrainsPlaybackWithoutProviderDoneEvent(t *testing.T) {
	rt, _ := completionFixture(t)
	s := &gracefulLiveFixture{fakeRealtimeSession: newFakeRealtimeSession(), closed: make(chan struct{})}
	rt.replaceSession(s)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	rt.delivery.playbackReady()
	rt.requestCompletion("done-live")
	done := runCompletionFixture(rt)
	s.events <- RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "goodbye", Audio: make([]byte, 4800)}
	s.events <- RealtimeEvent{Type: RealtimeEventTranscriptOutput, ItemID: "goodbye", Transcript: "Au revoir.", TranscriptDelta: true}
	var frame RealtimeAudioFrame
	select {
	case frame = <-rt.audioOut:
	case <-time.After(time.Second):
		t.Fatal("closing audio lost")
	}
	select {
	case <-s.closed:
		t.Fatal("provider closed before socket write/playback")
	case <-time.After(150 * time.Millisecond):
	}
	frame.written(true)
	rt.acknowledgePlayback("goodbye", 100)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("continuous completion waited for nonexistent provider turn end")
	}
	if got := shutdownReceipt(t, rt)["graceful_completion_outcome"]; got != "completed" {
		t.Fatalf("completion=%v", got)
	}
	for _, e := range readSessionEntriesForTest(t, rt.Thinker.session.path) {
		if e.RealtimeTranscript != nil && e.RealtimeTranscript.Status == "provider_final" {
			t.Fatal("invented final transcript")
		}
	}
}

func TestContinuousOverlapDoesNotCancelSpeechOrTruncateProvider(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, ProviderInterruption: true}
	rt.rendererSpeechStarted()
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventSpeechStarted})
	if s.interrupts != 0 || len(s.truncations) != 0 {
		t.Fatal("continuous model's interruption policy was overridden")
	}
}

type correctionLiveProvider struct {
	scriptedRetryProvider
	release   chan struct{}
	cancelled chan struct{}
}

func (p *correctionLiveProvider) SupportsNativeTools() bool { return true }
func (p *correctionLiveProvider) Chat(ctx context.Context, m []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	p.calls++
	n := p.calls
	p.messages = append(p.messages, cloneMessages(m))
	p.mu.Unlock()
	if n == 1 {
		p.startedOne.Do(func() { close(p.started) })
		select {
		case <-p.release:
		case <-ctx.Done():
			close(p.cancelled)
			return ChatResponse{}, ctx.Err()
		}
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "stale-booking", Name: "book"}}}, nil
	}
	return ChatResponse{Text: "La demande actuelle concerne vendredi."}, nil
}

func TestContinuousCorrectionKeepsInferenceAliveAndRejectsItsOldAction(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, CoreExecution: true, TranscriptFragments: true, ProviderInterruption: true}
	p := &correctionLiveProvider{scriptedRetryProvider: scriptedRetryProvider{name: "fixture", started: make(chan struct{})}, release: make(chan struct{}), cancelled: make(chan struct{})}
	rt.Thinker.provider = p
	rt.pool = &ProviderPool{providers: map[string]LLMProvider{"fixture": p}, default_: "fixture"}
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	var tools sync.Mutex
	toolCalls := 0
	rt.handleTools = func(_ *Thinker, calls []toolCall, _ []string) ([]string, []string, []ToolResult) {
		if len(calls) == 0 {
			return nil, nil, nil
		}
		tools.Lock()
		toolCalls++
		tools.Unlock()
		return nil, nil, nil
	}
	worker, err := rt.ensureExecutionThread()
	if err != nil {
		t.Fatal(err)
	}
	worker.Thinker.handleTools = rt.handleTools
	done := runCompletionFixture(rt)
	s.events <- RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "Jeudi.", TranscriptDelta: true, TranscriptEndMS: 500}
	s.events <- RealtimeEvent{Type: RealtimeEventDelegation, DelegationID: "request-one", DelegationOffsetMS: 1000}
	select {
	case <-p.started:
	case <-time.After(2 * time.Second):
		t.Fatal("delegation incorrectly required transcript to reach a later notification timestamp")
	}
	s.events <- RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: "Non, vendredi.", TranscriptDelta: true, TranscriptStartMS: 1100, TranscriptEndMS: 1800}
	// Wait for the event loop to consume the correction before releasing inference.
	correctionDeadline := time.Now().Add(time.Second)
	for time.Now().Before(correctionDeadline) {
		rt.transcriptMu.Lock()
		seen := false
		for _, m := range rt.messages {
			seen = seen || strings.Contains(m.Content, "Non, vendredi.")
		}
		rt.transcriptMu.Unlock()
		if seen {
			break
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case <-p.cancelled:
		t.Fatal("caller correction cancelled inference")
	default:
	}
	close(p.release)
	deadline := time.Now().Add(2 * time.Second)
	found := false
	for time.Now().Before(deadline) {
		s.mu.Lock()
		for _, m := range s.texts {
			found = found || strings.Contains(m.Content, "vendredi")
		}
		s.mu.Unlock()
		if found {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	rt.Thinker.Stop()
	waitCompletionFixture(t, done)
	if !found {
		t.Fatal("corrected request did not complete")
	}
	tools.Lock()
	defer tools.Unlock()
	if toolCalls != 0 {
		t.Fatal("stale booking executed")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.calls != 2 {
		t.Fatalf("inference attempts=%d", p.calls)
	}
}

func TestDelegatedVisionConsumesOnlyImagesSeenByThatRequest(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{CoreExecution: true}
	p := newRecordingThreadEventProvider()
	rt.Thinker.provider = p
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	defer rt.stopExecutionThread()
	rt.handleBusEvent(Event{Type: EventInbox, From: "user", Text: "Read this screenshot", Parts: []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.invalid/first.png"}}}})
	rt.backend.lastInput = time.Now().Add(-time.Second)
	rt.maybeStartExecution()
	select {
	case request := <-p.requests:
		if transientAttachmentCount(request) != 1 {
			t.Fatal("screenshot missing from managed child's request")
		}
	case <-time.After(time.Second):
		t.Fatal("vision handoff stalled")
	}
	rt.handleBusEvent(Event{Type: EventInbox, From: "user", Text: "Now this screenshot", Parts: []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.invalid/second.png"}}}})
	if transientAttachmentCount(rt.messages) != 1 {
		t.Fatal("handoff consumed a newer screenshot or retained an already transferred one")
	}
	if len(s.toolResults) != 0 {
		t.Fatal("vision evidence sent as native voice tool outputs")
	}
}
