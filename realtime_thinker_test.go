package core

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRealtimePromptKeepsReasoningPrivateWithoutChangingNormalWorkers(t *testing.T) {
	normal := formatThreadBasePrompt(false, false, "worker", "main coordinator")
	if !strings.Contains(normal, "Think out loud") || strings.Contains(normal, "LIVE TURN-TAKING") {
		t.Fatalf("normal worker prompt changed unexpectedly:\n%s", normal)
	}

	realtime := formatThreadBasePrompt(false, true, "voice", "main coordinator") + realtimeConversationPrompt
	for _, forbidden := range []string{"Think out loud", "You MUST report results", "pace(sleep=\"5m\")"} {
		if strings.Contains(realtime, forbidden) {
			t.Fatalf("realtime prompt contains %q:\n%s", forbidden, realtime)
		}
	}
	for _, required := range []string{
		"Reason privately",
		"Spoken output contains only words intended for the caller",
		"at most one brief, natural sentence",
		"After sending work to main, do not speak again",
		"Do not use pace as a conversational wait mechanism",
	} {
		if !strings.Contains(realtime, required) {
			t.Fatalf("realtime prompt missing %q:\n%s", required, realtime)
		}
	}
}

type fakeRealtimeToolResult struct {
	callID  string
	content string
	isError bool
}

type fakeRealtimeSession struct {
	mu          sync.Mutex
	events      chan RealtimeEvent
	toolResults []fakeRealtimeToolResult
	responses   int
	updates     []RealtimeSessionOpts
	restored    [][]Message
	texts       []Message
	interrupts  int
	truncations []struct {
		itemID string
		endMS  int
	}
}

func newFakeRealtimeSession() *fakeRealtimeSession {
	return &fakeRealtimeSession{events: make(chan RealtimeEvent, 16)}
}

func (s *fakeRealtimeSession) SendAudio([]byte) error { return nil }
func (s *fakeRealtimeSession) SendText(role, text string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.texts = append(s.texts, Message{Role: role, Content: text})
	return nil
}
func (s *fakeRealtimeSession) SendToolResult(callID, result string, isError bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.toolResults = append(s.toolResults, fakeRealtimeToolResult{callID: callID, content: result, isError: isError})
	return nil
}
func (s *fakeRealtimeSession) RequestResponse() error {
	s.mu.Lock()
	s.responses++
	s.mu.Unlock()
	return nil
}
func (s *fakeRealtimeSession) UpdateConfiguration(instructions string, tools []NativeTool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updates = append(s.updates, RealtimeSessionOpts{Instructions: instructions, Tools: tools})
	return nil
}
func (s *fakeRealtimeSession) RestoreConversation(messages []Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.restored = append(s.restored, cloneMessages(messages))
	return nil
}
func (s *fakeRealtimeSession) Interrupt() error {
	s.mu.Lock()
	s.interrupts++
	s.mu.Unlock()
	return nil
}
func (s *fakeRealtimeSession) Truncate(itemID string, audioEndMS int) error {
	s.mu.Lock()
	s.truncations = append(s.truncations, struct {
		itemID string
		endMS  int
	}{itemID: itemID, endMS: audioEndMS})
	s.mu.Unlock()
	return nil
}
func (s *fakeRealtimeSession) Events() <-chan RealtimeEvent { return s.events }
func (s *fakeRealtimeSession) Close() error                 { return nil }

type fakeRealtimeProvider struct {
	mu       sync.Mutex
	sessions []*fakeRealtimeSession
	opens    []RealtimeSessionOpts
}

func (p *fakeRealtimeProvider) Name() string { return "fake-realtime" }
func (p *fakeRealtimeProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "fake-large", ModelMedium: "fake-medium", ModelSmall: "fake-small"}
}
func (p *fakeRealtimeProvider) Pricing(string) RealtimePricing { return RealtimePricing{} }
func (p *fakeRealtimeProvider) DefaultVoice() string           { return "test-voice" }
func (p *fakeRealtimeProvider) DefaultTranscriptionModel() string {
	return "test-transcribe"
}
func (p *fakeRealtimeProvider) Open(_ context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.opens = append(p.opens, opts)
	index := len(p.opens) - 1
	if index < len(p.sessions) {
		return p.sessions[index], nil
	}
	session := newFakeRealtimeSession()
	p.sessions = append(p.sessions, session)
	return session, nil
}

func TestRealtimeOpenUsesFullCorePromptAndSelectedProfile(t *testing.T) {
	thinker := newTestThinker()
	thinker.messages[0].Content = "FULL CORE PROMPT\n[DIRECTIVE]\nanswer calls"
	thinker.registry = NewToolRegistry("test")
	thinker.toolAllowlist = map[string]bool{"send": true, "done": true, "pace": true, "evolve": true}
	thinker.agentModel = ModelSmall
	thinker.agentReasoning = ReasoningHigh
	session := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{session}}
	rt, err := startRealtimeThinker(context.Background(), thinker, provider, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.cancel()

	provider.mu.Lock()
	opts := provider.opens[0]
	provider.mu.Unlock()
	if opts.Instructions != thinker.messages[0].Content {
		t.Fatalf("realtime received partial prompt:\n%s", opts.Instructions)
	}
	if opts.Model != "fake-small" || opts.Reasoning != "high" || opts.Voice != "test-voice" {
		t.Fatalf("profile not forwarded: %#v", opts)
	}
	if !opts.TranscribeInput || opts.SafetyIdentifier == "" {
		t.Fatalf("transcription/safety identifier missing: %#v", opts)
	}
}

func TestOpenAIRealtimeUsesLowReasoningOnlyWhenProfileIsAuto(t *testing.T) {
	thinker := newTestThinker()
	thinker.agentReasoning = ReasoningAuto
	rt := newRealtimeThinker(context.Background(), thinker, NewOpenAIRealtimeProvider("test"), "", nil, nil, nil)
	defer rt.cancel()
	if rt.opts.Reasoning != "low" {
		t.Fatalf("auto reasoning = %q, want OpenAI realtime default low", rt.opts.Reasoning)
	}

	thinker.agentReasoning = ReasoningHigh
	explicit := newRealtimeThinker(context.Background(), thinker, NewOpenAIRealtimeProvider("test"), "", nil, nil, nil)
	defer explicit.cancel()
	if explicit.opts.Reasoning != "high" {
		t.Fatalf("explicit reasoning = %q, want high", explicit.opts.Reasoning)
	}
}

func TestRealtimeCoreToolsUseNormalThreadHandlerAndRefreshPrompt(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.registry = NewToolRegistry("test")
	parent.config.RealtimeEnabled = true
	session := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{session}}
	parent.pool = &ProviderPool{
		providers: map[string]LLMProvider{"fireworks": parent.provider}, order: []string{"fireworks"}, default_: "fireworks",
		realtimeProviders: map[string]RealtimeProvider{"fake-realtime": provider}, realtimeOrder: []string{"fake-realtime"}, realtimeDefault: "fake-realtime",
	}
	if err := parent.threads.SpawnWithOpts("voice", "old plain directive", nil, SpawnOpts{
		Realtime: true, ProviderName: "fake-realtime", DeferRun: true,
	}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["voice"]
	rt := thread.Realtime
	rt.replaceSession(session)

	rt.dispatchToolCall(RealtimeEvent{Type: RealtimeEventToolCall, ToolCallID: "e1", ToolName: "evolve", ToolArgs: `{"directive":"new durable directive"}`})
	if thread.Directive != "new durable directive" {
		t.Fatalf("directive = %q", thread.Directive)
	}
	session.mu.Lock()
	if len(session.toolResults) != 1 || session.toolResults[0].callID != "e1" || session.toolResults[0].isError {
		t.Fatalf("evolve result = %#v", session.toolResults)
	}
	if len(session.updates) != 1 || session.updates[0].Instructions != thread.Thinker.messages[0].Content {
		t.Fatalf("updated prompt not pushed: %#v", session.updates)
	}
	if !strings.Contains(session.updates[0].Instructions, "[REALTIME CONVERSATION]") || strings.Contains(session.updates[0].Instructions, "Think out loud") {
		t.Fatalf("realtime speech contract lost after evolve:\n%s", session.updates[0].Instructions)
	}
	session.mu.Unlock()

	stored := parent.config.GetThreads()
	if len(stored) != 1 || !stored[0].Realtime || stored[0].Provider != "fake-realtime" {
		t.Fatalf("realtime metadata lost on evolve: %#v", stored)
	}
	if err := parent.threads.Update("voice", "Voice Agent", "updated again", []string{"send"}); err != nil {
		t.Fatal(err)
	}
	if err := parent.threads.Rename("voice", "voice-renamed"); err != nil {
		t.Fatal(err)
	}
	stored = parent.config.GetThreads()
	if len(stored) != 1 || stored[0].ID != "voice-renamed" || !stored[0].Realtime || stored[0].Voice != "" || stored[0].Provider != "fake-realtime" {
		t.Fatalf("realtime metadata lost on update/rename: %#v", stored)
	}
	if rt.opts.SafetyIdentifier != realtimeSafetyIdentifier("voice-renamed") {
		t.Fatalf("safety identifier not refreshed on rename: %q", rt.opts.SafetyIdentifier)
	}
}

func TestRealtimeGreetingStartsOnceAfterAudioBridgeConnects(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.config.RealtimeEnabled = true
	session := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{session}}
	parent.pool = &ProviderPool{
		providers: map[string]LLMProvider{"fireworks": parent.provider}, order: []string{"fireworks"}, default_: "fireworks",
		realtimeProviders: map[string]RealtimeProvider{"fake-realtime": provider}, realtimeOrder: []string{"fake-realtime"}, realtimeDefault: "fake-realtime",
	}
	if err := parent.threads.SpawnWithOpts("voice-greeting", "answer the phone", nil, SpawnOpts{
		Realtime: true, ProviderName: "fake-realtime", InitialMessage: "Greet the caller.", DeferRun: true,
	}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["voice-greeting"]
	thread.Realtime.replaceSession(session)

	parent.threads.realtimeBridgeConnected("voice-greeting")
	parent.threads.realtimeBridgeConnected("voice-greeting")

	session.mu.Lock()
	defer session.mu.Unlock()
	if len(session.texts) != 1 || session.texts[0].Role != "user" || session.texts[0].Content != "Greet the caller." {
		t.Fatalf("greeting messages = %#v", session.texts)
	}
	if session.responses != 1 {
		t.Fatalf("greeting responses = %d", session.responses)
	}
}

func TestRealtimeParallelToolBatchContinuesOnceAfterEveryResult(t *testing.T) {
	session := newFakeRealtimeSession()
	rt := newRealtimeThinker(context.Background(), newTestThinker(), &fakeRealtimeProvider{}, "", nil, nil, nil)
	rt.replaceSession(session)

	rt.beginToolCall(RealtimeEvent{ResponseID: "response-1", ToolCallID: "call-1"}, session)
	rt.beginToolCall(RealtimeEvent{ResponseID: "response-1", ToolCallID: "call-2"}, session)
	rt.submitToolResult(session, "call-1", "one", false)
	rt.completeToolResponse("response-1")

	session.mu.Lock()
	if session.responses != 0 {
		t.Fatalf("continued before all tool results: %d", session.responses)
	}
	session.mu.Unlock()

	rt.submitToolResult(session, "call-2", "two", false)
	session.mu.Lock()
	defer session.mu.Unlock()
	if len(session.toolResults) != 2 || session.responses != 1 {
		t.Fatalf("results=%#v continuations=%d", session.toolResults, session.responses)
	}
}

func TestRealtimeRenewsEndedSessionAndRestoresBoundedTranscript(t *testing.T) {
	t.Chdir(t.TempDir())
	thinker := newTestThinker()
	thinker.session = NewSession(".", "voice-renew")
	thinker.registry = NewToolRegistry("test")
	first := newFakeRealtimeSession()
	second := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{first, second}}
	rt, err := startRealtimeThinker(context.Background(), thinker, provider, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.appendTranscript("user", "hello")
	rt.appendTranscript("assistant", "hi")
	done := make(chan struct{})
	go func() {
		rt.Run()
		close(done)
	}()
	close(first.events)

	deadline := time.Now().Add(2 * time.Second)
	for {
		provider.mu.Lock()
		opens := len(provider.opens)
		provider.mu.Unlock()
		if opens >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("provider session was not renewed")
		}
		time.Sleep(10 * time.Millisecond)
	}
	second.mu.Lock()
	if len(second.restored) != 1 || len(second.restored[0]) != 2 || second.restored[0][0].Content != "hello" {
		t.Fatalf("restored transcript = %#v", second.restored)
	}
	second.mu.Unlock()
	thinker.Stop()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("realtime run did not stop")
	}
}

func TestRealtimeSpeechStartTruncatesAndClearsPlayback(t *testing.T) {
	thinker := newTestThinker()
	session := newFakeRealtimeSession()
	control := make(chan string, 1)
	audioOut := make(chan []byte, 1)
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, audioOut, control)
	rt.replaceSession(session)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "item-7", Audio: make([]byte, realtimePCMBytesPerSecond/2)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventSpeechStarted})
	session.mu.Lock()
	if len(session.truncations) != 1 || session.truncations[0].itemID != "item-7" || session.truncations[0].endMS != 500 {
		t.Fatalf("truncations = %#v", session.truncations)
	}
	session.mu.Unlock()
	if got := <-control; got != "interrupt" {
		t.Fatalf("control = %q", got)
	}
}

func TestRealtimeConversationStateTelemetrySeparatesThinkingAndSpeaking(t *testing.T) {
	thinker := newTestThinker()
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, nil, nil)
	defer rt.cancel()

	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseStarted, ResponseID: "response-1"})
	rt.handleSessionEvent(RealtimeEvent{
		Type: RealtimeEventAudioOut, ResponseID: "response-1", ItemID: "item-1", Phase: "commentary", Audio: []byte{1, 2},
	})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "response-1"})

	events, _ := thinker.telemetry.StoredEvents(0)
	var states []string
	var speakingPhase string
	for _, event := range events {
		if event.Type != "realtime.state" {
			continue
		}
		var data map[string]any
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatal(err)
		}
		state, _ := data["state"].(string)
		states = append(states, state)
		if state == "speaking" {
			speakingPhase, _ = data["phase"].(string)
		}
	}
	if got := strings.Join(states, ","); got != "thinking,speaking,listening" {
		t.Fatalf("states = %q", got)
	}
	if speakingPhase != "commentary" {
		t.Fatalf("speaking phase = %q", speakingPhase)
	}
}

func TestSpawnSchemaAdvertisesRealtimeFields(t *testing.T) {
	spawn := NewToolRegistry("test").Get("spawn")
	if spawn == nil {
		t.Fatal("spawn tool missing")
	}
	properties := spawn.InputSchema["properties"].(map[string]any)
	if properties["realtime"] == nil || properties["voice"] == nil {
		t.Fatalf("spawn schema missing realtime fields: %#v", properties)
	}
}
