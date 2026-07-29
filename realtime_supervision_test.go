package core

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestGoogleTextProviderDefaultsToLatestStableFlash(t *testing.T) {
	provider := NewGoogleProvider("test")
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		if got := provider.Models()[tier]; got != "gemini-3.6-flash" {
			t.Fatalf("Google %s model = %q, want gemini-3.6-flash", tier, got)
		}
	}
}

func TestEphemeralRealtimeThreadHasAppliedMetadataButNoEvolve(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.config.RealtimeEnabled = true
	realtimeProvider := &fakeRealtimeProvider{}
	parent.pool = &ProviderPool{
		providers: map[string]LLMProvider{"fireworks": parent.provider},
		order:     []string{"fireworks"}, default_: "fireworks",
		realtimeProviders: map[string]RealtimeProvider{
			realtimeProvider.Name(): realtimeProvider,
		},
		realtimeOrder: []string{realtimeProvider.Name()}, realtimeDefault: realtimeProvider.Name(),
	}
	if err := parent.threads.SpawnWithOpts("live-call", "Welcome the caller.", []string{"lookup"}, SpawnOpts{
		Realtime: true, Ephemeral: true, ProviderName: realtimeProvider.Name(), DeferRun: true,
	}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["live-call"]
	if thread.Tools["evolve"] {
		t.Fatalf("ephemeral realtime tools unexpectedly include evolve: %#v", thread.Tools)
	}
	for _, required := range []string{"send", "done", "pace", "search_tools", "lookup"} {
		if !thread.Tools[required] {
			t.Fatalf("ephemeral realtime tools missing %q: %#v", required, thread.Tools)
		}
	}
	if strings.Contains(thread.Thinker.messages[0].Content, "[DIRECTIVE MANAGEMENT]") {
		t.Fatalf("ephemeral realtime prompt advertises durable directive mutation:\n%s", thread.Thinker.messages[0].Content)
	}

	parent.threads.realtimeBridgeConnected("live-call")
	dynamic := buildDynamicTurnContext(parent.threads.ListAgentVisible(), "")
	for _, want := range []string{
		"realtime; ephemeral", "audio bridge connected", "listed configuration already applied",
	} {
		if !strings.Contains(dynamic, want) {
			t.Fatalf("active-thread context missing %q:\n%s", want, dynamic)
		}
	}
}

type immutableRealtimeSession struct {
	*fakeRealtimeSession
	mu          sync.Mutex
	fingerprint string
	closes      int
}

func (s *immutableRealtimeSession) PreviewConfigurationUpdate(instructions string, tools []NativeTool) RealtimeConfigurationDisposition {
	next := googleRealtimeConfigFingerprint(instructions, tools)
	s.mu.Lock()
	defer s.mu.Unlock()
	if next == s.fingerprint {
		return RealtimeConfigurationUnchanged
	}
	return RealtimeConfigurationRestartRequired
}

func (s *immutableRealtimeSession) Close() error {
	s.mu.Lock()
	s.closes++
	s.mu.Unlock()
	return nil
}

func TestActiveImmutableRealtimeUpdateIsNoOpOrRequiresExplicitRestart(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.config.RealtimeEnabled = true
	realtimeProvider := &fakeRealtimeProvider{}
	parent.pool = &ProviderPool{
		providers: map[string]LLMProvider{"fireworks": parent.provider},
		order:     []string{"fireworks"}, default_: "fireworks",
		realtimeProviders: map[string]RealtimeProvider{
			realtimeProvider.Name(): realtimeProvider,
		},
		realtimeOrder: []string{realtimeProvider.Name()}, realtimeDefault: realtimeProvider.Name(),
	}
	if err := parent.threads.SpawnWithOpts("live-call", "Already configured.", []string{"lookup"}, SpawnOpts{
		Realtime: true, Ephemeral: true, ProviderName: realtimeProvider.Name(), DeferRun: true,
	}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["live-call"]
	instructions, tools := thread.Realtime.configurationSnapshot()
	session := &immutableRealtimeSession{
		fakeRealtimeSession: newFakeRealtimeSession(),
		fingerprint:         googleRealtimeConfigFingerprint(instructions, tools),
	}
	thread.Realtime.replaceSession(session)
	parent.threads.realtimeBridgeConnected("live-call")

	result, err := parent.threads.UpdateWithOpts("live-call", "", "Already configured.", []string{"lookup"}, ThreadUpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Changed || result.RealtimeRestarted || session.closes != 0 {
		t.Fatalf("identical update = %#v, closes=%d", result, session.closes)
	}

	if _, err := parent.threads.UpdateWithOpts("live-call", "", "Changed while live.", nil, ThreadUpdateOptions{}); !errors.Is(err, ErrRealtimeConfigurationRestartRequired) {
		t.Fatalf("changed update error = %v, want restart-required", err)
	}
	if thread.Directive != "Already configured." || session.closes != 0 {
		t.Fatalf("rejected update mutated live state: directive=%q closes=%d", thread.Directive, session.closes)
	}

	result, err = parent.threads.UpdateWithOpts("live-call", "", "Changed while live.", nil, ThreadUpdateOptions{RestartRealtime: true})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Changed || !result.RealtimeRestarted || session.closes != 1 {
		t.Fatalf("explicit update = %#v, closes=%d", result, session.closes)
	}
}

func TestRealtimeGreetingReplaysOnlyWhenReconnectPrecedesFirstAudio(t *testing.T) {
	thinker := newTestThinker()
	first := newFakeRealtimeSession()
	second := newFakeRealtimeSession()
	provider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{first, second}}
	audioOut := make(chan RealtimeAudioFrame, 4)
	rt, err := startRealtimeThinker(context.Background(), thinker, provider, "", nil, audioOut, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.setInitialMessage("Greet the caller.")
	done := make(chan struct{})
	go func() {
		defer close(done)
		rt.Run()
	}()
	defer func() {
		rt.cancel()
		<-done
	}()

	rt.audioBridgeConnected()
	waitForRealtimeTexts(t, first, 1)
	close(first.events)
	waitForRealtimeTexts(t, second, 1)

	second.events <- RealtimeEvent{
		Type: RealtimeEventAudioOut, Audio: []byte{1, 2, 3, 4}, ItemID: "greeting",
	}
	select {
	case <-audioOut:
	case <-time.After(time.Second):
		t.Fatal("assistant audio was not forwarded")
	}
	waitForRealtimeTelemetryType(t, thinker.telemetry, "realtime.first_audio", time.Second)
	close(second.events)

	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		provider.mu.Lock()
		opened := len(provider.opens)
		var third *fakeRealtimeSession
		if len(provider.sessions) >= 3 {
			third = provider.sessions[2]
		}
		provider.mu.Unlock()
		if opened >= 3 && third != nil {
			time.Sleep(30 * time.Millisecond)
			third.mu.Lock()
			greetings := len(third.texts)
			third.mu.Unlock()
			if greetings != 0 {
				t.Fatalf("greeting replayed after assistant audio: %d messages", greetings)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("realtime session did not reconnect a second time")
}

func waitForRealtimeTexts(t *testing.T, session *fakeRealtimeSession, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		session.mu.Lock()
		got := len(session.texts)
		session.mu.Unlock()
		if got >= want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("realtime text count did not reach %d", want)
}

func waitForRealtimeTelemetryType(t *testing.T, telemetry *Telemetry, eventType string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		events, _ := telemetry.StoredEvents(0)
		for _, event := range events {
			if event.Type == eventType {
				return
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("%s telemetry was not emitted", eventType)
}

// TestGoogleMainLeavesConfiguredRealtimeThreadAlone is an opt-in paid
// behavioral check. It exercises the normal Gemini text/main model, while the
// deterministic tests above enforce the runtime guard independently of model
// judgment.
//
//	RUN_GOOGLE_REALTIME_SUPERVISION_SMOKE=1 GOOGLE_API_KEY=... \
//	  go test -v -run TestGoogleMainLeavesConfiguredRealtimeThreadAlone -timeout 2m .
func TestGoogleMainLeavesConfiguredRealtimeThreadAlone(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_GOOGLE_REALTIME_SUPERVISION_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_SUPERVISION_SMOKE=1 for the paid Gemini supervision check")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}
	provider := NewGoogleProvider(key)
	registry := NewToolRegistry("test")
	realtimeProvider := NewGoogleRealtimeProvider(key)
	pool := &ProviderPool{
		providers: map[string]LLMProvider{"google": provider},
		order:     []string{"google"}, default_: "google",
		realtimeProviders: map[string]RealtimeProvider{
			"google-realtime": realtimeProvider,
		},
		realtimeOrder: []string{"google-realtime"}, realtimeDefault: "google-realtime",
	}
	systemPrompt := buildSystemPrompt(
		"# Role\nGovern active work without disrupting live calls.",
		ModeAutonomous,
		registry,
		"",
		nil,
		nil,
		pool,
		nil,
	)
	activeContext := buildDynamicTurnContext([]ThreadInfo{{
		ID: "tel-live", Name: "Reception", Realtime: true, Ephemeral: true, BridgeConnected: true,
		Directive: "Welcome callers, check availability, and book confirmed appointments.",
		Tools:     []string{"availability_check", "appointment_book", "send", "done", "pace"},
	}}, "")
	tools := registry.NativeTools(nil, nil, false)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	response, err := provider.Chat(ctx, []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: `[console] The telephone call is active. Its configured receptionist instructions and booking tools are correct. Do not interrupt it. If no new information needs delivery, simply pace.

` + activeContext},
	}, provider.Models()[ModelLarge], tools, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, call := range response.ToolCalls {
		if call.Name == "update" {
			t.Fatalf("Gemini attempted a redundant realtime update: %#v", call)
		}
	}
	t.Logf("model=%s text=%q tool_calls=%v", provider.Models()[ModelLarge], response.Text, response.ToolCalls)
}
