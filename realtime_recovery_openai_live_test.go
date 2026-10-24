package core

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// This proxy injects a labeled local failure and one second of connection delay;
// both provider sessions and all speech/transcription still use the real API.
type recoveryLiveProxy struct {
	RealtimeProvider
	opens   atomic.Int32
	fail    atomic.Bool
	entered chan struct{}
	sent    atomic.Int64
}
type recoveryLiveProxySession struct {
	RealtimeSession
	proxy *recoveryLiveProxy
	first bool
}

func (s *recoveryLiveProxySession) SendAudio(audio []byte) error {
	if s.first && s.proxy.fail.Load() {
		return realtimeAdapterError(RealtimeErrorTransport, errors.New("injected live recovery failure"))
	}
	err := s.RealtimeSession.SendAudio(audio)
	if err == nil {
		s.proxy.sent.Add(int64(len(audio)))
	}
	return err
}
func (p *recoveryLiveProxy) Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	n := p.opens.Add(1)
	if n == 2 {
		close(p.entered)
		timer := time.NewTimer(time.Second)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	session, err := p.RealtimeProvider.Open(ctx, opts)
	if err != nil {
		return nil, err
	}
	return &recoveryLiveProxySession{RealtimeSession: session, proxy: p, first: n == 1}, nil
}
func TestOpenAILiveRecoveryWithContinuousIntakeLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_OPENAI_LIVE_RECOVERY") != "1" {
		t.Skip("set RUN_OPENAI_LIVE_RECOVERY=1 for paid Live recovery verification")
	}
	loadIntegrationEnv()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY required")
	}
	caller := renderLiveCallerText(t, key, "Mon mot de référence est abricot. Répétez simplement ce mot, s'il vous plaît.")
	rt, _ := completionFixture(t)
	rt.Thinker.threadID = "main"
	rt.directive = "Parlez uniquement en français. Écoutez la personne et répétez brièvement son mot de référence lorsqu'elle le demande. Ne terminez pas la conversation. N'inventez pas de mot."
	rt.messages[0].Content = rt.directive
	backend := NewOpenAINativeProvider(key)
	applyModelOverrides(backend, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
	rt.Thinker.provider = backend
	rt.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": backend}, default_: "openai", order: []string{"openai"}}
	p := &recoveryLiveProxy{RealtimeProvider: NewOpenAIRealtimeProvider(key), entered: make(chan struct{})}
	rt.provider = p
	rt.opts.Model = "gpt-live-1"
	rt.opts.Voice = "brise"
	// The wrapper forwards the provider API, while the production capability
	// selection uses the underlying registered provider's protocol contract.
	rt.capabilities = capabilitiesForRealtime(p.RealtimeProvider, rt.opts.Model)
	rt.registry = NewToolRegistry("")
	input := make(chan []byte, 6)
	rt.audioIn = input
	ctx, cancel := context.WithTimeout(context.Background(), 65*time.Second)
	defer cancel()
	rt.cancel()
	rt.ctx, rt.cancel = context.WithCancel(ctx)
	if err := rt.openSession(false); err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	done := runCompletionFixture(rt)
	defer func() {
		rt.Thinker.Stop()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("recovery worker failed to stop")
		}
	}()
	microphoneErrors := make(chan error, 1)
	startCaller := make(chan struct{})
	var received atomic.Int64
	go func() {
		timer := time.NewTicker(20 * time.Millisecond)
		defer timer.Stop()
		position := -1
		signal := startCaller
		for {
			select {
			case <-ctx.Done():
				return
			case <-signal:
				position = 0
				signal = nil
			case <-timer.C:
				frame := make([]byte, 960)
				if position >= 0 && position < len(caller) {
					copy(frame, caller[position:min(position+960, len(caller))])
					position += 960
				}
				if err := enqueueLiveRendererInput(ctx, input, frame); err != nil {
					select {
					case microphoneErrors <- err:
					default:
					}
					return
				}
				received.Add(1)
			}
		}
	}()
	// Trigger a genuine session replacement while audio continues through Core's
	// external intake (rather than sending directly into the provider adapter).
	p.fail.Store(true)
	select {
	case <-p.entered:
	case err := <-microphoneErrors:
		t.Fatal(err)
	case <-ctx.Done():
		t.Fatal("recovery did not start")
	}
	before := received.Load()
	timer := time.NewTimer(700 * time.Millisecond)
	select {
	case err := <-microphoneErrors:
		t.Fatal(err)
	case <-timer.C:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if delta := received.Load() - before; delta < 20 {
		t.Fatalf("only %d microphone frames admitted during blocked Open", delta)
	}
	deadline := time.NewTicker(20 * time.Millisecond)
	defer deadline.Stop()
	for generationForAudioFrame(rt) < 3 {
		select {
		case err := <-microphoneErrors:
			t.Fatal(err)
		case <-ctx.Done():
			t.Fatal("replacement session not ready")
		case <-deadline.C:
		}
	}
	close(startCaller)
	heard, spoken := false, false
	audioBytes := 0
	cursor := 0
	for {
		select {
		case err := <-microphoneErrors:
			t.Fatal(err)
		case <-ctx.Done():
			t.Fatalf("real recovery failed: heard=%v spoken=%v output_bytes=%d", heard, spoken, audioBytes)
		case frame := <-rt.audioOut:
			audioBytes += len(frame.Audio)
			frame.written(true)
		case <-deadline.C:
			events, next := rt.telemetry.StoredEvents(cursor)
			cursor = next
			for _, e := range events {
				if e.Type == "realtime.user" && strings.Contains(strings.ToLower(string(e.Data)), "abricot") {
					heard = true
				}
				if e.Type == "realtime.assistant" && strings.Contains(strings.ToLower(string(e.Data)), "abricot") {
					spoken = true
				}
				if e.Type == "realtime.error" {
					t.Logf("provider error: %s", e.Data)
				}
			}
			if heard && spoken && audioBytes > 0 {
				if p.opens.Load() != 2 {
					t.Fatalf("unexpected extra recoveries: %d", p.opens.Load())
				}
				for _, e := range recoveryEvents(rt, "realtime.audio_drop") {
					if e["reason"] != "delivery_uncertain" {
						t.Fatalf("unexpected input loss: %+v", e)
					}
				}
				t.Logf("real Live recovered: microphone_frames=%d admitted_audio_ms=%d output_bytes=%d confirmed French caller and reply", received.Load(), p.sent.Load()*1000/realtimePCMBytesPerSecond, audioBytes)
				for _, e := range recoveryEvents(rt, "realtime.reconnect") {
					t.Logf("recovery: %+v", e)
				}
				return
			}
		}
	}
}
