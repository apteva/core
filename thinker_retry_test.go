package core

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestThinkerStopIsSafeConcurrently(t *testing.T) {
	thinker := &Thinker{quit: make(chan struct{})}
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			thinker.Stop()
		}()
	}
	wg.Wait()
	select {
	case <-thinker.quit:
	default:
		t.Fatal("Stop did not close quit")
	}
}

type scriptedRetryProvider struct {
	name       string
	failures   int
	response   ChatResponse
	block      bool
	started    chan struct{}
	startedOne sync.Once

	mu       sync.Mutex
	calls    int
	messages [][]Message
}

func (p *scriptedRetryProvider) Chat(ctx context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	p.calls++
	call := p.calls
	p.messages = append(p.messages, cloneMessages(messages))
	p.mu.Unlock()
	if p.started != nil {
		p.startedOne.Do(func() { close(p.started) })
	}
	if p.block {
		<-ctx.Done()
		return ChatResponse{}, ctx.Err()
	}
	if call <= p.failures {
		return ChatResponse{}, errors.New("provider HTTP 401 token_expired")
	}
	return p.response, nil
}

func (p *scriptedRetryProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "test", ModelMedium: "test", ModelSmall: "test"}
}
func (p *scriptedRetryProvider) Name() string                           { return p.name }
func (p *scriptedRetryProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *scriptedRetryProvider) SupportsNativeTools() bool              { return false }
func (p *scriptedRetryProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *scriptedRetryProvider) SetBuiltinTools([]string)               {}
func (p *scriptedRetryProvider) WithBuiltins([]string) LLMProvider      { return p }

func retryTestThinker(provider LLMProvider) *Thinker {
	bus := NewEventBus()
	return &Thinker{
		provider:   provider,
		messages:   []Message{{Role: "system", Content: "system"}, {Role: "user", Content: "original work"}},
		bus:        bus,
		sub:        bus.Subscribe("main", 10),
		quit:       make(chan struct{}),
		threadID:   "main",
		model:      ModelLarge,
		retryDelay: func(error, int) time.Duration { return time.Millisecond },
	}
}

func TestCallLLMWithRetryPreservesPreparedTurn(t *testing.T) {
	provider := &scriptedRetryProvider{name: "primary", failures: 2, response: ChatResponse{Text: "done"}}
	thinker := retryTestThinker(provider)
	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil {
		t.Fatalf("callLLMWithRetry: %v", err)
	}
	if resp.Text != "done" || provider.calls != 3 {
		t.Fatalf("response=%q calls=%d", resp.Text, provider.calls)
	}
	if len(thinker.messages) != 2 || thinker.messages[1].Content != "original work" {
		t.Fatalf("retry mutated prepared context: %#v", thinker.messages)
	}
	for i, seen := range provider.messages {
		if len(seen) != 2 || seen[1].Content != "original work" {
			t.Fatalf("attempt %d saw changed context: %#v", i+1, seen)
		}
	}
}

func TestCallLLMWithRetryFallbackDoesNotBecomePermanent(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 100}
	fallback := &scriptedRetryProvider{name: "fallback", response: ChatResponse{Text: "fallback result"}}
	thinker := retryTestThinker(primary)
	thinker.pool = &ProviderPool{
		providers: map[string]LLMProvider{"primary": primary, "fallback": fallback},
		order:     []string{"primary", "fallback"},
		default_:  "primary",
	}
	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "fallback result" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if thinker.provider != primary {
		t.Fatal("successful fallback replaced the configured primary provider")
	}
}

func TestStopCancelsInFlightProviderCall(t *testing.T) {
	provider := &scriptedRetryProvider{name: "blocking", block: true, started: make(chan struct{})}
	thinker := retryTestThinker(provider)
	ctx, cancel := context.WithCancel(context.Background())
	thinker.runContextMu.Lock()
	thinker.runCancel = cancel
	thinker.runContextMu.Unlock()
	done := make(chan error, 1)
	go func() {
		_, err := thinker.callLLMWithRetry(ctx)
		done <- err
	}()
	<-provider.started
	thinker.Stop()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Stop did not cancel provider call")
	}
}

func TestProviderRetryDelayClassifiesFailures(t *testing.T) {
	if got := providerRetryDelay(errors.New("401 token_expired"), 1); got != 30*time.Second {
		t.Fatalf("auth delay = %s", got)
	}
	if got := providerRetryDelay(errors.New("HTTP 429 rate limit"), 1); got != 15*time.Second {
		t.Fatalf("rate delay = %s", got)
	}
	if got := providerRetryDelay(errors.New("HTTP 400 invalid schema"), 1); got != time.Minute {
		t.Fatalf("permanent delay = %s", got)
	}
	if got := providerRetryDelay(errors.New("temporary network error"), 20); got != 2*time.Minute {
		t.Fatalf("transient cap = %s", got)
	}
}
