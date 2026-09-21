package core

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
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
	failureErr error
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
		if p.failureErr != nil {
			return ChatResponse{}, p.failureErr
		}
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

type tierRetryProvider struct {
	models    map[ModelTier]string
	responses map[string]ChatResponse
	errors    map[string]error
	calls     []string
}

func (p *tierRetryProvider) Chat(_ context.Context, _ []Message, model string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.calls = append(p.calls, model)
	if err := p.errors[model]; err != nil {
		return ChatResponse{}, err
	}
	return p.responses[model], nil
}

func (p *tierRetryProvider) Models() map[ModelTier]string           { return p.models }
func (p *tierRetryProvider) Name() string                           { return "tiered" }
func (p *tierRetryProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *tierRetryProvider) SupportsNativeTools() bool              { return false }
func (p *tierRetryProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *tierRetryProvider) SetBuiltinTools([]string)               {}
func (p *tierRetryProvider) WithBuiltins([]string) LLMProvider      { return p }

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

func TestCallLLMWithRetryUsesAlternateTierWhenModelIsOverloaded(t *testing.T) {
	for _, tc := range []struct {
		name     string
		selected ModelTier
		want     []string
	}{
		{name: "lighter model", selected: ModelMedium, want: []string{"medium-model", "small-model"}},
		{name: "stronger model", selected: ModelSmall, want: []string{"small-model", "medium-model"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := &tierRetryProvider{
				models: map[ModelTier]string{
					ModelLarge:  "large-model",
					ModelMedium: "medium-model",
					ModelSmall:  "small-model",
				},
				responses: map[string]ChatResponse{
					"medium-model": {Text: "medium recovered"},
					"small-model":  {Text: "small recovered"},
				},
				errors: map[string]error{tc.want[0]: errors.New("provider API error 429: model overloaded")},
			}
			thinker := retryTestThinker(provider)
			thinker.model = tc.selected

			resp, err := thinker.callLLMWithRetry(context.Background())
			if err != nil {
				t.Fatalf("callLLMWithRetry: %v", err)
			}
			if len(provider.calls) != len(tc.want) {
				t.Fatalf("model calls = %v, want %v", provider.calls, tc.want)
			}
			for i := range tc.want {
				if provider.calls[i] != tc.want[i] {
					t.Fatalf("model calls = %v, want %v", provider.calls, tc.want)
				}
			}
			if resp.Model != tc.want[1] {
				t.Fatalf("response model = %q, want %q", resp.Model, tc.want[1])
			}
			if thinker.model != tc.selected {
				t.Fatalf("configured tier changed to %s, want %s", thinker.model, tc.selected)
			}
		})
	}
}

func TestCallLLMWithRetryQuarantinesRejectedAttachmentAndContinues(t *testing.T) {
	provider := &scriptedRetryProvider{
		name:       "primary",
		failures:   1,
		failureErr: errors.New(`OpenAI Responses API error 400: {"message":"Error while downloading file. Upstream status code: 403.","param":"url"}`),
		response:   ChatResponse{Text: "recovered"},
	}
	thinker := retryTestThinker(provider)
	thinker.messages[1] = Message{
		Role:    "user",
		Content: "Continue even if the attachment is unavailable.",
		Parts: []ContentPart{
			{Type: "text", Text: "Continue even if the attachment is unavailable."},
			{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/frame.jpg?x=1%26y=2%26z=3"}},
		},
	}
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}

	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "recovered" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if provider.calls != 2 {
		t.Fatalf("provider calls = %d, want rejected call plus immediate clean retry", provider.calls)
	}
	if transientAttachmentCount(provider.messages[0]) != 1 {
		t.Fatalf("first request did not contain the attachment: %#v", provider.messages[0])
	}
	if transientAttachmentCount(provider.messages[1]) != 0 {
		t.Fatalf("clean retry retained the attachment: %#v", provider.messages[1])
	}
	if transientAttachmentCount(thinker.messages) != 0 || !strings.Contains(thinker.messages[1].Content, transientAttachmentHistoryNotice) {
		t.Fatalf("live history was not quarantined: %#v", thinker.messages)
	}
	events, _ := thinker.telemetry.Events(0)
	quarantined := 0
	for _, event := range events {
		if event.Type == "attachment.quarantined" {
			quarantined++
		}
	}
	if quarantined != 1 {
		t.Fatalf("attachment.quarantined events = %d, want 1", quarantined)
	}
}

func TestAttachmentDownloadFailureDoesNotFanOutToFallback(t *testing.T) {
	primary := &scriptedRetryProvider{
		name:       "primary",
		failures:   1,
		failureErr: errors.New("Unable to download the file. Please verify the URL and try again."),
		response:   ChatResponse{Text: "primary recovered"},
	}
	fallback := &scriptedRetryProvider{name: "fallback", response: ChatResponse{Text: "fallback"}}
	thinker := retryTestThinker(primary)
	thinker.messages[1].Parts = []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/broken.jpg"}}}
	thinker.pool = &ProviderPool{
		providers: map[string]LLMProvider{"primary": primary, "fallback": fallback},
		order:     []string{"primary", "fallback"},
		default_:  "primary",
	}

	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "primary recovered" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if primary.calls != 2 || fallback.calls != 0 {
		t.Fatalf("provider calls primary=%d fallback=%d, want 2/0", primary.calls, fallback.calls)
	}
}

func TestSuccessfulAttachmentIsVisibleOnceThenConsumed(t *testing.T) {
	provider := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "seen"}}
	thinker := retryTestThinker(provider)
	thinker.messages[1].Parts = []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/frame.jpg"}}}

	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	if transientAttachmentCount(provider.messages[0]) != 1 || transientAttachmentCount(provider.messages[1]) != 0 {
		t.Fatalf("attachment counts across requests = %d, %d; want 1, 0",
			transientAttachmentCount(provider.messages[0]), transientAttachmentCount(provider.messages[1]))
	}
}

func TestThinkWithProviderPersistsAttributedLLMStart(t *testing.T) {
	provider := &scriptedRetryProvider{name: "test-provider", response: ChatResponse{Text: "done"}}
	thinker := retryTestThinker(provider)
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}

	resp, err := thinker.thinkWithProviderMessages(context.Background(), provider, thinker.messages)
	if err != nil {
		t.Fatalf("thinkWithProviderMessages: %v", err)
	}
	if resp.Provider != "test-provider" || resp.Model != "test" {
		t.Fatalf("response attribution = provider %q model %q", resp.Provider, resp.Model)
	}

	allEvents, _ := thinker.telemetry.Events(0)
	var events []TelemetryEvent
	for _, event := range allEvents {
		if event.Type == "llm.request_budget" || event.Type == "llm.start" {
			events = append(events, event)
		}
	}
	if len(events) != 2 || events[0].Type != "llm.request_budget" || events[1].Type != "llm.start" {
		t.Fatalf("expected request budget followed by attributed llm.start, got %d events", len(events))
	}
	var budget requestBudget
	if err := json.Unmarshal(events[0].Data, &budget); err != nil || budget.OverBudget || budget.InputTokens == 0 {
		t.Fatalf("invalid preflight budget: %+v, %v", budget, err)
	}
	var data map[string]any
	if err := json.Unmarshal(events[1].Data, &data); err != nil {
		t.Fatalf("decode llm.start: %v", err)
	}
	if data["provider"] != "test-provider" || data["model"] != "test" {
		t.Fatalf("llm.start attribution = %#v", data)
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
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
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
		events, _ := thinker.telemetry.Events(0)
		cancelled := false
		for _, event := range events {
			if event.Type == "llm.cancelled" {
				cancelled = true
			}
		}
		if !cancelled {
			t.Fatalf("stored telemetry = %#v, want llm.cancelled", events)
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
