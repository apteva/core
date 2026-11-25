package core

import (
	"context"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

// emptyAnswerProvider plays back scripted answers, then answers with text.
type emptyAnswerProvider struct {
	mu        sync.Mutex
	model     string
	responses []ChatResponse
	requests  [][]Message
}

// reasoningOnly is the answer some models give to a purely informational
// message: reasoning, but no text, tool calls or files.
func reasoningOnly(usage TokenUsage) ChatResponse {
	return ChatResponse{Reasoning: "Nothing to do for this notice.", Usage: usage}
}

func (p *emptyAnswerProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.requests = append(p.requests, cloneMessages(messages))
	if n := len(p.requests); n <= len(p.responses) {
		return p.responses[n-1], nil
	}
	return ChatResponse{Text: "Handled."}, nil
}

func (p *emptyAnswerProvider) calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.requests)
}

func (p *emptyAnswerProvider) request(call int) []Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	return cloneMessages(p.requests[call-1])
}

func (p *emptyAnswerProvider) Models() map[ModelTier]string {
	model := p.model
	if model == "" {
		model = "empty-test-model"
	}
	return map[ModelTier]string{ModelLarge: model, ModelMedium: model, ModelSmall: model}
}
func (p *emptyAnswerProvider) Name() string                           { return "empty-answer-provider" }
func (p *emptyAnswerProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *emptyAnswerProvider) SupportsNativeTools() bool              { return true }
func (p *emptyAnswerProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *emptyAnswerProvider) SetBuiltinTools([]string)               {}
func (p *emptyAnswerProvider) WithBuiltins([]string) LLMProvider      { return p }

func runEmptyAnswerThinker(t *testing.T, provider *emptyAnswerProvider) (*Thinker, *Config) {
	t.Helper()
	t.Chdir(t.TempDir())
	cfg := &Config{path: "config.json", Directive: "Coordinate the conversation threads."}
	thinker := NewThinker("", provider, cfg)
	done := make(chan struct{})
	go func() { defer close(done); thinker.Run() }()
	t.Cleanup(func() {
		thinker.Stop()
		<-done
		thinker.blobs.Close()
	})
	return thinker, cfg
}

// A model that keeps answering with reasoning alone ends the turn after a few
// tries instead of being asked again forever. The thread then waits like after
// any other turn, and the next event starts a new one.
func TestEmptyAnswersEndTheTurnAndWaitForTheNextEvent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		provider := &emptyAnswerProvider{}
		for range maxEmptyModelResponses {
			provider.responses = append(provider.responses, reasoningOnly(TokenUsage{PromptTokens: 7_000}))
		}
		thinker, cfg := runEmptyAnswerThinker(t, provider)

		time.Sleep(20 * time.Minute)
		synctest.Wait()
		if got := provider.calls(); got != maxEmptyModelResponses {
			t.Fatalf("model asked %d times, want %d and then a wait", got, maxEmptyModelResponses)
		}
		state := cfg.GetMainPace()
		if state == nil || state.WaitForEvents || state.NextWakeAt.IsZero() || !state.NextWakeAt.After(time.Now()) {
			t.Fatalf("turn did not end with the normal idle wake: %#v", state)
		}

		thinker.InjectConsole("Is anyone there?")
		synctest.Wait()
		if got := provider.calls(); got != maxEmptyModelResponses+1 {
			t.Fatalf("next event made %d calls in all, want %d", got, maxEmptyModelResponses+1)
		}
		next := provider.request(maxEmptyModelResponses + 1)
		if !strings.Contains(messagesText(next), "Is anyone there?") {
			t.Fatal("the new event is missing from the next request")
		}
		for _, message := range next {
			if message.Role == "assistant" && message.TextContent() == "" && len(message.ToolCalls) == 0 && len(message.Parts) == 0 {
				t.Fatal("an empty answer was kept in the history")
			}
		}
	})
}

// Near the context limit Core compacts once. When that removes nothing, the
// turn ends instead of backing off and asking again in a loop.
func TestEmptyAnswerNearTheLimitEndsTheTurnWhenCompactionCannotHelp(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		provider := &emptyAnswerProvider{model: "kimi-k2.6"}
		for range 10 {
			provider.responses = append(provider.responses, reasoningOnly(TokenUsage{PromptTokens: 205_000}))
		}
		runEmptyAnswerThinker(t, provider)

		time.Sleep(20 * time.Minute)
		synctest.Wait()
		if got := provider.calls(); got != 1 {
			t.Fatalf("model asked %d times, want 1 and then a wait", got)
		}
	})
}

// An assignment the model never answers is reported to the parent as failed,
// and the worker waits for its next assignment.
func TestEmptyAnswersFailTheAssignment(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		parent := newTestThinkerFull()
		parent.session = NewSession(t.TempDir(), "main")
		parent.registry = NewToolRegistry("")
		provider := &emptyAnswerProvider{}
		for range maxEmptyModelResponses {
			provider.responses = append(provider.responses, reasoningOnly(TokenUsage{}))
		}
		parent.provider = provider
		defer parent.Stop()
		if err := parent.threads.SpawnWithOpts("worker", "Return concise results", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
			t.Fatal(err)
		}
		defer parent.threads.KillAll()
		if err := parent.threads.Start("worker"); err != nil {
			t.Fatal(err)
		}
		w := ThreadWork{ID: "task", Revision: 1, Origin: "main", Kind: "request"}
		if err := parent.threads.QueueWork("worker", w, "Answer this assignment", nil); err != nil {
			t.Fatal(err)
		}

		time.Sleep(time.Minute)
		synctest.Wait()
		if got := provider.calls(); got != maxEmptyModelResponses {
			t.Fatalf("model asked %d times, want %d", got, maxEmptyModelResponses)
		}
		var result *Event
		for _, e := range parent.sub.DrainTargeted() {
			if e.Work != nil && e.Work.Kind == "result" {
				result = &e
			}
		}
		if result == nil || result.Work.Status != "failed" || result.Text != "The model returned no answer." || result.Work.Worker != "worker" {
			t.Fatalf("result = %+v", result)
		}
		if parent.threads.Count() != 1 {
			t.Fatal("the failed assignment stopped the worker")
		}
	})
}
