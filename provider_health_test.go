package core

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

var (
	errKeyRejected = errors.New("Anthropic API error 401: invalid x-api-key")
	errOpenAIKey   = errors.New("API error 401: Incorrect API key provided (request_ids=[])")
	errUpstream    = errors.New("API error 500: upstream failure (request_ids=[])")
	errRateLimited = errors.New("API error 429: rate limited (request_ids=[])")
)

// poolTestThinker builds a thinker whose pool holds the providers in order,
// the first being the default, with a controllable clock on provider health.
func poolTestThinker(t *testing.T, providers ...*scriptedRetryProvider) (*Thinker, *time.Time) {
	t.Helper()
	pool := &ProviderPool{providers: map[string]LLMProvider{}, default_: providers[0].name}
	for _, provider := range providers {
		pool.providers[provider.name] = provider
		pool.order = append(pool.order, provider.name)
	}
	thinker := retryTestThinker(providers[0])
	thinker.pool = pool
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	now := time.Date(2026, 11, 24, 10, 16, 37, 0, time.UTC)
	pool.healthStore().now = func() time.Time { return now }
	return thinker, &now
}

func providerNames(events []map[string]any) []string {
	out := []string{}
	for _, event := range events {
		name, _ := event["provider"].(string)
		out = append(out, name)
	}
	return out
}

func TestFallbackWalksPastAFailedFallback(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected}
	dead := &scriptedRetryProvider{name: "openai", failures: 100, failureErr: errOpenAIKey}
	spare := &scriptedRetryProvider{name: "claude-subscription", response: ChatResponse{Text: "from the third provider"}}
	thinker, _ := poolTestThinker(t, primary, dead, spare)

	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "from the third provider" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if primary.calls != 1 || dead.calls != 1 || spare.calls != 1 {
		t.Fatalf("calls primary=%d openai=%d spare=%d, want 1/1/1", primary.calls, dead.calls, spare.calls)
	}
	if got := providerNames(traceEvents(t, thinker.telemetry, "llm.provider_error")); strings.Join(got, ",") != "anthropic,openai" {
		t.Fatalf("llm.provider_error providers = %v", got)
	}
	unhealthy := traceEvents(t, thinker.telemetry, "llm.provider_unhealthy")
	if got := providerNames(unhealthy); strings.Join(got, ",") != "anthropic,openai" || unhealthy[0]["reason"] != "credentials rejected (HTTP 401)" {
		t.Fatalf("llm.provider_unhealthy = %v", unhealthy)
	}
	if thinker.provider != primary {
		t.Fatal("a fallback replaced the configured primary provider")
	}
}

func TestUnhealthyProvidersAreSkippedUntilTheCooldownPasses(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected, response: ChatResponse{Text: "primary is back"}}
	dead := &scriptedRetryProvider{name: "openai", failures: 100, failureErr: errOpenAIKey}
	spare := &scriptedRetryProvider{name: "claude-subscription", response: ChatResponse{Text: "spare"}}
	thinker, now := poolTestThinker(t, primary, dead, spare)
	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Within the cooldown the rejected providers get no request at all.
	*now = now.Add(5 * time.Minute)
	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "spare" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if primary.calls != 1 || dead.calls != 1 || spare.calls != 2 {
		t.Fatalf("calls primary=%d openai=%d spare=%d, want 1/1/2", primary.calls, dead.calls, spare.calls)
	}
	if got := traceEvents(t, thinker.telemetry, "llm.provider_error"); len(got) != 2 {
		t.Fatalf("a skipped provider was reported as failing again: %v", got)
	}
	health := thinker.pool.ProviderHealth()
	if len(health) != 2 || health[0].Provider != "anthropic" || health[1].Provider != "openai" || health[0].State != "unhealthy" {
		t.Fatalf("provider health = %+v", health)
	}

	// After the cooldown one real request finds out the key works again.
	*now = now.Add(6 * time.Minute)
	primary.failures = 1
	resp, err = thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "primary is back" || primary.calls != 2 {
		t.Fatalf("response=%q err=%v primary calls=%d", resp.Text, err, primary.calls)
	}
	recovered := traceEvents(t, thinker.telemetry, "llm.provider_recovered")
	if len(recovered) != 1 || recovered[0]["provider"] != "anthropic" || recovered[0]["unhealthy_seconds"] != float64(660) {
		t.Fatalf("llm.provider_recovered = %v", recovered)
	}
	if health := thinker.pool.ProviderHealth(); len(health) != 1 || health[0].Provider != "openai" {
		t.Fatalf("provider health after recovery = %+v", health)
	}
}

func TestStillRejectedProviderIsTriedOncePerCooldown(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected}
	spare := &scriptedRetryProvider{name: "fireworks", response: ChatResponse{Text: "spare"}}
	thinker, now := poolTestThinker(t, primary, spare)
	for i := 0; i < 3; i++ {
		if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if primary.calls != 1 {
		t.Fatalf("primary calls=%d within the cooldown, want 1", primary.calls)
	}
	*now = now.Add(providerUnhealthyCooldown)
	for i := 0; i < 3; i++ {
		if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if primary.calls != 2 {
		t.Fatalf("primary calls=%d after one cooldown, want 2", primary.calls)
	}
	health := thinker.pool.ProviderHealth()
	if len(health) != 1 || health[0].Failures != 2 || !health[0].RetryAt.Equal(now.Add(providerUnhealthyCooldown)) {
		t.Fatalf("provider health = %+v", health)
	}
	if got := traceEvents(t, thinker.telemetry, "llm.provider_unhealthy"); len(got) != 1 {
		t.Fatalf("llm.provider_unhealthy emitted %d times, want once per outage", len(got))
	}
}

func TestTransientFailureDoesNotMarkAProviderUnhealthy(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected}
	limited := &scriptedRetryProvider{name: "fireworks", failures: 100, failureErr: errRateLimited}
	spare := &scriptedRetryProvider{name: "opencode-go", response: ChatResponse{Text: "spare"}}
	thinker, _ := poolTestThinker(t, primary, limited, spare)
	for i := 0; i < 2; i++ {
		if resp, err := thinker.callLLMWithRetry(context.Background()); err != nil || resp.Text != "spare" {
			t.Fatalf("request %d: response=%q err=%v", i+1, resp.Text, err)
		}
	}
	if limited.calls != 2 {
		t.Fatalf("rate-limited fallback calls=%d, want 2: a 429 must not stop it being tried", limited.calls)
	}
	if health := thinker.pool.ProviderHealth(); len(health) != 1 || health[0].Provider != "anthropic" {
		t.Fatalf("provider health = %+v", health)
	}
}

// Before the walk, every retry of a transient primary failure picked the same
// first fallback again, so a later provider was never reached.
func TestTransientPrimaryFailureReachesALaterFallback(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errUpstream}
	dead := &scriptedRetryProvider{name: "openai", failures: 100, failureErr: errOpenAIKey}
	flaky := &scriptedRetryProvider{name: "opencode-go", failures: 1, failureErr: errUpstream, response: ChatResponse{Text: "second attempt"}}
	thinker, _ := poolTestThinker(t, primary, dead, flaky)
	resp, err := thinker.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "second attempt" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if primary.calls != 2 || dead.calls != 1 || flaky.calls != 2 {
		t.Fatalf("calls primary=%d openai=%d opencode-go=%d, want 2/1/2", primary.calls, dead.calls, flaky.calls)
	}
}

func TestWhenNoProviderAnswersTheErrorNamesEach(t *testing.T) {
	primary := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected}
	dead := &scriptedRetryProvider{name: "openai", failures: 100, failureErr: errOpenAIKey}
	down := &scriptedRetryProvider{name: "opencode-go", failures: 100, failureErr: errUpstream}
	thinker, _ := poolTestThinker(t, primary, dead, down)

	_, err := thinker.callLLMWithRetry(context.Background())
	for _, want := range []string{"no provider answered (3 tried or skipped)", "primary anthropic: Anthropic API error 401", "fallback openai: API error 401", "fallback opencode-go: API error 500"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("error %v lacks %q", err, want)
		}
	}
	if !errors.Is(err, errUpstream) || !permanentProviderError(err) {
		t.Fatalf("error lost its causes: %v", err)
	}
	if got := traceEvents(t, thinker.telemetry, "llm.error"); len(got) != 1 || got[0]["error"] != err.Error() {
		t.Fatalf("llm.error = %v", got)
	}

	// The next request names the providers it skipped and why.
	_, err = thinker.callLLMWithRetry(context.Background())
	for _, want := range []string{"primary anthropic: not called: credentials rejected (HTTP 401) since 2026-11-24T10:16:37Z", "fallback openai: not called: credentials rejected", "fallback opencode-go: API error 500"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("second error %v lacks %q", err, want)
		}
	}
	if primary.calls != 1 || dead.calls != 1 || down.calls != 2 {
		t.Fatalf("calls primary=%d openai=%d opencode-go=%d, want 1/1/2", primary.calls, dead.calls, down.calls)
	}
}

func TestASoleProviderIsAlwaysCalled(t *testing.T) {
	only := &scriptedRetryProvider{name: "anthropic", failures: 100, failureErr: errKeyRejected}
	thinker, _ := poolTestThinker(t, only)
	for i := 0; i < 2; i++ {
		if _, err := thinker.callLLMWithRetry(context.Background()); err == nil {
			t.Fatal("rejected key answered")
		}
	}
	if only.calls != 2 {
		t.Fatalf("sole provider calls=%d, want 2: nothing could answer in its place", only.calls)
	}
	if health := thinker.pool.ProviderHealth(); len(health) != 1 || health[0].Failures != 2 {
		t.Fatalf("provider health = %+v", health)
	}
}

func TestProviderCredentialFailureIsNarrowerThanPermanent(t *testing.T) {
	for _, tc := range []struct {
		err       error
		unhealthy bool
	}{
		{errors.New("Anthropic API error 401: invalid x-api-key"), true},
		{errors.New("API error 402: payment required"), true},
		{errors.New("Gemini API error 403: PERMISSION_DENIED"), true},
		{errors.New("provider said invalid_api_key"), true},
		{errors.New("API error 400: messages: tool_use ids must be unique"), false},
		{errors.New("API error 404: model not found"), false},
		{errors.New("API error 422: unprocessable"), false},
		{errors.New("API error 429: rate limited"), false},
		{errors.New("API error 500: upstream failure"), false},
		{errors.New("context_length_exceeded"), false},
		{context.DeadlineExceeded, false},
		{&providerUnhealthyError{}, false},
	} {
		if _, got := providerCredentialFailure(tc.err); got != tc.unhealthy {
			t.Errorf("%v: unhealthy=%v, want %v", tc.err, got, tc.unhealthy)
		}
	}
}

func TestProviderConfigChangeKeepsHealthOfUnchangedProviders(t *testing.T) {
	configs := []ProviderConfig{
		{Name: "anthropic", Default: true, Models: map[string]string{"large": "claude"}},
		{Name: "openai", Models: map[string]string{"large": "gpt"}},
	}
	oldPool := &ProviderPool{}
	oldPool.healthStore().markUnhealthy("anthropic", "credentials rejected (HTTP 401)", errKeyRejected)
	oldPool.healthStore().markUnhealthy("openai", "credentials rejected (HTTP 401)", errOpenAIKey)
	names := func(pool *ProviderPool) string {
		out := []string{}
		for _, status := range pool.ProviderHealth() {
			out = append(out, status.Provider)
		}
		return strings.Join(out, ",")
	}

	// A settings save that resends the same providers shares one record, so
	// child threads that keep the old pool see the same state.
	same := &ProviderPool{}
	same.inheritHealth(oldPool, configs, cloneProviderConfigs(configs), false)
	if same.healthStore() != oldPool.healthStore() {
		t.Fatal("unchanged providers did not share the health record")
	}

	// Changing which provider is the default is not a change to either one.
	switched := cloneProviderConfigs(configs)
	switched[0].Default, switched[1].Default = false, true
	moved := &ProviderPool{}
	moved.inheritHealth(oldPool, configs, switched, false)
	if names(moved) != "anthropic,openai" || moved.healthStore() == oldPool.healthStore() {
		t.Fatalf("default switch: health=%q shared=%v", names(moved), moved.healthStore() == oldPool.healthStore())
	}

	// A changed provider starts fresh; the others keep their state.
	changed := cloneProviderConfigs(configs)
	changed[0].Models = map[string]string{"large": "claude-next"}
	fresh := &ProviderPool{}
	fresh.inheritHealth(oldPool, configs, changed, false)
	if names(fresh) != "openai" {
		t.Fatalf("after a model change health=%q, want openai only", names(fresh))
	}

	// A removed provider is dropped.
	removed := &ProviderPool{}
	removed.inheritHealth(oldPool, configs, configs[:1], false)
	if names(removed) != "anthropic" {
		t.Fatalf("after removing openai health=%q", names(removed))
	}
}

func TestStatusReportsUnhealthyProviders(t *testing.T) {
	api, thinker := newTestAPI()
	read := func() map[string]any {
		w := httptest.NewRecorder()
		api.status(w, httptest.NewRequest("GET", "/status", nil))
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		return body
	}
	if _, listed := read()["provider_health"]; listed {
		t.Fatal("provider_health listed with no pool")
	}
	thinker.pool = &ProviderPool{}
	if _, listed := read()["provider_health"]; listed {
		t.Fatal("provider_health listed while every provider is healthy")
	}
	thinker.pool.healthStore().markUnhealthy("anthropic", "credentials rejected (HTTP 401)", errKeyRejected)
	health, _ := read()["provider_health"].([]any)
	if len(health) != 1 {
		t.Fatalf("provider_health = %v", health)
	}
	row := health[0].(map[string]any)
	if row["provider"] != "anthropic" || row["state"] != "unhealthy" || row["reason"] != "credentials rejected (HTTP 401)" || !strings.Contains(row["error"].(string), "invalid x-api-key") {
		t.Fatalf("provider_health row = %v", row)
	}
}
