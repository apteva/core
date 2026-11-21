package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// clearProviderEnv unsets every provider credential so a test sees only the
// variables it sets itself.
func clearProviderEnv(t *testing.T) {
	t.Helper()
	t.Setenv("CORE_PROVIDER", "")
	for _, keys := range providerRequiredEnv {
		for _, key := range keys {
			t.Setenv(key, "")
		}
	}
}

func TestProviderRequiredEnvMatchesWhatBuildsEachProvider(t *testing.T) {
	for name, keys := range providerRequiredEnv {
		t.Run(name, func(t *testing.T) {
			clearProviderEnv(t)
			for _, key := range keys {
				t.Setenv(key, "http://127.0.0.1:1")
			}
			if createProviderByName(name) == nil {
				t.Fatalf("%s did not build with %v set", name, keys)
			}
			for _, key := range keys {
				t.Setenv(key, "")
				if createProviderByName(name) != nil {
					t.Fatalf("%s built without %s", name, key)
				}
				if got := missingProviderEnv(name); len(got) != 1 || got[0] != key {
					t.Fatalf("missing = %v, want [%s]", got, key)
				}
				t.Setenv(key, "http://127.0.0.1:1")
			}
		})
	}
}

// The report that led here: a switch to the Claude subscription before the
// core had its connection id ran the agent on Anthropic instead.
func claudeSubscriptionWithoutConnection(t *testing.T) {
	t.Helper()
	clearProviderEnv(t)
	t.Setenv("ANTHROPIC_API_KEY", "test")
	t.Setenv("SERVER_URL", "http://127.0.0.1:5280")
	t.Setenv("APTEVA_API_KEY", "core-key")
}

func TestPoolKeepsAChosenDefaultThatCannotRun(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	pool, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: "anthropic"}, {Name: "claude-subscription", Default: true}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := pool.Default().Name(); got != "claude-subscription" {
		t.Fatalf("default = %s", got)
	}
	info := pool.Unavailable()
	if info == nil || info.Name != "claude-subscription" || info.Reason != "CLAUDE_SUBSCRIPTION_CONNECTION_ID was not set when the agent started" {
		t.Fatalf("unavailable = %#v", info)
	}
	// The other provider stays usable for an explicit choice.
	if pool.Get("anthropic") == nil || pool.Count() != 1 || pool.Get("claude-subscription") != nil {
		t.Fatalf("providers = %v", pool.Names())
	}
	_, err = pool.Default().Chat(context.Background(), nil, "", nil, nil, nil, nil)
	if !errors.Is(err, ErrProviderUnavailable) || !strings.Contains(err.Error(), "CLAUDE_SUBSCRIPTION_CONNECTION_ID") || !strings.Contains(err.Error(), "restart the agent") {
		t.Fatalf("chat err = %v", err)
	}
}

func TestPoolDoesNotAutoDetectAroundAChosenDefault(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	pool, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: "claude-subscription", Default: true}}})
	if err != nil {
		t.Fatalf("an agent whose only provider can't run should still start and say why: %v", err)
	}
	if pool.Count() != 0 || pool.Default().Name() != "claude-subscription" || pool.Unavailable() == nil {
		t.Fatalf("providers=%v default=%s", pool.Names(), pool.Default().Name())
	}
}

func TestPoolChoicesThatCanRunAreUnchanged(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	for name, tc := range map[string]struct {
		providers []ProviderConfig
		env       string
		want      string
	}{
		"no explicit default":      {[]ProviderConfig{{Name: "claude-subscription"}, {Name: "anthropic"}}, "", "anthropic"},
		"later default wins":       {[]ProviderConfig{{Name: "claude-subscription", Default: true}, {Name: "anthropic", Default: true}}, "", "anthropic"},
		"CORE_PROVIDER still wins": {[]ProviderConfig{{Name: "claude-subscription", Default: true}}, "anthropic", "anthropic"},
		"nothing configured":       {nil, "", "anthropic"},
		"default that can run":     {[]ProviderConfig{{Name: "anthropic", Default: true}}, "", "anthropic"},
		"unbuildable non-default":  {[]ProviderConfig{{Name: "anthropic", Default: true}, {Name: "claude-subscription"}}, "", "anthropic"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Setenv("CORE_PROVIDER", tc.env)
			pool, err := buildProviderPool(&Config{Providers: tc.providers})
			if err != nil {
				t.Fatal(err)
			}
			if got := pool.Default().Name(); got != tc.want || pool.Unavailable() != nil {
				t.Fatalf("default=%s unavailable=%#v", got, pool.Unavailable())
			}
		})
	}
}

func TestAnUnavailableDefaultIsNotRetriedOrHandedToAnotherProvider(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	other := &scriptedRetryProvider{name: "anthropic", response: ChatResponse{Text: "answered by anthropic"}}
	another := &scriptedRetryProvider{name: "openai", response: ChatResponse{Text: "answered by openai"}}
	pool := &ProviderPool{providers: map[string]LLMProvider{"anthropic": other, "openai": another}, order: []string{"anthropic", "openai"}}
	pool.markDefaultUnavailable("claude-subscription")

	th := retryTestThinker(pool.Default())
	th.pool = pool
	retries := 0
	th.retryDelay = func(error, int) time.Duration { retries++; return 0 }
	resp, err := th.callLLMWithRetry(context.Background())
	if !errors.Is(err, ErrProviderUnavailable) || resp.Text != "" {
		t.Fatalf("resp=%q err=%v", resp.Text, err)
	}
	if other.calls != 0 || another.calls != 0 || retries != 0 {
		t.Fatalf("anthropic=%d openai=%d retries=%d", other.calls, another.calls, retries)
	}
	if got := th.inferenceSnapshot().BlockedReason; !strings.HasPrefix(got, "provider_unavailable: ") {
		t.Fatalf("blocked reason = %q", got)
	}
}

func TestNewThinkerKeepsTheUnavailableDefault(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	cfg := &Config{Providers: []ProviderConfig{{Name: "anthropic"}, {Name: "claude-subscription", Default: true}}}
	provider, err := selectProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	th := NewThinker("", provider, cfg)
	if got := th.provider.Name(); got != "claude-subscription" {
		t.Fatalf("thinker runs on %s", got)
	}
	if th.pool.Get("claude-subscription") != nil {
		t.Fatal("the stand-in was added to the pool as if it could run")
	}
	if info := th.status().ProviderUnavailable; info == nil || info.Name != "claude-subscription" {
		t.Fatalf("status unavailable = %#v", info)
	}
}

func TestPutConfigReportsAProviderThatCannotRunAndRecoversOnAnother(t *testing.T) {
	claudeSubscriptionWithoutConnection(t)
	api, thinker := newPinTestAPI(t, newPinTestProvider("fake"))
	putProviderConfig(t, api, `{"providers":[{"name":"anthropic","default":true}]}`)
	if unavailableIn(t, api.status) != nil {
		t.Fatal("anthropic reported unavailable")
	}

	// Switch to the subscription: the change is taken, nothing else runs in its place.
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest(http.MethodPut, "/config", bytes.NewBufferString(`{"providers":[{"name":"anthropic"},{"name":"claude-subscription","default":true}]}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("PUT /config status=%d: %s", w.Code, w.Body.String())
	}
	var put struct {
		ProviderUnavailable *ProviderUnavailable `json:"provider_unavailable"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &put); err != nil || put.ProviderUnavailable == nil || put.ProviderUnavailable.Name != "claude-subscription" {
		t.Fatalf("PUT /config = %s", w.Body.String())
	}
	if thinker.provider.Name() != "claude-subscription" {
		t.Fatalf("thinker runs on %s", thinker.provider.Name())
	}
	for name, handler := range map[string]http.HandlerFunc{"/status": api.status, "/config": api.config} {
		if info := unavailableIn(t, handler); info == nil || info.Name != "claude-subscription" || !strings.Contains(info.Reason, "CLAUDE_SUBSCRIPTION_CONNECTION_ID") {
			t.Fatalf("GET %s unavailable = %#v", name, info)
		}
	}

	// Choosing a provider that can run applies live, as before.
	putProviderConfig(t, api, `{"providers":[{"name":"anthropic","default":true},{"name":"claude-subscription"}]}`)
	if thinker.provider.Name() != "anthropic" || unavailableIn(t, api.status) != nil || unavailableIn(t, api.config) != nil {
		t.Fatalf("thinker runs on %s; status still reports it unavailable", thinker.provider.Name())
	}
}

func unavailableIn(t *testing.T, handler http.HandlerFunc) *ProviderUnavailable {
	t.Helper()
	w := httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodGet, "/", nil))
	var body struct {
		ProviderUnavailable *ProviderUnavailable `json:"provider_unavailable"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v: %s", err, w.Body.String())
	}
	return body.ProviderUnavailable
}
