package core

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func anthropicOutputLimit(t *testing.T, pool *ProviderPool) int {
	t.Helper()
	provider, ok := pool.Get("anthropic").(*AnthropicProvider)
	if !ok {
		t.Fatalf("anthropic provider = %#v", pool.Get("anthropic"))
	}
	return provider.maxOutputTokens
}

func savedOutputLimit(t *testing.T, cfg *Config) int {
	t.Helper()
	providers := cfg.GetProviders()
	if len(providers) != 1 {
		t.Fatalf("providers = %#v", providers)
	}
	return providers[0].MaxOutputTokens
}

func TestMaxOutputTokensReachesTheAnthropicProvider(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	pool, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: "anthropic", Default: true, MaxOutputTokens: 4096}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := anthropicOutputLimit(t, pool); got != 4096 {
		t.Fatalf("limit = %d", got)
	}
	provider := pool.Get("anthropic").(*AnthropicProvider)
	if got := provider.outputTokenLimit("claude-fable-5"); got != 4096 {
		t.Fatalf("requests would ask for %d output tokens", got)
	}
}

func TestMaxOutputTokensRange(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	t.Setenv("OPENAI_API_KEY", "test")
	for name, tc := range map[string]struct {
		provider string
		limit    int
		valid    bool
	}{
		"unset":              {"anthropic", 0, true},
		"lowest":             {"anthropic", 1, true},
		"highest":            {"anthropic", 64000, true},
		"too high":           {"anthropic", 64001, false},
		"negative":           {"anthropic", -1, false},
		"unset elsewhere":    {"openai", 0, true},
		"another provider":   {"openai", 8000, false},
		"negative elsewhere": {"openai", -1, false},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: tc.provider, Default: true, MaxOutputTokens: tc.limit}}})
			if tc.valid {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), `"`+tc.provider+`"`) {
				t.Fatalf("err = %v, want one naming %s", err, tc.provider)
			}
		})
	}
}

func TestPutConfigRefusesAnInvalidOutputLimit(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	api, thinker := newPinTestAPI(t, newPinTestProvider("fake"))
	putProviderConfig(t, api, `{"providers":[{"name":"anthropic","default":true,"max_output_tokens":8000}]}`)

	for _, payload := range []string{
		`{"providers":[{"name":"anthropic","default":true,"max_output_tokens":999999}]}`,
		`{"provider":{"name":"anthropic","max_output_tokens":999999}}`,
	} {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest(http.MethodPut, "/config", bytes.NewBufferString(payload)))
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "max_output_tokens") {
			t.Fatalf("%s: status=%d: %s", payload, w.Code, w.Body.String())
		}
	}
	if got := savedOutputLimit(t, thinker.config); got != 8000 {
		t.Fatalf("a refused request changed the limit to %d", got)
	}
}

func TestPutConfigKeepsAndAppliesTheOutputLimit(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	api, thinker := newPinTestAPI(t, newPinTestProvider("fake"))
	putProviderConfig(t, api, `{"providers":[{"name":"anthropic","default":true,"max_output_tokens":8000}]}`)
	if got := anthropicOutputLimit(t, thinker.pool); got != 8000 {
		t.Fatalf("live limit = %d", got)
	}

	// Requests about something else leave it alone.
	putProviderConfig(t, api, `{"realtime_enabled":true}`)
	if got := savedOutputLimit(t, thinker.config); got != 8000 {
		t.Fatalf("a realtime toggle changed the saved limit to %d", got)
	}
	if got := anthropicOutputLimit(t, thinker.pool); got != 8000 {
		t.Fatalf("a realtime toggle changed the live limit to %d", got)
	}
	putProviderConfig(t, api, `{"provider":{"name":"anthropic","models":{"large":"claude-fable-5"}}}`)
	if got := savedOutputLimit(t, thinker.config); got != 8000 {
		t.Fatalf("a provider update changed the saved limit to %d", got)
	}

	// The single-provider form can set it.
	putProviderConfig(t, api, `{"provider":{"name":"anthropic","max_output_tokens":4000}}`)
	if got := savedOutputLimit(t, thinker.config); got != 4000 {
		t.Fatalf("saved limit = %d", got)
	}
	if got := anthropicOutputLimit(t, thinker.pool); got != 4000 {
		t.Fatalf("live limit = %d", got)
	}

	// GET returns it, so a client that reads and saves the config keeps it.
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest(http.MethodGet, "/config", nil))
	var body struct {
		Providers []ProviderConfig `json:"providers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Providers) != 1 || body.Providers[0].MaxOutputTokens != 4000 {
		t.Fatalf("GET /config = %s", w.Body.String())
	}

	// And a limit-only edit reaches running workers.
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	putProviderConfig(t, api, `{"provider":{"name":"anthropic","max_output_tokens":2000}}`)
	if got := anthropicOutputLimit(t, worker.Thinker.pool); got != 2000 {
		t.Fatalf("worker limit = %d", got)
	}
}

func TestStartingIgnoresAnInvalidSavedOutputLimit(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	t.Setenv("OPENAI_API_KEY", "test")
	t.Chdir(t.TempDir())
	saved := `{"directive":"Work.","providers":[
		{"name":"anthropic","default":true,"max_output_tokens":999999},
		{"name":"openai","max_output_tokens":8000}
	]}`
	if err := os.WriteFile(configFile, []byte(saved), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := NewConfig()
	if err := cfg.LoadError(); err != nil {
		t.Fatalf("a bad limit stopped the config loading: %v", err)
	}
	for _, provider := range cfg.GetProviders() {
		if provider.MaxOutputTokens != 0 {
			t.Fatalf("%s kept the invalid limit %d", provider.Name, provider.MaxOutputTokens)
		}
	}
	// The same call main makes at startup.
	if _, err := selectProvider(cfg); err != nil {
		t.Fatalf("the agent would not start: %v", err)
	}
}

func TestLoadingKeepsAValidSavedOutputLimit(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	t.Chdir(t.TempDir())
	if err := os.WriteFile(configFile, []byte(`{"directive":"Work.","provider":{"name":"anthropic","max_output_tokens":4096}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := NewConfig()
	if err := cfg.LoadError(); err != nil {
		t.Fatal(err)
	}
	if got := savedOutputLimit(t, cfg); got != 4096 {
		t.Fatalf("limit after migrating the legacy provider field = %d", got)
	}
	pool, err := buildProviderPool(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := anthropicOutputLimit(t, pool); got != 4096 {
		t.Fatalf("live limit = %d", got)
	}
}

func TestGetProvidersReturnsAnIndependentCopy(t *testing.T) {
	cfg := &Config{Providers: []ProviderConfig{{
		Name:              "anthropic",
		MaxOutputTokens:   4096,
		Models:            map[string]string{"large": "a"},
		RealtimeExecution: RealtimeExecutionConfig{PreloadTools: []string{"original"}},
	}}}
	copied := cfg.GetProviders()
	if copied[0].MaxOutputTokens != 4096 {
		t.Fatalf("limit = %d", copied[0].MaxOutputTokens)
	}
	copied[0].MaxOutputTokens = 1
	copied[0].Models["large"] = "changed"
	copied[0].RealtimeExecution.PreloadTools[0] = "changed"
	stored := cfg.Providers[0]
	if stored.MaxOutputTokens != 4096 || stored.Models["large"] != "a" || stored.RealtimeExecution.PreloadTools[0] != "original" {
		t.Fatalf("editing the copy changed the config: %#v", stored)
	}
}
