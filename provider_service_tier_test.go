package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func writeServiceTierTestStream(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n")
	_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n")
}

func TestOpenAICodexServiceTierIsSentAndPreservedByClones(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		writeServiceTierTestStream(w)
	}))
	defer srv.Close()

	p := (&OpenAINativeProvider{
		name:                   "openai-codex",
		apiKey:                 "token",
		responsesURL:           srv.URL,
		forceStoreFalse:        true,
		serviceTierUnsupported: &atomic.Bool{},
	}).WithServiceTier("priority")
	clone := p.WithReasoning(ReasoningSettings{Level: ReasoningLow}).(*OpenAINativeProvider)
	if clone.requestServiceTier() != "priority" {
		t.Fatalf("clone service tier = %q, want priority", clone.requestServiceTier())
	}
	if _, err := clone.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "gpt-6-sol", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if body["service_tier"] != "priority" {
		t.Fatalf("service_tier = %#v, want priority", body["service_tier"])
	}
}

func TestOpenAIServiceTierDoesNotLeakFromCodexProfile(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		writeServiceTierTestStream(w)
	}))
	defer srv.Close()

	p := (&OpenAINativeProvider{
		name:                   "openai",
		apiKey:                 "token",
		responsesURL:           srv.URL,
		serviceTier:            "priority",
		serviceTierUnsupported: &atomic.Bool{},
	})
	if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "gpt-6-sol", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, ok := body["service_tier"]; ok {
		t.Fatalf("OpenAI request leaked service_tier: %#v", body["service_tier"])
	}
}

func TestOpenAICodexServiceTierFallsBackAndDisablesAfterUnsupportedError(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		if _, ok := body["service_tier"]; ok {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"detail":"Unsupported service_tier: priority"}`)
			return
		}
		writeServiceTierTestStream(w)
	}))
	defer srv.Close()

	p := (&OpenAINativeProvider{
		name:                   "openai-codex",
		apiKey:                 "token",
		responsesURL:           srv.URL,
		forceStoreFalse:        true,
		serviceTierUnsupported: &atomic.Bool{},
	}).WithServiceTier("priority")
	messages := []Message{{Role: "user", Content: "hello"}}
	if _, err := p.Chat(context.Background(), messages, "gpt-6-sol", nil, nil, nil, nil); err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if _, err := p.Chat(context.Background(), messages, "gpt-6-sol", nil, nil, nil, nil); err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	if len(bodies) != 3 {
		t.Fatalf("requests = %d, want initial tier request, fallback, and remembered fallback", len(bodies))
	}
	if bodies[0]["service_tier"] != "priority" {
		t.Fatalf("initial service_tier = %#v, want priority", bodies[0]["service_tier"])
	}
	for i, body := range bodies[1:] {
		if _, ok := body["service_tier"]; ok {
			t.Fatalf("request %d retained unsupported service_tier: %#v", i+2, body["service_tier"])
		}
	}
}

func TestProviderConfigServiceTierBuildsCodexAndRoundTrips(t *testing.T) {
	t.Setenv("OPENAI_CODEX_ACCESS_TOKEN", "config-token")
	cfg := &Config{Providers: []ProviderConfig{{
		Name:        "openai-codex",
		Default:     true,
		ServiceTier: "priority",
		Models:      map[string]string{"large": "gpt-6-sol"},
	}}}

	providers := cfg.GetProviders()
	if len(providers) != 1 || providers[0].ServiceTier != "priority" {
		t.Fatalf("provider config copy = %#v", providers)
	}
	pool, err := buildProviderPool(cfg)
	if err != nil {
		t.Fatalf("buildProviderPool: %v", err)
	}
	native, ok := pool.Default().(*OpenAINativeProvider)
	if !ok {
		t.Fatalf("default provider = %T, want *OpenAINativeProvider", pool.Default())
	}
	if native.requestServiceTier() != "priority" {
		t.Fatalf("configured service tier = %q, want priority", native.requestServiceTier())
	}
	if native.Models()[ModelLarge] != "gpt-6-sol" {
		t.Fatalf("configured large model = %q", native.Models()[ModelLarge])
	}

	merged := mergeProviderConfig(providers, ProviderConfig{Name: "openai-codex", ServiceTier: "priority"})
	if len(merged) != 1 || merged[0].ServiceTier != "priority" {
		t.Fatalf("merged provider config = %#v", merged)
	}
}

func TestProviderConfigRejectsServiceTierForOtherProviders(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "api-key")
	_, err := buildProviderPool(&Config{Providers: []ProviderConfig{{
		Name:        "openai",
		Default:     true,
		ServiceTier: "priority",
	}}})
	if err == nil || !strings.Contains(err.Error(), "does not support service_tier") {
		t.Fatalf("buildProviderPool error = %v, want unsupported service_tier", err)
	}
}

func TestProviderConfigRejectsUnknownCodexServiceTier(t *testing.T) {
	t.Setenv("OPENAI_CODEX_ACCESS_TOKEN", "config-token")
	_, err := buildProviderPool(&Config{Providers: []ProviderConfig{{
		Name:        "openai-codex",
		Default:     true,
		ServiceTier: "fast",
	}}})
	if err == nil || !strings.Contains(err.Error(), "supported: priority") {
		t.Fatalf("buildProviderPool error = %v, want invalid tier", err)
	}
}
