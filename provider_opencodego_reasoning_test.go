package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenCodeGoReasoning_AutoUsesMediumBaselineAndExplicitOverride(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()

	provider := newOpenCodeGoTestProvider(t, srv.URL)
	resp, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "kimi-k3", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("default Chat: %v", err)
	}
	if got := bodies[0]["reasoning_effort"]; got != "medium" {
		t.Fatalf("default reasoning_effort = %#v, want medium", got)
	}
	if resp.RequestedReasoningEffort != "auto" || resp.EffectiveReasoningEffort != "medium" {
		t.Fatalf("default reasoning telemetry = requested %q effective %q", resp.RequestedReasoningEffort, resp.EffectiveReasoningEffort)
	}

	resp, err = provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "glm-5.2", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("GLM auto Chat: %v", err)
	}
	if got := bodies[1]["reasoning_effort"]; got != "medium" {
		t.Fatalf("GLM auto reasoning_effort = %#v, want medium", got)
	}
	if resp.RequestedReasoningEffort != "auto" || resp.EffectiveReasoningEffort != "medium" {
		t.Fatalf("GLM auto reasoning telemetry = requested %q effective %q", resp.RequestedReasoningEffort, resp.EffectiveReasoningEffort)
	}

	high := provider.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).(*OpenCodeGoProvider)
	resp, err = high.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "kimi-k3", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("explicit Chat: %v", err)
	}
	if got := bodies[2]["reasoning_effort"]; got != "high" {
		t.Fatalf("explicit reasoning_effort = %#v, want high", got)
	}
	if resp.RequestedReasoningEffort != "high" || resp.EffectiveReasoningEffort != "high" {
		t.Fatalf("explicit reasoning telemetry = requested %q effective %q", resp.RequestedReasoningEffort, resp.EffectiveReasoningEffort)
	}
}

func TestOpenCodeGoReasoning_UnsupportedModelRetriesAndRemembers(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		bodies = append(bodies, body)
		if body["model"] == "kimi-k3" && body["reasoning_effort"] != nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"reasoning_effort only supports max for this model"}}`))
			return
		}
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()

	provider := newOpenCodeGoTestProvider(t, srv.URL)
	resp, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "kimi-k3", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if resp.EffectiveReasoningEffort != "provider-default" {
		t.Fatalf("first effective reasoning = %q, want provider-default", resp.EffectiveReasoningEffort)
	}

	resp, err = provider.Chat(context.Background(), []Message{{Role: "user", Content: "again"}}, "kimi-k3", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("remembered Chat: %v", err)
	}
	if resp.EffectiveReasoningEffort != "provider-default" {
		t.Fatalf("remembered effective reasoning = %q, want provider-default", resp.EffectiveReasoningEffort)
	}

	resp, err = provider.Chat(context.Background(), []Message{{Role: "user", Content: "other"}}, "glm-5.2", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("other-model Chat: %v", err)
	}
	if resp.EffectiveReasoningEffort != "medium" {
		t.Fatalf("other-model effective reasoning = %q, want medium", resp.EffectiveReasoningEffort)
	}

	if len(bodies) != 4 {
		t.Fatalf("requests = %d, want rejected request + retry + remembered call + other model", len(bodies))
	}
	if bodies[0]["reasoning_effort"] != "medium" {
		t.Fatalf("first request reasoning_effort = %#v", bodies[0]["reasoning_effort"])
	}
	if _, present := bodies[1]["reasoning_effort"]; present {
		t.Fatalf("fallback request retained reasoning_effort: %#v", bodies[1])
	}
	if _, present := bodies[2]["reasoning_effort"]; present {
		t.Fatalf("remembered model retried reasoning_effort: %#v", bodies[2])
	}
	if bodies[3]["reasoning_effort"] != "medium" {
		t.Fatalf("GLM did not receive its medium auto default: %#v", bodies[3])
	}
}

func TestOpenCodeGoReasoning_ModelOverridesReachWrappedProvider(t *testing.T) {
	provider := NewOpenCodeGoProvider("test")
	applyModelOverrides(provider, map[string]string{
		"large": "kimi-k3", "medium": "kimi-k2.7-code", "small": "glm-5.2",
	})
	models := provider.Models()
	if models[ModelLarge] != "kimi-k3" || models[ModelMedium] != "kimi-k2.7-code" || models[ModelSmall] != "glm-5.2" {
		t.Fatalf("model overrides did not reach wrapped provider: %#v", models)
	}
}

func newOpenCodeGoTestProvider(t *testing.T, url string) *OpenCodeGoProvider {
	t.Helper()
	provider, ok := NewOpenCodeGoProvider("test-key").(*OpenCodeGoProvider)
	if !ok {
		t.Fatalf("NewOpenCodeGoProvider returned %T", NewOpenCodeGoProvider("test-key"))
	}
	provider.compat.url = url
	return provider
}

func writeOpenCodeGoTestStream(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
}
