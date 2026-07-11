package core

import "testing"

func TestModelCapabilitiesOverrideContextAndEffectiveWindow(t *testing.T) {
	resetRuntimeModelCapabilities()
	t.Cleanup(resetRuntimeModelCapabilities)
	registerModelCapabilities(map[string]ModelCapabilities{
		"gpt-5.6-terra": {
			ContextWindow:                 400000,
			EffectiveContextWindowPercent: 95,
		},
	})

	if got := ModelContextWindow("GPT-5.6-TERRA"); got != 400000 {
		t.Fatalf("ModelContextWindow = %d, want 400000", got)
	}
	if got := ModelEffectiveContextWindow("gpt-5.6-terra"); got != 380000 {
		t.Fatalf("ModelEffectiveContextWindow = %d, want 380000", got)
	}
}

func TestModelReasoningEffortUsesCatalogLevels(t *testing.T) {
	resetRuntimeModelCapabilities()
	t.Cleanup(resetRuntimeModelCapabilities)
	registerModelCapabilities(map[string]ModelCapabilities{
		"gpt-5.6-luna": {
			SupportedReasoningLevels: []ModelReasoningCapability{
				{Effort: "low"},
				{Effort: "medium"},
				{Effort: "high"},
			},
		},
	})

	if got := modelReasoningEffort("gpt-5.6-luna", "xhigh"); got != "high" {
		t.Fatalf("modelReasoningEffort = %q, want high", got)
	}
	if got := modelReasoningEffort("gpt-5.6-luna", "medium"); got != "medium" {
		t.Fatalf("modelReasoningEffort = %q, want medium", got)
	}
}

func TestBuildProviderPoolReplacesRuntimeCapabilities(t *testing.T) {
	t.Setenv("OPENAI_CODEX_ACCESS_TOKEN", "test-token")
	trueValue := true
	first := &Config{Providers: []ProviderConfig{{
		Name: "openai-codex",
		ModelCapabilities: map[string]ModelCapabilities{
			"gpt-5.6-sol": {ContextWindow: 400000, SupportsParallelToolCalls: &trueValue},
		},
	}}}
	if _, err := buildProviderPool(first); err != nil {
		t.Fatalf("first buildProviderPool: %v", err)
	}
	if _, ok := capabilitiesForModel("gpt-5.6-sol"); !ok {
		t.Fatal("first capabilities were not registered")
	}

	second := &Config{Providers: []ProviderConfig{{
		Name: "openai-codex",
		ModelCapabilities: map[string]ModelCapabilities{
			"gpt-5.6-terra": {ContextWindow: 400000},
		},
	}}}
	if _, err := buildProviderPool(second); err != nil {
		t.Fatalf("second buildProviderPool: %v", err)
	}
	if _, ok := capabilitiesForModel("gpt-5.6-sol"); ok {
		t.Fatal("stale capabilities survived provider pool rebuild")
	}
	if _, ok := capabilitiesForModel("gpt-5.6-terra"); !ok {
		t.Fatal("replacement capabilities were not registered")
	}
}
