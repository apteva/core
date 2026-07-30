package core

import (
	"os"
	"strings"
	"testing"
)

// TestOpenCodeGoGLM52PaceBehaviorSuite runs the same agent-owned wake
// scenarios as the Codex smoke through OpenCode Go's default GLM 5.2 model.
func TestOpenCodeGoGLM52PaceBehaviorSuite(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_PACE_SMOKE") != "1" {
		t.Skip("set RUN_OPENCODE_GO_PACE_SMOKE=1 to run the GLM 5.2 pace smoke")
	}
	if testing.Short() {
		t.Skip("skipping GLM 5.2 pace smoke in short mode")
	}
	key := openCodeGoPaceKey(t)
	newProvider := func() LLMProvider {
		provider := NewOpenCodeGoProvider(key)
		if got := provider.Models()[ModelLarge]; got != "glm-5.2" {
			t.Fatalf("OpenCode Go large model = %q, want glm-5.2", got)
		}
		return provider
	}
	runLivePaceBehaviorSuite(t, newProvider)
}

// TestOpenCodeGoKimiK3PaceBehaviorSuite runs the provider-neutral wake
// scenarios through Kimi K3 while retaining the production OpenCode Go
// transport and reasoning policy.
func TestOpenCodeGoKimiK3PaceBehaviorSuite(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_KIMI_K3_PACE_SMOKE") != "1" {
		t.Skip("set RUN_OPENCODE_GO_KIMI_K3_PACE_SMOKE=1 to run the Kimi K3 pace smoke")
	}
	if testing.Short() {
		t.Skip("skipping Kimi K3 pace smoke in short mode")
	}
	key := openCodeGoPaceKey(t)
	newProvider := func() LLMProvider {
		provider := NewOpenCodeGoProvider(key)
		for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
			provider.Models()[tier] = "kimi-k3"
		}
		if got := provider.Models()[ModelLarge]; got != "kimi-k3" {
			t.Fatalf("OpenCode Go large model = %q, want kimi-k3", got)
		}
		return provider
	}
	runLivePaceBehaviorSuite(t, newProvider)
}

func runLivePaceBehaviorSuite(t *testing.T, newProvider func() LLMProvider) {
	t.Helper()
	t.Run("early event preserves planned wake", func(t *testing.T) {
		runEarlyEventPreservesPlannedWakeSmoke(t, newProvider())
	})
	t.Run("recurring owner replans several responsibilities", func(t *testing.T) {
		runRecurringOwnerReplansSeveralResponsibilitiesSmoke(t, newProvider())
	})
}

func openCodeGoPaceKey(t *testing.T) string {
	t.Helper()
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	return key
}
