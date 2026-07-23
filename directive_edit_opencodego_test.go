package core

import (
	"os"
	"strings"
	"testing"
)

// TestOpenCodeGoRecurringInstructionUsesMainWakeLoop runs the recurring-work
// behavioral gate through OpenCode Go's default model. It shares the exact
// evolve/pace/no-spawn/no-scheduler assertions used by the Codex smoke.
//
//	RUN_OPENCODE_GO_RECURRING_EVOLVE_SMOKE=1 go test -v -run TestOpenCodeGoRecurringInstructionUsesMainWakeLoop -timeout 5m .
func TestOpenCodeGoRecurringInstructionUsesMainWakeLoop(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_RECURRING_EVOLVE_SMOKE") == "" {
		t.Skip("set RUN_OPENCODE_GO_RECURRING_EVOLVE_SMOKE=1 to run the OpenCode Go recurring evolve smoke")
	}
	if testing.Short() {
		t.Skip("skipping OpenCode Go recurring evolve smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}

	provider := NewOpenCodeGoProvider(key)
	if got := provider.Models()[ModelMedium]; got != "glm-5.2" {
		t.Fatalf("OpenCode Go medium model = %q, want glm-5.2", got)
	}
	runRecurringInstructionUsesMainWakeLoop(t, provider)
}

// TestOpenCodeGoKimiCode27RecurringInstructionUsesMainWakeLoop runs the same
// behavioral gate with OpenCode Go's Kimi Code 2.7 model while leaving GLM 5.2
// as the provider default.
//
//	RUN_OPENCODE_GO_KIMI_CODE_SMOKE=1 go test -v -run TestOpenCodeGoKimiCode27RecurringInstructionUsesMainWakeLoop -timeout 5m .
func TestOpenCodeGoKimiCode27RecurringInstructionUsesMainWakeLoop(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_KIMI_CODE_SMOKE") == "" {
		t.Skip("set RUN_OPENCODE_GO_KIMI_CODE_SMOKE=1 to run the Kimi Code 2.7 recurring evolve smoke")
	}
	if testing.Short() {
		t.Skip("skipping Kimi Code 2.7 recurring evolve smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}

	provider := NewOpenCodeGoProvider(key)
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		provider.Models()[tier] = "kimi-k2.7-code"
	}
	runRecurringInstructionUsesMainWakeLoop(t, provider)
}

// TestOpenCodeGoMiniMaxM3RecurringInstructionUsesMainWakeLoop runs the same
// behavioral gate with OpenCode Go's MiniMax M3 model.
//
//	RUN_OPENCODE_GO_MINIMAX_M3_SMOKE=1 go test -v -run TestOpenCodeGoMiniMaxM3RecurringInstructionUsesMainWakeLoop -timeout 5m .
func TestOpenCodeGoMiniMaxM3RecurringInstructionUsesMainWakeLoop(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_MINIMAX_M3_SMOKE") == "" {
		t.Skip("set RUN_OPENCODE_GO_MINIMAX_M3_SMOKE=1 to run the MiniMax M3 recurring evolve smoke")
	}
	if testing.Short() {
		t.Skip("skipping MiniMax M3 recurring evolve smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}

	provider := NewOpenCodeGoProvider(key)
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		provider.Models()[tier] = "minimax-m3"
	}
	runRecurringInstructionUsesMainWakeLoop(t, provider)
}

// TestOpenCodeGoDirectiveBehaviorSuite runs every provider-neutral directive
// behavior smoke through OpenCode Go's default GLM 5.2 model.
func TestOpenCodeGoDirectiveBehaviorSuite(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_DIRECTIVE_SUITE") == "" {
		t.Skip("set RUN_OPENCODE_GO_DIRECTIVE_SUITE=1 to run the GLM directive behavior suite")
	}
	if testing.Short() {
		t.Skip("skipping GLM directive behavior suite in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	newProvider := func() LLMProvider { return NewOpenCodeGoProvider(key) }
	runProviderDirectiveBehaviorSuite(t, newProvider)
}

func TestOpenCodeGoBoundedOneOffStaysOnMainSmoke(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_DIRECTIVE_SUITE") == "" {
		t.Skip("set RUN_OPENCODE_GO_DIRECTIVE_SUITE=1 to run the GLM bounded-work smoke")
	}
	if testing.Short() {
		t.Skip("skipping GLM bounded-work smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	runBoundedOneOffStaysOnMainSmoke(t, NewOpenCodeGoProvider(key))
}

func TestOpenCodeGoOwnershipScalingSmoke(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_OWNERSHIP_SMOKE") == "" {
		t.Skip("set RUN_OPENCODE_GO_OWNERSHIP_SMOKE=1 to run the GLM ownership smoke")
	}
	if testing.Short() {
		t.Skip("skipping GLM ownership smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	provider := NewOpenCodeGoProvider(key)
	t.Run("recurring responsibilities scale out", func(t *testing.T) {
		runRecurringResponsibilitiesScaleOutSmoke(t, provider)
	})
	t.Run("substantial parallel work delegates", func(t *testing.T) {
		runSubstantialWorkDelegatesSmoke(t, provider)
	})
}

// TestOpenCodeGoMiniMaxM3DirectiveBehaviorSuite runs the complete
// provider-neutral directive behavior suite through MiniMax M3.
func TestOpenCodeGoMiniMaxM3DirectiveBehaviorSuite(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_MINIMAX_M3_SUITE") == "" {
		t.Skip("set RUN_OPENCODE_GO_MINIMAX_M3_SUITE=1 to run the MiniMax M3 directive behavior suite")
	}
	if testing.Short() {
		t.Skip("skipping MiniMax M3 directive behavior suite in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	newProvider := func() LLMProvider {
		provider := NewOpenCodeGoProvider(key)
		for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
			provider.Models()[tier] = "minimax-m3"
		}
		return provider
	}
	runProviderDirectiveBehaviorSuite(t, newProvider)
}

func runProviderDirectiveBehaviorSuite(t *testing.T, newProvider func() LLMProvider) {
	t.Helper()
	t.Run("structured edit", func(t *testing.T) { runDirectiveEditSmoke(t, newProvider()) })
	t.Run("empty directive initialization", func(t *testing.T) { runEmptyDirectiveSectionInitSmoke(t, newProvider()) })
	t.Run("redundant heading recovery", func(t *testing.T) { runRedundantDirectiveHeadingSmoke(t, newProvider()) })
	t.Run("persistent intent", func(t *testing.T) { runPersistentIntentAutoEvolveSmoke(t, newProvider()) })
	t.Run("subthread persistent intent", func(t *testing.T) { runSubthreadPersistentIntentAutoEvolveSmoke(t, newProvider()) })
	t.Run("intent boundaries", func(t *testing.T) { runPersistentIntentBoundariesSmoke(t, newProvider()) })
	t.Run("full replacement guard", func(t *testing.T) { runMarkdownDirectiveRejectsFullReplaceSmoke(t, newProvider()) })
}
