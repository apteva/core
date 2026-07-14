package core

import (
	"os"
	"strings"
	"testing"
)

func TestOpenCodeGoMemoryBehaviorSuite(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_MEMORY_SUITE") == "" {
		t.Skip("set RUN_OPENCODE_GO_MEMORY_SUITE=1 to run the GLM memory behavior suite")
	}
	if testing.Short() {
		t.Skip("skipping GLM memory behavior suite in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	newProvider := func() LLMProvider { return NewOpenCodeGoProvider(key) }

	t.Run("lexical recall without embeddings", func(t *testing.T) {
		runUsesLexicalMemoryWithoutEmbeddings(t, newProvider())
	})
	t.Run("ephemeral recall across turns", func(t *testing.T) {
		runUsesEphemeralMemoryAcrossTurns(t, newProvider())
	})
	t.Run("auto-spawned unconscious consolidation", func(t *testing.T) {
		runAutoSpawnedUnconsciousCreatesLexicalMemory(t, newProvider())
	})
	t.Run("threshold persistence and reload", func(t *testing.T) {
		runUnconsciousHistoryGrowthCreatesPersistentMemory(t, newProvider())
	})
}
