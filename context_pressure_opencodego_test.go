package core

import (
	"os"
	"strings"
	"testing"
)

func openCodeGoContextPressureProvider(t *testing.T) LLMProvider {
	t.Helper()
	if os.Getenv("RUN_OPENCODE_GO_CONTEXT_PRESSURE_SMOKE") == "" {
		t.Skip("set RUN_OPENCODE_GO_CONTEXT_PRESSURE_SMOKE=1 to run the OpenCode Go context-pressure smokes")
	}
	if testing.Short() {
		t.Skip("skipping OpenCode Go context-pressure smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	return NewOpenCodeGoProvider(key)
}

func TestOpenCodeGoLargeToolResultPreservedSmoke(t *testing.T) {
	runLargeToolResultPreservedSmoke(t, openCodeGoContextPressureProvider(t))
}

func TestOpenCodeGoSemanticCompactionSmoke(t *testing.T) {
	runSemanticCompactionSmoke(t, openCodeGoContextPressureProvider(t))
}
