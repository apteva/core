package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexSpawnDirectiveIsFocusedAndSelfContained verifies that
// the spawn contract changes model behavior, not merely registry prose. Main
// receives deliberately broad standing context plus one narrow operation and
// must author a compact child directive that carries the operation's complete
// objective, constraints, scope, success criteria, and stop condition without
// copying unrelated parent responsibilities.
//
// Run:
//
//	RUN_CODEX_SPAWN_DIRECTIVE_SMOKE=1 go test -v -run TestIntegration_CodexSpawnDirectiveIsFocusedAndSelfContained -timeout 4m .
func TestIntegration_CodexSpawnDirectiveIsFocusedAndSelfContained(t *testing.T) {
	if os.Getenv("RUN_CODEX_SPAWN_DIRECTIVE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_SPAWN_DIRECTIVE_SMOKE=1 to run the Codex focused-spawn smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex focused-spawn smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := NewOpenAICodexProvider(token)
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Coordinate specialized workers for the operator.",
			"",
			"# Broad responsibilities",
			"UNRELATED_PAYROLL_CONTEXT: reconcile payroll exports and tax reports.",
			"UNRELATED_CRM_CONTEXT: supervise the CRM migration and sales pipeline cleanup.",
			"UNRELATED_EMAIL_CONTEXT: review newsletter and email-campaign analytics.",
			"UNRELATED_WAREHOUSE_CONTEXT: monitor inventory and warehouse deliveries.",
			"",
			"# Delegation",
			"Create focused workers when the operator explicitly asks for one.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	defer thinker.threads.KillAll()

	started := time.Now()
	go thinker.Run()
	thinker.InjectConsole(strings.Join([]string{
		"Create exactly one paused worker named patreon-schedule-guard for this operation; do not perform the operation yet.",
		"Its objective is to schedule only existing Patreon draft 166563499 for 14 August 2026 at 21:00 Europe/Madrid.",
		"It must never publish immediately and must never modify another post.",
		"Success requires verifying that Patreon shows the draft as Scheduled for that exact slot.",
		"If the target cannot be verified or no control explicitly labeled Schedule is available, it must stop without changes and report the blocker.",
	}, " "))

	const workerID = "patreon-schedule-guard"
	var directive string
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != "main" || event.Type != "tool.call" || event.Time.Before(started) {
				continue
			}
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) == nil && data.Name == "spawn" && data.Args["id"] == workerID {
				directive = strings.TrimSpace(data.Args["directive"])
				break
			}
		}
		if directive != "" {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if directive == "" {
		t.Fatal("Codex did not emit the requested focused spawn call")
	}
	t.Logf("Codex worker directive (%d chars):\n%s", len(directive), directive)

	lower := strings.ToLower(directive)
	for _, required := range []string{
		"166563499",
		"21:00",
		"europe/madrid",
		"schedule",
		"scheduled",
	} {
		if !strings.Contains(lower, required) {
			t.Errorf("spawned directive missing %q: %s", required, directive)
		}
	}
	if !containsAny(lower, "never publish", "do not publish", "must not publish", "without publishing") {
		t.Errorf("spawned directive lost the non-publication constraint: %s", directive)
	}
	if !containsAny(lower, "only existing", "only draft", "draft 166563499 only") {
		t.Errorf("spawned directive lost the exact-scope restriction: %s", directive)
	}
	if !containsAny(lower, "stop without changes", "leave unchanged", "make no changes", "do not change") ||
		!containsAny(lower, "report the blocker", "report a blocker", "report the specific blocker") {
		t.Errorf("spawned directive lost its stop/report condition: %s", directive)
	}
	for _, unrelated := range []string{"payroll", "crm migration", "email campaign", "warehouse"} {
		if strings.Contains(lower, unrelated) {
			t.Errorf("spawned directive copied unrelated parent context %q: %s", unrelated, directive)
		}
	}
	if len(directive) > 1600 {
		t.Errorf("spawned directive is not compact: %d chars", len(directive))
	}
}
