package core

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestCodexDirectiveEditSmoke is a release-gate smoke for the real Codex
// provider. It verifies that the live model can use the new Markdown patch
// surface instead of rewriting the whole directive.
//
//	RUN_CODEX_DIRECTIVE_EDIT_SMOKE=1 OPENAI_CODEX_ACCESS_TOKEN=... go test -run TestCodexDirectiveEditSmoke -timeout 5m .
func TestCodexDirectiveEditSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_DIRECTIVE_EDIT_SMOKE") == "" {
		t.Skip("set RUN_CODEX_DIRECTIVE_EDIT_SMOKE=1 to run the Codex directive edit smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex directive edit smoke in short mode")
	}
	token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"))
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set")
	}

	t.Chdir(t.TempDir())
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"You maintain this directive when asked.",
			"",
			"# Schedule",
			"- daily_check: 09:00 Europe/Madrid",
			"",
			"# Goals",
			"- Keep the directive structured.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	provider := NewOpenAICodexProvider(token)
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()

	go thinker.Run()
	thinker.InjectConsole(strings.Join([]string{
		"Update your own directive now.",
		"Change only the Schedule line for daily_check to 07:30 Europe/Madrid.",
		`Use the evolve tool with edit_mode="section_replace_line", section="Schedule", match="daily_check:", and content="- daily_check: 07:30 Europe/Madrid".`,
		"Do not rewrite the full directive. Do not spawn, update children, or change tools.",
		"After the tool succeeds, reply exactly: RESULT: directive patched",
	}, "\n"))

	deadline := time.After(4 * time.Minute)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-deadline:
			t.Fatalf("directive was not patched; current directive:\n%s", cfg.GetDirective())
		case <-tick.C:
			directive := cfg.GetDirective()
			if strings.Contains(directive, "- daily_check: 07:30 Europe/Madrid") &&
				strings.Contains(directive, "# Goals\n- Keep the directive structured.") &&
				!strings.Contains(directive, "09:00 Europe/Madrid") {
				t.Logf("final directive:\n%s", directive)
				return
			}
		}
	}
}
