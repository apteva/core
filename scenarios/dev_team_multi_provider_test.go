package scenarios

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var devTeamMultiProviderScenario = Scenario{
	Name: "DevTeamMultiProvider",
	Directive: `You manage a small development team maintaining a Todo SaaS app.
The codebase is in the "app/" directory. It is a Go package with todo.go and todo_test.go.

You have TWO providers available:
- fireworks (default) — fast and cheap, use for coordination, support, and QA
- openai — powerful (gpt-4.1), use for the dev thread that writes code

Spawn and maintain 3 threads:
1. "support" — monitors helpdesk tickets, triages them (bug vs feature), reports to main with recommendations.
   Tools: helpdesk_list_tickets, helpdesk_reply_ticket, helpdesk_close_ticket, send, done
2. "dev" — reads/writes code, implements features and fixes. MUST use provider="openai" for better code quality. Always reads existing code before modifying.
   Tools: codebase_read_file, codebase_write_file, codebase_list_files, codebase_search, send, done
3. "qa" — runs the test suite and reports results. Triggered by main after dev finishes.
   Tools: codebase_run_tests, codebase_read_file, send, done

Workflow:
- Support finds a ticket and tells you what it is
- You decide what to do and tell dev to implement it
- After dev is done, tell qa to run tests
- If tests fail, send dev back to fix. If pass, tell support to close the ticket.`,
	Providers: []ProviderConfig{
		{Name: "fireworks", Default: true},
		{Name: "openai"},
	},
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		seedTodoApp(t, dir)
	},
	Phases: []Phase{
		{
			Name:    "Startup — multi-provider pool ready",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Gate on the real precondition for this scenario — the
				// two-provider pool is configured. Whether the agent
				// then spawns a standing team or works inline is its
				// own call; the thread-provider check below is
				// informational.
				return th.Pool() != nil && th.Pool().Count() >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify pool has both providers
				if th.Pool() == nil {
					t.Error("expected provider pool")
					return
				}
				if th.Pool().Count() < 2 {
					t.Errorf("expected 2 providers in pool, got %d", th.Pool().Count())
				}
				if th.Pool().Get("fireworks") == nil {
					t.Error("expected fireworks in pool")
				}
				if th.Pool().Get("openai") == nil {
					t.Error("expected openai in pool")
				}
				// Log each thread's actual provider
				threads := th.Threads().List()
				for _, thread := range threads {
					t.Logf("thread %s: provider=%s model=%s", thread.ID, thread.Provider, thread.Model)
				}
				// Check if dev got openai
				for _, thread := range threads {
					if thread.ID == "dev" && thread.Provider == "openai" {
						t.Logf("OK: dev thread correctly using openai")
					} else if thread.ID == "dev" {
						t.Logf("NOTE: dev thread using %s (directive asked for openai)", thread.Provider)
					}
				}
			},
		},
		{
			Name:    "Feature request — add priority field (dev uses openai)",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				WriteJSONFile(t, dir, "tickets.json", []map[string]string{
					{"id": "T-101", "question": "Feature request: Please add a Priority field to todos. It should be a string with values low, medium, or high. Default to low. The Create function should accept an optional priority parameter."},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				code, err := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if err != nil {
					return false
				}
				if !strings.Contains(string(code), "Priority") {
					return false
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				code, _ := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if !strings.Contains(string(code), "Priority") {
					t.Error("expected Priority field in todo.go")
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("tests should pass after feature: %s", string(out))
				}
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_DevTeamMultiProvider(t *testing.T) {
	// Require both API keys
	if os.Getenv("FIREWORKS_API_KEY") == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	helpdeskBin := BuildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := devTeamMultiProviderScenario
	s.MCPServers[0].Command = helpdeskBin
	s.MCPServers[1].Command = codebaseBin
	RunScenario(t, s)
}
