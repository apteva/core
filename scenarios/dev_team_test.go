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

var devTeamScenario = Scenario{
	Name: "DevTeam",
	Directive: `You manage a small development team maintaining a Todo SaaS app.
The codebase is in the "app/" directory. It is a Go package with todo.go and todo_test.go.

Spawn and maintain 3 threads:
1. "support" — monitors helpdesk tickets, triages them (bug vs feature), reports to main with recommendations.
   Tools: helpdesk_list_tickets, helpdesk_reply_ticket, helpdesk_close_ticket, send, done
2. "dev" — reads/writes code, implements features and fixes. Always reads existing code before modifying.
   Tools: codebase_read_file, codebase_write_file, codebase_list_files, codebase_search, send, done
3. "qa" — runs the test suite and reports results. Triggered by main after dev finishes.
   Tools: codebase_run_tests, codebase_read_file, send, done

Workflow:
- Support finds a ticket and tells you what it is
- You decide what to do and tell dev to implement it
- After dev is done, tell qa to run tests
- If tests fail, send dev back to fix. If pass, tell support to close the ticket.`,
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		seedTodoApp(t, dir)
	},
	Phases: []Phase{
		// No "startup — N threads spawned" phase: how the agent
		// organises itself (a standing team vs. handling tickets
		// inline) is its call. The phases below assert on the code
		// that lands and the tests that pass.
		{
			Name:    "Feature request — add priority field",
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
		{
			Name:    "Bug fix — empty title validation",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				WriteJSONFile(t, dir, "tickets.json", []map[string]string{
					{"id": "T-102", "question": "Bug report: Creating a todo with an empty title succeeds but it should not. The Create function should return an error when the title is empty."},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				code, err := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if err != nil {
					return false
				}
				if !strings.Contains(string(code), "error") && !strings.Contains(string(code), "Error") {
					return false
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				code, _ := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if !strings.Contains(string(code), "error") && !strings.Contains(string(code), "Error") {
					t.Error("expected error handling for empty title in todo.go")
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("tests should pass after bug fix: %s", string(out))
				}
			},
		},
	},
	Timeout: 8 * time.Minute,
}

func TestScenario_DevTeam(t *testing.T) {
	helpdeskBin := BuildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := devTeamScenario
	s.MCPServers[0].Command = helpdeskBin
	s.MCPServers[1].Command = codebaseBin
	RunScenario(t, s)
}
