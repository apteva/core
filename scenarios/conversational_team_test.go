package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var conversationalTeamScenario = Scenario{
	Name: "ConversationalTeam",
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DATA_DIR": "{{dataDir}}"}},
	},
	Directive: "You are a helpful coordinator. You manage threads and route work. You do NOT have a pre-defined team — the user will tell you what to set up.",
	Phases: []Phase{
		{
			Name: "User asks to create a support thread",
			Setup: func(t *testing.T, dir string) {
				// Seed a helpdesk ticket so support has something to find
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "tickets.json"), []byte(`[{"id":"T-001","question":"My login is broken, please help","status":"open"}]`), 0644)
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Inject the user request on first call
				if th.Iteration() <= 2 {
					th.Inject("[console] Create a support thread that monitors the helpdesk for tickets. Give it access to the helpdesk MCP server.")
				}
				return th.Threads().Count() >= 1
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				threads := th.Threads().List()
				found := false
				for _, thr := range threads {
					if strings.Contains(strings.ToLower(thr.ID), "support") || strings.Contains(strings.ToLower(thr.Directive), "support") || strings.Contains(strings.ToLower(thr.Directive), "helpdesk") {
						found = true
						t.Logf("support thread found: id=%s", thr.ID)
					}
				}
				if !found {
					t.Errorf("expected a support/helpdesk thread, got: %v", threads)
				}
			},
			Timeout: 60 * time.Second,
		},
		{
			Name: "User asks to add a dev thread",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 4 {
					th.Inject("[console] Now create a dev thread that can read and write code using the codebase tools. It should fix bugs that support finds.")
				}
				return th.Threads().Count() >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				threads := th.Threads().List()
				foundDev := false
				for _, thr := range threads {
					if strings.Contains(strings.ToLower(thr.ID), "dev") || strings.Contains(strings.ToLower(thr.Directive), "code") {
						foundDev = true
						t.Logf("dev thread found: id=%s", thr.ID)
					}
				}
				if !foundDev {
					t.Errorf("expected a dev/code thread, got: %v", threads)
				}
			},
			Timeout: 60 * time.Second,
		},
		{
			Name: "User asks to check for tickets — support finds T-001",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 6 {
					th.Inject("[console] Ask support to check for open tickets right now.")
				}
				// Wait for support to find and report the ticket
				for _, m := range th.Messages() {
					if strings.Contains(m.Content, "T-001") || strings.Contains(m.Content, "login") {
						return true
					}
				}
				return false
			},
			Timeout: 90 * time.Second,
		},
		{
			Name: "User asks for status — coordinator reports team state",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 10 {
					th.Inject("[console] What threads do we have running and what are they doing?")
				}
				// Wait for a response that mentions the threads
				for i := len(th.Messages()) - 1; i >= 0; i-- {
					m := th.Messages()[i]
					if m.Role == "assistant" && (strings.Contains(strings.ToLower(m.Content), "support") && strings.Contains(strings.ToLower(m.Content), "dev")) {
						return true
					}
				}
				return false
			},
			Timeout: 60 * time.Second,
		},
	},
	Timeout: 5 * time.Minute,
}

func TestScenario_ConversationalTeam(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	helpdeskBin := BuildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := conversationalTeamScenario
	s.MCPServers[0].Command = helpdeskBin
	s.MCPServers[1].Command = codebaseBin
	RunScenario(t, s)
}
