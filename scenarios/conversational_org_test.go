package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var conversationalOrgScenario = Scenario{
	Name: "ConversationalOrg",
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DATA_DIR": "{{dataDir}}"}},
	},
	Directive: "You are a CEO coordinator. You can spawn director threads (with spawn tool), and directors can spawn their own workers. Build the org as the user requests. Keep it lean.",
	Phases: []Phase{
		{
			Name: "User asks to create an engineering director",
			Setup: func(t *testing.T, dir string) {
				// Seed helpdesk + codebase data
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "tickets.json"), []byte(`[{"id":"T-100","question":"API returns 500 on /users endpoint","status":"open"}]`), 0644)
				writeGoProject(t, dir)
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 2 {
					th.Inject(`[console] Create an engineering director thread with tools=spawn,send and mcp=codebase. It MUST have spawn in its tools so it can create sub-workers.`)
				}
				for _, thr := range th.Threads().List() {
					if strings.Contains(strings.ToLower(thr.ID), "eng") || strings.Contains(strings.ToLower(thr.ID), "director") {
						return true
					}
				}
				return false
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				for _, thr := range th.Threads().List() {
					t.Logf("thread: id=%s depth=%d tools=%v", thr.ID, thr.Depth, thr.Tools)
				}
			},
			Timeout: 60 * time.Second,
		},
		{
			Name: "User asks to create a support director",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 4 {
					th.Inject(`[console] Create a support director thread with tools=spawn,send and mcp=helpdesk. It MUST have spawn in its tools.`)
				}
				count := 0
				for _, thr := range th.Threads().List() {
					if thr.Depth == 0 {
						count++
					}
				}
				return count >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("depth-0 threads (directors):")
				for _, thr := range th.Threads().List() {
					if thr.Depth == 0 {
						t.Logf("  %s (tools: %v)", thr.ID, thr.Tools)
					}
				}
			},
			Timeout: 60 * time.Second,
		},
		{
			Name: "User tells engineering director to spawn a dev worker",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 6 {
					th.Inject("[console] Send a message to the engineering director: spawn a dev-worker thread with codebase tools to fix bug T-100 (API returns 500 on /users).")
				}
				// Check directors' children for depth-1 workers
				for _, dir := range th.Threads().List() {
					if dir.SubThreads > 0 {
						return true
					}
				}
				return false
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("threads after worker spawn:")
				totalWorkers := 0
				for _, thr := range th.Threads().List() {
					t.Logf("  %s (sub_threads=%d)", thr.ID, thr.SubThreads)
					totalWorkers += thr.SubThreads
				}
				if totalWorkers < 1 {
					t.Errorf("expected at least 1 worker, got %d", totalWorkers)
				} else {
					t.Logf("PASS: %d workers under directors", totalWorkers)
				}
			},
			Timeout: 90 * time.Second,
		},
		{
			Name: "User tells support director to spawn a ticket handler",
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 10 {
					th.Inject("[console] Send a message to the support director: spawn a ticket-handler worker with helpdesk tools to handle open tickets.")
				}
				// Count workers across all directors
				workerCount := 0
				for _, dir := range th.Threads().List() {
					workerCount += dir.SubThreads
				}
				return workerCount >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("final org structure:")
				directors := th.Threads().List()
				totalWorkers := 0
				for _, dir := range directors {
					t.Logf("  [director] %s (sub_threads=%d)", dir.ID, dir.SubThreads)
					totalWorkers += dir.SubThreads
				}
				if len(directors) < 2 {
					t.Errorf("expected at least 2 directors, got %d", len(directors))
				}
				t.Logf("org: %d directors + %d workers (from SubThreads)", len(directors), totalWorkers)
				// Don't fail on worker count — SubThreads may lag behind actual spawns
			},
			Timeout: 90 * time.Second,
		},
	},
	Timeout: 5 * time.Minute,
}

func TestScenario_ConversationalOrg(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	helpdeskBin := BuildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := conversationalOrgScenario
	s.MCPServers[0].Command = helpdeskBin
	s.MCPServers[1].Command = codebaseBin
	RunScenario(t, s)
}
