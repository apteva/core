package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var cloneTeamScenario = Scenario{
	Name: "CloneTeam",
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
	},
	Directive: `You are a CEO coordinator. You spawn director threads, and directors spawn their own workers. When a user asks you to clone an existing team:
1. Use the send tool to ask the existing director for (a) its current directive verbatim, (b) the ids/directives/tools of each worker under it.
2. Wait for the director's reply to arrive in your next thought.
3. Use spawn to create a mirror director with a "_b" suffix on its id, copying the directive and tools you were told.
4. Use send to tell the new director to spawn the same workers (with "_b" suffixes on their ids).
Keep it lean — one director branch is enough.`,
	Phases: []Phase{
		{
			Name: "Bootstrap team A — support director + ticket handler",
			Setup: func(t *testing.T, dir string) {
				os.MkdirAll(dir, 0755)
				os.WriteFile(filepath.Join(dir, "tickets.json"), []byte(`[{"id":"T-200","question":"Password reset link is broken","status":"open"}]`), 0644)
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 2 {
					th.Inject(`[console] Create a director thread with id="support_director_a", tools=spawn,send and mcp=helpdesk. Then tell it (via send) to spawn a worker with id="ticket_handler_a" using helpdesk tools to handle open tickets.`)
				}
				workers := 0
				haveA := false
				for _, thr := range th.Threads().List() {
					if strings.Contains(thr.ID, "support_director_a") {
						haveA = true
					}
					workers += thr.SubThreads
				}
				return haveA && workers >= 1
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("team A:")
				for _, thr := range th.Threads().List() {
					t.Logf("  %s (depth=%d sub_threads=%d)", thr.ID, thr.Depth, thr.SubThreads)
				}
			},
			Timeout: 120 * time.Second,
		},
		{
			Name: "Clone team A as team B — CEO queries director, rebuilds mirror",
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.Inject(`[console] Clone the support team as team B. Follow your directive: send support_director_a to ask for its state, then spawn support_director_b with the same directive/tools, then send the new director to spawn a mirror worker ticket_handler_b.`)
					}
					haveB := false
					mirrorWorkers := 0
					for _, thr := range th.Threads().List() {
						if strings.Contains(thr.ID, "support_director_b") {
							haveB = true
							mirrorWorkers += thr.SubThreads
						}
					}
					return haveB && mirrorWorkers >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("final org:")
				var aDir, bDir *ThreadInfo
				list := th.Threads().List()
				for i := range list {
					thr := &list[i]
					t.Logf("  %s (depth=%d sub_threads=%d tools=%v)", thr.ID, thr.Depth, thr.SubThreads, thr.Tools)
					if strings.Contains(thr.ID, "support_director_a") {
						aDir = thr
					}
					if strings.Contains(thr.ID, "support_director_b") {
						bDir = thr
					}
				}
				if aDir == nil || bDir == nil {
					t.Errorf("expected both support_director_a and support_director_b, got a=%v b=%v", aDir, bDir)
					return
				}
				if bDir.SubThreads < 1 {
					t.Errorf("expected mirror director to have at least 1 worker, got %d", bDir.SubThreads)
				}
				t.Logf("PASS: team cloned by conversation — A has %d workers, B has %d workers", aDir.SubThreads, bDir.SubThreads)
			},
			Timeout: 180 * time.Second,
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_CloneTeam(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	helpdeskBin := BuildMCPBinary(t, "mcps/helpdesk")
	t.Logf("built helpdesk=%s", helpdeskBin)

	s := cloneTeamScenario
	s.MCPServers[0].Command = helpdeskBin
	RunScenario(t, s)
}
