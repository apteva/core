package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var rubricLearningScenario = Scenario{
	Name: "RubricLearning",
	Directive: `You coordinate sales-call QA across multiple call centers. Each
center has its OWN rubric, dimension set, training pool, and test
pool. Your job is to dispatch — you do NOT grade calls yourself.

The sales_qa MCP server is shared across centers. Every tool except
list_centers takes a center id.

YOUR PROCESS:

1. Call list_centers to see what's available. Note each center's id
   and dimension set — they differ deliberately.

2. For EACH center returned, spawn ONE sub-thread:
   [[spawn id="qa-<center_id>" mcp="sales_qa" tools="remember,evolve" directive="<directive below, with <center_id> filled in>"]]

   Sub-thread directive template (substitute <center_id> in 3 places):

   You are a QA analyst calibrating to call center "<center_id>". Use
   the sales_qa MCP, always passing center="<center_id>".

   PHASE A — STUDY:
     1. get_rubric(center="<center_id>") — learn the dimension criteria.
     2. list_training_calls(center="<center_id>"), then get_training_call
        for EVERY id. Read the transcript + ratings + notes. The notes
        are the most valuable signal — they tell you WHY each call
        scored what it did.
     3. Identify per-dimension patterns. What distinguishes a 1 from a 5?
     4. [[remember]] each heuristic — at least one per dimension.

   PHASE B — RATE TEST CALLS:
     5. list_test_calls(center="<center_id>"). For each test call:
        a. get_test_call(center="<center_id>", id=...).
        b. Quote concrete evidence for each rating dimension.
        c. submit_rating(center="<center_id>", call_id=..., ratings={...})
           where ratings is an OBJECT with one integer per dimension
           in this center's dimension set.
        d. Read server feedback. If <3 dims matched, refine heuristics
           with [[remember]] before grading the next call.
     6. After every test call has a submission, pace down. You're done.

3. After spawning all centers, pace down. Wait for the sub-threads to
   finish. Do NOT call the rubric/training/test/submit tools yourself.

CRITICAL for sub-threads: pass center="<center_id>" on EVERY tool call.
The MCP rejects calls missing the center arg. The dimension set varies
per center — never copy heuristics or dimension names across centers.`,
	MCPServers: []MCPServerConfig{
		{Name: "sales_qa", Command: "", Env: map[string]string{"SALES_QA_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Nothing to seed — the MCP bakes its own data. We just create
		// the dir so submissions.jsonl has somewhere to land.
	},
	Phases: []Phase{
		{
			Name:    "Dispatch sub-threads per center, study + rate all test calls",
			Timeout: 20 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Done when every center has at least one submission per
				// of its test calls. Expected counts are baked into the
				// MCP binary; rather than re-import them, we hard-code
				// here. Keeping this in sync with the MCP is the test
				// author's job — drifting it is loud (timeout vs. silent
				// pass) so it's easy to catch.
				expected := map[string]int{
					"saas_demo":        3,
					"telesales":        2,
					"support_recovery": 2,
				}
				data, err := os.ReadFile(filepath.Join(dir, "submissions.jsonl"))
				if err != nil {
					return false
				}
				seen := make(map[string]map[string]bool) // center → call_id → seen
				for _, ln := range strings.Split(strings.TrimSpace(string(data)), "\n") {
					if strings.TrimSpace(ln) == "" {
						continue
					}
					var row struct {
						Center string `json:"center"`
						CallID string `json:"call_id"`
					}
					if err := json.Unmarshal([]byte(ln), &row); err != nil {
						continue
					}
					if seen[row.Center] == nil {
						seen[row.Center] = map[string]bool{}
					}
					seen[row.Center][row.CallID] = true
				}
				for center, want := range expected {
					if len(seen[center]) < want {
						return false
					}
				}
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, err := os.ReadFile(filepath.Join(dir, "submissions.jsonl"))
				if err != nil {
					t.Fatalf("submissions.jsonl missing: %v", err)
				}
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				t.Logf("total submissions: %d (across all centers)", len(lines))

				type submission struct {
					Center      string         `json:"center"`
					CallID      string         `json:"call_id"`
					Submitted   map[string]int `json:"submitted"`
					GroundTruth map[string]int `json:"ground_truth"`
					Deltas      map[string]int `json:"deltas"`
				}
				// Latest submission per (center, call_id) — agents may
				// resubmit after seeing feedback; only the last attempt
				// counts.
				latest := make(map[string]submission)
				for _, ln := range lines {
					if strings.TrimSpace(ln) == "" {
						continue
					}
					var s submission
					if err := json.Unmarshal([]byte(ln), &s); err != nil {
						continue
					}
					latest[s.Center+"|"+s.CallID] = s
				}

				// Group by center for per-center reporting + thresholds.
				expectedTests := map[string]int{
					"saas_demo":        3,
					"telesales":        2,
					"support_recovery": 2,
				}
				perCenter := make(map[string][]submission)
				for _, s := range latest {
					perCenter[s.Center] = append(perCenter[s.Center], s)
				}

				for center, want := range expectedTests {
					subs := perCenter[center]
					t.Logf("center=%s: %d/%d submissions", center, len(subs), want)
					if len(subs) < want {
						gotIDs := make([]string, 0, len(subs))
						for _, s := range subs {
							gotIDs = append(gotIDs, s.CallID)
						}
						t.Errorf("center=%s: missing submissions, got %v want %d", center, gotIDs, want)
					}

					totalMatches := 0
					totalDims := 0
					for _, s := range subs {
						matches := 0
						dims := 0
						for _, delta := range s.Deltas {
							dims++
							if delta <= 1 {
								matches++
							}
						}
						totalDims += dims
						totalMatches += matches
						t.Logf("  %s/%s: %d/%d within ±1 — submitted=%v truth=%v deltas=%v",
							center, s.CallID, matches, dims, s.Submitted, s.GroundTruth, s.Deltas)
						// Per-call floor: at least 3/N dims within ±1 (more
						// forgiving than the old 4/5 because dim counts
						// vary per center; 3/5 = 60%, applied as ratio).
						if dims > 0 && matches*5 < dims*3 {
							t.Errorf("%s/%s: only %d/%d dims matched (need ≥60%%)", center, s.CallID, matches, dims)
						}
					}
					if totalDims > 0 {
						pct := float64(totalMatches) / float64(totalDims) * 100
						t.Logf("center=%s overall: %d/%d dims matched (%.1f%%)",
							center, totalMatches, totalDims, pct)
						// Center-level floor: 60% of dims across all test calls.
						if totalMatches*5 < totalDims*3 {
							t.Errorf("center=%s overall accuracy too low: %d/%d (need ≥60%%)",
								center, totalMatches, totalDims)
						}
					}
				}

				t.Logf("bonus: memory=%d entries on main, directive=%d chars",
					th.Memory().Count(), len(th.Config().GetDirective()))
			},
		},
	},
	Timeout: 30 * time.Minute,
	// 1 main + 1 sub per center. Sub-threads run in parallel.
}

func TestScenario_RubricLearning(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	bin := BuildMCPBinary(t, "mcps/sales_qa")
	t.Logf("built sales_qa=%s", bin)

	s := rubricLearningScenario
	s.MCPServers[0].Command = bin
	RunScenario(t, s)
}
