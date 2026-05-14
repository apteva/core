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

var conglomerateScenario = Scenario{
	Name: "Conglomerate",
	Directive: `You are the CEO of ByteVentures, a one-person conglomerate running 3 microbusinesses through AI agents. Your store's sales are declining badly and you need all 3 businesses working in parallel to fix it.

YOU MUST BUILD A 3-LEVEL HIERARCHY. You do NOT do any store work yourself. You spawn directors, directors spawn team leads, team leads spawn workers. Workers are the only ones who call store tools.

Spawn exactly 3 DIRECTORS (depth 0):

1. "retail-director" — runs the Electronics Retail business. Responsible for making sure products are in stock and competitively priced.
   Tools: send (NO store tools — directors delegate, they don't execute)
   Their job: spawn 2 team leads:
   - "supply-chain-lead" with tools: store_get_inventory, store_check_supplier, store_restock_item, send
     This lead should spawn workers to check stock and handle supplier logistics in parallel.
   - "pricing-lead" with tools: store_get_competitors, store_adjust_price, send
     This lead should spawn workers to scan competitor prices and execute adjustments.

2. "growth-director" — runs the Customer Growth business. Responsible for retention and acquisition.
   Tools: send (NO store tools)
   Their job: spawn 2 team leads:
   - "retention-lead" with tools: store_get_customer_segments, store_send_promotion, send
     This lead should spawn workers to analyze churn and send winback campaigns.
   - "acquisition-lead" with tools: store_get_traffic_sources, store_get_analytics, store_send_promotion, send
     This lead should spawn workers to find traffic opportunities and launch campaigns.

3. "expansion-director" — runs the New Markets business. Responsible for finding and launching new products.
   Tools: send (NO store tools)
   Their job: spawn 2 team leads:
   - "research-lead" with tools: store_get_reviews, store_get_traffic_sources, store_get_analytics, send
     This lead should spawn workers to mine reviews and spot trends.
   - "launch-lead" with tools: store_add_product, store_restock_item, send
     This lead should spawn workers to list new products when opportunities are found.

WORKFLOW:
- Directors spawn their team leads immediately.
- Team leads spawn 1-2 workers each to parallelize their tasks.
- Workers call store tools, report findings to their team lead.
- Team leads synthesize worker findings, take action, report to their director.
- Directors report business summaries to you (the CEO).
- You synthesize cross-business insights and send strategic directives back down.

CRITICAL: The value of this structure is PARALLELISM and SPECIALIZATION. All 3 businesses investigate simultaneously. Information flows UP (workers→leads→directors→CEO), decisions flow DOWN (CEO→directors→leads→workers).`,
	MCPServers: []MCPServerConfig{
		{Name: "store", Command: "", Env: map[string]string{"STORE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedStoreData,
	Phases: []Phase{
		{
			Name:    "Phase 1 — Directors spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				ids := ThreadIDs(th)
				t.Logf("  main threads: %v", ids)
				return len(ids) >= 3
			},
		},
		{
			Name:    "Phase 2 — Team leads spawned (depth 1)",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				all := AllThreadInfos(th.Threads())
				byDepth := map[int]int{}
				for _, info := range all {
					byDepth[info.Depth]++
				}
				t.Logf("  threads by depth: d0=%d d1=%d d2=%d (total %d)",
					byDepth[0], byDepth[1], byDepth[2], len(all))
				// Need at least 3 directors + 4 team leads
				return byDepth[0] >= 3 && byDepth[1] >= 4
			},
		},
		{
			Name:    "Phase 3 — Workers spawned (depth 2) — full 3-level tree",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				all := AllThreadInfos(th.Threads())
				byDepth := map[int]int{}
				for _, info := range all {
					byDepth[info.Depth]++
				}
				t.Logf("  threads by depth: d0=%d d1=%d d2=%d (total %d)",
					byDepth[0], byDepth[1], byDepth[2], len(all))
				for _, info := range all {
					indent := strings.Repeat("  ", info.Depth)
					t.Logf("    %s%s (parent=%s, depth=%d)", indent, info.ID, info.ParentID, info.Depth)
				}
				// Need depth-2 workers to exist (at least 3)
				return byDepth[2] >= 3
			},
		},
		{
			Name:    "Phase 4 — Workers execute store tools",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				if err != nil {
					return false
				}
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				actions := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						actions[a] = true
					}
				}
				t.Logf("  store actions: %d calls, %d types: %v", len(lines), len(actions), actions)
				// Need at least 4 different data sources explored
				return len(actions) >= 4
			},
		},
		{
			Name:    "Phase 5 — Full chain: actions taken + directors report to CEO",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Nudge the CEO to demand reports
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Board meeting in 5 minutes. All directors: submit your business unit reports NOW with findings, actions taken, and recommendations.")
						injected = true
					}
					// Check for write actions (restock, price adjust, promo, or new product)
					data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
					s := string(data)
					writeActions := 0
					if strings.Contains(s, "\"action\":\"restock_item\"") {
						writeActions++
					}
					if strings.Contains(s, "\"action\":\"adjust_price\"") {
						writeActions++
					}
					if strings.Contains(s, "\"action\":\"send_promotion\"") {
						writeActions++
					}
					if strings.Contains(s, "\"action\":\"add_product\"") {
						writeActions++
					}
					// Also check CEO received at least 2 director reports
					ceoReports := 0
					for _, msg := range th.Messages() {
						if strings.Contains(msg.Content, "[from:retail-director]") ||
							strings.Contains(msg.Content, "[from:growth-director]") ||
							strings.Contains(msg.Content, "[from:expansion-director]") {
							ceoReports++
						}
					}
					t.Logf("  write actions: %d/4, CEO reports received: %d/3", writeActions, ceoReports)
					return writeActions >= 2 && ceoReports >= 2
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				all := AllThreadInfos(th.Threads())
				t.Log("=== CONGLOMERATE REPORT ===")
				t.Logf("total threads alive: %d", len(all))

				// Print tree
				byDepth := map[int]int{}
				for _, info := range all {
					byDepth[info.Depth]++
				}
				t.Logf("by depth: directors=%d, leads=%d, workers=%d", byDepth[0], byDepth[1], byDepth[2])

				for _, info := range all {
					indent := strings.Repeat("  ", info.Depth)
					role := "worker"
					if info.Depth == 0 {
						role = "director"
					} else if info.Depth == 1 {
						role = "lead"
					}
					t.Logf("  %s%s [%s] (parent=%s)", indent, info.ID, role, info.ParentID)
				}

				// Action summary
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				t.Logf("total store actions: %d", len(lines))
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					t.Logf("  [%s] %v", entry["action"], entry)
				}

				// Verify 3-level tree
				if byDepth[2] == 0 {
					t.Error("expected depth-2 workers (3-level tree)")
				}

				// Log CEO messages from directors
				t.Log("--- CEO inbox (director reports) ---")
				for _, msg := range th.Messages() {
					if strings.Contains(msg.Content, "[from:retail-director]") ||
						strings.Contains(msg.Content, "[from:growth-director]") ||
						strings.Contains(msg.Content, "[from:expansion-director]") {
						preview := msg.Content
						if len(preview) > 200 {
							preview = preview[:200] + "..."
						}
						t.Logf("  %s", preview)
					}
				}
			},
		},
	},
	Timeout: 12 * time.Minute,
}

func TestScenario_Conglomerate(t *testing.T) {
	storeBin := BuildMCPBinary(t, "mcps/store")
	t.Logf("built store=%s", storeBin)

	s := conglomerateScenario
	s.MCPServers[0].Command = storeBin
	RunScenario(t, s)
}
