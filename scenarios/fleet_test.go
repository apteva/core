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

var fleetScenario = Scenario{
	Name: "Fleet",
	Directive: `You are the CEO of a small online electronics store. Your sales are declining and you need to turn things around.

You operate as a FLEET — you do NOT do the work yourself. Instead, you spawn TEAM LEADS who each manage their own workers. You coordinate at the strategic level only.

Spawn exactly 3 team leads:

1. "ops-lead" — Operations lead. Responsible for inventory and supply chain.
   Give them tools: store_get_inventory, store_check_supplier, store_restock_item, send
   Their job: investigate stock-outs, find working suppliers, restock critical items.
   They should spawn their own workers for parallel tasks (e.g. one to check inventory, one to handle restocking).

2. "sales-lead" — Sales & pricing lead. Responsible for revenue optimization.
   Give them tools: store_get_sales, store_get_competitors, store_adjust_price, store_get_analytics, send
   Their job: analyze sales trends, compare competitor prices, adjust pricing to be competitive.
   They should spawn workers to investigate different aspects in parallel.

3. "marketing-lead" — Marketing lead. Responsible for customer engagement and growth.
   Give them tools: store_get_customer_segments, store_get_traffic_sources, store_get_reviews, store_send_promotion, store_add_product, send
   Their job: identify customer segments to target, find new product opportunities, run promotions.
   They should spawn workers for research and execution.

CRITICAL RULES:
- You ONLY talk to your 3 leads. Never call store tools directly.
- Leads report findings and actions back to you.
- Wait for all 3 leads to report before making final strategic decisions.
- After receiving reports, send a strategic summary to each lead with any cross-team insights.`,
	MCPServers: []MCPServerConfig{
		{Name: "store", Command: "", Env: map[string]string{"STORE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedStoreData,
	Phases: []Phase{
		{
			Name:    "Startup — 3 team leads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				ids := ThreadIDs(th)
				t.Logf("  main threads: %v", ids)
				return len(ids) >= 3
			},
		},
		{
			Name:    "Tree forms — leads spawn workers (depth 1+)",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				all := AllThreadInfos(th.Threads())
				depth0 := 0
				depth1 := 0
				for _, info := range all {
					if info.Depth == 0 {
						depth0++
					} else if info.Depth >= 1 {
						depth1++
					}
				}
				t.Logf("  total threads: %d (leads: %d, workers: %d)", len(all), depth0, depth1)
				for _, info := range all {
					t.Logf("    %s (parent=%s, depth=%d)", info.ID, info.ParentID, info.Depth)
				}
				// Need at least 3 leads + 3 workers (at least 1 worker per lead)
				return depth0 >= 3 && depth1 >= 3
			},
		},
		{
			Name:    "Execution — workers take actions via store tools",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				if err != nil {
					return false
				}
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				// Count distinct action types
				actions := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						actions[a] = true
					}
				}
				t.Logf("  actions taken: %d calls, %d types: %v", len(lines), len(actions), actions)
				// Need at least 3 different action types (investigation + execution)
				return len(actions) >= 3
			},
		},
		{
			Name:    "Coordination — leads report back, real actions taken",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				s := string(data)
				// Need at least one write action (restock, price change, or promotion)
				hasRestock := strings.Contains(s, "\"action\":\"restock_item\"")
				hasPrice := strings.Contains(s, "\"action\":\"adjust_price\"")
				hasPromo := strings.Contains(s, "\"action\":\"send_promotion\"")
				hasNewProduct := strings.Contains(s, "\"action\":\"add_product\"")
				writeActions := 0
				if hasRestock {
					writeActions++
				}
				if hasPrice {
					writeActions++
				}
				if hasPromo {
					writeActions++
				}
				if hasNewProduct {
					writeActions++
				}
				t.Logf("  write actions: restock=%v price=%v promo=%v newProduct=%v (%d/4)",
					hasRestock, hasPrice, hasPromo, hasNewProduct, writeActions)
				return writeActions >= 2
			},
		},
		{
			Name:    "Upward reporting — leads report results to main (full chain)",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Nothing to set up — just need to wait for leads to report
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Status check: all leads please report what actions you've taken and results so far.")
						injected = true
					}
					// Check if main received messages from leads (main's iteration advanced beyond startup)
					return th.Iteration() >= 4
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Final report
				all := AllThreadInfos(th.Threads())
				t.Log("=== FLEET REPORT ===")
				t.Logf("total threads alive: %d", len(all))
				for _, info := range all {
					indent := ""
					for i := 0; i < info.Depth; i++ {
						indent += "  "
					}
					role := "worker"
					if info.Depth == 0 {
						role = "lead"
					}
					t.Logf("  %s%s [%s] (parent=%s, depth=%d)", indent, info.ID, role, info.ParentID, info.Depth)
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

				// Verify tree structure existed
				maxDepth := 0
				for _, info := range all {
					if info.Depth > maxDepth {
						maxDepth = info.Depth
					}
				}
				if maxDepth < 1 {
					t.Error("expected tree structure with depth >= 1 (leads spawning workers)")
				}

				// Check main received lead reports (check iteration count and message history)
				t.Logf("main iterations: %d", th.Iteration())
				// Log last few messages in main's context to see if leads reported
				for i, msg := range th.Messages() {
					if i == 0 {
						continue // skip system prompt
					}
					preview := msg.Content
					if len(preview) > 120 {
						preview = preview[:120] + "..."
					}
					if strings.Contains(msg.Content, "[from:") {
						t.Logf("  main msg[%d] role=%s: %s", i, msg.Role, preview)
					}
				}
			},
		},
	},
	Timeout: 10 * time.Minute,
}

func TestScenario_Fleet(t *testing.T) {
	storeBin := BuildMCPBinary(t, "mcps/store")
	t.Logf("built store=%s", storeBin)

	s := fleetScenario
	s.MCPServers[0].Command = storeBin
	RunScenario(t, s)
}
