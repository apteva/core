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

var learningAgentScenario = Scenario{
	Name: "LearningAgent",
	Directive: `You manage a warehouse. You do NOT know the business rules — discover them by trying actions and learning from failures.

CRITICAL RULES FOR LEARNING:
1. When ANY action fails, you MUST call [[remember text="..."]] with the rule you learned. This is mandatory.
2. After learning 2+ rules, call [[evolve directive="..."]] to update your directive with all learned rules.
3. Your memory persists across sessions. Your conversation does NOT. Only remembered facts survive.

Process orders and shipments as requested via console events. When something fails, learn why, remember it, and retry correctly.`,
	MCPServers: []MCPServerConfig{
		{Name: "warehouse", Command: "", Env: map[string]string{"WAREHOUSE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "stock.json", map[string]int{
			"widgets":   500,
			"gadgets":   200,
			"chemicals": 300,
			"batteries": 150,
		})
		WriteJSONFile(t, dir, "orders.json", []any{})
		WriteJSONFile(t, dir, "shipments.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Phase 1: Order fails — learns and remembers max qty rule",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Order 200 widgets immediately.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(data, &orders)
					hasFailed := false
					hasSuccess := false
					for _, o := range orders {
						if o["status"] == "failed" {
							hasFailed = true
						}
						if o["status"] == "fulfilled" {
							hasSuccess = true
						}
					}
					return hasFailed && hasSuccess && th.Memory().Count() > 0
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 1: %d entries", th.Memory().Count())
				if th.Memory().Count() == 0 {
					t.Error("agent did not use [[remember]] after learning qty rule")
				}
			},
		},
		{
			Name:    "Phase 2: Ship to Japan + remember customs rule",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Ship the fulfilled widgets order to Japan, weight 20kg. Remember any rules you discover.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
					var shipments []map[string]any
					json.Unmarshal(data, &shipments)
					for _, s := range shipments {
						if s["status"] == "shipped" {
							return true
						}
					}
					return false
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 2: %d entries", th.Memory().Count())
			},
		},
		{
			Name:    "Phase 2b: Force hazardous rule discovery",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Order 50 chemicals. Remember any new rules you discover about ordering.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(data, &orders)
					for _, o := range orders {
						if o["item"] == "chemicals" && o["status"] == "fulfilled" {
							return true
						}
					}
					return false
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 2b: %d entries", th.Memory().Count())
				// Should have at least 2 memories now (qty + hazardous or customs)
				if th.Memory().Count() < 2 {
					t.Logf("NOTE: expected 2+ memories, got %d", th.Memory().Count())
				}
			},
		},
		{
			Name:    "Phase 3: Context reset — apply knowledge from memory only",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Reset order/shipment files for clean phase 3
				WriteJSONFile(t, dir, "orders.json", []any{})
				WriteJSONFile(t, dir, "shipments.json", []any{})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						// Clear conversation history — agent must rely on memory
						th.ResetConversation()
						t.Logf("conversation reset — %d memory entries available", th.Memory().Count())
						th.InjectConsole("Order 150 chemicals and ship them to Germany, weight 30kg. Apply everything you know about warehouse rules.")
						injected = true
					}
					orderData, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(orderData, &orders)
					chemFulfilled := 0
					for _, o := range orders {
						if o["item"] == "chemicals" && o["status"] == "fulfilled" {
							chemFulfilled++
						}
					}
					shipData, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
					var shipments []map[string]any
					json.Unmarshal(shipData, &shipments)
					germanyShipped := false
					for _, s := range shipments {
						dest, _ := s["destination"].(string)
						if strings.EqualFold(dest, "germany") && s["status"] == "shipped" {
							germanyShipped = true
						}
					}
					return chemFulfilled >= 2 && germanyShipped
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Count failures in phase 3 — fewer failures = better memory recall
				orderData, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				var orders []map[string]any
				json.Unmarshal(orderData, &orders)
				failures := 0
				successes := 0
				for _, o := range orders {
					if o["status"] == "failed" {
						failures++
					}
					if o["status"] == "fulfilled" {
						successes++
					}
				}
				t.Logf("phase 3 orders: %d fulfilled, %d failed (fewer failures = better memory)", successes, failures)

				shipData, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
				var shipments []map[string]any
				json.Unmarshal(shipData, &shipments)
				shipOK := 0
				shipFail := 0
				for _, s := range shipments {
					if s["status"] == "shipped" {
						shipOK++
					} else {
						shipFail++
					}
				}
				t.Logf("phase 3 shipments: %d shipped, %d failed", shipOK, shipFail)
			},
		},
		{
			Name:    "Phase 4: Final summary",
			Timeout: 10 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return true // always pass — just log results
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				directive := th.Config().GetDirective()
				evolved := len(directive) > 700
				t.Logf("directive evolved: %v (%d chars)", evolved, len(directive))
				t.Logf("final memory count: %d", th.Memory().Count())

				auditData, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(auditData)), "\n")
				t.Logf("total audit trail: %d entries", len(lines))

				if th.Memory().Count() < 2 {
					t.Error("expected at least 2 memory entries from learning")
				}
			},
		},
	},
	Timeout:    10 * time.Minute,
	MaxThreads: 3,
}

func TestScenario_LearningAgent(t *testing.T) {
	warehouseBin := BuildMCPBinary(t, "mcps/warehouse")
	t.Logf("built warehouse=%s", warehouseBin)

	s := learningAgentScenario
	s.MCPServers[0].Command = warehouseBin
	RunScenario(t, s)
}
