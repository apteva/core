package scenarios

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var emergentScenario = Scenario{
	Name:      "Emergent",
	Directive: `You run a small online electronics store. Sales have been declining. Diagnose the root causes and take action to turn things around. Go deep — surface-level fixes won't be enough.`,
	MCPServers: []MCPServerConfig{
		{Name: "store", Command: "", Env: map[string]string{"STORE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedStoreData,
	Phases: []Phase{
		{
			Name:    "Deep investigation — agent explores multiple data layers",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				tools := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						tools[a] = true
					}
				}
				// Must dig into at least 5 different data sources (not just the obvious ones)
				return len(tools) >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				tools := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						tools[a] = true
					}
				}
				t.Logf("data sources explored: %v (%d)", tools, len(tools))
			},
		},
		{
			Name:    "Multi-layered action — fixes surface + deeper issues",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				s := string(data)
				// Must take at least 3 different ACTION types
				actionTypes := 0
				if strings.Contains(s, "\"action\":\"restock_item\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"adjust_price\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"send_promotion\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"add_product\"") {
					actionTypes++
				}
				return actionTypes >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				var actions []string
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					switch entry["action"] {
					case "restock_item":
						actions = append(actions, fmt.Sprintf("RESTOCK: %s ×%s (supplier: %s)", entry["product"], entry["quantity"], entry["supplier"]))
					case "adjust_price":
						actions = append(actions, fmt.Sprintf("PRICE: %s → $%s", entry["product"], entry["new_price"]))
					case "send_promotion":
						actions = append(actions, fmt.Sprintf("PROMO: \"%s\" (%s to %s)", entry["subject"], entry["discount"], entry["target_segment"]))
					case "add_product":
						actions = append(actions, fmt.Sprintf("NEW PRODUCT: %s at $%s", entry["name"], entry["price"]))
					}
				}
				for _, a := range actions {
					t.Logf("  %s", a)
				}
				// Check for smart decisions
				usedAltSupplier := strings.Contains(string(data), "alt_supplier")
				addedProduct := strings.Contains(string(data), "add_product")
				t.Logf("discovered alt supplier: %v", usedAltSupplier)
				t.Logf("added new product (blog opportunity): %v", addedProduct)
			},
		},
		{
			Name:    "Emergence score — threads, memory, strategy, creativity",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				score := 0
				if th.Threads() != nil && th.Threads().Count() > 0 {
					score += 2
				}
				if th.Memory().Count() > 0 {
					score++
				}
				directive := th.Config().GetDirective()
				if len(directive) > 200 {
					score++
				}
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				s := string(data)
				if strings.Contains(s, "alt_supplier") {
					score++ // discovered supply chain workaround
				}
				if strings.Contains(s, "add_product") {
					score++ // spotted market opportunity
				}
				actions := strings.Count(s, "\"action\":")
				if actions >= 6 {
					score++ // took comprehensive action
				}
				return score >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Log("=== EMERGENCE REPORT ===")
				t.Logf("threads active: %d", th.Threads().Count())
				t.Logf("memory entries: %d", th.Memory().Count())
				directive := th.Config().GetDirective()
				t.Logf("directive evolved: %v (%d chars)", len(directive) > 200, len(directive))

				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				s := string(data)

				score := 0
				if th.Threads().Count() > 0 {
					score += 2
					t.Log("  ✓ spawned worker threads (self-organization)")
				}
				if th.Memory().Count() > 0 {
					score++
					t.Log("  ✓ remembered findings (persistent learning)")
				}
				if len(directive) > 200 {
					score++
					t.Log("  ✓ evolved directive (self-improvement)")
				}
				if strings.Contains(s, "alt_supplier") {
					score++
					t.Log("  ✓ discovered alt supplier workaround (problem-solving)")
				}
				if strings.Contains(s, "add_product") {
					score++
					t.Log("  ✓ spotted new product opportunity (creativity)")
				}
				if len(lines) >= 8 {
					score++
					t.Log("  ✓ took comprehensive multi-step action (initiative)")
				}

				t.Logf("EMERGENCE SCORE: %d/7", score)
				t.Logf("total tool calls: %d", len(lines))

				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					t.Logf("  [%s] %s", entry["action"], entry)
				}
			},
		},
	},
	Timeout: 10 * time.Minute,
}

func TestScenario_Emergent(t *testing.T) {
	storeBin := BuildMCPBinary(t, "mcps/store")
	t.Logf("built store=%s", storeBin)

	s := emergentScenario
	s.MCPServers[0].Command = storeBin
	RunScenario(t, s)
}
