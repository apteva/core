package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var ecommerceScenario = Scenario{
	Name: "Ecommerce",
	Directive: `You manage order fulfillment for an online bakery.

Spawn and maintain 3 threads:
1. "warehouse" — checks inventory for pending orders, reserves stock, marks orders as ready.
   Tools: inventory_check_stock, inventory_use_stock, inventory_list_stock, orders_get_orders, orders_get_order, orders_update_order, send, done
2. "shipping" — picks up ready orders, marks them as shipped, stores tracking info.
   Tools: orders_get_orders, orders_update_order, storage_store, send, done
3. "comms" — sends customer notifications when orders ship.
   Tools: pushover_send_notification, storage_get, send, done

Workflow:
- When you receive a console event about new orders, tell warehouse to process them.
- Warehouse checks stock, reserves ingredients, marks order as "ready".
- If out of stock, warehouse reports to you and you notify comms.
- When warehouse finishes, tell shipping to dispatch.
- When shipping finishes, tell comms to notify the customer.`,
	MCPServers: []MCPServerConfig{
		{Name: "orders", Command: "", Env: map[string]string{"ORDERS_DATA_DIR": "{{dataDir}}"}},
		{Name: "inventory", Command: "", Env: map[string]string{"INVENTORY_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "stock.json", map[string]int{
			"chocolate cake": 10, "croissant": 50, "baguette": 30, "muffin": 25, "chocolate truffle": 8,
		})
		WriteJSONFile(t, dir, "orders.json", []map[string]any{})
	},
	Phases: []Phase{
		// No "startup — N threads spawned" phase: the phases below
		// assert on orders getting fulfilled, not on the agent's
		// internal thread layout.
		{
			Name:    "Order fulfillment — process and ship",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				WriteJSONFile(t, dir, "orders.json", []map[string]string{
					{"id": "ORD-001", "item": "chocolate cake", "qty": "2", "status": "pending"},
					{"id": "ORD-002", "item": "croissant", "qty": "12", "status": "pending"},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New orders received: ORD-001 (chocolate cake x2), ORD-002 (croissant x12). Please process them.")
						injected = true
					}
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					// Check if at least one order was updated beyond pending
					return strings.Contains(string(data), "ready") || strings.Contains(string(data), "shipped")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				s := string(data)
				if !strings.Contains(s, "ready") && !strings.Contains(s, "shipped") {
					t.Error("expected at least one order to be ready or shipped")
				}
			},
		},
		{
			Name:    "Out of stock — chocolate depleted",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				WriteJSONFile(t, dir, "stock.json", map[string]int{
					"chocolate cake": 10, "croissant": 50, "baguette": 30, "muffin": 25, "chocolate truffle": 0,
				})
				WriteJSONFile(t, dir, "orders.json", []map[string]string{
					{"id": "ORD-003", "item": "chocolate truffle", "qty": "5", "status": "pending"},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New order: ORD-003 (chocolate truffle x5). Please process. If out of stock, mark the order as cancelled.")
						injected = true
					}
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "cancelled")
				}
			}(),
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_Ecommerce(t *testing.T) {
	ordersBin := BuildMCPBinary(t, "mcps/orders")
	inventoryBin := BuildMCPBinary(t, "mcps/inventory")
	storageBin := BuildMCPBinary(t, "mcps/storage")
	pushoverBin := BuildMCPBinary(t, "mcps/pushover")
	t.Logf("built orders=%s inventory=%s storage=%s pushover=%s", ordersBin, inventoryBin, storageBin, pushoverBin)

	s := ecommerceScenario
	s.MCPServers[0].Command = ordersBin
	s.MCPServers[1].Command = inventoryBin
	s.MCPServers[2].Command = storageBin
	s.MCPServers[3].Command = pushoverBin
	RunScenario(t, s)
}
