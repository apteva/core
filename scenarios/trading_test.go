package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var tradingScenario = Scenario{
	Name: "Trading",
	Directive: `You manage a simple trading portfolio. Starting cash: $10,000.
Available symbols: AAPL, GOOGL, MSFT, TSLA.

Spawn and maintain 3 threads:
1. "data-feed" — reads prices periodically, stores history, reports significant moves to you.
   Tools: market_get_prices, market_get_history, storage_store, send, done
2. "analyst" — analyzes price data, identifies buy/sell signals based on price changes.
   Tools: market_get_history, market_get_prices, storage_get, storage_store, send, done
3. "executor" — places trades and manages stop-losses based on analyst signals.
   Tools: market_place_order, market_get_portfolio, market_set_stop_loss, market_get_orders, send, done

Workflow:
- Data-feed monitors prices and reports to you.
- You ask analyst to evaluate when significant moves occur.
- If analyst recommends a trade, you tell executor to place it with exact symbol, side, qty.
- Executor sets stop-losses on new positions (10% below buy price).

`,
	MCPServers: []MCPServerConfig{
		{Name: "market", Command: "", Env: map[string]string{"MARKET_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed initial prices
		WriteJSONFile(t, dir, "prices.json", map[string]float64{
			"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 420.10, "TSLA": 178.90,
		})
		// Seed price history (simulated recent data)
		var history []map[string]any
		now := time.Now()
		symbols := map[string]float64{"AAPL": 180.0, "GOOGL": 140.0, "MSFT": 415.0, "TSLA": 185.0}
		for i := 10; i >= 1; i-- {
			ts := now.Add(-time.Duration(i) * time.Minute).UTC().Format(time.RFC3339)
			for sym, base := range symbols {
				drift := (float64(10-i) / 10.0) * 5.0 // gradual increase
				history = append(history, map[string]any{
					"symbol": sym, "price": base + drift, "timestamp": ts,
				})
			}
		}
		WriteJSONFile(t, dir, "history.json", history)
		// Portfolio: $10k cash, no holdings
		WriteJSONFile(t, dir, "portfolio.json", map[string]any{
			"cash": 10000.0, "holdings": map[string]any{},
		})
	},
	Phases: []Phase{
		// No "startup — N threads spawned" phase: the phases below
		// assert on orders getting placed and stop-losses firing,
		// not on the agent's thread layout.
		{
			Name:    "Trading — buy signal and execution",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Market is open. AAPL shows strong upward trend from $180 to $185.50 over the last 10 periods. Buy 10 shares of AAPL now and set a stop-loss at $170.")
						injected = true
					}
					// Check if any order was placed
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "filled")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				if !strings.Contains(string(data), "buy") {
					t.Error("expected at least one buy order")
				}
			},
		},
		{
			Name:    "Stop-loss — price drop triggers sell",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Crash TSLA price to trigger stop-loss (if they bought it)
				// Or crash whatever they bought
				WriteJSONFile(t, dir, "prices.json", map[string]float64{
					"AAPL": 150.00, "GOOGL": 110.00, "MSFT": 380.00, "TSLA": 120.00,
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("MARKET CRASH: Prices just dropped hard. AAPL is now $150. Sell all AAPL positions immediately to limit losses.")
						injected = true
					}
					// Check if portfolio was updated (stop-loss triggered or manual sell)
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "sell")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				if !strings.Contains(string(data), "sell") {
					t.Error("expected at least one sell order after crash")
				}
			},
		},
	},
	Timeout: 8 * time.Minute,
}

func TestScenario_Trading(t *testing.T) {
	marketBin := BuildMCPBinary(t, "mcps/market")
	storageBin := BuildMCPBinary(t, "mcps/storage")
	t.Logf("built market=%s storage=%s", marketBin, storageBin)

	s := tradingScenario
	s.MCPServers[0].Command = marketBin
	s.MCPServers[1].Command = storageBin
	RunScenario(t, s)
}
