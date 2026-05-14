package core

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestIntegration_SpawnWithMCP_PerModel exercises the real-LLM
// spawn-with-MCP path across multiple Fireworks models so we can catch
// "agent gets stuck trying to spawn" regressions in specific models
// (e.g. minimax-m2p7 has historically hung here).
//
// Each sub-test:
//   - Stands up a tiny httptest MCP server exposing one tool (`echo`).
//   - Creates a Thinker pinned to the target Fireworks model via
//     Providers config (so the pool picks that model for all tiers).
//   - Injects a crisp directive asking main to spawn a worker that
//     calls `mock_echo` once, then `done`.
//   - Subscribes to the bus and asserts, within a timeout, that:
//       1. main invoked a `spawn` (at least the sub-thread appeared).
//       2. the sub-thread actually called `mock_echo` — i.e. the MCP
//          wiring worked end-to-end with this model.
//
// Cost: ~1-3 iterations per model on a healthy model, so ~$0.003–0.01
// each. Models that hang get caught by the per-model timeout and fail
// loudly with the metric log. Gate with FIREWORKS_API_KEY +
// RUN_SPAWN_MCP_TEST=1 so it stays out of the default CI matrix.
func TestIntegration_SpawnWithMCP_PerModel(t *testing.T) {
	apiKey := getAPIKey(t)
	if os.Getenv("RUN_SPAWN_MCP_TEST") == "" {
		t.Skip("set RUN_SPAWN_MCP_TEST=1 to run real-LLM spawn/MCP model comparison")
	}

	// Default model list — the two Fireworks models we're comparing.
	// Override via SPAWN_MCP_MODELS="a,b,c" to add more.
	models := []string{
		"accounts/fireworks/models/kimi-k2p5",
		"accounts/fireworks/models/minimax-m2p7",
	}
	if override := os.Getenv("SPAWN_MCP_MODELS"); override != "" {
		models = nil
		for _, m := range strings.Split(override, ",") {
			if m = strings.TrimSpace(m); m != "" {
				models = append(models, m)
			}
		}
	}

	for _, model := range models {
		model := model
		name := model
		if idx := strings.LastIndex(name, "/"); idx >= 0 {
			name = name[idx+1:]
		}
		t.Run(name, func(t *testing.T) {
			runSpawnMCPModelTest(t, apiKey, model)
		})
	}
}

func runSpawnMCPModelTest(t *testing.T, apiKey, model string) {
	t.Helper()

	var echoCalls atomic.Int64
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if req.ID == nil {
			w.WriteHeader(http.StatusAccepted)
			return
		}
		var result any
		switch req.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2025-03-26",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]string{"name": "mock", "version": "1.0"},
			}
		case "tools/list":
			result = map[string]any{"tools": []map[string]any{{
				"name":        "echo",
				"description": "echoes the input back",
				"inputSchema": map[string]any{
					"type":       "object",
					"properties": map[string]any{"text": map[string]any{"type": "string"}},
					"required":   []string{"text"},
				},
			}}}
		case "tools/call":
			echoCalls.Add(1)
			var p struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments"`
			}
			json.Unmarshal(req.Params, &p)
			text := "pong"
			if v, ok := p.Arguments["text"].(string); ok {
				text = v
			}
			result = map[string]any{
				"content": []map[string]any{{"type": "text", "text": text}},
			}
		default:
			result = map[string]any{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0", "id": req.ID, "result": result,
		})
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	directive := `You coordinate a single worker. On your first turn, spawn ONE
sub-thread:

[[spawn id="w" mcps="mock" tools="mock_echo,done" directive="Call mock_echo once with text='ok'. Then call done. Nothing else."]]

After spawning, pace down and wait. Do not call mock_echo yourself.`

	mcpCfg := []MCPServerConfig{{
		Name:      "mock",
		Transport: "http",
		URL:       srv.URL + "/mcp",
	}}

	providers := []ProviderConfig{{
		Name:    "fireworks",
		Default: true,
		Models: map[string]string{
			"large":  model,
			"medium": model,
			"small":  model,
		},
	}}

	thinker := newScenarioThinker(t, apiKey, directive, mcpCfg, providers)
	defer thinker.Stop()

	// Observer: track tokens, cost, and whether the worker actually
	// called mock_echo. A "stuck" model shows up as the 90s timeout
	// expiring with echoFromWorker still false.
	var (
		totalPrompt, totalCached, totalCompletion atomic.Int64
		iterCount                                 atomic.Int64
		spawnSeen                                 atomic.Bool
		echoFromWorker                            atomic.Bool
		firstToolAt                               atomic.Int64
	)
	start := time.Now()
	obs := thinker.bus.SubscribeAll("model-test-observer", 500)
	var wg sync.WaitGroup
	wg.Add(1)
	stopObs := make(chan struct{})
	go func() {
		defer wg.Done()
		for {
			select {
			case ev := <-obs.C:
				if ev.Type != EventThinkDone {
					continue
				}
				iterCount.Add(1)
				totalPrompt.Add(int64(ev.Usage.PromptTokens))
				totalCached.Add(int64(ev.Usage.CachedTokens))
				totalCompletion.Add(int64(ev.Usage.CompletionTokens))
				for _, tc := range ev.ToolCalls {
					if tc == "spawn" {
						spawnSeen.Store(true)
					}
					if ev.From == "w" && tc == "mock_echo" {
						if firstToolAt.Load() == 0 {
							firstToolAt.Store(int64(time.Since(start).Milliseconds()))
						}
						echoFromWorker.Store(true)
					}
				}
			case <-stopObs:
				return
			}
		}
	}()

	go thinker.Run()

	// A single nudge is enough — main's directive already instructs it
	// to spawn on first turn. We inject "Begin." so the first iteration
	// has a user message to react to.
	thinker.InjectConsole("Begin.")

	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		if echoFromWorker.Load() && echoCalls.Load() > 0 {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}

	close(stopObs)
	wg.Wait()

	duration := time.Since(start)
	cost := calculateCostForProvider(thinker.provider, TokenUsage{
		PromptTokens:     int(totalPrompt.Load()),
		CachedTokens:     int(totalCached.Load()),
		CompletionTokens: int(totalCompletion.Load()),
	})

	t.Logf("────────────────────────────────────────")
	t.Logf("model: %s", model)
	t.Logf("duration: %s | iterations: %d | spawn seen: %v | worker called mock_echo: %v",
		duration.Round(time.Second), iterCount.Load(), spawnSeen.Load(), echoFromWorker.Load())
	if firstToolAt.Load() > 0 {
		t.Logf("time to first worker tool call: %dms", firstToolAt.Load())
	}
	t.Logf("tokens: %d in / %d cached / %d out ($%.4f)",
		totalPrompt.Load(), totalCached.Load(), totalCompletion.Load(), cost)
	t.Logf("mock echo endpoint hits: %d", echoCalls.Load())
	t.Logf("────────────────────────────────────────")

	if !spawnSeen.Load() {
		t.Errorf("model %q never emitted a spawn tool call within 90s", model)
	}
	if !echoFromWorker.Load() {
		t.Errorf("model %q: worker thread never called mock_echo", model)
	}
	if echoCalls.Load() == 0 {
		t.Errorf("model %q: mock MCP server never received a tools/call", model)
	}
}
