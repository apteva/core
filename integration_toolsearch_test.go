package core

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/joho/godotenv"
)

// Real-LLM coverage for the agent-driven tool discovery path
// (apteva/core MCP refactor): search_tools discovery and per-turn
// BM25 preload. Uses OpenCode Go (flat-rate subscription) for the
// model calls so a full run costs effectively nothing.
//
// Gated behind RUN_TOOLSEARCH_TEST=1 + OPENCODE_GO_API_KEY so it
// stays out of the default unit run. `go test -short` also skips it.

// toolSearchGate is the common skip/key logic for this file.
func toolSearchGate(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping LLM integration test in short mode")
	}
	if os.Getenv("RUN_TOOLSEARCH_TEST") == "" {
		t.Skip("set RUN_TOOLSEARCH_TEST=1 to run the real-LLM tool-search tests")
	}
	godotenv.Load()
	key := os.Getenv("OPENCODE_GO_API_KEY")
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set (needed for opencode-go provider)")
	}
	return key
}

// mockToolMCP stands up a one-tool Streamable-HTTP MCP server. The
// tool just records that it was called and echoes back a marker
// string so the test can confirm the round-trip. Returns the server
// (caller defers Close) and a counter for tools/call hits.
func mockToolMCP(t *testing.T, toolName, toolDesc string) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var calls atomic.Int64
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
		if req.ID == nil { // notification
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
				"name":        toolName,
				"description": toolDesc,
				"inputSchema": map[string]any{
					"type":       "object",
					"properties": map[string]any{"input": map[string]any{"type": "string"}},
				},
			}}}
		case "tools/call":
			calls.Add(1)
			result = map[string]any{
				"content": []map[string]any{{"type": "text", "text": "MOCK_TOOL_OK"}},
			}
		default:
			result = map[string]any{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
	})
	srv := httptest.NewServer(mux)
	return srv, &calls
}

// mockMultiToolMCP is the multi-tool variant used by capability-scope tests.
// Counters are keyed by the local MCP tool name (before Core prefixes it with
// the server name).
func mockMultiToolMCP(t *testing.T, tools []mcpToolDef) (*httptest.Server, map[string]*atomic.Int64) {
	t.Helper()
	calls := make(map[string]*atomic.Int64, len(tools))
	for _, tool := range tools {
		calls[tool.Name] = &atomic.Int64{}
	}
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
				"serverInfo":      map[string]string{"name": "mock-multi", "version": "1.0"},
			}
		case "tools/list":
			listed := make([]map[string]any, 0, len(tools))
			for _, tool := range tools {
				listed = append(listed, map[string]any{
					"name": tool.Name, "description": tool.Description, "inputSchema": tool.InputSchema,
				})
			}
			result = map[string]any{"tools": listed}
		case "tools/call":
			var params struct {
				Name string `json:"name"`
			}
			_ = json.Unmarshal(req.Params, &params)
			if counter := calls[params.Name]; counter != nil {
				counter.Add(1)
			}
			result = map[string]any{
				"content": []map[string]any{{"type": "text", "text": "MOCK_" + params.Name + "_OK"}},
			}
		default:
			result = map[string]any{}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
	})
	return httptest.NewServer(mux), calls
}

// toolCallObserver watches the bus for EventThinkDone and records,
// per thread, which tools were called. Caller stops it via the
// returned func (which also waits for the goroutine to drain).
type toolCallObserver struct {
	mu     sync.Mutex
	byTool map[string]int // tool name → call count across all threads
}

func newToolCallObserver(t *testing.T, thinker *Thinker) (*toolCallObserver, func()) {
	t.Helper()
	obs := &toolCallObserver{byTool: map[string]int{}}
	sub := thinker.bus.SubscribeAll("toolsearch-observer", 500)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case ev := <-sub.C:
				if ev.Type != EventThinkDone {
					continue
				}
				obs.mu.Lock()
				for _, tc := range ev.ToolCalls {
					obs.byTool[tc]++
				}
				obs.mu.Unlock()
				// Turn-by-turn trace — invaluable when a test fails and
				// you need to see whether the model searched, guessed,
				// or had the tool preloaded.
				t.Logf("[iter %d from=%s] tool calls: %v", ev.Iteration, ev.From, ev.ToolCalls)
			case <-stop:
				return
			}
		}
	}()
	return obs, func() { close(stop); wg.Wait() }
}

func (o *toolCallObserver) count(tool string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.byTool[tool]
}

func (o *toolCallObserver) snapshot() map[string]int {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make(map[string]int, len(o.byTool))
	for k, v := range o.byTool {
		out[k] = v
	}
	return out
}

// waitFor polls cond until true or the deadline expires.
func waitFor(d time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(250 * time.Millisecond)
	}
	return cond()
}

// TestIntegration_ToolSearch_DiscoverAndCall verifies the cold-discovery
// path: an MCP tool the agent has never seen is NOT in its tool list,
// and the ONLY way to reach it is search_tools.
//
// The tool name is deliberately opaque ("op_4471") and semantically
// unrelated to the task. This matters: the registry dispatches any
// tool by name even when its schema wasn't in the LLM's tool list, so
// a *guessable* name lets the model skip search entirely (which is
// exactly what an earlier draft of this test discovered). With an
// unguessable name, a tools/call landing on the mock server is itself
// proof that search_tools ran, surfaced the real name, and activated
// the schema — there is no other path to it.
func TestIntegration_ToolSearch_DiscoverAndCall(t *testing.T) {
	toolSearchGate(t)

	srv, calls := mockToolMCP(t, "op_4471",
		"Submit a completed compliance form to the records system. "+
			"Required step before any audit can close.")
	defer srv.Close()

	directive := `You are the main thread. Your job: submit a completed compliance
form to the records system.

You do NOT have a tool for this in your current tool list, and you do NOT
know its name — it is not guessable. To get it:
  1. Call search_tools with a query describing the capability you need
     (something like "submit compliance form records").
  2. The matching tool's schema loads on your NEXT turn — read the name
     it returns.
  3. Call that exact tool name with a reasonable input value.

Do this YOURSELF on the main thread. Do NOT spawn a sub-thread. Do NOT
guess tool names. After the tool returns, call pace to sleep.`

	mcpCfg := []MCPServerConfig{{Name: "catalog", Transport: "http", URL: srv.URL + "/mcp"}}
	providers := []ProviderConfig{{Name: "opencode-go", Default: true}}

	thinker := newScenarioThinker(t, os.Getenv("FIREWORKS_API_KEY"), directive, mcpCfg, providers)
	defer thinker.Stop()
	// This case isolates explicit search_tools. Keep the directive in the
	// already-built system prompt so the model sees the task, but remove it
	// from BM25's preload query; otherwise the intentional preload path can
	// surface the exact tool before the model has a chance to search for it.
	thinker.directive = ""

	// Sanity: the mock tool is indexed but hidden from main's tool list
	// until search_tools activates it.
	if _, ok := thinker.toolIndex.Get("catalog_op_4471"); !ok {
		t.Fatal("setup: catalog_op_4471 not in the tool index")
	}
	for _, nt := range thinker.registry.NativeTools(nil, thinker.activeTools) {
		if nt.Name == "catalog_op_4471" {
			t.Fatal("setup: catalog_op_4471 should be hidden from main before any search")
		}
	}

	obs, stopObs := newToolCallObserver(t, thinker)
	defer stopObs()

	go thinker.Run()
	thinker.InjectConsole("Begin.")

	ok := waitFor(150*time.Second, func() bool {
		return calls.Load() > 0
	})

	t.Logf("tool calls observed: %v", obs.snapshot())
	t.Logf("mock server tools/call hits: %d", calls.Load())

	// The load-bearing assertion: the mock server was hit. Since the
	// tool name is unguessable, that can only happen via search_tools.
	if calls.Load() == 0 {
		t.Error("mock MCP server never received a tools/call — the discovery path (search_tools → activate → call) did not complete")
	}
	if obs.count("search_tools") == 0 {
		t.Error("agent never called search_tools — cold discovery path not exercised")
	}
	if obs.count("catalog_op_4471") == 0 {
		t.Error("agent never emitted a catalog_op_4471 call")
	}
	if !ok {
		t.Error("timed out waiting for the discovered tool to be called within 150s")
	}
}

// TestIntegration_ToolSearch_PreloadAvoidsSearch verifies the per-turn
// BM25 preload: when the user's message wording strongly matches a
// tool's name + description, that tool's schema is surfaced for the
// turn WITHOUT the agent having to call search_tools first. Success =
// the tool gets called and search_tools is never invoked.
func TestIntegration_ToolSearch_PreloadAvoidsSearch(t *testing.T) {
	toolSearchGate(t)

	// Tool name + description densely overlap the console message below
	// so the BM25 preload ranks it into the top-5 for that turn.
	srv, calls := mockToolMCP(t, "upload_invoice_pdf",
		"Upload an invoice PDF document to the billing folder.")
	defer srv.Close()

	directive := `You are the main thread. When the user asks you to do something,
the tool you need is ALREADY in your tool list — it was surfaced for you
based on the request. Call it directly with a reasonable input value.

Do NOT call search_tools — everything you need is already available.
Do NOT spawn a sub-thread. After the tool returns, call pace to sleep.`

	mcpCfg := []MCPServerConfig{{Name: "billing", Transport: "http", URL: srv.URL + "/mcp"}}
	providers := []ProviderConfig{{Name: "opencode-go", Default: true}}

	thinker := newScenarioThinker(t, os.Getenv("FIREWORKS_API_KEY"), directive, mcpCfg, providers)
	defer thinker.Stop()

	obs, stopObs := newToolCallObserver(t, thinker)
	defer stopObs()

	go thinker.Run()
	// The console wording deliberately echoes the tool's name + description
	// ("upload", "invoice", "pdf", "billing") so the preload BM25 search
	// ranks billing_upload_invoice_pdf into this turn's tool list.
	thinker.InjectConsole("Please upload the invoice PDF document to the billing folder.")

	ok := waitFor(120*time.Second, func() bool {
		return calls.Load() > 0
	})

	t.Logf("tool calls observed: %v", obs.snapshot())
	t.Logf("mock server tools/call hits: %d", calls.Load())

	if calls.Load() == 0 {
		t.Error("mock MCP server never received a tools/call — preload did not surface the tool, or the agent ignored it")
	}
	if obs.count("billing_upload_invoice_pdf") == 0 {
		t.Error("agent never emitted a billing_upload_invoice_pdf call")
	}
	if obs.count("search_tools") > 0 {
		t.Errorf("agent called search_tools %d× — preload should have made that unnecessary", obs.count("search_tools"))
	}
	if !ok {
		t.Error("timed out waiting for the preloaded tool to be called within 120s")
	}
}
