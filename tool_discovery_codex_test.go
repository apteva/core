package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

type discoveryWireCall struct {
	Server, Tool string
	Args         map[string]string
}

type discoveryMCPFixture struct {
	server   *httptest.Server
	mu       sync.Mutex
	calls    []discoveryWireCall
	receipt  string
	catalogs map[string][]mcpToolDef
}

func newDiscoveryMCPFixture(t *testing.T, missing bool, opaque bool) *discoveryMCPFixture {
	t.Helper()
	f := &discoveryMCPFixture{receipt: "READ-RECEIPT-" + newULID(), catalogs: patreonCatalog(t)}
	for server, tools := range f.catalogs {
		var kept []mcpToolDef
		for _, tool := range tools {
			if server == "tasks" && tool.Name == "get" && missing {
				continue
			}
			tool.InputSchema = map[string]any{"type": "object", "properties": map[string]any{"task_id": map[string]any{"type": "string"}}, "required": []string{"task_id"}}
			if server == "tasks" && tool.Name == "complete" {
				tool.InputSchema = map[string]any{"type": "object", "properties": map[string]any{"task_id": map[string]any{"type": "string"}, "result": map[string]any{"type": "string"}}, "required": []string{"task_id", "result"}}
			}
			if opaque && server == "tasks" && tool.Name == "get" {
				tool.Name = "op_7319"
				tool.Description = "Read the authoritative occurrence record and its chronological event history. Returns a unique verification receipt. This operation is read-only."
			}
			kept = append(kept, tool)
		}
		f.catalogs[server] = kept
	}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     any    `json:"id"`
			Method string `json:"method"`
			Params struct {
				Name      string                     `json:"name"`
				Arguments map[string]json.RawMessage `json:"arguments"`
			} `json:"params"`
		}
		if json.NewDecoder(r.Body).Decode(&req) != nil {
			http.Error(w, "invalid JSON", 400)
			return
		}
		if req.ID == nil {
			w.WriteHeader(202)
			return
		}
		server := strings.Trim(r.URL.Path, "/")
		var result any
		switch req.Method {
		case "initialize":
			result = map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": server, "version": "fixture"}}
		case "tools/list":
			result = map[string]any{"tools": f.catalogs[server]}
		case "tools/call":
			args := map[string]string{}
			for k, v := range req.Params.Arguments {
				var value string
				if json.Unmarshal(v, &value) != nil {
					value = string(v)
				}
				args[k] = value
			}
			f.mu.Lock()
			f.calls = append(f.calls, discoveryWireCall{server, req.Params.Name, args})
			f.mu.Unlock()
			ok := server == "tasks" && (req.Params.Name == "get" || (opaque && req.Params.Name == "op_7319")) && args["task_id"] == "occurrence-fixture"
			content := "Rejected unexpected operation; no fixture state changed."
			if ok {
				content = fmt.Sprintf(`{"task_id":"occurrence-fixture","state":"running","receipt":%q}`, f.receipt)
			}
			result = map[string]any{"content": []map[string]any{{"type": "text", "text": content}}, "isError": !ok}
		default:
			result = map[string]any{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *discoveryMCPFixture) configs() []MCPServerConfig {
	var names []string
	for name := range f.catalogs {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []MCPServerConfig
	for _, name := range names {
		cfg := MCPServerConfig{Name: name, Transport: "http", URL: f.server.URL + "/" + name, ToolLoading: &MCPToolLoadingConfig{Default: ToolLoadDeferred}}
		if name == "tasks" {
			cfg.ToolAliases = map[string]string{"occurrence_reader": "get"}
		}
		out = append(out, cfg)
	}
	return out
}

// Opt-in consumes the saved Codex allowance. The LLM is real; every MCP
// operation terminates at a local fixture with no external business effects.
func TestDiscoveryCodexProductionCatalog(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_DISCOVERY") != "1" {
		t.Skip("set RUN_CODEX_DISCOVERY=1 for live Codex discovery regression")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live Codex requested but no valid saved Codex authentication")
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	cases := []struct {
		name, query                           string
		k                                     int
		worker, missing, prerequisite, opaque bool
	}{
		{name: "exact_k1", query: "tasks_get", k: 1},
		{name: "lily_verbose_k5", query: "tasks_get exact occurrence by task id", k: 5},
		{name: "registered_alias", query: "occurrence_reader", k: 1},
		{name: "scoped_worker", query: "tasks_get", k: 1, worker: true},
		{name: "missing_reader", query: "tasks_get", k: 1, missing: true},
		{name: "execution_prerequisite", prerequisite: true},
		{name: "missing_prerequisite", prerequisite: true, missing: true},
		{name: "descriptive_opaque_reader", query: "read authoritative occurrence chronological event history", k: 3, opaque: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Chdir(t.TempDir())
			f := newDiscoveryMCPFixture(t, tc.missing, tc.opaque)
			instruction := fmt.Sprintf("First call search_tools with query %q and k=%d, then await its result. Use the returned reader to read task_id occurrence-fixture. After the read succeeds, call verification_report with the exact receipt string returned by that read.", tc.query, tc.k)
			if tc.prerequisite {
				instruction = "An incoming tracked execution requires an authoritative read. Follow required_first_action exactly with its task_id. Then call verification_report with the exact receipt returned by the reader."
			}
			instruction += " This is retrieval only. Never complete, update, recover, create, fail, or assign a Tasks record. If no authorized reader is available, call verification_report with result capability_unavailable. Do not substitute a mutation. Do not spawn or send. After verification_report, call pace clear_wake=true and wait."
			provider := &fixedModelProvider{LLMProvider: NewOpenAICodexProvider(token), model: "gpt-5.6-terra"}
			cfg := &Config{path: "config.json", Directive: instruction, Mode: ModeAutonomous, MCPServers: f.configs()}
			if err := cfg.Save(); err != nil {
				t.Fatal(err)
			}
			parent := NewThinker(token, provider, cfg)
			defer stopAuditThinker(t, parent)
			reports := make(chan string, 8)
			parent.registry.Register(&ToolDef{Name: "verification_report", Description: "Record the exact reader receipt, or capability_unavailable if discovery has no authorized reader.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"result": map[string]any{"type": "string"}}, "required": []string{"result"}}, Handler: func(args map[string]string) ToolResponse {
				reports <- args["result"]
				return ToolResponse{Text: "verification recorded"}
			}})
			thinker := parent
			if tc.worker {
				err := parent.threads.SpawnWithOpts("discovery-worker", instruction, []string{"verification_report"}, SpawnOpts{ParentID: "main", MCPNames: []string{"tasks"}, DeferRun: true})
				if err != nil {
					t.Fatal(err)
				}
				thinker = parent.threads.threads["discovery-worker"].Thinker
			}
			// The system prompt retains the instruction; disabling directive preload
			// forces ordinary cases through discovery instead of preloading the answer.
			thinker.directive = ""
			if tc.prerequisite {
				_, err := parent.QueueMainEvents([]PersistentThreadEvent{{ID: "discovery-execution", TrackLifecycle: true, Text: `{"type":"task.ready","required_first_action":{"tool":"tasks_get","task_id":"occurrence-fixture"}}`}})
				if err != nil {
					t.Fatal(err)
				}
			} else {
				thinker.Inject("Begin the verification now.")
			}
			started := time.Now()
			go thinker.Run()
			select {
			case result := <-reports:
				want := f.receipt
				if tc.missing {
					want = "capability_unavailable"
				}
				if result != want {
					t.Errorf("verification result=%q, want actual MCP receipt %q", result, want)
				}
			case <-time.After(100 * time.Second):
				events, _ := parent.telemetry.StoredEvents(0)
				for _, ev := range events {
					if ev.Type == "llm.error" || ev.Type == "tool.blocked" || ev.Type == "tool.discovery" {
						t.Logf("%s %s", ev.Type, ev.Data)
					}
				}
				t.Fatal("Codex did not finish discovery verification")
			}
			stopAuditThinker(t, thinker)
			f.mu.Lock()
			calls := append([]discoveryWireCall(nil), f.calls...)
			f.mu.Unlock()
			expectedTool := "get"
			if tc.opaque {
				expectedTool = "op_7319"
			}
			wantCalls := 1
			if tc.missing {
				wantCalls = 0
			}
			if len(calls) != wantCalls {
				t.Errorf("MCP calls=%v, want %d reads only", calls, wantCalls)
			}
			for _, call := range calls {
				if call.Server != "tasks" || call.Tool != expectedTool || call.Args["task_id"] != "occurrence-fixture" {
					t.Errorf("wrong dispatched identity or args: %+v", call)
				}
			}
			events, _ := parent.telemetry.StoredEvents(0)
			searches, manifests, dispatched, rawReads := 0, 0, 0, 0
			for _, ev := range events {
				if ev.ThreadID != thinker.threadID {
					continue
				}
				switch ev.Type {
				case "tool.discovery":
					searches++
				case "tool.manifest":
					manifests++
					if tc.prerequisite && !tc.missing && manifests == 1 && !strings.Contains(string(ev.Data), `"name":"tasks_get"`) {
						t.Error("first execution request omitted required reader")
					}
				case "tool.dispatch":
					dispatched++
				case "tool.arguments":
					var data ToolArgumentsData
					json.Unmarshal(ev.Data, &data)
					if data.Stage == "provider_raw" && strings.HasPrefix(data.Name, "tasks_") {
						if data.Name != "tasks_"+expectedTool {
							t.Errorf("model selected mutation despite read intent: %s", data.Name)
						} else {
							rawReads++
						}
					}
				}
			}
			if !tc.prerequisite && searches == 0 {
				t.Error("live test bypassed discovery")
			}
			if manifests == 0 || dispatched != wantCalls || rawReads != wantCalls {
				t.Errorf("incomplete identity trace: manifests=%d dispatch=%d rawReads=%d", manifests, dispatched, rawReads)
			}
			if tc.worker && parent.activeTools["tasks_get"] {
				t.Error("worker discovery leaked into main")
			}
			t.Logf("LIVE_DISCOVERY model=gpt-5.6-terra case=%s catalog=142 search_calls=%d raw_reads=%d dispatched_reads=%d elapsed_ms=%d", tc.name, searches, rawReads, len(calls), time.Since(started).Milliseconds())
		})
	}
}

func TestDiscoveryMCPWireIdentityAndAsyncGuard(t *testing.T) {
	t.Chdir(t.TempDir())
	f := newDiscoveryMCPFixture(t, false, false)
	cfg := &Config{path: "config.json", MCPServers: f.configs()}
	th := NewThinker("", newParkedAPIProvider(), cfg)
	defer stopAuditThinker(t, th)
	th.touchActiveTool("tasks_get")
	th.touchActiveTool("tasks_recover_occurrence")
	th.prepareNativeTools("openai-codex")
	// Real async queue and HTTP MCP boundary; no model needed for this guard test.
	for i := 0; i < 4; i++ {
		call := toolCall{Name: "tasks_recover_occurrence", NativeID: fmt.Sprintf("guard-%d", i), Args: map[string]string{"task_id": "occurrence-fixture", "reason": fmt.Sprint(i)}}
		queueTool(th, call)
		result := awaitDiscoveryToolResult(t, th, call.NativeID)
		if i == 3 && (result.Success || !strings.Contains(result.Result, "no_progress")) {
			t.Fatalf("fourth call was not explicitly blocked: %+v", result)
		}
	}
	f.mu.Lock()
	calls := append([]discoveryWireCall(nil), f.calls...)
	f.mu.Unlock()
	if len(calls) != 3 {
		t.Fatalf("guard allowed %d wire calls, want 3 then block", len(calls))
	}
	for _, c := range calls {
		if c.Server != "tasks" || c.Tool != "recover_occurrence" {
			t.Fatal(c)
		}
	}
}

func awaitDiscoveryToolResult(t *testing.T, thinker *Thinker, id string) ToolResultData {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.Type == "tool.result" {
				var result ToolResultData
				if json.Unmarshal(event.Data, &result) == nil && result.ID == id {
					return result
				}
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("tool result missing: %s", id)
	return ToolResultData{}
}

func TestDiscoveryPrerequisiteBlocksCompletionAtWire(t *testing.T) {
	t.Chdir(t.TempDir())
	f := newDiscoveryMCPFixture(t, false, false)
	cfg := &Config{path: "config.json", MCPServers: f.configs()}
	th := NewThinker("", newParkedAPIProvider(), cfg)
	defer stopAuditThinker(t, th)
	registerEventExecutionsLocked(cfg, "main", []PersistentThreadEvent{{ID: "first-action", ExecutionID: "exe", TrackLifecycle: true, Text: `{"required_first_action":{"tool":"tasks_get","task_id":"occurrence-fixture"}}`}})
	th.addEventExecutions([]string{"exe"})
	th.touchActiveTool("tasks_complete")
	th.prepareNativeTools("openai-codex")
	queueTool(th, toolCall{Name: "tasks_complete", NativeID: "wrong-complete", Args: map[string]string{"task_id": "occurrence-fixture", "result": "placeholder"}})
	result := awaitDiscoveryToolResult(t, th, "wrong-complete")
	if result.Success || !strings.Contains(result.Result, "prerequisite_required") {
		t.Fatal(result)
	}
	f.mu.Lock()
	before := len(f.calls)
	f.mu.Unlock()
	if before != 0 {
		t.Fatal("wrong completion reached MCP")
	}
	queueTool(th, toolCall{Name: "tasks_get", NativeID: "correct-get", Args: map[string]string{"task_id": "occurrence-fixture"}})
	result = awaitDiscoveryToolResult(t, th, "correct-get")
	if !result.Success || !strings.Contains(result.Result, f.receipt) {
		t.Fatal(result)
	}
	if !th.requiredActions([]string{"exe"})["exe"].Satisfied {
		t.Fatal("successful read did not durably satisfy prerequisite")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.calls) != 1 || f.calls[0].Tool != "get" {
		t.Fatal("unexpected wire operations", f.calls)
	}
}
