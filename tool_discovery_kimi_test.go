package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// Real Kimi K3 inference through Core's thinker and MCP transport. Business
// actions terminate at an isolated HTTP fixture; no real ticket is changed.
func TestDiscoveryKimiK3Tickets(t *testing.T) {
	key := getOpenCodeGoKey(t)
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	for _, tc := range []struct {
		name, query string
		automatic   bool
	}{
		{"local_names", "tickets_get tickets_update tickets note comment status assign", false},
		{"schema_search", "read ticket details assign ticket to agent assignee", false},
		{"automatic_context", "", true},
		{"automatic_default", "", true},
		{"automatic_backup", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Chdir(t.TempDir())
			var mu sync.Mutex
			var calls []string
			receipt := "ASSIGN-" + newULID()
			read, assigned := false, false
			fixture := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					ID     any    `json:"id"`
					Method string `json:"method"`
					Params struct {
						Name      string         `json:"name"`
						Arguments map[string]any `json:"arguments"`
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
				var result any
				switch req.Method {
				case "initialize":
					result = map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{"tools": map[string]any{}}, "serverInfo": map[string]any{"name": "tickets", "version": "fixture"}}
				case "tools/list":
					result = map[string]any{"tools": ticketDiscoveryCatalog()}
				case "tools/call":
					mu.Lock()
					calls = append(calls, req.Params.Name)
					content, failed := "Rejected unexpected operation or arguments; no state changed.", true
					args := req.Params.Arguments
					if args["id"] == float64(2) {
						if req.Params.Name == "tickets_get" && !read {
							read = true
							content, failed = `{"id":2,"title":"About page crash","assignee_kind":"","assignee_ref":""}`, false
						} else if req.Params.Name == "tickets_update" && read && !assigned && args["assignee_kind"] == "agent" && args["assignee_ref"] == "1118" && args["assignee_name"] == "Coding Agent" {
							assigned = true
							content, failed = fmt.Sprintf(`{"id":2,"assignee_kind":"agent","assignee_ref":"1118","assignee_name":"Coding Agent","receipt":%q}`, receipt), false
						}
					}
					mu.Unlock()
					result = map[string]any{"content": []map[string]any{{"type": "text", "text": content}}, "isError": failed}
				default:
					result = map[string]any{}
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
			}))
			defer fixture.Close()
			directive := fmt.Sprintf(`Discover tools by calling search_tools exactly once with query %q and k=5, and wait for its result. Then read ticket 2 and assign that ticket to agent 1118, name Coding Agent. Use the returned tool schemas and set all three assignee fields (kind agent, ref 1118, name Coding Agent). Read before assigning. After successful assignment call verification_report with the exact receipt from the update result. Do not modify status, add notes, create tickets or areas, spawn, send, or repeat operations. After reporting, call pace(clear_wake=true).`, tc.query)
			cfg := &Config{path: "config.json", Directive: directive, MCPServers: []MCPServerConfig{{Name: "tickets", Transport: "http", URL: fixture.URL, ToolLoading: &MCPToolLoadingConfig{Default: ToolLoadDeferred}}}}
			if tc.automatic {
				directive = "Read ticket 2, then assign it to agent 1118, name Coding Agent (assignee_kind agent, assignee_ref 1118, assignee_name Coding Agent). After successful assignment call verification_report with the exact receipt from the update result. Use available tools directly; search only if a necessary tool is missing. Do not change status, add notes, create tickets or areas, spawn, send or repeat operations. After reporting call pace(clear_wake=true)."
				cfg.Directive = directive
				cfg.AutomaticToolLoading = &AutomaticToolLoadingConfig{Enabled: true, WorkflowTools: []string{"tickets_tickets_get"}, MaxTools: 3, MaxSchemaTokens: 2000, IncludeMemory: true}
			}
			switch tc.name {
			case "automatic_default":
				cfg.AutomaticToolLoading = nil
			case "automatic_backup":
				cfg.AutomaticToolLoading = &AutomaticToolLoadingConfig{Enabled: true, MaxSchemaTokens: 1}
				cfg.Directive += " If the read or assignment schema is missing, use one search_tools call for tickets_get tickets_update to load both."
			}
			if err := cfg.Save(); err != nil {
				t.Fatal(err)
			}
			provider := &fixedModelProvider{LLMProvider: NewOpenCodeGoProvider(key), model: "kimi-k3"}
			th := NewThinker(key, provider, cfg)
			defer stopAuditThinker(t, th)
			reports := make(chan string, 8)
			th.registry.Register(&ToolDef{Name: "verification_report", Description: "Record the actual assignment receipt after the ticket update succeeds.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"receipt": map[string]any{"type": "string"}}, "required": []string{"receipt"}}, Handler: func(args map[string]string) ToolResponse {
				reports <- args["receipt"]
				return ToolResponse{Text: "verified"}
			}})
			// Keep the instruction in the system prompt but force this regression
			// through search, so automatic preload cannot conceal a discovery failure.
			if !tc.automatic {
				th.directive = ""
			}
			th.Inject("Begin the verification now.")
			started := time.Now()
			go th.Run()
			select {
			case got := <-reports:
				if got != receipt {
					t.Errorf("receipt=%q, want actual assignment receipt %q", got, receipt)
				}
			case <-time.After(150 * time.Second):
				t.Error("Kimi K3 did not finish ticket discovery and assignment")
			}
			stopAuditThinker(t, th)
			mu.Lock()
			gotCalls, didAssign := append([]string(nil), calls...), assigned
			mu.Unlock()
			if !didAssign || !reflect.DeepEqual(gotCalls, []string{"tickets_get", "tickets_update"}) {
				t.Errorf("MCP calls=%v assigned=%v", gotCalls, didAssign)
			}
			events, _ := th.telemetry.StoredEvents(0)
			searches, requests, autoloads := 0, 0, 0
			firstManifest := ""
			for _, ev := range events {
				switch ev.Type {
				case "tool.autoload":
					autoloads++
				case "tool.manifest":
					if firstManifest == "" {
						firstManifest = string(ev.Data)
					}
				case "tool.discovery":
					searches++
				case "llm.request.finished":
					requests++
				case "tool.dispatch":
					if strings.Contains(string(ev.Data), "tickets_tickets_") {
						t.Logf("canonical dispatch: %s", ev.Data)
					}
				case "llm.error", "tool.blocked":
					t.Errorf("%s: %s", ev.Type, ev.Data)
				}
			}
			wantSearches := 1
			if tc.automatic && tc.name != "automatic_backup" {
				wantSearches = 0
				if autoloads == 0 || !strings.Contains(firstManifest, "tickets_tickets_get") || !strings.Contains(firstManifest, "tickets_tickets_update") {
					t.Errorf("automatic schemas missing before first inference: autoloads=%d manifest=%s", autoloads, firstManifest)
				}
			}
			if tc.automatic && !strings.Contains(firstManifest, "search_tools") {
				t.Error("automatic loading removed search backup")
			}
			if tc.name == "automatic_backup" && (autoloads == 0 || strings.Contains(firstManifest, "tickets_tickets_update")) {
				t.Error("backup scenario did not exercise an automatic budget miss")
			}
			if searches != wantSearches {
				t.Errorf("searches=%d, want %d", searches, wantSearches)
			}
			t.Logf("kimi-k3: searches=%d requests=%d MCP=%v assigned=%v elapsed=%s", searches, requests, gotCalls, didAssign, time.Since(started).Round(time.Millisecond))
		})
	}
}
