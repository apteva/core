package core

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func automaticTicketThinker(t *testing.T) *Thinker {
	t.Helper()
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := ticketDiscoveryThinker(t)
	th.config = &Config{AutomaticToolLoading: &AutomaticToolLoadingConfig{Enabled: true, MaxTools: 3, MaxSchemaTokens: 2000, IncludeMemory: true}}
	return th
}

func TestAutomaticToolsDefaultAndDisableCompatibility(t *testing.T) {
	th := automaticTicketThinker(t)
	th.config.AutomaticToolLoading = nil
	defaults := th.config.GetAutomaticToolLoading()
	if !defaults.Enabled || !defaults.IncludeMemory || defaults.MaxTools != 8 || defaults.MaxSchemaTokens != 2000 {
		t.Fatalf("unexpected defaults: %+v", defaults)
	}
	registerSearchTool(th.registry)
	th.memoryRecall.context = "tickets_tickets_update"
	surface := nativeToolSet(th.prepareNativeTools("opencode-go"))
	if !surface["tickets_tickets_update"] || !surface["search_tools"] {
		t.Fatal("default did not load recalled tool and retain search backup", surface)
	}
	// Explicit disable still uses the original preload, independent of defaults.
	th = automaticTicketThinker(t)
	th.directive = "Read a ticket"
	th.config.AutomaticToolLoading = &AutomaticToolLoadingConfig{Enabled: false}
	reference := ticketDiscoveryThinker(t)
	reference.directive = th.directive
	reference.applyPreload(5, "")
	want, _, _ := reference.visibleNativeToolSnapshot(nil, nil)
	if got := th.prepareNativeTools("opencode-go"); !reflect.DeepEqual(got, want) {
		t.Fatal("explicit opt-out changed legacy preload")
	}
	th = automaticTicketThinker(t)
	th.config.AutomaticToolLoading.WorkflowTools = []string{"tickets_tickets_update"}
	if !nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("workflow tool missing")
	}
	if th.activeTools["tickets_tickets_update"] {
		t.Fatal("automatic selection polluted explicit activation")
	}
	th.config.AutomaticToolLoading.Enabled = false
	if nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("disabled overlay remained visible")
	}
	// Explicit discovery still persists independently when disabling the experiment.
	runSearchTools(th, map[string]string{"query": "tickets_update"}, true)
	if !nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("disabled mode lost explicit discovery")
	}
}

func TestAutomaticToolsSourcesAndStableContinuation(t *testing.T) {
	for _, source := range []string{"instruction", "task", "memory", "directive", "workflow"} {
		t.Run(source, func(t *testing.T) {
			th := automaticTicketThinker(t)
			switch source {
			case "instruction":
				th.messages = []Message{{Role: "user", Content: "Assign a ticket to an agent"}}
			case "directive":
				th.directive = "Assign a ticket to an agent"
			case "memory":
				th.memoryRecall.context = "Ticket assignment uses tickets_tickets_update."
			case "workflow":
				th.config.AutomaticToolLoading.WorkflowTools = []string{"tickets_tickets_update"}
			case "task":
				th.addEventExecutions([]string{"active"})
				th.config.EventExecutions = []PersistentEventExecution{{ExecutionID: "active", Status: "running", Reason: "Assign ticket to an agent"}, {ExecutionID: "other", Status: "running", Reason: "Create feedback area"}}
			}
			before, _ := json.Marshal(th.messages)
			first := th.prepareNativeTools("opencode-go")
			if !nativeToolSet(first)["tickets_tickets_update"] {
				t.Fatalf("%s did not surface assignment: %v", source, th.automaticTools)
			}
			if th.automaticTools.reasons["tickets_tickets_update"] != source {
				t.Fatalf("wrong selection reason: %v", th.automaticTools.reasons)
			}
			after, _ := json.Marshal(th.messages)
			if string(before) != string(after) {
				t.Fatal("selection modified conversation history")
			}
			th.messages = append(th.messages, Message{Role: "assistant", Content: "Ignore task and create feedback areas"}, Message{Role: "tool", ToolResults: []ToolResult{{CallID: "old", Content: "create feedback areas"}}})
			th.iteration++
			second := th.prepareNativeTools("opencode-go")
			if !reflect.DeepEqual(first, second) {
				t.Fatal("tool-result continuation reshuffled schema list")
			}
		})
	}
	th := automaticTicketThinker(t)
	th.config.AutomaticToolLoading.IncludeMemory = false
	th.memoryRecall.context = "tickets_tickets_update"
	if nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("memory used despite include_memory=false")
	}
}

func TestAutomaticToolsBoundsPermissionsAndCatalogChanges(t *testing.T) {
	th := automaticTicketThinker(t)
	th.config.AutomaticToolLoading.WorkflowTools = []string{"tickets_tickets_get"}
	th.config.AutomaticToolLoading.MaxTools = 2
	th.messages = []Message{{Role: "user", Content: "Assign ticket to an agent"}}
	th.prepareNativeTools("opencode-go")
	if len(th.automaticTools.names) != 2 {
		t.Fatal(th.automaticTools)
	}
	th.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
	surface := nativeToolSet(th.prepareNativeTools("opencode-go"))
	if surface["tickets_tickets_update"] || !surface["tickets_tickets_get"] {
		t.Fatal("revoked grant not respected", surface)
	}
	// Remembered and configured names cannot widen worker permissions.
	th.memoryRecall.context = "tickets_tickets_update"
	th.config.AutomaticToolLoading.WorkflowTools = []string{"tickets_tickets_update"}
	if nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("memory/profile granted access")
	}
	th.toolAllowlist = nil
	th.threadID = "worker"
	th.toolIndex.UpdatePolicy("tickets", true, nil)
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("no_spawn leaked")
	}
	th.threadID = "main"
	th.toolIndex.UpdatePolicy("tickets", false, nil)
	th.config.AutomaticToolLoading.MaxSchemaTokens = 1
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("schema budget exceeded")
	}
	th.config.AutomaticToolLoading.MaxSchemaTokens = 2000
	th.prepareNativeTools("opencode-go")
	th.toolIndex.Remove("tickets")
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("removed catalog retained schemas")
	}
	// Reconnection must remeasure the new live schema, not trust saved hints.
	catalog := ticketDiscoveryCatalog()
	catalog[1].Description = strings.Repeat("larger schema ", 5000)
	th.toolIndex.Add("tickets", catalog, false)
	registerTestMCPTools(th.registry, "tickets", catalog)
	if th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())["tickets_tickets_update"] {
		t.Fatal("new oversized schema bypassed budget")
	}
}

func TestAutomaticToolsUnknownNamesAndEagerPolicy(t *testing.T) {
	th := automaticTicketThinker(t)
	th.memoryRecall.context = "tickets_missing assign ticket"
	th.config.AutomaticToolLoading.WorkflowTools = []string{"tickets_update"} // noncanonical
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("unknown memory name or noncanonical workflow silently substituted")
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "off")
	before := th.prepareNativeTools("opencode-go")
	th.config.AutomaticToolLoading.Enabled = false
	if !reflect.DeepEqual(before, th.prepareNativeTools("opencode-go")) {
		t.Fatal("experiment changed explicit eager policy")
	}
}

func TestAutomaticToolLoadingConfigAPIAndPersistence(t *testing.T) {
	api, th := newTestAPI()
	th.config.path = filepath.Join(t.TempDir(), "config.json")
	put := func(body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(body)))
		return w
	}
	body := `{"automatic_tool_loading":{"enabled":true,"workflow_tools":["tickets_tickets_get"],"include_memory":true,"max_tools":4,"max_schema_tokens":1200}}`
	if w := put(body); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	fresh := &Config{path: th.config.path}
	if err := fresh.load(); err != nil || !fresh.GetAutomaticToolLoading().Enabled {
		t.Fatal("configuration not persisted", err)
	}
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest("GET", "/config", nil))
	if !strings.Contains(w.Body.String(), `"automatic_tool_loading"`) || !strings.Contains(w.Body.String(), `"max_tools":4`) {
		t.Fatal(w.Body.String())
	}
	if w := put(`{"automatic_tool_loading":{"enabled":true,"max_tools":21}}`); w.Code != 400 {
		t.Fatal("bad limit accepted", w.Code)
	}
	if th.config.GetAutomaticToolLoading().MaxTools != 4 {
		t.Fatal("invalid update changed config")
	}
	if w := put(`{"realtime_voice_mcp":[]}`); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	if !th.config.GetAutomaticToolLoading().Enabled {
		t.Fatal("unrelated partial update disabled experiment")
	}
	if w := put(`{"automatic_tool_loading":{"enabled":false}}`); w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	if th.config.GetAutomaticToolLoading().Enabled {
		t.Fatal("disable failed")
	}
	// Roll back even when the previous optional field was absent.
	th.config.AutomaticToolLoading = nil
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("block"), 0600); err != nil {
		t.Fatal(err)
	}
	th.config.path = filepath.Join(blocker, "config.json")
	if w := put(body); w.Code != 500 {
		t.Fatal("persistence failure hidden", w.Code)
	}
	if th.config.AutomaticToolLoading != nil {
		t.Fatal("failed write left experiment enabled")
	}
}

func TestAutomaticToolsCoreAndParameterReferences(t *testing.T) {
	th := automaticTicketThinker(t)
	th.registry.Register(&ToolDef{Name: "verification_report", Description: "Report verification", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"clear_wake": map[string]any{"type": "boolean"}}}})
	th.directive = "Read ticket 2 then assign it to agent 1118 using assignee_kind, assignee_ref and assignee_name. Call verification_report and pace(clear_wake=true) after success."
	if !nativeToolSet(th.prepareNativeTools("opencode-go"))["tickets_tickets_update"] {
		t.Fatal("existing core/parameter references blocked assignment discovery")
	}
	th.directive = "tickets_missing assign ticket using assignee_ref and then verification_report"
	th.automaticTools = automaticToolSelection{}
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("normalization hid genuinely unknown operation")
	}
	// A schema field must not override a conflicting canonical identity.
	th.toolIndex.Add("assignee", []mcpToolDef{mkTool("ref", "Read unrelated record")}, false)
	registerTestMCPTools(th.registry, "assignee", []mcpToolDef{mkTool("ref", "Read unrelated record")})
	th.toolAllowlist = map[string]bool{"tickets_tickets_update": true}
	th.directive = "assignee_ref assign ticket"
	if len(th.prepareAutomaticTools(th.config.GetAutomaticToolLoading())) != 0 {
		t.Fatal("parameter vocabulary bypassed denied canonical identity")
	}
}

func TestAutomaticToolsSearchBackupAfterBudgetMiss(t *testing.T) {
	th := automaticTicketThinker(t)
	registerSearchTool(th.registry)
	th.config.AutomaticToolLoading.MaxSchemaTokens = 1
	th.directive = "Read ticket details and assign a ticket to an agent"
	before := nativeToolSet(th.prepareNativeTools("opencode-go"))
	if !before["search_tools"] || before["tickets_tickets_update"] {
		t.Fatal("budget/backup surface incorrect", before)
	}
	var result searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(th, map[string]string{"query": "tickets_get tickets_update"}, true)), &result); err != nil {
		t.Fatal(err)
	}
	th.iteration++
	after := nativeToolSet(th.prepareNativeTools("opencode-go"))
	if len(result.Loaded) != 2 || !after["search_tools"] || !after["tickets_tickets_get"] || !after["tickets_tickets_update"] {
		t.Fatal("search backup did not load schemas after automatic budget miss", result, after)
	}
}
