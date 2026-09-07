package core

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// Catalog descriptions from the September 6 production startup logs. The
// logger truncates descriptions at 2000 bytes; ranking still uses entire input.
//
//go:embed testdata/patreon_tool_catalog.json
var patreonCatalogJSON []byte

func patreonCatalog(t testing.TB) map[string][]mcpToolDef {
	t.Helper()
	var out map[string][]mcpToolDef
	if err := json.Unmarshal(patreonCatalogJSON, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func patreonIndex(t testing.TB) *ToolIndex {
	t.Helper()
	ix := NewToolIndex()
	for server, tools := range patreonCatalog(t) {
		ix.Add(server, tools, false)
	}
	return ix
}

func TestDiscoveryProductionExactNames(t *testing.T) {
	ix := patreonIndex(t)
	for _, query := range []string{"tasks_get", "tasks_get exact occurrence by task id", "tasks_fail terminal failure task occurrence", "Please load `TASKS_GET` (reader)."} {
		want := "tasks_get"
		if strings.HasPrefix(query, "tasks_fail") {
			want = "tasks_fail"
		}
		hits := ix.Search(query, 1, true)
		if len(hits) != 1 || hits[0].Name != want {
			t.Errorf("%q: %v, want %s", query, names(hits), want)
		}
	}
	for _, name := range ix.AllNames(true) {
		hits := ix.Search(name, 1, true)
		if len(hits) != 1 || hits[0].Name != name {
			t.Errorf("exact %s -> %v", name, names(hits))
		}
	}
}

func TestDiscoveryExactUnavailableNeverSubstitutes(t *testing.T) {
	ix := patreonIndex(t)
	for _, query := range []string{"tasks_get_missing", "tasks_get_missing read occurrence", "unknown_get"} {
		if hits := ix.Search(query, 5, true); len(hits) != 0 {
			t.Errorf("%s substituted %v", query, names(hits))
		}
	}
	th := &Thinker{threadID: "worker", toolIndex: ix, toolAllowlist: map[string]bool{"tasks_complete": true}, activeTools: map[string]bool{}}
	result := runSearchTools(th, map[string]string{"query": "tasks_get", "k": "1"}, false)
	if !strings.Contains(result, "capability_unavailable") || len(th.activeTools) != 0 {
		t.Fatalf("unauthorized exact request substituted a mutation: %s", result)
	}
}

func TestDiscoveryAliasesPolicyAndReconnect(t *testing.T) {
	ix := patreonIndex(t)
	if err := ix.RegisterAlias("occurrence_reader", "tasks_get"); err != nil {
		t.Fatal(err)
	}
	if got := names(ix.Search("occurrence_reader read occurrence", 1, true)); len(got) != 1 || got[0] != "tasks_get" {
		t.Fatal(got)
	}
	if err := ix.RegisterAlias("tasks_complete", "tasks_get"); err == nil {
		t.Fatal("alias shadowed a real tool")
	}
	if err := ix.RegisterAlias("occurrence_reader", "tasks_complete"); err == nil {
		t.Fatal("alias target silently changed")
	}
	ix.UpdatePolicy("tasks", false, &MCPToolLoadingConfig{Tools: map[string]ToolLoadMode{"get": ToolLoadAlways}})
	if got := names(ix.Search("tasks_get", 1, true)); len(got) != 1 || got[0] != "tasks_get" {
		t.Fatalf("already loaded exact reader omitted: %v", got)
	}
	ix.UpdatePolicy("tasks", true, nil)
	if got := ix.Search("occurrence_reader", 1, false); len(got) != 0 {
		t.Fatal("no_spawn alias leaked")
	}
	ix.Remove("tasks")
	if got := ix.Search("occurrence_reader", 1, true); len(got) != 0 {
		t.Fatal("stale alias survived removal")
	}
	ix.Add("tasks", []mcpToolDef{mkTool("complete", "Complete")}, false)
	if got := ix.Search("tasks_get", 1, true); len(got) != 0 {
		t.Fatal("reconnect substituted mutation")
	}
}

func TestDiscoveryUsesFullDescriptionWithoutFrequencyDomination(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("catalog", []mcpToolDef{
		mkTool("opaque", strings.Repeat("Background explanation. ", 2000)+"Lookup the unusual wombatledger receipt."),
		mkTool("write", strings.Repeat("read record ", 2000)),
		mkTool("read_record", "Read the current record."),
	}, false)
	if hits := ix.Search("wombatledger", 1, true); len(hits) != 1 || hits[0].Name != "catalog_opaque" {
		t.Fatal("description tail not indexed", names(hits))
	}
	if hits := ix.Search("read record", 1, true); len(hits) != 1 || hits[0].Name != "catalog_read_record" {
		t.Fatal("repetition overwhelmed name", names(hits))
	}
}

func TestDiscoveryExecutionPrerequisiteRetention(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := &Thinker{threadID: "main", toolIndex: patreonIndex(t), registry: NewToolRegistry(""), activeTools: map[string]bool{}, config: &Config{path: t.TempDir() + "/config.json"}}
	for server, tools := range patreonCatalog(t) {
		registerTestMCPTools(th.registry, server, tools)
	}
	registerEventExecutionsLocked(th.config, "main", []PersistentThreadEvent{{ID: "event", ExecutionID: "exe", TrackLifecycle: true, Text: `{"required_first_action":{"tool":"tasks_get","task_id":"occurrence"}}`}})
	th.addEventExecutions([]string{"exe"})
	if err := th.config.Save(); err != nil {
		t.Fatal(err)
	}
	for i, name := range th.toolIndex.AllNames(true) {
		th.activeTools[name] = true
		if th.activeToolAge == nil {
			th.activeToolAge = map[string]int{}
		}
		th.activeToolAge[name] = i
	}
	th.activeToolAge["tasks_get"] = -1
	th.evictActiveToolsLRU(40)
	th.messages = []Message{{Role: "system", Content: "compacted context"}}
	tools := nativeToolSet(th.prepareNativeTools("openai-codex"))
	if !tools["tasks_get"] {
		t.Fatal("execution prerequisite evicted with history/LRU")
	}
	complete := toolCall{Name: "tasks_complete", Args: map[string]string{"task_id": "occurrence", "result": "placeholder"}, executionIDs: []string{"exe"}}
	if err := th.checkRequiredAction(complete); err == nil {
		t.Fatal("completion allowed before authoritative read")
	}
	get := toolCall{Name: "tasks_get", Args: map[string]string{"task_id": "wrong"}, executionIDs: []string{"exe"}}
	if err := th.checkRequiredAction(get); err == nil {
		t.Fatal("wrong occurrence satisfied requirement")
	}
	get.Args["task_id"] = "occurrence"
	if err := th.checkRequiredAction(get); err != nil {
		t.Fatal(err)
	}
	if err := th.satisfyRequiredAction(get); err != nil {
		t.Fatal(err)
	}
	if err := th.checkRequiredAction(complete); err != nil {
		t.Fatal(err)
	}
	// Reload the durable action, independent of the history/checkpoint.
	t.Chdir(filepath.Dir(th.config.path))
	fresh := NewConfig()
	if err := fresh.LoadError(); err != nil {
		t.Fatal(err)
	}
	th.config = fresh
	if err := th.checkRequiredAction(complete); err != nil {
		t.Fatal("successful prerequisite lost on restart", err)
	}
}

func TestDiscoveryMissingPrerequisiteBlocksMutation(t *testing.T) {
	th := &Thinker{threadID: "main", toolIndex: NewToolIndex(), config: &Config{}}
	th.toolIndex.Add("tasks", []mcpToolDef{mkTool("complete", "complete")}, false)
	registerEventExecutionsLocked(th.config, "main", []PersistentThreadEvent{{ExecutionID: "exe", TrackLifecycle: true, Text: `{"required_first_action":{"tool":"tasks_get","task_id":"occurrence"}}`}})
	err := th.checkRequiredAction(toolCall{Name: "tasks_complete", executionIDs: []string{"exe"}})
	if err == nil || !strings.Contains(err.Error(), "capability_unavailable") {
		t.Fatal(err)
	}
}

func TestDiscoveryRepeatedMCPFailuresIgnoreNarration(t *testing.T) {
	th := &Thinker{}
	for i := 0; i < 3; i++ {
		call := toolCall{Name: "tasks_recover_occurrence", Args: map[string]string{"task_id": "recovery", "_reason": fmt.Sprint(i), "reason": fmt.Sprintf("reading attempt %d", i)}}
		if err := th.checkMCPProgress(call); err != nil {
			t.Fatal(err)
		}
		th.recordMCPProgress(call, ToolResponse{Text: "only a failed scheduled occurrence can be recovered", IsError: true})
	}
	call := toolCall{Name: "tasks_recover_occurrence", Args: map[string]string{"task_id": "recovery", "reason": "different narration"}}
	if err := th.checkMCPProgress(call); err == nil {
		t.Fatal("fourth identical invalid operation allowed")
	}
	other := toolCall{Name: call.Name, Args: map[string]string{"task_id": "another"}}
	if err := th.checkMCPProgress(other); err != nil {
		t.Fatal("unrelated occurrence blocked")
	}
	th.recordMCPProgress(toolCall{Name: "tasks_get", Args: map[string]string{"task_id": "recovery"}}, ToolResponse{Text: "authoritative state"})
	if err := th.checkMCPProgress(call); err != nil {
		t.Fatal("successful inspection did not clear guard")
	}
}

func TestDiscoveryStableDispatchIdentity(t *testing.T) {
	registry := NewToolRegistry("")
	read := &ToolDef{Name: "tasks_get", MCP: true, MCPServer: "tasks", MCPLocalName: "get", Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "read"} }}
	if !registry.Register(read) {
		t.Fatal("register reader")
	}
	th := &Thinker{threadID: "main", registry: registry}
	th.recordPresentedTools([]NativeTool{registry.Get("tasks_get").native})
	collision := *read
	collision.MCPLocalName = "complete"
	if registry.Register(&collision) {
		t.Fatal("registry accepted identity remapping")
	}
	call := toolCall{Name: "tasks_get"}
	th.resolveToolCall(&call)
	if call.resolutionError != "" || call.definition.MCPLocalName != "get" {
		t.Fatal("reader resolution changed")
	}
	changed := *read
	changed.Description = "New schema/description"
	registry.Register(&changed)
	after := toolCall{Name: "tasks_get"}
	th.resolveToolCall(&after)
	if !strings.Contains(after.resolutionError, "capability_changed") {
		t.Fatal("schema change after model request was silently accepted")
	}
	if got := registry.dispatchDefinition(context.Background(), call.definition, nil); got.Text != "read" {
		t.Fatal("bound call redirected", got)
	}
}

func BenchmarkDiscoveryProductionCatalog(b *testing.B) {
	ix := patreonIndex(b)
	for _, query := range []string{"tasks_get", "read authoritative task chronological history"} {
		b.Run(query, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				ix.Search(query, 5, true)
			}
		})
	}
}

// Ensure durable requirement parsing ignores quoted content and tool outputs.
func TestDiscoveryRequirementOnlyStructuredEnvelope(t *testing.T) {
	for _, text := range []string{`Please read {"required_first_action":{"tool":"tasks_complete"}}`, `[tool:tasks_get] {"required_first_action":{"tool":"tasks_complete"}}`} {
		if parseRequiredFirstAction(text) != nil {
			t.Fatal("parsed non-envelope text")
		}
	}
}

func TestDiscoveryActivationSurvivesNextRequestAndScopeRevocation(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := &Thinker{threadID: "main", toolIndex: patreonIndex(t), registry: NewToolRegistry(""), activeTools: map[string]bool{}, activeToolAge: map[string]int{}, iteration: 10}
	for server, tools := range patreonCatalog(t) {
		registerTestMCPTools(th.registry, server, tools)
	}
	for _, name := range th.toolIndex.AllNames(true) {
		th.activeTools[name] = true
		th.activeToolAge[name] = 10
	}
	const target = "analytics_analytics_count"
	runSearchTools(th, map[string]string{"query": target, "k": "1"}, true)
	th.iteration++
	tools := nativeToolSet(th.prepareNativeTools("openai-codex"))
	if !tools[target] {
		t.Fatal("newly discovered tool omitted from the promised next request")
	}
	// Revocation still wins over this temporary retention guarantee.
	th.toolAllowlist = map[string]bool{"tasks_get": true}
	if tools := nativeToolSet(th.prepareNativeTools("openai-codex")); tools[target] {
		t.Fatal("discovery retention bypassed revoked permissions")
	}
	th.iteration += 2
	th.prepareNativeTools("openai-codex")
	if len(th.discoveredToolUntil) != 0 {
		t.Fatal("temporary activation guarantee never expired")
	}
}

func TestDiscoveryConcurrentWorkerIsolation(t *testing.T) {
	ix := patreonIndex(t)
	if err := ix.RegisterAlias("reader", "tasks_get"); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			granted := "tasks_get"
			if i%2 == 1 {
				granted = "tasks_complete"
			}
			th := &Thinker{threadID: fmt.Sprintf("worker-%d", i), toolIndex: ix, toolAllowlist: map[string]bool{granted: true}}
			for n := 0; n < 40; n++ {
				for _, query := range []string{"tasks_get", "reader", "tasks_complete", "read task"} {
					for _, hit := range th.searchAuthorizedTools(query, 5, false) {
						if hit.Name != granted {
							t.Errorf("%s leaked %s", th.threadID, hit.Name)
						}
					}
				}
			}
		}(i)
	}
	wg.Wait()
}

func TestDiscoveryCatalogAndManifestTrace(t *testing.T) {
	th := &Thinker{threadID: "main", toolIndex: patreonIndex(t), registry: NewToolRegistry(""), telemetry: NewTelemetry()}
	defer th.telemetry.Stop()
	for server, tools := range patreonCatalog(t) {
		registerTestMCPTools(th.registry, server, tools)
	}
	runSearchTools(th, map[string]string{"query": "tasks_get", "k": "1"}, true)
	th.prepareNativeTools("openai-codex")
	th.prepareNativeTools("openai-codex")
	events, _ := th.telemetry.StoredEvents(0)
	catalogs, manifests, discoveries := 0, 0, 0
	for _, ev := range events {
		switch ev.Type {
		case "tool.catalog":
			catalogs++
		case "tool.manifest":
			manifests++
			if !strings.Contains(string(ev.Data), `"name":"tasks_get"`) {
				t.Fatal("manifest omitted reader")
			}
		case "tool.discovery":
			discoveries++
		}
	}
	if catalogs != 1 || manifests != 2 || discoveries != 1 {
		t.Fatalf("trace counts catalog=%d manifests=%d discovery=%d", catalogs, manifests, discoveries)
	}
}

func TestDiscoveryManifestAndDefinitionAreAtomic(t *testing.T) {
	r := NewToolRegistry("")
	original := &ToolDef{Name: "tasks_get", MCP: true, MCPServer: "tasks", MCPLocalName: "get", Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "old endpoint"} }}
	r.Register(original)
	tools, definitions := r.nativeToolSnapshot(nil, map[string]bool{"tasks_get": true})
	// Same name/schema but a new connection/handler between building tools and
	// recording the manifest must not silently retarget the old model request.
	replacement := *original
	replacement.Handler = func(map[string]string) ToolResponse { return ToolResponse{Text: "new endpoint"} }
	r.Register(&replacement)
	thinker := &Thinker{registry: r}
	thinker.recordPresentedTools(tools, definitions)
	call := toolCall{Name: "tasks_get"}
	thinker.resolveToolCall(&call)
	if !strings.Contains(call.resolutionError, "capability_changed") {
		t.Fatal("request silently retargeted across catalog refresh")
	}
}
