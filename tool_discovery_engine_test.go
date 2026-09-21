package core

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func productionDiscoveryThinker(t testing.TB) *Thinker {
	t.Helper()
	index := NewToolIndex()
	registry := NewToolRegistry("")
	add := func(server string, noSpawn bool, tools ...mcpToolDef) {
		index.Add(server, tools, noSpawn)
		registerTestMCPTools(registry, server, tools)
	}
	read := func(name, description string) mcpToolDef {
		tool := mkTool(name, description)
		tool.Annotations = map[string]any{"readOnlyHint": true, "destructiveHint": false}
		return tool
	}
	add("code", false,
		read("list_files", "List repository files and directories."),
		read("read_file", "Read repository file contents."),
		read("grep", "Search repository source code."),
		read("repos_list", "List connected code repositories."),
		mkTool("create_file", "Create a repository file."),
		mkTool("update_file", "Update a repository file."))
	add("analytics", false,
		read("topics", "Read analytics topics and traffic performance."),
		read("query", "Query analytics performance metrics."),
		read("top", "List top analytics performance results."))
	add("editorial", false,
		read("list_items", "List editorial items and publication state."),
		read("get_item", "Read one editorial item."),
		mkTool("create_item", "Create an editorial item."),
		mkTool("update_item", "Update an editorial item."),
		mkTool("delete_item", "Delete an editorial item."))
	add("podcast", false,
		read("list_episodes", "List podcast episodes."),
		mkTool("publish_episode", "Publish a podcast episode."))
	add("crm", false,
		read("list_contacts", "List CRM contacts and accounts."),
		mkTool("update_contact", "Update a CRM contact."))
	add("private-admin", true, read("list_secrets", "List private administrative secrets."))
	decoys := map[string][]mcpToolDef{"seo": {}, "crm-decoys": {}}
	for i := 0; i < 180; i++ {
		server := "seo"
		if i%2 == 1 {
			server = "crm-decoys"
		}
		decoys[server] = append(decoys[server], read(fmt.Sprintf("search_report_%03d", i), "Search editorial analytics performance repository files contacts campaigns and traffic reports."))
	}
	for server, tools := range decoys {
		add(server, false, tools...)
	}
	return &Thinker{threadID: "main", toolIndex: index, registry: registry, activeTools: map[string]bool{}, config: &Config{AutomaticToolLoading: &AutomaticToolLoadingConfig{Enabled: true, MaxTools: 8, MaxSchemaTokens: 2000, IncludeMemory: true}}}
}

func TestDiscoveryEngineCompoundServerCoverageSurvivesCrowding(t *testing.T) {
	thinker := productionDiscoveryThinker(t)
	out := runSearchTools(thinker, map[string]string{"query": "Editorial items, Code repository files, Analytics performance", "k": "3", "access": "read_only"}, true)
	var result searchToolsResult
	if err := json.Unmarshal([]byte(out), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 3 {
		t.Fatalf("compound query produced %d groups, want one per named server: %s", len(result.Results), out)
	}
	wantServers := map[string]bool{"analytics": true, "code": true, "editorial": true}
	for _, group := range result.Results {
		if len(group.RequestedServers) != 1 || !wantServers[group.RequestedServers[0]] {
			t.Fatalf("wrong hard scope: %+v", group)
		}
		if group.Coverage != "satisfied" || !reflect.DeepEqual(group.RequestedServers, group.RepresentedServers) {
			t.Fatalf("named server was crowded out: %+v", group)
		}
		for _, candidate := range group.Candidates {
			if candidate.Loaded && candidate.Access != ToolAccessRead {
				t.Fatalf("read_only loaded %s access=%s", candidate.Name, candidate.Access)
			}
		}
	}
	if len(result.Loaded) != 3 {
		t.Fatalf("loaded=%v, want one representative per server", result.Loaded)
	}
}

func TestDiscoveryEngineStructuredPartialAndLexicalMiss(t *testing.T) {
	thinker := productionDiscoveryThinker(t)
	queries := `[{
		"query":"repository files",
		"servers":["code","not-attached"],
		"access":"read_only",
		"limit":3
	}]`
	var partial searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(thinker, map[string]string{"queries": queries, "k": "3"}, true)), &partial); err != nil {
		t.Fatal(err)
	}
	if len(partial.Results) != 1 || partial.Results[0].Coverage != "partial" || !reflect.DeepEqual(partial.Results[0].MissingServers, []string{"not-attached"}) {
		t.Fatalf("partial coverage was not explicit: %+v", partial.Results)
	}
	var miss searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(thinker, map[string]string{"query": "wombatledger", "server_names": `["code"]`, "access": "read_only"}, true)), &miss); err != nil {
		t.Fatal(err)
	}
	if len(miss.Results) != 1 || miss.Results[0].Coverage != "lexical_miss" || !strings.Contains(miss.Error, "does not prove") || strings.Contains(miss.Error, "capability_unavailable") {
		t.Fatalf("lexical miss made an availability claim: %+v", miss)
	}
}

func TestDiscoveryEngineStrictReadOnlyAndMCPAnnotations(t *testing.T) {
	var tool mcpToolDef
	if err := json.Unmarshal([]byte(`{"name":"opaque_reader","description":"Inspect an item","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":true,"destructiveHint":false}}`), &tool); err != nil {
		t.Fatal(err)
	}
	index := NewToolIndex()
	falseReader := mkTool("get_false", "Inspect an item")
	falseReader.Annotations = map[string]any{"readOnlyHint": false}
	destructiveReader := mkTool("get_destructive", "Inspect an item")
	destructiveReader.Annotations = map[string]any{"destructiveHint": true}
	index.Add("opaque", []mcpToolDef{tool, falseReader, destructiveReader, mkTool("create_item", "Inspect and create an item"), mkTool("mystery", "Inspect an item")}, false)
	if entry, ok := index.Get("opaque_opaque_reader"); !ok || entry.Access != ToolAccessRead || entry.SchemaCost <= 0 {
		t.Fatalf("MCP annotations/schema cost were not retained: %#v", entry)
	}
	thinker := &Thinker{threadID: "main", toolIndex: index, activeTools: map[string]bool{}}
	result := thinker.discoverTools(DiscoveryRequest{Intents: []DiscoveryIntent{{Query: "inspect item", Servers: []string{"opaque"}, Access: DiscoveryAccessReadOnly, Limit: 10, Priority: 90}}, AllowNoSpawn: true, MaxTools: 10, MaxSchemaTokens: 16000, Activate: true})
	if !reflect.DeepEqual(result.Loaded, []string{"opaque_opaque_reader"}) {
		t.Fatalf("strict read-only admitted write/unknown tools: %+v", result)
	}
}

func TestDiscoveryEngineAutomaticCurrentIntentOutranksMemory(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	thinker := productionDiscoveryThinker(t)
	thinker.config.AutomaticToolLoading.MaxTools = 2
	thinker.messages = []Message{{Role: "user", Content: "Use Code repository files and Analytics performance."}}
	thinker.memoryRecall.context = "CRM contacts and accounts are frequently important."
	selected := thinker.prepareAutomaticTools(thinker.config.GetAutomaticToolLoading())
	if !selected["code_list_files"] || !selected["analytics_top"] && !selected["analytics_query"] && !selected["analytics_topics"] {
		t.Fatalf("current multi-server intent lost coverage: %v (%+v)", selected, thinker.automaticTools)
	}
	if selected["crm_list_contacts"] {
		t.Fatalf("memory displaced the current request: %v", selected)
	}
}

func TestDiscoveryEngineScopeNoSpawnAndDeterministicBudget(t *testing.T) {
	thinker := productionDiscoveryThinker(t)
	thinker.threadID = "worker"
	thinker.toolAllowlist = map[string]bool{"code_read_file": true}
	structured := `[{"query":"repository file contents","servers":["code"],"access":"read_only","limit":10},{"query":"administrative secrets","servers":["private-admin"],"access":"read_only","limit":10}]`
	var result searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(thinker, map[string]string{"queries": structured, "k": "2"}, false)), &result); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Loaded, []string{"code_read_file"}) || thinker.activeTools["private-admin_list_secrets"] {
		t.Fatalf("scope/no_spawn boundary failed: %+v", result)
	}
	if len(result.Results) != 2 || result.Results[1].Coverage != "unavailable_in_scope" {
		t.Fatalf("hidden scope coverage was not explicit: %+v", result.Results)
	}

	fresh := productionDiscoveryThinker(t)
	request := DiscoveryRequest{Intents: compileDiscoveryIntent("Code repository files and Analytics performance", nil, DiscoveryAccessReadOnly, 8, "explicit", 90, fresh.authorizedDiscoveryServers(true)), AllowNoSpawn: true, MaxTools: 2, MaxSchemaTokens: 16000}
	first, second := fresh.discoverTools(request), fresh.discoverTools(request)
	if !reflect.DeepEqual(first.Loaded, second.Loaded) || first.SchemaTokens != second.SchemaTokens || first.CatalogRevision != second.CatalogRevision {
		t.Fatalf("planner is not deterministic: first=%+v second=%+v", first, second)
	}
}

func BenchmarkDiscoveryEngineCrowdedCompoundQuery(b *testing.B) {
	thinker := productionDiscoveryThinker(b)
	available := thinker.authorizedDiscoveryServers(true)
	intents := compileDiscoveryIntent("Editorial items, Code repository files, Analytics performance", nil, DiscoveryAccessReadOnly, 5, "explicit", 90, available)
	request := DiscoveryRequest{Intents: intents, AllowNoSpawn: true, MaxTools: 5, MaxSchemaTokens: 16000}
	b.ReportAllocs()
	for b.Loop() {
		thinker.discoverTools(request)
	}
}
