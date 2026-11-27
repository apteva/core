package core

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// automaticTicketThinker is a main thread in search mode over the ticket
// catalog.
func automaticTicketThinker(t *testing.T) *Thinker {
	t.Helper()
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := ticketDiscoveryThinker(t)
	th.config = &Config{}
	return th
}

func mcpToolSet(tools []NativeTool, prefix string) []string {
	var out []string
	for _, tool := range tools {
		if strings.HasPrefix(tool.Name, prefix) {
			out = append(out, tool.Name)
		}
	}
	return out
}

// The mode follows what the catalog costs to send, not how many tools it
// holds: many small tools go out whole, a few large ones do not.
func TestToolModeFollowsSchemaTokensNotToolCount(t *testing.T) {
	small := NewToolIndex()
	var many []mcpToolDef
	for i := 0; i < 200; i++ {
		many = append(many, mkTool(fmt.Sprintf("tool_%03d", i), "A small tool."))
	}
	small.Add("many", many, false)
	if tokens := small.SchemaTokens(nil); tokens > defaultDirectToolTokens {
		t.Fatalf("fixture: 200 small tools cost %d tokens", tokens)
	}
	if th := (&Thinker{threadID: "main", toolIndex: small}); !th.useEagerTools() {
		t.Fatal("a catalog of many small tools under the budget was not sent whole")
	}

	large := NewToolIndex()
	wide := map[string]any{"type": "object", "properties": map[string]any{}}
	for i := 0; i < 400; i++ {
		wide["properties"].(map[string]any)[fmt.Sprintf("field_%03d", i)] = map[string]any{"type": "string", "description": strings.Repeat("A long field description. ", 8)}
	}
	var few []mcpToolDef
	for i := 0; i < 5; i++ {
		few = append(few, mcpToolDef{Name: fmt.Sprintf("big_%d", i), Description: "A large tool.", InputSchema: wide})
	}
	large.Add("few", few, false)
	th := &Thinker{threadID: "main", toolIndex: large}
	if tokens := large.SchemaTokens(nil); tokens <= defaultDirectToolTokens {
		t.Fatalf("fixture: 5 large tools cost only %d tokens", tokens)
	}
	if th.useEagerTools() {
		t.Fatal("a catalog over the budget was sent whole")
	}
	t.Setenv("APTEVA_DIRECT_TOOL_TOKENS", "10000000")
	if !th.useEagerTools() {
		t.Fatal("APTEVA_DIRECT_TOOL_TOKENS did not raise the budget")
	}
	// A tool its app defers is never part of what direct mode would send.
	deferred := NewToolIndex()
	deferred.Add("few", few, false, &MCPToolLoadingConfig{Default: ToolLoadDeferred})
	if deferred.SchemaTokens(nil) != 0 {
		t.Fatal("explicitly deferred tools counted toward the direct budget")
	}
}

// A worker scoped to a few tools gets them directly, however large the
// agent's whole catalog is.
func TestToolModeIsDecidedPerThread(t *testing.T) {
	th := ticketDiscoveryThinker(t)
	t.Setenv("APTEVA_DIRECT_TOOL_TOKENS", fmt.Sprint(th.toolIndex.SchemaTokens(nil)-1))
	if th.useEagerTools() {
		t.Fatal("fixture: the whole catalog should be over the budget")
	}
	worker := &Thinker{threadID: "worker", toolIndex: th.toolIndex, registry: th.registry, activeTools: map[string]bool{},
		toolAllowlist: map[string]bool{"tickets_tickets_get": true, "tickets_tickets_update": true}}
	if !worker.useEagerTools() {
		t.Fatal("a worker allowed two tools was put in search mode")
	}
	got := mcpToolSet(worker.prepareNativeTools("openai-codex"), "tickets_")
	if len(got) != 2 {
		t.Fatalf("worker tool list = %v, want its two tools", got)
	}
}

// In search mode nothing loads because some text mentions it. The model
// loads a tool by searching, and it then stays loaded.
func TestSearchModeLoadsOnlyWhatTheModelAsksFor(t *testing.T) {
	th := automaticTicketThinker(t)
	registerSearchTool(th.registry)
	th.directive = "Use tickets_tickets_update to assign every ticket. Read ticket history."
	th.messages = []Message{{Role: "user", Content: "Assign ticket 4 with tickets_tickets_update"}}
	th.memoryRecall.context = "Last week we used tickets_tickets_get."
	first := th.prepareNativeTools("openai-codex")
	if got := mcpToolSet(first, "tickets_"); len(got) != 0 {
		t.Fatalf("tools loaded without the model asking: %v", got)
	}
	if !nativeToolSet(first)["search_tools"] {
		t.Fatal("search_tools missing in search mode")
	}

	runSearchTools(th, map[string]string{"query": "tickets_tickets_update", "k": "1"}, true)
	th.iteration++
	if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); len(got) != 1 || got[0] != "tickets_tickets_update" {
		t.Fatalf("search did not load exactly the tool asked for: %v", got)
	}
	// It stays loaded on later turns, whatever they are about.
	for i := 0; i < 5; i++ {
		th.iteration++
		th.messages = append(th.messages, Message{Role: "user", Content: "Create a feedback area"})
		if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); len(got) != 1 {
			t.Fatalf("turn %d changed the tool list: %v", i, got)
		}
	}
	// Grants still bound it: narrowed to another tool, the loaded one goes.
	th.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
	if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); containsString(got, "tickets_tickets_update") {
		t.Fatalf("a loaded tool outlived its grant: %v", got)
	}
}

// workflow_tools is the operator's explicit list of tools to keep loaded.
func TestPinnedToolsStayLoadedInSearchMode(t *testing.T) {
	th := automaticTicketThinker(t)
	th.config.AutomaticToolLoading = &AutomaticToolLoadingConfig{Enabled: true, WorkflowTools: []string{"tickets_tickets_get", "tickets_tickets_create", "tickets_not_a_tool"}}
	if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); len(got) != 2 {
		t.Fatalf("pinned tools = %v, want the two that exist", got)
	}
	if th.activeTools["tickets_tickets_get"] {
		t.Fatal("pinning marked the tool as loaded by the model")
	}
	th.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
	if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); len(got) != 1 || got[0] != "tickets_tickets_get" {
		t.Fatalf("pinning bypassed grants: %v", got)
	}
	// The other settings of the old automatic loading are stored and ignored.
	th = automaticTicketThinker(t)
	th.config.AutomaticToolLoading = &AutomaticToolLoadingConfig{Enabled: true, MaxTools: 8, MaxSchemaTokens: 6000, IncludeMemory: true}
	th.directive = "Read tickets"
	if got := mcpToolSet(th.prepareNativeTools("openai-codex"), "tickets_"); len(got) != 0 {
		t.Fatalf("legacy settings loaded tools: %v", got)
	}
}

func TestToolDirectoryInPrompt(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("tickets", ticketDiscoveryCatalog(), false)
	ix.Add("pinned", []mcpToolDef{mkTool("status", "Report status. Always on.")}, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	catalog := computeMCPCatalog(ix)

	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	prompt := buildSystemPrompt("d", nil, "", nil, nil, nil, catalog)
	for _, want := range []string{"[TOOL DIRECTORY]", "- tickets_tickets_get: Fetch one ticket", "- tickets_ticket_areas_create: Create a feedback area.", "call search_tools"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("search-mode prompt lacks %q:\n%s", want, prompt)
		}
	}
	if strings.Contains(prompt, "- pinned_status") {
		t.Fatal("a tool already in the tool list was listed as loadable")
	}

	t.Setenv("APTEVA_TOOL_SEARCH", "auto")
	if prompt := buildSystemPrompt("d", nil, "", nil, nil, nil, catalog); strings.Contains(prompt, "[TOOL DIRECTORY]") {
		t.Fatal("direct mode listed a directory for tools that are already sent")
	}

	// A directory past the cap is dropped; the per-server counts remain.
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	var many []mcpToolDef
	for i := 0; i < 400; i++ {
		many = append(many, mkTool(fmt.Sprintf("operation_number_%03d", i), "Do one specific thing with a record in this system."))
	}
	huge := NewToolIndex()
	huge.Add("huge", many, false)
	prompt = buildSystemPrompt("d", nil, "", nil, nil, nil, computeMCPCatalog(huge))
	if strings.Contains(prompt, "[TOOL DIRECTORY]") || !strings.Contains(prompt, "- huge (400 tools") {
		t.Fatalf("oversized catalog was not reduced to counts")
	}

	// A thread with narrower grants lists only what it may use.
	scoped := computeMCPCatalog(ix, func(e IndexEntry) bool { return e.Name == "tickets_tickets_get" })
	if len(scoped) != 1 || scoped[0].ToolCount != 1 || renderToolDirectory(scoped, false) != "- tickets_tickets_get: Fetch one ticket with comments, attachments, links, and chronological history.\n" {
		t.Fatalf("scoped catalog = %+v", scoped)
	}
	if got := toolDirectorySummary("First sentence here. Second sentence.\nThird."); got != "First sentence here." {
		t.Fatalf("summary = %q", got)
	}
	if got := toolDirectorySummary(strings.Repeat("word ", 60)); len(got) > toolDirectorySummaryChars+4 || !strings.HasSuffix(got, "…") {
		t.Fatalf("long summary not cut: %q", got)
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
		t.Fatal("unrelated partial update changed the stored setting")
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
		t.Fatal("failed write left the new setting in place")
	}
}
