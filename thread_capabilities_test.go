package core

import (
	"net/http"
	"slices"
	"strings"
	"testing"
)

func registerDelegationFixture(parent *Thinker) {
	parent.registry.Register(&ToolDef{
		Name: "local_inspect", Description: "Inspect a local fixture.",
		InputSchema: map[string]any{"type": "object"},
		Handler:     func(map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
	})
	for _, fixture := range []struct {
		server  string
		tool    string
		noSpawn bool
	}{
		{server: "tasks", tool: "get"},
		{server: "tasks", tool: "list"},
		{server: "crm", tool: "lookup"},
		{server: "host", tool: "admin", noSpawn: true},
	} {
		fullName := fixture.server + "_" + fixture.tool
		def := mcpToolDef{Name: fixture.tool, Description: "Capability " + fullName, InputSchema: map[string]any{"type": "object"}}
		parent.registry.Register(&ToolDef{
			Name: fullName, Description: def.Description, InputSchema: def.InputSchema,
			MCP: true, MCPServer: fixture.server, MCPLocalName: fixture.tool,
			Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
		})
		parent.toolIndex.Add(fixture.server, []mcpToolDef{def}, fixture.noSpawn)
	}
}

func TestSpawnOmittedCapabilitiesInheritsParentCeiling(t *testing.T) {
	parent := newTestThinkerFull()
	parent.registry = NewToolRegistry("test")
	parent.toolIndex = NewToolIndex()
	registerDelegationFixture(parent)
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.Stop()
	})

	if err := parent.threads.SpawnWithOpts(
		"inherited", "Inspect tasks and CRM records.", nil,
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["inherited"]
	if thread == nil || !thread.InheritCapabilities {
		t.Fatalf("thread did not record inherited capabilities: %#v", thread)
	}
	for _, name := range []string{"local_inspect", "spawn", "send", "done", "search_tools"} {
		if !thread.Tools[name] {
			t.Errorf("inherited thread missing %q: %v", name, toolSetToSlice(thread.Tools))
		}
	}
	if !slices.Contains(thread.MCPNames, "tasks") || !slices.Contains(thread.MCPNames, "crm") {
		t.Fatalf("inherited MCP scopes = %v", thread.MCPNames)
	}
	if slices.Contains(thread.MCPNames, "host") || thread.Tools["host_admin"] {
		t.Fatalf("no_spawn capability leaked: tools=%v mcp=%v", thread.Tools, thread.MCPNames)
	}
	if thread.Children == nil {
		t.Fatal("spawn capability was not inherited within the depth limit")
	}
	state, err := parent.threads.PersistentState("inherited")
	if err != nil || !state.InheritCapabilities {
		t.Fatalf("persistent inheritance marker: state=%#v err=%v", state, err)
	}
}

func TestExplicitLegacySpawnProfileStillNarrows(t *testing.T) {
	parent := newTestThinkerFull()
	parent.registry = NewToolRegistry("test")
	parent.toolIndex = NewToolIndex()
	registerDelegationFixture(parent)
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.Stop()
	})

	emptyMCP := []string{}
	if err := parent.threads.SpawnWithOpts(
		"strict", "Use one local capability.", []string{"local_inspect"},
		SpawnOpts{DeferRun: true, ParentID: "main", MCPNames: emptyMCP},
	); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["strict"]
	if thread.InheritCapabilities || !thread.Tools["local_inspect"] || thread.Tools["spawn"] || len(thread.MCPNames) != 0 {
		t.Fatalf("legacy strict profile widened: inherited=%v tools=%v mcp=%v", thread.InheritCapabilities, thread.Tools, thread.MCPNames)
	}
}

func TestNestedSpawnCannotWidenParentCapabilityCeiling(t *testing.T) {
	parent := newTestThinkerFull()
	parent.registry = NewToolRegistry("test")
	parent.toolIndex = NewToolIndex()
	registerDelegationFixture(parent)
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.Stop()
	})

	if err := parent.threads.SpawnWithOpts(
		"lead", "Coordinate task readers.", []string{"spawn"},
		SpawnOpts{DeferRun: true, ParentID: "main", MCPNames: []string{"tasks"}},
	); err != nil {
		t.Fatal(err)
	}
	lead := parent.threads.threads["lead"]
	if lead == nil || lead.Children == nil {
		t.Fatal("lead was not spawn-capable")
	}
	if err := lead.Children.SpawnWithOpts(
		"reader", "Read assigned tasks.", nil,
		SpawnOpts{DeferRun: true, ParentID: "lead", Depth: 1},
	); err != nil {
		t.Fatal(err)
	}
	reader := lead.Children.threads["reader"]
	if !reader.InheritCapabilities || !slices.Equal(reader.MCPNames, []string{"tasks"}) {
		t.Fatalf("nested inheritance = inherited:%v mcp:%v", reader.InheritCapabilities, reader.MCPNames)
	}

	err := lead.Children.SpawnWithOpts(
		"escalated-server", "Reach CRM.", nil,
		SpawnOpts{DeferRun: true, ParentID: "lead", Depth: 1, MCPNames: []string{"crm"}},
	)
	if err == nil || !strings.Contains(err.Error(), "capability_not_delegable") {
		t.Fatalf("server escalation error = %v", err)
	}
	err = lead.Children.SpawnWithOpts(
		"escalated-tool", "Reach CRM.", []string{"crm_lookup"},
		SpawnOpts{DeferRun: true, ParentID: "lead", Depth: 1},
	)
	if err == nil || !strings.Contains(err.Error(), "capability_not_delegable") {
		t.Fatalf("tool escalation error = %v", err)
	}
	if _, err := lead.Children.UpdateWithOpts(
		"reader", "", "", []string{"crm_lookup"},
		ThreadUpdateOptions{ReplaceTools: true},
	); err == nil || !strings.Contains(err.Error(), "capability_not_delegable") {
		t.Fatalf("update escalation error = %v", err)
	}
	if err := lead.Children.SpawnWithOpts(
		"exact-reader", "List tasks.", []string{"tasks_list"},
		SpawnOpts{DeferRun: true, ParentID: "lead", Depth: 1},
	); err != nil {
		t.Fatalf("exact subset of parent MCP scope was rejected: %v", err)
	}
}

func TestAPIThreadCapabilityInheritanceAndExplicitEmptyCompatibility(t *testing.T) {
	api, parent, _ := newPersistentThreadTestAPI(t)
	registerDelegationFixture(parent)

	inherited := postThreadForTest(t, api, "api-inherited", map[string]any{
		"directive": "Inspect tasks without a manually supplied tool profile.",
	})
	if inherited.Code != http.StatusOK {
		t.Fatalf("inherited API spawn = %d: %s", inherited.Code, inherited.Body.String())
	}
	inheritedState, err := parent.threads.PersistentState("api-inherited")
	if err != nil || !inheritedState.InheritCapabilities || !slices.Contains(inheritedState.MCPNames, "tasks") || !slices.Contains(inheritedState.Tools, "local_inspect") {
		t.Fatalf("API omitted-profile state=%#v err=%v", inheritedState, err)
	}

	strict := postThreadForTest(t, api, "api-strict-empty", map[string]any{
		"directive": "Remain capability-minimal.",
		"tools":     []string{},
		"mcp":       []string{},
	})
	if strict.Code != http.StatusOK {
		t.Fatalf("strict API spawn = %d: %s", strict.Code, strict.Body.String())
	}
	strictState, err := parent.threads.PersistentState("api-strict-empty")
	if err != nil || strictState.InheritCapabilities || len(strictState.MCPNames) != 0 || slices.Contains(strictState.Tools, "local_inspect") {
		t.Fatalf("API explicit-empty state=%#v err=%v", strictState, err)
	}
}
