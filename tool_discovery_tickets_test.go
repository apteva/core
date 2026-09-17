package core

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// Original terse descriptions from the incident, deliberately not the improved
// app descriptions: assignment must be discoverable from the schema alone.
func ticketDiscoveryCatalog() []mcpToolDef {
	tools := []mcpToolDef{
		mkTool("tickets_get", "Fetch one ticket with comments, attachments, links, and chronological history."),
		mkTool("tickets_update", "Patch ticket fields and record every changed value."),
		mkTool("tickets_list", "Search and filter tickets."),
		mkTool("tickets_set_status", "Move a ticket through its workflow and record the transition."),
		mkTool("tickets_add_internal_note", "Add an internal team/agent note hidden from the client."),
		mkTool("tickets_history", "Read append-only ticket history."),
		mkTool("tickets_create", "Create a client feedback or support ticket."),
		mkTool("ticket_areas_list", "List configurable feedback areas."),
		mkTool("ticket_areas_create", "Create a feedback area."),
		mkTool("ticket_areas_update", "Update or archive a feedback area."),
	}
	for i := range tools {
		tools[i].InputSchema = map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer"}}, "required": []string{"id"}}
		if tools[i].Name == "tickets_update" {
			props := tools[i].InputSchema["properties"].(map[string]any)
			for _, name := range []string{"assignee_kind", "assignee_ref", "assignee_name"} {
				props[name] = map[string]any{"type": "string"}
			}
		}
	}
	return tools
}

func ticketDiscoveryThinker(t *testing.T) *Thinker {
	t.Helper()
	ix := NewToolIndex()
	tools := ticketDiscoveryCatalog()
	ix.Add("tickets", tools, false)
	r := NewToolRegistry("")
	registerTestMCPTools(r, "tickets", tools)
	return &Thinker{threadID: "main", toolIndex: ix, registry: r, activeTools: map[string]bool{}}
}

func TestDiscoveryTicketsIncidentQueries(t *testing.T) {
	th := ticketDiscoveryThinker(t)
	for _, tc := range []struct {
		query string
		want  []string
		k     int
	}{
		{"tickets_get tickets_update tickets note comment status assign", []string{"tickets_tickets_get", "tickets_tickets_update"}, 2},
		{"tickets_get", []string{"tickets_tickets_get"}, 1},
		{"tickets_get read ticket details", []string{"tickets_tickets_get"}, 1},
		{"tickets_tickets_get", []string{"tickets_tickets_get"}, 1},
		{"tickets_update assign ticket to agent assignee field", []string{"tickets_tickets_update"}, 1},
		{"tickets assign assignee ticket", []string{"tickets_tickets_update"}, 1},
		{"tickets view ticket single ticket contents", []string{"tickets_tickets_get"}, 1},
		{"update ticket status assignee internal note", []string{"tickets_tickets_update", "tickets_tickets_set_status", "tickets_tickets_add_internal_note"}, 5},
	} {
		t.Run(tc.query, func(t *testing.T) {
			got := names(th.toolIndex.Search(tc.query, tc.k, true))
			for _, want := range tc.want {
				found := false
				for _, name := range got {
					found = found || name == want
				}
				if !found {
					t.Errorf("got %v, missing %s", got, want)
				}
			}
		})
	}
	th.applyPreload(3, "Read ticket details and assign the ticket to an agent")
	for _, name := range []string{"tickets_tickets_get", "tickets_tickets_update"} {
		if !th.activeTools[name] {
			t.Errorf("preload did not activate %s: %v", name, th.activeTools)
		}
	}
}

func TestDiscoverySuggestionsAreDistinctAndScoped(t *testing.T) {
	th := ticketDiscoveryThinker(t)
	var res searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(th, map[string]string{"query": "tickets_missing assign ticket to assignee", "k": "1"}, true)), &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Hits) != 0 || !reflect.DeepEqual(res.Unresolved, []string{"tickets_missing"}) || len(res.Suggestions) != 1 || res.Suggestions[0].Name != "tickets_tickets_update" || res.Suggestions[0].Match != "descriptive_suggestion" || !strings.Contains(res.Note, "not exact matches") {
		t.Fatalf("unsafe or missing distinction: %+v", res)
	}
	if !reflect.DeepEqual(res.Loaded, []string{"tickets_tickets_update"}) {
		t.Fatal("suggested schema not available in same search", res)
	}
	// Preload cannot hide the unresolved-name warning by silently activating it.
	fresh := ticketDiscoveryThinker(t)
	fresh.applyPreload(5, "tickets_missing assign ticket to assignee")
	if len(fresh.activeTools) != 0 {
		t.Fatal("preload silently substituted unknown operation")
	}
	for _, query := range []string{"tickets_missing", "tickets_tickets_update assign ticket", "tickets_update assign ticket"} {
		fresh.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
		got := fresh.searchAuthorizedToolsDetailed(query, 5, true)
		if len(got.Hits)+len(got.Suggestions) != 0 {
			t.Fatalf("unauthorized/unknown exact lookup substituted: %s %+v", query, got)
		}
	}
	fresh.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
	got := fresh.searchAuthorizedToolsDetailed("tickets_missing read ticket", 5, true)
	for _, e := range got.Suggestions {
		if e.Name != "tickets_tickets_get" {
			t.Fatal("suggestion leaked permission", e.Name)
		}
	}
}

func TestDiscoveryLocalAliasCollisionsAndReconnect(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("a", []mcpToolDef{mkTool("tickets_get", "Read ticket")}, false)
	ix.Add("b", []mcpToolDef{mkTool("tickets_get", "Read ticket")}, true)
	got := ix.searchDetailed("tickets_get", 5, false, nil)
	if len(got.Hits)+len(got.Suggestions) != 0 || !reflect.DeepEqual(got.Ambiguous["tickets_get"], []string{"a_tickets_get"}) {
		t.Fatal("ambiguous alias resolved or leaked hidden tool", got)
	}
	ix.Remove("b")
	if got := names(ix.Search("tickets_get", 1, false)); !reflect.DeepEqual(got, []string{"a_tickets_get"}) {
		t.Fatal(got)
	}
	// A real canonical identity takes precedence over a local shorthand.
	ix.Add("tickets", []mcpToolDef{mkTool("get", "Read another ticket")}, false)
	if got := names(ix.Search("tickets_get", 1, false)); !reflect.DeepEqual(got, []string{"tickets_get"}) {
		t.Fatal(got)
	}
	if err := ix.RegisterAlias("tickets_get", "a_tickets_get"); err == nil {
		t.Fatal("alias shadowed canonical name")
	}
	ix.Add("a", []mcpToolDef{mkTool("other", "Unrelated")}, false)
	ix.Remove("tickets")
	if got := ix.Search("tickets_get", 1, true); len(got) != 0 {
		t.Fatal("stale local alias", got)
	}
}

func TestDiscoverySchemaMetadataAndBounds(t *testing.T) {
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"opaque": map[string]any{"description": "Escalate the case to a supervisor", "enum": []any{"urgent", "routine"}},
		"nested": map[string]any{"items": map[string]any{"oneOf": []any{map[string]any{"properties": map[string]any{"delivery_region": map[string]any{"enum": []string{"antarctica"}}}}}}},
	}, "default": "secret_default", "examples": []any{"secret_example"}}
	ix := NewToolIndex()
	tool := mkTool("opaque", "Perform operation")
	tool.InputSchema = schema
	ix.Add("support", []mcpToolDef{tool}, false)
	for _, q := range []string{"escalate supervisor", "urgent", "delivery region", "antarctica"} {
		if got := names(ix.Search(q, 1, true)); !reflect.DeepEqual(got, []string{"support_opaque"}) {
			t.Errorf("%q: %v", q, got)
		}
	}
	tokens := schemaDiscoveryTokens(schema)
	if tokens["secret"] != 0 {
		t.Fatal("indexed example/default values")
	}
	schema["items"] = schema
	if len(schemaDiscoveryTokens(schema)) == 0 {
		t.Fatal("cyclic schema lost searchable metadata")
	}
	huge := map[string]any{"description": strings.Repeat("x ", 40000), "properties": map[string]any{"tail": map[string]any{"description": "beyondbudget"}}}
	if schemaDiscoveryTokens(huge)["beyondbudget"] != 0 {
		t.Fatal("schema byte budget not enforced")
	}
}
