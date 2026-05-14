package core

import (
	"reflect"
	"testing"
)

// mkTool is a terse constructor for the index test fixtures.
func mkTool(name, desc string) mcpToolDef {
	return mcpToolDef{Name: name, Description: desc, InputSchema: map[string]any{"type": "object"}}
}

func TestToolIndex_AddGetSearch(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("storage", []mcpToolDef{
		mkTool("files_upload", "Upload a file to a folder"),
		mkTool("files_list", "List files in a folder"),
	}, false)
	ix.Add("social", []mcpToolDef{
		mkTool("post_slack", "Post a message to a Slack channel"),
	}, false)

	// Get resolves the fully-qualified name.
	e, ok := ix.Get("storage_files_upload")
	if !ok {
		t.Fatal("Get(storage_files_upload) miss")
	}
	if e.Server != "storage" {
		t.Errorf("entry server = %q, want storage", e.Server)
	}

	// Search hits the right tool. "upload" is in storage_files_upload's
	// name + description, nowhere in social.
	hits := ix.Search("upload file", 5, true)
	if len(hits) == 0 {
		t.Fatal("Search(upload file) returned nothing")
	}
	if hits[0].Name != "storage_files_upload" {
		t.Errorf("top hit = %q, want storage_files_upload", hits[0].Name)
	}

	// "slack" only matches social_post_slack.
	hits = ix.Search("slack", 5, true)
	if len(hits) != 1 || hits[0].Name != "social_post_slack" {
		t.Errorf("Search(slack) = %v, want [social_post_slack]", names(hits))
	}
}

func TestToolIndex_NoSpawnFilter(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("apteva-server", []mcpToolDef{
		mkTool("create_agent", "Create a new agent instance"),
	}, true) // no_spawn
	ix.Add("storage", []mcpToolDef{
		mkTool("files_upload", "Upload a file to create it in storage"),
	}, false)

	// allowNoSpawn=true (main): both servers' tools are searchable.
	all := ix.Search("create", 5, true)
	if len(all) != 2 {
		t.Errorf("main search(create) = %v, want both tools", names(all))
	}

	// allowNoSpawn=false (sub-thread): the no_spawn server is invisible.
	sub := ix.Search("create", 5, false)
	for _, h := range sub {
		if h.Server == "apteva-server" {
			t.Errorf("sub-thread search leaked no_spawn tool %q", h.Name)
		}
	}
	if len(sub) != 1 || sub[0].Name != "storage_files_upload" {
		t.Errorf("sub search(create) = %v, want [storage_files_upload]", names(sub))
	}
}

func TestToolIndex_ReAddReplaces(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("storage", []mcpToolDef{mkTool("old_tool", "the old surface")}, false)
	// Reconnect / hot-reload: same server, different tools.
	ix.Add("storage", []mcpToolDef{mkTool("new_tool", "the new surface")}, false)

	if _, ok := ix.Get("storage_old_tool"); ok {
		t.Error("re-Add did not evict storage_old_tool")
	}
	if _, ok := ix.Get("storage_new_tool"); !ok {
		t.Error("re-Add did not register storage_new_tool")
	}
	if got := ix.ToolsForServer("storage"); !reflect.DeepEqual(got, []string{"storage_new_tool"}) {
		t.Errorf("ToolsForServer after re-Add = %v, want [storage_new_tool]", got)
	}
}

func TestToolIndex_Remove(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("storage", []mcpToolDef{mkTool("files_list", "list files")}, false)
	ix.Add("social", []mcpToolDef{mkTool("post_slack", "post to slack")}, false)

	ix.Remove("storage")
	if _, ok := ix.Get("storage_files_list"); ok {
		t.Error("Remove(storage) left storage_files_list behind")
	}
	if _, ok := ix.Get("social_post_slack"); !ok {
		t.Error("Remove(storage) wrongly dropped social's tools")
	}
	if got := ix.Servers(); !reflect.DeepEqual(got, []string{"social"}) {
		t.Errorf("Servers() after Remove = %v, want [social]", got)
	}
}

func TestToolIndex_ToolsForServerSorted(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("storage", []mcpToolDef{
		mkTool("zeta", "z"), mkTool("alpha", "a"), mkTool("mike", "m"),
	}, false)
	got := ix.ToolsForServer("storage")
	want := []string{"storage_alpha", "storage_mike", "storage_zeta"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ToolsForServer = %v, want sorted %v", got, want)
	}
	if ix.ToolsForServer("nonexistent") != nil {
		t.Error("ToolsForServer(nonexistent) should be nil")
	}
}

// TestToolIndex_ScoringNameBoost pins the ranking rule: a query term
// that hits a tool's NAME outranks one that only hits a description.
// Without the name boost in scoreEntry, "report" would tie and the
// stable sort would fall back to alphabetical, surfacing the wrong tool.
func TestToolIndex_ScoringNameBoost(t *testing.T) {
	ix := NewToolIndex()
	ix.Add("srv", []mcpToolDef{
		// "report" only in the description.
		mkTool("aaa_generate", "Build a financial report from raw numbers"),
		// "report" in the name itself — should win.
		mkTool("zzz_report", "Produce output"),
	}, false)

	hits := ix.Search("report", 5, true)
	if len(hits) < 1 {
		t.Fatal("Search(report) returned nothing")
	}
	if hits[0].Name != "srv_zzz_report" {
		t.Errorf("top hit = %q, want srv_zzz_report (name hit must outrank description-only hit)", hits[0].Name)
	}
}

func TestToolIndex_SearchEmptyAndCaps(t *testing.T) {
	ix := NewToolIndex()
	for _, n := range []string{"a", "b", "c", "d", "e", "f"} {
		ix.Add("srv"+n, []mcpToolDef{mkTool("match_tool", "matchable surface")}, false)
	}
	// k caps the result count.
	if got := ix.Search("matchable", 3, true); len(got) != 3 {
		t.Errorf("Search with k=3 returned %d hits, want 3", len(got))
	}
	// No query terms → no results, no panic.
	if got := ix.Search("", 5, true); got != nil {
		t.Errorf("Search(\"\") = %v, want nil", names(got))
	}
	// Query with only stopwords → no results.
	if got := ix.Search("the and for", 5, true); len(got) != 0 {
		t.Errorf("Search(stopwords-only) = %v, want empty", names(got))
	}
	// k<=0 → nil.
	if got := ix.Search("matchable", 0, true); got != nil {
		t.Errorf("Search with k=0 = %v, want nil", names(got))
	}
}

func TestToolIndex_Tokenize(t *testing.T) {
	// indexTokens: lowercased, split on non-alphanumeric, <2-char and
	// stopwords dropped, term-frequency counted.
	got := indexTokens("Upload-a FILE for the user, use_it")
	// "a" dropped (<2 char), "for"/"the"/"use" dropped (stopwords),
	// "upload"/"file"/"user"/"it" kept... "it" is 2 chars, kept.
	for _, want := range []string{"upload", "file", "user", "it"} {
		if got[want] == 0 {
			t.Errorf("indexTokens missing expected token %q (got %v)", want, got)
		}
	}
	for _, drop := range []string{"a", "for", "the", "use"} {
		if got[drop] != 0 {
			t.Errorf("indexTokens kept token %q that should be dropped", drop)
		}
	}

	// Term frequency: a repeated word counts up.
	tf := indexTokens("post post post")
	if tf["post"] != 3 {
		t.Errorf("indexTokens term-freq for repeated word = %d, want 3", tf["post"])
	}

	// Query tokens are deduplicated (TF collapsed to a set).
	q := indexQueryTokens("post post slack")
	if len(q) != 2 {
		t.Errorf("indexQueryTokens(post post slack) = %v, want 2 unique terms", q)
	}
}

// TestToolIndex_NilSafe verifies the nil-receiver guards — call sites
// pass a *ToolIndex that can legitimately be nil (a thinker with no
// MCPs attached), and none of these should panic.
func TestToolIndex_NilSafe(t *testing.T) {
	var ix *ToolIndex
	ix.Add("srv", []mcpToolDef{mkTool("t", "d")}, false) // no-op
	ix.Remove("srv")                                     // no-op
	if _, ok := ix.Get("x"); ok {
		t.Error("nil index Get should miss")
	}
	if ix.Search("q", 5, true) != nil {
		t.Error("nil index Search should be nil")
	}
	if ix.ToolsForServer("srv") != nil {
		t.Error("nil index ToolsForServer should be nil")
	}
	if ix.Servers() != nil {
		t.Error("nil index Servers should be nil")
	}
	if computeMCPCatalog(nil) != nil {
		t.Error("computeMCPCatalog(nil) should be nil")
	}
}

// names extracts the tool names from a hit slice for terse assertions.
func names(hits []IndexEntry) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Name
	}
	return out
}
