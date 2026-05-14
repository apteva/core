package core

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// indexedThinker returns a bare Thinker with a populated ToolIndex —
// enough surface to exercise runSearchTools / applyPreload
// without standing up the full agent loop.
func indexedThinker(threadID string) *Thinker {
	ix := NewToolIndex()
	ix.Add("storage", []mcpToolDef{
		mkTool("files_upload", "Upload a file to a folder in storage"),
		mkTool("files_list", "List files in a storage folder"),
	}, false)
	ix.Add("apteva-server", []mcpToolDef{
		mkTool("create_agent", "Create a new agent to upload and manage things"),
	}, true) // no_spawn
	return &Thinker{
		threadID:    threadID,
		toolIndex:   ix,
		activeTools: map[string]bool{},
		messages:    []Message{{Role: "system", Content: "sys"}},
	}
}

func TestRunSearchTools_PopulatesActiveTools(t *testing.T) {
	th := indexedThinker("main")
	out := runSearchTools(th, map[string]string{"query": "upload file"}, true)

	var res searchToolsResult
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("result is not valid JSON: %v\n%s", err, out)
	}
	if len(res.Hits) == 0 {
		t.Fatalf("expected hits for 'upload file', got none: %s", out)
	}
	// Every hit must also have been activated on the thinker.
	for _, h := range res.Hits {
		if !th.activeTools[h.Name] {
			t.Errorf("hit %q not added to activeTools", h.Name)
		}
	}
	// Loaded mirrors the hit names — it's the signal the runtime uses.
	if len(res.Loaded) != len(res.Hits) {
		t.Errorf("loaded (%d) and hits (%d) length mismatch", len(res.Loaded), len(res.Hits))
	}
	if !th.activeTools["storage_files_upload"] {
		t.Error("expected storage_files_upload to be activated")
	}
}

// TestRunSearchTools_NoSpawnExclusion is the security-relevant case:
// a sub-thread (allowNoSpawn=false) searching a term that matches a
// no_spawn server's tool must NOT discover or activate it.
func TestRunSearchTools_NoSpawnExclusion(t *testing.T) {
	th := indexedThinker("worker-1")

	// "upload" matches both storage_files_upload AND
	// apteva-server_create_agent (its description says "upload").
	out := runSearchTools(th, map[string]string{"query": "upload"}, false /* sub-thread */)

	var res searchToolsResult
	json.Unmarshal([]byte(out), &res)
	for _, h := range res.Hits {
		if h.Server == "apteva-server" {
			t.Errorf("sub-thread search leaked no_spawn tool %q", h.Name)
		}
	}
	if th.activeTools["apteva-server_create_agent"] {
		t.Error("sub-thread activated a no_spawn tool — privilege escalation")
	}
	// Main, by contrast, sees it.
	main := indexedThinker("main")
	mainOut := runSearchTools(main, map[string]string{"query": "upload"}, true)
	if !strings.Contains(mainOut, "apteva-server_create_agent") {
		t.Errorf("main search should surface apteva-server_create_agent, got: %s", mainOut)
	}
}

func TestRunSearchTools_EmptyResultNote(t *testing.T) {
	th := indexedThinker("main")
	out := runSearchTools(th, map[string]string{"query": "xyzzy nonexistent capability"}, true)

	var res searchToolsResult
	json.Unmarshal([]byte(out), &res)
	if len(res.Hits) != 0 {
		t.Fatalf("expected no hits, got %v", res.Hits)
	}
	if res.Note == "" {
		t.Error("empty-result search should carry a note listing attached servers")
	}
	// The note should mention at least one real attached server so the
	// LLM can refine instead of guessing.
	if !strings.Contains(res.Note, "storage") {
		t.Errorf("note should mention attached servers, got %q", res.Note)
	}
}

func TestRunSearchTools_MissingQuery(t *testing.T) {
	th := indexedThinker("main")
	out := runSearchTools(th, map[string]string{}, true)
	if !strings.Contains(out, "error") {
		t.Errorf("missing query should return an error JSON, got: %s", out)
	}
	if len(th.activeTools) != 0 {
		t.Error("failed search must not mutate activeTools")
	}
}

func TestRunSearchTools_NilIndex(t *testing.T) {
	th := &Thinker{threadID: "main", activeTools: map[string]bool{}}
	out := runSearchTools(th, map[string]string{"query": "anything"}, true)
	if !strings.Contains(out, "error") {
		t.Errorf("search with nil index should return an error, got: %s", out)
	}
}

func TestRunSearchTools_KClamp(t *testing.T) {
	th := indexedThinker("main")
	// k beyond the cap (20) must not panic and must not over-return.
	out := runSearchTools(th, map[string]string{"query": "file storage folder", "k": "999"}, true)
	var res searchToolsResult
	if err := json.Unmarshal([]byte(out), &res); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if len(res.Hits) > 20 {
		t.Errorf("k=999 returned %d hits, cap is 20", len(res.Hits))
	}
}

func TestApplyPreload(t *testing.T) {
	// applyPreload queries the DIRECTIVE only (not the user turn) — that's
	// the cache fix: a stable query → stable hits → the active set seeds
	// once and then holds, so the tools array stops churning.
	th := indexedThinker("main")
	th.directive = "Keep the storage folder tidy — upload new files as they arrive and list what's there."

	th.applyPreload(5)
	if len(th.activeTools) == 0 {
		t.Fatal("applyPreload activated nothing for a storage-related directive")
	}
	if !th.activeTools["storage_files_upload"] {
		t.Errorf("activeTools %v should include storage_files_upload", keysOf(th.activeTools))
	}
	if _, ok := th.activeToolAge["storage_files_upload"]; !ok {
		t.Error("applyPreload did not record activeToolAge for the surfaced tool")
	}

	// Idempotent: same directive → same hits → already active → the set
	// neither grows nor shrinks. This is what lets the tools array go
	// stable and the prompt cache start hitting.
	before := len(th.activeTools)
	th.applyPreload(5)
	if len(th.activeTools) != before {
		t.Errorf("applyPreload not idempotent: %d → %d on a repeat call with the same directive", before, len(th.activeTools))
	}

	// Sub-thread preload excludes no_spawn servers.
	sub := indexedThinker("worker-9")
	sub.directive = "Create and upload things for the team."
	sub.applyPreload(5)
	if sub.activeTools["apteva-server_create_agent"] {
		t.Error("sub-thread preload leaked a no_spawn tool into activeTools")
	}

	// No directive → no-op (the user turn is deliberately NOT consulted).
	bare := indexedThinker("main")
	bare.messages = append(bare.messages, Message{Role: "user", Content: "I need to upload a file"})
	bare.applyPreload(5)
	if len(bare.activeTools) != 0 {
		t.Errorf("applyPreload with no directive activated %v — must ignore the user turn", keysOf(bare.activeTools))
	}

	// No index → no-op, no panic.
	noIx := &Thinker{threadID: "main", directive: "upload files"}
	noIx.applyPreload(5)
	if len(noIx.activeTools) != 0 {
		t.Error("applyPreload with no index should be a no-op")
	}
}

// TestEvictActiveToolsLRU pins the cache-bounding behaviour: the sticky
// set grows freely up to the cap, then drops the least-recently-touched
// tools in a batch (to ~70% of cap) so the next turn doesn't re-evict.
func TestEvictActiveToolsLRU(t *testing.T) {
	th := &Thinker{threadID: "main", activeTools: map[string]bool{}, activeToolAge: map[string]int{}}
	// Touch 50 tools across iterations 1..50 — tool-N touched at iter N.
	for i := 1; i <= 50; i++ {
		th.iteration = i
		th.touchActiveTool(fmt.Sprintf("srv_tool%02d", i))
	}
	if len(th.activeTools) != 50 {
		t.Fatalf("setup: expected 50 active tools, got %d", len(th.activeTools))
	}

	th.evictActiveToolsLRU(40)
	// 50 > 40 → evict down to 70% of 40 = 28.
	if len(th.activeTools) != 28 {
		t.Errorf("after eviction expected 28 tools, got %d", len(th.activeTools))
	}
	// The OLDEST (lowest iteration) must be gone; the NEWEST kept.
	if th.activeTools["srv_tool01"] {
		t.Error("evictActiveToolsLRU kept the oldest tool (tool01)")
	}
	if !th.activeTools["srv_tool50"] {
		t.Error("evictActiveToolsLRU dropped the newest tool (tool50)")
	}
	// activeToolAge stays in lockstep with activeTools.
	if len(th.activeToolAge) != len(th.activeTools) {
		t.Errorf("activeToolAge (%d) out of sync with activeTools (%d)", len(th.activeToolAge), len(th.activeTools))
	}

	// Under the cap → no-op.
	before := len(th.activeTools)
	th.evictActiveToolsLRU(40)
	if len(th.activeTools) != before {
		t.Errorf("evictActiveToolsLRU under cap should be a no-op, %d → %d", before, len(th.activeTools))
	}
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// TestToolVisible_Matrix exercises every branch of the visibility
// rule that NativeTools delegates to. This is the load-bearing logic
// of the whole refactor: get it wrong and either MCP tools leak into
// every prompt (token bloat) or never appear at all (agent is blind).
func TestToolVisible_Matrix(t *testing.T) {
	core := &ToolDef{Name: "pace", Core: true}
	mcp := &ToolDef{Name: "storage_files_upload", MCP: true, MCPServer: "storage"}
	threadOnly := &ToolDef{Name: "reply", ThreadOnly: true}
	systemOnly := &ToolDef{Name: "consolidate", SystemOnly: true}

	cases := []struct {
		name      string
		tool      *ToolDef
		allowlist map[string]bool
		active    map[string]bool
		want      bool
	}{
		// Main thread (nil allowlist).
		{"main sees core tool", core, nil, nil, true},
		{"main hides MCP tool by default", mcp, nil, nil, false},
		{"main sees MCP tool once active", mcp, nil, map[string]bool{"storage_files_upload": true}, true},
		{"main hides thread-only tool", threadOnly, nil, nil, false},
		{"main hides system-only tool", systemOnly, nil, nil, false},

		// Sub-thread (allowlist set).
		{"sub sees allowlisted tool", core, map[string]bool{"pace": true}, nil, true},
		{"sub hides non-allowlisted tool", core, map[string]bool{"other": true}, nil, false},
		{"sub sees MCP tool via allowlist", mcp, map[string]bool{"storage_files_upload": true}, nil, true},
		{"sub sees MCP tool via active even if not in allowlist", mcp, map[string]bool{"pace": true}, map[string]bool{"storage_files_upload": true}, true},
		{"sub hides system-only even if allowlisted", systemOnly, map[string]bool{"consolidate": true}, nil, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := toolVisible(c.tool, c.allowlist, c.active); got != c.want {
				t.Errorf("toolVisible(%s) = %v, want %v", c.tool.Name, got, c.want)
			}
		})
	}
}

// TestNativeTools_ActiveOverridesMCPHidden is the end-to-end check on
// the registry: an MCP tool is absent from main's list until its name
// lands in the active set, then it appears.
func TestNativeTools_ActiveOverridesMCPHidden(t *testing.T) {
	tr := NewToolRegistry("test")
	tr.Register(&ToolDef{
		Name: "storage_files_upload", MCP: true, MCPServer: "storage",
		Description: "Upload a file", InputSchema: map[string]any{"type": "object"},
	})

	// Main, no active set — MCP tool hidden.
	before := nativeNames(tr.NativeTools(nil, nil))
	if before["storage_files_upload"] {
		t.Error("MCP tool should be hidden from main with empty active set")
	}

	// Main, tool activated — now visible.
	after := nativeNames(tr.NativeTools(nil, map[string]bool{"storage_files_upload": true}))
	if !after["storage_files_upload"] {
		t.Error("MCP tool should be visible once in active set")
	}
	// Core scaffolding (search_tools, pace, …) is always present.
	if !after["search_tools"] {
		t.Error("search_tools scaffolding tool missing from main's list")
	}
}

func nativeNames(nts []NativeTool) map[string]bool {
	out := map[string]bool{}
	for _, nt := range nts {
		out[nt.Name] = true
	}
	return out
}
