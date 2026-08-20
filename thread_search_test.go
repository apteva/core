package core

import (
	"fmt"
	"strings"
	"sync"
	"testing"
)

// newSearchManager builds a ThreadManager tree without booting real Thinkers.
// ThreadManager.List calls t.Thinker.status(), which reads an atomic.Value
// snapshot, so a zero Thinker is sufficient and keeps the test hermetic.
func newSearchManager(t *testing.T, infos ...ThreadInfo) *ThreadManager {
	t.Helper()
	tm := &ThreadManager{threads: make(map[string]*Thread)}
	for _, info := range infos {
		tm.threads[info.ID] = &Thread{
			ID:        info.ID,
			Name:      info.Name,
			System:    info.System,
			ParentID:  info.ParentID,
			Depth:     info.Depth,
			Directive: info.Directive,
			MCPNames:  info.MCPNames,
			Tools:     toolSetFromSlice(info.Tools),
			Thinker:   &Thinker{},
		}
	}
	return tm
}

func toolSetFromSlice(names []string) map[string]bool {
	out := make(map[string]bool, len(names))
	for _, n := range names {
		out[n] = true
	}
	return out
}

func attachChildren(parent *ThreadManager, parentID string, children *ThreadManager) {
	parent.threads[parentID].Children = children
}

func TestListThreadsFiltersAcrossSearchableFields(t *testing.T) {
	tm := newSearchManager(t,
		ThreadInfo{ID: "chat-0001", Name: "Acme Billing", Directive: "resolve invoice disputes"},
		ThreadInfo{ID: "chat-0002", Directive: "handle shipping questions", Tools: []string{"web_fetch"}},
		ThreadInfo{ID: "chat-0003", Directive: "unrelated", MCPNames: []string{"stripe"}},
	)

	cases := map[string]string{
		"billing":  "chat-0001", // name
		"invoice":  "chat-0001", // directive
		"shipping": "chat-0002", // directive
		"web_":     "chat-0002", // tool name
		"stripe":   "chat-0003", // mcp scope
	}
	for filter, wantID := range cases {
		got := runListThreads(tm, map[string]string{"filter": filter})
		if !strings.Contains(got, wantID) {
			t.Errorf("filter %q should match %s; got:\n%s", filter, wantID, got)
		}
	}
}

func TestListThreadsFilterIsCaseInsensitive(t *testing.T) {
	tm := newSearchManager(t, ThreadInfo{ID: "chat-1", Directive: "Handle BILLING disputes"})
	if got := runListThreads(tm, map[string]string{"filter": "billing"}); !strings.Contains(got, "chat-1") {
		t.Errorf("filter must be case-insensitive; got:\n%s", got)
	}
}

func TestListThreadsNaturalKeywordQueryRanksByOverlap(t *testing.T) {
	tm := newSearchManager(t,
		ThreadInfo{ID: "billing-lead", Directive: "Coordinate billing work."},
		ThreadInfo{ID: "acme-payment-owner", Directive: "Own ACME payment-dispute requests."},
		ThreadInfo{ID: "unrelated", Directive: "Own warehouse operations."},
	)
	got := runListThreads(tm, map[string]string{"filter": "ACME payment dispute billing dispute payment"})
	ownerAt := strings.Index(got, "acme-payment-owner")
	leadAt := strings.Index(got, "billing-lead")
	if ownerAt < 0 || leadAt < 0 {
		t.Fatalf("multi-keyword search omitted relevant matches:\n%s", got)
	}
	if ownerAt > leadAt {
		t.Errorf("stronger keyword overlap must rank first:\n%s", got)
	}
	if strings.Contains(got, "unrelated") {
		t.Errorf("zero-overlap thread leaked into filtered result:\n%s", got)
	}
}

// Literal substring search cannot prove semantic non-ownership: an "invoice"
// owner may not match a "billing" query. The receipt must not authorize a
// duplicate spawn from that negative alone.
func TestListThreadsNoMatchDoesNotAuthorizeSpawn(t *testing.T) {
	tm := newSearchManager(t, ThreadInfo{ID: "chat-1", Directive: "shipping"})
	got := runListThreads(tm, map[string]string{"filter": "billing"})
	if !strings.Contains(got, "0 of 1") {
		t.Errorf("no-match result must state how many were searched; got:\n%s", got)
	}
	if strings.Contains(strings.ToLower(got), "spawning a new one is safe") {
		t.Errorf("substring no-match must not claim spawning is safe; got:\n%s", got)
	}
	if !strings.Contains(got, "does not prove") || !strings.Contains(got, "broaden") {
		t.Errorf("no-match result must explain its bounded meaning; got:\n%s", got)
	}
}

func TestListThreadsNilManagerIsEmptyNotError(t *testing.T) {
	got := runListThreads(nil, map[string]string{})
	if strings.Contains(strings.ToLower(got), "error") {
		t.Errorf("a thread with no children is a valid empty result, not an error; got %q", got)
	}
	if !strings.HasPrefix(got, "0 threads") {
		t.Errorf("expected an empty-result answer, got %q", got)
	}
}

// List covers direct children only. Searching for an existing owner has to see
// grandchildren or it produces duplicate spawns.
func TestListThreadsTreeScopeFindsGrandchildren(t *testing.T) {
	grandchildren := newSearchManager(t, ThreadInfo{ID: "deep-worker", Directive: "billing reconciliation"})
	children := newSearchManager(t, ThreadInfo{ID: "mid-leader", Directive: "team lead"})
	attachChildren(children, "mid-leader", grandchildren)
	root := newSearchManager(t, ThreadInfo{ID: "top-leader", Directive: "top"})
	attachChildren(root, "top-leader", children)

	tree := runListThreads(root, map[string]string{"filter": "billing"})
	if !strings.Contains(tree, "deep-worker") {
		t.Errorf("default tree scope must reach grandchildren; got:\n%s", tree)
	}

	direct := runListThreads(root, map[string]string{"filter": "billing", "scope": "children"})
	if strings.Contains(direct, "deep-worker") {
		t.Errorf("scope=children must not recurse; got:\n%s", direct)
	}
	if !strings.Contains(direct, "direct children only") {
		t.Errorf("result must state the scope it searched; got:\n%s", direct)
	}
}

func TestListTreeExcludesSystemThreadsAndSortsByID(t *testing.T) {
	child := newSearchManager(t, ThreadInfo{ID: "aaa-child"})
	root := newSearchManager(t,
		ThreadInfo{ID: "zzz-leader"},
		ThreadInfo{ID: "unconscious", System: true},
	)
	attachChildren(root, "zzz-leader", child)

	got := root.ListTreeAgentVisible()
	if len(got) != 2 {
		t.Fatalf("expected 2 visible threads, got %d", len(got))
	}
	if got[0].ID != "aaa-child" || got[1].ID != "zzz-leader" {
		t.Errorf("tree listing must sort by id for stable paging, got %v, %v", got[0].ID, got[1].ID)
	}
	for _, info := range got {
		if info.System {
			t.Error("system threads must never surface to the model")
		}
	}
}

func TestListThreadsPaginationIsStableAndReportsRemainder(t *testing.T) {
	infos := make([]ThreadInfo, 0, 60)
	for i := 0; i < 60; i++ {
		infos = append(infos, ThreadInfo{ID: fmt.Sprintf("chat-%03d", i)})
	}
	tm := newSearchManager(t, infos...)

	first := runListThreads(tm, map[string]string{"limit": "10"})
	if !strings.Contains(first, "showing 1-10") {
		t.Errorf("expected a 1-10 window; got:\n%s", first)
	}
	if !strings.Contains(first, "50 more match") {
		t.Errorf("expected the remainder count; got:\n%s", first)
	}
	if !strings.Contains(first, "chat-000") || strings.Contains(first, "chat-059") {
		t.Errorf("first page must hold the lowest ids; got:\n%s", first)
	}

	second := runListThreads(tm, map[string]string{"limit": "10", "offset": "10"})
	if !strings.Contains(second, "showing 11-20") {
		t.Errorf("expected an 11-20 window; got:\n%s", second)
	}
	if strings.Contains(second, "- chat-000 ") {
		t.Error("offset page must not repeat the first page")
	}
}

func TestListThreadsLimitIsCappedNotRejected(t *testing.T) {
	q := parseListThreadsQuery(map[string]string{"limit": "500"})
	if q.limit != listThreadsMaxLimit {
		t.Errorf("oversized limit must clamp to %d, got %d", listThreadsMaxLimit, q.limit)
	}
	// Rejecting a malformed arg costs a whole turn; the intent is unambiguous.
	if q := parseListThreadsQuery(map[string]string{"limit": "abc"}); q.limit != listThreadsDefaultLimit {
		t.Errorf("unparseable limit must fall back to the default, got %d", q.limit)
	}
	if q := parseListThreadsQuery(map[string]string{"limit": "-5"}); q.limit != listThreadsDefaultLimit {
		t.Errorf("negative limit must fall back to the default, got %d", q.limit)
	}
	if q := parseListThreadsQuery(nil); q.scope != "tree" {
		t.Errorf("default scope must be tree, got %q", q.scope)
	}
}

func TestListThreadsOffsetPastEndIsEmptyNotPanic(t *testing.T) {
	tm := newSearchManager(t, ThreadInfo{ID: "only-one"})
	got := runListThreads(tm, map[string]string{"offset": "100"})
	if strings.Contains(got, "only-one") {
		t.Errorf("offset past the end must yield no entries; got:\n%s", got)
	}
}

// A result at or above largeToolResultThresholdBytes is SHA-archived and later
// projected down to a preview, so the roster would silently truncate mid-task.
func TestListThreadsOutputStaysUnderArchiveThreshold(t *testing.T) {
	wide := make([]string, 60)
	for i := range wide {
		wide[i] = fmt.Sprintf("some_long_mcp_tool_name_%02d", i)
	}
	infos := make([]ThreadInfo, 0, listThreadsMaxLimit)
	for i := 0; i < listThreadsMaxLimit; i++ {
		infos = append(infos, ThreadInfo{
			ID:        fmt.Sprintf("wide-%03d", i),
			Directive: strings.Repeat("y", 400),
			Tools:     wide,
		})
	}
	tm := newSearchManager(t, infos...)

	got := runListThreads(tm, map[string]string{"limit": fmt.Sprint(listThreadsMaxLimit)})
	if len(got) >= largeToolResultThresholdBytes {
		t.Errorf("result of %d bytes would be archived and silently truncated (threshold %d)",
			len(got), largeToolResultThresholdBytes)
	}
	if !strings.Contains(got, "output truncated") {
		t.Errorf("a size-truncated result must say so rather than look complete; got tail:\n%s",
			got[max(0, len(got)-200):])
	}
}

func TestListThreadsToolIsRegisteredAndGatedLikeKill(t *testing.T) {
	tr := NewToolRegistry("")

	def := tr.Get("list_threads")
	if def == nil {
		t.Fatal("list_threads must be registered")
	}
	// MainOnly keeps it out of worker prompt docs and out of AllToolNames, so
	// the model cannot try to grant it via spawn(tools=...). Wire visibility
	// is the per-thread allowlist, exactly as for kill/update.
	if !def.MainOnly {
		t.Error("list_threads must be MainOnly, matching kill/update")
	}
	if !def.Core {
		t.Error("list_threads must be Core so it is not treated as a discoverable MCP tool")
	}
	for _, name := range tr.AllToolNames() {
		if name == "list_threads" {
			t.Error("list_threads must not appear in spawn's grantable tool list")
		}
	}
	if !strings.Contains(tr.CoreDocs(true), "list_threads") {
		t.Error("main's prompt docs must document list_threads")
	}
	if strings.Contains(tr.CoreDocs(false), "list_threads") {
		t.Error("worker docs must exclude it; leaders are documented by leaderThreadPromptTemplate instead")
	}
}

// CoreDocs(false) strips MainOnly tools from sub-thread prompts, so a leader
// that receives list_threads on the wire is otherwise undocumented.
func TestLeaderPromptDocumentsListThreads(t *testing.T) {
	if !strings.Contains(leaderThreadPromptTemplate, "list_threads") {
		t.Error("leaderThreadPromptTemplate must document list_threads, as it already does for kill/update")
	}
}

// A leader's roster is its own children, so it hits the same digest threshold
// main does and needs the same escape hatch. A plain worker must not get it —
// it has no children to search.
func TestListThreadsGrantedToLeadersNotWorkers(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	if err := thinker.threads.SpawnWithOpts("lead", "team lead",
		[]string{"spawn", "web"}, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn leader: %v", err)
	}
	defer thinker.threads.Kill("lead")

	if err := thinker.threads.SpawnWithOpts("plain", "worker",
		[]string{"web"}, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	defer thinker.threads.Kill("plain")

	infos := thinker.threads.List()
	tools := map[string][]string{}
	for _, info := range infos {
		tools[info.ID] = info.Tools
	}

	if !contains(tools["lead"], "list_threads") {
		t.Errorf("leader must receive list_threads alongside kill/update; got %v", tools["lead"])
	}
	if contains(tools["plain"], "list_threads") {
		t.Errorf("non-leader must not receive list_threads; got %v", tools["plain"])
	}
}

func TestUpdatingLeaderToolsRetainsListThreads(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	if err := thinker.threads.SpawnWithOpts("lead", "team lead",
		[]string{"spawn", "web"}, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn leader: %v", err)
	}
	defer thinker.threads.Kill("lead")

	if err := thinker.threads.Update("lead", "", "", []string{"spawn", "files"}); err != nil {
		t.Fatalf("update leader tools: %v", err)
	}
	thread := thinker.threads.threads["lead"]
	if thread == nil || !thread.Tools["list_threads"] {
		t.Fatalf("live leader lost list_threads after tool update: %#v", thread)
	}
}

func TestListThreadsSchedulesExactlyOneRequestedContinuation(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	call := toolCall{Name: "list_threads", NativeID: "list-main", Args: map[string]string{"filter": "billing"}}
	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{call}, nil)
	if len(results) != 1 || results[0].CallID != "list-main" {
		t.Fatalf("main list_threads result=%+v", results)
	}
	if !thinker.kickNextTurn {
		t.Fatal("main list_threads must schedule a result-consuming continuation")
	}
	// Simulate Run consuming the one-shot kick. Merely invoking the handler on
	// a later turn without another list_threads call must not recreate it.
	thinker.kickNextTurn = false
	mainToolHandler(thinker)(thinker, nil, nil)
	if thinker.kickNextTurn {
		t.Fatal("list_threads continuation repeated without another model call")
	}

	if err := thinker.threads.SpawnWithOpts("lead", "team lead", []string{"spawn"}, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn leader: %v", err)
	}
	defer thinker.threads.Kill("lead")
	lead := thinker.threads.threads["lead"]
	lead.Thinker.kickNextTurn = false
	_, _, results = threadToolHandler(lead, thinker.threads)(lead.Thinker, []toolCall{{
		Name: "list_threads", NativeID: "list-lead", Args: map[string]string{},
	}}, nil)
	if len(results) != 1 || results[0].CallID != "list-lead" {
		t.Fatalf("leader list_threads result=%+v", results)
	}
	if !lead.Thinker.kickNextTurn {
		t.Fatal("leader list_threads must schedule a result-consuming continuation")
	}
}

// ListTree recurses across manager boundaries. threadExistsInTree documents
// that recursing while holding tm.mu deadlocks against spawnInternal's write
// lock, so this hammers the walk against concurrent spawns and kills. Run
// under -race; a deadlock surfaces as the test timing out.
func TestListTreeConcurrentWithSpawnAndKill(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	if err := thinker.threads.SpawnWithOpts("lead", "team lead",
		[]string{"spawn"}, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn leader: %v", err)
	}
	defer thinker.threads.Kill("lead")

	done := make(chan struct{})
	var readers sync.WaitGroup
	for i := 0; i < 4; i++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-done:
					return
				default:
					thinker.threads.ListTreeAgentVisible()
					runListThreads(thinker.threads, map[string]string{"filter": "work"})
				}
			}
		}()
	}

	for i := 0; i < 40; i++ {
		id := fmt.Sprintf("worker-%d", i)
		if err := thinker.threads.SpawnWithOpts(id, "work unit",
			nil, SpawnOpts{DeferRun: true}); err != nil {
			t.Errorf("spawn %s: %v", id, err)
			break
		}
		thinker.threads.Kill(id)
	}

	close(done)
	readers.Wait()
}

func TestListThreadsEntryCarriesRoutingFields(t *testing.T) {
	tm := newSearchManager(t, ThreadInfo{
		ID: "worker-1", Name: "Worker One", ParentID: "main", Depth: 1,
		Directive: "do the thing", Tools: []string{"web"}, MCPNames: []string{"billing"},
	})
	got := runListThreads(tm, map[string]string{})
	for _, want := range []string{"worker-1", "Worker One", "parent=main", "depth=1", "wake=events-only", "do the thing", "web", "mcp scopes: billing"} {
		if !strings.Contains(got, want) {
			t.Errorf("entry must include %q; got:\n%s", want, got)
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
