package core

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// makeRosterThreads builds n threads whose rendered entry width is close to
// production shape: a realistic id, a directive past the 150-char truncation
// point, and the builtin tool set a spawned worker always receives.
func makeRosterThreads(n int) []ThreadInfo {
	out := make([]ThreadInfo, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, ThreadInfo{
			ID:        fmt.Sprintf("chat-%04d", i),
			Directive: strings.Repeat("handle the customer conversation and escalate when needed. ", 5),
			Tools:     []string{"send", "done", "search_tools", "evolve", "pace"},
		})
	}
	return out
}

func TestRosterFullBelowBudgetAssertsCompleteness(t *testing.T) {
	threads := makeRosterThreads(5)
	got := buildDynamicTurnContext(threads, "")

	if !strings.HasPrefix(got, "[ACTIVE THREADS]") {
		t.Fatalf("block must open with the literal [ACTIVE THREADS] token, got %q", truncateStr(got, 60))
	}
	if !strings.Contains(got, "this is the complete list") {
		t.Error("small rosters must assert completeness, or the model burns a turn calling list_threads to check")
	}
	if strings.Contains(got, "partial view") {
		t.Error("a 5-thread roster must not digest")
	}
	for _, th := range threads {
		if !strings.Contains(got, th.ID) {
			t.Errorf("full roster must list every thread; %s missing", th.ID)
		}
	}
}

func TestActiveThreadRosterIncludesGrandchildrenBeforeClaimingCompleteness(t *testing.T) {
	grandchildren := newSearchManager(t, ThreadInfo{
		ID: "acme-payment-owner", ParentID: "billing-lead", Depth: 1,
		Directive: "Own ACME payment disputes.",
	})
	root := newSearchManager(t, ThreadInfo{
		ID: "billing-lead", ParentID: "main", Directive: "Coordinate billing work.",
	})
	attachChildren(root, "billing-lead", grandchildren)

	active := activeThreadRoster(root)
	got := buildDynamicTurnContext(active, "")
	if !strings.Contains(got, "this is the complete list") {
		t.Fatalf("small hierarchy should render completely; got:\n%s", got)
	}
	for _, id := range []string{"billing-lead", "acme-payment-owner"} {
		if !strings.Contains(got, id) {
			t.Errorf("complete hierarchy omitted %s; got:\n%s", id, got)
		}
	}
}

func TestActiveThreadRosterDigestsOnTotalHierarchySize(t *testing.T) {
	children := make([]ThreadInfo, 0, rosterInlineMaxEntries)
	for i := 0; i < rosterInlineMaxEntries; i++ {
		children = append(children, ThreadInfo{
			ID: fmt.Sprintf("worker-%02d", i), ParentID: "lead", Depth: 1,
		})
	}
	descendants := newSearchManager(t, children...)
	root := newSearchManager(t, ThreadInfo{ID: "lead", ParentID: "main"})
	attachChildren(root, "lead", descendants)

	active := activeThreadRoster(root)
	if len(active) != rosterInlineMaxEntries+1 {
		t.Fatalf("active hierarchy has %d threads, want %d", len(active), rosterInlineMaxEntries+1)
	}
	if got := buildDynamicTurnContext(active, ""); !strings.Contains(got, "partial view") {
		t.Fatalf("hierarchy over the total entry budget must digest; got:\n%s", got)
	}
}

func TestRosterDigestsAboveEntryBudget(t *testing.T) {
	got := buildDynamicTurnContext(makeRosterThreads(rosterInlineMaxEntries+1), "")

	if !strings.HasPrefix(got, "[ACTIVE THREADS]") {
		t.Fatalf("digest must keep the section token, got %q", truncateStr(got, 60))
	}
	if !strings.Contains(got, "partial view") {
		t.Error("over-budget roster must announce itself as partial")
	}
	if strings.Contains(got, "this is the complete list") {
		t.Error("digest must not claim completeness")
	}
	if !strings.Contains(got, "list_threads") {
		t.Error("digest must point at list_threads")
	}
	if strings.Contains(got, "chat-0000\n  directive:") {
		t.Error("digest must not render full per-thread entries")
	}
}

// The char budget must be able to trip on its own: a few leaders with wide
// tool grants exceed 8 KB well under the entry cap.
func TestRosterDigestsOnCharBudgetAloneWithWideToolGrants(t *testing.T) {
	wide := make([]string, 40)
	for i := range wide {
		wide[i] = fmt.Sprintf("some_mcp_tool_name_%02d", i)
	}
	threads := make([]ThreadInfo, 0, 10)
	for i := 0; i < 10; i++ {
		threads = append(threads, ThreadInfo{
			ID:        fmt.Sprintf("leader-%d", i),
			Directive: strings.Repeat("x", 200),
			Tools:     wide,
		})
	}
	if len(threads) > rosterInlineMaxEntries {
		t.Fatalf("fixture must stay under the entry cap to isolate the char gate")
	}
	if got := buildDynamicTurnContext(threads, ""); !strings.Contains(got, "partial view") {
		t.Errorf("char budget must trip independently of the entry count; got full roster of %d bytes", len(got))
	}
}

func TestRosterHysteresisHoldsDigestNearBoundary(t *testing.T) {
	// Just under the cap, but above the 75% re-expand line.
	near := makeRosterThreads(rosterInlineMaxEntries - 1)

	fresh := buildDynamicTurnContextView(near, "", rosterView{})
	if strings.Contains(fresh, "partial view") {
		t.Fatal("under the cap from a non-digested state must render full")
	}

	sticky := buildDynamicTurnContextView(near, "", rosterView{digested: true})
	if !strings.Contains(sticky, "partial view") {
		t.Error("already-digested roster must stay digested near the boundary, or it flip-flops every turn")
	}

	// Well below the re-expand line: must return to full.
	reexpandLine := float64(rosterInlineMaxEntries) * rosterReexpandFraction
	low := makeRosterThreads(int(reexpandLine) - 2)
	back := buildDynamicTurnContextView(low, "", rosterView{digested: true})
	if strings.Contains(back, "partial view") {
		t.Errorf("roster must re-expand once comfortably below both budgets (%d threads)", len(low))
	}
}

func TestRosterDeltaReportsSpawnsAndExits(t *testing.T) {
	current := makeRosterThreads(rosterInlineMaxEntries + 1)
	previous := make(map[string]bool, len(current))
	for _, th := range current[:len(current)-2] {
		previous[th.ID] = true
	}
	previous["chat-9999"] = true // ended since last turn

	got := buildDynamicTurnContextView(current, "", rosterView{digested: true, previous: previous})

	if !strings.Contains(got, "+2 spawned") {
		t.Errorf("delta must report the 2 new threads; got:\n%s", got)
	}
	if !strings.Contains(got, "-1 ended") || !strings.Contains(got, "chat-9999") {
		t.Errorf("delta must report the ended thread; got:\n%s", got)
	}
}

// A nil previous set is the first turn. Reporting every thread as newly
// spawned there would be both wrong and unbounded.
func TestRosterDeltaSilentOnFirstTurn(t *testing.T) {
	got := buildDynamicTurnContextView(makeRosterThreads(rosterInlineMaxEntries+1), "", rosterView{digested: true})
	if strings.Contains(got, "since last turn") {
		t.Errorf("first turn has no baseline and must not emit a delta; got:\n%s", got)
	}
}

func TestRosterDeltaNamesAreCapped(t *testing.T) {
	current := makeRosterThreads(200)
	added, _ := rosterDelta(current, map[string]bool{})
	if len(added) != 200 {
		t.Fatalf("expected 200 added, got %d", len(added))
	}
	line := formatRosterDeltaNames(added)
	if strings.Count(line, ",") > rosterDigestDeltaNames {
		t.Errorf("delta names must be capped so a spawn burst cannot become unbounded; got %q", line)
	}
	if !strings.Contains(line, "+195 more") {
		t.Errorf("capped delta must state the true overflow count; got %q", line)
	}
}

// The dedupe instruction in both system prompts is only sound against a
// complete roster. The digest has to carry the correction itself.
func TestRosterDigestCarriesSpawnDedupeWarning(t *testing.T) {
	got := buildDynamicTurnContext(makeRosterThreads(rosterInlineMaxEntries+1), "")
	if !strings.Contains(got, "before spawning") {
		t.Errorf("digest must warn that absence is not evidence of non-existence; got:\n%s", got)
	}
}

func TestRosterDigestCountsAreDerivable(t *testing.T) {
	threads := makeRosterThreads(rosterInlineMaxEntries + 1)
	threads[0].NextWakeAt = time.Now().Add(time.Hour)
	threads[1].SubThreads = 3

	got := buildDynamicTurnContext(threads, "")
	if !strings.Contains(got, "1 with a pending wake") {
		t.Errorf("expected pending-wake count; got:\n%s", got)
	}
	if !strings.Contains(got, "1 with sub-threads") {
		t.Errorf("expected leader count; got:\n%s", got)
	}
	// ThreadInfo.Running is hardcoded true by List, so any running/paused
	// split would be fabricated.
	if strings.Contains(got, "paused") {
		t.Error("digest must not report a paused count that ThreadInfo cannot support")
	}
}

// System threads (e.g. unconscious) are excluded before budgeting, so they
// must not push a roster over the threshold.
func TestRosterExcludesSystemThreadsFromBudget(t *testing.T) {
	threads := makeRosterThreads(rosterInlineMaxEntries)
	for i := 0; i < 10; i++ {
		threads = append(threads, ThreadInfo{ID: fmt.Sprintf("sys-%d", i), System: true})
	}
	got := buildDynamicTurnContext(threads, "")
	if strings.Contains(got, "partial view") {
		t.Error("system threads must not count toward the roster budget")
	}
	if strings.Contains(got, "sys-0") {
		t.Error("system threads must never appear in the roster")
	}
}

// context_breakdown.go splits on "\n\n[ACTIVE THREADS]" and
// retrieval_context.go prefix-matches "[ACTIVE THREADS]". Both renderings must
// keep that token first on the line.
func TestRosterSectionTokenStableAcrossModes(t *testing.T) {
	for name, threads := range map[string][]ThreadInfo{
		"full":   makeRosterThreads(3),
		"digest": makeRosterThreads(rosterInlineMaxEntries + 1),
	} {
		got := buildDynamicTurnContext(threads, "recall block")
		if !strings.HasPrefix(got, "[ACTIVE THREADS]") {
			t.Errorf("%s: section token must lead the block, got %q", name, truncateStr(got, 60))
		}
		if !strings.Contains(got, "recall block") {
			t.Errorf("%s: recall context must still be appended", name)
		}
	}
}

func TestRosterEmptyProducesNoSection(t *testing.T) {
	if got := buildDynamicTurnContext(nil, ""); got != "" {
		t.Errorf("no threads and no recall must produce an empty block, got %q", got)
	}
	if got := buildDynamicTurnContext([]ThreadInfo{}, ""); strings.Contains(got, "[ACTIVE THREADS]") {
		t.Error("empty slice must not emit a section")
	}
}

// advanceRoster is what makes hysteresis and the delta work across turns.
func TestAdvanceRosterTracksStateAcrossTurns(t *testing.T) {
	th := &Thinker{}

	th.advanceRoster(makeRosterThreads(3))
	if th.roster.digested {
		t.Error("3 threads must not set the digested flag")
	}
	if len(th.roster.previous) != 3 {
		t.Errorf("expected 3 ids recorded, got %d", len(th.roster.previous))
	}

	th.advanceRoster(makeRosterThreads(rosterInlineMaxEntries + 1))
	if !th.roster.digested {
		t.Error("over-budget turn must set the digested flag for the next turn")
	}

	// Boundary: stays digested.
	th.advanceRoster(makeRosterThreads(rosterInlineMaxEntries - 1))
	if !th.roster.digested {
		t.Error("hysteresis must hold the digested flag near the boundary")
	}

	th.advanceRoster(makeRosterThreads(2))
	if th.roster.digested {
		t.Error("well below both budgets must clear the digested flag")
	}
	if len(th.roster.previous) != 2 {
		t.Errorf("previous set must track the latest turn, got %d", len(th.roster.previous))
	}
}
