package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"
)

func continuityTestThinker(t *testing.T) *Thinker {
	t.Helper()
	thinker := newTestThinker()
	t.Cleanup(thinker.Stop)
	tools := ticketDiscoveryCatalog()
	thinker.toolIndex.Add("tickets", tools, false)
	thinker.registry = NewToolRegistry("")
	registerTestMCPTools(thinker.registry, "tickets", tools)
	thinker.retiredTools = newRetiredToolSets()
	return thinker
}

func spawnedThinker(t *testing.T, tm *ThreadManager, id string) *Thinker {
	t.Helper()
	_, thread := tm.findManagedThread(id)
	if thread == nil || thread.Thinker == nil {
		t.Fatalf("thread %q not found", id)
	}
	return thread.Thinker
}

func loadedNames(th *Thinker) []string {
	var out []string
	for _, name := range th.loadedToolsSnapshot() {
		if strings.HasPrefix(name, "tickets_") {
			out = append(out, name)
		}
	}
	return out
}

func TestThreadContinuesToolsFromEndedAndLiveThreads(t *testing.T) {
	thinker := continuityTestThinker(t)
	tm := thinker.threads
	scope := SpawnOpts{MCPNames: []string{"tickets"}, DeferRun: true}
	if err := tm.SpawnWithOpts("worker", "Handle step one.", nil, scope); err != nil {
		t.Fatal(err)
	}
	first := spawnedThinker(t, tm, "worker")
	runSearchTools(first, map[string]string{"queries": `["tickets_tickets_get","tickets_tickets_update"]`}, false)
	want := []string{"tickets_tickets_get", "tickets_tickets_update"}
	if got := loadedNames(first); !reflect.DeepEqual(got, want) {
		t.Fatalf("first worker loaded %v", got)
	}

	// A fresh worker without continuity starts empty, which is what made
	// replacement workers search again.
	if err := tm.SpawnWithOpts("unrelated", "Handle step two.", nil, scope); err != nil {
		t.Fatal(err)
	}
	if got := loadedNames(spawnedThinker(t, tm, "unrelated")); len(got) != 0 {
		t.Fatalf("plain spawn loaded %v", got)
	}

	tm.Kill("worker")
	next := scope
	next.ContinueToolsFrom = "worker"
	if err := tm.SpawnWithOpts("worker-g1", "Handle step two.", nil, next); err != nil {
		t.Fatal(err)
	}
	second := spawnedThinker(t, tm, "worker-g1")
	if got := loadedNames(second); !reflect.DeepEqual(got, want) {
		t.Fatalf("replacement did not continue the ended worker's tools: %v", got)
	}
	if visible := nativeToolSet(second.registry.NativeTools(second.toolAllowlist, second.activeTools, false)); !visible["tickets_tickets_update"] {
		t.Fatalf("continued tool not presented: %v", visible)
	}

	// A live thread can be the source too.
	live := scope
	live.ContinueToolsFrom = "worker-g1"
	if err := tm.SpawnWithOpts("helper", "Help.", nil, live); err != nil {
		t.Fatal(err)
	}
	if got := loadedNames(spawnedThinker(t, tm, "helper")); !reflect.DeepEqual(got, want) {
		t.Fatalf("live source not continued: %v", got)
	}

	// Continuity never widens access: a thread without the tickets scope
	// receives none of them.
	narrow := SpawnOpts{MCPNames: []string{}, DeferRun: true, ContinueToolsFrom: "worker-g1"}
	if err := tm.SpawnWithOpts("narrow", "Narrow.", []string{}, narrow); err != nil {
		t.Fatal(err)
	}
	if got := loadedNames(spawnedThinker(t, tm, "narrow")); len(got) != 0 {
		t.Fatalf("continuity widened access: %v", got)
	}

	// An unknown source is not an error; the thread starts empty.
	unknown := scope
	unknown.ContinueToolsFrom = "never-existed"
	if err := tm.SpawnWithOpts("orphan", "Orphan.", nil, unknown); err != nil {
		t.Fatal(err)
	}
	if got := loadedNames(spawnedThinker(t, tm, "orphan")); len(got) != 0 {
		t.Fatalf("unknown source loaded %v", got)
	}
}

func TestRetiredToolSetsExpireAndStayBounded(t *testing.T) {
	sets := newRetiredToolSets()
	now := time.Date(2026, 11, 26, 10, 0, 0, 0, time.UTC)
	sets.record("old", []string{"a_tool"}, now)
	if got := sets.get("old", now.Add(retiredToolSetTTL+time.Second)); got != nil {
		t.Fatalf("expired set returned %v", got)
	}
	for i := 0; i < retiredToolSetLimit+5; i++ {
		sets.record(fmt.Sprintf("t%d", i), []string{"a_tool"}, now.Add(time.Duration(i)*time.Second))
	}
	if len(sets.sets) != retiredToolSetLimit {
		t.Fatalf("retained %d sets, limit %d", len(sets.sets), retiredToolSetLimit)
	}
	if sets.get("t0", now.Add(time.Hour)) != nil || sets.get(fmt.Sprintf("t%d", retiredToolSetLimit+4), now.Add(time.Hour)) == nil {
		t.Fatal("oldest set was not the one dropped")
	}
	sets.record("empty", nil, now)
	if _, ok := sets.sets["empty"]; ok {
		t.Fatal("empty set recorded")
	}
}

func TestAPISpawnContinueToolsFrom(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	tools := ticketDiscoveryCatalog()
	thinker.toolIndex.Add("tickets", tools, false)
	registerTestMCPTools(thinker.registry, "tickets", tools)
	thinker.retiredTools = newRetiredToolSets()

	if w := postThreadForTest(t, api, "step-worker", map[string]any{"directive": "Step one.", "mcp": []string{"tickets"}}); w.Code != http.StatusOK {
		t.Fatalf("spawn: %d %s", w.Code, w.Body.String())
	}
	first := spawnedThinker(t, thinker.threads, "step-worker")
	// Activation normally happens on the worker's own loop; publish from here
	// the way that loop would after a search.
	first.touchActiveTool("tickets_tickets_history")
	first.publishActiveTools()
	thinker.threads.Kill("step-worker")

	if w := postThreadForTest(t, api, "step-worker-g1", map[string]any{"directive": "Step two.", "mcp": []string{"tickets"}, "continue_tools_from": "step-worker"}); w.Code != http.StatusOK {
		t.Fatalf("spawn continuation: %d %s", w.Code, w.Body.String())
	}
	if got := loadedNames(spawnedThinker(t, thinker.threads, "step-worker-g1")); !reflect.DeepEqual(got, []string{"tickets_tickets_history"}) {
		t.Fatalf("API continuation loaded %v", got)
	}

	w := postThreadForTest(t, api, "bad-source", map[string]any{"directive": "x", "continue_tools_from": "bad id!"})
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "continue_tools_from") {
		t.Fatalf("invalid source accepted: %d %s", w.Code, w.Body.String())
	}
}

func searchResult(t *testing.T, th *Thinker, args map[string]string) searchToolsResult {
	t.Helper()
	var res searchToolsResult
	if err := json.Unmarshal([]byte(runSearchTools(th, args, true)), &res); err != nil {
		t.Fatal(err)
	}
	return res
}

func TestSearchToolsSaysWhenNothingClearlyMatches(t *testing.T) {
	th := ticketDiscoveryThinker(t)
	weak := searchResult(t, th, map[string]string{"query": "export quarterly payroll invoices tickets"})
	if len(weak.Hits) == 0 || !weak.NoClearMatch || !strings.Contains(weak.Note, "No clear match") {
		t.Fatalf("weak results not flagged: %+v", weak)
	}
	for _, hit := range weak.Hits {
		if !hit.WeakMatch {
			t.Fatalf("hit %s not marked weak", hit.Name)
		}
	}
	clear := searchResult(t, th, map[string]string{"query": "fetch ticket history"})
	if clear.NoClearMatch || len(clear.Hits) == 0 {
		t.Fatalf("clear match flagged: %+v", clear)
	}
	exact := searchResult(t, th, map[string]string{"query": "tickets_tickets_get"})
	if exact.NoClearMatch {
		t.Fatalf("exact name flagged: %+v", exact)
	}
	miss := searchResult(t, th, map[string]string{"query": "wombatledger"})
	if !strings.Contains(miss.Error, "does not prove") || !strings.Contains(miss.Error, "instead of searching again") {
		t.Fatalf("miss does not bound retries: %s", miss.Error)
	}
}

func sendRankingThinker(t *testing.T) *Thinker {
	t.Helper()
	tools := []mcpToolDef{
		mkTool("contacts_reply", "Reply to a contact's latest inbound conversation message. Sends a real external message to the contact on their conversation channel."),
		mkTool("contacts_send_message", "Send a new message to a contact about their latest inbound conversation."),
		mkTool("contacts_get_conversation", "Get one contact conversation."),
	}
	ix := NewToolIndex()
	ix.Add("crm", tools, false)
	r := NewToolRegistry("")
	registerTestMCPTools(r, "crm", tools)
	return &Thinker{threadID: "main", toolIndex: ix, registry: r, activeTools: map[string]bool{}}
}

func TestSearchToolsRanksSendToolsLowerForReadQueries(t *testing.T) {
	th := sendRankingThinker(t)
	query := "latest inbound conversation message contact"
	plain := th.toolIndex.searchDetailedWithOptions(query, 3, true, nil, indexSearchOptions{})
	if len(plain.Hits) == 0 || !externalSendTool(plain.Hits[0]) {
		t.Fatalf("fixture no longer ranks a send tool first without demotion: %v", names(plain.Hits))
	}
	res := searchResult(t, th, map[string]string{"query": "get " + query})
	if len(res.Hits) == 0 || res.Hits[0].Name != "crm_contacts_get_conversation" {
		t.Fatalf("read query still ranks a send tool first: %+v", res.Hits)
	}
	// A query asking to send keeps send tools on top.
	res = searchResult(t, th, map[string]string{"query": "send reply to " + query})
	if len(res.Hits) == 0 || !strings.Contains(res.Hits[0].Name, "reply") && !strings.Contains(res.Hits[0].Name, "send") {
		t.Fatalf("send query lost send tools: %+v", res.Hits)
	}
	for query, want := range map[string]bool{
		"list latest messages":          true,
		"find contact by email":         true,
		"send a reply":                  false,
		"set conversation status":       false,
		"conversation message contact":  false,
		"check status then update deal": false,
	} {
		if got := readIntentQuery(query); got != want {
			t.Errorf("readIntentQuery(%q) = %v, want %v", query, got, want)
		}
	}
}
