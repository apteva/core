package core

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestUsedToolsSurviveRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history", usedToolsFile)
	now := time.Now().Add(-time.Minute)
	store := newUsedToolStore(path, now)
	store.record("tickets_tickets_get", now.Add(-8*24*time.Hour)) // too old to keep
	store.record("tickets_tickets_list", now.Add(-2*time.Hour))
	store.record("tickets_tickets_update", now.Add(-time.Hour))
	store.record("other_server_tool", now.Add(-30*time.Minute)) // server not connected
	store.record("tickets_tickets_create", now)                 // no longer granted

	th := ticketDiscoveryThinker(t)
	th.toolAllowlist = map[string]bool{"tickets_tickets_list": true, "tickets_tickets_update": true, "tickets_tickets_get": true}
	th.usedTools = newUsedToolStore(path, now.Add(time.Minute))
	th.restoreUsedTools()

	want := map[string]bool{"tickets_tickets_list": true, "tickets_tickets_update": true}
	if !reflect.DeepEqual(th.activeTools, want) {
		t.Fatalf("restored tools = %v, want %v", th.activeTools, want)
	}
	if th.activeTools["tickets_tickets_create"] {
		t.Fatal("a saved name widened access past the current grants")
	}
	// The name whose server has not connected yet keeps waiting, then loads.
	if got := th.usedTools.pending; !reflect.DeepEqual(got, []string{"other_server_tool"}) {
		t.Fatalf("pending = %v", got)
	}
	tools := []mcpToolDef{mkTool("server_tool", "A tool on a server that connects late.")}
	th.toolIndex.Add("other", tools, false)
	registerTestMCPTools(th.registry, "other", tools)
	th.toolAllowlist["other_server_tool"] = true
	th.restoreUsedTools()
	if !th.activeTools["other_server_tool"] || len(th.usedTools.pending) != 0 {
		t.Fatalf("late server's tool was not restored: active=%v pending=%v", th.activeTools, th.usedTools.pending)
	}
}

func TestUsedToolsStopWaitingForMissingServer(t *testing.T) {
	path := filepath.Join(t.TempDir(), usedToolsFile)
	now := time.Now()
	newUsedToolStore(path, now).record("gone_server_tool", now)
	store := newUsedToolStore(path, now.Add(-2*usedToolRestoreWindow))
	if got := store.takePending(now, func(string) bool { return false }); got != nil || len(store.pending) != 0 {
		t.Fatalf("ready=%v pending=%v, want nothing left waiting", got, store.pending)
	}
}

func TestUsedToolsWriteOnlyWhenStale(t *testing.T) {
	path := filepath.Join(t.TempDir(), usedToolsFile)
	now := time.Now()
	store := newUsedToolStore(path, now)
	store.record("tickets_tickets_get", now)
	first, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	store.record("tickets_tickets_get", now.Add(time.Minute))
	if again, _ := os.ReadFile(path); string(again) != string(first) {
		t.Fatal("a repeated call within the resave interval rewrote the file")
	}
	store.record("tickets_tickets_get", now.Add(2*usedToolResaveAfter))
	if again, _ := os.ReadFile(path); string(again) == string(first) {
		t.Fatal("a stale saved time was not refreshed")
	}
	store.clear()
	if len(store.tools) != 0 {
		t.Fatal("clear kept saved names")
	}
	// A damaged file is an empty store, not an error.
	if err := os.WriteFile(path, []byte("{not json"), 0600); err != nil {
		t.Fatal(err)
	}
	if got := newUsedToolStore(path, now); len(got.tools) != 0 || len(got.pending) != 0 {
		t.Fatalf("damaged file produced %v", got.tools)
	}
}
