package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestIntegration_UnconsciousConsolidation_OneCycle drives the unconscious
// thread through the SAME path production uses (parent thinker spawns the
// child via t.threads.SpawnWithOpts). Earlier versions of this test
// constructed a Thinker directly and patched threadID after the fact —
// that left the bus subscription bound to "main" while we published
// tool results to "unconscious", silently losing every result and
// causing the model to spin on review_history. Don't do that. Spawn
// like prod and the wiring just works.
//
// Run gating: opt-in via RUN_UNCONSCIOUS_INTEGRATION=1.
func TestIntegration_UnconsciousConsolidation_OneCycle(t *testing.T) {
	if os.Getenv("RUN_UNCONSCIOUS_INTEGRATION") == "" {
		t.Skip("set RUN_UNCONSCIOUS_INTEGRATION=1 to run the live unconscious cycle test")
	}
	tp := getTestProvider(t)
	t.Logf("provider=%s", tp.Source)

	// Sandbox: temp cwd so memory.jsonl + history/ + skills/ are isolated.
	dir := t.TempDir()
	prev, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	// Plant history/main.jsonl: three turns of clear signal a sensible
	// agent should consolidate.
	if err := os.MkdirAll("history", 0755); err != nil {
		t.Fatal(err)
	}
	historyLines := []string{
		`{"ts":"2026-04-26T10:00:00Z","role":"user","content":"[chat] Hi! Quick note: I always prefer terse, no-fluff replies. Save the verbose stuff for when I explicitly ask."}`,
		`{"ts":"2026-04-26T10:00:05Z","role":"assistant","content":"Got it — terse by default."}`,
		`{"ts":"2026-04-26T10:01:00Z","role":"user","content":"[chat] Also — my Postgres runs on port 6543 (custom), not 5432. Worth knowing for any DB stuff."}`,
		`{"ts":"2026-04-26T10:01:08Z","role":"assistant","content":"Noted — Postgres at 6543."}`,
		`{"ts":"2026-04-26T10:30:00Z","role":"user","content":"[chat] The dashboard pulse strip task is done. Shipped this morning."}`,
		`{"ts":"2026-04-26T10:30:05Z","role":"assistant","content":"Great — closed out that work item."}`,
	}
	if err := os.WriteFile(filepath.Join("history", "main.jsonl"),
		[]byte(strings.Join(historyLines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Build the parent thinker. We don't run main; we just need it as
	// the host for the ThreadManager so SpawnWithOpts can wire the
	// unconscious's bus subscription correctly. Memory store is shared
	// — the unconscious writes through it.
	cfg := NewConfig()
	cfg.Directive = "Test parent — does nothing while the unconscious works."
	cfg.Save()

	parent := NewThinker(tp.APIKey, tp.Provider, cfg)
	defer parent.Stop()

	// Spawn the unconscious through the production path. This is the
	// exact tool list and directive the production launcher uses (see
	// the auto-spawn block in NewThinker around config.Unconscious).
	tools := []string{
		"review_history", "memory_search", "memory_list",
		"memory_remember", "memory_supersede", "memory_drop",
		"skill_write", "pace",
	}
	if err := parent.threads.SpawnWithOpts(
		"unconscious",
		unconsciousDirectiveV2,
		tools,
		SpawnOpts{ParentID: "main", Depth: 0, System: true},
	); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}

	// The wake event mirrors what unconsciousSafetyFloors publishes
	// in production when history/main.jsonl grows past the threshold.
	parent.bus.Publish(Event{
		Type: EventInbox,
		To:   "unconscious",
		Text: "[wake] new history available — please consolidate",
	})

	// Wait until memory.jsonl has at least one record OR the budget
	// elapses. The unconscious uses its own pace; with default slow-
	// rate sleep between iterations a typical cycle (review_history
	// → 2-3 remembers → pace) lands inside 2-3 minutes.
	deadline := time.After(6 * time.Minute)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
poll:
	for {
		select {
		case <-deadline:
			break poll
		case <-tick.C:
			if parent.memory.Count() > 0 {
				// Give it 3 more seconds for any follow-up writes
				// in the same iteration.
				time.Sleep(3 * time.Second)
				break poll
			}
		}
	}

	// --- Assertions ---

	data, err := os.ReadFile("memory.jsonl")
	if err != nil {
		t.Fatalf("memory.jsonl not written: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("memory.jsonl is empty — unconscious wrote nothing within budget")
	}

	// Every line is a valid v2 record.
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	for i, ln := range lines {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		var rec MemoryRecord
		if err := json.Unmarshal([]byte(ln), &rec); err != nil {
			t.Errorf("line %d: invalid JSON: %v", i, err)
			continue
		}
		if rec.ID == "" {
			t.Errorf("line %d: missing id", i)
		}
		if rec.TS.IsZero() {
			t.Errorf("line %d: missing ts", i)
		}
		if !rec.Tombstone {
			if strings.TrimSpace(rec.Content) == "" {
				t.Errorf("line %d: non-tombstone record with empty content", i)
			}
			if rec.Weight < 0 || rec.Weight > 1 {
				t.Errorf("line %d: weight out of range: %v", i, rec.Weight)
			}
		} else {
			if rec.IDTarget == "" {
				t.Errorf("line %d: tombstone without id_target", i)
			}
			if rec.Reason == "" {
				t.Errorf("line %d: tombstone without reason", i)
			}
		}
	}

	// At least one active memory survived.
	if n := parent.memory.Count(); n == 0 {
		t.Fatal("expected at least one active memory, got 0")
	} else {
		t.Logf("active memories after one cycle: %d", n)
		for _, r := range parent.memory.Active() {
			idShort := r.ID
			if len(idShort) > 8 {
				idShort = idShort[:8]
			}
			t.Logf("  id=%s w=%.2f tags=%v content=%q",
				idShort, r.Weight, r.Tags, snippet(r.Content, 100))
		}
	}

	// Spot-check: at least one memory mentions one of the planted
	// signals. Models phrase things differently — we accept any of
	// several keyword groups.
	found := false
	for _, r := range parent.memory.Active() {
		c := strings.ToLower(r.Content)
		if strings.Contains(c, "terse") || strings.Contains(c, "concise") ||
			strings.Contains(c, "6543") || strings.Contains(c, "postgres") ||
			strings.Contains(c, "dashboard") || strings.Contains(c, "pulse") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("none of the planted signals (terse/postgres/dashboard) appear in any active memory")
		for _, r := range parent.memory.Active() {
			t.Logf("  active: %s", r.Content)
		}
	}

	// Supersede chain integrity.
	for _, r := range parent.memory.All() {
		if r.Supersedes != "" {
			if _, ok := parent.memory.byID[r.Supersedes]; !ok {
				t.Errorf("record %s supersedes unknown id %s", r.ID, r.Supersedes)
			}
		}
		if r.Tombstone && r.IDTarget != "" {
			if _, ok := parent.memory.byID[r.IDTarget]; !ok {
				t.Errorf("tombstone %s targets unknown id %s", r.ID, r.IDTarget)
			}
		}
	}
}

// snippet — local helper to avoid clashing with truncate variants in
// the rest of the package.
func snippet(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
