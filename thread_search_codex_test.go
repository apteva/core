package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexThreadRosterSearchBehavior proves the model-facing
// contract at both sides of the roster threshold:
//
//   - a small complete hierarchy already exposes a grandchild owner, so Codex
//     reuses that knowledge without spending a turn on list_threads;
//   - a large hierarchy is explicitly partial, so Codex calls list_threads and
//     does not spawn a duplicate after the existing owner is returned.
//
// Run:
//
//	RUN_CODEX_THREAD_ROSTER_SMOKE=1 go test -v -run TestIntegration_CodexThreadRosterSearchBehavior -timeout 8m .
func TestIntegration_CodexThreadRosterSearchBehavior(t *testing.T) {
	if os.Getenv("RUN_CODEX_THREAD_ROSTER_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_THREAD_ROSTER_SMOKE=1 to run the Codex thread-roster smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex thread-roster smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")

	t.Run("small complete hierarchy avoids list_threads", func(t *testing.T) {
		runCodexThreadRosterCase(t, token, false)
	})
	t.Run("large partial hierarchy searches before spawn", func(t *testing.T) {
		runCodexThreadRosterCase(t, token, true)
	})
}

func runCodexThreadRosterCase(t *testing.T, token string, large bool) {
	t.Helper()
	t.Chdir(t.TempDir())
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Coordinate work ownership without duplicating existing owners.",
			"",
			"# Operating rules",
			"Wait for operator events. Reuse existing ownership whenever it exists.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	thinker := NewThinker("", NewOpenAICodexProvider(token), cfg)
	defer thinker.Stop()

	if err := thinker.threads.SpawnWithOpts(
		"billing-lead", "Coordinate billing and payment ownership.", []string{"spawn"},
		SpawnOpts{DeferRun: true, ParentID: "main", Depth: 0},
	); err != nil {
		t.Fatalf("spawn billing leader: %v", err)
	}
	lead := thinker.threads.threads["billing-lead"]
	if lead == nil || lead.Children == nil {
		t.Fatal("billing leader has no child manager")
	}
	if err := lead.Children.SpawnWithOpts(
		"acme-payment-owner", "Own ACME payment-dispute requests and their operational state.", nil,
		SpawnOpts{DeferRun: true, ParentID: "billing-lead", Depth: 1},
	); err != nil {
		t.Fatalf("spawn existing payment owner: %v", err)
	}

	if large {
		// leader + grandchild + 29 direct workers = 31 total, just beyond the
		// entry threshold. The target owner therefore disappears from the
		// inline digest and must be found through list_threads.
		for i := 0; i < rosterInlineMaxEntries-1; i++ {
			id := fmt.Sprintf("unrelated-owner-%02d", i)
			directive := fmt.Sprintf("Own unrelated operational domain %02d.", i)
			if err := thinker.threads.SpawnWithOpts(id, directive, nil,
				SpawnOpts{DeferRun: true, ParentID: "main", Depth: 0}); err != nil {
				t.Fatalf("spawn filler %s: %v", id, err)
			}
		}
	}

	roster := buildDynamicTurnContext(activeThreadRoster(thinker.threads), "")
	if large && !strings.Contains(roster, "partial view") {
		t.Fatalf("large fixture did not produce a partial roster:\n%s", roster)
	}
	if !large {
		if !strings.Contains(roster, "this is the complete list") ||
			!strings.Contains(roster, "acme-payment-owner") {
			t.Fatalf("small fixture did not expose the complete hierarchy:\n%s", roster)
		}
	}

	started := time.Now()
	thinker.InjectConsole(strings.Join([]string{
		"An ACME payment-dispute request needs an owner.",
		"Reuse an existing owner if one exists; create a new paused owner only if none exists.",
		"This is an ownership review only, so do not message or modify an existing owner.",
		"After deciding, pace normally.",
	}, " "))
	go thinker.Run()

	type observedCall struct {
		name string
		args map[string]string
	}
	seenEvents := map[string]bool{}
	var calls []observedCall
	var listResult string
	llmDone := 0
	deadline := time.Now().Add(4 * time.Minute)
	for time.Now().Before(deadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != "main" || event.Time.Before(started) || seenEvents[event.ID] {
				continue
			}
			seenEvents[event.ID] = true
			switch event.Type {
			case "tool.call":
				var data ToolCallData
				if json.Unmarshal(event.Data, &data) == nil {
					calls = append(calls, observedCall{name: data.Name, args: data.Args})
				}
			case "tool.result":
				var data ToolResultData
				if json.Unmarshal(event.Data, &data) == nil && data.Name == "list_threads" {
					listResult = data.Result
				}
			case "llm.done":
				llmDone++
			}
		}

		for _, call := range calls {
			if call.name == "spawn" {
				t.Fatalf("Codex spawned a duplicate owner (large=%v): args=%v calls=%v", large, call.args, calls)
			}
		}

		if !large && llmDone >= 1 {
			break
		}
		if large {
			sawList := false
			for _, call := range calls {
				if call.name == "list_threads" {
					sawList = true
					break
				}
			}
			// Require a later model completion so this validates that Codex
			// consumed the list result rather than merely emitting the call.
			if sawList && listResult != "" && llmDone >= 2 {
				break
			}
		}
		time.Sleep(100 * time.Millisecond)
	}

	var names []string
	listCalls := 0
	spawnCalls := 0
	for _, call := range calls {
		names = append(names, call.name)
		switch call.name {
		case "list_threads":
			listCalls++
		case "spawn":
			spawnCalls++
		}
	}
	if spawnCalls != 0 {
		t.Fatalf("spawn calls=%d, want 0; calls=%v", spawnCalls, calls)
	}
	if large {
		if listCalls == 0 {
			t.Fatalf("Codex did not search the partial hierarchy; llm_done=%d calls=%v", llmDone, calls)
		}
		if !strings.Contains(listResult, "acme-payment-owner") {
			t.Fatalf("list_threads result did not expose the existing owner: %q", listResult)
		}
		if llmDone < 2 {
			t.Fatalf("Codex did not process the list result; llm_done=%d calls=%v", llmDone, calls)
		}
	} else if listCalls != 0 {
		t.Fatalf("Codex unnecessarily called list_threads for a complete small hierarchy: calls=%v", calls)
	}
	t.Logf("Codex roster case large=%v: llm_done=%d calls=%s list_result=%q",
		large, llmDone, strings.Join(names, ","), truncateStr(listResult, 300))
}
