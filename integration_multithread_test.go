package main

import (
	"strings"
	"testing"
	"time"
)

// TestIntegration_MultiThreadCoordination tests the full flow:
// 1. Main receives a console event asking to spawn 3 worker threads
// 2. Each thread does a simple task and calls done
// 3. Main receives all done messages
//
// This uses real LLM calls and tests the complete agent loop.
func TestIntegration_MultiThreadCoordination(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	directive := `You coordinate worker threads. When you receive a console event asking to "run tasks", spawn exactly 3 threads:
- id="task-a" directive="Say the word 'alpha' then immediately call [[done message="alpha complete"]]"
- id="task-b" directive="Say the word 'beta' then immediately call [[done message="beta complete"]]"
- id="task-c" directive="Say the word 'gamma' then immediately call [[done message="gamma complete"]]"

Each thread has tools="done". After spawning all 3, set pace to sleep and wait for their results.
When all 3 threads report done, say "ALL TASKS COMPLETE" in your thought.`

	provider, err := selectProvider(NewConfig())
	if err != nil {
		t.Skip("no provider available")
	}

	thinker := NewThinker(apiKey, provider)
	thinker.config = &Config{Directive: directive}
	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, ModeAutonomous, thinker.registry, "", nil, nil)}

	// Set up event filter and tool handler (same as normal startup)
	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(thinker.config.GetDirective(), ModeAutonomous, thinker.registry, toolDocs, thinker.mcpServers, nil)
	}

	go thinker.Run()
	defer thinker.Stop()

	// Observer to track events
	obs := thinker.bus.SubscribeAll("test", 500)
	defer thinker.bus.Unsubscribe("test")

	// Track done messages received by main
	doneMessages := make(map[string]bool)
	var allComplete bool

	// Hard timeout
	timeout := time.After(3 * time.Minute)

	// Inject the trigger event after first iteration
	time.Sleep(2 * time.Second)
	thinker.InjectConsole("run tasks")
	t.Log("injected 'run tasks' console event")

	// Monitor events
	for {
		select {
		case ev := <-obs.C:
			switch ev.Type {
			case EventThreadStart:
				t.Logf("thread started: %s", ev.From)
			case EventThreadDone:
				t.Logf("thread done: %s msg=%q", ev.From, ev.Text)
			case EventThinkDone:
				// Check for done messages in main's consumed events
				for _, consumed := range ev.ConsumedEvents {
					if strings.Contains(consumed, "task-a done") {
						doneMessages["task-a"] = true
						t.Log("main received: task-a done")
					}
					if strings.Contains(consumed, "task-b done") {
						doneMessages["task-b"] = true
						t.Log("main received: task-b done")
					}
					if strings.Contains(consumed, "task-c done") {
						doneMessages["task-c"] = true
						t.Log("main received: task-c done")
					}
				}
				if len(doneMessages) == 3 {
					allComplete = true
					t.Log("all 3 tasks reported done!")
				}
			}
			if allComplete {
				goto done
			}
		case <-timeout:
			t.Fatalf("timeout: only got %d/3 done messages: %v", len(doneMessages), doneMessages)
		}
	}

done:
	// Verify
	if !doneMessages["task-a"] {
		t.Error("missing done from task-a")
	}
	if !doneMessages["task-b"] {
		t.Error("missing done from task-b")
	}
	if !doneMessages["task-c"] {
		t.Error("missing done from task-c")
	}

	// All threads should be cleaned up
	time.Sleep(500 * time.Millisecond)
	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 threads at end, got %d", thinker.threads.Count())
	}

	t.Log("multi-thread coordination test passed")
}
