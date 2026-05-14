package core

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// Cache-friendliness audit tests.
//
// Two complementary tests live here:
//
//   TestPromptStability        — offline, no LLM. Builds the system prompt
//                                under two states (fresh vs. mid-session)
//                                and computes the longest common prefix.
//                                A free, deterministic upper bound on what
//                                a real prefix cache can hit. Run anytime.
//
//   TestIntegration_CacheHitRatio_Kimi — real LLM (Fireworks kimi-k2p6).
//                                Issues N user turns, captures cached vs.
//                                prompt tokens per turn, reports steady-
//                                state ratio. Skips if FIREWORKS_API_KEY
//                                is unset.
//
// Both log human-readable summaries so before/after-a-prompt-change runs
// can be eyeballed without parsing test output.

// commonPrefixLen — bytes that match between a and b starting from offset 0.
// This is exactly the size of the prefix a naïve cache can reuse.
func commonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// renderContext returns ±N chars around an offset, with newlines escaped
// so the diagnostic stays on one log line.
func renderContext(s string, offset, radius int) string {
	start := offset - radius
	if start < 0 {
		start = 0
	}
	end := offset + radius
	if end > len(s) {
		end = len(s)
	}
	out := s[start:end]
	out = strings.ReplaceAll(out, "\n", "\\n")
	return out
}

func TestPromptStability(t *testing.T) {
	directive := "You are a coordinating agent. Spawn workers when needed."
	mode := ModeAutonomous

	// State A: fresh session, no per-turn context.
	promptA := buildSystemPrompt(directive, mode, nil, "", nil, nil, nil, nil)

	// After the cache fix, buildSystemPrompt no longer renders activeThreads
	// or extraToolDocs — those moved to buildDynamicTurnContext. So
	// passing varied dynamic state to buildSystemPrompt should now produce
	// IDENTICAL output (regression test).
	now := time.Now()
	activeThreads := []ThreadInfo{
		{
			ID:         "worker-1",
			ParentID:   "main",
			Depth:      1,
			Directive:  "Do the thing",
			Tools:      []string{"send", "done", "pace", "evolve", "remember"},
			Started:    now.Add(-30 * time.Second),
			Iteration:  5,
			Rate:       RateFast,
			Model:      ModelLarge,
			SubThreads: 0,
		},
	}
	promptB := buildSystemPrompt(directive, mode, nil, "", nil, activeThreads, nil, nil)

	activeThreadsC := []ThreadInfo{activeThreads[0]}
	activeThreadsC[0].Started = now.Add(-35 * time.Second)
	activeThreadsC[0].Iteration = 6
	promptC := buildSystemPrompt(directive, mode, nil, "", nil, activeThreadsC, nil, nil)

	promptD := buildSystemPrompt(directive, mode, nil, "[CANDIDATE TOOL]\nfoo: …", nil, activeThreads, nil, nil)

	report := func(label, base, variant string) {
		baseLen := len(base)
		variantLen := len(variant)
		common := commonPrefixLen(base, variant)
		ratio := 0.0
		if baseLen > 0 {
			ratio = float64(common) / float64(baseLen)
		}
		t.Logf("[%s]", label)
		t.Logf("    base bytes: %d", baseLen)
		t.Logf("    variant bytes: %d", variantLen)
		t.Logf("    common prefix bytes: %d (%.1f%% of base)", common, ratio*100)
		if common < baseLen {
			t.Logf("    first divergence @ offset %d", common)
			t.Logf("    base context:    …%s…", renderContext(base, common, 60))
			t.Logf("    variant context: …%s…", renderContext(variant, common, 60))
		}
		t.Logf("")
	}

	t.Log("────────────────────────────────────────────────")
	t.Log("Prompt-stability snapshot — measures the upper")
	t.Log("bound on prefix-cache hit ratio for messages[0].")
	t.Log("Higher is better. ≥99% means caching is healthy.")
	t.Log("────────────────────────────────────────────────")
	report("fresh → after-spawn (worker added)", promptA, promptB)
	report("after-spawn → +5s tick (steady state)", promptB, promptC)
	report("after-spawn → +RAG tool docs", promptB, promptD)

	// Post-fix hard requirements: dynamic state changes must NOT
	// change messages[0]. If any of these triples diverge, a future
	// change has reintroduced volatile content into the static prompt.
	if promptA != promptB {
		t.Errorf("regression: spawning a worker changed messages[0] (delta = %d bytes)", len(promptB)-len(promptA))
	}
	if promptB != promptC {
		t.Errorf("regression: time/iteration tick changed messages[0]")
	}
	if promptB != promptD {
		t.Errorf("regression: RAG tool docs leaked into messages[0]")
	}

	// Sanity: buildDynamicTurnContext IS allowed to vary — that's its job.
	// Just verify it produces non-empty output when state is non-empty,
	// and that the output is deterministic given the same input.
	dyn1 := buildDynamicTurnContext(activeThreads, "")
	dyn2 := buildDynamicTurnContext(activeThreads, "")
	if dyn1 != dyn2 {
		t.Errorf("buildDynamicTurnContext is non-deterministic")
	}
	if dyn1 == "" {
		t.Errorf("buildDynamicTurnContext returned empty when active threads were provided")
	}
	t.Logf("[dynamic context size with 1 worker] %d bytes", len(dyn1))
}

// TestIntegration_CacheHitRatio_Kimi exercises a multi-turn session
// against Fireworks kimi-k2p6 and reports the steady-state cache hit
// ratio. Iteration 1 is excluded from the ratio (cold cache).
//
// Pass FIREWORKS_API_KEY in env. Honors -short by skipping.
func TestIntegration_CacheHitRatio_Kimi(t *testing.T) {
	apiKey := getAPIKey(t)

	// Use the Fireworks provider with the kimi-k2p6 default. The provider
	// constructor wires the model id table, so no per-call override is
	// needed.
	thinker := NewThinker(apiKey, NewFireworksProvider(apiKey))

	// Multi-turn dialog — five short user turns, content varied just
	// enough that the model has to read the new turn each time. The
	// system prompt + tools + earlier turns should be cached.
	turns := []string{
		"Reply with exactly one word: hello",
		"Reply with exactly one word: world",
		"Reply with exactly one word: foo",
		"Reply with exactly one word: bar",
		"Reply with exactly one word: baz",
	}

	type turnStat struct {
		Prompt    int
		Cached    int
		Completed int
		Reply     string
	}
	stats := make([]turnStat, 0, len(turns))

	stop := drainEvents(thinker)
	for i, msg := range turns {
		thinker.messages = append(thinker.messages, Message{Role: "user", Content: msg})
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		_ = ctx
		resp, err := thinker.think()
		cancel()
		if err != nil {
			stop()
			t.Fatalf("turn %d think() error: %v", i+1, err)
		}
		thinker.messages = append(thinker.messages, Message{Role: "assistant", Content: resp.Text})
		stats = append(stats, turnStat{
			Prompt:    resp.Usage.PromptTokens,
			Cached:    resp.Usage.CachedTokens,
			Completed: resp.Usage.CompletionTokens,
			Reply:     resp.Text,
		})
	}
	stop()

	// Per-turn breakdown.
	t.Log("────────────────────────────────────────────────")
	t.Log("Cache-hit-ratio test — Fireworks kimi-k2p6")
	t.Log("────────────────────────────────────────────────")
	t.Logf("%-5s %-10s %-10s %-10s %-8s %s", "turn", "prompt", "cached", "ratio", "out", "reply")
	for i, s := range stats {
		ratio := 0.0
		if s.Prompt > 0 {
			ratio = float64(s.Cached) / float64(s.Prompt) * 100
		}
		reply := strings.ReplaceAll(strings.TrimSpace(s.Reply), "\n", " ")
		if len(reply) > 40 {
			reply = reply[:40] + "…"
		}
		t.Logf("%-5d %-10d %-10d %-9.1f%% %-8d %q", i+1, s.Prompt, s.Cached, ratio, s.Completed, reply)
	}

	// Steady-state ratio: iterations 2..N (turn 1 is cold, no cache yet).
	if len(stats) < 2 {
		t.Fatal("need at least 2 turns to compute steady-state ratio")
	}
	var promptSum, cachedSum int
	for _, s := range stats[1:] {
		promptSum += s.Prompt
		cachedSum += s.Cached
	}
	ratio := 0.0
	if promptSum > 0 {
		ratio = float64(cachedSum) / float64(promptSum) * 100
	}
	t.Logf("────────────────────────────────────────────────")
	t.Logf("Steady state (turns 2-%d):", len(stats))
	t.Logf("  prompt:  %d", promptSum)
	t.Logf("  cached:  %d", cachedSum)
	t.Logf("  ratio:   %.1f%%", ratio)
	t.Logf("────────────────────────────────────────────────")

	// Soft assertion: any usable cache wiring should beat 30% on a
	// trivial multi-turn dialog. The number is intentionally generous —
	// the goal is to catch regressions, not lock in a specific value.
	// Update the floor when prompt-cache improvements land.
	if ratio < 30.0 {
		t.Errorf("steady-state cache hit ratio %.1f%% is below the 30%% floor — investigate", ratio)
	}

	// Surface the headline number on stdout too so a CI summary reader
	// can grep it without parsing testing's verbose output.
	fmt.Fprintf(os.Stdout, "[CACHE-RATIO] kimi-k2p6 turns=%d ratio=%.1f%% prompt=%d cached=%d\n",
		len(stats), ratio, promptSum, cachedSum)
}
