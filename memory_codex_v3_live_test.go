package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// Opt-in, real subscription-backed Codex calls. All tiers are pinned so the
// runtime cannot silently select another model. No provider fallback is used.
//
// RUN_CODEX_MEMORY_V3_LIVE=1 go test -v -count=1 -run '^TestIntegration_CodexGPT6SolMemory$' -timeout 15m .
func TestIntegration_CodexGPT6SolMemory(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_MEMORY_V3_LIVE") != "1" {
		t.Skip("set RUN_CODEX_MEMORY_V3_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live verification requested but no valid Codex test credential is available")
	}
	newProvider := func() LLMProvider {
		p := NewOpenAICodexProvider(token).(*OpenAINativeProvider)
		p.models = map[ModelTier]string{ModelLarge: "gpt-6-sol", ModelMedium: "gpt-6-sol", ModelSmall: "gpt-6-sol"}
		// Use only the test credential, never a configured production connection.
		p.runtimeTokenURL = ""
		p.serverAPIKey = ""
		return p
	}
	t.Log("live provider=openai-codex model=gpt-6-sol; synthetic data in temporary directories")
	t.Run("lexical_recall", func(t *testing.T) {
		runUsesLexicalMemoryWithoutEmbeddings(t, newProvider())
	})
	t.Run("full_thinker_ephemeral_recall", func(t *testing.T) {
		runUsesEphemeralMemoryAcrossTurns(t, newProvider())
	})
	t.Run("checkpointed_consolidation_restart_and_scopes", func(t *testing.T) {
		runCodexCheckpointedMemory(t, newProvider())
	})
	t.Run("history_fallback_isolation_and_forgetting", func(t *testing.T) {
		runCodexScopedHistorySearch(t, newProvider())
	})
	t.Run("full_thinker_multithread_consolidation", func(t *testing.T) {
		runCodexFullThinkerConsolidation(t, newProvider())
	})
}

func runCodexFullThinkerConsolidation(t *testing.T, provider LLMProvider) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	inTempCwd(t)
	if err := os.Mkdir("history", 0700); err != nil {
		t.Fatal(err)
	}
	seed := newMemoryHistory(&MemoryStore{path: memoryFile}, nil)
	codes := map[string]string{"chat-live-a": "copper-" + newULID(), "chat-live-b": "teal-" + newULID()}
	for id, code := range codes {
		writeMemoryHistory(t, seed, id, SessionEntry{Sequence: 1, Role: "user", Content: "Please remember my lasting deployment verification code exactly: " + code})
	}
	cfg := NewConfig()
	cfg.Directive = "Test supervisor. The unconscious owns memory consolidation."
	cfg.Unconscious = true
	parent := NewThinker("", provider, cfg)
	t.Cleanup(func() { parent.threads.KillAll(); parent.Stop() })
	parent.bus.Publish(Event{Type: EventInbox, To: "unconscious", Text: "[wake] Consolidate every available history batch now, preserving exact deployment codes from explicit user statements. Commit each batch after its writes succeed; continue reading until batch is null, then pace."})
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	deadline := time.NewTimer(4 * time.Minute)
	defer deadline.Stop()
	for {
		h := parent.memory.history
		h.mu.Lock()
		finished := h.state.Pending == nil && len(h.state.Seen) == 2
		h.mu.Unlock()
		if finished && parent.memory.Count() == 2 {
			break
		}
		select {
		case <-ticker.C:
		case <-deadline.C:
			t.Fatalf("full thinker did not finish both history batches; memories=%d", parent.memory.Count())
		}
	}
	for thread, code := range codes {
		matches := parent.memory.recallIndexed([]string{"deployment verification code"}, 5, thread)
		if len(matches) != 1 || !strings.Contains(matches[0].Record.Content, code) || len(matches[0].Record.Sources) != 1 || matches[0].Record.Sources[0].ThreadID != thread {
			t.Fatalf("full runtime scoped consolidation failed for %s", thread)
		}
	}
	t.Log("verified production auto-spawn, thinker loop, native dispatch, two source-thread batches, commits, provenance and scoped recall")
}

// runCodexMemoryToolLoop uses the real provider, real tool schemas and handlers,
// and stateless Responses continuation state. Only the bounded scheduler is a
// test harness; it never generates the model's tool calls or final answer.
func runCodexMemoryToolLoop(t *testing.T, provider LLMProvider, directive, prompt string,
	tools []NativeTool, dispatch func(NativeToolCall) ToolResponse) (string, []string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	messages := []Message{{Role: "system", Content: directive}, {Role: "user", Content: prompt}}
	var calls []string
	for step := 0; step < 16; step++ {
		started := time.Now()
		resp, err := provider.Chat(ctx, messages, "gpt-6-sol", tools, nil, nil, nil)
		if err != nil {
			t.Fatalf("live gpt-6-sol request failed: %v", err)
		}
		t.Logf("gpt-6-sol step=%d duration=%s tool_calls=%d", step+1, time.Since(started).Round(time.Millisecond), len(resp.ToolCalls))
		messages = append(messages, Message{Role: "assistant", Content: resp.Text, Reasoning: resp.Reasoning, ToolCalls: resp.ToolCalls, ProviderState: resp.ProviderState})
		if len(resp.ToolCalls) == 0 {
			return resp.Text, calls
		}
		var results []ToolResult
		for _, call := range resp.ToolCalls {
			calls = append(calls, call.Name)
			r := dispatch(call)
			t.Logf("tool=%s error=%v", call.Name, r.IsError)
			results = append(results, ToolResult{CallID: call.ID, ToolName: call.Name, Content: r.Text, IsError: r.IsError})
		}
		messages = append(messages, Message{Role: "tool", ToolResults: results})
	}
	t.Fatal("live memory tool loop exceeded 16 model requests")
	return "", calls
}

func runCodexCheckpointedMemory(t *testing.T, provider LLMProvider) {
	m, h := newHistoryTestStore(t)
	a := "saffron-" + newULID()
	b := "indigo-" + newULID()
	writeMemoryHistory(t, h, "chat-a", SessionEntry{Sequence: 1, Role: "user", Content: "Remember my permanent deployment verification code exactly: " + a})
	writeMemoryHistory(t, h, "chat-b", SessionEntry{Sequence: 1, Role: "user", Content: "Remember my permanent deployment verification code exactly: " + b})
	registry := NewToolRegistry("")
	registerSystemTools(registry, m, h.config)
	allow := map[string]bool{"review_history": true, "memory_remember": true, "memory_search": true, "memory_list": true, "memory_supersede": true, "memory_drop": true}
	native := registry.NativeTools(allow, nil, true)
	ctx := withMemoryCaller(context.Background(), "unconscious")
	restarted := false
	committed := map[string]bool{}
	dispatch := func(call NativeToolCall) ToolResponse {
		if !allow[call.Name] {
			return ToolResponse{Text: "unauthorized test tool", IsError: true}
		}
		r, ok := registry.DispatchContext(ctx, call.Name, call.Args)
		if !ok {
			t.Fatalf("unregistered tool %s", call.Name)
		}
		if r.IsError {
			t.Fatalf("consolidation tool %s failed: %s", call.Name, r.Text)
		}
		if call.Name == "review_history" && call.Args["action"] == "commit" {
			committed[call.Args["batch_id"]] = true
		}
		if call.Name == "memory_remember" && !restarted {
			// Simulate process loss after a durable write but before commit.
			// Reload real journal/checkpoint and replay the exact LLM-generated
			// write; it must not duplicate the memory.
			oldBatch := m.history.state.Pending
			if oldBatch == nil {
				t.Fatal("write had no durable review batch")
			}
			loaded := &MemoryStore{path: m.path, byID: map[string]int{}}
			loaded.load()
			m = loaded
			registerSystemTools(registry, m, h.config)
			replayed := readMemoryBatch(t, m.history, "50")
			if replayed == nil || replayed.ID != oldBatch.ID {
				t.Fatal("restart lost pending batch")
			}
			count := m.Count()
			again, _ := registry.DispatchContext(ctx, call.Name, call.Args)
			if again.IsError || m.Count() != count {
				t.Fatal("replayed model write was not idempotent")
			}
			restarted = true
		}
		return r
	}
	_, calls := runCodexMemoryToolLoop(t, provider, unconsciousDirectiveV2+
		"\nFor this bounded verification, process every available batch before finishing. Preserve exact deployment verification codes. After review_history reports batch=null, respond CONSOLIDATION_COMPLETE. No pacing tool is available.",
		"Consolidate the available retained histories now.", native, dispatch)
	if !restarted || len(committed) != 2 {
		t.Fatalf("incomplete protocol: restarted=%v commits=%d calls=%v", restarted, len(committed), calls)
	}
	for thread, code := range map[string]string{"chat-a": a, "chat-b": b} {
		found := false
		for _, rec := range m.Active() {
			if !strings.Contains(rec.Content, code) {
				continue
			}
			found = true
			if rec.Scope != thread || len(rec.Sources) != 1 || rec.Sources[0].ThreadID != thread || rec.Sources[0].Role != "user" {
				t.Fatalf("incorrect source scope: %+v", rec)
			}
		}
		if !found {
			t.Fatalf("LLM did not preserve %s's code", thread)
		}
	}
	if m.Count() != 2 {
		t.Fatalf("expected two deduplicated memories, got %d", m.Count())
	}
	if batch := readMemoryBatch(t, m.history, "50"); batch != nil {
		t.Fatal("history remains uncommitted")
	}
	for thread, own := range map[string]string{"chat-a": a, "chat-b": b} {
		ranked := m.recallIndexed([]string{"deployment verification code"}, 5, thread)
		if len(ranked) != 1 || !strings.Contains(ranked[0].Record.Content, own) {
			t.Fatalf("scoped recall incorrect for %s", thread)
		}
		answer, _ := runCodexMemoryToolLoop(t, provider, "Answer from supplied memory only. Reply with the exact code and nothing else.",
			m.BuildContext([]MemoryRecord{ranked[0].Record})+"\nWhat is my deployment verification code?", nil,
			func(NativeToolCall) ToolResponse { t.Fatal("unexpected tool call"); return ToolResponse{} })
		if !strings.Contains(answer, own) {
			t.Fatalf("LLM did not use scoped learned memory: %q", answer)
		}
	}
	t.Log("verified: two source threads, two committed batches, durable restart/replayed-write deduplication, provenance and scoped LLM recall")
}

func runCodexScopedHistorySearch(t *testing.T, provider LLMProvider) {
	m, h := newHistoryTestStore(t)
	a := "amber-" + newULID()
	b := "violet-" + newULID()
	writeMemoryHistory(t, h, "chat-a", SessionEntry{Role: "user", Content: "The agreed deployment recovery code is " + a})
	writeMemoryHistory(t, h, "chat-b", SessionEntry{Role: "user", Content: "The agreed deployment recovery code is " + b})
	registry := NewToolRegistry("")
	registerSystemTools(registry, m, h.config)
	native := registry.NativeTools(map[string]bool{"history_search": true}, nil, false)
	query := func(thread string) (string, []string) {
		return runCodexMemoryToolLoop(t, provider,
			"Answer using retained history. If a requested fact is not in your current context, use history_search. Never invent codes. If unavailable, reply NOT_FOUND.",
			"What deployment recovery code did we agree on earlier? Return the exact code, or NOT_FOUND if unavailable.", native,
			func(call NativeToolCall) ToolResponse {
				if call.Name != "history_search" {
					t.Fatalf("unexpected tool %s", call.Name)
				}
				r, ok := registry.DispatchContext(withMemoryCaller(context.Background(), thread), call.Name, call.Args)
				if !ok || r.IsError {
					t.Fatalf("history search failed: %s", r.Text)
				}
				if thread == "chat-a" && strings.Contains(r.Text, b) {
					t.Fatal("sibling history leaked into tool result")
				}
				return r
			})
	}
	answer, calls := query("chat-a")
	if len(calls) == 0 || !strings.Contains(answer, a) || strings.Contains(answer, b) {
		t.Fatalf("fallback failed: calls=%v answer=%q", calls, answer)
	}
	if m.Count() != 0 {
		t.Fatal("history fallback unexpectedly wrote memories")
	}
	if _, err := m.Remember("Deployment recovery code "+a, nil, 1, MemoryMetadata{Scope: "chat-a"}); err != nil {
		t.Fatal(err)
	}
	if n, err := m.ForgetSource("chat-a", "synthetic privacy test"); err != nil || n != 1 {
		t.Fatalf("forget failed: %d %v", n, err)
	}
	if got := m.recallIndexed([]string{"deployment recovery code"}, 5, "chat-a"); len(got) != 0 {
		t.Fatal("forgotten fact still recalled")
	}
	// Fresh model context: forgetting does not claim to erase already-sent text.
	answer, calls = query("chat-a")
	if len(calls) == 0 || !strings.Contains(answer, "NOT_FOUND") || strings.Contains(answer, a) || strings.Contains(answer, b) {
		t.Fatalf("forgotten source resurfaced: %q", answer)
	}
	answer, _ = query("chat-b")
	if !strings.Contains(answer, b) || strings.Contains(answer, a) {
		t.Fatalf("forgetting damaged unrelated thread: %q", answer)
	}
	t.Log("verified: LLM chose history_search, isolated sibling history, respected forgetting in fresh context, preserved unrelated history")
}
