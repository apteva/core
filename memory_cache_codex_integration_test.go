package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type recordingMemoryCacheProvider struct {
	LLMProvider
	mu       sync.Mutex
	requests [][]Message
}

func (p *recordingMemoryCacheProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	p.requests = append(p.requests, cloneMessages(messages))
	p.mu.Unlock()
	return p.LLMProvider.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
}

func (p *recordingMemoryCacheProvider) capturedRequests() [][]Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]Message, len(p.requests))
	for i := range p.requests {
		out[i] = cloneMessages(p.requests[i])
	}
	return out
}

// TestIntegration_CodexMultiSkillToolCycleKeepsMemoryAndCacheStable proves
// the complete live behavior behind long browser-style workflows:
//
//   - one external event selects relevant guidance from many large records,
//   - irrelevant guidance and the total memory block remain bounded,
//   - six result-driven model continuations reuse one memory snapshot,
//   - the stable prompt prefix and cache epoch do not churn per result,
//   - live Codex uses the recalled secret to complete the tool workflow.
//
// Run:
//
//	RUN_CODEX_MEMORY_CACHE_SMOKE=1 go test -v -run TestIntegration_CodexMultiSkillToolCycleKeepsMemoryAndCacheStable -timeout 8m .
func TestIntegration_CodexMultiSkillToolCycleKeepsMemoryAndCacheStable(t *testing.T) {
	if os.Getenv("RUN_CODEX_MEMORY_CACHE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_MEMORY_CACHE_SMOKE=1 to run the Codex memory/cache smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex memory/cache smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := &recordingMemoryCacheProvider{LLMProvider: NewOpenAICodexProvider(token)}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Supervise the live memory/cache verification worker without doing its work.",
		Mode:      ModeAutonomous,
	}
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()

	const (
		threadID  = "codex-memory-cache-worker"
		toolName  = "browser_validation_step"
		secret    = "VIOLET-SEAL-927"
		finalText = "CODEX_MEMORY_CACHE_OK"
		steps     = 6
	)

	relevantRecords := []struct {
		id      string
		content string
		tags    []string
	}{
		{
			id: "skill_patreon_validation",
			content: "PATREON_VALIDATION_SKILL\nThe Patreon browser validation code is " + secret +
				". Supply this exact code to every browser_validation_step call.\n" +
				strings.Repeat("Patreon browser validation procedure for an existing draft. ", 125),
			tags: []string{"skill", "patreon", "browser", "validation"},
		},
		{
			id: "skill_computer_observation",
			content: "COMPUTER_OBSERVATION_SKILL\nUse each structured browser observation result before selecting the next step.\n" +
				strings.Repeat("Computer browser observation procedure and structured tool result guidance. ", 110),
			tags: []string{"skill", "computer", "browser", "observation"},
		},
	}
	for _, rec := range relevantRecords {
		if _, err := parent.memory.RememberWithID(rec.id, rec.content, rec.tags, 0.95); err != nil {
			t.Fatalf("remember relevant skill: %v", err)
		}
	}
	for i := 0; i < 10; i++ {
		content := fmt.Sprintf("UNRELATED_SKILL_%02d\n", i) + strings.Repeat("Payroll tax kitchen inventory warehouse archive. ", 130)
		if _, err := parent.memory.RememberWithID(
			fmt.Sprintf("skill_unrelated_%02d", i), content,
			[]string{"skill", "unrelated", "archive"}, 0.9,
		); err != nil {
			t.Fatalf("remember distractor skill: %v", err)
		}
	}

	var toolMu sync.Mutex
	toolCalls := 0
	parent.registry.Register(&ToolDef{
		Name:        toolName,
		Description: "Advance exactly one step of the current browser validation. Start at step 1, then use next_step from each successful result. The code must come from recalled guidance. Call once per model turn.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"step": map[string]any{"type": "string", "description": "Current sequential step number"},
				"code": map[string]any{"type": "string", "description": "Exact validation code from recalled guidance"},
			},
			"required": []string{"step", "code"},
		},
		Handler: func(args map[string]string) ToolResponse {
			step, err := strconv.Atoi(args["step"])
			if err != nil {
				return ToolResponse{Text: `{"ok":false,"error":"step must be a decimal string"}`, IsError: true}
			}
			toolMu.Lock()
			defer toolMu.Unlock()
			expected := toolCalls + 1
			if args["code"] != secret || step != expected {
				return ToolResponse{
					Text:    fmt.Sprintf(`{"ok":false,"expected_step":%d,"instruction":"retry with the exact recalled code and expected step"}`, expected),
					IsError: true,
				}
			}
			toolCalls++
			if step == steps {
				return ToolResponse{Text: `{"ok":true,"complete":true,"instruction":"Reply exactly CODEX_MEMORY_CACHE_OK and do not call another tool."}` + strings.Repeat(" final-state", 1800)}
			}
			return ToolResponse{Text: fmt.Sprintf(`{"ok":true,"completed_step":%d,"next_step":"%d","instruction":"Call browser_validation_step once for next_step using the same recalled code."}`, step, step+1) + strings.Repeat(" observation-state", 1500)}
		},
	})

	directive := strings.Join([]string{
		"# Role",
		"Complete this bounded validation directly. Do not spawn, send, evolve, or delegate.",
		"",
		"# Workflow",
		"On the external request, read the automatically recalled Patreon and Computer guidance.",
		"Call browser_validation_step for step 1 using the exact recalled validation code.",
		"After each successful result, call it exactly once for next_step. Do not skip or repeat steps.",
		"When the tool reports complete=true, reply exactly CODEX_MEMORY_CACHE_OK with no other text and wait for events.",
	}, "\n")
	if err := parent.threads.SpawnWithOpts(
		threadID, directive, []string{toolName, "pace"},
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatalf("spawn live worker: %v", err)
	}
	thread := parent.threads.threads[threadID]
	if thread == nil {
		t.Fatal("live worker missing")
	}
	// Keep the wake deliberately vague: the worker's standing directive must
	// remain part of RAG relevance instead of being replaced by this event.
	thread.Thinker.InjectConsole("Begin now and use the shared operating guidance.")
	go thread.Thinker.Run()

	deadline := time.Now().Add(6 * time.Minute)
	sawFinal := false
	for time.Now().Before(deadline) {
		events, _ := parent.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != threadID || event.Type != "llm.done" {
				continue
			}
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) == nil && strings.Contains(data.Message, finalText) {
				sawFinal = true
				break
			}
		}
		if sawFinal {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	toolMu.Lock()
	completedCalls := toolCalls
	toolMu.Unlock()
	if !sawFinal || completedCalls != steps {
		requests := provider.capturedRequests()
		events, _ := parent.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != threadID {
				continue
			}
			switch event.Type {
			case "memory.recall", "llm.error", "llm.done":
				t.Logf("%s: %s", event.Type, event.Data)
			}
		}
		t.Fatalf("live workflow incomplete: final=%v successful_tool_calls=%d/%d model_requests=%d", sawFinal, completedCalls, steps, len(requests))
	}

	events, _ := parent.telemetry.StoredEvents(0)
	recalls := 0
	stableHashes := map[string]bool{}
	cacheEpochs := map[uint64]bool{}
	var cachedTokens []int
	for _, event := range events {
		if event.ThreadID != threadID {
			continue
		}
		switch event.Type {
		case "memory.recall":
			recalls++
			var data map[string]any
			if json.Unmarshal(event.Data, &data) != nil {
				t.Fatalf("decode memory telemetry: %s", event.Data)
			}
			if chars, _ := data["chars"].(float64); int(chars) > automaticMemoryRecallMaxChars {
				t.Fatalf("memory context chars = %d, limit = %d", int(chars), automaticMemoryRecallMaxChars)
			}
			if !strings.Contains(string(event.Data), "skill_patreon_validation") || strings.Contains(string(event.Data), "skill_unrelated") {
				t.Fatalf("unexpected recalled records: %s", event.Data)
			}
		case "llm.prompt_cache_reset":
			if strings.Contains(string(event.Data), "tool_result_retention_expired") {
				t.Fatalf("rolling retention reset observed: %s", event.Data)
			}
		case "llm.done":
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) == nil {
				stableHashes[data.PromptCacheStablePrefixHash] = true
				cacheEpochs[data.PromptCacheEpoch] = true
				cachedTokens = append(cachedTokens, data.TokensCached)
			}
		}
	}
	if recalls != 1 {
		t.Fatalf("memory recalls = %d, want exactly one external-cycle retrieval", recalls)
	}
	delete(stableHashes, "")
	if len(stableHashes) != 1 {
		t.Fatalf("stable prefix hashes changed: %v", stableHashes)
	}
	if len(cacheEpochs) != 1 {
		t.Fatalf("prompt cache epoch churned during six result continuations: %v", cacheEpochs)
	}

	requests := provider.capturedRequests()
	if len(requests) < steps+1 {
		t.Fatalf("recorded model requests = %d, want at least %d", len(requests), steps+1)
	}
	for i, request := range requests {
		encoded, _ := json.Marshal(request)
		body := string(encoded)
		if strings.Count(body, "[memories — surfaced") != 1 {
			t.Fatalf("request %d contains a duplicated/missing memory block", i+1)
		}
		if !strings.Contains(body, "PATREON_VALIDATION_SKILL") || !strings.Contains(body, secret) {
			t.Fatalf("request %d lost the selected skill", i+1)
		}
		if strings.Contains(body, "UNRELATED_SKILL_") {
			t.Fatalf("request %d contains an unrelated skill", i+1)
		}
	}

	t.Logf("Codex completed %d sequential tool calls with one memory retrieval; requests=%d cache_epochs=%v cached_tokens=%v",
		completedCalls, len(requests), cacheEpochs, cachedTokens)
}
