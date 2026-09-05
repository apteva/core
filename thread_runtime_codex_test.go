package core

import (
	"encoding/json"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type fixedModelProvider struct {
	LLMProvider
	model string
}

func (p *fixedModelProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{
		ModelLarge: p.model, ModelMedium: p.model, ModelSmall: p.model,
	}
}

// TestCodexTerraOneShotWorkerReturnsFinalThroughDoneSmoke exercises the full
// runtime contract against a real model: a worker consumes an asynchronous
// tool result, returns its final result once through done(message), and is
// removed without a duplicate send.
//
// RUN_CODEX_THREAD_LIFECYCLE_SMOKE=1 go test -v -run TestCodexTerraOneShotWorkerReturnsFinalThroughDoneSmoke -timeout 5m .
func TestCodexTerraOneShotWorkerReturnsFinalThroughDoneSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_THREAD_LIFECYCLE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_THREAD_LIFECYCLE_SMOKE=1 to run the Codex/Terra thread lifecycle smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex/Terra thread lifecycle smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Chdir(t.TempDir())
	provider := &fixedModelProvider{
		LLMProvider: NewOpenAICodexProvider(token),
		model:       "gpt-5.6-terra",
	}
	thinker := NewThinker("", provider, &Config{Directive: "Coordinate worker ownership.", Mode: ModeAutonomous})
	defer thinker.Stop()
	defer thinker.threads.KillAll()

	const toolName = "contract_lookup"
	const marker = "FINAL-TERRA-739"
	var lookupCalls atomic.Int32
	thinker.registry.Register(&ToolDef{
		Name:        toolName,
		Description: "Return the authoritative marker for a one-shot contract test.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"key": map[string]any{"type": "string"},
			},
			"required": []string{"key"},
		},
		Handler: func(args map[string]string) ToolResponse {
			lookupCalls.Add(1)
			return ToolResponse{Text: `{"ok":true,"marker":"` + marker + `"}`}
		},
	})

	started := time.Now()
	err := thinker.threads.SpawnWithOpts(
		"terra-one-shot",
		"This is a one-shot assignment. Call contract_lookup exactly once with key alpha. After its result arrives, call done alone with message exactly \""+marker+"\". Do not call send and do not pace.",
		[]string{toolName},
		SpawnOpts{ExecutionIDs: []string{"exe-terra-done"}},
	)
	if err != nil {
		t.Fatalf("spawn Terra worker: %v", err)
	}

	deadline := time.Now().Add(3 * time.Minute)
	for time.Now().Before(deadline) {
		if thinker.threads.Count() == 0 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if thinker.threads.Count() != 0 {
		t.Fatal("Terra one-shot worker did not terminate through done")
	}
	if lookupCalls.Load() != 1 {
		t.Fatalf("lookup calls=%d want 1", lookupCalls.Load())
	}

	var events []TelemetryEvent
	telemetryDeadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(telemetryDeadline) {
		events, _ = thinker.telemetry.StoredEvents(0)
		seenThreadDone := false
		for _, event := range events {
			if event.ThreadID == "terra-one-shot" && event.Type == "thread.done" {
				seenThreadDone = true
				break
			}
		}
		if seenThreadDone {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	doneCalls := 0
	sendCalls := 0
	doneResults := 0
	doneResultIndex := -1
	threadDoneIndex := -1
	for eventIndex, event := range events {
		if event.ThreadID != "terra-one-shot" || event.Time.Before(started) {
			continue
		}
		switch event.Type {
		case "tool.call":
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) != nil {
				continue
			}
			switch data.Name {
			case "done":
				doneCalls++
			case "send":
				sendCalls++
			}
		case "tool.result":
			var data ToolResultData
			if json.Unmarshal(event.Data, &data) == nil && data.Name == "done" {
				doneResults++
				doneResultIndex = eventIndex
				if !data.Success || len(data.ExecutionIDs) != 1 || data.ExecutionIDs[0] != "exe-terra-done" {
					t.Fatalf("done tool result = %+v", data)
				}
			}
		case "thread.done":
			threadDoneIndex = eventIndex
		}
	}
	if doneCalls != 1 || doneResults != 1 || sendCalls != 0 {
		t.Fatalf("worker terminal events: done calls=%d results=%d send=%d", doneCalls, doneResults, sendCalls)
	}
	if threadDoneIndex < 0 || doneResultIndex >= threadDoneIndex {
		t.Fatalf("done tool result index=%d must precede thread.done index=%d", doneResultIndex, threadDoneIndex)
	}

	mainEvents := thinker.drainEventTexts()
	finals := 0
	for _, event := range mainEvents {
		if strings.Contains(event, "[thread:terra-one-shot done]") {
			finals++
			if !strings.Contains(event, marker) {
				t.Fatalf("done event missing marker: %q", event)
			}
		}
		if strings.Contains(event, "[from:terra-one-shot]") {
			t.Fatalf("worker duplicated its final result through send: %q", event)
		}
	}
	if finals != 1 {
		t.Fatalf("done events=%d want 1; main events=%v", finals, mainEvents)
	}
	t.Logf("gpt-5.6-terra consumed the tool result and returned exactly one final done message")
}
