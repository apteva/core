package core

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestCodexToolReadySmoke asks the live Codex provider for several tool
// calls in one response and checks that every call is reported ready, once,
// under the id the response returns it with. The log shows when each call
// started streaming and when it became ready.
//
//	RUN_CODEX_TOOL_READY_SMOKE=1 go test -run TestCodexToolReadySmoke -timeout 5m -v .
//	CODEX_TOOL_READY_MODEL overrides the model (default gpt-6.2-sol).
func TestCodexToolReadySmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_TOOL_READY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_TOOL_READY_SMOKE=1 to run the Codex tool-ready smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex tool-ready smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	model := os.Getenv("CODEX_TOOL_READY_MODEL")
	if model == "" {
		model = "gpt-6.2-sol"
	}

	const want = 6
	var mu sync.Mutex
	start := time.Now()
	firstChunk := map[string]time.Duration{}
	ready := map[string]time.Duration{}
	readyCount := map[string]int{}
	var order []string
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = withToolReadyObserver(ctx, func(_, callID string, _ int) {
		mu.Lock()
		defer mu.Unlock()
		ready[callID] = time.Since(start)
		readyCount[callID]++
		order = append(order, callID)
	})
	tools := []NativeTool{{
		Name:        "todo_create",
		Description: "Create one todo item.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"title", "notes"},
			"properties": map[string]any{
				"title": map[string]any{"type": "string"},
				"notes": map[string]any{"type": "string", "description": "Three full sentences describing the work."},
			},
		},
	}}
	resp, err := NewOpenAICodexProvider(token).Chat(ctx, []Message{
		{Role: "system", Content: "You create todo items with the todo_create tool. Issue every requested call in a single response, in parallel. Do not answer in text."},
		{Role: "user", Content: fmt.Sprintf("Create %d todos for planning a product launch. Call todo_create %d times in this one response.", want, want)},
	}, model, tools, nil, nil, func(_, callID, _ string) {
		mu.Lock()
		defer mu.Unlock()
		if _, seen := firstChunk[callID]; !seen {
			firstChunk[callID] = time.Since(start)
		}
	})
	done := time.Since(start)
	if err != nil {
		t.Fatalf("Codex tool-ready smoke failed: %v", err)
	}
	if len(resp.ToolCalls) < 2 {
		t.Fatalf("model returned %d tool calls, need several in one response (text=%q)", len(resp.ToolCalls), resp.Text)
	}
	if len(order) != len(resp.ToolCalls) {
		t.Fatalf("ready reports = %d, tool calls = %d", len(order), len(resp.ToolCalls))
	}
	var lines []string
	for i, call := range resp.ToolCalls {
		if readyCount[call.ID] != 1 {
			t.Fatalf("call %s reported ready %d times, want 1", call.ID, readyCount[call.ID])
		}
		if order[i] != call.ID {
			t.Fatalf("ready order %v does not follow the response's call order", order)
		}
		lines = append(lines, fmt.Sprintf("call %d: first args %6dms  ready %6dms  waited for stream end %6dms",
			i+1, firstChunk[call.ID].Milliseconds(), ready[call.ID].Milliseconds(), (done-ready[call.ID]).Milliseconds()))
	}
	t.Logf("model=%s calls=%d stream ended at %dms\n%s", model, len(resp.ToolCalls), done.Milliseconds(), strings.Join(lines, "\n"))
}
