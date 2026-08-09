package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// recordingCheckpointProvider forces the tool call that crosses the history
// checkpoint boundary, then records and delegates every continuation to the
// real provider.
type recordingCheckpointProvider struct {
	LLMProvider
	mu       sync.Mutex
	forced   bool
	response ChatResponse
	requests [][]Message
}

func (p *recordingCheckpointProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	if !p.forced {
		p.forced = true
		p.mu.Unlock()
		return p.response, nil
	}
	p.requests = append(p.requests, cloneMessages(messages))
	p.mu.Unlock()
	return p.LLMProvider.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
}

func (p *recordingCheckpointProvider) recordedRequests() [][]Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]Message, len(p.requests))
	for i := range p.requests {
		out[i] = cloneMessages(p.requests[i])
	}
	return out
}

// TestCodexPendingToolSurvivesHistoryCheckpointSmoke exercises the complete
// regression with a real Codex continuation. The first asynchronous tool is
// blocked until Core has checkpointed history. Codex must then receive that
// call and its successful result, continue to the requested lookup exactly
// once, and not repeat the acknowledgement.
//
//	RUN_CODEX_HISTORY_CHECKPOINT_SMOKE=1 go test -v -run TestCodexPendingToolSurvivesHistoryCheckpointSmoke -timeout 5m .
func TestCodexPendingToolSurvivesHistoryCheckpointSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_HISTORY_CHECKPOINT_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_HISTORY_CHECKPOINT_SMOKE=1 to run the Codex history-checkpoint smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex history-checkpoint smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Chdir(t.TempDir())
	const (
		ackCallID = "checkpoint-ack-call"
		ackTool   = "checkpoint_acknowledge"
		listTool  = "checkpoint_tasks_list"
		finalText = "CHECKPOINT_HISTORY_OK"
	)
	provider := &recordingCheckpointProvider{
		LLMProvider: NewOpenAICodexProvider(token),
		response: ChatResponse{ToolCalls: []NativeToolCall{{
			ID: ackCallID, Name: ackTool,
			Args: map[string]string{"message": "I’ll check your currently active recurring tasks.", "_reason": "Acknowledge once before lookup"},
		}}},
	}
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Complete this tiny foreground checkpoint verification directly; do not delegate, spawn, send, or evolve.",
			"",
			"# Required workflow",
			"- Call checkpoint_acknowledge exactly once.",
			"- After its successful result, never acknowledge again; call checkpoint_tasks_list exactly once.",
			"- After the list result, reply exactly CHECKPOINT_HISTORY_OK and wait.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	thinker.maxHistory = maxHistoryWorker

	ackStarted := make(chan struct{})
	releaseAck := make(chan struct{})
	var ackStartOnce sync.Once
	var ackCalls, listCalls atomic.Int32
	thinker.registry.Register(&ToolDef{
		Name:        ackTool,
		Description: "Send the one required acknowledgement. After this tool returns success, never call it again for the same request.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"message": map[string]any{"type": "string"},
			},
			"required": []string{"message"},
		},
		Handler: func(map[string]string) ToolResponse {
			ackCalls.Add(1)
			ackStartOnce.Do(func() { close(ackStarted) })
			select {
			case <-releaseAck:
				return ToolResponse{Text: `{"delivered":true,"instruction":"acknowledgement complete; do not acknowledge again"}`}
			case <-time.After(30 * time.Second):
				return ToolResponse{Text: "error: test acknowledgement release timed out", IsError: true}
			}
		},
	})
	thinker.registry.Register(&ToolDef{
		Name:        listTool,
		Description: "List active recurring tasks after acknowledgement delivery. Call exactly once.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		Handler: func(map[string]string) ToolResponse {
			listCalls.Add(1)
			return ToolResponse{Text: `{"active_tasks":["hourly Gmail review"]}`}
		},
	})

	// A worker checkpoints above 24 retained history messages. Alternating
	// completed turns keep this realistic while ensuring the forced async call
	// is the newest assistant message at the boundary.
	for i := 0; i < 12; i++ {
		thinker.messages = append(thinker.messages,
			Message{Role: "user", Content: "Historical completed request."},
			Message{Role: "assistant", Content: "Historical request completed."},
		)
	}

	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Error("Codex checkpoint thinker did not stop")
		}
	}()

	select {
	case <-ackStarted:
	case <-time.After(10 * time.Second):
		t.Fatal("forced acknowledgement tool did not start")
	}

	checkpointDeadline := time.Now().Add(10 * time.Second)
	sawCheckpoint := false
	for !sawCheckpoint && time.Now().Before(checkpointDeadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.Type != "llm.prompt_cache_reset" {
				continue
			}
			var data map[string]any
			if json.Unmarshal(event.Data, &data) == nil && data["reset_reason"] == "history_checkpoint" {
				sawCheckpoint = true
				break
			}
		}
		if !sawCheckpoint {
			time.Sleep(20 * time.Millisecond)
		}
	}
	if !sawCheckpoint {
		t.Fatal("history checkpoint did not occur while acknowledgement was pending")
	}
	close(releaseAck)

	deadline := time.Now().Add(150 * time.Second)
	sawFinal := false
	for time.Now().Before(deadline) {
		if ackCalls.Load() > 1 {
			t.Fatalf("Codex repeated acknowledgement after its receipt: calls=%d", ackCalls.Load())
		}
		if listCalls.Load() > 1 {
			t.Fatalf("Codex repeated task lookup: calls=%d", listCalls.Load())
		}
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.Type != "llm.done" || event.ThreadID != "main" {
				continue
			}
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) == nil && strings.Contains(data.Message, finalText) {
				sawFinal = true
				break
			}
		}
		if sawFinal && listCalls.Load() == 1 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !sawFinal || listCalls.Load() != 1 || ackCalls.Load() != 1 {
		t.Fatalf("live Codex workflow incomplete: checkpoint=%v acknowledgement_calls=%d list_calls=%d final=%v",
			sawCheckpoint, ackCalls.Load(), listCalls.Load(), sawFinal)
	}

	requests := provider.recordedRequests()
	if len(requests) == 0 {
		t.Fatal("real Codex received no continuation request")
	}
	callSeen, resultSeen := false, false
	for _, message := range requests[0] {
		for _, call := range message.ToolCalls {
			if call.ID == ackCallID && call.Name == ackTool {
				callSeen = true
			}
		}
		for _, result := range message.ToolResults {
			if result.CallID == ackCallID && strings.Contains(result.Content, `"delivered":true`) {
				resultSeen = true
			}
		}
	}
	if !callSeen || !resultSeen {
		t.Fatalf("first real Codex continuation lost checkpoint pair: call=%v result=%v", callSeen, resultSeen)
	}
	t.Logf("Codex retained the pending call/result across checkpoint and completed without duplicates: acknowledgement=%d lookup=%d requests=%d",
		ackCalls.Load(), listCalls.Load(), len(requests))
}
