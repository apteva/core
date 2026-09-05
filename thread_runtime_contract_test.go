package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type terminalDoneProvider struct {
	*recordingThreadEventProvider
}

func (p *terminalDoneProvider) Chat(_ context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	return ChatResponse{ToolCalls: []NativeToolCall{{
		ID: "runtime-done", Name: "done", Args: map[string]string{"message": "runtime final"},
	}}}, nil
}

func hasNativeToolNamed(tools []NativeTool, name string) bool {
	for _, tool := range tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

func TestCoreThreadRuntimeContractUsesNoModelLifecycleBookkeeping(t *testing.T) {
	t.Setenv("APTEVA_MODEL_MCP_MANAGEMENT", "")
	registry := NewToolRegistry("test")
	if registry.Get("resolve_event") != nil {
		t.Fatal("resolve_event must not be a model-facing Core tool")
	}
	if hasNativeToolNamed(registry.NativeTools(nil, nil), "done") {
		t.Fatal("root main must not receive the worker-only done tool")
	}
	if strings.Contains(registry.CoreDocs(true), "  done —") {
		t.Fatal("root main documentation exposes worker-only done")
	}
	workerTools := map[string]bool{"send": true, "pace": true, "done": true}
	if !hasNativeToolNamed(registry.NativeTools(workerTools, nil), "done") {
		t.Fatal("ordinary workers must receive done")
	}
	done := registry.Get("done")
	if done == nil || done.TurnDisposition != ToolTurnTerminate {
		t.Fatalf("done disposition = %#v, want terminate", done)
	}
	pace := registry.Get("pace")
	if pace == nil || pace.TurnDisposition != ToolTurnYield {
		t.Fatalf("pace disposition = %#v, want yield", pace)
	}
}

func TestModelMCPManagementRequiresExplicitHostCapability(t *testing.T) {
	t.Setenv("APTEVA_MODEL_MCP_MANAGEMENT", "")
	disabled := NewToolRegistry("test")
	if disabled.Get("connect") != nil || disabled.Get("disconnect") != nil {
		t.Fatal("model-facing MCP mutation tools must be hidden by default")
	}
	if disabled.Get("list_connected") == nil {
		t.Fatal("read-only MCP inventory should remain available")
	}

	thinker := newTestThinkerFull()
	defer thinker.Stop()
	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "disconnect", Args: map[string]string{"name": "example"}, NativeID: "disabled-disconnect",
	}}, nil)
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, "not enabled") {
		t.Fatalf("disabled administrative call result = %+v", results)
	}

	t.Setenv("APTEVA_MODEL_MCP_MANAGEMENT", "true")
	enabled := NewToolRegistry("test")
	if enabled.Get("connect") == nil || enabled.Get("disconnect") == nil {
		t.Fatal("explicit host capability did not expose MCP mutation tools")
	}
}

func TestInlineToolResultsContinueExceptSuccessfulPace(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	handler := mainToolHandler(thinker)

	_, _, results := handler(thinker, []toolCall{{
		Name: "pace", Args: map[string]string{"sleep": "1h"}, NativeID: "pace-ok",
	}}, nil)
	if len(results) != 1 || results[0].IsError || thinker.kickNextTurn {
		t.Fatalf("successful pace must yield: kick=%v results=%+v", thinker.kickNextTurn, results)
	}

	_, _, results = handler(thinker, []toolCall{{
		Name: "pace", Args: map[string]string{"sleep": "not-a-duration"}, NativeID: "pace-error",
	}}, nil)
	if len(results) != 1 || !results[0].IsError || !thinker.kickNextTurn {
		t.Fatalf("failed pace must continue: kick=%v results=%+v", thinker.kickNextTurn, results)
	}

	thinker.kickNextTurn = false
	_, _, results = handler(thinker, []toolCall{
		{Name: "pace", Args: map[string]string{"sleep": "1h"}, NativeID: "pace-mixed"},
		{Name: "list_connected", Args: map[string]string{}, NativeID: "list-mixed"},
	}, nil)
	if len(results) != 2 || !thinker.kickNextTurn {
		t.Fatalf("ordinary receipt in a mixed batch must continue: kick=%v results=%+v", thinker.kickNextTurn, results)
	}
}

func TestInlineContinuationGuardCountsOnlyIdenticalNoProgress(t *testing.T) {
	thinker := &Thinker{registry: NewToolRegistry("test")}
	apply := func(id, argument, result string) {
		thinker.kickNextTurn = false
		thinker.applyInlineToolTurnDisposition(
			[]toolCall{{Name: "inline_test", Args: map[string]string{"value": argument}, NativeID: id}},
			[]ToolResult{{ToolName: "inline_test", CallID: id, Content: result}},
			false,
		)
	}
	for i := 0; i < 5; i++ {
		apply(fmt.Sprintf("progress-%d", i), fmt.Sprint(i), "ok")
		if !thinker.kickNextTurn {
			t.Fatalf("different arguments were mistaken for no progress at %d", i)
		}
	}
	thinker.resetInlineToolContinuation()
	for i := 1; i <= maxRepeatedInlineToolResults; i++ {
		apply(fmt.Sprintf("repeat-%d", i), "same", "same receipt")
		if want := i < maxRepeatedInlineToolResults; thinker.kickNextTurn != want {
			t.Fatalf("repeat %d continuation=%v want %v", i, thinker.kickNextTurn, want)
		}
	}
	if len(thinker.inlineResultFingerprint) != 64 {
		t.Fatal("continuation guard should retain only a SHA-256 digest")
	}
	apply("changed-result", "same", "new information")
	if !thinker.kickNextTurn {
		t.Fatal("new result information must restart normal continuation")
	}
}

func TestDoneMustBeAloneAndDoesNotTerminateMixedBatch(t *testing.T) {
	parent := newTestThinkerFull()
	defer parent.Stop()
	if err := parent.threads.SpawnWithOpts("one-shot", "Return one result.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	parent.drainEventTexts()
	worker := parent.threads.threads["one-shot"]
	worker.Thinker.addEventExecutions([]string{"exe-mixed-done"})

	_, _, results := threadToolHandler(worker, parent.threads)(worker.Thinker, []toolCall{
		{Name: "done", Args: map[string]string{"message": "final"}, NativeID: "done-mixed"},
		{Name: "pace", Args: map[string]string{"sleep": "1h"}, NativeID: "pace-mixed"},
	}, nil)
	if len(results) != 2 || !results[0].IsError || !strings.Contains(results[0].Content, "done must be called alone") {
		t.Fatalf("mixed done results = %+v", results)
	}
	if !worker.Thinker.kickNextTurn {
		t.Fatal("rejected mixed done must give the model a correction turn")
	}
	select {
	case <-worker.Thinker.quit:
		t.Fatal("mixed done batch terminated the worker")
	default:
	}
	if got := parent.drainEventTexts(); len(got) != 0 {
		t.Fatalf("mixed done batch sent a false completion: %v", got)
	}
	var failed, succeeded int
	events, _ := parent.telemetry.StoredEvents(0)
	for _, event := range events {
		if event.ThreadID != "one-shot" || event.Type != "tool.result" {
			continue
		}
		var data ToolResultData
		if json.Unmarshal(event.Data, &data) == nil && data.ID == "done-mixed" {
			if data.Success {
				succeeded++
			} else {
				failed++
			}
		}
	}
	if failed != 1 || succeeded != 0 {
		t.Fatalf("mixed done terminal results: failed=%d successful=%d", failed, succeeded)
	}
}

func TestDonePersistenceFailureReturnsErrorAndKeepsWorkerAlive(t *testing.T) {
	parent := newTestThinkerFull()
	defer parent.Stop()
	blocked := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blocked, []byte("blocked"), 0600); err != nil {
		t.Fatal(err)
	}
	parent.telemetry.durableDir = filepath.Join(blocked, "outbox")
	if err := parent.threads.SpawnWithOpts("one-shot", "Return one result.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	parent.drainEventTexts()
	worker := parent.threads.threads["one-shot"]

	_, _, results := threadToolHandler(worker, parent.threads)(worker.Thinker, []toolCall{{
		Name: "done", Args: map[string]string{"message": "must not escape"}, NativeID: "done-persist-failure",
	}}, nil)
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, "persist terminal tool result") {
		t.Fatalf("failed terminal persistence results=%+v", results)
	}
	select {
	case <-worker.Thinker.quit:
		t.Fatal("worker terminated despite terminal receipt persistence failure")
	default:
	}
	if got := parent.drainEventTexts(); len(got) != 0 {
		t.Fatalf("failed done delivered a parent completion: %v", got)
	}
}

func TestOneShotDoneReturnsFinalOnceWhilePersistentWorkerStaysAlive(t *testing.T) {
	parent := newTestThinkerFull()
	defer parent.Stop()
	for _, id := range []string{"one-shot", "persistent"} {
		if err := parent.threads.SpawnWithOpts(id, "Perform assigned work.", nil, SpawnOpts{DeferRun: true}); err != nil {
			t.Fatalf("spawn %s: %v", id, err)
		}
	}
	parent.drainEventTexts()

	oneShot := parent.threads.threads["one-shot"]
	oneShot.Thinker.addEventExecutions([]string{"exe-done-final"})
	_, _, results := threadToolHandler(oneShot, parent.threads)(oneShot.Thinker, []toolCall{{
		Name: "done", Args: map[string]string{"message": "authoritative final result"}, NativeID: "done-final",
	}}, nil)
	if len(results) != 1 || results[0].IsError || results[0].Content != "stopping" {
		t.Fatalf("done result = %+v", results)
	}
	select {
	case <-oneShot.Thinker.quit:
	default:
		t.Fatal("one-shot done did not terminate the worker")
	}
	if got := parent.drainEventTexts(); len(got) != 1 || got[0] != "[thread:one-shot done] authoritative final result" {
		t.Fatalf("one-shot parent result = %v", got)
	}
	var terminalResults []ToolResultData
	events, _ := parent.telemetry.StoredEvents(0)
	for _, event := range events {
		if event.ThreadID != "one-shot" || event.Type != "tool.result" {
			continue
		}
		var data ToolResultData
		if json.Unmarshal(event.Data, &data) == nil && data.ID == "done-final" {
			terminalResults = append(terminalResults, data)
		}
	}
	if len(terminalResults) != 1 || !terminalResults[0].Success || terminalResults[0].Name != "done" ||
		len(terminalResults[0].ExecutionIDs) != 1 || terminalResults[0].ExecutionIDs[0] != "exe-done-final" {
		t.Fatalf("successful done telemetry = %+v", terminalResults)
	}

	persistent := parent.threads.threads["persistent"]
	_, _, results = threadToolHandler(persistent, parent.threads)(persistent.Thinker, []toolCall{
		{Name: "send", Args: map[string]string{"id": "parent", "message": "cycle result"}, NativeID: "send-cycle"},
		{Name: "pace", Args: map[string]string{"sleep": "1h"}, NativeID: "pace-cycle"},
	}, nil)
	if len(results) != 2 || !persistent.Thinker.kickNextTurn {
		t.Fatalf("persistent report results = %+v kick=%v", results, persistent.Thinker.kickNextTurn)
	}
	select {
	case <-persistent.Thinker.quit:
		t.Fatal("persistent send+pace terminated the worker")
	default:
	}
	if got := parent.drainEventTexts(); len(got) != 1 || got[0] != "[from:persistent] cycle result" {
		t.Fatalf("persistent parent report = %v", got)
	}
}

func TestDoneRuntimeRecordsResultBeforeParentDeliveryAndThreadDone(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := &terminalDoneProvider{recordingThreadEventProvider: newRecordingThreadEventProvider()}
	config := &Config{path: "config.json", Directive: "Coordinate workers.", Mode: ModeAutonomous}
	parent := NewThinker("", provider, config)
	defer parent.Stop()
	defer parent.threads.KillAll()

	if err := parent.threads.SpawnWithOpts("terminal-worker", "Finish once.", nil, SpawnOpts{
		DeferRun: true, ExecutionIDs: []string{"exe-runtime-done"},
	}); err != nil {
		t.Fatalf("spawn terminal worker: %v", err)
	}
	parent.drainEventTexts()
	if err := config.SaveThread(PersistentThread{
		ID: "terminal-worker", ParentID: "main", Directive: "Finish once.",
	}); err != nil {
		t.Fatalf("persist terminal worker: %v", err)
	}
	worker := parent.threads.threads["terminal-worker"]
	worker.runStarted = true
	go worker.Thinker.Run()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && parent.threads.Count() != 0 {
		time.Sleep(10 * time.Millisecond)
	}
	if parent.threads.Count() != 0 {
		t.Fatal("done worker did not terminate")
	}
	for time.Now().Before(deadline) {
		events, _ := parent.telemetry.StoredEvents(0)
		seen := false
		for _, event := range events {
			if event.ThreadID == "terminal-worker" && event.Type == "thread.done" {
				seen = true
				break
			}
		}
		if seen {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	events, _ := parent.telemetry.StoredEvents(0)
	resultIndex, doneIndex, successfulResults := -1, -1, 0
	for index, event := range events {
		if event.ThreadID != "terminal-worker" {
			continue
		}
		switch event.Type {
		case "tool.result":
			var data ToolResultData
			if json.Unmarshal(event.Data, &data) == nil && data.ID == "runtime-done" {
				if data.Success && data.Name == "done" && len(data.ExecutionIDs) == 1 && data.ExecutionIDs[0] == "exe-runtime-done" {
					successfulResults++
					resultIndex = index
				}
			}
		case "thread.done":
			doneIndex = index
		}
	}
	if successfulResults != 1 || resultIndex < 0 || doneIndex < 0 || resultIndex >= doneIndex {
		t.Fatalf("terminal telemetry order: results=%d result_index=%d thread_done_index=%d events=%+v", successfulResults, resultIndex, doneIndex, events)
	}
	if got := parent.drainEventTexts(); len(got) != 1 || got[0] != "[thread:terminal-worker done] runtime final" {
		t.Fatalf("parent terminal delivery=%v", got)
	}
	if _, exists := persistentThreadByID(config.GetThreads(), "terminal-worker"); exists {
		t.Fatal("completed worker remained persisted for restart")
	}

	restarted := NewThinker("", provider, config)
	defer restarted.Stop()
	defer restarted.threads.KillAll()
	if restarted.threads.Count() != 0 {
		t.Fatal("completed worker was restored and could duplicate completion")
	}
	if got := restarted.drainEventTexts(); len(got) != 1 || got[0] != "[thread:terminal-worker done] runtime final" {
		t.Fatalf("restart must replay the completion until parent history acknowledges it: %v", got)
	}
}
