package core

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestPendingToolBarrierRoutineNotificationWaitsForFastResult(t *testing.T) {
	th := retryTestThinker(nil)
	th.registry = NewToolRegistry("")
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	th.registry.Register(&ToolDef{Name: "fast_claim", HandlerContext: func(_ context.Context, _ map[string]string) ToolResponse {
		started <- struct{}{}
		th.bus.Publish(Event{Type: EventInbox, To: "main", From: "processes", Text: "next workflow step is ready"})
		<-release
		return ToolResponse{Text: "authoritative claim result"}
	}})
	queueTool(th, toolCall{Name: "fast_claim", NativeID: "claim-1", Args: map[string]string{}})
	<-started
	var results []ToolResult
	var texts []string
	var parts []ContentPart
	done := make(chan struct{})
	go func() { th.waitForPendingTools(&results, &texts, &parts, time.Second); close(done) }()
	select {
	case <-done:
		t.Fatal("routine notification bypassed pending-tool barrier")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("barrier did not finish after fast result")
	}
	if len(results) != 1 || results[0].CallID != "claim-1" {
		t.Fatalf("results=%+v", results)
	}
	if len(texts) < 1 || !strings.Contains(texts[0], "next workflow") {
		t.Fatalf("texts=%q", texts)
	}
	if th.pendingToolCount() != 0 {
		t.Fatal("tool remained pending")
	}
}

func TestPendingToolBarrierLateResultTextDoesNotBypass(t *testing.T) {
	th := retryTestThinker(nil)
	th.pendingTools.Store("claim-2", "fast_claim")
	th.bus.Publish(Event{Type: EventInbox, To: "main", From: "processes", Text: "[late-result] Tool previous completed: ok"})
	var results []ToolResult
	var texts []string
	var parts []ContentPart
	done := make(chan struct{})
	go func() { th.waitForPendingTools(&results, &texts, &parts, 40*time.Millisecond); close(done) }()
	select {
	case <-done:
		t.Fatal("late-result text bypassed pending-tool barrier")
	case <-time.After(10 * time.Millisecond):
	}
	<-done
	if len(results) != 0 || len(texts) != 1 {
		t.Fatalf("results=%+v texts=%q", results, texts)
	}
}
