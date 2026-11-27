package core

import (
	"context"
	"strings"
	"testing"
	"time"
)

// A turn waits for every tool call it dispatched, however long each takes
// within its own timeout. Nothing is answered with a placeholder, so no
// result arrives later as text and wakes the thread for another model call.
func TestTurnWaitsForAllItsToolCalls(t *testing.T) {
	if pendingToolWaitLimit < maxConnectorCallTimeout {
		t.Fatalf("wait limit %s is shorter than the longest tool timeout %s", pendingToolWaitLimit, maxConnectorCallTimeout)
	}
	th := retryTestThinker(nil)
	th.registry = NewToolRegistry("")
	for name, delay := range map[string]time.Duration{"quick": 10 * time.Millisecond, "slow": 350 * time.Millisecond} {
		delay := delay
		th.registry.Register(&ToolDef{Name: name, HandlerContext: func(context.Context, map[string]string) ToolResponse {
			time.Sleep(delay)
			return ToolResponse{Text: "done"}
		}})
	}
	queueTool(th, toolCall{Name: "quick", NativeID: "call-quick", Args: map[string]string{}})
	queueTool(th, toolCall{Name: "slow", NativeID: "call-slow", Args: map[string]string{}})

	var results []ToolResult
	var texts []string
	var parts []ContentPart
	th.waitForPendingTools(&results, &texts, &parts, pendingToolWaitLimit)
	if th.pendingToolCount() != 0 || len(results) != 2 {
		t.Fatalf("turn went on with calls still running: pending=%d results=%+v", th.pendingToolCount(), results)
	}
	before := len(results)
	th.injectPlaceholdersForPending(&results)
	if len(results) != before {
		t.Fatalf("a finished call got a placeholder: %+v", results)
	}
	for _, text := range texts {
		if strings.Contains(text, "late-result") {
			t.Fatalf("a result arrived as late text: %q", text)
		}
	}
}

// The operator can still cut the wait short. Only then is a call that is
// still running answered with a placeholder.
func TestOperatorMessageEndsTheWaitForToolCalls(t *testing.T) {
	th := retryTestThinker(nil)
	th.registry = NewToolRegistry("")
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	th.registry.Register(&ToolDef{Name: "slow", HandlerContext: func(context.Context, map[string]string) ToolResponse {
		started <- struct{}{}
		<-release
		return ToolResponse{Text: "done"}
	}})
	defer close(release)
	queueTool(th, toolCall{Name: "slow", NativeID: "call-slow", Args: map[string]string{}})
	<-started

	var results []ToolResult
	var texts []string
	var parts []ContentPart
	done := make(chan struct{})
	go func() { th.waitForPendingTools(&results, &texts, &parts, pendingToolWaitLimit); close(done) }()
	select {
	case <-done:
		t.Fatal("the wait ended while the call was still running")
	case <-time.After(50 * time.Millisecond):
	}
	th.bus.Publish(Event{Type: EventInbox, To: "main", From: "operator", Text: "[console] stop, do this first"})
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("an operator message did not end the wait")
	}
	th.injectPlaceholdersForPending(&results)
	if len(results) != 1 || results[0].CallID != "call-slow" || !strings.Contains(results[0].Content, "In progress") {
		t.Fatalf("running call was not answered with a placeholder: %+v", results)
	}
}
