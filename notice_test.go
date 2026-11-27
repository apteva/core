package core

import (
	"encoding/json"
	"strings"
	"testing"
)

func strPtr(s string) *string { return &s }

// A user message records what it asked for when it is built. The lookup for
// the current instruction reads that, and never parses the message text.
func TestInstructionLookupUsesWhatTheMessageRecorded(t *testing.T) {
	asked := Message{Role: "user", Content: "[t] Events:\n• Assign the ticket\n• [late-result] Tool x completed: ok\n", Instruction: strPtr("Assign the ticket")}
	onlyNotices := Message{Role: "user", Content: "[t] Events:\n• [thread:w1 done] Closed ticket 4 and created ticket 9.\n", Instruction: strPtr("")}
	legacy := Message{Role: "user", Content: "Close the old ticket"}

	if query, source := recallQueryForTurn(nil, []Message{asked, onlyNotices}, "directive"); query != "Assign the ticket" || source != "latest_user_context" {
		t.Fatalf("query=%q source=%q, want the earlier real instruction", query, source)
	}
	if query, source := recallQueryForTurn(nil, []Message{onlyNotices}, "directive"); query != "directive" || source != "directive" {
		t.Fatalf("query=%q source=%q, want the directive", query, source)
	}
	if query, source := recallQueryForTurn([]string{"Reopen ticket 2"}, []Message{asked}, "directive"); query != "Reopen ticket 2" || source != "event" {
		t.Fatalf("query=%q source=%q, want this turn's instruction", query, source)
	}
	// A message stored before the field existed is still read by its content.
	if query, _ := recallQueryForTurn(nil, []Message{legacy}, ""); query != "Close the old ticket" {
		t.Fatalf("legacy message query = %q", query)
	}
	// The record survives the session journal, and an empty one is kept
	// distinct from a missing one.
	for _, message := range []Message{asked, onlyNotices, legacy} {
		raw, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		var back Message
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatal(err)
		}
		if (message.Instruction == nil) != (back.Instruction == nil) || message.Instruction != nil && *message.Instruction != *back.Instruction {
			t.Fatalf("instruction did not round-trip: %s", raw)
		}
	}
}

// What counts as an instruction is decided from the event, not its text.
func TestEventInstructionIsDecidedByTheEvent(t *testing.T) {
	result := &ToolResult{CallID: "c1", Content: "ok"}
	for _, tc := range []struct {
		event drainedEvent
		want  string
	}{
		{drainedEvent{Text: "  Close ticket 4  "}, "Close ticket 4"},
		{drainedEvent{Text: "[from:worker-1] Closed ticket 4"}, "[from:worker-1] Closed ticket 4"},
		{drainedEvent{Text: "[late-result] Tool x completed: ok", Notice: true}, ""},
		{drainedEvent{Text: "[tool:x] ok", ToolResult: result}, ""},
		// The text alone proves nothing: an unmarked event that happens to
		// look like a notice is still someone's message.
		{drainedEvent{Text: "[late-result] is what the log line said"}, "[late-result] is what the log line said"},
	} {
		if got := tc.event.instruction(); got != tc.want {
			t.Errorf("instruction(%+v) = %q, want %q", tc.event, got, tc.want)
		}
	}
	if isUrgentPendingToolInterruption(Event{From: "operator", Text: "[thread:w1 done] finished", Notice: true}) {
		t.Fatal("one of Core's own notices interrupted the wait for tool calls")
	}
	if isUrgentPendingToolInterruption(Event{From: "operator", Text: "[tool:x] ok", ToolResult: result}) {
		t.Fatal("a tool result interrupted the wait for tool calls")
	}
	if !isUrgentPendingToolInterruption(Event{From: "operator", Text: "stop"}) {
		t.Fatal("an operator message no longer interrupts the wait")
	}
}

// Core marks its own notices where it creates them.
func TestCoreMarksItsOwnNotices(t *testing.T) {
	// A result that arrives after its call was answered with a placeholder.
	th := retryTestThinker(nil)
	th.placeholdersSent.Store("late-1", placeholderInfo{toolName: "slow"})
	th.publishToolFailure(toolCall{Name: "slow", NativeID: "late-1"}, th.toolGeneration.Load(), nil, "upstream failed")
	late := th.drainEvents()
	if len(late) != 1 || !late[0].Notice || late[0].ToolResult != nil || !strings.Contains(late[0].Text, "late-result") {
		t.Fatalf("late result = %+v", late)
	}
	// The same failure on time is a tool result, not a notice.
	th.publishToolFailure(toolCall{Name: "slow", NativeID: "on-time"}, th.toolGeneration.Load(), nil, "upstream failed")
	if onTime := th.drainEvents(); len(onTime) != 1 || onTime[0].Notice || onTime[0].ToolResult == nil {
		t.Fatalf("on-time result = %+v", onTime)
	}

	// Queued events keep the mark on their way to the thread.
	bus := NewEventBus()
	sub := bus.Subscribe("main", 10)
	publishThreadInboxEvents(bus, "main", []PersistentThreadEvent{
		{ID: "completion/w1/1", Text: "[thread:w1 done] Closed ticket 4.", Notice: true},
		{ID: "api/1", Text: "Close ticket 5"},
	})
	got := sub.DrainTargeted()
	if len(got) != 2 || !got[0].Notice || got[1].Notice {
		t.Fatalf("queued events lost or gained the mark: %+v", got)
	}
	raw, _ := json.Marshal(PersistentThreadEvent{ID: "x", Text: "t", Notice: true})
	var back PersistentThreadEvent
	if json.Unmarshal(raw, &back) != nil || !back.Notice {
		t.Fatalf("the mark is not persisted: %s", raw)
	}
}

// While a turn waits for its tool calls, instructions that arrive are
// collected apart from notices and results.
func TestWaitCollectsInstructionsApartFromNotices(t *testing.T) {
	th := retryTestThinker(nil)
	th.bus.Publish(Event{Type: EventInbox, To: "main", From: "processes", Text: "next workflow step is ready"})
	th.bus.Publish(Event{Type: EventInbox, To: "main", Text: "[thread:w1] started (provider: p, role: worker, tools: a_b)", Notice: true})
	th.pendingTools.Store("call-1", "slow")
	go func() {
		th.pendingTools.Delete("call-1")
		th.bus.Publish(Event{Type: EventInbox, To: "main", Text: "[tool:slow] ok", ToolResult: &ToolResult{CallID: "call-1", Content: "ok"}})
	}()
	var results []ToolResult
	var consumed, ids, executions, instructions []string
	var parts []ContentPart
	th.waitForPendingTools(&results, &consumed, &parts, pendingToolWaitLimit, &ids, &executions, &instructions)
	if len(instructions) != 1 || instructions[0] != "next workflow step is ready" {
		t.Fatalf("instructions = %q, consumed = %q", instructions, consumed)
	}
	if len(consumed) < 2 {
		t.Fatalf("the thread must still read the notice: consumed = %q", consumed)
	}
}
