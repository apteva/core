package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func longToolRunFixture() []Message {
	messages := []Message{
		{Role: "system", Content: "Preserve pending work; never duplicate completed external actions."},
		{Role: "user", Content: "[COMPACTED CONTEXT]\nPOST-881 already published. Release remains pending."},
		{Role: "user", Content: "Inspect the records, report pending work, and do not publish anything.", EventIDs: []string{"long-run-event"}},
	}
	for i := 0; i < 24; i++ {
		id := fmt.Sprintf("inspect-%d", i)
		messages = append(messages,
			Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "inspect", Args: map[string]string{"record": id}}}},
			Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: strings.Repeat("small observation ", 350)}}},
		)
	}
	return messages
}

func TestRecoverySummarizesToolRunAfterLatestInstruction(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "POST-881 exists. Release pending; do not publish."}}
	messages := longToolRunFixture()
	th := recoveryTestThinker(t, p, messages)
	next, err := th.recoverOversizedRequest(context.Background(), p, messages, fmt.Errorf("context_length_exceeded"), 0, ModelLarge)
	if err != nil {
		t.Fatal(err)
	}
	if estimatedContextTokens(next) >= estimatedContextTokens(messages)/2 {
		t.Fatal("completed tool run did not shrink")
	}
	if err := validateToolHistory(next); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range next {
		if m.Content == messages[2].Content {
			found = len(m.EventIDs) == 1 && m.EventIDs[0] == "long-run-event"
		}
	}
	if !found {
		t.Fatal("latest instruction or execution correlation lost")
	}
	// A later long run must be able to merge the previous summary again.
	continuation := cloneMessages(messages[3:])
	for i := range continuation {
		for j := range continuation[i].ToolCalls {
			continuation[i].ToolCalls[j].ID = "again-" + continuation[i].ToolCalls[j].ID
		}
		for j := range continuation[i].ToolResults {
			continuation[i].ToolResults[j].CallID = "again-" + continuation[i].ToolResults[j].CallID
		}
	}
	more := append(cloneMessages(next), continuation...)
	if _, err := th.recoverOversizedRequest(context.Background(), p, more, fmt.Errorf("context_length_exceeded"), 0, ModelLarge); err != nil {
		t.Fatalf("second recovery: %v", err)
	}
	loaded, _ := th.session.LoadTail(100)
	if !th.session.EventIDs()["long-run-event"] || !strings.Contains(fmt.Sprint(loaded), "POST-881") {
		t.Fatal("restart lost recovery state")
	}
}

func TestRecoveryBoundsWideJSONAndPreservesPendingResult(t *testing.T) {
	rows := make([]map[string]any, 1000)
	for i := range rows {
		rows[i] = map[string]any{"id": i, "text": strings.Repeat("observation ", 20)}
	}
	raw, _ := json.Marshal(map[string]any{"rows": rows, "pending": true})
	messages := []Message{
		{Role: "system", Content: "Keep pending work."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "complete", Name: "inspect"}, {ID: "pending", Name: "inspect"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "complete", Content: string(raw)}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "pending", Content: string(raw)}}},
	}
	next := projectBulkyContext(messages, "history/original.json", map[string]bool{"pending": true})
	for _, m := range next {
		for _, r := range m.ToolResults {
			if r.CallID == "pending" && r.Content != string(raw) {
				t.Error("pending result changed")
			}
			if r.CallID == "complete" && (len(r.Content) > 10000 || !strings.Contains(r.Content, "history/original.json")) {
				t.Error("wide JSON did not become a bounded, recoverable preview")
			}
		}
	}
	if messages[2].ToolResults[0].Content != string(raw) {
		t.Fatal("original observation mutated")
	}
}

func TestRecoveryAcceptsSmallReductionThatFitsCompleteBudget(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary"}
	messages := []Message{
		{Role: "system", Content: strings.Repeat("fixed instruction ", 20000)},
		{Role: "user", Content: "Read the current observation."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "inspect", Name: "inspect"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "inspect", Content: strings.Repeat("observation ", 800)}}},
	}
	th := recoveryTestThinker(t, p, messages)
	b := estimatePreparedRequest(p.Name(), "test", messages, nil)
	b.InputTokens += 4000 // Schemas and other wire overhead remain fixed.
	b.InputBudget = b.InputTokens - 100
	next, err := th.recoverOversizedRequest(context.Background(), p, messages, &contextBudgetError{Budget: b}, 0, ModelLarge)
	if err != nil {
		t.Fatal(err)
	}
	after := estimatePreparedRequest(p.Name(), "test", next, nil)
	if after.InputTokens+4000 > b.InputBudget || estimatedContextTokens(next) <= estimatedContextTokens(messages)*95/100 {
		t.Fatal("fixture must fit after a reduction smaller than five percent")
	}
	if p.calls != 0 || next[0].Content != messages[0].Content {
		t.Fatal("fitting projection unnecessarily summarized or altered instructions")
	}
}

func TestRecoveryKeepsParallelPendingGroupTogether(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "POST-881 exists. Release pending."}}
	messages := longToolRunFixture()
	group := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "still-pending", Name: "inspect"}, {ID: "already-complete", Name: "inspect"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "already-complete", Content: "exact completed observation"}}},
	}
	messages = append(append(cloneMessages(messages[:3]), group...), messages[3:]...)
	th := recoveryTestThinker(t, p, messages)
	th.pendingTools.Store("still-pending", true)
	next, err := th.recoverOversizedRequest(context.Background(), p, messages, fmt.Errorf("context_length_exceeded"), 0, ModelLarge)
	if err != nil {
		t.Fatal(err)
	}
	calls, results := map[string]bool{}, map[string]string{}
	for _, m := range next {
		for _, c := range m.ToolCalls {
			calls[c.ID] = true
		}
		for _, r := range m.ToolResults {
			results[r.CallID] = r.Content
		}
	}
	if !calls["still-pending"] || !calls["already-complete"] || results["already-complete"] != "exact completed observation" {
		t.Fatal("summary split the parallel group containing a pending call")
	}
}
