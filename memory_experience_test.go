package core

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestExperienceUpgradesOnlyThePreviousBuiltinMemoryPrompt(t *testing.T) {
	old, err := os.ReadFile("testdata/memory/unconscious_v2_prompt.txt")
	if err != nil {
		t.Fatal(err)
	}
	if !legacyBuiltinMemoryDirective(string(old)) {
		t.Fatal("previous platform memory thread would retain obsolete learning instructions")
	}
	if legacyBuiltinMemoryDirective(string(old)+"\nCustom owner policy.") || legacyBuiltinMemoryDirective(unconsciousDirectiveV2) {
		t.Fatal("upgrade would overwrite a customized or current memory prompt")
	}
}

func TestExperienceUsesExistingMemoryThreadAndKeepsDirectiveAuthority(t *testing.T) {
	registry := NewToolRegistry("test")
	for role, prompt := range map[string]string{
		"main":   buildSystemPrompt("Keep the assigned objective.", registry, "", nil, nil, nil, nil),
		"worker": formatThreadBasePrompt(false, false, "worker", "main"),
		"leader": formatThreadBasePrompt(true, false, "leader", "main"),
		"voice":  formatThreadBasePrompt(false, true, "voice", "main"),
		"evolve": registry.Get("evolve").Rules,
	} {
		if !strings.Contains(prompt, selfImprovementMemoryContract) {
			t.Errorf("%s lacks the common memory contract", role)
		}
		if strings.Contains(prompt, "call evolve once to retain that improvement") {
			t.Errorf("%s still learns by rewriting its directive", role)
		}
	}
	for _, want := range []string{"EXPERIENCE LESSONS", "memory_supersede", "retries, duplicate delivery", "Pending/delayed outcomes stay unknown", "existing source-thread scope"} {
		if !strings.Contains(unconsciousDirectiveV2, want) {
			t.Errorf("memory thread lacks %q", want)
		}
	}
}

func TestExperienceHistoryPreservesOutcomeIdentityWithoutArgumentsOrMedia(t *testing.T) {
	entry := SessionEntry{Sequence: 12, Role: "tool_result", Content: "observed reply", Reasoning: "private reasoning", ToolCalls: []NativeToolCall{{ID: "call-12", Name: "call_customer", Args: map[string]string{"secret": "private argument"}}}, ToolResults: []ToolResult{{CallID: "call-12", ToolName: "call_customer", Content: "Connected; recipient declined.", IsError: false}, {CallID: "call-13", ToolName: "lookup", Content: "Temporary failure", IsError: true, ContentIsPreview: true, Image: []byte("private pixels")}}}
	projected, ok := historyEntry("chat-a", entry)
	if !ok || len(projected.ToolResults) != 2 {
		t.Fatal("missing projected results")
	}
	if projected.ToolResults[0].CallID != "call-12" || projected.ToolResults[0].IsError || !projected.ToolResults[1].IsError || !projected.ToolResults[1].ContentIsPreview {
		t.Fatal("lost execution identity or uncertainty")
	}
	raw, _ := json.Marshal(projected)
	for _, secret := range []string{"private argument", "private reasoning", "private pixels"} {
		if strings.Contains(string(raw), secret) {
			t.Fatal("unnecessary private state entered consolidation")
		}
	}
	// Added projection metadata must not cause all old history to be relearned.
	changed := entry
	changed.ToolResults = append([]ToolResult(nil), entry.ToolResults...)
	changed.ToolResults[0].CallID = "other-call"
	again, _ := historyEntry("chat-a", changed)
	if again.Source.EntryID != projected.Source.EntryID {
		t.Fatal("projection upgrade invalidated existing history checkpoints")
	}
}

func TestExperienceHistoryMetadataIsBounded(t *testing.T) {
	entry := SessionEntry{Role: "tool_result", Content: "batch"}
	for i := 0; i < 100; i++ {
		entry.ToolResults = append(entry.ToolResults, ToolResult{CallID: strings.Repeat("id", 1000), ToolName: "lookup", Content: "result"})
	}
	projected, _ := historyEntry("chat-a", entry)
	raw, _ := json.Marshal(projected.ToolResults)
	if !projected.ToolMetadataTruncated || len(raw) > 4096 {
		t.Fatal("unbounded evidence metadata")
	}
}

func TestExperienceFailureRefreshIsBoundedAndDoesNotRepeatForRetries(t *testing.T) {
	var state memoryRecallCycleState
	failure := []ToolResult{{CallID: "one", ToolName: "booking", Content: "Verification required. " + strings.Repeat("x", 3000), IsError: true}}
	if !state.observeToolFailures(failure, false) || len(state.failureQuery) > 600 {
		t.Fatal("new failure did not produce bounded recall context")
	}
	failure[0].CallID = "retry"
	if state.observeToolFailures(failure, false) {
		t.Fatal("same failure refreshed recall for each retry")
	}
	if state.observeToolFailures([]ToolResult{{ToolName: "booking", Content: "recipient declined", IsError: false}}, false) {
		t.Fatal("inferred technical failure from a business response")
	}
	state.observeToolFailures(nil, true)
	if state.failureQuery != "" {
		t.Fatal("old failure contaminated a new external request")
	}
	failures := []ToolResult{{ToolName: "a", Content: "verification required", IsError: true}, {ToolName: "b", Content: "missing setup", IsError: true}}
	state.observeToolFailures(failures, false)
	if state.observeToolFailures([]ToolResult{failures[1], failures[0]}, false) {
		t.Fatal("result ordering refreshed an identical failure set")
	}
}
