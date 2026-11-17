package core

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func bookingResultPolicy() DirectToolResultPolicy {
	return DirectToolResultPolicy{Tool: "book", SuccessField: "status", SuccessValue: "confirmed", PublicFields: []string{"status", "date", "reference"}}
}

func TestRealtimeExecutionDefaultsAndIsolation(t *testing.T) {
	rt, _ := completionFixture(t)
	original := NewOpenAINativeProvider("test")
	applyModelOverrides(original, map[string]string{"large": "original-model"})
	rt.Thinker.provider = original
	rt.provider = NewOpenAIRealtimeProvider("test")
	rt.opts.Model = "gpt-live-1"
	rt.capabilities = capabilitiesForRealtime(rt.provider, rt.opts.Model)
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	if got := modelIDForProvider(rt.backend.provider, ModelMedium); got != "gpt-6-luna" || rt.backend.reasoning != ReasoningNone {
		t.Fatalf("defaults: %s %s", got, rt.backend.reasoning)
	}
	if original.Models()[ModelLarge] != "original-model" {
		t.Fatal("shared model mapping mutated")
	}
	rt.realtimeExecution = &RealtimeExecutionConfig{Model: "large", Reasoning: "low"}
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	if rt.backend.provider != original || rt.backend.reasoning != ReasoningLow {
		t.Fatal("explicit configuration lost")
	}
	rt.realtimeExecution = &RealtimeExecutionConfig{ModelID: "gpt-6.1-sol", Reasoning: "none"}
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	if modelIDForProvider(rt.backend.provider, ModelMedium) != "gpt-6.1-sol" {
		t.Fatal("exact model override lost")
	}
	if !rt.provider.(realtimeExecutionDefaultProvider).DefaultRealtimeExecution("gpt-realtime-2.1").isZero() {
		t.Fatal("changed direct-tool model")
	}
}

func TestRealtimeExecutionConfigClonesAndValidates(t *testing.T) {
	c := RealtimeExecutionConfig{PreloadTools: []string{"book"}, ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	cp := cloneRealtimeExecutionConfig(&c)
	cp.PreloadTools[0] = "other"
	cp.ResultTools[0].PublicFields[0] = "other"
	if c.PreloadTools[0] != "book" || c.ResultTools[0].PublicFields[0] != "status" {
		t.Fatal("shared config")
	}
	for _, c := range []RealtimeExecutionConfig{{Model: "large", ModelID: "x"}, {Reasoning: "bogus"}, {PreloadTools: make([]string, 21)}, {ResultTools: []DirectToolResultPolicy{{Tool: "book"}}}} {
		if validateRealtimeExecutionConfig(c) == nil {
			t.Fatalf("accepted invalid config %+v", c)
		}
	}
}

func TestDirectReceiptGuards(t *testing.T) {
	for _, name := range []string{"success", "error", "failed_status", "missing_field", "nested_field", "oversize", "media", "not_terminal", "missing_decision", "stale", "parallel", "external", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			th := newTestThinkerFull()
			defer th.Stop()
			th.work = &threadWorkState{}
			th.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
			w := ThreadWork{ID: "request", Revision: 1, Origin: "main", Kind: "request"}
			th.work.observe(&w)
			args := map[string]string{completeAssignmentArg: "true"}
			response := ToolResponse{Text: `{"status":"confirmed","date":"2026-11-03T14:00:00+01:00","reference":"R42","private":"NEVER DISCLOSE"}`}
			switch name {
			case "error":
				response.IsError = true
			case "failed_status":
				response.Text = strings.Replace(response.Text, "confirmed", "pending", 1)
			case "missing_field":
				response.Text = `{"status":"confirmed"}`
			case "nested_field":
				response.Text = `{"status":"confirmed","date":{},"reference":"R42"}`
			case "oversize":
				response.Text = `{"status":"confirmed","date":"` + strings.Repeat("x", 450) + `","reference":"R42"}`
			case "media":
				response.Image = []byte{1}
			case "missing_decision":
				delete(args, completeAssignmentArg)
			case "not_terminal":
				args[completeAssignmentArg] = "false"
			case "stale":
				newer := w
				newer.Revision++
				th.work.advance(newer)
			}
			call := toolCall{Name: "book", NativeID: "call", Args: args, work: &w}
			receipt := th.directToolReceipt(call, response)
			want := name == "success" || name == "parallel" || name == "external" || name == "duplicate"
			if (receipt != nil) != want {
				t.Fatalf("receipt=%+v want=%v", receipt, want)
			}
			if receipt == nil {
				return
			}
			if strings.Contains(receipt.Content, "private") || strings.Contains(receipt.Content, "NEVER") {
				t.Fatal("private result leaked")
			}
			calls := []NativeToolCall{{ID: "call", Name: "book", Args: args}}
			if name == "parallel" {
				calls = append(calls, NativeToolCall{ID: "other", Name: "email"})
			}
			th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: calls})
			result := ToolResult{CallID: "call", ToolName: "book", Content: response.Text, DirectReceipt: receipt}
			done, err := th.completeDirectToolResult([]ToolResult{result}, name == "external")
			if err != nil {
				t.Fatal(err)
			}
			if done != (name == "success" || name == "duplicate") {
				t.Fatalf("completed=%v", done)
			}
			if name == "duplicate" {
				again, err := th.completeDirectToolResult([]ToolResult{result}, false)
				if again || err != nil {
					t.Fatal("duplicate completion", err)
				}
			}
		})
	}
}

func TestExecutionPreloadsAndSchemasRespectScope(t *testing.T) {
	th := newTestThinkerFull()
	defer th.Stop()
	th.threadID = "worker"
	th.registry = NewToolRegistry("")
	th.work = &threadWorkState{}
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	for _, name := range []string{"book", "denied"} {
		th.registry.Register(&ToolDef{Name: name, MCP: true, MCPServer: "appointments", InputSchema: schema})
	}
	th.toolAllowlist = map[string]bool{"book": true, "search_tools": true}
	th.executionPolicy = &RealtimeExecutionConfig{PreloadTools: []string{"book", "denied"}, ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	overlay := th.preloadExecutionTools(nil)
	if !overlay["book"] || overlay["denied"] {
		t.Fatalf("overlay=%v", overlay)
	}
	tools := []NativeTool{{Name: "book", Parameters: schema}}
	th.decorateDirectResultTools(tools)
	if tools[0].Parameters["properties"].(map[string]any)[completeAssignmentArg] == nil {
		t.Fatal("no completion hint")
	}
	required := tools[0].Parameters["required"].([]any)
	if len(required) != 1 || required[0] != completeAssignmentArg {
		t.Fatal("completion decision is optional")
	}
	if schema["properties"].(map[string]any)[completeAssignmentArg] != nil {
		t.Fatal("mutated shared schema")
	}
	args := toolDispatchArgs(th, toolCall{Name: "book", Args: map[string]string{completeAssignmentArg: "true", "date": "Tuesday"}})
	if _, ok := args[completeAssignmentArg]; ok {
		t.Fatal("Core hint leaked to tool")
	}
}

type receiptTestProvider struct {
	omitDecision bool
	LLMProvider
	calls atomic.Int32
}

func (p *receiptTestProvider) Chat(_ context.Context, _ []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	if p.calls.Add(1) == 1 {
		if p.omitDecision {
			return ChatResponse{ToolCalls: []NativeToolCall{{ID: "call", Name: "book", Args: map[string]string{}}}}, nil
		}
		return ChatResponse{ToolCalls: []NativeToolCall{{ID: "call", Name: "book", Args: map[string]string{completeAssignmentArg: "true"}}}}, nil
	}
	return ChatResponse{Text: "unexpected narration"}, nil
}
func (p *receiptTestProvider) WithBuiltins([]string) LLMProvider { return p }

func TestManagedWorkerDirectReceiptSkipsInferenceAndPersists(t *testing.T) {
	parent := newTestThinkerFull()
	parent.session = NewSession(t.TempDir(), "main")
	parent.registry = NewToolRegistry("")
	p := &receiptTestProvider{LLMProvider: NewOpenAINativeProvider("test")}
	parent.provider = p
	var executions atomic.Int32
	parent.registry.Register(&ToolDef{Name: "book", Description: "Book appointment", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}, Handler: func(args map[string]string) ToolResponse {
		executions.Add(1)
		return ToolResponse{Text: `{"status":"confirmed","date":"Tuesday at 14:00","reference":"R42"}`}
	}})
	defer parent.Stop()
	if err := parent.threads.SpawnWithOpts("worker", "Book once", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["worker"].Thinker
	worker.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	if err := parent.threads.Start("worker"); err != nil {
		t.Fatal(err)
	}
	w := ThreadWork{ID: "request", Revision: 1, Origin: "main", Kind: "request"}
	if err := parent.threads.QueueWork("worker", w, "Book Tuesday at 14:00", nil); err != nil {
		t.Fatal(err)
	}
	deadline := time.After(3 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("no direct result")
		case <-parent.sub.Wake:
			for _, ev := range parent.sub.DrainTargeted() {
				if ev.Work != nil && ev.Work.Kind == "result" {
					if !strings.Contains(ev.Text, "R42") {
						t.Fatalf("unexpected result %s", ev.Text)
					}
					time.Sleep(30 * time.Millisecond)
					if p.calls.Load() != 1 || executions.Load() != 1 {
						t.Fatalf("inference=%d tools=%d", p.calls.Load(), executions.Load())
					}
					parent.threads.KillAll()
					entries := readSessionEntriesForTest(t, worker.session.path)
					paired, completed := false, false
					for _, m := range entries {
						paired = paired || len(m.ToolResults) > 0
						completed = completed || m.Work != nil && m.Work.Kind == "result"
					}
					if !paired || !completed {
						t.Fatal("missing durable evidence/completion")
					}
					return
				}
			}
		}
	}
}

func TestLiveProviderExecutionReasoningSerialized(t *testing.T) {
	data, err := buildOpenAILiveStart(RealtimeSessionOpts{Model: "gpt-live-1", Voice: "brise", ExecutionMode: "provider", ExecutionModel: "gpt-6-luna", ExecutionReasoning: "none"}, "brise")
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err = json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	responses := body["session"].(map[string]any)["delegation"].(map[string]any)["responses"].(map[string]any)
	if responses["reasoning"].(map[string]any)["effort"] != "none" {
		t.Fatalf("settings=%v", responses)
	}
}

func TestExecutionPreloadStableAndDiscoveryStillAvailable(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := newTestThinkerFull()
	defer th.Stop()
	th.threadID = "worker"
	th.registry = NewToolRegistry("")
	th.work = &threadWorkState{}
	defs := []mcpToolDef{mkTool("book", "Book appointment"), mkTool("reschedule", "Reschedule appointment")}
	registerTestMCPTools(th.registry, "appointments", defs)
	th.toolIndex.Add("appointments", defs, false, &MCPToolLoadingConfig{Default: ToolLoadDeferred})
	th.toolAllowlist = map[string]bool{"search_tools": true}
	th.toolMCPScopes = map[string]bool{"appointments": true}
	th.executionPolicy = &RealtimeExecutionConfig{PreloadTools: []string{"appointments_book"}}
	for i := 0; i < 3; i++ {
		th.iteration = i * 100
		th.activeTools = map[string]bool{}
		tools := nativeToolSet(th.prepareNativeTools("openai"))
		if !tools["appointments_book"] || tools["appointments_reschedule"] || !tools["search_tools"] {
			t.Fatalf("unexpected tools: %v", tools)
		}
	}
	runSearchTools(th, map[string]string{"query": "appointments_reschedule"}, false)
	tools := nativeToolSet(th.prepareNativeTools("openai"))
	if !tools["appointments_book"] || !tools["appointments_reschedule"] {
		t.Fatalf("discovery lost: %v", tools)
	}
}

func TestDirectReceiptReservedArgumentPreserved(t *testing.T) {
	th := newTestThinkerFull()
	defer th.Stop()
	th.registry = NewToolRegistry("")
	th.work = &threadWorkState{}
	th.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	th.registry.Register(&ToolDef{Name: "book", InputSchema: map[string]any{"type": "object", "properties": map[string]any{completeAssignmentArg: map[string]any{"type": "string"}}}})
	args := toolDispatchArgs(th, toolCall{Name: "book", Args: map[string]string{completeAssignmentArg: "original"}})
	if args[completeAssignmentArg] != "original" || th.directResultPolicy("book") != nil {
		t.Fatal("overrode app argument")
	}
}

func TestDirectReceiptPendingAndRevisedAfterResult(t *testing.T) {
	for _, mode := range []string{"pending", "revised"} {
		t.Run(mode, func(t *testing.T) {
			th := newTestThinkerFull()
			defer th.Stop()
			th.work = &threadWorkState{}
			th.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
			w := ThreadWork{ID: "request", Revision: 1, Origin: "main", Kind: "request"}
			th.work.observe(&w)
			args := map[string]string{completeAssignmentArg: "true"}
			receipt := th.directToolReceipt(toolCall{Name: "book", NativeID: "call", Args: args, work: &w}, ToolResponse{Text: `{"status":"confirmed","date":"Tuesday","reference":"R42"}`})
			if receipt == nil {
				t.Fatal("fixture missing receipt")
			}
			th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call", Name: "book", Args: args}}})
			if mode == "pending" {
				th.pendingTools.Store("other", "email")
			} else {
				w.Revision++
				th.work.advance(w)
			}
			done, err := th.completeDirectToolResult([]ToolResult{{CallID: "call", ToolName: "book", DirectReceipt: receipt}}, false)
			if done || err != nil {
				t.Fatal("completed pending/superseded work", err)
			}
		})
	}
}

func TestProviderModePreloadsReceiptToolsWithoutShortcut(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.backend.mode = "provider"
	rt.backend.policy = RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	worker := newTestThinkerFull()
	defer worker.Stop()
	worker.work = &threadWorkState{}
	rt.configureExecutionWorker(worker)
	if len(worker.executionPolicy.ResultTools) != 0 || len(worker.executionPolicy.PreloadTools) != 1 || worker.executionPolicy.PreloadTools[0] != "book" {
		t.Fatal("provider-mode preload/shortcut policy lost")
	}
	if len(rt.backend.policy.ResultTools) != 1 {
		t.Fatal("modified caller config")
	}
}

func TestDirectReceiptRequiredDecisionPreservesAppSchema(t *testing.T) {
	th := newTestThinkerFull()
	defer th.Stop()
	th.work = &threadWorkState{}
	th.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	schema := map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}}, "required": []string{"name"}}
	tools := []NativeTool{{Name: "book", Parameters: schema}}
	th.decorateDirectResultTools(tools)
	th.decorateDirectResultTools(tools)
	required := tools[0].Parameters["required"].([]any)
	if len(required) != 2 || required[0] != "name" || required[1] != completeAssignmentArg {
		t.Fatalf("requirements=%v", required)
	}
	if len(schema["required"].([]string)) != 1 {
		t.Fatal("mutated original schema")
	}
}

func TestDirectReceiptMissingDecisionContinuesWithoutRepeatingAction(t *testing.T) {
	parent := newTestThinkerFull()
	defer parent.Stop()
	parent.session = NewSession(t.TempDir(), "main")
	p := &receiptTestProvider{LLMProvider: NewOpenAINativeProvider("test"), omitDecision: true}
	parent.provider = p
	parent.registry = NewToolRegistry("")
	var calls atomic.Int32
	parent.registry.Register(&ToolDef{Name: "book", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}, Handler: func(args map[string]string) ToolResponse {
		calls.Add(1)
		return ToolResponse{Text: `{"status":"confirmed","date":"Tuesday","reference":"R42"}`}
	}})
	if err := parent.threads.SpawnWithOpts("worker", "Book once", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["worker"].Thinker
	worker.executionPolicy = &RealtimeExecutionConfig{ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
	if err := parent.threads.Start("worker"); err != nil {
		t.Fatal(err)
	}
	if err := parent.threads.QueueWork("worker", ThreadWork{ID: "request", Revision: 1, Origin: "main", Kind: "request"}, "Book Tuesday", nil); err != nil {
		t.Fatal(err)
	}
	deadline := time.After(3 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("missing completion decision stalled worker")
		case <-parent.sub.Wake:
			for _, e := range parent.sub.DrainTargeted() {
				if e.Work != nil && e.Work.Kind == "result" {
					parent.threads.KillAll()
					if calls.Load() != 1 || p.calls.Load() != 2 {
						t.Fatalf("calls=%d requests=%d", calls.Load(), p.calls.Load())
					}
					events, _ := parent.telemetry.StoredEvents(0)
					found := false
					for _, event := range events {
						if event.Type == "thread.work_direct_result" {
							t.Fatal("missing flag became completion")
						}
						if event.Type == "thread.work_direct_result_evaluated" && strings.Contains(string(event.Data), "completion_decision_missing") {
							found = true
						}
					}
					if !found {
						t.Fatal("missing decision not diagnosed")
					}
					return
				}
			}
		}
	}
}
