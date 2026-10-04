package core

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"
)

func TestToolActivityReasonSelection(t *testing.T) {
	registry := NewToolRegistry("")
	registry.Register(&ToolDef{Name: "billing_invoices_search", Description: "Filter invoices. Args: q, limit."})
	thinker := &Thinker{registry: registry}
	for _, test := range []struct {
		name, reason, want, source string
		definition                 *ToolDef
	}{
		{"model", " Searching customer invoices ", "Searching customer invoices", "model", nil},
		{"missing", "", "Filter invoices", "tool_description", nil},
		{"whitespace", " \n\t ", "Filter invoices", "tool_description", nil},
		{"captured definition", "", "Look up invoices", "tool_description", &ToolDef{Description: "Look up invoices; internal usage guidance."}},
	} {
		t.Run(test.name, func(t *testing.T) {
			call := toolCall{Name: "billing_invoices_search", NativeID: "call-1", definition: test.definition, Args: map[string]string{"_reason": test.reason, "q": "PRIVATE_ARGUMENT", "limit": "0"}}
			data := thinker.prepareToolCallData(&call, []string{"execution-1"})
			if data.Reason != test.want || data.ReasonSource != test.source || data.ID != "call-1" || !reflect.DeepEqual(data.ExecutionIDs, []string{"execution-1"}) {
				t.Fatalf("activity=%+v", data)
			}
			if _, ok := call.Args["_reason"]; ok || call.Args["q"] != "PRIVATE_ARGUMENT" || call.Args["limit"] != "0" {
				t.Fatalf("observability field leaked or operation arguments changed: %#v", call.Args)
			}
			if strings.Contains(data.Reason, "PRIVATE_ARGUMENT") || strings.Contains(data.Reason, "internal usage") || strings.Contains(data.Reason, "Args:") {
				t.Fatalf("fallback included arguments/guidance: %q", data.Reason)
			}
		})
	}
	for _, name := range []string{"billing_invoices_search", ""} {
		call := toolCall{Name: name}
		data := (&Thinker{}).prepareToolCallData(&call, nil)
		if data.Reason == "" || data.ReasonSource != "tool_name" {
			t.Fatalf("unknown tool activity=%+v", data)
		}
	}
	call := toolCall{Name: "long", definition: &ToolDef{Description: strings.Repeat("é", 300)}}
	data := thinker.prepareToolCallData(&call, nil)
	if utf8.RuneCountInString(data.Reason) != 160 || !utf8.ValidString(data.Reason) || !strings.HasSuffix(data.Reason, "…") {
		t.Fatalf("fallback not safely bounded: %q", data.Reason)
	}
}

// Reproduce the production case starting with the exact provider JSON shape:
// the model omits _reason, Core preserves its raw arguments, runs the operation
// exactly once, and emits a useful call label before a successful result.
func TestToolActivityReasonProviderOmissionStillExecutes(t *testing.T) {
	args := `{"q":"Circlewise","limit":200,"sort":"due_date"}`
	item := map[string]any{"type": "response.output_item.done", "item": map[string]any{"id": "fc-billing", "type": "function_call", "call_id": "call-billing", "name": "billing_invoices_search", "arguments": args}}
	encoded, _ := json.Marshal(item)
	stream := "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc-billing\",\"type\":\"function_call\",\"call_id\":\"call-billing\",\"name\":\"billing_invoices_search\"}}\n\n" +
		fmt.Sprintf("data: %s\n\ndata: {\"type\":\"response.completed\"}\n\ndata: [DONE]\n\n", encoded)
	response, err := (&OpenAINativeProvider{}).streamResponse(strings.NewReader(stream), nil, nil, nil)
	if err != nil || len(response.ToolCalls) != 1 {
		t.Fatalf("parse response=%+v error=%v", response, err)
	}
	native := response.ToolCalls[0]
	if native.RawArgs != args || string(native.CanonicalArgs) != args || native.Args["_reason"] != "" {
		t.Fatalf("parser fabricated or changed provider arguments: %+v", native)
	}
	registry := NewToolRegistry("")
	var executed atomic.Int64
	registry.Register(&ToolDef{
		Name: native.Name, Description: "Filter invoices. q searches invoice numbers; use customers_search for names.",
		Handler: func(got map[string]string) ToolResponse {
			executed.Add(1)
			if !reflect.DeepEqual(got, map[string]string{"q": "Circlewise", "limit": "200", "sort": "due_date"}) {
				t.Errorf("operation arguments changed: %#v", got)
			}
			return ToolResponse{Text: `{"count":0}`}
		},
	})
	bus := NewEventBus()
	telemetry := NewTelemetry()
	t.Cleanup(telemetry.Stop)
	thinker := &Thinker{registry: registry, bus: bus, sub: bus.Subscribe("main", 100), telemetry: telemetry, threadID: "main", quit: make(chan struct{})}
	// Raw telemetry remains decisive evidence of the model omission even when
	// the user-visible tool.call label uses a Core fallback.
	telemetry.Emit("tool.arguments", thinker.threadID, newToolArgumentsData(native.ID, native.Name, "provider_raw", native.RawArgs))
	executeTool(thinker, toolCall{Name: native.Name, NativeID: native.ID, Args: native.Args})
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		events, _ := telemetry.StoredEvents(0)
		var call ToolCallData
		var result ToolResultData
		for _, event := range events {
			switch event.Type {
			case "tool.call":
				if err := json.Unmarshal(event.Data, &call); err != nil {
					t.Fatal(err)
				}
			case "tool.result":
				if err := json.Unmarshal(event.Data, &result); err != nil {
					t.Fatal(err)
				}
			}
		}
		if result.ID == native.ID && thinker.asyncToolsActive.Load() == 0 {
			if call.Reason != "Filter invoices" || call.ReasonSource != "tool_description" || !result.Success || executed.Load() != 1 {
				t.Fatalf("call=%+v result=%+v executions=%d", call, result, executed.Load())
			}
			if string(native.CanonicalArgs) != args {
				t.Fatal("fallback mutated replay arguments")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("tool did not finish")
}

func TestToolActivityReasonInlinePaths(t *testing.T) {
	for _, path := range []string{"main", "worker", "worker unavailable"} {
		t.Run(path, func(t *testing.T) {
			thinker := newTestThinker()
			t.Cleanup(thinker.telemetry.Stop)
			thinker.registry = NewToolRegistry("")
			call := toolCall{Name: "list_threads", NativeID: "call-inline", Args: map[string]string{}}
			if path == "main" {
				mainToolHandler(thinker)(thinker, []toolCall{call}, nil)
			} else {
				thinker.threadID = "worker"
				thread := &Thread{ID: "worker", Thinker: thinker, Children: thinker.threads, Tools: map[string]bool{}}
				if path == "worker" {
					thread.Tools[call.Name] = true
				} else {
					call.Name = "unavailable_operation"
				}
				threadToolHandler(thread, thinker.threads)(thinker, []toolCall{call}, nil)
			}
			events, _ := thinker.telemetry.StoredEvents(0)
			calls := 0
			for _, event := range events {
				if event.Type != "tool.call" {
					continue
				}
				calls++
				var data ToolCallData
				if err := json.Unmarshal(event.Data, &data); err != nil {
					t.Fatal(err)
				}
				if data.Reason == "" || data.ReasonSource == "model" {
					t.Fatalf("inline activity=%+v", data)
				}
				if _, ok := data.Args["_reason"]; ok {
					t.Fatal("inline observability field leaked into arguments")
				}
			}
			if calls != 1 {
				t.Fatalf("tool.call events=%d, want 1", calls)
			}
		})
	}
}
