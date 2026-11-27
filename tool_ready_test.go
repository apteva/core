package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// toolReadyLog records chunk and ready callbacks in the order they arrive.
type toolReadyLog struct{ events []string }

func (l *toolReadyLog) chunk(_, callID, _ string) { l.events = append(l.events, "chunk:"+callID) }
func (l *toolReadyLog) ready(_, callID string, _ int) {
	l.events = append(l.events, "ready:"+callID)
}
func (l *toolReadyLog) context() context.Context {
	return withToolReadyObserver(context.Background(), l.ready)
}
func (l *toolReadyLog) index(event string) int {
	for i, e := range l.events {
		if e == event {
			return i
		}
	}
	return -1
}
func (l *toolReadyLog) count(event string) int {
	n := 0
	for _, e := range l.events {
		if e == event {
			n++
		}
	}
	return n
}

// The first call must be reported before the model starts writing the second.
func (l *toolReadyLog) assertFirstReadyBeforeSecond(t *testing.T, first, second string) {
	t.Helper()
	if l.count("ready:"+first) != 1 || l.count("ready:"+second) != 1 {
		t.Fatalf("each call must be reported once: %v", l.events)
	}
	if ready, next := l.index("ready:"+first), l.index("chunk:"+second); ready < 0 || next < 0 || ready > next {
		t.Fatalf("first call was not ready before the second was written: %v", l.events)
	}
}

func sseServer(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func compatToolDelta(index int, id, name, args string) string {
	call := map[string]any{"index": index, "function": map[string]any{"arguments": args}}
	if id != "" {
		call["id"] = id
		call["function"].(map[string]any)["name"] = name
	}
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{call}}}}})
	return string(b)
}

func compatToolReadyChat(t *testing.T, log *toolReadyLog, events ...string) (ChatResponse, error) {
	t.Helper()
	provider := &OpenAICompatProvider{
		name: "test", apiKey: "test", authHeader: "Bearer", url: sseServer(t, events...).URL,
		models: map[ModelTier]string{ModelMedium: "test-model"},
	}
	return provider.Chat(log.context(), []Message{{Role: "user", Content: "hello"}}, "test-model",
		[]NativeTool{{Name: "create", Parameters: map[string]any{"type": "object"}}}, nil, nil, log.chunk)
}

func TestChatCompletionsToolReadyWhenNextCallStarts(t *testing.T) {
	log := &toolReadyLog{}
	resp, err := compatToolReadyChat(t, log,
		compatToolDelta(0, "call-a", "create", `{"title":`),
		compatToolDelta(0, "", "", `"one"}`),
		compatToolDelta(1, "call-b", "create", `{"title":`),
		compatToolDelta(1, "", "", `"two"}`),
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		"[DONE]",
	)
	if err != nil || len(resp.ToolCalls) != 2 {
		t.Fatalf("calls=%+v err=%v", resp.ToolCalls, err)
	}
	log.assertFirstReadyBeforeSecond(t, "call-a", "call-b")
}

// A gateway that interleaves two calls must not have either reported while
// its arguments are still partial.
func TestChatCompletionsToolReadyWaitsForCompleteArguments(t *testing.T) {
	log := &toolReadyLog{}
	resp, err := compatToolReadyChat(t, log,
		compatToolDelta(0, "call-a", "create", `{"title":`),
		compatToolDelta(1, "call-b", "create", `{"title":`),
		compatToolDelta(0, "", "", `"one"}`),
		compatToolDelta(1, "", "", `"two"}`),
		"[DONE]",
	)
	if err != nil || len(resp.ToolCalls) != 2 {
		t.Fatalf("calls=%+v err=%v", resp.ToolCalls, err)
	}
	if log.count("ready:call-a") != 1 || log.count("ready:call-b") != 1 {
		t.Fatalf("each call must be reported once: %v", log.events)
	}
	if ready, last := log.index("ready:call-a"), 2; ready <= last {
		t.Fatalf("call-a was reported before its arguments were complete: %v", log.events)
	}
}

func TestChatCompletionsToolReadySkipsFailedStream(t *testing.T) {
	log := &toolReadyLog{}
	_, err := compatToolReadyChat(t, log, compatToolDelta(0, "call-a", "create", `{"title":`))
	if err == nil {
		t.Fatal("truncated stream must fail")
	}
	if len(log.events) != 1 || log.events[0] != "chunk:call-a" {
		t.Fatalf("a truncated call must not be reported ready: %v", log.events)
	}
}

func TestAnthropicToolReadyAtBlockStop(t *testing.T) {
	log := &toolReadyLog{}
	p := NewAnthropicProvider("test").(*AnthropicProvider)
	p.url = sseServer(t,
		`{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"create"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"title\":\"one\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_b","name":"create"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"title\":\"two\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`,
		`{"type":"message_stop"}`,
	).URL
	resp, err := p.Chat(log.context(), []Message{{Role: "user", Content: "hello"}}, "claude-sonnet-5-5",
		[]NativeTool{{Name: "create", Parameters: map[string]any{"type": "object"}}}, nil, nil, log.chunk)
	if err != nil || len(resp.ToolCalls) != 2 {
		t.Fatalf("calls=%+v err=%v", resp.ToolCalls, err)
	}
	log.assertFirstReadyBeforeSecond(t, "toolu_a", "toolu_b")
}

func TestResponsesToolReadyAtOutputItemDone(t *testing.T) {
	log := &toolReadyLog{}
	item := func(id, call, args string) string {
		return `data: {"type":"response.output_item.added","item":{"type":"function_call","id":"` + id + `","call_id":"` + call + `","name":"create"}}` + "\n\n" +
			`data: {"type":"response.function_call_arguments.delta","item_id":"` + id + `","delta":` + string(mustJSON(t, args)) + `}` + "\n\n" +
			`data: {"type":"response.output_item.done","item":{"type":"function_call","id":"` + id + `","call_id":"` + call + `","name":"create","arguments":` + string(mustJSON(t, args)) + `}}` + "\n\n"
	}
	stream := item("fc_a", "call-a", `{"title":"one"}`) + item("fc_b", "call-b", `{"title":"two"}`) +
		`data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}` + "\n\n"
	resp, err := (&OpenAINativeProvider{}).streamResponseReady(strings.NewReader(stream), nil, nil, log.chunk, log.ready)
	if err != nil || len(resp.ToolCalls) != 2 {
		t.Fatalf("calls=%+v err=%v", resp.ToolCalls, err)
	}
	log.assertFirstReadyBeforeSecond(t, "call-a", "call-b")
}

func TestGeminiToolReadyWithEachCall(t *testing.T) {
	log := &toolReadyLog{}
	stream := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call-a","name":"create","args":{"title":"one"}}}]}}]}` + "\n" +
		`data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call-b","name":"create","args":{"title":"two"}}}]},"finishReason":"STOP"}]}` + "\n"
	resp, err := parseGeminiStreamReady(strings.NewReader(stream), nil, log.chunk, log.ready)
	if err != nil || len(resp.ToolCalls) != 2 {
		t.Fatalf("calls=%+v err=%v", resp.ToolCalls, err)
	}
	first, second := resp.ToolCalls[0].ID, resp.ToolCalls[1].ID
	log.assertFirstReadyBeforeSecond(t, first, second)
}

// toolReadyProvider reports one call ready, as a streaming provider would.
type toolReadyProvider struct{ scriptedRetryProvider }

func (p *toolReadyProvider) Chat(ctx context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	notifyToolReady(toolReadyObserverFromContext(ctx), "create", "call-a", `{"title":"one"}`)
	return ChatResponse{Text: "done"}, nil
}

func TestThinkerEmitsToolReadyEvent(t *testing.T) {
	provider := &toolReadyProvider{scriptedRetryProvider{name: "primary"}}
	thinker := retryTestThinker(provider)
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatalf("callLLMWithRetry: %v", err)
	}
	events, _ := thinker.telemetry.Events(0)
	var got []map[string]any
	for _, event := range events {
		if event.Type == "llm.tool_ready" {
			var data map[string]any
			if err := json.Unmarshal(event.Data, &data); err != nil {
				t.Fatalf("event data: %v", err)
			}
			got = append(got, data)
		}
	}
	if len(got) != 1 {
		t.Fatalf("llm.tool_ready events = %d, want 1", len(got))
	}
	if got[0]["id"] != "call-a" || got[0]["tool"] != "create" || got[0]["name"] != "create" || got[0]["arg_bytes"] != float64(15) {
		t.Fatalf("llm.tool_ready data = %v", got[0])
	}
}
