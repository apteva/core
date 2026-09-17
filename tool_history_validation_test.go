package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"testing/synctest"
)

const malformedToolFixture = "uncensored_tool_call>tasks_get"

func historyCall(id, name string) Message {
	return Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: name, Args: map[string]string{"id": "record-123"}, CanonicalArgs: json.RawMessage(`{"id":"record-123","count":3,"enabled":true}`)}}}
}
func historyResult(id, outcome string) Message {
	return Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: outcome}}}
}
func badHistoryFixture() []Message {
	return []Message{historyCall("call_0", "lookup"), historyResult("call_0", "already completed: first"), historyCall("call_0", malformedToolFixture), historyResult("call_0", "unknown tool: recorded failure"), historyCall("call_0", "lookup"), historyResult("call_0", "already completed: last")}
}

func TestToolNameValidation(t *testing.T) {
	for _, name := range []string{"tasks_get", "flexylead-bookings_create_booking", "A9_Z-0"} {
		if !validToolName(name) {
			t.Errorf("rejected valid name %q", name)
		}
	}
	for _, name := range []string{"", malformedToolFixture, "tasks get", " tasks_get", "tasks_get\n", "tools/get", "tasks.get", "tâsks_get", "tasks\x00get"} {
		if validToolName(name) {
			t.Errorf("accepted invalid name %q", name)
		}
	}
}

func TestMalformedToolOutputRejectedByAdapters(t *testing.T) {
	fixtures := map[string]string{
		"compat":    `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"BAD_NAME","arguments":"{}"}}]}}]}` + "\n\ndata: [DONE]\n\n",
		"anthropic": `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_0","name":"BAD_NAME","input":{}}}` + "\n\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}` + "\n\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n" + `data: {"type":"message_stop"}` + "\n\n",
		"responses": `data: {"type":"response.output_item.added","item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"BAD_NAME"}}` + "\n\n" + `data: {"type":"response.output_item.done","item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"BAD_NAME","arguments":"{}"}}` + "\n\n" + `data: {"type":"response.completed"}` + "\n\n",
		"google":    `data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"call_0","name":"BAD_NAME","args":{}}}]},"finishReason":"STOP"}]}` + "\n\n",
	}
	for adapter, fixture := range fixtures {
		t.Run(adapter, func(t *testing.T) {
			for _, name := range []string{malformedToolFixture, "tasks get", "tasks/get"} {
				stream := strings.ReplaceAll(fixture, "BAD_NAME", name)
				var response ChatResponse
				var err error
				switch adapter {
				case "responses":
					response, err = (&OpenAINativeProvider{}).streamResponse(strings.NewReader(stream), nil, nil, nil)
				case "google":
					response, err = parseGeminiStream(strings.NewReader(stream), nil, nil)
				default:
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, stream)
					}))
					var provider LLMProvider = &OpenAICompatProvider{name: "audit", apiKey: "test", authHeader: "Bearer", url: server.URL}
					if adapter == "anthropic" {
						provider = &AnthropicProvider{apiKey: "test", url: server.URL}
					}
					response, err = provider.Chat(context.Background(), []Message{{Role: "user", Content: "test"}}, "test", nil, nil, nil, nil)
					server.Close()
				}
				var invalid *toolNameError
				if !errors.As(err, &invalid) || invalid.Source != "provider" || len(response.ToolCalls) != 0 || response.ProviderState != nil {
					t.Fatalf("name=%q response=%+v error=%v", name, response, err)
				}
			}
		})
	}
}

func TestMalformedToolOutputNeverDispatchedOrPersisted(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		t.Chdir(t.TempDir())
		provider := &scriptedRetryProvider{name: "fixture", response: ChatResponse{ToolCalls: historyCall("call_0", malformedToolFixture).ToolCalls}}
		thinker := NewThinker("", provider, &Config{path: "config.json", Directive: "Test malformed tool output."})
		thinker.pool = nil // No environment-discovered provider may be called by this test.
		defer thinker.blobs.Close()
		dispatches := 0
		thinker.handleTools = func(_ *Thinker, _ []toolCall, _ []string) ([]string, []string, []ToolResult) {
			dispatches++
			return nil, nil, nil
		}
		done := make(chan struct{})
		go func() { defer close(done); thinker.Run() }()
		synctest.Wait()
		thinker.Stop()
		<-done
		if dispatches != 0 || provider.calls != 1 {
			t.Fatalf("dispatches=%d provider calls=%d", dispatches, provider.calls)
		}
		for _, message := range thinker.messages {
			if len(message.ToolCalls) > 0 {
				t.Fatal("malformed call entered in-memory history")
			}
		}
		if raw, err := os.ReadFile(thinker.session.path); err == nil && bytes.Contains(raw, []byte(`"tool_calls"`)) {
			t.Fatalf("persisted tool call: %s", raw)
		}
		if health := thinker.inferenceSnapshot(); health.ConsecutiveFailures != 1 || !strings.Contains(health.BlockedReason, "provider_invalid_tool_name") {
			t.Fatalf("health=%+v", health)
		}
	})
}

func TestToolHistoryRepairPreservesNeighborsArchiveAndIsIdempotent(t *testing.T) {
	original := badHistoryFixture()
	before, _ := json.Marshal(original)
	session := NewSession(t.TempDir(), "main")
	// Legacy data enters through the raw archive API, bypassing the new append guard.
	for _, m := range original {
		if err := session.Append(SessionEntry{Role: m.Role, Content: m.Content, ToolCalls: m.ToolCalls, ToolResults: m.ToolResults}); err != nil {
			t.Fatal(err)
		}
	}
	archivedBefore, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	projected, changed := projectMalformedToolHistory(original)
	if !changed || validateToolHistory(projected) != nil {
		t.Fatal("history was not repaired")
	}
	for _, i := range []int{0, 1, 4, 5} {
		if !reflect.DeepEqual(projected[i], original[i]) {
			t.Fatalf("valid neighbor %d changed", i)
		}
	}
	if len(projected[2].ToolCalls) != 0 || len(projected[3].ToolResults) != 0 || !strings.Contains(projected[3].Content, "recorded failure") {
		t.Fatal("bad pair was not converted to diagnostic text")
	}
	twice, changedAgain := projectMalformedToolHistory(projected)
	if changedAgain || !reflect.DeepEqual(twice, projected) {
		t.Fatal("repair not idempotent")
	}
	after, _ := json.Marshal(original)
	if !bytes.Equal(before, after) {
		t.Fatal("repair mutated input")
	}
	loaded, _ := session.LoadTail(50)
	if validateToolHistory(loaded) != nil || !strings.Contains(messagesText(loaded), "recorded failure") {
		t.Fatal("session load failed to project corruption")
	}
	archivedAfter, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(archivedBefore, archivedAfter) {
		t.Fatal("loading rewrote original archive")
	}
	if err := session.AppendMessage(historyCall("new", malformedToolFixture), 1, TokenUsage{}); !isInvalidToolHistory(err) {
		t.Fatalf("append guard=%v", err)
	}
}

func TestToolPairRepairReusedIDsAndPendingCalls(t *testing.T) {
	t.Run("old result does not satisfy new call", func(t *testing.T) {
		original := []Message{historyCall("call_0", "first"), historyResult("call_0", "done"), historyCall("call_0", "pending")}
		cleaned := sanitizeToolPairs(original)
		if len(cleaned[2].ToolCalls) != 0 {
			t.Fatal("old result satisfied later call")
		}
		pending := sanitizeToolPairs(original, map[string]bool{"call_0": true})
		if len(pending[2].ToolCalls) != 1 || len(pending[1].ToolResults) != 1 {
			t.Fatal("pending call or old result lost")
		}
	})
	t.Run("old unanswered call does not claim new result", func(t *testing.T) {
		original := []Message{historyCall("call_0", malformedToolFixture), historyCall("call_0", "valid"), historyResult("call_0", "valid outcome")}
		cleaned := sanitizeToolPairs(original)
		if len(cleaned[1].ToolCalls) != 1 || len(cleaned[2].ToolResults) != 1 || cleaned[2].ToolResults[0].Content != "valid outcome" {
			t.Fatal("repair consumed valid reused-ID neighbor")
		}
	})
	t.Run("future call does not claim orphan result", func(t *testing.T) {
		original := []Message{historyResult("call_0", "old"), historyCall("call_0", "pending")}
		cleaned := sanitizeToolPairs(original, map[string]bool{"call_0": true})
		if len(cleaned) != 1 || len(cleaned[0].ToolCalls) != 1 {
			t.Fatal("orphan result paired with future pending call")
		}
	})
}

func TestToolHistoryRepairOpaqueStateAndTypedArguments(t *testing.T) {
	for _, stateOnly := range []bool{false, true} {
		t.Run(fmt.Sprint(stateOnly), func(t *testing.T) {
			m := historyCall("good", "lookup")
			m.ProviderState = &ProviderResponseState{Provider: openAIResponsesStateProvider, Items: []json.RawMessage{
				json.RawMessage(`{"type":"function_call","call_id":"bad","name":"uncensored_tool_call>tasks_get","arguments":"{}"}`),
				json.RawMessage(`{"type":"function_call","call_id":"good","name":"lookup","arguments":"{\"id\":\"record-123\",\"count\":3,\"enabled\":true}"}`),
			}}
			if stateOnly {
				m.ToolCalls = nil
			}
			original := []Message{m, {Role: "user", ToolResults: []ToolResult{{CallID: "bad", Content: "failed"}, {CallID: "good", Content: "valid recorded outcome"}}}}
			repaired := sanitizeToolPairs(original)
			if repaired[0].ProviderState != nil || len(repaired[0].ToolCalls) != 1 || repaired[0].ToolCalls[0].ID != "good" {
				t.Fatalf("calls=%+v", repaired[0])
			}
			if len(repaired[1].ToolResults) != 1 || repaired[1].ToolResults[0].Content != "valid recorded outcome" {
				t.Fatal("valid result lost")
			}
			items := (&OpenAINativeProvider{}).buildInput(repaired)
			raw, err := json.Marshal(items)
			if err != nil {
				t.Fatal(err)
			}
			var wire any
			json.Unmarshal(raw, &wire)
			if err := validateWireToolNames(wire, "history"); err != nil {
				t.Fatal(err)
			}
			found := false
			for _, item := range items {
				if item.Type == "function_call" {
					var args map[string]any
					json.Unmarshal([]byte(item.Arguments), &args)
					if args["count"] != float64(3) || args["enabled"] != true {
						t.Fatalf("typed args lost: %s", item.Arguments)
					}
					found = true
				}
			}
			if !found {
				t.Fatal("valid neighbor not replayed")
			}
			again, changed := projectMalformedToolHistory(repaired)
			if changed || !reflect.DeepEqual(again, repaired) {
				t.Fatal("opaque repair not idempotent")
			}
		})
	}
}

func TestFinalProviderRequestRejectsInvalidNamesWithoutInspectingData(t *testing.T) {
	for _, body := range []string{
		`{"input":[{"type":"function_call","name":"bad>name","arguments":"{}"}]}`,
		`{"messages":[{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"bad name","arguments":"{}"}}]}]}`,
		`{"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"bad/name","input":{}}]}]}`,
		`{"contents":[{"parts":[{"functionCall":{"name":"bad>name","args":{}}}]}]}`,
		`{"tools":[{"functionDeclarations":[{"name":"bad>name"}]}]}`,
	} {
		if err := observeProviderRequest(context.Background(), "test", "test", []byte(body)); !isInvalidToolHistory(err) {
			t.Fatalf("guard=%v body=%s", err, body)
		}
	}
	body := `{"messages":[{"role":"assistant","content":[{"type":"tool_use","name":"valid","input":{"type":"function_call","name":"arbitrary data >"}}]}]}`
	if err := observeProviderRequest(context.Background(), "test", "test", []byte(body)); err != nil {
		t.Fatal(err)
	}
}

func TestHistoryInvalidStopsWithoutRetryOrFallback(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 100, failureErr: &toolNameError{Source: "history", Name: malformedToolFixture}}
	fallback := &scriptedRetryProvider{name: "fallback", response: ChatResponse{Text: "must not run"}}
	th := retryTestThinker(primary)
	th.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "fallback": fallback}, order: []string{"primary", "fallback"}, default_: "primary"}
	_, err := th.callLLMWithRetry(context.Background())
	if !isInvalidToolHistory(err) || primary.calls != 1 || fallback.calls != 0 {
		t.Fatalf("error=%v calls=%d/%d", err, primary.calls, fallback.calls)
	}
	if h := th.inferenceSnapshot(); h.ConsecutiveFailures != 1 || !strings.HasPrefix(h.BlockedReason, "history_invalid_tool_name") {
		t.Fatalf("health=%+v", h)
	}
}

func TestToolHistoryPrimaryFallbackPrimaryRegression(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 1, failureErr: errors.New("provider API error 503: unavailable"), response: ChatResponse{Text: "primary resumed"}}
	fallback := &scriptedRetryProvider{name: "fallback", response: ChatResponse{ToolCalls: historyCall("call_0", malformedToolFixture).ToolCalls}}
	th := retryTestThinker(primary)
	th.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "fallback": fallback}, order: []string{"primary", "fallback"}, default_: "primary"}
	th.messages = append(th.messages, badHistoryFixture()...)
	response, err := th.callLLMWithRetry(context.Background())
	if err != nil || response.Text != "primary resumed" || primary.calls != 2 || fallback.calls != 1 {
		t.Fatalf("response=%+v err=%v calls=%d/%d", response, err, primary.calls, fallback.calls)
	}
	for _, request := range append(primary.messages, fallback.messages...) {
		if err := validateToolHistory(request); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(messagesText(request), "recorded failure") {
			t.Fatal("lost historical failure")
		}
	}
	if h := th.inferenceSnapshot(); h.ConsecutiveFailures != 0 || h.LastSuccessfulInference == nil || h.BlockedReason != "" {
		t.Fatalf("health=%+v", h)
	}
}

func TestInferenceHealthTracksFailureAndRecovery(t *testing.T) {
	p := &scriptedRetryProvider{name: "fixture", response: ChatResponse{Text: "ok"}}
	th := retryTestThinker(p)
	if _, err := th.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	successful := th.inferenceSnapshot().LastSuccessfulInference
	p.failures = 100
	p.failureErr = errors.New("provider API error 400: rejected")
	for i := 1; i <= 2; i++ {
		th.callLLMWithRetry(context.Background())
		h := th.inferenceSnapshot()
		if h.ConsecutiveFailures != i || h.LastSuccessfulInference != successful || h.BlockedReason == "" || th.workflowHealthy() {
			t.Fatalf("health=%+v", h)
		}
	}
	w := httptest.NewRecorder()
	(&APIServer{thinker: th}).health(w, httptest.NewRequest("GET", "/health", nil))
	var health map[string]bool
	json.Unmarshal(w.Body.Bytes(), &health)
	if !health["ok"] || health["workflow_healthy"] {
		t.Fatalf("liveness/workflow response=%s", w.Body.String())
	}
	p.failures = 0
	th.callLLMWithRetry(context.Background())
	if h := th.inferenceSnapshot(); h.ConsecutiveFailures != 0 || h.BlockedReason != "" || !th.workflowHealthy() {
		t.Fatalf("health=%+v", h)
	}
}

func TestToolHistoryWireGuardAndOneRepairBeforeSubmission(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var root map[string]any
		if err := json.NewDecoder(r.Body).Decode(&root); err != nil {
			t.Error(err)
		}
		if err := validateWireToolNames(root, "history"); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"recovered\"}\n\ndata: {\"type\":\"response.completed\"}\n\n")
	}))
	defer server.Close()
	provider := &OpenAINativeProvider{name: "openai", apiKey: "test", responsesURL: server.URL, models: map[ModelTier]string{ModelLarge: "test"}}
	messages := []Message{historyCall("call_0", "lookup"), historyResult("call_0", "already done")}
	messages[0].ProviderState = &ProviderResponseState{Provider: openAIResponsesStateProvider, Items: []json.RawMessage{json.RawMessage(`{"type":"function_call","call_id":"call_0","name":"uncensored_tool_call>tasks_get","arguments":"{}"}`)}}
	// Direct adapter use has no core request observer, but still validates wire bytes.
	_, err := provider.Chat(context.Background(), messages, "test", nil, nil, nil, nil)
	if !isInvalidToolHistory(err) || requests != 0 {
		t.Fatalf("invalid request was sent: requests=%d error=%v", requests, err)
	}
	th := retryTestThinker(provider)
	th.messages = append(th.messages, messages...)
	response, err := th.callLLMWithRetry(context.Background())
	if err != nil || response.Text != "recovered" || requests != 1 {
		t.Fatalf("response=%+v err=%v requests=%d", response, err, requests)
	}
}

func TestToolHistoryMixedBatchPreservesValidRecords(t *testing.T) {
	valid := historyCall("same", "valid").ToolCalls[0]
	invalid := historyCall("bad", malformedToolFixture).ToolCalls[0]
	original := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{invalid, valid}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "bad", Content: "bad call failed"}, {CallID: "same", Content: "valid call completed"}}},
		historyCall("same", "pending"),
	}
	clean := sanitizeToolPairs(original, map[string]bool{"same": true})
	if len(clean) != 4 || len(clean[0].ToolCalls) != 1 || !reflect.DeepEqual(clean[0].ToolCalls[0], valid) || len(clean[1].ToolResults) != 1 || clean[1].ToolResults[0].Content != "valid call completed" || !strings.Contains(clean[2].Content, "bad call failed") || len(clean[3].ToolCalls) != 1 {
		t.Fatalf("bad projection: %+v", clean)
	}
	if again := sanitizeToolPairs(clean, map[string]bool{"same": true}); !reflect.DeepEqual(clean, again) {
		t.Fatal("sanitization not idempotent")
	}
}
