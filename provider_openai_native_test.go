package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenAINativeRuntimeTokenRefreshIsSingleFlight(t *testing.T) {
	var requests atomic.Int32
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
		_ = json.NewEncoder(w).Encode(map[string]string{"access_token": "new-token"})
	}))
	defer server.Close()
	p := &OpenAINativeProvider{apiKey: "old-token", runtimeTokenURL: server.URL, serverAPIKey: "server-key"}

	const callers = 20
	var wg sync.WaitGroup
	errs := make(chan error, callers)
	for i := 0; i < callers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- p.refreshRuntimeToken(context.Background(), true)
		}()
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("refresh request did not start")
	}
	time.Sleep(25 * time.Millisecond)
	close(release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("refresh HTTP requests = %d, want 1", got)
	}
	if got := p.token(); got != "new-token" {
		t.Fatalf("token = %q", got)
	}
}

func TestOpenAINativeBuildInput_SystemPromptHandling(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system rules"},
		{Role: "user", Content: "hello"},
	}

	regular := (&OpenAINativeProvider{}).buildInput(messages)
	if len(regular) != 2 {
		t.Fatalf("regular input len = %d, want 2", len(regular))
	}
	if regular[0].Role != "developer" || regular[0].Content != "system rules" {
		t.Fatalf("regular system item = %#v", regular[0])
	}

	codex := (&OpenAINativeProvider{forceStoreFalse: true}).buildInput(messages)
	if len(codex) != 1 {
		t.Fatalf("codex input len = %d, want 1", len(codex))
	}
	if codex[0].Role != "user" || codex[0].Content != "hello" {
		b, _ := json.Marshal(codex)
		t.Fatalf("codex input = %s", b)
	}
}

func TestOpenAINativeChat_BuildsFunctionToolsOnly(t *testing.T) {
	p := &OpenAINativeProvider{apiKey: "test", name: "openai-codex"}
	tools := p.buildAPITools("gpt-5.5", []NativeTool{
		{Name: "app_tool", Description: "tool", Parameters: map[string]any{"type": "object"}},
	})
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	b, _ := json.Marshal(tools[0])
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("tool JSON: %v", err)
	}
	if decoded["type"] != "function" || decoded["name"] != "app_tool" {
		t.Fatalf("tool = %#v", decoded)
	}
}

func TestOpenAINativeChat_SendsPromptCacheHints(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"type":"response.output_text.delta","delta":"ok"}`,
			`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":1,"input_tokens_details":{"cached_tokens":5,"cache_write_tokens":2}}}}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer srv.Close()

	p := &OpenAINativeProvider{
		name:            "openai-codex",
		apiKey:          "test",
		responsesURL:    srv.URL,
		forceStoreFalse: true,
	}
	scope := openAIPromptCacheScope{Identity: "agent:10/thread:main", Epoch: 7}
	tools := []NativeTool{{Name: "app_tool", Description: "tool", Parameters: map[string]any{"type": "object"}}}
	resp, err := p.Chat(withOpenAIPromptCacheScope(context.Background(), scope),
		[]Message{{Role: "system", Content: "stable system"}, {Role: "user", Content: "hello"}},
		"gpt-5.5",
		tools,
		nil, nil, nil,
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Text != "ok" || resp.Usage.CachedTokens != 5 || resp.Usage.CacheWriteTokens != 2 {
		t.Fatalf("response = %+v", resp)
	}
	if _, ok := body["prompt_cache_retention"]; ok {
		t.Fatalf("Codex request sent unsupported prompt_cache_retention: %#v", body)
	}
	key, _ := body["prompt_cache_key"].(string)
	if !strings.HasPrefix(key, "apteva-v2-") {
		t.Fatalf("prompt_cache_key = %q", key)
	}
	wantKey := openAIPromptCacheHintsForScope("openai-codex", "gpt-5.5", "stable system", p.buildAPITools("gpt-5.5", tools), scope).Key
	if key != wantKey {
		t.Fatalf("prompt_cache_key = %q, want scoped key %q", key, wantKey)
	}
	if body["instructions"] != "stable system" {
		t.Fatalf("instructions = %v", body["instructions"])
	}
	include, _ := body["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v, want encrypted reasoning", body["include"])
	}
}

func TestOpenAICodexChatUsesAccountAndCatalogCapabilities(t *testing.T) {
	var body map[string]any
	var accountID string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		accountID = r.Header.Get("ChatGPT-Account-ID")
		defer r.Body.Close()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	parallel := true
	p := &OpenAINativeProvider{
		name:            "openai-codex",
		apiKey:          "token",
		accountID:       "account-a",
		responsesURL:    srv.URL,
		forceStoreFalse: true,
		modelCapabilities: map[string]ModelCapabilities{
			"gpt-5.6-terra": {
				SupportsParallelToolCalls: &parallel,
				SupportedReasoningLevels:  []ModelReasoningCapability{{Effort: "low"}, {Effort: "high"}},
			},
		},
		reasoning: ReasoningSettings{Level: ReasoningXHigh},
	}
	if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "gpt-5.6-terra", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if accountID != "account-a" {
		t.Fatalf("ChatGPT-Account-ID = %q", accountID)
	}
	if body["parallel_tool_calls"] != true {
		t.Fatalf("parallel_tool_calls = %#v", body["parallel_tool_calls"])
	}
	if key, _ := body["prompt_cache_key"].(string); key == "" {
		t.Fatalf("GPT-5.6 Codex request omitted stable prompt_cache_key: %#v", body)
	}
	if _, ok := body["prompt_cache_retention"]; ok {
		t.Fatalf("GPT-5.6 Codex request included deprecated retention: %#v", body)
	}
	reasoning, _ := body["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v, want catalog-clamped high", reasoning)
	}
}

func TestOpenAICodexReasoningOmitsUnsupportedSummary(t *testing.T) {
	summaries := false
	p := &OpenAINativeProvider{
		name: "openai-codex",
		modelCapabilities: map[string]ModelCapabilities{
			"gpt-5.6-luna": {SupportsReasoningSummaries: &summaries},
		},
	}
	if got := p.requestReasoning("gpt-5.6-luna"); got != nil {
		t.Fatalf("auto reasoning = %#v, want nil when summaries are unsupported", got)
	}
	p.reasoning = ReasoningSettings{Level: ReasoningHigh}
	got := p.requestReasoning("gpt-5.6-luna")
	if got == nil || got.Effort != "high" || got.Summary != "" {
		t.Fatalf("high reasoning = %#v, want effort without summary", got)
	}
}

func TestOpenAINativeChat_RetriesWithoutUnsupportedPromptCacheHints(t *testing.T) {
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		bodies = append(bodies, body)
		if _, hasHint := body["prompt_cache_key"]; hasHint {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"Unknown parameter: prompt_cache_key"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"type":"response.output_text.delta","delta":"ok"}`,
			`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer srv.Close()

	p := &OpenAINativeProvider{name: "openai-codex", apiKey: "test", responsesURL: srv.URL, forceStoreFalse: true}
	if _, err := p.Chat(context.Background(), []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "hi"}}, "gpt-5.5", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	clone := p.WithReasoning(ReasoningSettings{Level: ReasoningLow})
	if _, err := clone.Chat(context.Background(), []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "again"}}, "gpt-5.5", nil, nil, nil, nil); err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	if len(bodies) != 3 {
		t.Fatalf("requests = %d, want first failure + retry + one remembered no-hint request", len(bodies))
	}
	if bodies[0]["prompt_cache_key"] == "" {
		t.Fatalf("first request missing cache key: %#v", bodies[0])
	}
	if _, ok := bodies[0]["prompt_cache_retention"]; ok {
		t.Fatalf("first Codex request included retention: %#v", bodies[0])
	}
	if _, ok := bodies[1]["prompt_cache_key"]; ok {
		t.Fatalf("retry kept prompt_cache_key: %#v", bodies[1])
	}
	if _, ok := bodies[2]["prompt_cache_key"]; ok {
		t.Fatalf("later call forgot unsupported-hint downgrade: %#v", bodies[2])
	}
}

func TestOpenAINativeRequestReasoning(t *testing.T) {
	codex := NewOpenAICodexProvider("token").(*OpenAINativeProvider)
	if got := codex.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "" {
		t.Fatalf("codex auto reasoning = %#v, want summary auto only", got)
	}

	high := codex.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).(*OpenAINativeProvider)
	if got := high.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "high" {
		t.Fatalf("codex high reasoning = %#v, want summary auto effort high", got)
	}
	minimal := codex.WithReasoning(ReasoningSettings{Level: ReasoningMinimal}).(*OpenAINativeProvider)
	if got := minimal.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "low" {
		t.Fatalf("codex minimal reasoning = %#v, want summary auto effort low", got)
	}
	xhigh := codex.WithReasoning(ReasoningSettings{Level: ReasoningXHigh}).(*OpenAINativeProvider)
	if got := xhigh.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "xhigh" {
		t.Fatalf("codex xhigh reasoning = %#v, want summary auto effort xhigh", got)
	}

	openai := NewOpenAINativeProvider("key").(*OpenAINativeProvider)
	if got := openai.requestReasoning("gpt-5.4-mini"); got != nil {
		t.Fatalf("openai auto reasoning = %#v, want nil", got)
	}
	low := openai.WithReasoning(ReasoningSettings{Level: ReasoningLow}).(*OpenAINativeProvider)
	if got := low.requestReasoning("gpt-5.4-mini"); got == nil || got.Summary != "auto" || got.Effort != "low" {
		t.Fatalf("openai low reasoning = %#v, want summary auto effort low", got)
	}
	none := openai.WithReasoning(ReasoningSettings{Level: ReasoningNone}).(*OpenAINativeProvider)
	if got := none.requestReasoning("gpt-5.4-mini"); got == nil || got.Summary != "" || got.Effort != "none" {
		t.Fatalf("openai none reasoning = %#v, want effort none without summary", got)
	}
}

func TestOpenAINativeBuildInput_FunctionImageOutputs(t *testing.T) {
	messages := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{
			ID:   "call_image",
			Name: "app_tool",
			Args: map[string]string{"action": "capture"},
		}}},
		{Role: "user", ToolResults: []ToolResult{{
			CallID:  "call_image",
			Content: "screenshot attached",
			Image:   []byte{0x89, 0x50, 0x4e, 0x47},
		}}},
	}

	items := (&OpenAINativeProvider{}).buildInput(messages)
	if len(items) != 2 {
		t.Fatalf("items len = %d, want 2: %#v", len(items), items)
	}
	if items[0].Type != "function_call" || items[0].Name != "app_tool" {
		t.Fatalf("replay item = %#v", items[0])
	}
	if items[1].Type != "function_call_output" {
		t.Fatalf("image result type = %q, want function_call_output", items[1].Type)
	}
	blocks, ok := items[1].Output.([]oaiContentBlock)
	if !ok || len(blocks) != 2 || blocks[0].Type != "input_text" || blocks[1].Type != "input_image" {
		t.Fatalf("image output = %#v", items[1].Output)
	}
}

func TestOpenAINativeBuildInput_ReplaysProviderItemsBeforeToolResults(t *testing.T) {
	reasoning := json.RawMessage(`{"id":"rs_123","type":"reasoning","summary":[{"type":"summary_text","text":"**Checking state**"}],"encrypted_content":"opaque-state"}`)
	message := json.RawMessage(`{"id":"msg_123","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"I’ll check."}]}`)
	functionCall := json.RawMessage(`{"id":"fc_123","type":"function_call","status":"completed","call_id":"call_123","name":"lookup","arguments":"{\"query\":\"x\"}"}`)
	messages := []Message{
		{
			Role:      "assistant",
			Content:   "I’ll check.",
			ToolCalls: []NativeToolCall{{ID: "call_123", OutputItemID: "fc_123", Name: "lookup", Args: map[string]string{"query": "x"}}},
			ProviderState: &ProviderResponseState{
				Provider: openAIResponsesStateProvider,
				Items:    []json.RawMessage{reasoning, message, functionCall},
			},
		},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call_123", Content: `{"ok":true}`}}},
	}

	items := (&OpenAINativeProvider{}).buildInput(messages)
	if len(items) != 4 {
		t.Fatalf("items len = %d, want 4", len(items))
	}
	for i, want := range []json.RawMessage{reasoning, message, functionCall} {
		got, err := json.Marshal(items[i])
		if err != nil {
			t.Fatalf("marshal replay item %d: %v", i, err)
		}
		if !jsonEqual(got, want) {
			t.Fatalf("replay item %d = %s, want %s", i, got, want)
		}
	}
	if items[3].Type != "function_call_output" || items[3].CallID != "call_123" || items[3].Output != `{"ok":true}` {
		t.Fatalf("tool result = %#v", items[3])
	}
}

func TestOpenAINativeBuildInput_ReplaysParallelCallsInOriginalOrder(t *testing.T) {
	rawItems := []json.RawMessage{
		json.RawMessage(`{"id":"rs_parallel","type":"reasoning","encrypted_content":"opaque-parallel"}`),
		json.RawMessage(`{"id":"fc_a","type":"function_call","status":"completed","call_id":"call_a","name":"first","arguments":"{}"}`),
		json.RawMessage(`{"id":"fc_b","type":"function_call","status":"completed","call_id":"call_b","name":"second","arguments":"{}"}`),
	}
	messages := []Message{
		{
			Role:      "assistant",
			ToolCalls: []NativeToolCall{{ID: "call_a", Name: "first"}, {ID: "call_b", Name: "second"}},
			ProviderState: &ProviderResponseState{
				Provider: openAIResponsesStateProvider,
				Items:    rawItems,
			},
		},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call_a", Content: "a"}, {CallID: "call_b", Content: "b"}}},
	}

	items := (&OpenAINativeProvider{}).buildInput(messages)
	if len(items) != 5 {
		t.Fatalf("items len = %d, want 5", len(items))
	}
	for i, wantID := range []string{"rs_parallel", "fc_a", "fc_b"} {
		var got map[string]any
		raw, _ := json.Marshal(items[i])
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("decode item %d: %v", i, err)
		}
		if got["id"] != wantID {
			t.Fatalf("item %d id = %v, want %s", i, got["id"], wantID)
		}
	}
	if items[3].CallID != "call_a" || items[4].CallID != "call_b" {
		t.Fatalf("result order = %q, %q", items[3].CallID, items[4].CallID)
	}
}

func TestOpenAINativeBuildInput_InvalidProviderStateFallsBackToLegacyReplay(t *testing.T) {
	messages := []Message{{
		Role:      "assistant",
		Content:   "working",
		ToolCalls: []NativeToolCall{{ID: "call_legacy", Name: "lookup", Args: map[string]string{"query": "x"}}},
		ProviderState: &ProviderResponseState{
			Provider: openAIResponsesStateProvider,
			Items:    []json.RawMessage{json.RawMessage(`{"type":"reasoning"}`), json.RawMessage(`not-json`)},
		},
	}}

	items := (&OpenAINativeProvider{}).buildInput(messages)
	if len(items) != 2 {
		t.Fatalf("items len = %d, want legacy message + call", len(items))
	}
	if items[0].Type != "message" || items[0].Content != "working" {
		t.Fatalf("legacy message = %#v", items[0])
	}
	if items[1].Type != "function_call" || items[1].CallID != "call_legacy" {
		t.Fatalf("legacy call = %#v", items[1])
	}
}

func jsonEqual(a, b []byte) bool {
	var av, bv any
	return json.Unmarshal(a, &av) == nil && json.Unmarshal(b, &bv) == nil && reflect.DeepEqual(av, bv)
}

func TestOpenAINativeStreamResponse_ReasoningSummary(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.reasoning_summary_text.delta","delta":"checking "}`,
		`data: {"type":"response.reasoning_summary_text.delta","delta":"state"}`,
		`data: {"type":"response.output_text.delta","delta":"done"}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var thinking strings.Builder
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if resp.Text != "done" {
		t.Fatalf("Text = %q, want done", resp.Text)
	}
	if resp.Reasoning != "checking state" {
		t.Fatalf("Reasoning = %q, want checking state", resp.Reasoning)
	}
	if thinking.String() != resp.Reasoning {
		t.Fatalf("onThinking = %q, want %q", thinking.String(), resp.Reasoning)
	}
	if resp.Usage.PromptTokens != 3 || resp.Usage.CompletionTokens != 2 || resp.Usage.CachedTokens != 1 {
		t.Fatalf("Usage = %+v", resp.Usage)
	}
}

func TestOpenAINativeStreamResponse_CapturesReasoningAndOutputItems(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.done","item":{"id":"rs_123","type":"reasoning","summary":[{"type":"summary_text","text":"**Checking state**"}],"encrypted_content":"opaque-state"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"msg_123","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"I’ll check."}]}}`,
		`data: {"type":"response.output_item.added","item":{"id":"fc_123","type":"function_call","call_id":"call_123","name":"lookup"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_123","type":"function_call","status":"completed","call_id":"call_123","name":"lookup","arguments":"{\"query\":\"x\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if resp.Reasoning != "**Checking state**" {
		t.Fatalf("Reasoning = %q", resp.Reasoning)
	}
	if resp.ProviderState == nil || resp.ProviderState.Provider != openAIResponsesStateProvider {
		t.Fatalf("ProviderState = %#v", resp.ProviderState)
	}
	if len(resp.ProviderState.Items) != 3 {
		t.Fatalf("provider items = %d, want 3", len(resp.ProviderState.Items))
	}
	var captured []map[string]any
	for i, raw := range resp.ProviderState.Items {
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatalf("provider item %d: %v", i, err)
		}
		captured = append(captured, item)
	}
	if captured[0]["id"] != "rs_123" || captured[0]["encrypted_content"] != "opaque-state" {
		t.Fatalf("reasoning item = %#v", captured[0])
	}
	if captured[1]["id"] != "msg_123" || captured[2]["id"] != "fc_123" {
		t.Fatalf("item order = %#v", captured)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_123" || resp.ToolCalls[0].OutputItemID != "fc_123" || resp.ToolCalls[0].Status != "completed" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponse_ToolChunkUsesCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"id\":\"main\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send","arguments":"{\"id\":\"main\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunkTool, chunkID, chunk string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunkTool = toolName
		chunkID = callID
		chunk = argChunk
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if chunkTool != "send" {
		t.Fatalf("chunk tool = %q, want send", chunkTool)
	}
	if chunkID != "call_final_456" {
		t.Fatalf("chunk id = %q, want final call_id", chunkID)
	}
	if chunk != `{"id":"main"}` {
		t.Fatalf("chunk = %q", chunk)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != chunkID {
		t.Fatalf("final call id = %q, want chunk id %q", resp.ToolCalls[0].ID, chunkID)
	}
}

func TestOpenAINativeStreamResponse_ReasoningSummaryPartDone(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"summary only"}}`,
		`data: {"type":"response.output_text.delta","delta":"done"}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var thinking strings.Builder
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if resp.Reasoning != "summary only" {
		t.Fatalf("Reasoning = %q, want summary only", resp.Reasoning)
	}
	if thinking.String() != resp.Reasoning {
		t.Fatalf("onThinking = %q, want %q", thinking.String(), resp.Reasoning)
	}
}

func TestOpenAINativeStreamResponse_BuffersToolChunkUntilCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call","name":"send"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"id\":\"main\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send","arguments":"{\"id\":\"main\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks len = %d, want 1: %#v", len(chunks), chunks)
	}
	if chunks[0] != `send|call_final_456|{"id":"main"}` {
		t.Fatalf("chunk = %q", chunks[0])
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_final_456" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponse_BuffersToolChunkUntilNameAndCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"text\":\"hello\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"channels_respond","arguments":"{\"text\":\"hello\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks len = %d, want 1: %#v", len(chunks), chunks)
	}
	if chunks[0] != `channels_respond|call_final_456|{"text":"hello"}` {
		t.Fatalf("chunk = %q", chunks[0])
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "channels_respond" || resp.ToolCalls[0].ID != "call_final_456" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponse_FinalArgumentsEmitToolChunkWhenNoDeltas(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"channels_respond","arguments":"{\"text\":\"hello\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 || chunks[0] != `channels_respond|call_final_456|{"text":"hello"}` {
		t.Fatalf("chunks = %#v", chunks)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Args["text"] != "hello" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponseReturnsProviderFailure(t *testing.T) {
	stream := strings.NewReader("data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"upstream failed\"}}}\n\n")
	_, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "upstream failed") {
		t.Fatalf("error = %v", err)
	}
}

func TestOpenAINativeStreamResponseReturnsScannerError(t *testing.T) {
	stream := strings.NewReader("data: " + strings.Repeat("x", 1024*1024+1))
	_, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "stream read error") {
		t.Fatalf("error = %v", err)
	}
}
