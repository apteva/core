package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestXAIProviderStreamsToolsContinuationReasoningUsageAndCacheHeader(t *testing.T) {
	type capturedRequest struct {
		Authorization  string
		ConversationID string
		Body           map[string]any
	}
	var requests []capturedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		requests = append(requests, capturedRequest{
			Authorization:  r.Header.Get("Authorization"),
			ConversationID: r.Header.Get("x-grok-conv-id"),
			Body:           body,
		})

		w.Header().Set("Content-Type", "text/event-stream")
		messages, _ := body["messages"].([]any)
		hasToolResult := false
		for _, rawMessage := range messages {
			message, _ := rawMessage.(map[string]any)
			if message["role"] == "tool" {
				hasToolResult = true
			}
		}
		if hasToolResult {
			_, _ = w.Write([]byte(strings.Join([]string{
				`data: {"choices":[{"delta":{"content":"sunny"}}]}`,
				`data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":21}}}`,
				`data: [DONE]`,
				``,
			}, "\n")))
			return
		}
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"choices":[{"delta":{"reasoning_content":"checking"}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Madrid\"}"}}]}}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":7}}}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer srv.Close()

	provider := NewXAIProvider("xai-test-key").(*XAIProvider)
	provider.compat.url = srv.URL
	provider.now = func() time.Time { return time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC) }
	provider = provider.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).(*XAIProvider)
	ctx := withOpenAIPromptCacheScope(context.Background(), openAIPromptCacheScope{
		Identity: "agent:286/thread:main",
		Epoch:    3,
	})

	var thinking strings.Builder
	var toolChunks strings.Builder
	tools := []NativeTool{{
		Name:        "weather",
		Description: "Get weather",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"city"},
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
		},
	}}
	first, err := provider.Chat(ctx, []Message{
		{Role: "system", Content: "Use the weather tool."},
		{Role: "user", Content: "Weather in Madrid?"},
	}, "grok-4.5", tools, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, func(_, _, chunk string) {
		toolChunks.WriteString(chunk)
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if thinking.String() != "checking" || first.Reasoning != "checking" {
		t.Fatalf("reasoning callback=%q response=%q", thinking.String(), first.Reasoning)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].ID != "call_weather" || first.ToolCalls[0].Name != "weather" || first.ToolCalls[0].Args["city"] != "Madrid" {
		t.Fatalf("tool calls = %+v", first.ToolCalls)
	}
	if !strings.Contains(toolChunks.String(), `"city":"Madrid"`) {
		t.Fatalf("tool chunks = %q", toolChunks.String())
	}
	if first.Usage.PromptTokens != 20 || first.Usage.CachedTokens != 7 || first.Usage.CompletionTokens != 4 {
		t.Fatalf("first usage = %+v", first.Usage)
	}

	second, err := provider.Chat(ctx, []Message{
		{Role: "system", Content: "Use the weather tool."},
		{Role: "user", Content: "Weather in Madrid?"},
		{Role: "assistant", Reasoning: first.Reasoning, ToolCalls: first.ToolCalls},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call_weather", ToolName: "weather", Content: `{"condition":"sunny"}`}}},
	}, "grok-4.5", tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	if second.Text != "sunny" || second.Usage.CachedTokens != 21 {
		t.Fatalf("second response = %+v", second)
	}

	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	for i, request := range requests {
		if request.Authorization != "Bearer xai-test-key" {
			t.Fatalf("request %d authorization = %q", i+1, request.Authorization)
		}
		if !strings.HasPrefix(request.ConversationID, "apteva-xai-v1-") {
			t.Fatalf("request %d conversation id = %q", i+1, request.ConversationID)
		}
		if request.Body["reasoning_effort"] != "high" {
			t.Fatalf("request %d reasoning_effort = %#v", i+1, request.Body["reasoning_effort"])
		}
		streamOptions, _ := request.Body["stream_options"].(map[string]any)
		if streamOptions["include_usage"] != true {
			t.Fatalf("request %d stream_options = %#v", i+1, streamOptions)
		}
		if _, present := request.Body["prompt_cache_key"]; present {
			t.Fatalf("request %d used Responses/OpenAI cache field: %#v", i+1, request.Body)
		}
	}
	if requests[0].ConversationID != requests[1].ConversationID {
		t.Fatalf("normal append changed conversation id: %q != %q", requests[0].ConversationID, requests[1].ConversationID)
	}
}

func TestXAIConversationIDSeparatesScopeEpochModelAndReuseWindow(t *testing.T) {
	baseTime := time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC)
	baseScope := openAIPromptCacheScope{Identity: "agent:1/thread:main", Epoch: 2}
	base := xAIConversationID(baseScope, "grok-4.5", baseTime)
	if !strings.HasPrefix(base, "apteva-xai-v1-") {
		t.Fatalf("base id = %q", base)
	}
	if got := xAIConversationID(baseScope, "grok-4.5", baseTime.Add(time.Hour)); got != base {
		t.Fatalf("same reuse window changed id: %q != %q", got, base)
	}
	for label, got := range map[string]string{
		"identity": xAIConversationID(openAIPromptCacheScope{Identity: "agent:2/thread:main", Epoch: 2}, "grok-4.5", baseTime),
		"epoch":    xAIConversationID(openAIPromptCacheScope{Identity: baseScope.Identity, Epoch: 3}, "grok-4.5", baseTime),
		"model":    xAIConversationID(baseScope, "grok-4.3", baseTime),
		"window":   xAIConversationID(baseScope, "grok-4.5", baseTime.Add(24*time.Hour)),
	} {
		if got == base {
			t.Fatalf("%s did not change conversation id", label)
		}
	}
	if got := xAIConversationID(openAIPromptCacheScope{}, "grok-4.5", baseTime); got != "" {
		t.Fatalf("empty scope id = %q, want no shared cache identity", got)
	}
}

func TestXAIProviderParsesParallelFunctionCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"first","arguments":"{\"value\":1}"}},{"index":1,"id":"call_b","type":"function","function":{"name":"second","arguments":"{\"value\":2}"}}]}}]}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer srv.Close()
	provider := NewXAIProvider("test").(*XAIProvider)
	provider.compat.url = srv.URL
	response, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "call both"}}, "grok-4.5", []NativeTool{
		{Name: "first", Parameters: map[string]any{"type": "object"}},
		{Name: "second", Parameters: map[string]any{"type": "object"}},
	}, nil, nil, nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(response.ToolCalls) != 2 || response.ToolCalls[0].ID != "call_a" || response.ToolCalls[0].Args["value"] != "1" || response.ToolCalls[1].ID != "call_b" || response.ToolCalls[1].Args["value"] != "2" {
		t.Fatalf("parallel calls = %+v", response.ToolCalls)
	}
}

func TestXAIReasoningUsesCatalogAndSafeFallbacks(t *testing.T) {
	resetRuntimeModelCapabilities()
	t.Cleanup(resetRuntimeModelCapabilities)
	registerModelCapabilities(map[string]ModelCapabilities{
		"catalog-grok": {
			SupportedReasoningLevels: []ModelReasoningCapability{{Effort: "low"}, {Effort: "medium"}, {Effort: "high"}},
		},
	})
	if got := xAIReasoningEffort("catalog-grok", ReasoningNone); got != "low" {
		t.Fatalf("catalog-clamped none = %q, want low", got)
	}
	if got := xAIReasoningEffort("catalog-grok", ReasoningXHigh); got != "high" {
		t.Fatalf("catalog-clamped xhigh = %q, want high", got)
	}
	if got := xAIReasoningEffort("grok-4.5", ReasoningNone); got != "low" {
		t.Fatalf("grok-4.5 none = %q, want low", got)
	}
	if got := xAIReasoningEffort("grok-4.5", ReasoningXHigh); got != "high" {
		t.Fatalf("grok-4.5 xhigh = %q, want high", got)
	}
	if got := xAIReasoningEffort("grok-4.20-multi-agent", ReasoningXHigh); got != "xhigh" {
		t.Fatalf("multi-agent xhigh = %q, want xhigh", got)
	}
	if got := xAIReasoningEffort("grok-4.5", ReasoningAuto); got != "" {
		t.Fatalf("auto = %q, want omitted", got)
	}
}

func TestXAIProviderFactoryAndModelOverrides(t *testing.T) {
	t.Setenv("XAI_API_KEY", "factory-key")
	provider := createProviderByName("xai")
	xai, ok := provider.(*XAIProvider)
	if !ok || xai.Name() != "xai" {
		t.Fatalf("provider = %T/%v, want *XAIProvider", provider, provider)
	}
	if xai.compat.url != "https://api.x.ai/v1/chat/completions" || xai.compat.apiKey != "factory-key" {
		t.Fatalf("xAI transport = url %q key %q", xai.compat.url, xai.compat.apiKey)
	}
	applyModelOverrides(xai, map[string]string{
		"large":  "large-grok",
		"medium": "medium-grok",
		"small":  "small-grok",
	})
	if xai.Models()[ModelLarge] != "large-grok" || xai.Models()[ModelMedium] != "medium-grok" || xai.Models()[ModelSmall] != "small-grok" {
		t.Fatalf("models = %#v", xai.Models())
	}
}

func TestOpenAICompatDoesNotReceiveXAIRequestOptions(t *testing.T) {
	var conversationID string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conversationID = r.Header.Get("x-grok-conv-id")
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	provider := &OpenAICompatProvider{name: "fireworks", apiKey: "test", authHeader: "Bearer", url: srv.URL}
	ctx := withOpenAIPromptCacheScope(context.Background(), openAIPromptCacheScope{Identity: "agent:1/thread:main"})
	if _, err := provider.Chat(ctx, []Message{{Role: "user", Content: "hello"}}, "kimi", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if conversationID != "" {
		t.Fatalf("non-xAI conversation id = %q", conversationID)
	}
	if _, present := body["reasoning_effort"]; present {
		t.Fatalf("non-xAI request received reasoning_effort: %#v", body)
	}
}
