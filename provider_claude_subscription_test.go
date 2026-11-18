package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClaudeSubscriptionProviderNeedsServerAndConnection(t *testing.T) {
	t.Setenv("CLAUDE_SUBSCRIPTION_CONNECTION_ID", "")
	t.Setenv("SERVER_URL", "http://127.0.0.1:5280")
	t.Setenv("APTEVA_API_KEY", "core_test")
	if p := createProviderByName("claude-subscription"); p != nil {
		t.Fatal("provider created without a connection")
	}
	t.Setenv("CLAUDE_SUBSCRIPTION_CONNECTION_ID", "7")
	t.Setenv("SERVER_URL", "")
	if p := createProviderByName("claude-subscription"); p != nil {
		t.Fatal("provider created without a server URL")
	}
	t.Setenv("SERVER_URL", "http://127.0.0.1:5280/")
	p := createProviderByName("claude-subscription")
	if p == nil {
		t.Fatal("provider not created")
	}
	compat, ok := p.(*OpenAICompatProvider)
	if !ok {
		t.Fatalf("provider type %T", p)
	}
	if compat.Name() != "claude-subscription" || compat.url != "http://127.0.0.1:5280/api/llm/claude-subscription/chat/completions" {
		t.Fatalf("provider = %+v", compat)
	}
	if !compat.SupportsNativeTools() {
		t.Fatal("native tools disabled")
	}
	if in, cached, out := compat.CostPer1M(); in != 0 || cached != 0 || out != 0 {
		t.Fatal("subscription usage must not be priced per token")
	}
	if m := compat.Models(); m[ModelLarge] != "opus" || m[ModelMedium] != "sonnet" || m[ModelSmall] != "haiku" {
		t.Fatalf("models = %+v", m)
	}
}

// The server route speaks plain OpenAI chat-completion streaming; this
// checks core's generic client handles its tool calls and usage.
func TestClaudeSubscriptionProviderStreamsToolCalls(t *testing.T) {
	var gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range []string{
			`: waiting for a Claude process`,
			`data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking"},"finish_reason":null}]}`,
			`data: {"choices":[{"index":0,"delta":{"content":"Let me check."},"finish_reason":null}]}`,
			`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"toolu_1","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`,
			`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}`,
			`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]},"finish_reason":null}]}`,
			`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`data: {"choices":[],"usage":{"prompt_tokens":115,"completion_tokens":7,"total_tokens":122,"prompt_tokens_details":{"cached_tokens":100,"cache_write_tokens":5}}}`,
			`data: [DONE]`,
		} {
			_, _ = io.WriteString(w, chunk+"\n\n")
		}
	}))
	defer srv.Close()

	p := NewClaudeSubscriptionProvider(srv.URL, "core_test")
	tools := []NativeTool{{Name: "lookup", Description: "Look up", Parameters: map[string]any{"type": "object"}}}
	resp, err := p.Chat(context.Background(), []Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "hi"}}, "opus", tools, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer core_test" || gotBody["model"] != "opus" || gotBody["stream"] != true {
		t.Fatalf("request auth=%q body=%v", gotAuth, gotBody)
	}
	if defs, _ := gotBody["tools"].([]any); len(defs) != 1 {
		t.Fatalf("tools not sent: %v", gotBody["tools"])
	}
	if resp.Text != "Let me check." || resp.Reasoning != "thinking" {
		t.Fatalf("response text=%q reasoning=%q", resp.Text, resp.Reasoning)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "toolu_1" || resp.ToolCalls[0].Name != "lookup" || resp.ToolCalls[0].Args["q"] != "x" {
		t.Fatalf("tool calls = %+v", resp.ToolCalls)
	}
	if resp.Usage.PromptTokens != 115 || resp.Usage.CachedTokens != 100 || resp.Usage.CacheWriteTokens != 5 || resp.Usage.CompletionTokens != 7 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if strings.Contains(gotAuth, "sk-") {
		t.Fatal("unexpected credential")
	}
}
