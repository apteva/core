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

func TestOpenAICompatChat_SendsPromptCacheHintsOnlyForOpenAI(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"choices":[{"delta":{"content":"ok"}}]}`,
			`data: {"usage":{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":5}}}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer srv.Close()

	p := &OpenAICompatProvider{name: "openai", apiKey: "test", authHeader: "Bearer", url: srv.URL}
	resp, err := p.Chat(context.Background(),
		[]Message{{Role: "system", Content: "stable system"}, {Role: "user", Content: "hello"}},
		"gpt-5.5",
		[]NativeTool{{Name: "app_tool", Description: "tool", Parameters: map[string]any{"type": "object"}}},
		nil, nil, nil,
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Text != "ok" || resp.Usage.CachedTokens != 5 {
		t.Fatalf("response = %+v", resp)
	}
	if body["prompt_cache_retention"] != "24h" {
		t.Fatalf("prompt_cache_retention = %v, want 24h", body["prompt_cache_retention"])
	}
	if key, _ := body["prompt_cache_key"].(string); !strings.HasPrefix(key, "apteva-v1-") {
		t.Fatalf("prompt_cache_key = %q", key)
	}

	body = nil
	fireworks := &OpenAICompatProvider{name: "fireworks", apiKey: "test", authHeader: "Bearer", url: srv.URL}
	if _, err := fireworks.Chat(context.Background(), []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "hello"}}, "kimi", nil, nil, nil, nil); err != nil {
		t.Fatalf("fireworks Chat: %v", err)
	}
	if _, ok := body["prompt_cache_key"]; ok {
		t.Fatalf("fireworks request unexpectedly had prompt_cache_key: %#v", body)
	}
}
