package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// DeepSeek requires reasoning from ordinary assistant replies as well as tool
// calls when tools are present. Check the serialized HTTP request, including
// multimodal replies and messages that must not carry reasoning.
func TestOpenCodeGoReasoning_ReplaysAllAssistantTurns(t *testing.T) {
	requests := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- body
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()

	messages := []Message{
		{Role: "system", Content: "Review the code.", Reasoning: "must not be sent"},
		{Role: "user", Content: "Inspect it.", Reasoning: "must not be sent"},
		{Role: "assistant", Reasoning: "Inspect before reviewing.", ToolCalls: []NativeToolCall{{ID: "call_inspect", Name: "inspect_source", Args: map[string]string{}}}},
		{Role: "tool", ToolResults: []ToolResult{{CallID: "call_inspect", Content: "return a - b"}}},
		{Role: "assistant", Content: "Use addition.", Reasoning: "The function subtracts instead of adding."},
		{Role: "user", Content: "Show the corrected source."},
		{Role: "assistant", Parts: []ContentPart{{Type: "text", Text: "return a + b"}}, Reasoning: "Preserve the argument names."},
		{Role: "assistant", Content: "Ready."},
		{Role: "assistant"}, // Empty failed turns must still be skipped.
		{Role: "user", Content: "Check once more."},
	}
	tools := []NativeTool{{Name: "inspect_source", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
	_, err := newOpenCodeGoTestProvider(t, srv.URL).Chat(context.Background(), messages, "deepseek-v4.1-flash", tools, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	body := <-requests
	if body["model"] != "deepseek-v4.1-flash" || len(body["tools"].([]any)) != 1 {
		t.Fatal("expected DeepSeek request with tools")
	}
	wire := body["messages"].([]any)
	if len(wire) != 9 {
		t.Fatalf("wire message count = %d, want 9", len(wire))
	}
	wantReasoning := map[int]string{
		2: messages[2].Reasoning,
		4: messages[4].Reasoning,
		6: messages[6].Reasoning,
	}
	for i, raw := range wire {
		msg := raw.(map[string]any)
		got, present := msg["reasoning_content"]
		if want, ok := wantReasoning[i]; ok {
			if got != want {
				t.Errorf("message[%d] reasoning_content = %v, want %q", i, got, want)
			}
		} else if present {
			t.Errorf("message[%d] unexpectedly carries reasoning_content", i)
		}
	}
	if wire[2].(map[string]any)["content"] != "" || wire[4].(map[string]any)["content"] != messages[4].Content {
		t.Error("assistant text content changed")
	}
	parts := wire[6].(map[string]any)["content"].([]any)
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "return a + b" {
		t.Error("assistant content parts changed")
	}
}
