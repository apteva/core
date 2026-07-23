package core

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAICompatProviderReportsStreamPhaseTiming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Request-ID", "req-timing-test")
		_, _ = w.Write([]byte(
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"query\\\":\\\"hello\\\"}\"}}]}}]}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	defer server.Close()

	provider := &OpenAICompatProvider{
		name: "test", apiKey: "test", authHeader: "Bearer", url: server.URL,
		models: map[ModelTier]string{ModelMedium: "test-model"},
	}
	resp, err := provider.Chat(
		context.Background(),
		[]Message{{Role: "user", Content: "hello"}},
		"test-model",
		[]NativeTool{{Name: "lookup", Parameters: map[string]any{"type": "object"}}},
		nil, nil, nil,
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	timing := resp.ProviderTiming
	if timing.RequestAttempts != 1 {
		t.Fatalf("request attempts = %d, want 1", timing.RequestAttempts)
	}
	if timing.ResponseHeadersMs == nil || timing.FirstChunkMs == nil || timing.FirstToolCallMs == nil {
		t.Fatalf("missing provider phase timings: %+v", timing)
	}
	if timing.TerminalPhase != "completed" {
		t.Fatalf("terminal phase = %q, want completed", timing.TerminalPhase)
	}
	if timing.StreamChunks != 2 {
		t.Fatalf("stream chunks = %d, want 2", timing.StreamChunks)
	}
	if timing.ProviderRequestIDs["x-request-id"] != "req-timing-test" {
		t.Fatalf("provider request IDs = %#v", timing.ProviderRequestIDs)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "lookup" {
		t.Fatalf("tool calls = %#v", resp.ToolCalls)
	}
}

func TestOpenAICompatProviderRetainsTimingOnStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"error\":{\"message\":\"upstream overloaded\",\"type\":\"server_error\"}}\n\n"))
	}))
	defer server.Close()

	provider := &OpenAICompatProvider{
		name: "test", apiKey: "test", authHeader: "Bearer", url: server.URL,
		models: map[ModelTier]string{ModelMedium: "test-model"},
	}
	resp, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "test-model", nil, nil, nil, nil)
	if err == nil {
		t.Fatal("expected stream error")
	}
	if resp.ProviderTiming.ResponseHeadersMs == nil || resp.ProviderTiming.FirstChunkMs == nil {
		t.Fatalf("error lost provider phase timings: %+v", resp.ProviderTiming)
	}
	if resp.ProviderTiming.TerminalPhase != "stream_error" {
		t.Fatalf("terminal phase = %q, want stream_error", resp.ProviderTiming.TerminalPhase)
	}
}
