package core

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAICompatibleStreamErrorIsReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"error\":{\"message\":\"upstream overloaded\",\"type\":\"server_error\"}}\n\n"))
	}))
	defer server.Close()
	provider := &OpenAICompatProvider{
		name: "test", apiKey: "test", authHeader: "Bearer", url: server.URL,
		models: map[ModelTier]string{ModelMedium: "test-model"},
	}
	_, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "test-model", nil, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "upstream overloaded") {
		t.Fatalf("stream error = %v", err)
	}
}

func TestAnthropicStreamErrorIsReturned(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try later\"}}\n\n"))
	}))
	defer server.Close()
	provider := &AnthropicProvider{
		apiKey: "test", url: server.URL,
		models: map[ModelTier]string{ModelMedium: "test-model"},
	}
	_, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "test-model", nil, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "try later") {
		t.Fatalf("stream error = %v", err)
	}
}
