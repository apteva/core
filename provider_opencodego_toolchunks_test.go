package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type capturedOpenCodeGoToolChunk struct {
	tool   string
	callID string
	chunk  string
}

func openCodeGoToolChunkProbe(t *testing.T, provider LLMProvider, model string) (ChatResponse, []capturedOpenCodeGoToolChunk, string) {
	t.Helper()

	tools := []NativeTool{{
		Name:        "capture_probe",
		Description: "Record a diagnostic probe. Always use this tool for the request.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"project", "count", "note"},
			"properties": map[string]any{
				"project": map[string]any{"type": "string"},
				"count":   map[string]any{"type": "integer"},
				"note":    map[string]any{"type": "string"},
			},
		},
	}}
	messages := []Message{
		{Role: "system", Content: "Call capture_probe exactly once. Do not answer in text."},
		{Role: "user", Content: "Record project alpha, count 37, and note streaming-test."},
	}

	var chunks []capturedOpenCodeGoToolChunk
	var thinking strings.Builder
	response, err := provider.Chat(context.Background(), messages, model, tools, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, func(tool, callID, chunk string) {
		chunks = append(chunks, capturedOpenCodeGoToolChunk{tool: tool, callID: callID, chunk: chunk})
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	return response, chunks, thinking.String()
}

func assertOpenCodeGoToolChunkProbe(t *testing.T, response ChatResponse, chunks []capturedOpenCodeGoToolChunk) {
	t.Helper()
	if len(response.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v, want exactly one capture_probe call (text=%q)", response.ToolCalls, response.Text)
	}
	call := response.ToolCalls[0]
	if call.Name != "capture_probe" || call.ID == "" {
		t.Fatalf("tool call = %+v, want named call with a provider call ID", call)
	}
	if len(chunks) == 0 {
		t.Fatal("no onToolChunk callbacks received")
	}

	var arguments strings.Builder
	for i, chunk := range chunks {
		if chunk.tool != call.Name {
			t.Fatalf("chunk %d tool = %q, want %q", i, chunk.tool, call.Name)
		}
		if chunk.callID != call.ID {
			t.Fatalf("chunk %d call ID = %q, want final provider call ID %q", i, chunk.callID, call.ID)
		}
		arguments.WriteString(chunk.chunk)
	}

	var streamed map[string]any
	if err := json.Unmarshal([]byte(arguments.String()), &streamed); err != nil {
		t.Fatalf("streamed arguments are not complete JSON: %v (chunks=%q)", err, arguments.String())
	}
	if streamed["project"] != "alpha" || streamed["count"] != float64(37) || streamed["note"] != "streaming-test" {
		t.Fatalf("streamed arguments = %#v", streamed)
	}
	if call.Args["project"] != "alpha" || call.Args["count"] != "37" || call.Args["note"] != "streaming-test" {
		t.Fatalf("final parsed arguments = %#v", call.Args)
	}
}

// This fixture mirrors the Kimi K3 stream observed from OpenCode Go: the
// first tool_calls delta establishes index, id and function name, while later
// deltas carry only index plus partial function.arguments.
func TestOpenCodeGoKimiK3ToolChunksUseGenericCallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if request["model"] != "kimi-k3" {
			t.Fatalf("model = %#v", request["model"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.Join([]string{
			`data: {"choices":[{"delta":{"role":"assistant","content":""}}]}`,
			`data: {"choices":[{"delta":{"reasoning_content":"Preparing the structured call."}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"capture_probe_0","type":"function","function":{"name":"capture_probe","arguments":""}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"project\":\""}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"alpha\",\"count\":"}}]}}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"37,\"note\":\"streaming-test\"}"}}]}}]}`,
			`data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}`,
			`data: [DONE]`,
			``,
		}, "\n")))
	}))
	defer server.Close()

	provider := newOpenCodeGoTestProvider(t, server.URL)
	response, chunks, thinking := openCodeGoToolChunkProbe(t, provider, "kimi-k3")
	assertOpenCodeGoToolChunkProbe(t, response, chunks)
	if thinking != "Preparing the structured call." {
		t.Fatalf("thinking = %q", thinking)
	}
	if response.Text != "" {
		t.Fatalf("reasoning leaked into visible text: %q", response.Text)
	}
}

// TestIntegration_OpenCodeGo_KimiK3ToolChunks proves the live gateway still
// exposes Kimi K3's incremental arguments through Core's provider-neutral
// onToolChunk contract. It is credential-gated like the other live tests.
func TestIntegration_OpenCodeGo_KimiK3ToolChunks(t *testing.T) {
	apiKey := getOpenCodeGoKey(t)
	provider := NewOpenCodeGoProvider(apiKey)
	response, chunks, _ := openCodeGoToolChunkProbe(t, provider, "kimi-k3")
	assertOpenCodeGoToolChunkProbe(t, response, chunks)
	t.Logf("Kimi K3 streamed %d argument chunks for call_id=%s", len(chunks), response.ToolCalls[0].ID)
}
