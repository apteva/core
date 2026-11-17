package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
)

// Optional paid-test instrumentation, never runtime behavior. Ask the upstream
// endpoint to compare consecutive responses without modifying their prompts.
// Codex support is intentionally tested, rather than inferred from public API
// documentation. Responses are buffered only in this diagnostic test mode.
func installSolCacheDiagnostics(t *testing.T) {
	t.Helper()
	original := llmHTTPClient
	client := *original
	transport := original.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	var mu sync.Mutex
	previous := map[string]string{}
	client.Transport = auditRoundTripper(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodPost || req.Header.Get("session_id") == "" {
			return transport.RoundTrip(req)
		}
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
		var root map[string]any
		if err := json.Unmarshal(body, &root); err != nil {
			return nil, err
		}
		id := req.Header.Get("session_id")
		mu.Lock()
		baseline := previous[id]
		mu.Unlock()
		if baseline != "" {
			root["prompt_cache_options"] = map[string]any{"comparison_response_id": baseline}
			body, err = json.Marshal(root)
			if err != nil {
				return nil, err
			}
		}
		copy := req.Clone(req.Context())
		copy.Body = io.NopCloser(bytes.NewReader(body))
		copy.ContentLength = int64(len(body))
		copy.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
		response, err := transport.RoundTrip(copy)
		if err != nil {
			return nil, err
		}
		wire, err := io.ReadAll(response.Body)
		_ = response.Body.Close()
		if err != nil {
			return nil, err
		}
		response.Body = io.NopCloser(bytes.NewReader(wire))
		t.Logf("DIAGNOSTIC session=%s cache_key=%s status=%d response_turn_state=%t", promptCacheShortHash([]byte(id)), promptCacheShortHash([]byte(fmt.Sprint(root["prompt_cache_key"]))), response.StatusCode, response.Header.Get("x-codex-turn-state") != "")
		for _, line := range bytes.Split(wire, []byte{'\n'}) {
			line = bytes.TrimPrefix(line, []byte("data: "))
			var event struct {
				Type     string `json:"type"`
				Response struct {
					ID          string          `json:"id"`
					Diagnostics json.RawMessage `json:"prompt_cache_diagnostics"`
				} `json:"response"`
			}
			if json.Unmarshal(line, &event) == nil && event.Type == "response.completed" {
				mu.Lock()
				previous[id] = event.Response.ID
				mu.Unlock()
				t.Logf("DIAGNOSTIC session=%s comparison_requested=%t result=%s", promptCacheShortHash([]byte(id)), baseline != "", event.Response.Diagnostics)
			}
		}
		return response, nil
	})
	llmHTTPClient = &client
	t.Cleanup(func() { llmHTTPClient = original })
}
