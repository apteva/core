package core

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// TestMCPHTTP_FollowsRedirectAndParsesSSE verifies the two fixes in
// connectMCPHTTP:
//
//  1. The client manually follows 307/308 redirects with the POST body
//     preserved (Go's default client drops the body on 307 for POSTs).
//  2. The client parses SSE-framed response bodies (event: / data: lines)
//     in addition to plain JSON.
//
// The mock server mimics Composio's hosted MCP:
//   - POST /mcp → 307 Location /mcp/inner
//   - POST /mcp/inner → 200 with Content-Type: text/event-stream and a
//     single data: JSON-RPC response line.
func TestMCPHTTP_FollowsRedirectAndParsesSSE(t *testing.T) {
	var innerHits atomic.Int64

	mux := http.NewServeMux()
	mux.HandleFunc("/mcp/inner", func(w http.ResponseWriter, r *http.Request) {
		innerHits.Add(1)
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
			http.Error(w, "method", 405)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req jsonRPCRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("body not valid JSON-RPC: %v body=%q", err, string(body))
			http.Error(w, "bad body", 400)
			return
		}
		// Body MUST be present on the redirected POST — this is what Go's
		// default client would drop.
		if req.Method == "" {
			t.Errorf("request body empty after redirect — body was dropped")
		}

		// Respond based on method. Every response is SSE-framed to exercise
		// decodeMCPBody.
		var result map[string]any
		switch req.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2025-03-26",
				"serverInfo":      map[string]string{"name": "mock", "version": "1.0"},
				"capabilities":    map[string]any{"tools": map[string]any{}},
			}
		case "tools/list":
			result = map[string]any{
				"tools": []map[string]any{
					{
						"name":        "echo",
						"description": "return input",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{"text": map[string]any{"type": "string"}},
						},
					},
				},
			}
		case "tools/call":
			result = map[string]any{
				"content": []map[string]any{
					{"type": "text", "text": "echoed"},
				},
			}
		case "notifications/initialized":
			// Notifications — no response needed; respond with empty SSE frame
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(200)
			_, _ = io.WriteString(w, "event: message\ndata: {}\n\n")
			return
		default:
			t.Errorf("unexpected method %q", req.Method)
			http.Error(w, "bad method", 400)
			return
		}
		resp := jsonRPCResponse{JSONRPC: "2.0", ID: req.ID}
		raw, _ := json.Marshal(result)
		resp.Result = raw
		envelope, _ := json.Marshal(resp)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = fmt.Fprintf(w, "event: message\ndata: %s\n\n", envelope)
	})
	// Root: always redirect to /mcp/inner with a 307
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", "/mcp/inner")
		w.WriteHeader(http.StatusTemporaryRedirect)
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	rootURL := server.URL + "/mcp"

	// Connect — this runs initialize through the redirect path.
	srv, err := connectMCPHTTP("mockmcp", rootURL)
	if err != nil {
		t.Fatalf("connectMCPHTTP: %v", err)
	}
	defer srv.Close()

	// The client should have captured the resolved URL so subsequent calls
	// skip the redirect entirely.
	srv.mu.Lock()
	finalURL := srv.url
	srv.mu.Unlock()
	if !strings.HasSuffix(finalURL, "/mcp/inner") {
		t.Errorf("expected srv.url to be updated to /mcp/inner, got %q", finalURL)
	}

	// tools/list should succeed and return one tool
	tools, err := srv.ListTools()
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "echo" {
		t.Errorf("unexpected tools: %+v", tools)
	}

	// tools/call should succeed and return the echoed text
	got, err := srv.CallTool("echo", map[string]string{"text": "hi"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if got.Text != "echoed" {
		t.Errorf("CallTool: got %q, want %q", got.Text, "echoed")
	}
	if got.IsError {
		t.Errorf("CallTool: IsError=true, want false")
	}

	// At least 3 hits on /mcp/inner (initialize + tools/list + tools/call);
	// more OK (notifications/initialized, etc.)
	if innerHits.Load() < 3 {
		t.Errorf("expected >=3 inner hits, got %d", innerHits.Load())
	}
}

func TestMCPToolMetaWakeOnResultRegistered(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req jsonRPCRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("body not valid JSON-RPC: %v", err)
			http.Error(w, "bad body", http.StatusBadRequest)
			return
		}
		var result map[string]any
		switch req.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2025-03-26",
				"serverInfo":      map[string]string{"name": "mock", "version": "1.0"},
				"capabilities":    map[string]any{"tools": map[string]any{}},
			}
		case "tools/list":
			result = map[string]any{
				"tools": []map[string]any{{
					"name":        "respond",
					"description": "send a message",
					"inputSchema": map[string]any{"type": "object"},
					"_meta":       map[string]any{wakeOnResultMetaKey: "on_error"},
				}},
			}
		case "notifications/initialized":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
			return
		default:
			t.Errorf("unexpected method %q", req.Method)
			http.Error(w, "bad method", http.StatusBadRequest)
			return
		}
		resp := jsonRPCResponse{JSONRPC: "2.0", ID: req.ID}
		resp.Result, _ = json.Marshal(result)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	registry := NewToolRegistry("")
	index := NewToolIndex()
	conns := connectAndRegisterMCP([]MCPServerConfig{{
		Name:      "channels",
		Transport: "http",
		URL:       server.URL + "/mcp",
	}}, registry, index, nil)
	defer func() {
		for _, c := range conns {
			c.Close()
		}
	}()

	def := registry.Get("channels_respond")
	if def == nil {
		t.Fatal("channels_respond not registered")
	}
	if def.WakeOnResult != WakeOnResultOnError {
		t.Fatalf("WakeOnResult=%q, want %q", def.WakeOnResult, WakeOnResultOnError)
	}
}

// TestMCPHTTP_DecodeBody verifies the SSE / JSON discriminator directly so
// regressions in the parser surface quickly.
func TestMCPHTTP_DecodeBody(t *testing.T) {
	cases := []struct {
		name  string
		ct    string
		body  string
		want  string
		isErr bool
	}{
		{
			name: "plain JSON",
			ct:   "application/json",
			body: `{"jsonrpc":"2.0","id":1,"result":{}}`,
			want: `{"jsonrpc":"2.0","id":1,"result":{}}`,
		},
		{
			name: "SSE single frame",
			ct:   "text/event-stream",
			body: "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":42}\n\n",
			want: `{"jsonrpc":"2.0","id":1,"result":42}`,
		},
		{
			name: "SSE multiple frames (last wins)",
			ct:   "text/event-stream",
			body: "event: message\ndata: {\"stale\":true}\n\nevent: message\ndata: {\"fresh\":true}\n\n",
			want: `{"fresh":true}`,
		},
		{
			name: "SSE with CRLF line endings",
			ct:   "text/event-stream",
			body: "event: message\r\ndata: {\"ok\":1}\r\n\r\n",
			want: `{"ok":1}`,
		},
		{
			name:  "SSE missing data line",
			ct:    "text/event-stream",
			body:  "event: message\n\n",
			isErr: true,
		},
		{
			name: "SSE-shaped body without header",
			ct:   "", // empty CT should still sniff by prefix
			body: "event: message\ndata: {\"from\":\"sniff\"}\n\n",
			want: `{"from":"sniff"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := decodeMCPBody(tc.ct, []byte(tc.body))
			if tc.isErr {
				if err == nil {
					t.Errorf("expected error, got %q", string(got))
				}
				return
			}
			if err != nil {
				t.Fatalf("decodeMCPBody: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("got %q, want %q", string(got), tc.want)
			}
		})
	}
}
