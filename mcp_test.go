package main

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"sync/atomic"
	"testing"
)

// TestCallToolParsesJSONArgs verifies that string values containing JSON arrays
// and objects are parsed into proper types before being sent over MCP.
// This prevents the bug where account_ids=["33"] was sent as the string "[\"33\"]".
func TestCallToolParsesJSONArgs(t *testing.T) {
	// Create a pipe-based mock MCP server that captures the raw request
	serverReader, clientWriter := io.Pipe()
	clientReader, serverWriter := io.Pipe()

	// Track what the server receives
	type capturedCall struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	captured := make(chan capturedCall, 1)

	// Mock MCP server goroutine
	go func() {
		scanner := bufio.NewScanner(serverReader)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			var req jsonRPCRequest
			json.Unmarshal([]byte(line), &req)

			switch req.Method {
			case "tools/call":
				// Capture the raw params to verify arg types
				raw, _ := json.Marshal(req.Params)
				var call capturedCall
				json.Unmarshal(raw, &call)
				captured <- call

				// Send success response
				resp := jsonRPCResponse{
					JSONRPC: "2.0",
					ID:      req.ID,
				}
				resultJSON, _ := json.Marshal(map[string]any{
					"content": []map[string]string{{"type": "text", "text": "ok"}},
				})
				resp.Result = resultJSON
				data, _ := json.Marshal(resp)
				serverWriter.Write(append(data, '\n'))
			}
		}
	}()

	// Create MCPServer with our pipes
	srv := &MCPServer{
		Name:    "test",
		stdin:   clientWriter,
		scanner: bufio.NewScanner(clientReader),
		pending: make(map[int64]chan jsonRPCResponse),
	}
	srv.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	go srv.readLoop()

	tests := []struct {
		name     string
		args     map[string]string
		checkKey string
		wantType string // "array", "object", "string"
	}{
		{
			name:     "JSON array is parsed",
			args:     map[string]string{"account_ids": `["33"]`, "text": "hello"},
			checkKey: "account_ids",
			wantType: "array",
		},
		{
			name:     "JSON object is parsed",
			args:     map[string]string{"config": `{"key":"value"}`, "name": "test"},
			checkKey: "config",
			wantType: "object",
		},
		{
			name:     "plain string stays string",
			args:     map[string]string{"message": "hello world"},
			checkKey: "message",
			wantType: "string",
		},
		{
			name:     "string starting with bracket but invalid JSON stays string",
			args:     map[string]string{"note": "[not valid json"},
			checkKey: "note",
			wantType: "string",
		},
		{
			name:     "nested array is parsed",
			args:     map[string]string{"ids": `[1, 2, 3]`, "label": "items"},
			checkKey: "ids",
			wantType: "array",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			go srv.CallTool("test_tool", tt.args)

			call := <-captured

			val, ok := call.Arguments[tt.checkKey]
			if !ok {
				t.Fatalf("missing key %q in arguments", tt.checkKey)
			}

			switch tt.wantType {
			case "array":
				if _, ok := val.([]any); !ok {
					t.Errorf("expected %q to be []any (JSON array), got %T: %v", tt.checkKey, val, val)
				}
			case "object":
				if _, ok := val.(map[string]any); !ok {
					t.Errorf("expected %q to be map[string]any (JSON object), got %T: %v", tt.checkKey, val, val)
				}
			case "string":
				if _, ok := val.(string); !ok {
					t.Errorf("expected %q to be string, got %T: %v", tt.checkKey, val, val)
				}
			}
		})
	}

	// Verify non-checked args are also correct
	t.Run("mixed args preserve all types", func(t *testing.T) {
		go srv.CallTool("test_tool", map[string]string{
			"ids":   `["a","b"]`,
			"text":  "plain",
			"obj":   `{"x":1}`,
			"empty": "",
		})

		call := <-captured

		if _, ok := call.Arguments["ids"].([]any); !ok {
			t.Error("ids should be array")
		}
		if v, ok := call.Arguments["text"].(string); !ok || v != "plain" {
			t.Error("text should be string 'plain'")
		}
		if _, ok := call.Arguments["obj"].(map[string]any); !ok {
			t.Error("obj should be object")
		}
		if v, ok := call.Arguments["empty"].(string); !ok || v != "" {
			t.Error("empty should be empty string")
		}
	})

	_ = strings.NewReader // keep import
	_ = atomic.Int64{}    // keep import
}
