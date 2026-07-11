package core

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"io"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestMCPReadFailureImmediatelyFailsPendingAndFutureCalls(t *testing.T) {
	stdinReader, stdinWriter := io.Pipe()
	stdoutReader, stdoutWriter := io.Pipe()
	defer stdinReader.Close()
	go io.Copy(io.Discard, stdinReader)

	srv := &MCPServer{
		Name:    "broken",
		stdin:   stdinWriter,
		scanner: bufio.NewScanner(stdoutReader),
		pending: make(map[int64]chan jsonRPCResponse),
	}
	srv.scanner.Buffer(make([]byte, 64*1024), 16<<20)
	go srv.readLoop()

	result := make(chan error, 1)
	go func() {
		_, err := srv.call("tools/list", nil)
		result <- err
	}()
	deadline := time.Now().Add(time.Second)
	for {
		srv.pendMu.Lock()
		pending := len(srv.pending)
		srv.pendMu.Unlock()
		if pending == 1 || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	_ = stdoutWriter.Close()

	select {
	case err := <-result:
		if err == nil || !strings.Contains(err.Error(), "connection closed") {
			t.Fatalf("pending call error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("pending call waited for the three-minute timeout after transport EOF")
	}

	start := time.Now()
	if _, err := srv.call("tools/list", nil); err == nil {
		t.Fatal("future call succeeded on dead MCP connection")
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatal("future call did not fail immediately")
	}
}

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

func TestMCPArgumentsFromStringsUsesInputSchemaForScalars(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"recursive":       map[string]any{"type": "boolean"},
			"duration_max_ms": map[string]any{"type": "integer"},
			"threshold":       map[string]any{"type": "number"},
			"folder":          map[string]any{"type": "string"},
			"ids":             map[string]any{"type": "array"},
		},
	}

	got := mcpArgumentsFromStrings(map[string]string{
		"recursive":       "true",
		"duration_max_ms": "60000",
		"threshold":       "0.75",
		"folder":          "/monika",
		"ids":             `["1","2"]`,
	}, schema)

	if v, ok := got["recursive"].(bool); !ok || !v {
		t.Fatalf("recursive = %#v (%T), want true bool", got["recursive"], got["recursive"])
	}
	if v, ok := got["duration_max_ms"].(int64); !ok || v != 60000 {
		t.Fatalf("duration_max_ms = %#v (%T), want int64 60000", got["duration_max_ms"], got["duration_max_ms"])
	}
	if v, ok := got["threshold"].(float64); !ok || v != 0.75 {
		t.Fatalf("threshold = %#v (%T), want float64 0.75", got["threshold"], got["threshold"])
	}
	if v, ok := got["folder"].(string); !ok || v != "/monika" {
		t.Fatalf("folder = %#v (%T), want string /monika", got["folder"], got["folder"])
	}
	if _, ok := got["ids"].([]any); !ok {
		t.Fatalf("ids = %#v (%T), want []any", got["ids"], got["ids"])
	}
}

func TestMCPArgumentsFromStringsKeepsNumericStringsWithoutSchema(t *testing.T) {
	got := mcpArgumentsFromStrings(map[string]string{
		"file_id": "00123",
		"flag":    "true",
	})

	if v, ok := got["file_id"].(string); !ok || v != "00123" {
		t.Fatalf("file_id = %#v (%T), want string 00123", got["file_id"], got["file_id"])
	}
	if v, ok := got["flag"].(string); !ok || v != "true" {
		t.Fatalf("flag = %#v (%T), want string true", got["flag"], got["flag"])
	}
}

func TestExtractMCPResultImageFromComputerScreenshot(t *testing.T) {
	rawImage := []byte{0x89, 0x50, 0x4e, 0x47}
	result := map[string]any{
		"text": "Success: screenshot action completed. Screenshot attached.",
		"screenshot": map[string]any{
			"_binary":  true,
			"base64":   base64.StdEncoding.EncodeToString(rawImage),
			"mimeType": "image/png",
			"size":     len(rawImage),
		},
		"screenshot_b64": base64.StdEncoding.EncodeToString(rawImage),
		"current_url":    "https://example.com",
	}
	raw, _ := json.Marshal(result)

	text, image := extractMCPResultImage(string(raw))
	if string(image) != string(rawImage) {
		t.Fatalf("image extraction mismatch: got %v want %v", image, rawImage)
	}
	if strings.Contains(text, "screenshot_b64") {
		t.Fatalf("compacted text should not retain screenshot_b64: %s", text)
	}
	if !strings.Contains(text, "attached as image") || !strings.Contains(text, "https://example.com") {
		t.Fatalf("compacted text missing useful fields: %s", text)
	}
}
