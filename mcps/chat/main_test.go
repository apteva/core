package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type mcpClient struct {
	stdin   io.WriteCloser
	scanner *bufio.Scanner
	nextID  int64
}

func startServer(t *testing.T, dataDir string) *mcpClient {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "mcp-chat")
	build := exec.Command("go", "build", "-o", bin, ".")
	build.Dir = "."
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "CHAT_DATA_DIR="+dataDir)
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { stdin.Close(); cmd.Process.Kill(); cmd.Wait() })

	c := &mcpClient{stdin: stdin, scanner: bufio.NewScanner(stdout)}
	c.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Initialize
	c.call(t, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]string{"name": "test", "version": "1.0.0"},
	})
	data, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	fmt.Fprintf(stdin, "%s\n", data)

	return c
}

func (c *mcpClient) call(t *testing.T, method string, params any) jsonRPCResponse {
	t.Helper()
	c.nextID++
	req := map[string]any{"jsonrpc": "2.0", "id": c.nextID, "method": method, "params": params}
	data, _ := json.Marshal(req)
	fmt.Fprintf(c.stdin, "%s\n", data)
	if !c.scanner.Scan() {
		t.Fatal("no response")
	}
	var resp jsonRPCResponse
	json.Unmarshal([]byte(c.scanner.Text()), &resp)
	return resp
}

func (c *mcpClient) callTool(t *testing.T, name string, args map[string]string) string {
	t.Helper()
	resp := c.call(t, "tools/call", map[string]any{"name": name, "arguments": args})
	if resp.Error != nil {
		t.Fatalf("tool %s error: %s", name, resp.Error.Message)
	}
	raw, _ := json.Marshal(resp.Result)
	var result struct {
		Content []struct{ Type, Text string } `json:"content"`
	}
	json.Unmarshal(raw, &result)
	var texts []string
	for _, c := range result.Content {
		if c.Type == "text" {
			texts = append(texts, c.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func writeJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	data, _ := json.MarshalIndent(v, "", "  ")
	os.WriteFile(filepath.Join(dir, name), data, 0644)
}

func TestToolsList(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	resp := c.call(t, "tools/list", nil)
	raw, _ := json.Marshal(resp.Result)
	var result struct {
		Tools []struct{ Name string } `json:"tools"`
	}
	json.Unmarshal(raw, &result)
	names := map[string]bool{}
	for _, tool := range result.Tools {
		names[tool.Name] = true
	}
	if !names["get_messages"] || !names["send_reply"] {
		t.Errorf("missing tools: %v", names)
	}
}

func TestGetMessages_Empty(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_messages", map[string]string{})
	var msgs []UserMessage
	json.Unmarshal([]byte(result), &msgs)
	if len(msgs) != 0 {
		t.Errorf("expected 0, got %d", len(msgs))
	}
}

func TestGetMessages_WithMessages(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "messages.json", []UserMessage{
		{User: "alice", Text: "Hello!"},
		{User: "alice", Text: "How are you?"},
	})
	c := startServer(t, dir)

	result := c.callTool(t, "get_messages", map[string]string{})
	var msgs []UserMessage
	json.Unmarshal([]byte(result), &msgs)
	if len(msgs) != 2 {
		t.Fatalf("expected 2, got %d", len(msgs))
	}

	// Should be consumed — second call returns empty
	result2 := c.callTool(t, "get_messages", map[string]string{})
	json.Unmarshal([]byte(result2), &msgs)
	if len(msgs) != 0 {
		t.Errorf("expected 0 after consume, got %d", len(msgs))
	}
}

func TestSendReply(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "send_reply", map[string]string{
		"user": "alice", "message": "Hi Alice!",
	})
	if !strings.Contains(result, "alice") {
		t.Errorf("expected confirmation, got: %s", result)
	}
}

func TestSendReply_MissingMessage(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	resp := c.call(t, "tools/call", map[string]any{
		"name": "send_reply", "arguments": map[string]string{"user": "alice"},
	})
	if resp.Error == nil {
		t.Error("expected error for missing message")
	}
}

func TestAuditLog(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "messages.json", []UserMessage{{User: "bob", Text: "Hey"}})
	c := startServer(t, dir)

	c.callTool(t, "get_messages", map[string]string{})
	c.callTool(t, "send_reply", map[string]string{"user": "bob", "message": "Hello Bob!"})

	data, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 audit entries, got %d", len(lines))
	}

	var e1, e2 AuditEntry
	json.Unmarshal([]byte(lines[0]), &e1)
	json.Unmarshal([]byte(lines[1]), &e2)
	if e1.Tool != "get_messages" {
		t.Errorf("expected get_messages, got %s", e1.Tool)
	}
	if e2.Tool != "send_reply" || e2.Args["message"] != "Hello Bob!" {
		t.Errorf("expected send_reply with message, got %+v", e2)
	}
}
