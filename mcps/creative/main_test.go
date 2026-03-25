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
	bin := filepath.Join(t.TempDir(), "mcp-creative")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "CREATIVE_DATA_DIR="+dataDir)
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { stdin.Close(); cmd.Process.Kill(); cmd.Wait() })
	c := &mcpClient{stdin: stdin, scanner: bufio.NewScanner(stdout)}
	c.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	c.call(t, "initialize", map[string]any{
		"protocolVersion": "2024-11-05", "capabilities": map[string]any{},
		"clientInfo": map[string]string{"name": "test", "version": "1.0.0"},
	})
	data, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	fmt.Fprintf(stdin, "%s\n", data)
	return c
}

func (c *mcpClient) call(t *testing.T, method string, params any) jsonRPCResponse {
	t.Helper()
	c.nextID++
	data, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": c.nextID, "method": method, "params": params})
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

func TestGeneratePost_Twitter(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "generate_post", map[string]string{
		"channel": "twitter", "topic": "Monday coffee special",
	})
	if !strings.Contains(result, "coffee") {
		t.Errorf("expected coffee-related content, got: %s", result)
	}
	if !strings.Contains(result, "#") {
		t.Errorf("expected hashtags for twitter, got: %s", result)
	}
	t.Logf("Twitter post: %s", result)
}

func TestGeneratePost_Instagram(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "generate_post", map[string]string{
		"channel": "instagram", "topic": "Latte art",
	})
	if !strings.Contains(result, "Bean & Brew") {
		t.Errorf("expected brand mention, got: %s", result)
	}
	t.Logf("Instagram post: %s", result)
}

func TestGeneratePost_LinkedIn(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "generate_post", map[string]string{
		"channel": "linkedin", "topic": "Hiring baristas",
	})
	if !strings.Contains(result, "Hiring baristas") {
		t.Errorf("expected topic in post, got: %s", result)
	}
	t.Logf("LinkedIn post: %s", result)
}

func TestGenerateImage(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "generate_image", map[string]string{
		"topic": "Latte art", "style": "warm aesthetic",
	})
	if !strings.Contains(result, "Generated image") {
		t.Errorf("expected image description, got: %s", result)
	}
	if !strings.Contains(result, "url:") {
		t.Errorf("expected URL, got: %s", result)
	}
	t.Logf("Image: %s", result)
}

func TestGeneratePost_MissingTopic(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	resp := c.call(t, "tools/call", map[string]any{
		"name": "generate_post", "arguments": map[string]string{"channel": "twitter"},
	})
	if resp.Error == nil {
		t.Error("expected error for missing topic")
	}
}

func TestAuditLog(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	c.callTool(t, "generate_post", map[string]string{"channel": "twitter", "topic": "Coffee"})
	c.callTool(t, "generate_image", map[string]string{"topic": "Coffee"})

	data, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 audit entries, got %d", len(lines))
	}
	var e1 AuditEntry
	json.Unmarshal([]byte(lines[0]), &e1)
	if e1.Tool != "generate_post" {
		t.Errorf("expected generate_post, got %s", e1.Tool)
	}
	if e1.Result == "" {
		t.Error("expected result in audit")
	}
}
