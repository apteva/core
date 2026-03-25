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
	bin := filepath.Join(t.TempDir(), "mcp-social")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "SOCIAL_DATA_DIR="+dataDir)
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

func TestGetChannels(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_channels", map[string]string{})
	if !strings.Contains(result, "twitter") || !strings.Contains(result, "instagram") || !strings.Contains(result, "linkedin") {
		t.Errorf("expected all 3 channels, got: %s", result)
	}
}

func TestPost(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "post", map[string]string{
		"channel": "twitter", "content": "Hello world! #test",
	})
	if !strings.Contains(result, "post-1") {
		t.Errorf("expected post ID, got: %s", result)
	}
}

func TestPost_WithImage(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "post", map[string]string{
		"channel": "instagram", "content": "Beautiful latte art",
		"image": "https://cdn.example.com/latte.jpg",
	})
	if !strings.Contains(result, "instagram") {
		t.Errorf("expected instagram confirmation, got: %s", result)
	}
}

func TestPost_Scheduled(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	c.callTool(t, "post", map[string]string{
		"channel": "linkedin", "content": "We're hiring!",
		"scheduled_time": "09:00",
	})

	// Verify in get_posts
	result := c.callTool(t, "get_posts", map[string]string{})
	var posts []Post
	json.Unmarshal([]byte(result), &posts)
	if len(posts) != 1 {
		t.Fatalf("expected 1 post, got %d", len(posts))
	}
	if posts[0].ScheduledTime != "09:00" {
		t.Errorf("expected scheduled_time 09:00, got %s", posts[0].ScheduledTime)
	}
}

func TestGetPosts_Empty(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_posts", map[string]string{})
	var posts []Post
	json.Unmarshal([]byte(result), &posts)
	if len(posts) != 0 {
		t.Errorf("expected 0, got %d", len(posts))
	}
}

func TestPost_MissingArgs(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	resp := c.call(t, "tools/call", map[string]any{
		"name": "post", "arguments": map[string]string{"channel": "twitter"},
	})
	if resp.Error == nil {
		t.Error("expected error for missing content")
	}
}

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	// Post to all 3 channels
	c.callTool(t, "post", map[string]string{"channel": "twitter", "content": "Morning coffee! #coffee"})
	c.callTool(t, "post", map[string]string{"channel": "instagram", "content": "Latte art", "image": "https://img.fake/latte.jpg"})
	c.callTool(t, "post", map[string]string{"channel": "linkedin", "content": "Join our team!", "scheduled_time": "10:00"})

	result := c.callTool(t, "get_posts", map[string]string{})
	var posts []Post
	json.Unmarshal([]byte(result), &posts)
	if len(posts) != 3 {
		t.Fatalf("expected 3 posts, got %d", len(posts))
	}

	channels := map[string]bool{}
	for _, p := range posts {
		channels[p.Channel] = true
	}
	if !channels["twitter"] || !channels["instagram"] || !channels["linkedin"] {
		t.Errorf("expected all 3 channels, got: %v", channels)
	}
	t.Logf("Posted to %d channels", len(posts))
}
