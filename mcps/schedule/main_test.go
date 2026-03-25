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
	bin := filepath.Join(t.TempDir(), "mcp-schedule")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "SCHEDULE_DATA_DIR="+dataDir)
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

func writeJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	data, _ := json.MarshalIndent(v, "", "  ")
	os.WriteFile(filepath.Join(dir, name), data, 0644)
}

func TestGetSchedule_Empty(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_schedule", map[string]string{})
	var slots []Slot
	json.Unmarshal([]byte(result), &slots)
	if len(slots) != 0 {
		t.Errorf("expected 0, got %d", len(slots))
	}
}

func TestGetSchedule_WithSlots(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "schedule.json", []Slot{
		{ID: "s1", Channel: "twitter", Topic: "Coffee special", Time: "09:00", Status: "planned"},
		{ID: "s2", Channel: "instagram", Topic: "Latte art", Time: "12:00", Status: "planned"},
	})
	c := startServer(t, dir)
	result := c.callTool(t, "get_schedule", map[string]string{})
	var slots []Slot
	json.Unmarshal([]byte(result), &slots)
	if len(slots) != 2 {
		t.Fatalf("expected 2, got %d", len(slots))
	}
}

func TestUpdateSlot(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "schedule.json", []Slot{
		{ID: "s1", Channel: "twitter", Topic: "Coffee", Time: "09:00", Status: "planned"},
	})
	c := startServer(t, dir)

	result := c.callTool(t, "update_slot", map[string]string{"id": "s1", "status": "content_ready"})
	if !strings.Contains(result, "content_ready") {
		t.Errorf("expected content_ready, got: %s", result)
	}

	// Verify
	sched := c.callTool(t, "get_schedule", map[string]string{})
	var slots []Slot
	json.Unmarshal([]byte(sched), &slots)
	if slots[0].Status != "content_ready" {
		t.Errorf("expected content_ready, got %s", slots[0].Status)
	}
}

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "schedule.json", []Slot{
		{ID: "s1", Channel: "twitter", Topic: "Morning special", Time: "09:00", Status: "planned"},
	})
	c := startServer(t, dir)

	// planned → content_ready → posted
	c.callTool(t, "update_slot", map[string]string{"id": "s1", "status": "content_ready"})
	c.callTool(t, "update_slot", map[string]string{"id": "s1", "status": "posted"})

	sched := c.callTool(t, "get_schedule", map[string]string{})
	var slots []Slot
	json.Unmarshal([]byte(sched), &slots)
	if slots[0].Status != "posted" {
		t.Errorf("expected posted, got %s", slots[0].Status)
	}
}
