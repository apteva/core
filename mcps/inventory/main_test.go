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
	bin := filepath.Join(t.TempDir(), "mcp-inventory")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "INVENTORY_DATA_DIR="+dataDir)
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

func TestCheckStock(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"croissant": 10, "baguette": 5})
	c := startServer(t, dir)

	result := c.callTool(t, "check_stock", map[string]string{"item": "croissant"})
	if !strings.Contains(result, "10") {
		t.Errorf("expected 10 croissants, got: %s", result)
	}
}

func TestCheckStock_NotInInventory(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"croissant": 10})
	c := startServer(t, dir)

	result := c.callTool(t, "check_stock", map[string]string{"item": "muffin"})
	if !strings.Contains(result, "not in inventory") {
		t.Errorf("expected not in inventory, got: %s", result)
	}
}

func TestListStock(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"croissant": 10, "baguette": 5})
	c := startServer(t, dir)

	result := c.callTool(t, "list_stock", map[string]string{})
	var stock map[string]int
	json.Unmarshal([]byte(result), &stock)
	if stock["croissant"] != 10 || stock["baguette"] != 5 {
		t.Errorf("unexpected stock: %v", stock)
	}
}

func TestUseStock_Success(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"croissant": 10})
	c := startServer(t, dir)

	result := c.callTool(t, "use_stock", map[string]string{"item": "croissant", "qty": "3"})
	if !strings.Contains(result, "OK") || !strings.Contains(result, "7 remaining") {
		t.Errorf("expected OK with 7 remaining, got: %s", result)
	}

	// Verify persisted
	check := c.callTool(t, "check_stock", map[string]string{"item": "croissant"})
	if !strings.Contains(check, "7") {
		t.Errorf("stock not updated, got: %s", check)
	}
}

func TestUseStock_Insufficient(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"muffin": 2})
	c := startServer(t, dir)

	result := c.callTool(t, "use_stock", map[string]string{"item": "muffin", "qty": "5"})
	if !strings.Contains(result, "FAILED") {
		t.Errorf("expected FAILED, got: %s", result)
	}

	// Stock unchanged
	check := c.callTool(t, "check_stock", map[string]string{"item": "muffin"})
	if !strings.Contains(check, "2") {
		t.Errorf("stock should be unchanged, got: %s", check)
	}
}

func TestUseStock_NotInInventory(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{})
	c := startServer(t, dir)

	result := c.callTool(t, "use_stock", map[string]string{"item": "cake", "qty": "1"})
	if !strings.Contains(result, "FAILED") {
		t.Errorf("expected FAILED, got: %s", result)
	}
}

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "stock.json", map[string]int{"croissant": 10, "baguette": 5, "muffin": 0})
	c := startServer(t, dir)

	// Check stock
	c.callTool(t, "check_stock", map[string]string{"item": "croissant"})

	// Use some
	result := c.callTool(t, "use_stock", map[string]string{"item": "croissant", "qty": "3"})
	if !strings.Contains(result, "OK") {
		t.Fatalf("expected OK, got: %s", result)
	}

	// Use more
	result = c.callTool(t, "use_stock", map[string]string{"item": "baguette", "qty": "5"})
	if !strings.Contains(result, "OK") {
		t.Fatalf("expected OK, got: %s", result)
	}

	// Try muffin (0 stock)
	result = c.callTool(t, "use_stock", map[string]string{"item": "muffin", "qty": "1"})
	if !strings.Contains(result, "FAILED") {
		t.Fatalf("expected FAILED for muffin, got: %s", result)
	}

	// Final stock
	list := c.callTool(t, "list_stock", map[string]string{})
	var stock map[string]int
	json.Unmarshal([]byte(list), &stock)
	if stock["croissant"] != 7 {
		t.Errorf("expected 7 croissants, got %d", stock["croissant"])
	}
	if stock["baguette"] != 0 {
		t.Errorf("expected 0 baguettes, got %d", stock["baguette"])
	}
	if stock["muffin"] != 0 {
		t.Errorf("expected 0 muffins, got %d", stock["muffin"])
	}
	t.Logf("Final stock: %v", stock)
}
