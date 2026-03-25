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
	bin := filepath.Join(t.TempDir(), "mcp-orders")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "ORDERS_DATA_DIR="+dataDir)
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

func TestGetOrders_Empty(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_orders", map[string]string{})
	var orders []Order
	json.Unmarshal([]byte(result), &orders)
	if len(orders) != 0 {
		t.Errorf("expected 0, got %d", len(orders))
	}
}

func TestGetOrders_Pending(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "orders.json", []Order{
		{ID: "o1", Item: "croissant", Qty: 2, Status: "pending"},
		{ID: "o2", Item: "baguette", Qty: 1, Status: "ready"},
	})
	c := startServer(t, dir)
	result := c.callTool(t, "get_orders", map[string]string{})
	var orders []Order
	json.Unmarshal([]byte(result), &orders)
	if len(orders) != 1 {
		t.Fatalf("expected 1 pending, got %d", len(orders))
	}
	if orders[0].ID != "o1" {
		t.Errorf("expected o1, got %s", orders[0].ID)
	}
}

func TestGetOrder(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "orders.json", []Order{
		{ID: "o1", Item: "croissant", Qty: 2, Status: "pending"},
	})
	c := startServer(t, dir)
	result := c.callTool(t, "get_order", map[string]string{"id": "o1"})
	var order Order
	json.Unmarshal([]byte(result), &order)
	if order.Item != "croissant" || order.Qty != 2 {
		t.Errorf("unexpected order: %+v", order)
	}
}

func TestGetOrder_NotFound(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)
	result := c.callTool(t, "get_order", map[string]string{"id": "nope"})
	if !strings.Contains(result, "not found") {
		t.Errorf("expected not found, got: %s", result)
	}
}

func TestUpdateOrder(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "orders.json", []Order{
		{ID: "o1", Item: "croissant", Qty: 2, Status: "pending"},
	})
	c := startServer(t, dir)

	result := c.callTool(t, "update_order", map[string]string{"id": "o1", "status": "preparing"})
	if !strings.Contains(result, "preparing") {
		t.Errorf("expected preparing confirmation, got: %s", result)
	}

	// Verify persisted
	detail := c.callTool(t, "get_order", map[string]string{"id": "o1"})
	var order Order
	json.Unmarshal([]byte(detail), &order)
	if order.Status != "preparing" {
		t.Errorf("expected preparing, got %s", order.Status)
	}
}

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "orders.json", []Order{
		{ID: "o1", Item: "croissant", Qty: 2, Status: "pending"},
		{ID: "o2", Item: "muffin", Qty: 1, Status: "pending"},
	})
	c := startServer(t, dir)

	// Get pending
	result := c.callTool(t, "get_orders", map[string]string{})
	var orders []Order
	json.Unmarshal([]byte(result), &orders)
	if len(orders) != 2 {
		t.Fatalf("expected 2 pending, got %d", len(orders))
	}

	// Process o1: preparing → ready
	c.callTool(t, "update_order", map[string]string{"id": "o1", "status": "preparing"})
	c.callTool(t, "update_order", map[string]string{"id": "o1", "status": "ready"})

	// Cancel o2
	c.callTool(t, "update_order", map[string]string{"id": "o2", "status": "cancelled"})

	// No more pending
	result = c.callTool(t, "get_orders", map[string]string{})
	json.Unmarshal([]byte(result), &orders)
	if len(orders) != 0 {
		t.Errorf("expected 0 pending after processing, got %d", len(orders))
	}
}
