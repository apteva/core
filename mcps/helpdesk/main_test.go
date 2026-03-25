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

// mcpClient talks to the helpdesk MCP server over stdio.
type mcpClient struct {
	stdin   io.WriteCloser
	scanner *bufio.Scanner
	nextID  int64
}

func startServer(t *testing.T, dataDir string) *mcpClient {
	t.Helper()

	// Build the server binary
	bin := filepath.Join(t.TempDir(), "mcp-helpdesk")
	build := exec.Command("go", "build", "-o", bin, ".")
	build.Dir = "."
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	cmd := exec.Command(bin)
	cmd.Env = append(os.Environ(), "HELPDESK_DATA_DIR="+dataDir)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		stdin.Close()
		cmd.Process.Kill()
		cmd.Wait()
	})

	c := &mcpClient{
		stdin:   stdin,
		scanner: bufio.NewScanner(stdout),
	}
	c.scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Initialize
	resp := c.call(t, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]string{"name": "test", "version": "1.0.0"},
	})
	if resp.Error != nil {
		t.Fatalf("initialize error: %s", resp.Error.Message)
	}

	// Send initialized notification (no id)
	data, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	})
	fmt.Fprintf(stdin, "%s\n", data)

	return c
}

func (c *mcpClient) call(t *testing.T, method string, params any) jsonRPCResponse {
	t.Helper()
	c.nextID++
	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      c.nextID,
		"method":  method,
		"params":  params,
	}
	data, _ := json.Marshal(req)
	if _, err := fmt.Fprintf(c.stdin, "%s\n", data); err != nil {
		t.Fatalf("write: %v", err)
	}

	if !c.scanner.Scan() {
		t.Fatal("no response from server")
	}
	var resp jsonRPCResponse
	if err := json.Unmarshal([]byte(c.scanner.Text()), &resp); err != nil {
		t.Fatalf("parse response: %v\nraw: %s", err, c.scanner.Text())
	}
	return resp
}

func (c *mcpClient) callTool(t *testing.T, name string, args map[string]string) string {
	t.Helper()
	resp := c.call(t, "tools/call", map[string]any{
		"name":      name,
		"arguments": args,
	})
	if resp.Error != nil {
		t.Fatalf("tool %s error: %s", name, resp.Error.Message)
	}
	// Extract text from MCP content response
	raw, _ := json.Marshal(resp.Result)
	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
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
	if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
		t.Fatal(err)
	}
}

func readAudit(t *testing.T, dir string) []AuditEntry {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	if err != nil {
		return nil
	}
	var entries []AuditEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var e AuditEntry
		json.Unmarshal([]byte(line), &e)
		entries = append(entries, e)
	}
	return entries
}

// --- Tests ---

func TestToolsList(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	resp := c.call(t, "tools/list", nil)
	if resp.Error != nil {
		t.Fatalf("tools/list error: %s", resp.Error.Message)
	}

	raw, _ := json.Marshal(resp.Result)
	var result struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	json.Unmarshal(raw, &result)

	names := make(map[string]bool)
	for _, tool := range result.Tools {
		names[tool.Name] = true
	}

	for _, want := range []string{"list_tickets", "reply_ticket", "close_ticket", "lookup_kb"} {
		if !names[want] {
			t.Errorf("missing tool %q", want)
		}
	}
	t.Logf("tools: %v", names)
}

func TestListTickets_Empty(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	result := c.callTool(t, "list_tickets", map[string]string{})
	var tickets []Ticket
	json.Unmarshal([]byte(result), &tickets)

	if len(tickets) != 0 {
		t.Errorf("expected 0 tickets, got %d", len(tickets))
	}
}

func TestListTickets_WithTickets(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "tickets.json", []Ticket{
		{ID: "t1", Question: "What are your hours?"},
		{ID: "t2", Question: "Do you deliver?"},
	})
	c := startServer(t, dir)

	result := c.callTool(t, "list_tickets", map[string]string{})
	var tickets []Ticket
	json.Unmarshal([]byte(result), &tickets)

	if len(tickets) != 2 {
		t.Fatalf("expected 2 tickets, got %d", len(tickets))
	}
	if tickets[0].ID != "t1" || tickets[1].ID != "t2" {
		t.Errorf("unexpected tickets: %+v", tickets)
	}
}

func TestLookupKB_Hit(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "kb.json", map[string]string{
		"hours":    "We are open Mon-Fri 9am-5pm.",
		"delivery": "We deliver within 10 miles for free.",
	})
	c := startServer(t, dir)

	result := c.callTool(t, "lookup_kb", map[string]string{"query": "What are your hours?"})
	if !strings.Contains(result, "Mon-Fri") {
		t.Errorf("expected hours answer, got: %s", result)
	}
}

func TestLookupKB_Miss(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "kb.json", map[string]string{
		"hours": "Mon-Fri 9-5",
	})
	c := startServer(t, dir)

	result := c.callTool(t, "lookup_kb", map[string]string{"query": "Do you accept Bitcoin?"})
	if result != "No results found" {
		t.Errorf("expected no results, got: %s", result)
	}
}

func TestLookupKB_MissingQuery(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	resp := c.call(t, "tools/call", map[string]any{
		"name":      "lookup_kb",
		"arguments": map[string]string{},
	})
	if resp.Error == nil {
		t.Error("expected error for missing query")
	}
}

func TestReplyTicket(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	result := c.callTool(t, "reply_ticket", map[string]string{
		"id":      "t1",
		"message": "We are open Mon-Fri 9am-5pm.",
	})
	if !strings.Contains(result, "t1") {
		t.Errorf("expected confirmation, got: %s", result)
	}
}

func TestReplyTicket_MissingArgs(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	resp := c.call(t, "tools/call", map[string]any{
		"name":      "reply_ticket",
		"arguments": map[string]string{"id": "t1"},
	})
	if resp.Error == nil {
		t.Error("expected error for missing message")
	}
}

func TestCloseTicket(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "tickets.json", []Ticket{
		{ID: "t1", Question: "Hours?"},
		{ID: "t2", Question: "Delivery?"},
	})
	c := startServer(t, dir)

	// Close t1
	result := c.callTool(t, "close_ticket", map[string]string{"id": "t1"})
	if !strings.Contains(result, "closed") {
		t.Errorf("expected closed confirmation, got: %s", result)
	}

	// Verify t1 removed from tickets.json
	remaining := c.callTool(t, "list_tickets", map[string]string{})
	var tickets []Ticket
	json.Unmarshal([]byte(remaining), &tickets)
	if len(tickets) != 1 {
		t.Fatalf("expected 1 remaining ticket, got %d", len(tickets))
	}
	if tickets[0].ID != "t2" {
		t.Errorf("expected t2 to remain, got %s", tickets[0].ID)
	}
}

func TestCloseTicket_NotFound(t *testing.T) {
	dir := t.TempDir()
	c := startServer(t, dir)

	result := c.callTool(t, "close_ticket", map[string]string{"id": "nonexistent"})
	if !strings.Contains(result, "not found") {
		t.Errorf("expected not found, got: %s", result)
	}
}

func TestAuditLog(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "tickets.json", []Ticket{
		{ID: "t1", Question: "Hours?"},
	})
	writeJSON(t, dir, "kb.json", map[string]string{
		"hours": "Mon-Fri 9-5",
	})
	c := startServer(t, dir)

	// Simulate a full ticket lifecycle: list → lookup → reply → close
	c.callTool(t, "list_tickets", map[string]string{})
	c.callTool(t, "lookup_kb", map[string]string{"query": "hours"})
	c.callTool(t, "reply_ticket", map[string]string{"id": "t1", "message": "Mon-Fri 9-5"})
	c.callTool(t, "close_ticket", map[string]string{"id": "t1"})

	entries := readAudit(t, dir)
	if len(entries) != 4 {
		t.Fatalf("expected 4 audit entries, got %d", len(entries))
	}

	expected := []string{"list_tickets", "lookup_kb", "reply_ticket", "close_ticket"}
	for i, want := range expected {
		if entries[i].Tool != want {
			t.Errorf("audit[%d]: expected %s, got %s", i, want, entries[i].Tool)
		}
	}

	// Verify args recorded
	if entries[2].Args["id"] != "t1" {
		t.Errorf("reply_ticket audit missing ticket id")
	}
	if entries[2].Args["message"] != "Mon-Fri 9-5" {
		t.Errorf("reply_ticket audit missing message")
	}
	t.Logf("audit entries: %+v", entries)
}

func TestFullLifecycle(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, dir, "tickets.json", []Ticket{
		{ID: "t1", Question: "What are your hours?"},
		{ID: "t2", Question: "Do you deliver?"},
	})
	writeJSON(t, dir, "kb.json", map[string]string{
		"hours":    "We are open Mon-Fri 9am-5pm.",
		"delivery": "We deliver within 10 miles for free.",
	})
	c := startServer(t, dir)

	// List tickets
	result := c.callTool(t, "list_tickets", map[string]string{})
	var tickets []Ticket
	json.Unmarshal([]byte(result), &tickets)
	if len(tickets) != 2 {
		t.Fatalf("expected 2 tickets, got %d", len(tickets))
	}

	// Handle each ticket: lookup → reply → close
	for _, ticket := range tickets {
		answer := c.callTool(t, "lookup_kb", map[string]string{"query": ticket.Question})
		if answer == "No results found" {
			t.Errorf("no KB answer for %q", ticket.Question)
		}
		c.callTool(t, "reply_ticket", map[string]string{"id": ticket.ID, "message": answer})
		c.callTool(t, "close_ticket", map[string]string{"id": ticket.ID})
	}

	// Verify all tickets closed
	result = c.callTool(t, "list_tickets", map[string]string{})
	json.Unmarshal([]byte(result), &tickets)
	if len(tickets) != 0 {
		t.Errorf("expected 0 tickets after closing all, got %d", len(tickets))
	}

	// Verify audit: should be list + (lookup + reply + close) × 2 + final list = 8
	entries := readAudit(t, dir)
	if len(entries) != 8 {
		t.Fatalf("expected 8 audit entries, got %d", len(entries))
	}
	t.Logf("full lifecycle: %d audit entries, all tickets resolved", len(entries))
}
