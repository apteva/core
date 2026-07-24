package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexSubThreadToolSearchDiscoverAndCall is the worker
// equivalent of TestIntegration_ToolSearch_DiscoverAndCall. It uses a real
// Codex model and an opaque MCP tool name so the only successful path is:
//
//	worker search_tools -> worker activeTools -> next-turn schema ->
//	worker dispatch -> MCP result back to that same worker.
//
//	RUN_CODEX_SUBTHREAD_TOOLSEARCH_SMOKE=1 go test -v -run TestIntegration_CodexSubThreadToolSearchDiscoverAndCall -timeout 5m .
func TestIntegration_CodexSubThreadToolSearchDiscoverAndCall(t *testing.T) {
	if os.Getenv("RUN_CODEX_SUBTHREAD_TOOLSEARCH_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_SUBTHREAD_TOOLSEARCH_SMOKE=1 to run the Codex worker tool-search smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex worker tool-search smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	t.Chdir(t.TempDir())

	mcpServer, mcpCalls := mockToolMCP(
		t,
		"op_9274",
		"Submit a verified partner record to the private compliance ledger.",
	)
	defer mcpServer.Close()

	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Coordinate worker tests.",
		Mode:      ModeAutonomous,
		MCPServers: []MCPServerConfig{{
			Name: "catalog", Transport: "http", URL: mcpServer.URL + "/mcp",
		}},
	}
	parent := NewThinker(token, NewOpenAICodexProvider(token), cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()
	defer func() {
		for _, server := range parent.mcpServers {
			server.Close()
		}
	}()

	const (
		threadID = "codex-discovery-worker"
		toolName = "catalog_op_9274"
	)
	directive := `You are an independent worker. Complete this task yourself:
submit a verified partner record to the private compliance ledger.

The required MCP tool is not currently visible and its name is intentionally
unknown. First call search_tools with a precise capability query. Read its
result. On the immediately following turn, call the exact returned tool once
with input "partner-verified". Wait for its real result.

Do not guess a tool name, spawn, send, evolve, or call done. After the external
tool succeeds, call pace with rate normal.`
	if err := parent.threads.SpawnWithOpts(
		threadID, directive, nil,
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatalf("spawn Codex discovery worker: %v", err)
	}
	thread := parent.threads.threads[threadID]
	if thread == nil {
		t.Fatal("Codex discovery worker missing")
	}
	// Keep the directive in the already-built system prompt, but suppress
	// directive BM25 preloading so this test proves explicit search_tools.
	thread.Thinker.directive = ""
	if thread.Tools[toolName] || thread.Thinker.activeTools[toolName] {
		t.Fatalf("setup: %s was already available to worker", toolName)
	}

	observer, stopObserver := newToolCallObserver(t, parent)
	defer stopObserver()
	go thread.Thinker.Run()

	if !waitFor(150*time.Second, func() bool { return mcpCalls.Load() > 0 }) {
		t.Fatalf("Codex worker never reached the discovered MCP tool; active=%v", thread.Thinker.activeTools)
	}

	deadline := time.Now().Add(10 * time.Second)
	for observer.count(toolName) == 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if observer.count("search_tools") == 0 {
		t.Fatal("Codex worker did not call search_tools")
	}
	if observer.count(toolName) == 0 {
		t.Fatalf("Codex worker did not emit %s after discovery; calls=%v", toolName, observer.snapshot())
	}
	if mcpCalls.Load() != 1 {
		t.Fatalf("MCP calls=%d, want exactly 1", mcpCalls.Load())
	}
	if !thread.Thinker.activeTools[toolName] {
		t.Fatalf("%s was not retained in the worker's activeTools", toolName)
	}
	if parent.activeTools[toolName] {
		t.Fatalf("%s leaked from worker activeTools into main", toolName)
	}

	events, _ := parent.telemetry.StoredEvents(0)
	var trace []string
	var sawSuccessfulResult bool
	for _, event := range events {
		if event.ThreadID != threadID {
			continue
		}
		switch event.Type {
		case "tool.call":
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) == nil {
				trace = append(trace, "CODEX WORKER → "+data.Name)
			}
		case "tool.result":
			var data ToolResultData
			if json.Unmarshal(event.Data, &data) == nil {
				trace = append(trace, "RESULT ← "+data.Name+": "+data.Result)
				if data.Name == toolName && data.Success && strings.Contains(data.Result, "MOCK_TOOL_OK") {
					sawSuccessfulResult = true
				}
			}
		}
	}
	if !sawSuccessfulResult {
		t.Fatalf("successful worker tool.result telemetry missing\n%s", strings.Join(trace, "\n"))
	}
	t.Logf("CODEX SUB-THREAD TOOL DISCOVERY TRACE\n%s", strings.Join(trace, "\n"))
}
