package core

import (
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type workerToolDiscoveryFixture struct {
	parent *Thinker
	thread *Thread
	calls  *atomic.Int64
}

func newWorkerToolDiscoveryFixture(
	t *testing.T,
	server, localName, description string,
	noSpawn bool,
	loading *MCPToolLoadingConfig,
) workerToolDiscoveryFixture {
	t.Helper()
	t.Setenv("APTEVA_TOOL_SEARCH", "on")

	parent := newTestThinkerFull()
	parent.registry = NewToolRegistry("test")
	parent.toolIndex = NewToolIndex()
	parent.telemetry = NewTelemetry()
	t.Cleanup(func() {
		parent.threads.KillAll()
		parent.telemetry.Stop()
		parent.Stop()
	})

	var calls atomic.Int64
	fullName := server + "_" + localName
	parent.registry.Register(&ToolDef{
		Name: fullName, Description: description,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{"type": "string"},
			},
		},
		MCP: true, MCPServer: server,
		Handler: func(map[string]string) ToolResponse {
			calls.Add(1)
			return ToolResponse{Text: "WORKER_TOOL_OK"}
		},
	})
	parent.toolIndex.Add(server, []mcpToolDef{{
		Name: localName, Description: description,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{"type": "string"},
			},
		},
	}}, noSpawn, loading)

	if err := parent.threads.SpawnWithOpts(
		"worker", "Wait for an explicit task.", nil,
		SpawnOpts{DeferRun: true},
	); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	thread := parent.threads.threads["worker"]
	if thread == nil {
		t.Fatal("worker missing after spawn")
	}
	if thread.Tools[fullName] {
		t.Fatalf("setup: runtime tool %q unexpectedly entered spawn-time thread.Tools", fullName)
	}
	return workerToolDiscoveryFixture{parent: parent, thread: thread, calls: &calls}
}

func nativeToolNames(tools []NativeTool) map[string]bool {
	names := make(map[string]bool, len(tools))
	for _, tool := range tools {
		names[tool.Name] = true
	}
	return names
}

func waitForWorkerToolResult(t *testing.T, thinker *Thinker, callID string) ToolResult {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, event := range thinker.drainEvents() {
			if event.ToolResult != nil && event.ToolResult.CallID == callID {
				return *event.ToolResult
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for tool result %q", callID)
	return ToolResult{}
}

func TestWorkerSearchToolsActivatesAndExecutesOnSameWorker(t *testing.T) {
	const toolName = "catalog_op_4471"
	fixture := newWorkerToolDiscoveryFixture(
		t, "catalog", "op_4471",
		"Submit a completed compliance form to the records system.",
		false, nil,
	)
	worker := fixture.thread.Thinker
	handler := threadToolHandler(fixture.thread, fixture.parent.threads)

	before := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if before[toolName] {
		t.Fatalf("%s was visible before worker discovery", toolName)
	}

	_, _, searchResults := handler(worker, []toolCall{{
		Name: "search_tools",
		Args: map[string]string{
			"query": "submit completed compliance form records",
		},
		Raw: "search_tools", NativeID: "search-call",
	}}, nil)
	if len(searchResults) != 1 || searchResults[0].IsError || !strings.Contains(searchResults[0].Content, toolName) {
		t.Fatalf("search results = %+v", searchResults)
	}
	if !worker.activeTools[toolName] {
		t.Fatalf("search_tools did not activate %s on the worker", toolName)
	}
	if fixture.parent.activeTools[toolName] {
		t.Fatalf("worker discovery leaked %s into main.activeTools", toolName)
	}

	after := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if !after[toolName] {
		t.Fatalf("%s was not presented to the worker on the next turn", toolName)
	}
	_, names, inline := handler(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "complete"},
		Raw: toolName, NativeID: "tool-call",
	}}, nil)
	if len(inline) != 0 {
		t.Fatalf("external MCP call unexpectedly returned inline results: %+v", inline)
	}
	if len(names) != 1 || names[0] != toolName {
		t.Fatalf("handled tool names = %v", names)
	}
	result := waitForWorkerToolResult(t, worker, "tool-call")
	if result.IsError || result.Content != "WORKER_TOOL_OK" || fixture.calls.Load() != 1 {
		t.Fatalf("worker result=%+v calls=%d", result, fixture.calls.Load())
	}
}

func TestWorkerEventPreloadActivatesAndExecutesTool(t *testing.T) {
	const toolName = "billing_upload_invoice_pdf"
	fixture := newWorkerToolDiscoveryFixture(
		t, "billing", "upload_invoice_pdf",
		"Upload an invoice PDF document to the billing folder.",
		false, nil,
	)
	worker := fixture.thread.Thinker
	worker.directive = ""
	worker.lastInboundForPreload = "Please upload the invoice PDF document to the billing folder."

	presented := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if !presented[toolName] || !worker.activeTools[toolName] {
		t.Fatalf("event preload did not present/activate %s: presented=%v active=%v", toolName, presented, worker.activeTools)
	}
	threadToolHandler(fixture.thread, fixture.parent.threads)(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "invoice.pdf"},
		Raw: toolName, NativeID: "preload-call",
	}}, nil)
	result := waitForWorkerToolResult(t, worker, "preload-call")
	if result.IsError || fixture.calls.Load() != 1 {
		t.Fatalf("preloaded worker result=%+v calls=%d", result, fixture.calls.Load())
	}
}

func TestWorkerAlwaysLoadedToolExecutesWithoutPollutingActiveTools(t *testing.T) {
	const toolName = "crm_lookup_contact"
	fixture := newWorkerToolDiscoveryFixture(
		t, "crm", "lookup_contact", "Look up a CRM contact.",
		false, &MCPToolLoadingConfig{Default: ToolLoadAlways},
	)
	worker := fixture.thread.Thinker
	presented := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if !presented[toolName] {
		t.Fatalf("always-loaded worker tool %s was not presented", toolName)
	}
	if worker.activeTools[toolName] {
		t.Fatalf("always-loaded tool %s polluted activeTools", toolName)
	}

	threadToolHandler(fixture.thread, fixture.parent.threads)(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "Ada"},
		Raw: toolName, NativeID: "always-call",
	}}, nil)
	result := waitForWorkerToolResult(t, worker, "always-call")
	if result.IsError || fixture.calls.Load() != 1 {
		t.Fatalf("always-loaded result=%+v calls=%d", result, fixture.calls.Load())
	}
}

func TestWorkerInactiveToolRejectedWithPairedResultTelemetryAndBoundedKick(t *testing.T) {
	const toolName = "catalog_hidden_operation"
	fixture := newWorkerToolDiscoveryFixture(
		t, "catalog", "hidden_operation", "Perform a capability unrelated to waiting.",
		false, nil,
	)
	worker := fixture.thread.Thinker
	handler := threadToolHandler(fixture.thread, fixture.parent.threads)
	presented := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if presented[toolName] || worker.activeTools[toolName] {
		t.Fatalf("setup: inactive tool became visible: presented=%v active=%v", presented, worker.activeTools)
	}

	_, names, results := handler(worker, []toolCall{
		{
			Name: toolName, Args: map[string]string{"input": "guess", "_reason": "Guessing"},
			Raw: toolName, NativeID: "rejected-call",
		},
		{
			Name: "pace", Args: map[string]string{"rate": "normal"},
			Raw: "pace", NativeID: "pace-call",
		},
	}, nil)
	if len(names) == 0 || names[0] != toolName || len(results) != 2 {
		t.Fatalf("names=%v results=%+v; every attempted call must receive a result", names, results)
	}
	resultByID := map[string]ToolResult{}
	for _, result := range results {
		resultByID[result.CallID] = result
	}
	rejected := resultByID["rejected-call"]
	if !rejected.IsError || !strings.Contains(rejected.Content, "not available") {
		t.Fatalf("rejected result = %+v", rejected)
	}
	if _, ok := resultByID["pace-call"]; !ok {
		t.Fatalf("pace call lost its paired result: %+v", results)
	}
	paired := sanitizeToolPairs([]Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{
			{ID: "rejected-call", Name: toolName},
			{ID: "pace-call", Name: "pace"},
		}},
		{Role: "user", ToolResults: results},
	})
	if len(paired) != 2 || len(paired[0].ToolCalls) != 2 || len(paired[1].ToolResults) != 2 {
		t.Fatalf("rejected call/result pairing did not survive sanitization: %+v", paired)
	}
	if fixture.calls.Load() != 0 {
		t.Fatalf("inactive tool executed %d times", fixture.calls.Load())
	}
	if !worker.kickNextTurn {
		t.Fatal("first rejected call did not schedule an immediate correction turn")
	}

	events, _ := worker.telemetry.StoredEvents(0)
	var sawCall, sawFailedResult bool
	for _, event := range events {
		if event.ThreadID != worker.threadID {
			continue
		}
		switch event.Type {
		case "tool.call":
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) == nil && data.ID == "rejected-call" && data.Name == toolName {
				sawCall = true
			}
		case "tool.result":
			var data ToolResultData
			if json.Unmarshal(event.Data, &data) == nil && data.ID == "rejected-call" && data.Name == toolName && !data.Success {
				sawFailedResult = true
			}
		}
	}
	if !sawCall || !sawFailedResult {
		t.Fatalf("rejection telemetry missing: call=%v failed_result=%v", sawCall, sawFailedResult)
	}

	worker.kickNextTurn = false
	handler(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "repeat"},
		Raw: toolName, NativeID: "rejected-again",
	}}, nil)
	if worker.kickNextTurn {
		t.Fatal("consecutive rejected calls created an unbounded correction loop")
	}

	handler(worker, []toolCall{{
		Name: "pace", Args: map[string]string{"rate": "normal"},
		Raw: "pace", NativeID: "clean-turn",
	}}, nil)
	handler(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "later"},
		Raw: toolName, NativeID: "rejected-after-reset",
	}}, nil)
	if !worker.kickNextTurn {
		t.Fatal("rejection guard did not reset after a clean tool turn")
	}
}

func TestWorkerNoSpawnToolCannotBeDiscoveredOrExecuted(t *testing.T) {
	const toolName = "channels_send"
	fixture := newWorkerToolDiscoveryFixture(
		t, "channels", "send", "Send a message through a privileged channel.",
		true, &MCPToolLoadingConfig{Default: ToolLoadAlways},
	)
	worker := fixture.thread.Thinker
	handler := threadToolHandler(fixture.thread, fixture.parent.threads)

	presented := nativeToolNames(worker.prepareNativeTools("openai-codex"))
	if presented[toolName] {
		t.Fatalf("no_spawn tool %s was presented to an ordinary worker", toolName)
	}
	_, _, searchResults := handler(worker, []toolCall{{
		Name: "search_tools", Args: map[string]string{"query": "send privileged channel message"},
		Raw: "search_tools", NativeID: "search-no-spawn",
	}}, nil)
	if len(searchResults) != 1 || strings.Contains(searchResults[0].Content, toolName) || worker.activeTools[toolName] {
		t.Fatalf("no_spawn discovery leaked: result=%+v active=%v", searchResults, worker.activeTools)
	}

	_, _, rejected := handler(worker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "guess"},
		Raw: toolName, NativeID: "no-spawn-call",
	}}, nil)
	if len(rejected) != 1 || !rejected[0].IsError || rejected[0].CallID != "no-spawn-call" {
		t.Fatalf("no_spawn guessed call was not explicitly rejected: %+v", rejected)
	}
	if fixture.calls.Load() != 0 {
		t.Fatalf("no_spawn tool executed %d times", fixture.calls.Load())
	}

	if err := fixture.parent.threads.SpawnWithOpts(
		"explicit-no-spawn", "Try an explicitly named privileged tool.",
		[]string{toolName},
		SpawnOpts{
			DeferRun: true, MCPNames: []string{"channels"},
		},
	); err != nil {
		t.Fatalf("spawn explicit no_spawn worker: %v", err)
	}
	explicit := fixture.parent.threads.threads["explicit-no-spawn"]
	if explicit.Tools[toolName] || explicit.Thinker.activeTools[toolName] {
		t.Fatalf("ordinary spawn retained exact no_spawn grant: tools=%v active=%v", explicit.Tools, explicit.Thinker.activeTools)
	}
	if nativeToolNames(explicit.Thinker.prepareNativeTools("openai-codex"))[toolName] {
		t.Fatalf("ordinary explicit no_spawn tool %s was presented", toolName)
	}
	_, _, explicitResult := threadToolHandler(explicit, fixture.parent.threads)(explicit.Thinker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "guess"},
		Raw: toolName, NativeID: "explicit-no-spawn-call",
	}}, nil)
	if len(explicitResult) != 1 || !explicitResult[0].IsError || fixture.calls.Load() != 0 {
		t.Fatalf("explicit no_spawn call result=%+v calls=%d", explicitResult, fixture.calls.Load())
	}
}

func TestWorkerMalformedSearchReturnsErrorAndImmediateBoundedCorrection(t *testing.T) {
	fixture := newWorkerToolDiscoveryFixture(
		t, "catalog", "lookup", "Look up a catalog record.",
		false, nil,
	)
	worker := fixture.thread.Thinker
	handler := threadToolHandler(fixture.thread, fixture.parent.threads)
	worker.prepareNativeTools("openai-codex")

	_, _, results := handler(worker, []toolCall{{
		Name: "search_tools", Args: map[string]string{},
		Raw: "search_tools", NativeID: "malformed-search",
	}}, nil)
	if len(results) != 1 || !results[0].IsError || results[0].CallID != "malformed-search" {
		t.Fatalf("malformed search result = %+v", results)
	}
	if !worker.kickNextTurn {
		t.Fatal("malformed search did not schedule a correction turn")
	}
	worker.kickNextTurn = false
	handler(worker, []toolCall{{
		Name: "search_tools", Args: map[string]string{},
		Raw: "search_tools", NativeID: "malformed-search-repeat",
	}}, nil)
	if worker.kickNextTurn {
		t.Fatal("repeated malformed search created an unbounded correction loop")
	}
}

func TestPrivilegedAPIStyleThreadKeepsExplicitNoSpawnGrant(t *testing.T) {
	const toolName = "channels_send"
	fixture := newWorkerToolDiscoveryFixture(
		t, "channels", "send", "Send a message through a privileged channel.",
		true, &MCPToolLoadingConfig{Default: ToolLoadAlways},
	)
	if err := fixture.parent.threads.SpawnWithOpts(
		"privileged-api-thread", "Handle authenticated API events.", nil,
		SpawnOpts{
			DeferRun: true,
			MCPNames: []string{"channels"}, BypassNoSpawn: true,
		},
	); err != nil {
		t.Fatalf("spawn privileged API thread: %v", err)
	}
	thread := fixture.parent.threads.threads["privileged-api-thread"]
	if thread == nil {
		t.Fatal("privileged API thread missing")
	}
	presented := nativeToolNames(thread.Thinker.prepareNativeTools("openai-codex"))
	if !presented[toolName] {
		t.Fatalf("explicit privileged no_spawn grant disappeared: presented=%v tools=%v", presented, thread.Tools)
	}
	threadToolHandler(thread, fixture.parent.threads)(thread.Thinker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "reply"},
		Raw: toolName, NativeID: "privileged-call",
	}}, nil)
	result := waitForWorkerToolResult(t, thread.Thinker, "privileged-call")
	if result.IsError || fixture.calls.Load() != 1 {
		t.Fatalf("privileged no_spawn result=%+v calls=%d", result, fixture.calls.Load())
	}
	persisted := persistentThreadState(thread)
	if !persisted.AllowNoSpawn {
		t.Fatal("privileged no_spawn grant was not persisted for restart")
	}

	if err := fixture.parent.threads.SpawnWithOpts(
		"privileged-restored", persisted.Directive, persisted.Tools,
		SpawnOpts{
			DeferRun: true,
			MCPNames: persisted.MCPNames, BypassNoSpawn: persisted.AllowNoSpawn,
		},
	); err != nil {
		t.Fatalf("restore privileged API thread: %v", err)
	}
	restored := fixture.parent.threads.threads["privileged-restored"]
	if !nativeToolNames(restored.Thinker.prepareNativeTools("openai-codex"))[toolName] {
		t.Fatalf("restored privileged no_spawn grant disappeared: tools=%v active=%v", restored.Tools, restored.Thinker.activeTools)
	}
	threadToolHandler(restored, fixture.parent.threads)(restored.Thinker, []toolCall{{
		Name: toolName, Args: map[string]string{"input": "reply after restart"},
		Raw: toolName, NativeID: "restored-privileged-call",
	}}, nil)
	restoredResult := waitForWorkerToolResult(t, restored.Thinker, "restored-privileged-call")
	if restoredResult.IsError || fixture.calls.Load() != 2 {
		t.Fatalf("restored privileged result=%+v calls=%d", restoredResult, fixture.calls.Load())
	}
}

func TestRealtimeToolSnapshotIncludesWorkerDiscovery(t *testing.T) {
	fixture := newWorkerToolDiscoveryFixture(
		t, "calendar", "availability", "Check callback availability.",
		false, nil,
	)
	worker := fixture.thread.Thinker
	worker.activeTools["calendar_availability"] = true
	tools := nativeToolNames(realtimeNativeTools(worker))
	if !tools["calendar_availability"] || !worker.modelToolCallable("calendar_availability", worker.toolAllowlist) {
		t.Fatalf("realtime worker did not retain its discovered tool: tools=%v", tools)
	}
	if !tools["interrupt"] {
		t.Fatal("realtime interrupt tool disappeared")
	}
}
