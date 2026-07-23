package core

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func newTestAPI() (*APIServer, *Thinker) {
	bus := NewEventBus()
	t := &Thinker{
		apiKey:    "test",
		messages:  []Message{{Role: "system", Content: "test"}},
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    &MemoryStore{path: "/dev/null"},
		config:    &Config{Directive: "test directive"},
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
		telemetry: NewTelemetry(),
	}
	t.threads = NewThreadManager(t)
	api := &APIServer{thinker: t, startTime: time.Now()}
	return api, t
}

func TestAPI_Health(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	api.health(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]bool
	json.Unmarshal(w.Body.Bytes(), &body)
	if !body["ok"] {
		t.Error("expected ok: true")
	}
}

func TestAPI_Status(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.iteration = 5
	thinker.rate = RateFast
	thinker.agentSleep = 2 * time.Second
	thinker.model = ModelLarge
	nextWake := time.Date(2026, 7, 23, 12, 34, 56, 789, time.FixedZone("test", 2*60*60))
	thinker.nextWakeAt = nextWake
	thinker.publishRuntimeStatus()

	req := httptest.NewRequest("GET", "/status", nil)
	w := httptest.NewRecorder()
	api.status(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["iteration"].(float64) != 5 {
		t.Errorf("expected iteration 5, got %v", body["iteration"])
	}
	if body["rate"] != "2.0s" {
		t.Errorf("expected rate 2.0s, got %v", body["rate"])
	}
	if body["model"] != "large" {
		t.Errorf("expected model large, got %v", body["model"])
	}
	if body["next_wake_at"] != nextWake.UTC().Format(time.RFC3339Nano) {
		t.Errorf("expected next_wake_at %q, got %v", nextWake.UTC().Format(time.RFC3339Nano), body["next_wake_at"])
	}
	if body["core_version"] == "" {
		t.Errorf("expected core_version in status")
	}
	if body["core_build_time"] == "" {
		t.Errorf("expected core_build_time in status")
	}
	if body["execution_control"] == nil {
		t.Errorf("expected execution_control in status")
	}
}

func TestAPI_ResetThreadReportsCleanupResultAndClearsTransientContext(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.session = NewSession(t.TempDir(), "main")
	thinker.messages = []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Content: "old request"},
		{Role: "assistant", Content: "old answer"},
	}
	for _, message := range thinker.messages[1:] {
		if err := thinker.session.AppendMessage(message, 1, TokenUsage{}); err != nil {
			t.Fatalf("append fixture message: %v", err)
		}
	}
	thinker.requestContext = ephemeralTurnContextState{
		snapshots: []ephemeralTurnContextSnapshot{{
			anchor:  1,
			message: Message{Role: "user", Content: "stale request context", RequestContext: true},
		}},
		signature: "stale",
		active:    true,
	}
	thinker.publishRuntimeStatus()
	thinker.publishContextStatus()

	req := httptest.NewRequest(http.MethodPost, "/threads/main/reset", nil)
	w := httptest.NewRecorder()
	api.threadAction(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var result contextResetResult
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode reset result: %v", err)
	}
	if result.Status != "reset" || result.ID != "main" {
		t.Fatalf("unexpected reset identity: %+v", result)
	}
	if result.BeforeCount != 3 || result.AfterCount != 1 || result.RemovedCount != 2 {
		t.Fatalf("unexpected reset counts: %+v", result)
	}
	if result.RemovedChars <= 0 || result.AfterChars != contextChars([]Message{{Role: "system", Content: "system prompt"}}) {
		t.Fatalf("unexpected reset chars: %+v", result)
	}
	if thinker.session.Count() != 0 {
		t.Fatalf("session count = %d, want 0", thinker.session.Count())
	}
	if len(thinker.messages) != 1 || thinker.messages[0].Role != "system" {
		t.Fatalf("messages were not reset: %#v", thinker.messages)
	}
	if thinker.requestContext.active || len(thinker.requestContext.snapshots) != 0 {
		t.Fatalf("transient request context survived reset: %#v", thinker.requestContext)
	}
	if snapshot := thinker.contextSnapshot(); len(snapshot.Messages) != 1 {
		t.Fatalf("published context has %d messages, want 1", len(snapshot.Messages))
	}
}

type fakeLiveMCP struct {
	name   string
	closed bool
}

func (f *fakeLiveMCP) GetName() string { return f.name }
func (f *fakeLiveMCP) ListTools() ([]mcpToolDef, error) {
	return nil, nil
}
func (f *fakeLiveMCP) Close() {
	f.closed = true
}
func (f *fakeLiveMCP) CallTool(name string, args map[string]string) (ToolResponse, error) {
	return ToolResponse{}, nil
}

func TestReconcileMCPKeepsLiveServerMissingFromPersistedConfig(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.registry = NewToolRegistry("")
	thinker.toolIndex = NewToolIndex()
	thinker.config = &Config{}

	live := &fakeLiveMCP{name: "apteva-server"}
	thinker.mcpServers = []MCPConn{live}

	api.reconcileMCP([]MCPServerConfig{{
		Name:      "apteva-server",
		Command:   "/tmp/apteva-server",
		Transport: "stdio",
		NoSpawn:   true,
	}})

	if live.closed {
		t.Fatal("reconcileMCP closed a live MCP that was present in desired but missing from persisted config")
	}
	if len(thinker.mcpServers) != 1 || thinker.mcpServers[0].GetName() != "apteva-server" {
		t.Fatalf("expected live apteva-server MCP to remain attached, got %#v", thinker.mcpServers)
	}
}

func TestReconcileMCPKeepsLiveNoSpawnServerWhenConfigDiffers(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.registry = NewToolRegistry("")
	thinker.toolIndex = NewToolIndex()
	thinker.config = &Config{}
	thinker.config.SaveMCPServer(MCPServerConfig{
		Name:      "apteva-server",
		Command:   "/old/apteva-server",
		Transport: "stdio",
		NoSpawn:   true,
	})

	live := &fakeLiveMCP{name: "apteva-server"}
	thinker.mcpServers = []MCPConn{live}

	api.reconcileMCP([]MCPServerConfig{{
		Name:      "apteva-server",
		Command:   "/new/apteva-server",
		Transport: "stdio",
		NoSpawn:   true,
	}})

	if live.closed {
		t.Fatal("reconcileMCP closed a live no_spawn MCP while it was still requested")
	}
	if len(thinker.mcpServers) != 1 || thinker.mcpServers[0].GetName() != "apteva-server" {
		t.Fatalf("expected live no_spawn MCP to remain attached, got %#v", thinker.mcpServers)
	}
}

func TestReconcileMCPUpdatesToolLoadingWithoutReconnect(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.registry = NewToolRegistry("")
	thinker.toolIndex = NewToolIndex()
	thinker.config = &Config{}
	oldCfg := MCPServerConfig{
		Name:      "channels",
		URL:       "http://127.0.0.1:7777",
		Transport: "http",
		NoSpawn:   true,
	}
	if err := thinker.config.SaveMCPServer(oldCfg); err != nil {
		t.Fatal(err)
	}
	thinker.toolIndex.Add("channels", []mcpToolDef{{Name: "send", Description: "Send a visible message"}}, true)
	live := &fakeLiveMCP{name: "channels"}
	thinker.mcpServers = []MCPConn{live}

	newCfg := oldCfg
	newCfg.ToolLoading = &MCPToolLoadingConfig{Default: ToolLoadAlways}
	if err := api.reconcileMCP([]MCPServerConfig{newCfg}); err != nil {
		t.Fatal(err)
	}
	if live.closed {
		t.Fatal("tool loading policy update reconnected the live Channels MCP")
	}
	entry, ok := thinker.toolIndex.Get("channels_send")
	if !ok || entry.LoadMode != ToolLoadAlways {
		t.Fatalf("updated index entry = %#v, %v", entry, ok)
	}
	persisted := thinker.config.GetMCPServers()
	if len(persisted) != 1 || persisted[0].ToolLoading == nil || persisted[0].ToolLoading.Default != ToolLoadAlways {
		t.Fatalf("persisted MCP policy = %#v", persisted)
	}
	if thinker.promptCacheEpoch != 1 || thinker.promptCacheResetReason != "tool_loading_policy_changed" {
		t.Fatalf("prompt cache state = epoch %d reason %q", thinker.promptCacheEpoch, thinker.promptCacheResetReason)
	}
}

func TestAPI_ControlStep(t *testing.T) {
	api, _ := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"action": "step"})
	req := httptest.NewRequest("POST", "/control", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.control(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	ec, _ := body["execution_control"].(map[string]any)
	if ec["mode"] != "step" {
		t.Fatalf("execution mode = %v, want step", ec["mode"])
	}
}

func TestAPI_RestoreCheckpointDoesNotCloseQuit(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.execution = NewExecutionController(ExecutionControlConfig{
		Mode:        ExecutionStep,
		Breakpoints: []string{string(ExecutionPhaseToolBefore)},
	})
	thinker.checkpoints = NewExecutionCheckpointStore()
	thinker.iteration = 1

	waitDone := make(chan bool, 1)
	go func() {
		waitDone <- thinker.executionGate(ExecutionPhaseToolBefore, ExecutionGate{
			Tool:    "smoke_tool",
			Summary: "smoke_tool ready",
			Args:    map[string]string{"message": "before"},
		})
	}()

	var checkpointID string
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		st := thinker.executionStatus()
		if st.Waiting && st.RestoreCheckpointID != "" {
			checkpointID = st.RestoreCheckpointID
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if checkpointID == "" {
		t.Fatal("checkpoint was not captured while waiting")
	}

	payload, _ := json.Marshal(map[string]string{
		"action":        "restore_checkpoint",
		"checkpoint_id": checkpointID,
	})
	req := httptest.NewRequest("POST", "/control", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.control(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	select {
	case <-thinker.quit:
		t.Fatal("restore closed thinker.quit; headless core would exit")
	default:
	}
	select {
	case proceed := <-waitDone:
		if proceed {
			t.Fatal("restore released the old gate instead of cancelling it")
		}
	case <-time.After(time.Second):
		t.Fatal("old execution gate was not cancelled")
	}
	close(thinker.quit)
}

func TestAPI_Threads_MainOnly(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body []map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 1 {
		t.Fatalf("expected 1 thread (main), got %d", len(body))
	}
	if body[0]["id"] != "main" {
		t.Errorf("expected id 'main', got %v", body[0]["id"])
	}
}

func TestAPI_Threads_WithSubThreads(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.threads.Spawn("test-thread", "test prompt", []string{"web"})
	defer thinker.threads.Kill("test-thread")

	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	var body []map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 2 {
		t.Fatalf("expected 2 threads (main + test-thread), got %d", len(body))
	}
	if body[0]["id"] != "main" {
		t.Errorf("expected first thread 'main', got %v", body[0]["id"])
	}
	if body[1]["id"] != "test-thread" {
		t.Errorf("expected second thread 'test-thread', got %v", body[1]["id"])
	}
}

func TestAPI_PostEvent(t *testing.T) {
	api, thinker := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"message": "test command"})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	// Check it was injected
	items := thinker.drainEventTexts()
	if len(items) != 1 {
		t.Fatalf("expected 1 item in events, got %d", len(items))
	}
	if items[0] != "[console] test command" {
		t.Errorf("expected '[console] test command', got %q", items[0])
	}
}

func TestAPI_PostEvent_EmptyMessage(t *testing.T) {
	api, _ := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"message": ""})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 400 {
		t.Errorf("expected 400 for empty message, got %d", w.Code)
	}
}

func TestAPI_PostEvent_InvalidJSON(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("POST", "/event", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 400 {
		t.Errorf("expected 400 for invalid JSON, got %d", w.Code)
	}
}

func TestAPI_PostEvent_WrongMethod(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/event", nil)
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405 for GET, got %d", w.Code)
	}
}

func TestAPI_PostEvent_ThreadTarget(t *testing.T) {
	api, thinker := newTestAPI()

	// Subscribe a "webhook-listener" on the bus
	listenerSub := thinker.bus.Subscribe("webhook-listener", 10)

	// Post event targeting the thread
	payload, _ := json.Marshal(map[string]string{
		"message":   "[webhook:omnikit] {\"event\":\"message.received\"}",
		"thread_id": "webhook-listener",
	})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Check response includes thread_id
	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["thread_id"] != "webhook-listener" {
		t.Errorf("expected thread_id 'webhook-listener', got %q", resp["thread_id"])
	}

	// The event should arrive on the listener subscription, NOT on main
	select {
	case ev := <-listenerSub.C:
		if ev.Text != "[webhook:omnikit] {\"event\":\"message.received\"}" {
			t.Errorf("unexpected event text: %q", ev.Text)
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("expected event on webhook-listener subscription")
	}

	// Main should NOT have received it
	mainEvents := thinker.drainEventTexts()
	if len(mainEvents) != 0 {
		t.Errorf("main should not receive thread-targeted event, got %d events", len(mainEvents))
	}
}

func TestAPI_PostEvent_NoThreadID_GoesToMain(t *testing.T) {
	api, thinker := newTestAPI()

	// Post event without thread_id — should go to main
	payload, _ := json.Marshal(map[string]string{"message": "hello main"})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["thread_id"] != "main" {
		t.Errorf("expected thread_id 'main', got %q", resp["thread_id"])
	}

	items := thinker.drainEventTexts()
	if len(items) != 1 {
		t.Fatalf("expected 1 event on main, got %d", len(items))
	}
}

func TestAPI_PostEvent_MainThreadID_GoesToMain(t *testing.T) {
	api, thinker := newTestAPI()

	// Explicitly targeting "main" should behave same as no thread_id
	payload, _ := json.Marshal(map[string]string{"message": "explicit main", "thread_id": "main"})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	items := thinker.drainEventTexts()
	if len(items) != 1 {
		t.Fatalf("expected 1 event on main, got %d", len(items))
	}
}

func TestAPI_Config_Get(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/config", nil)
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["directive"] != "test directive" {
		t.Errorf("expected 'test directive', got %v", body["directive"])
	}
	if body["mode"] != "autonomous" {
		t.Errorf("expected default mode 'autonomous', got %v", body["mode"])
	}
}

func TestAPIConfigToolLoadingRoundTripAndValidation(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.config.MCPServers = []MCPServerConfig{{
		Name:      "channels",
		URL:       "http://127.0.0.1:7777",
		Transport: "http",
		NoSpawn:   true,
		ToolLoading: &MCPToolLoadingConfig{
			Default: ToolLoadAlways,
		},
	}}

	req := httptest.NewRequest(http.MethodGet, "/config", nil)
	w := httptest.NewRecorder()
	api.config(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("GET status = %d: %s", w.Code, w.Body.String())
	}
	var body struct {
		MCPServers []MCPServerConfig `json:"mcp_servers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.MCPServers) != 1 || body.MCPServers[0].ToolLoading == nil || body.MCPServers[0].ToolLoading.Default != ToolLoadAlways {
		t.Fatalf("GET dropped tool_loading: %#v", body.MCPServers)
	}

	badPayload := []byte(`{"mcp_servers":[{"name":"bad","url":"http://example.test","tool_loading":{"default":"sometimes"}}]}`)
	req = httptest.NewRequest(http.MethodPut, "/config", bytes.NewReader(badPayload))
	w = httptest.NewRecorder()
	api.config(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("invalid policy status = %d, want 400: %s", w.Code, w.Body.String())
	}
}

func TestAPI_Config_Put(t *testing.T) {
	api, thinker := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"directive": "new directive"})
	req := httptest.NewRequest("PUT", "/config", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if thinker.config.GetDirective() != "new directive" {
		t.Errorf("directive not updated, got %q", thinker.config.GetDirective())
	}
	if thinker.directive != "new directive" {
		t.Errorf("live directive not reloaded, got %q", thinker.directive)
	}
	select {
	case ev := <-thinker.sub.C:
		t.Fatalf("directive config update should not inject a wake event, got %s %q", ev.Type, ev.Text)
	default:
	}
}

func TestAPIConfigPutReportsPersistenceFailureAndRollsBack(t *testing.T) {
	api, thinker := newTestAPI()
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	thinker.config.path = filepath.Join(blocker, "config.json")
	payload, _ := json.Marshal(map[string]string{"directive": "must-not-apply"})
	req := httptest.NewRequest(http.MethodPut, "/config", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.config(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body=%s", w.Code, w.Body.String())
	}
	if got := thinker.config.GetDirective(); got != "test directive" {
		t.Fatalf("directive changed after failed persistence: %q", got)
	}
}

func TestAPI_Config_WrongMethod(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("DELETE", "/config", nil)
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

// Full HTTP server integration test — verifies routing and real HTTP round-trips
func TestAPI_FullServer(t *testing.T) {
	api, _ := newTestAPI()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", api.health)
	mux.HandleFunc("/status", api.status)
	mux.HandleFunc("/threads", api.threads)
	mux.HandleFunc("/event", api.postEvent)
	mux.HandleFunc("/config", api.config)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// Health
	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("health: expected 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Status
	resp, err = http.Get(srv.URL + "/status")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	var status map[string]any
	json.NewDecoder(resp.Body).Decode(&status)
	resp.Body.Close()
	if _, ok := status["uptime_seconds"]; !ok {
		t.Error("status missing uptime_seconds")
	}

	// Post event
	payload, _ := json.Marshal(map[string]string{"message": "hello"})
	resp, err = http.Post(srv.URL+"/event", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("event: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("event: expected 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Config round-trip
	payload, _ = json.Marshal(map[string]string{"directive": "full server test"})
	req, _ := http.NewRequest("PUT", srv.URL+"/config", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("config PUT: %v", err)
	}
	resp.Body.Close()

	resp, err = http.Get(srv.URL + "/config")
	if err != nil {
		t.Fatalf("config GET: %v", err)
	}
	var cfg map[string]any
	json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if cfg["directive"] != "full server test" {
		t.Errorf("config round-trip failed, got %v", cfg["directive"])
	}

	t.Log("All endpoints working via real HTTP server")
}

// --- Supervised Mode Tests ---

func TestAPI_Status_IncludesMode(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/status", nil)
	w := httptest.NewRecorder()
	api.status(w, req)

	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["mode"] != "autonomous" {
		t.Errorf("expected default mode 'autonomous', got %v", body["mode"])
	}
}

func TestAPI_Config_SetMode(t *testing.T) {
	api, thinker := newTestAPI()

	// Set to cautious
	payload, _ := json.Marshal(map[string]string{"mode": "cautious"})
	req := httptest.NewRequest("PUT", "/config", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if thinker.config.GetMode() != ModeCautious {
		t.Errorf("expected cautious, got %s", thinker.config.GetMode())
	}

	// Verify via GET
	req = httptest.NewRequest("GET", "/config", nil)
	w = httptest.NewRecorder()
	api.config(w, req)

	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["mode"] != "cautious" {
		t.Errorf("expected cautious, got %v", body["mode"])
	}
}

// --- Multimodal API Tests ---

func TestAPI_PostEvent_PlainString(t *testing.T) {
	api, thinker := newTestAPI()
	payload := []byte(`{"message": "hello"}`)
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	items := thinker.drainEventTexts()
	if len(items) != 1 {
		t.Fatalf("expected 1 event, got %d", len(items))
	}
}

func TestAPI_PostEvent_ContentParts(t *testing.T) {
	api, thinker := newTestAPI()
	payload := []byte(`{"message": [
		{"type": "text", "text": "What is this?"},
		{"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
	]}`)
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Parts now flow through the event bus
	time.Sleep(50 * time.Millisecond)
	events := thinker.drainEvents()
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if len(events[0].Parts) != 2 {
		t.Fatalf("expected 2 parts on event, got %d", len(events[0].Parts))
	}
	if events[0].Parts[0].Type != "text" || events[0].Parts[0].Text != "What is this?" {
		t.Errorf("unexpected first part: %+v", events[0].Parts[0])
	}
	if events[0].Parts[1].Type != "image_url" || events[0].Parts[1].ImageURL == nil {
		t.Errorf("unexpected second part: %+v", events[0].Parts[1])
	}
}

func TestAPI_PostEvent_InvalidMessage(t *testing.T) {
	api, _ := newTestAPI()
	payload := []byte(`{"message": 123}`)
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 400 {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

// ---- POST /memory + DELETE /memory/by-id ------------------------------

// withWritableMemory swaps the test thinker's stub MemoryStore for one
// that writes to a temp memory.jsonl in a fresh temp dir. The stub at
// path=/dev/null + nil byID can't handle writes; this gives the upsert
// path a real journal to append to.
func withWritableMemory(t *testing.T, api *APIServer) {
	t.Helper()
	dir := t.TempDir()
	prev, _ := os.Getwd()
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	api.thinker.memory = &MemoryStore{
		path: memoryFile,
		byID: map[string]int{},
	}
}

func TestAPI_MemoryPost_InsertsWithoutID(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	payload, _ := json.Marshal(map[string]any{
		"content": "the agent should know X",
		"tags":    []string{"skill", "skill:foo:bar"},
		"weight":  0.85,
	})
	req := httptest.NewRequest("POST", "/memory", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.memoryRoot(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d (body=%s)", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["action"] != "inserted" {
		t.Errorf("expected action=inserted, got %v", resp["action"])
	}
	if id, _ := resp["id"].(string); id == "" {
		t.Error("expected non-empty id")
	}
	if api.thinker.memory.Count() != 1 {
		t.Errorf("expected 1 active memory, got %d", api.thinker.memory.Count())
	}
}

func TestAPI_MemoryPost_InsertsWithSuppliedID(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	payload, _ := json.Marshal(map[string]any{
		"id":      "skill_42_0",
		"content": "first version",
		"tags":    []string{"skill"},
	})
	req := httptest.NewRequest("POST", "/memory", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.memoryRoot(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d (body=%s)", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["id"] != "skill_42_0" {
		t.Errorf("expected id=skill_42_0, got %v", resp["id"])
	}
	if resp["action"] != "inserted" {
		t.Errorf("expected action=inserted, got %v", resp["action"])
	}
	if !api.thinker.memory.HasID("skill_42_0") {
		t.Error("expected store to have id skill_42_0")
	}
}

func TestAPI_MemoryPost_UpsertsExistingID(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	// Insert first.
	first, _ := json.Marshal(map[string]any{
		"id":      "skill_42_0",
		"content": "first version",
	})
	req1 := httptest.NewRequest("POST", "/memory", bytes.NewReader(first))
	w1 := httptest.NewRecorder()
	api.memoryRoot(w1, req1)
	if w1.Code != 200 {
		t.Fatalf("first POST: expected 200, got %d (%s)", w1.Code, w1.Body.String())
	}

	// Re-POST same id with new content → should supersede.
	second, _ := json.Marshal(map[string]any{
		"id":      "skill_42_0",
		"content": "second version",
		"reason":  "skill body changed",
	})
	req2 := httptest.NewRequest("POST", "/memory", bytes.NewReader(second))
	w2 := httptest.NewRecorder()
	api.memoryRoot(w2, req2)
	if w2.Code != 200 {
		t.Fatalf("second POST: expected 200, got %d (%s)", w2.Code, w2.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w2.Body.Bytes(), &resp)
	if resp["action"] != "upserted" {
		t.Errorf("expected action=upserted, got %v", resp["action"])
	}
	if resp["supersedes"] != "skill_42_0" {
		t.Errorf("expected supersedes=skill_42_0, got %v", resp["supersedes"])
	}
	// Active count should still be 1 (the new record), not 2.
	if api.thinker.memory.Count() != 1 {
		t.Errorf("expected 1 active after upsert, got %d", api.thinker.memory.Count())
	}
	// And the active record's content is the new content.
	active := api.thinker.memory.Active()
	if len(active) != 1 || active[0].Content != "second version" {
		t.Errorf("expected active to be 'second version', got %+v", active)
	}
}

func TestAPI_MemoryPost_RepeatedDeterministicUpsertKeepsOneActiveSkillMemory(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	for _, body := range []map[string]any{
		{"id": "skill_42_0", "content": "first version", "tags": []string{"skill"}},
		{"id": "skill_42_0", "content": "second version", "tags": []string{"skill"}, "reason": "skill changed"},
		{"id": "skill_42_0", "content": "third version", "tags": []string{"skill"}, "reason": "skill changed again"},
	} {
		payload, _ := json.Marshal(body)
		req := httptest.NewRequest("POST", "/memory", bytes.NewReader(payload))
		w := httptest.NewRecorder()
		api.memoryRoot(w, req)
		if w.Code != 200 {
			t.Fatalf("POST body=%v: expected 200, got %d (%s)", body, w.Code, w.Body.String())
		}
	}

	active := api.thinker.memory.Active()
	if len(active) != 1 {
		t.Fatalf("expected one active skill memory after repeated upserts, got %+v", active)
	}
	if active[0].Content != "third version" {
		t.Fatalf("active content = %q, want third version", active[0].Content)
	}
	if active[0].ID == "skill_42_0" {
		t.Fatalf("active replacement should use generated id after supersede, got original id")
	}
	target, ok := api.thinker.memory.UpsertTargetID("skill_42_0")
	if !ok || target != active[0].ID {
		t.Fatalf("upsert target = %q, %v; want active id %q", target, ok, active[0].ID)
	}
}

func TestAPI_MemoryPost_RequiresContent(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	payload, _ := json.Marshal(map[string]any{"id": "x", "content": ""})
	req := httptest.NewRequest("POST", "/memory", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.memoryRoot(w, req)
	if w.Code != 400 {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestAPI_MemoryPost_InvalidJSON(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)
	req := httptest.NewRequest("POST", "/memory", bytes.NewReader([]byte("{not json")))
	w := httptest.NewRecorder()
	api.memoryRoot(w, req)
	if w.Code != 400 {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestAPI_MemoryDeleteByID_DropsRecord(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	id, _ := api.thinker.memory.RememberWithID("skill_42_0", "body", []string{"skill"}, 0.7)
	if !api.thinker.memory.HasID(id) {
		t.Fatal("setup: id should exist")
	}

	req := httptest.NewRequest("DELETE", "/memory/by-id/"+id, nil)
	w := httptest.NewRecorder()
	api.memoryItem(w, req)
	if w.Code != 200 {
		t.Fatalf("expected 200, got %d (body=%s)", w.Code, w.Body.String())
	}
	if api.thinker.memory.Count() != 0 {
		t.Errorf("expected 0 active after delete, got %d", api.thinker.memory.Count())
	}
}

func TestAPI_MemoryDeleteByID_IdempotentOnMissing(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	req := httptest.NewRequest("DELETE", "/memory/by-id/never-existed", nil)
	w := httptest.NewRecorder()
	api.memoryItem(w, req)
	if w.Code != 200 {
		t.Errorf("expected 200 (idempotent no-op), got %d (body=%s)", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["noop"] != true {
		t.Errorf("expected noop=true, got %v", resp["noop"])
	}
}

func TestAPI_MemoryDeleteByID_RejectsNonDelete(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)

	req := httptest.NewRequest("GET", "/memory/by-id/abc", nil)
	w := httptest.NewRecorder()
	api.memoryItem(w, req)
	if w.Code != 405 {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestAPI_MemoryRoot_RejectsBadMethod(t *testing.T) {
	api, _ := newTestAPI()
	withWritableMemory(t, api)
	req := httptest.NewRequest("PATCH", "/memory", nil)
	w := httptest.NewRecorder()
	api.memoryRoot(w, req)
	if w.Code != 405 {
		t.Errorf("expected 405, got %d", w.Code)
	}
}
