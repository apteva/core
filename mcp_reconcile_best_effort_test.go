package core

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
)

// unauthorizedMCP answers every request with 401, like the app gateway does
// for a disabled app installation.
func unauthorizedMCP(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// deadAndLiveMCP sets up a thinker with a saved "tasks" connection that is not
// live and a live "conversations" connection, and returns the configs.
func deadAndLiveMCP(t *testing.T) (*APIServer, *Thinker, *fakeLiveMCP, MCPServerConfig, MCPServerConfig) {
	t.Helper()
	api, thinker := newTestAPI()
	thinker.registry = NewToolRegistry("")
	thinker.toolIndex = NewToolIndex()
	thinker.config = &Config{}
	tasks := MCPServerConfig{Name: "tasks", URL: unauthorizedMCP(t).URL + "/mcp", Transport: "http"}
	conversations := MCPServerConfig{Name: "conversations", URL: "http://127.0.0.1:1/old", Transport: "http"}
	for _, c := range []MCPServerConfig{tasks, conversations} {
		if err := thinker.config.SaveMCPServer(c); err != nil {
			t.Fatal(err)
		}
	}
	live := &fakeLiveMCP{name: "conversations"}
	thinker.mcpServers = []MCPConn{live}
	return api, thinker, live, tasks, conversations
}

func TestReconcileMCPUnchangedDeadEntryDoesNotBlockAnotherUpdate(t *testing.T) {
	api, thinker, live, tasks, conversations := deadAndLiveMCP(t)
	upgraded, _ := mockToolMCP(t, "send", "Send a message")
	t.Cleanup(upgraded.Close)
	conversations.URL = upgraded.URL + "/mcp"

	started := time.Now()
	if err := api.reconcileMCP([]MCPServerConfig{tasks, conversations}); err != nil {
		t.Fatalf("an unchanged, already failing connection blocked another app's update: %v", err)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("update took %v: the failing connection was retried with the full retry budget", elapsed)
	}
	if !live.closed {
		t.Fatal("the replaced conversations connection was not closed")
	}
	var names []string
	for _, srv := range thinker.mcpServers {
		names = append(names, srv.GetName())
	}
	if !reflect.DeepEqual(names, []string{"conversations"}) {
		t.Fatalf("live connections = %v, want only conversations", names)
	}
	if _, ok := thinker.toolIndex.Get("conversations_send"); !ok {
		t.Fatal("the upgraded app's tools were not registered")
	}
	saved := map[string]string{}
	for _, c := range thinker.config.GetMCPServers() {
		saved[c.Name] = c.URL
	}
	if saved["tasks"] != tasks.URL || saved["conversations"] != conversations.URL {
		t.Fatalf("saved configuration = %v, want tasks kept and conversations updated", saved)
	}
}

func TestReconcileMCPFailedRequiredEntryIsNamed(t *testing.T) {
	api, thinker, live, tasks, conversations := deadAndLiveMCP(t)
	conversations.URL = unauthorizedMCP(t).URL + "/mcp"

	payload, _ := json.Marshal(map[string]any{"mcp_servers": []MCPServerConfig{tasks, conversations}})
	req := httptest.NewRequest(http.MethodPut, "/config", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.config(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500; body=%s", w.Code, w.Body.String())
	}
	var body struct {
		Error  string   `json:"error"`
		Failed []string `json:"failed_mcp_servers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("error body is not JSON: %v: %s", err, w.Body.String())
	}
	if !reflect.DeepEqual(body.Failed, []string{"conversations"}) || body.Error == "" {
		t.Fatalf("failed_mcp_servers = %v error = %q, want only the changed conversations entry", body.Failed, body.Error)
	}
	if live.closed || len(thinker.mcpServers) != 1 {
		t.Fatal("a failed update changed the live connections")
	}
}
