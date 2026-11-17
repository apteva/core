package core

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestToolCatalogCanonicalOrderAndSemanticChanges(t *testing.T) {
	a := json.RawMessage(`{"revision":1,"aliases":{"z":"b","a":"a"},"tools":[{"identity":{"name":"b"},"parameters":{"z":1,"a":2},"loading":"auto","description":"B"},{"identity":{"name":"a"},"description":"A"}]}`)
	b := json.RawMessage(`{"tools":[{"description":"A","identity":{"name":"a"}},{"description":"B","loading":"auto","parameters":{"a":2,"z":1},"identity":{"name":"b"}}],"aliases":{"a":"a","z":"b"},"revision":999}`)
	hash, body, _, _, err := canonicalToolCatalog(a)
	if err != nil {
		t.Fatal(err)
	}
	other, otherBody, _, _, err := canonicalToolCatalog(b)
	if err != nil || hash != other || string(body) != string(otherBody) {
		t.Fatalf("canonical order/revision changed hash: %s %s %v", hash, other, err)
	}
	t.Logf("shared canonical fixture hash=%s", hash)
	for _, changed := range []string{strings.Replace(string(b), `"A"`, `"new A"`, 1), strings.Replace(string(b), `"auto"`, `"deferred"`, 1), strings.Replace(string(b), `"z":1`, `"z":2`, 1), strings.Replace(string(b), `"z":"b"`, `"z":"a"`, 1)} {
		next, _, _, _, e := canonicalToolCatalog(json.RawMessage(changed))
		if e != nil || next == hash {
			t.Fatalf("semantic change missed: %s %v", next, e)
		}
	}
}

func TestMCPTransportRefreshRetainsCacheAndStaleCallProtection(t *testing.T) {
	api, th := newTestAPI()
	defer th.telemetry.Stop()
	th.registry = NewToolRegistry("")
	th.toolIndex = NewToolIndex()
	var initializes atomic.Int32
	var schema atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		w.Header().Set("Content-Type", "application/json")
		var result any = map[string]any{}
		switch req.Method {
		case "initialize":
			initializes.Add(1)
			result = map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{}, "serverInfo": map[string]any{"name": "fixture", "version": "1"}}
		case "tools/list":
			kind := "string"
			if schema.Load() > 0 {
				kind = "number"
			}
			result = map[string]any{"tools": []any{map[string]any{"name": "transcribe", "description": "Transcribe audio", "inputSchema": map[string]any{"type": "object", "properties": map[string]any{"file_id": map[string]any{"type": kind}}}}}}
		case "notifications/initialized":
			w.WriteHeader(202)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
	}))
	defer server.Close()
	cfg := MCPServerConfig{Name: "media", URL: server.URL + "?generation=1", Transport: "http"}
	if err := api.reconcileMCP([]MCPServerConfig{cfg}); err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, conn := range th.mcpServers {
			conn.Close()
		}
	}()
	th.recordToolCatalog()
	epoch := th.promptCacheEpoch
	native, definitions := th.registry.nativeToolSnapshot(nil, map[string]bool{"media_transcribe": true})
	th.recordPresentedTools(native, definitions)
	cfg.URL = server.URL + "?generation=2"
	if err := api.reconcileMCP([]MCPServerConfig{cfg}); err != nil {
		t.Fatal(err)
	}
	th.reloadDirectiveNow()
	th.recordToolCatalog()
	if initializes.Load() != 2 {
		t.Fatalf("transport did not reconnect: %d", initializes.Load())
	}
	if th.promptCacheEpoch != epoch {
		t.Fatalf("identical reconnect reset cache: %d -> %d", epoch, th.promptCacheEpoch)
	}
	call := toolCall{Name: "media_transcribe"}
	th.resolveToolCall(&call)
	if !strings.Contains(call.resolutionError, "capability_changed") {
		t.Fatal("stale transport definition silently retargeted")
	}
	for i := 0; i < 10; i++ {
		if err := api.reconcileMCP([]MCPServerConfig{cfg}); err != nil {
			t.Fatal(err)
		}
		th.reloadDirectiveNow()
		th.recordToolCatalog()
	}
	if th.promptCacheEpoch != epoch || initializes.Load() != 2 {
		t.Fatal("no-op configuration churned transport or cache")
	}
	events, _ := th.telemetry.StoredEvents(0)
	full, refs := 0, 0
	for _, e := range events {
		if e.Type != "tool.catalog" {
			continue
		}
		var d map[string]json.RawMessage
		_ = json.Unmarshal(e.Data, &d)
		if _, ok := d["tools"]; ok {
			full++
		} else {
			refs++
		}
	}
	if full != 1 || refs < 1 {
		t.Fatalf("catalog full=%d references=%d", full, refs)
	}
	schema.Store(1)
	cfg.URL = server.URL + "?generation=3"
	if err := api.reconcileMCP([]MCPServerConfig{cfg}); err != nil {
		t.Fatal(err)
	}
	th.reloadDirectiveNow()
	th.recordToolCatalog()
	if th.promptCacheEpoch != epoch+1 {
		t.Fatalf("schema change reset count: %d expected%d", th.promptCacheEpoch, epoch+1)
	}
	cfg.ToolLoading = &MCPToolLoadingConfig{Default: ToolLoadAlways}
	if err := api.reconcileMCP([]MCPServerConfig{cfg}); err != nil {
		t.Fatal(err)
	}
	th.reloadDirectiveNow()
	th.recordToolCatalog()
	if th.promptCacheEpoch != epoch+2 || initializes.Load() != 3 {
		t.Fatalf("policy change reset/reconnect: epoch=%d connections=%d", th.promptCacheEpoch, initializes.Load())
	}
}

func TestToolCatalogRecoversMissingSnapshotFromServerAck(t *testing.T) {
	telemetry := NewTelemetry()
	defer telemetry.Stop()
	hash, body, tools, aliases, err := canonicalToolCatalog(json.RawMessage(`{"tools":[{"identity":{"name":"media_get"}}],"aliases":{}}`))
	if err != nil {
		t.Fatal(err)
	}
	telemetry.recordCatalog("worker", hash, body, 1, tools, aliases)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"inserted": 1, "missing_catalogs": []string{hash, hash, "unknown"}})
	}))
	defer server.Close()
	telemetry.telemetryURL = server.URL
	if !telemetry.postBatch(server.Client(), []byte(`[]`)) {
		t.Fatal("ingest ack failed")
	}
	events, _ := telemetry.StoredEvents(0)
	full, recovered := 0, 0
	for _, e := range events {
		if e.Type != "tool.catalog" {
			continue
		}
		var d map[string]any
		_ = json.Unmarshal(e.Data, &d)
		if _, ok := d["tools"]; ok {
			full++
		}
		if d["catalog_recovery"] == true {
			recovered++
			if d["catalog_hash"] != hash || e.ThreadID != "worker" {
				t.Fatal("recovery identity changed")
			}
		}
	}
	if full != 2 || recovered != 1 {
		t.Fatalf("recovery counts full=%d recovered=%d", full, recovered)
	}
}

func TestDirectiveReloadOnlyResetsForEffectiveChanges(t *testing.T) {
	_, th := newTestAPI()
	defer th.telemetry.Stop()
	th.registry = NewToolRegistry("")
	th.toolIndex = NewToolIndex()
	th.reloadDirectiveNow()
	epoch := th.promptCacheEpoch
	for i := 0; i < 20; i++ {
		th.reloadDirectiveNow()
	}
	if th.promptCacheEpoch != epoch {
		t.Fatal("identical directive refresh reset cache")
	}
	th.config.Directive = "Updated responsibility"
	th.reloadDirectiveNow()
	th.reloadDirectiveNow()
	if th.promptCacheEpoch != epoch+1 {
		t.Fatalf("genuine directive update reset %d times", th.promptCacheEpoch-epoch)
	}
}
