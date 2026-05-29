package core

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// ---------- BlobStore unit tests ----------

func TestBlobStore_PutGet(t *testing.T) {
	bs := NewBlobStore(1024*1024, time.Hour)
	defer bs.Close()

	data := []byte("hello binary world")
	ref := bs.Put("audio/mpeg", data)

	if !strings.HasPrefix(ref, blobRefPrefix) {
		t.Fatalf("ref missing prefix: %q", ref)
	}
	got, mime, ok := bs.Get(ref)
	if !ok {
		t.Fatal("blob not found by full ref")
	}
	if string(got) != string(data) || mime != "audio/mpeg" {
		t.Fatalf("mismatch: got=%q mime=%q", got, mime)
	}
	// Bare id lookup should also work.
	id := strings.TrimPrefix(ref, blobRefPrefix)
	if _, _, ok := bs.Get(id); !ok {
		t.Fatal("blob not found by bare id")
	}
}

func TestBlobStore_TTLEviction(t *testing.T) {
	bs := NewBlobStore(1024*1024, 50*time.Millisecond)
	defer bs.Close()

	ref := bs.Put("text/plain", []byte("x"))
	if bs.Count() != 1 {
		t.Fatalf("expected 1 blob, got %d", bs.Count())
	}
	// Call the internal sweep after the TTL passes — don't wait for the
	// 30s janitor. We're testing the eviction logic, not the ticker.
	time.Sleep(60 * time.Millisecond)
	bs.evictExpired()
	if bs.Count() != 0 {
		t.Fatalf("expected expired blob evicted, still have %d", bs.Count())
	}
	if _, _, ok := bs.Get(ref); ok {
		t.Fatal("expired blob still retrievable")
	}
}

func TestBlobStore_SizeCapEvictsOldest(t *testing.T) {
	// Cap = 100 bytes. Two 40-byte blobs fit; a third 40-byte forces the
	// first to be evicted (oldest first).
	bs := NewBlobStore(100, time.Hour)
	defer bs.Close()

	ref1 := bs.Put("x", make([]byte, 40))
	time.Sleep(2 * time.Millisecond) // ensure distinct created timestamps
	ref2 := bs.Put("x", make([]byte, 40))
	time.Sleep(2 * time.Millisecond)
	ref3 := bs.Put("x", make([]byte, 40))

	if _, _, ok := bs.Get(ref1); ok {
		t.Error("expected ref1 (oldest) evicted")
	}
	if _, _, ok := bs.Get(ref2); !ok {
		t.Error("ref2 should still be present")
	}
	if _, _, ok := bs.Get(ref3); !ok {
		t.Error("ref3 should be present (just added)")
	}
}

// ---------- RewriteBinaryToHandle (outbound, MCP result → LLM) ----------

func TestRewriteBinaryToHandle_PassthroughForText(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()
	const plain = `{"ok": true, "message": "done"}`
	got := bs.RewriteBinaryToHandle(plain)
	if got != plain {
		t.Fatalf("expected text unchanged, got %q", got)
	}
	if bs.Count() != 0 {
		t.Fatal("no blob should have been stored for plain text")
	}
}

func TestRewriteBinaryToHandle_ReplacesEnvelope(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	raw := []byte("pretend this is audio")
	envelope := map[string]any{
		"_binary":  true,
		"base64":   base64.StdEncoding.EncodeToString(raw),
		"mimeType": "audio/mpeg",
		"size":     len(raw),
	}
	envJSON, _ := json.Marshal(envelope)

	got := bs.RewriteBinaryToHandle(string(envJSON))

	var handle map[string]any
	if err := json.Unmarshal([]byte(got), &handle); err != nil {
		t.Fatalf("rewritten text is not JSON: %v\n%s", err, got)
	}
	if handle["_file"] != true {
		t.Errorf("expected _file: true, got %v", handle["_file"])
	}
	if handle["_binary"] != nil {
		t.Errorf("_binary leaked into handle: %v", handle["_binary"])
	}
	if handle["mimeType"] != "audio/mpeg" {
		t.Errorf("mimeType mismatch: %v", handle["mimeType"])
	}
	ref, _ := handle["ref"].(string)
	if !strings.HasPrefix(ref, blobRefPrefix) {
		t.Fatalf("ref missing prefix: %q", ref)
	}
	// Handle should be small — it's the whole point.
	if len(got) > 256 {
		t.Errorf("handle unexpectedly large: %d bytes", len(got))
	}
	// Bytes should be retrievable via the ref.
	stored, mime, ok := bs.Get(ref)
	if !ok || string(stored) != string(raw) || mime != "audio/mpeg" {
		t.Errorf("bytes not stored correctly: ok=%v mime=%q len=%d", ok, mime, len(stored))
	}
}

// ---------- RehydrateFileRefs (inbound, LLM args → MCP dispatch) ----------

func TestRehydrateFileRefs_ScalarRef(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	raw := []byte("audio bytes")
	ref := bs.Put("audio/mpeg", raw)

	// LLM passed the scalar ref directly.
	in := map[string]string{
		"audio":    ref,
		"language": "en",
	}
	out := bs.RehydrateFileRefs(in)

	if out["language"] != "en" {
		t.Errorf("unrelated arg mutated: %q", out["language"])
	}
	var env map[string]any
	if err := json.Unmarshal([]byte(out["audio"]), &env); err != nil {
		t.Fatalf("rehydrated audio is not JSON: %v\n%s", err, out["audio"])
	}
	if env["_binary"] != true {
		t.Errorf("expected _binary envelope, got %v", env)
	}
	decoded, _ := base64.StdEncoding.DecodeString(env["base64"].(string))
	if string(decoded) != string(raw) {
		t.Errorf("bytes not round-tripped correctly")
	}
}

func TestRehydrateFileRefs_ObjectForm(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	raw := []byte("abc")
	ref := bs.Put("application/octet-stream", raw)

	// LLM used the explicit object form.
	in := map[string]string{
		"audio": `{"_file_ref": "` + ref + `"}`,
	}
	out := bs.RehydrateFileRefs(in)
	var env map[string]any
	if err := json.Unmarshal([]byte(out["audio"]), &env); err != nil {
		t.Fatalf("object-form rehydration failed: %v\n%s", err, out["audio"])
	}
	if env["_binary"] != true {
		t.Errorf("envelope missing _binary: %v", env)
	}
}

func TestRehydrateFileRefs_UnknownRefLeftUntouched(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	// No Put — this ref doesn't exist.
	in := map[string]string{"audio": "blobref://nonexistent"}
	out := bs.RehydrateFileRefs(in)

	if out["audio"] != "blobref://nonexistent" {
		t.Errorf("unknown ref should pass through, got %q", out["audio"])
	}
}

// ---------- End-to-end wiring through mcpProxyHandler ----------

// handlerMCP is a minimal in-process MCPConn used to exercise the
// mcpProxyHandler interception path without a real subprocess. It
// captures the last args received so tests can assert what the
// downstream tool actually saw after rehydration.
type handlerMCP struct {
	name     string
	result   string
	lastArgs map[string]string
}

func (m *handlerMCP) GetName() string                  { return m.name }
func (m *handlerMCP) ListTools() ([]mcpToolDef, error) { return nil, nil }
func (m *handlerMCP) CallTool(_ string, a map[string]string) (ToolResponse, error) {
	m.lastArgs = a
	return ToolResponse{Text: m.result}, nil
}
func (m *handlerMCP) Close() {}

func TestMCPProxyHandler_WrapsBinaryResult(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	raw := []byte{0xff, 0xd8, 0xff, 0xe0} // jpeg magic
	envBytes, _ := json.Marshal(map[string]any{
		"_binary":  true,
		"base64":   base64.StdEncoding.EncodeToString(raw),
		"mimeType": "image/jpeg",
		"size":     len(raw),
	})
	mcp := &handlerMCP{name: "fake", result: string(envBytes)}

	h := mcpProxyHandler(mcp, "download", bs)
	resp := h(map[string]string{"id": "x"})

	if strings.Contains(resp.Text, `"_binary"`) || strings.Contains(resp.Text, "base64") {
		t.Fatalf("binary payload leaked into tool result text:\n%s", resp.Text)
	}
	if !strings.Contains(resp.Text, `"_file"`) || !strings.Contains(resp.Text, "blobref://") {
		t.Fatalf("expected handle in result, got:\n%s", resp.Text)
	}
	if bs.Count() != 1 {
		t.Fatalf("expected 1 blob stored, got %d", bs.Count())
	}
}

func TestMCPProxyHandler_RehydratesBeforeDispatch(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	raw := []byte("secret audio")
	ref := bs.Put("audio/wav", raw)

	mcp := &handlerMCP{name: "fake", result: "transcript: hi"}
	h := mcpProxyHandler(mcp, "transcribe", bs)
	_ = h(map[string]string{"audio": ref, "language": "en"})

	if mcp.lastArgs == nil {
		t.Fatal("handler never dispatched")
	}
	audioArg := mcp.lastArgs["audio"]
	if audioArg == ref {
		t.Fatalf("args were not rehydrated (still scalar ref):\n%s", audioArg)
	}
	var env map[string]any
	if err := json.Unmarshal([]byte(audioArg), &env); err != nil {
		t.Fatalf("downstream arg not JSON: %v\n%s", err, audioArg)
	}
	if env["_binary"] != true {
		t.Fatalf("expected _binary envelope dispatched, got %v", env)
	}
	decoded, _ := base64.StdEncoding.DecodeString(env["base64"].(string))
	if string(decoded) != string(raw) {
		t.Fatalf("bytes corrupted through rehydration")
	}
	// Unrelated args untouched.
	if mcp.lastArgs["language"] != "en" {
		t.Errorf("unrelated arg mutated: %q", mcp.lastArgs["language"])
	}
}

// TestBlobStore_DoesNotInterfereWithPlainToolText proves that blob
// rewriting only treats structured binary envelopes as blob material.
func TestBlobStore_DoesNotInterfereWithPlainToolText(t *testing.T) {
	bs := NewBlobStore(1<<20, time.Hour)
	defer bs.Close()

	toolText := "clicked at (100, 100) - element: <button>Submit</button>"
	if got := bs.RewriteBinaryToHandle(toolText); got != toolText {
		t.Fatalf("plain tool text mutated by blob rewrite:\nin:  %q\nout: %q", toolText, got)
	}
	if bs.Count() != 0 {
		t.Fatal("plain tool text must not create blob store entries")
	}

	toolArgs := map[string]string{
		"action":     "left_click",
		"coordinate": "[100, 100]",
		"text":       "Submit",
	}
	out := bs.RehydrateFileRefs(toolArgs)
	for k, v := range toolArgs {
		if out[k] != v {
			t.Errorf("plain tool arg %q was mutated: %q -> %q", k, v, out[k])
		}
	}

	// Pathological case: plain text that happens to contain
	// the substring "_binary" (e.g. a page screenshot's extracted text
	// mentions the word). Must NOT be treated as an envelope — the
	// JSON unmarshal will fail and the original text is returned.
	edgeText := `clicked on link "Download _binary blob from S3" at (50, 50)`
	if got := bs.RewriteBinaryToHandle(edgeText); got != edgeText {
		t.Fatalf("text containing '_binary' substring was mutated:\n%q", got)
	}
	if bs.Count() != 0 {
		t.Fatal("edge-case text must not create a blob")
	}
}

func TestMCPProxyHandler_NilBlobsPassthrough(t *testing.T) {
	// Legacy path — blobs nil means no interception.
	mcp := &handlerMCP{name: "fake", result: `{"_binary": true, "base64": "AQI=", "mimeType": "x", "size": 2}`}
	h := mcpProxyHandler(mcp, "t", nil)
	resp := h(map[string]string{"a": "blobref://x"})
	if !strings.Contains(resp.Text, `"_binary"`) {
		t.Errorf("nil blobs should not wrap result")
	}
	if mcp.lastArgs["a"] != "blobref://x" {
		t.Errorf("nil blobs should not rehydrate args, got %q", mcp.lastArgs["a"])
	}
}
