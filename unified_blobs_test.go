package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestUnifiedBlobLocalHandleAndNestedInput(t *testing.T) {
	store := NewBlobStore(1<<20, time.Hour)
	defer store.Close()
	output := store.RewriteBinaryToHandle(`{"_binary":true,"base64":"aGVsbG8=","mimeType":"text/plain","size":5,"filename":"hello.txt"}`)
	var handle FileRef
	if err := json.Unmarshal([]byte(output), &handle); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal([]ContentPart{{Type: "file_ref", FileRef: &handle}})
	if _, _, err := parseAPIEventMessage(raw); err != nil {
		t.Fatalf("tool output cannot be used as incoming file: %v", err)
	}
	for _, value := range []string{handle.Ref, output, `{"items":[` + output + `],"id":9007199254740993}`} {
		got := store.RehydrateFileRefs(map[string]string{"file": value})["file"]
		if !strings.Contains(got, `"_binary":true`) || !strings.Contains(got, `"filename":"hello.txt"`) || !strings.Contains(got, `"base64":"aGVsbG8="`) {
			t.Fatalf("local handle not usable as tool input: %s", got)
		}
		if strings.Contains(value, "9007199254740993") && !strings.Contains(got, "9007199254740993") {
			t.Fatal("nested input lost numeric precision")
		}
	}
	shared := `{"_file":true,"ref":"blobref://server-owned","mimeType":"text/plain","size":5}`
	if got := store.RehydrateFileRefs(map[string]string{"file": shared})["file"]; got != shared {
		t.Fatalf("shared handle was changed by core: %s", got)
	}
}

func TestUnifiedBlobThreadHeaderComesFromRuntimeContext(t *testing.T) {
	headers := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers <- r.Header.Get("X-Apteva-File-Thread")
		var rpc struct {
			ID int64 `json:"id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&rpc); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": rpc.ID, "result": map[string]any{"content": []any{map[string]any{"type": "text", "text": "ok"}}}})
	}))
	defer server.Close()
	ctx := withBlobCallerThread(context.Background(), "actual-thread")
	for _, tc := range []struct{ path, want string }{
		{"/mcp/1?file_agent=1&file_auth=signed", "actual-thread"},
		{"/mcp/1", ""},
	} {
		mcp := &MCPHTTPServer{Name: "gateway", url: server.URL + tc.path, client: server.Client()}
		_, err := mcp.CallToolContext(ctx, "use_file", map[string]string{"_apteva_caller_thread": "forged", "file": "blobref://shared"}, nil)
		mcp.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got := <-headers; got != tc.want {
			t.Fatalf("trusted header=%q want=%q", got, tc.want)
		}
	}
	if isAptevaBlobMCPURL("https://external.test/mcp/1?file_agent=1&file_auth=signed") {
		t.Fatal("runtime identity would leak to an external MCP")
	}
}
