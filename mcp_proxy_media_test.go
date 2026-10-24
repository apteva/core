package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Exercise the boundary the decoder-only and provider-only tests miss:
// MCP wire result -> transport decoder -> registered proxy handler.
func mediaResultMCP(t *testing.T, payload any) *MCPHTTPServer {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req jsonRPCRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid test request", 400)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: raw})
	}))
	t.Cleanup(server.Close)
	conn := &MCPHTTPServer{Name: "media", url: server.URL, client: server.Client()}
	t.Cleanup(conn.Close)
	return conn
}
func mediaResultPayload(content ...map[string]any) map[string]any {
	return map[string]any{"content": content}
}
func mediaImageBlock(mime string, data []byte) map[string]any {
	return map[string]any{"type": "image", "mimeType": mime, "data": base64.StdEncoding.EncodeToString(data)}
}

func TestMCPProxyPreservesDecodedContent(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47}
	jpeg := []byte{0xff, 0xd8, 0xff, 0xe0}
	text := map[string]any{"type": "text", "text": "Tool observation; no visual description."}
	mixed := mediaResultPayload(text, mediaImageBlock("image/png", png), mediaImageBlock("image/jpeg", jpeg), map[string]any{"type": "audio", "mimeType": "audio/wav", "data": "AQIDBA=="})
	mixed["isError"] = true
	mixed["structuredContent"] = map[string]any{"status": "partial"}
	cases := map[string]any{
		"png":                                mediaResultPayload(mediaImageBlock("image/png", png)),
		"jpeg":                               mediaResultPayload(text, mediaImageBlock("image/jpeg", jpeg)),
		"multiple_png":                       mediaResultPayload(mediaImageBlock("image/png", png), mediaImageBlock("image/png", []byte{1, 2, 3})),
		"wav":                                mediaResultPayload(map[string]any{"type": "audio", "mimeType": "audio/wav", "data": "AQIDBA=="}),
		"ogg":                                mediaResultPayload(map[string]any{"type": "audio", "mimeType": "audio/ogg", "data": "AQIDBA=="}),
		"mixed_error_and_structured_content": mixed,
		// Resource documents currently retain their text/JSON representation. A
		// preservation fix must not erase that evidence or imply native PDF support.
		"document_resource": mediaResultPayload(map[string]any{"type": "resource", "resource": map[string]any{"uri": "document:fixture", "mimeType": "application/pdf", "blob": "JVBERi0xLjcK"}}),
		"text_document":     mediaResultPayload(map[string]any{"type": "resource", "resource": map[string]any{"uri": "document:notes", "mimeType": "text/plain", "text": "Appointment is Tuesday at 10:00."}}),
		"resource_link":     mediaResultPayload(map[string]any{"type": "resource_link", "uri": "https://example.com/report.pdf"}),
	}
	for name, payload := range cases {
		for _, withBlobs := range []bool{false, true} {
			suffix := "/plain"
			if withBlobs {
				suffix = "/blob_store"
			}
			t.Run(name+suffix, func(t *testing.T) {
				conn := mediaResultMCP(t, payload)
				expected, err := conn.CallTool("read", nil)
				if err != nil {
					t.Fatal(err)
				}
				var blobs *BlobStore
				if withBlobs {
					blobs = NewBlobStore(1<<20, time.Hour)
					t.Cleanup(blobs.Close)
				}
				handler := mcpProxyHandlerContext(conn, "read", map[string]any{"type": "object"}, blobs)
				got := handler(context.Background(), nil)
				if !reflect.DeepEqual(got, expected) {
					t.Fatalf("proxy changed decoded response: image %d -> %d, parts %d -> %d, text_equal=%v error %v -> %v", len(expected.Image), len(got.Image), len(expected.Parts), len(got.Parts), expected.Text == got.Text, expected.IsError, got.IsError)
				}
			})
		}
	}
}

func TestMCPProxyPreservesNativeContentWithLegacyScreenshot(t *testing.T) {
	native := []byte{0x89, 0x50, 0x4e, 0x47}
	for _, duplicate := range []bool{false, true} {
		name := "different_images"
		if duplicate {
			name = "same_image"
		}
		t.Run(name, func(t *testing.T) {
			screenshot := []byte{0xff, 0xd8, 0xff, 0xe0}
			if duplicate {
				screenshot = native
			}
			metadata, _ := json.Marshal(map[string]any{"current_url": "https://example.com", "screenshot": map[string]any{"_binary": true, "mimeType": "image/jpeg", "base64": base64.StdEncoding.EncodeToString(screenshot)}, "screenshot_b64": "legacy duplicate"})
			conn := mediaResultMCP(t, mediaResultPayload(mediaImageBlock("image/png", native), map[string]any{"type": "audio", "mimeType": "audio/wav", "data": "AQIDBA=="}, map[string]any{"type": "text", "text": string(metadata)}))
			got := mcpProxyHandler(conn, "capture")(nil)
			if !bytes.Equal(got.Image, native) {
				t.Fatal("legacy screenshot overwrote original native image")
			}
			want := 2
			if duplicate {
				want = 1
			}
			if len(got.Parts) != want || got.Parts[0].InputAudio == nil {
				t.Fatalf("native media or screenshot lost: %+v", got.Parts)
			}
			if !duplicate && (got.Parts[1].ImageURL == nil || got.Parts[1].ImageURL.URL != "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString(screenshot)) {
				t.Fatal("additional screenshot missing or MIME type changed")
			}
			if strings.Contains(got.Text, "screenshot_b64") || !strings.Contains(got.Text, "attached as image") {
				t.Fatal("legacy text compaction changed")
			}
		})
	}
}

func TestMCPProxyPreservesMediaAlongsideBinaryHandles(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47}
	binary, _ := json.Marshal(map[string]any{"_binary": true, "mimeType": "application/pdf", "base64": "JVBERi0xLjcK", "filename": "report.pdf", "size": 9})
	conn := mediaResultMCP(t, mediaResultPayload(mediaImageBlock("image/png", png), mediaImageBlock("image/jpeg", []byte{0xff, 0xd8}), map[string]any{"type": "text", "text": string(binary)}))
	blobs := NewBlobStore(1<<20, time.Hour)
	defer blobs.Close()
	got := mcpProxyHandler(conn, "read", blobs)(nil)
	if !bytes.Equal(got.Image, png) || len(got.Parts) != 1 {
		t.Fatal("binary handle rewriting dropped native media")
	}
	if strings.Contains(got.Text, "JVBERi0xLjcK") || !strings.Contains(got.Text, "blobref://") || blobs.Count() != 1 {
		t.Fatalf("binary handle behavior changed: %s", got.Text)
	}
}

func TestMCPProxyInvalidNativeImageRemainsAnError(t *testing.T) {
	conn := mediaResultMCP(t, mediaResultPayload(map[string]any{"type": "image", "mimeType": "image/png", "data": "not-base64!"}))
	got := mcpProxyHandler(conn, "read")(nil)
	if !got.IsError || !strings.Contains(got.Text, "invalid MCP image") {
		t.Fatalf("invalid image hidden: %+v", got)
	}
}

func registerMediaFixtureTool(th *Thinker, conn MCPConn) *ToolDef {
	th.registry = NewToolRegistry("")
	def := &ToolDef{Name: "media_read", Description: "Read the supplied image and accompanying document. Return their original content for inspection.", MCP: true, MCPServer: "media", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}, Handler: mcpProxyHandler(conn, "read"), HandlerContext: mcpProxyHandlerContext(conn, "read")}
	th.registry.Register(def)
	return th.registry.Get(def.Name)
}
func dispatchMediaFixtureTool(t *testing.T, th *Thinker, call NativeToolCall) Event {
	t.Helper()
	executeTool(th, toolCall{Name: call.Name, NativeID: call.ID, Args: call.Args})
	timeout := time.NewTimer(3 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case event := <-th.sub.C:
			if event.ToolResult != nil {
				if event.ToolResult.IsError {
					t.Fatalf("tool dispatch failed: %s", event.ToolResult.Content)
				}
				return event
			}
		case <-timeout.C:
			t.Fatal("tool result was not published to the owning thread")
			return Event{}
		}
	}
}
func mediaEventMessages(event Event) []Message {
	messages := []Message{{Role: "user", ToolResults: []ToolResult{*event.ToolResult}}}
	// Core's inbox keeps native images in the paired ToolResult, and other
	// content parts in the following multimodal user message.
	if len(event.Parts) > 0 {
		messages = append(messages, Message{Role: "user", Parts: event.Parts})
	}
	return messages
}
func TestMCPNativeMediaSurvivesDispatchAndProviderProjection(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47}
	jpeg := []byte{0xff, 0xd8, 0xff, 0xe0}
	conn := mediaResultMCP(t, mediaResultPayload(map[string]any{"type": "text", "text": "Observation"}, mediaImageBlock("image/png", png), mediaImageBlock("image/jpeg", jpeg)))
	th := newTestThinker()
	defer th.Stop()
	registerMediaFixtureTool(th, conn)
	call := NativeToolCall{ID: "native-observation", Name: "media_read", Args: map[string]string{"_reason": "Inspect the source content"}}
	event := dispatchMediaFixtureTool(t, th, call)
	if !bytes.Equal(event.ToolResult.Image, png) || len(event.Parts) != 1 || event.Parts[0].ImageURL == nil {
		t.Fatal("dispatched native content was lost")
	}
	messages := append([]Message{{Role: "assistant", ToolCalls: []NativeToolCall{call}}}, mediaEventMessages(event)...)
	// Both the Responses/Codex and Chat Completions projections must carry
	// each original image exactly once, including the additional JPEG part.
	for name, request := range map[string]any{"responses": (&OpenAINativeProvider{}).buildInput(messages), "chat_completions": toOpenAIMessages(messages)} {
		raw, _ := json.Marshal(request)
		for _, data := range [][]byte{png, jpeg} {
			if strings.Count(string(raw), base64.StdEncoding.EncodeToString(data)) != 1 {
				t.Fatalf("%s lost or duplicated an image: %s", name, raw)
			}
		}
	}
}
