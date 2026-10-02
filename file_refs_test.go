package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testFileRefPart() ContentPart {
	return ContentPart{Type: "file_ref", FileRef: &FileRef{
		File: true, Ref: "blobref://opaque%2Fid?x=a%26b&y=1", Filename: "report.pdf", MimeType: "application/pdf", Size: 12345,
	}}
}

func requireFileRef(t *testing.T, parts []ContentPart, want FileRef) {
	t.Helper()
	for _, part := range parts {
		if part.Type == "file_ref" && part.FileRef != nil && *part.FileRef == want {
			return
		}
	}
	t.Fatalf("file reference missing or modified: %#v, want %#v", parts, want)
}

func TestFileRefAPIEventPersistenceAndDeduplication(t *testing.T) {
	t.Chdir(t.TempDir())
	api, thinker := newTestAPI()
	thinker.config.path = configFile
	part := testFileRefPart()
	payload, _ := json.Marshal(map[string]any{"event_id": "file-message", "message": []ContentPart{part}})
	post := func(body []byte) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		api.postEvent(rec, httptest.NewRequest(http.MethodPost, "/event", bytes.NewReader(body)))
		return rec
	}
	if rec := post(payload); rec.Code != http.StatusOK {
		t.Fatalf("POST file-only event: %d %s", rec.Code, rec.Body.String())
	}
	events := thinker.drainEvents()
	if len(events) != 1 {
		t.Fatalf("events = %#v", events)
	}
	requireFileRef(t, events[0].Parts, *part.FileRef)
	loaded := NewConfig()
	if loaded.LoadError() != nil || len(loaded.MainEvents) != 1 {
		t.Fatalf("pending event not durable: %v %#v", loaded.LoadError(), loaded.MainEvents)
	}
	requireFileRef(t, loaded.MainEvents[0].Parts, *part.FileRef)
	if rec := post(payload); rec.Code != http.StatusOK || len(thinker.drainEvents()) != 0 {
		t.Fatal("duplicate file event was not idempotent")
	}
	part.FileRef.Size++
	payload, _ = json.Marshal(map[string]any{"event_id": "file-message", "message": []ContentPart{part}})
	if rec := post(payload); rec.Code != http.StatusConflict {
		t.Fatalf("changed metadata reused event ID: %d %s", rec.Code, rec.Body.String())
	}
}

func TestFileRefValidation(t *testing.T) {
	for _, raw := range []string{
		`[{"type":"file_ref","file_ref":{"ref":"blobref://x","filename":"x","mimeType":"text/plain"}}]`,
		`[{"type":"file_ref","file_ref":{"ref":"blobref://x","filename":"x","mimeType":"text/plain","size":null}}]`,
		`[{"type":"file_ref","file_ref":{"ref":"blobref://x","filename":"x","mimeType":"text/plain","size":1,"base64":"AQI="}}]`,
	} {
		if _, _, err := parseAPIEventMessage(json.RawMessage(raw)); err == nil {
			t.Fatalf("invalid metadata accepted: %s", raw)
		}
	}
	for _, tc := range []struct {
		name   string
		change func(*ContentPart)
	}{
		{"missing metadata", func(p *ContentPart) { p.FileRef = nil }},
		{"empty ref", func(p *ContentPart) { p.FileRef.Ref = "" }},
		{"ref whitespace", func(p *ContentPart) { p.FileRef.Ref += " " }},
		{"wrong scheme", func(p *ContentPart) { p.FileRef.Ref = "fileref://old-handle" }},
		{"data URI", func(p *ContentPart) { p.FileRef.Ref = "data:text/plain;base64,AQI=" }},
		{"missing MIME type", func(p *ContentPart) { p.FileRef.MimeType = "" }},
		{"negative size", func(p *ContentPart) { p.FileRef.Size = -1 }},
		{"inline bytes", func(p *ContentPart) { p.InputAudio = &InputAudio{Data: "AQI=", Format: "wav"} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			part := testFileRefPart()
			tc.change(&part)
			raw, _ := json.Marshal([]ContentPart{part})
			if _, _, err := parseAPIEventMessage(raw); err == nil {
				t.Fatal("invalid file_ref accepted")
			}
		})
	}
	part := testFileRefPart()
	part.FileRef.Size = 0
	raw, _ := json.Marshal([]ContentPart{part})
	if _, _, err := parseAPIEventMessage(raw); err != nil {
		t.Fatalf("empty file rejected: %v", err)
	}
}

func TestFileRefSurvivesSessionMixedAttachmentsAndCompaction(t *testing.T) {
	part := testFileRefPart()
	msg := Message{Role: "user", Content: "Process these inputs.", Parts: []ContentPart{
		{Type: "text", Text: "Process these inputs."}, part,
		{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/private.jpg?secret=1"}},
		{Type: "audio_url", AudioURL: &AudioURL{URL: "https://example.test/private.wav?secret=2"}},
	}}
	session := NewSession(t.TempDir(), "refs")
	if err := session.AppendMessage(msg, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	loaded, _ := session.LoadTail(10)
	if len(loaded) != 1 || transientAttachmentCount(loaded) != 0 || !strings.Contains(loaded[0].TextContent(), transientAttachmentHistoryNotice) {
		t.Fatalf("mixed history = %#v", loaded)
	}
	requireFileRef(t, loaded[0].Parts, *part.FileRef)
	if strings.Contains(loaded[0].TextContent(), "secret=") || len(msg.Parts) != 4 {
		t.Fatal("history retained access URL or mutated live inputs")
	}
	consumeTransientAttachments([]Message{msg})
	for i := 0; i < 2; i++ {
		if err := session.AppendMessage(Message{Role: "assistant", Content: "Continue."}, 2+i, TokenUsage{}); err != nil {
			t.Fatal(err)
		}
		session.ForceCompact(1, func(input string) string {
			if !strings.Contains(input, part.FileRef.Ref) {
				t.Fatal("compactor did not see file reference metadata")
			}
			return "Summary without any file references."
		})
		_, summaries := session.LoadTail(10)
		if len(summaries) != 1 || !strings.Contains(summaries[0], part.FileRef.Ref) {
			t.Fatalf("compaction lost reference: %v", summaries)
		}
	}
	cloned := cloneContentParts(msg.Parts)
	cloned[1].FileRef.Ref = "changed"
	if msg.Parts[1].FileRef.Ref != part.FileRef.Ref {
		t.Fatal("checkpoint clone aliases reference metadata")
	}
}

func TestFileRefSemanticCompactionKeepsStructuredReference(t *testing.T) {
	provider := &scriptedRetryProvider{name: "summarizer", response: ChatResponse{Text: "Summary omitting the file."}}
	thinker := retryTestThinker(provider)
	part := testFileRefPart()
	thinker.messages[1] = Message{Role: "user", Parts: []ContentPart{part}}
	for i := 0; i < contextPressureKeepRecent+3; i++ {
		thinker.messages = append(thinker.messages, Message{Role: "assistant", Content: "Later work."})
	}
	result, err := thinker.semanticCompactContext("test")
	if err != nil {
		t.Fatal(err)
	}
	requireFileRef(t, result.messages[1].Parts, *part.FileRef)
	if !strings.Contains(provider.messages[0][1].Content, part.FileRef.Ref) {
		t.Fatal("summarizer did not receive reference metadata")
	}
	// Further summaries preserve the structured handle independently of text.
	again := summaryWithFileRefs("Further summary.", result.messages)
	requireFileRef(t, again.Parts, *part.FileRef)
}

func TestFileRefProvidersRenderMetadataWithoutFetching(t *testing.T) {
	var fetches atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fetches.Add(1)
		_, _ = w.Write([]byte("PRIVATE FILE CONTENT"))
	}))
	defer server.Close()
	part := testFileRefPart()
	part.FileRef.Ref = "blobref://" + server.URL + "/file?token=a%2Fb&other=1"
	msg := Message{Role: "user", Parts: []ContentPart{part}}
	wantText := fileRefText(part.FileRef)
	for name, value := range map[string]any{
		"OpenAI compatible": toOpenAIMessages([]Message{msg}),
		"OpenAI Responses":  (&OpenAINativeProvider{}).buildInput([]Message{msg}),
		"Anthropic":         toAnthropicBlocks(msg.Parts),
		"Gemini":            toGeminiParts(msg.Parts),
	} {
		t.Run(name, func(t *testing.T) {
			wire, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			encodedText, _ := json.Marshal(wantText)
			if !bytes.Contains(wire, encodedText) || bytes.Contains(wire, []byte(`"type":"file_ref"`)) || bytes.Contains(wire, []byte("PRIVATE FILE CONTENT")) {
				t.Fatalf("unexpected provider wire: %s", wire)
			}
		})
	}
	if fetches.Load() != 0 {
		t.Fatalf("core fetched file bytes %d times", fetches.Load())
	}
	requireFileRef(t, msg.Parts, *part.FileRef)
}

func TestFileRefRetryAndFallbackRetainMetadata(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 10}
	fallback := &scriptedRetryProvider{name: "fallback", failures: 1, response: ChatResponse{Text: "done"}}
	thinker := retryTestThinker(primary)
	thinker.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "fallback": fallback}, order: []string{"primary", "fallback"}, default_: "primary"}
	part := testFileRefPart()
	thinker.messages[1] = Message{Role: "user", Parts: []ContentPart{part}}
	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, provider := range []*scriptedRetryProvider{primary, fallback} {
		for _, messages := range provider.messages {
			if len(messages[1].Parts) != 1 || messages[1].Parts[0].Type != "text" || messages[1].Parts[0].Text != fileRefText(part.FileRef) {
				t.Fatalf("retry/fallback lost metadata: %#v", messages)
			}
		}
	}
	requireFileRef(t, thinker.messages[1].Parts, *part.FileRef)
	// Image rejection strips the image while retaining the durable reference.
	primary = &scriptedRetryProvider{name: "primary", failures: 1, failureErr: errors.New("invalid_image_url"), response: ChatResponse{Text: "done"}}
	thinker = retryTestThinker(primary)
	thinker.messages[1] = Message{Role: "user", Parts: []ContentPart{part, {Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/broken.jpg"}}}}
	if _, err := thinker.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	requireFileRef(t, thinker.messages[1].Parts, *part.FileRef)
	seen, _ := json.Marshal(primary.messages[1])
	encodedMetadata, _ := json.Marshal(fileRefText(part.FileRef))
	if primary.calls != 2 || !bytes.Contains(seen, encodedMetadata) {
		t.Fatal("image quarantine removed file reference")
	}
}

func TestFileRefGatewayForwardingDoesNotRehydrate(t *testing.T) {
	part := testFileRefPart()
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			ID     any `json:"id"`
			Params struct {
				Arguments map[string]any `json:"arguments"`
			} `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		got = request.Params.Arguments
		_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": map[string]any{"content": []any{map[string]any{"type": "text", "text": "accepted"}}}})
	}))
	defer server.Close()
	blobs := NewBlobStore(1<<20, time.Hour)
	defer blobs.Close()
	mcp := &MCPHTTPServer{Name: "gateway", url: server.URL, client: server.Client()}
	defer mcp.Close()
	handler := mcpProxyHandler(mcp, "process_file", blobs, map[string]any{"properties": map[string]any{"file": map[string]any{"type": "string"}}})
	args := map[string]string{"file": part.FileRef.Ref}
	resp := handler(args)
	if resp.IsError || got["file"] != part.FileRef.Ref || args["file"] != part.FileRef.Ref || blobs.Count() != 0 {
		t.Fatalf("reference changed or bytes stored: args=%v gateway=%v response=%#v blobs=%d", args, got, resp, blobs.Count())
	}
}

func TestFileRefRealtimeDeliveryAndReconnect(t *testing.T) {
	t.Chdir(t.TempDir())
	thinker := newTestThinker()
	thinker.session = NewSession(t.TempDir(), "voice-files")
	socket := newFakeRealtimeSession()
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, nil, nil)
	rt.replaceSession(socket)
	part := testFileRefPart()
	rt.handleBusEvent(Event{Type: EventInbox, Parts: []ContentPart{part}})
	if len(socket.texts) != 1 || socket.texts[0].Content != fileRefText(part.FileRef) {
		t.Fatalf("file-only realtime event lost: %#v", socket.texts)
	}
	history := rt.boundedTranscript()
	if len(history) != 1 || history[0].Content != fileRefText(part.FileRef) {
		t.Fatalf("reconnect lost reference: %#v", history)
	}
	loaded, _ := thinker.session.LoadTail(10)
	if len(loaded) != 1 {
		t.Fatalf("realtime reference not persisted: %#v", loaded)
	}
	requireFileRef(t, loaded[0].Parts, *part.FileRef)
	if !reflect.DeepEqual(*loaded[0].Parts[1].FileRef, *part.FileRef) {
		t.Fatal("realtime metadata changed")
	}
}
