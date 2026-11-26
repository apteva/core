package core

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// fiveFrameResult is an MCP result like a video-frame tool's: a summary,
// then each frame preceded by a short caption.
func fiveFrameResult(t *testing.T, frames [][]byte, retention string) mcpCallResult {
	t.Helper()
	blocks := []map[string]any{{"type": "text", "text": `{"frames":5,"note":"Frames follow as images in frame order."}`}}
	for i, frame := range frames {
		image := map[string]any{"type": "image", "mimeType": "image/jpeg", "data": base64.StdEncoding.EncodeToString(frame)}
		if retention != "" {
			image["_meta"] = map[string]any{toolImageRetentionMetaKey: retention}
		}
		blocks = append(blocks, map[string]any{"type": "text", "text": fmt.Sprintf("Frame at %d s", i*5)}, image)
	}
	raw, _ := json.Marshal(map[string]any{"content": blocks})
	var result mcpCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func testFrames(t *testing.T, n int) [][]byte {
	t.Helper()
	frames := make([][]byte, n)
	for i := range frames {
		frames[i] = computerScreenshotJPEG(t, 20_000+i)
	}
	return frames
}

func TestDecodeMCPResultKeepsEveryImageLabelledInOrder(t *testing.T) {
	frames := testFrames(t, 5)
	resp, err := decodeMCPResult(fiveFrameResult(t, frames, ToolImageRetentionKeep))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 5 || len(resp.Parts) != 0 {
		t.Fatalf("images=%d parts=%d, want 5 images on the result and no loose parts", len(resp.Images), len(resp.Parts))
	}
	for i, img := range resp.Images {
		if !bytes.Equal(img.Data, frames[i]) || img.MIMEType != "image/jpeg" || img.Retention != ToolImageRetentionKeep {
			t.Fatalf("image %d = mime %q retention %q", i, img.MIMEType, img.Retention)
		}
		if want := fmt.Sprintf("Frame at %d s", i*5); img.Label != want {
			t.Fatalf("image %d label = %q, want %q", i, img.Label, want)
		}
	}
	// The long JSON summary is a result body, never an image caption.
	if !strings.Contains(resp.Text, `"frames":5`) {
		t.Fatalf("summary text lost: %q", resp.Text)
	}
}

func TestDecodeMCPResultReadsLabelAndRetentionFromMeta(t *testing.T) {
	raw := `{"content":[{"type":"image","mimeType":"image/png","data":"iVBORw==","_meta":{"io.apteva/label":"Render, front view","io.apteva/retention":"LATEST"}},{"type":"image","mimeType":"image/png","data":"iVBORw==","_meta":{"io.apteva/retention":"forever"}}]}`
	var result mcpCallResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatal(err)
	}
	resp, err := decodeMCPResult(result)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 2 || resp.Images[0].Label != "Render, front view" || resp.Images[0].Retention != ToolImageRetentionLatest {
		t.Fatalf("first image = %+v", resp.Images)
	}
	if resp.Images[1].Retention != "" || resp.Images[1].Label != "" {
		t.Fatalf("unknown hint must be ignored: %+v", resp.Images[1])
	}
	labelled := labelToolImages(resp.Images, "design_render")
	if labelled[1].Label != "Image 2 of 2 from design_render" || resp.Images[1].Label != "" {
		t.Fatalf("labelToolImages = %q (input mutated: %q)", labelled[1].Label, resp.Images[1].Label)
	}
}

// appendToolTurn adds one tool call and its archived result to the thread.
func appendToolTurn(th *Thinker, id, name, content string, images []ToolImage) {
	th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: name}}})
	th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: content, Images: labelToolImages(images, name)}}}))
}

func requestImages(messages []Message, callID string) []ToolImage {
	for _, m := range messages {
		for _, r := range m.ToolResults {
			if r.CallID == callID {
				return r.Images
			}
		}
	}
	return nil
}

func requestContent(messages []Message, callID string) string {
	for _, m := range messages {
		for _, r := range m.ToolResults {
			if r.CallID == callID {
				return r.Content
			}
		}
	}
	return ""
}

func TestToolImagesSurviveLaterRequestsAndNewerScreenshots(t *testing.T) {
	frames := testFrames(t, 5)
	resp, err := decodeMCPResult(fiveFrameResult(t, frames, ToolImageRetentionKeep))
	if err != nil {
		t.Fatal(err)
	}
	th := &Thinker{threadID: "main", session: NewSession(t.TempDir(), "main")}
	th.messages = []Message{{Role: "system", Content: "Describe the video."}, {Role: "user", Content: "What happens in the video?"}}
	appendToolTurn(th, "frames", "computer_use", resp.Text, resp.Images)

	first := th.prepareToolResultRequest(th.messages)
	if got := requestImages(first, "frames"); len(got) != 5 || got[4].Label != "Frame at 20 s" {
		t.Fatalf("first request images = %d", len(got))
	}
	th.markToolResultsConsumed(first)

	// The agent clicks a cookie banner: two screenshots arrive. Unhinted
	// computer_* screenshots keep their "latest" behaviour.
	appendToolTurn(th, "screen-1", "computer_use", "Clicked Accept.", []ToolImage{{Data: computerScreenshotJPEG(t, 30_000)}})
	th.markToolResultsConsumed(th.prepareToolResultRequest(th.messages))
	latestScreen := computerScreenshotJPEG(t, 31_000)
	appendToolTurn(th, "screen-2", "computer_use", "Page settled.", []ToolImage{{Data: latestScreen}})

	next := th.prepareToolResultRequest(th.messages)
	got := requestImages(next, "frames")
	if len(got) != 5 {
		t.Fatalf("frames after later requests = %d, want all 5", len(got))
	}
	for i, img := range got {
		if !bytes.Equal(img.Data, frames[i]) || img.Label != fmt.Sprintf("Frame at %d s", i*5) {
			t.Fatalf("frame %d changed: label %q", i, img.Label)
		}
	}
	if len(requestImages(next, "screen-1")) != 0 || len(requestImages(next, "screen-2")) != 0 {
		t.Fatal("screenshots must leave their tool results for the request tail")
	}
	tail := next[len(next)-1]
	wantScreen := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(latestScreen)
	if !tail.RequestContext || len(tail.Parts) != 2 || tail.Parts[1].ImageURL == nil || tail.Parts[1].ImageURL.URL != wantScreen || !strings.Contains(tail.Parts[0].Text, "screen-2") {
		t.Fatalf("request tail = %+v, want only the newest screenshot", tail.Parts)
	}
	for _, m := range next {
		if strings.Contains(m.Content, transientAttachmentHistoryNotice) {
			t.Fatal("tool images must not be treated as one-request attachments")
		}
	}
}

func TestToolImageBudgetRemovesOldestWithSpecificNote(t *testing.T) {
	th := &Thinker{threadID: "main", session: NewSession(t.TempDir(), "main")}
	th.messages = []Message{{Role: "system", Content: "Compare renders."}, {Role: "user", Content: "Render each view."}}
	frames := testFrames(t, maxKeptToolImages+3)
	appendToolTurn(th, "old", "media_frames", "Older frames.", []ToolImage{
		{Data: frames[0], Label: "Frame at 0 s"}, {Data: frames[1], Label: "Frame at 5 s"}, {Data: frames[2], Label: "Frame at 10 s"},
	})
	th.markToolResultsConsumed(th.prepareToolResultRequest(th.messages))

	var recent []ToolImage
	for i := 3; i < len(frames); i++ {
		recent = append(recent, ToolImage{Data: frames[i], Label: fmt.Sprintf("View %d", i)})
	}
	appendToolTurn(th, "new", "media_frames", "Newer frames.", recent)

	// Unseen images are always sent once, even past the budget.
	first := th.prepareToolResultRequest(th.messages)
	if got := len(requestImages(first, "new")); got != maxKeptToolImages {
		t.Fatalf("new result images = %d, want %d", got, maxKeptToolImages)
	}
	if got := len(requestImages(first, "old")); got != 0 {
		t.Fatalf("old result kept %d images past the budget", got)
	}
	note := requestContent(first, "old")
	for _, want := range []string{"3 images returned by media_frames were removed to save context", "Frame at 0 s; Frame at 5 s; Frame at 10 s", "call media_frames again"} {
		if !strings.Contains(note, want) {
			t.Fatalf("removal note %q lacks %q", note, want)
		}
	}
	if live := th.messages[3].ToolResults[0]; len(live.Images) != 3 || strings.Contains(live.Content, "removed") {
		t.Fatal("the budget must only shape the request, not the live history")
	}
}

func TestToolImagesFromSameLatestToolReplaceEachOther(t *testing.T) {
	th := &Thinker{threadID: "main", session: NewSession(t.TempDir(), "main")}
	th.messages = []Message{{Role: "system", Content: "Watch the preview."}, {Role: "user", Content: "Iterate."}}
	appendToolTurn(th, "p1", "design_preview", "Preview 1.", []ToolImage{{Data: computerScreenshotJPEG(t, 10_000), Retention: ToolImageRetentionLatest}})
	second := computerScreenshotJPEG(t, 11_000)
	appendToolTurn(th, "p2", "design_preview", "Preview 2.", []ToolImage{{Data: second, Retention: ToolImageRetentionLatest, Label: "Front view"}})
	appendToolTurn(th, "photo", "camera_capture", "Captured.", []ToolImage{{Data: computerScreenshotJPEG(t, 12_000)}})

	next := th.prepareToolResultRequest(th.messages)
	if len(requestImages(next, "p1")) != 0 || len(requestImages(next, "p2")) != 0 || len(requestImages(next, "photo")) != 1 {
		t.Fatal("latest images must move to the tail; kept images stay on their result")
	}
	tail := next[len(next)-1]
	if !strings.Contains(tail.Parts[0].Text, "CURRENT IMAGE FROM design_preview") || tail.Parts[1].Text != "[Front view]" || tail.Parts[2].ImageURL == nil || tail.Parts[2].ImageURL.URL != toolImageDataURL(ToolImage{Data: second}) {
		t.Fatalf("tail = %+v", tail.Parts)
	}
	// Saved sessions drop replaced "latest" images but keep the others.
	saved := messageForSession(th.messages, th.messages[len(th.messages)-5])
	if len(saved.ToolResults[0].Images) != 0 {
		t.Fatal("latest image persisted")
	}
	if kept := messageForSession(th.messages, th.messages[len(th.messages)-1]); len(kept.ToolResults[0].Images) != 1 {
		t.Fatal("kept image dropped from the saved session")
	}
}

func TestProvidersSendLabelledToolImagesInsideTheToolResult(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	jpeg := []byte{0xff, 0xd8, 0xff, 0xe0}
	messages := []Message{
		{Role: "user", Content: "Look."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call_1", Name: "frames", Args: map[string]string{}}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call_1", ToolName: "frames", Content: "Two frames.", Images: []ToolImage{
			{MIMEType: "image/png", Data: png, Label: "Frame at 0 s"},
			{Data: jpeg, Label: "Frame at 5 s"},
		}}}},
	}
	order := func(t *testing.T, name, wire string, want ...string) {
		t.Helper()
		at := 0
		for _, w := range want {
			i := strings.Index(wire[at:], w)
			if i < 0 {
				t.Fatalf("%s: %q missing or out of order in %s", name, w, wire)
			}
			at += i + len(w)
		}
	}
	pngURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(png)
	jpegURL := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(jpeg)
	order(t, "responses", string(mustJSON(t, (&OpenAINativeProvider{}).buildInput(messages))), "Two frames.", "[Frame at 0 s]", pngURL, "[Frame at 5 s]", jpegURL)
	chat := string(mustJSON(t, toOpenAIMessages(messages)))
	if !strings.Contains(chat, `"role":"tool"`) {
		t.Fatalf("chat completions must keep images in the tool message: %s", chat)
	}
	order(t, "chat_completions", chat, "Two frames.", "[Frame at 0 s]", pngURL, "[Frame at 5 s]", jpegURL)

	var capture anthropicCapture
	server := capture.server(t, `{"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":1}}}`, `{"type":"message_stop"}`)
	p := &AnthropicProvider{apiKey: "test", url: server.URL}
	if _, err := p.Chat(context.Background(), messages, "claude-opus-5-5", nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	if len(capture.bodies) != 1 {
		t.Fatal("anthropic request not captured")
	}
	wire := string(mustJSON(t, capture.bodies[0]))
	// Images sit inside the tool_result block, after its text.
	block := wire[strings.Index(wire, `{"content":[{"text":"Two frames."`):]
	if !strings.Contains(block[:strings.Index(block, `"type":"tool_result"`)], `"media_type":"image/jpeg"`) {
		t.Fatalf("anthropic images are not inside the tool_result: %s", wire)
	}
	order(t, "anthropic", wire, "Two frames.", "[Frame at 0 s]", `"media_type":"image/png"`, "[Frame at 5 s]", `"media_type":"image/jpeg"`)
}

func TestToolResultReadsLegacySingleImage(t *testing.T) {
	var result ToolResult
	if err := json.Unmarshal([]byte(`{"call_id":"c1","content":"shot","image":"AQID"}`), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 || !bytes.Equal(result.Images[0].Data, []byte{1, 2, 3}) {
		t.Fatalf("legacy image = %+v", result.Images)
	}
	raw, _ := json.Marshal(result)
	if strings.Contains(string(raw), `"image":`) || !strings.Contains(string(raw), `"images":`) {
		t.Fatalf("results are written with the images list: %s", raw)
	}
}

func TestToolResultHashKeepsV1ForSinglePlainImage(t *testing.T) {
	// The v1 hash as computed before images became a list.
	legacy := func(content string, image []byte, isError bool) string {
		h := sha256.New()
		h.Write([]byte("apteva-tool-result-v1\x00"))
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(content)))
		h.Write(length[:])
		h.Write([]byte(content))
		binary.BigEndian.PutUint64(length[:], uint64(len(image)))
		h.Write(length[:])
		h.Write(image)
		if isError {
			h.Write([]byte{1})
		} else {
			h.Write([]byte{0})
		}
		return hex.EncodeToString(h.Sum(nil))
	}
	if got, want := toolResultPayloadSHA256("text", []ToolImage{{Data: []byte("px")}}, false), legacy("text", []byte("px"), false); got != want {
		t.Fatal("single plain image hash changed; archived results would stop verifying")
	}
	if got, want := toolResultPayloadSHA256("text", nil, true), legacy("text", nil, true); got != want {
		t.Fatal("text-only hash changed")
	}
	labelled := toolResultPayloadSHA256("text", []ToolImage{{Data: []byte("px"), Label: "Frame at 0 s"}}, false)
	if labelled == legacy("text", []byte("px"), false) || labelled == toolResultPayloadSHA256("text", []ToolImage{{Data: []byte("px"), Label: "Frame at 5 s"}}, false) {
		t.Fatal("labels must be part of the v2 hash")
	}
}

func TestArchivedToolResultWithImagesRoundTrips(t *testing.T) {
	archive := NewToolResultArchive(t.TempDir(), "main")
	stored, err := archive.Archive(ToolResult{CallID: "c1", ToolName: "frames", Content: "Two frames.", Images: []ToolImage{
		{MIMEType: "image/jpeg", Data: []byte{0xff, 0xd8}, Label: "Frame at 0 s", Retention: ToolImageRetentionKeep},
		{MIMEType: "image/jpeg", Data: []byte{0xff, 0xd9}, Label: "Frame at 5 s"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	object, err := archive.Read(stored.ArchiveRef)
	if err != nil {
		t.Fatal(err)
	}
	if len(object.Images) != 2 || object.Images[1].Label != "Frame at 5 s" || object.Images[0].Retention != ToolImageRetentionKeep {
		t.Fatalf("archived images = %+v", object.Images)
	}
	projected := projectArchivedToolResult(ToolResult{CallID: "c1", ToolName: "frames", Content: "Two frames.", Images: object.Images}, 4000)
	if len(projected.Images) != 0 || !strings.Contains(projected.Content, "2 images returned by frames were removed with older history: Frame at 0 s; Frame at 5 s") {
		t.Fatalf("projected = %q", projected.Content)
	}
}
