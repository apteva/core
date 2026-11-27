package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// liveFrame is one synthetic video frame: a solid colour with a white square
// in one corner. Both are random per run, so the model cannot guess them.
type liveFrame struct {
	color, corner string
	data          []byte
}

var liveFramePalette = []struct {
	name string
	rgba color.RGBA
}{
	{"red", color.RGBA{220, 30, 30, 255}},
	{"green", color.RGBA{30, 170, 60, 255}},
	{"blue", color.RGBA{30, 70, 220, 255}},
	{"yellow", color.RGBA{240, 220, 40, 255}},
	{"purple", color.RGBA{130, 50, 170, 255}},
	{"orange", color.RGBA{245, 140, 20, 255}},
}

var liveFrameCorners = []string{"top-left", "top-right", "bottom-left", "bottom-right"}

func liveVideoFrames(t *testing.T, rng *rand.Rand, n int) []liveFrame {
	t.Helper()
	order := rng.Perm(len(liveFramePalette))
	frames := make([]liveFrame, n)
	for i := range frames {
		paint := liveFramePalette[order[i%len(order)]]
		corner := liveFrameCorners[rng.Intn(len(liveFrameCorners))]
		const w, h, side, margin = 512, 288, 72, 20
		img := image.NewRGBA(image.Rect(0, 0, w, h))
		x0, y0 := margin, margin
		if strings.HasSuffix(corner, "right") {
			x0 = w - margin - side
		}
		if strings.HasPrefix(corner, "bottom") {
			y0 = h - margin - side
		}
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := paint.rgba
				if x >= x0 && x < x0+side && y >= y0 && y < y0+side {
					c = color.RGBA{255, 255, 255, 255}
				}
				img.SetRGBA(x, y, c)
			}
		}
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 85}); err != nil {
			t.Fatal(err)
		}
		frames[i] = liveFrame{color: paint.name, corner: corner, data: buf.Bytes()}
	}
	return frames
}

// liveFramesPayload is shaped like the Computer app's video_frames result: a
// JSON summary, then one image per frame with a label and the keep hint.
func liveFramesPayload(video string, frames []liveFrame) map[string]any {
	content := []map[string]any{{"type": "text", "text": fmt.Sprintf(`{"video":%q,"frames":%d,"note":"Frames follow as images in frame order."}`, video, len(frames))}}
	for i, f := range frames {
		content = append(content, map[string]any{
			"type": "image", "mimeType": "image/jpeg", "data": base64.StdEncoding.EncodeToString(f.data),
			"_meta": map[string]any{
				toolImageLabelMetaKey:     fmt.Sprintf("Video %s, frame at %d s", video, i*5),
				toolImageRetentionMetaKey: ToolImageRetentionKeep,
			},
		})
	}
	return map[string]any{"content": content}
}

func liveScreenshotJPEG(t *testing.T, shade uint8) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 512, 320))
	for y := 0; y < 320; y++ {
		for x := 0; x < 512; x++ {
			c := color.RGBA{shade, shade, shade, 255}
			if y < 48 {
				c = color.RGBA{60, 60, 70, 255}
			}
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// liveJSONReply reads the JSON object a model was asked to reply with.
// Values are lower-cased, with spaces and underscores turned into hyphens.
func liveJSONReply(t *testing.T, label, text string) map[string]string {
	t.Helper()
	start, end := strings.Index(text, "{"), strings.LastIndex(text, "}")
	if start < 0 || end < start {
		t.Fatalf("%s: no JSON object in reply %q", label, text)
	}
	var raw map[string]any
	if err := json.Unmarshal([]byte(text[start:end+1]), &raw); err != nil {
		t.Fatalf("%s: invalid JSON reply %q: %v", label, text, err)
	}
	out := make(map[string]string, len(raw))
	for k, v := range raw {
		s := strings.ToLower(strings.TrimSpace(fmt.Sprint(v)))
		out[k] = strings.NewReplacer(" ", "-", "_", "-").Replace(s)
	}
	return out
}

// captureLLMInputs records the input items of every provider request, so a
// test can tell a prefix Core changed from one the provider did not cache.
func captureLLMInputs(t *testing.T) func() [][]json.RawMessage {
	t.Helper()
	original := llmHTTPClient
	client := *original
	base := original.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	var mu sync.Mutex
	var inputs [][]json.RawMessage
	client.Transport = auditRoundTripper(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodPost || req.Body == nil {
			return base.RoundTrip(req)
		}
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		_ = req.Body.Close()
		var root struct {
			Input []json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &root) == nil && len(root.Input) > 0 {
			mu.Lock()
			inputs = append(inputs, root.Input)
			mu.Unlock()
		}
		clone := req.Clone(req.Context())
		clone.Body = io.NopCloser(bytes.NewReader(body))
		clone.ContentLength = int64(len(body))
		clone.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
		return base.RoundTrip(clone)
	})
	llmHTTPClient = &client
	t.Cleanup(func() { llmHTTPClient = original })
	return func() [][]json.RawMessage {
		mu.Lock()
		defer mu.Unlock()
		return append([][]json.RawMessage(nil), inputs...)
	}
}

// firstChangedInput returns the first input item of prev that next does not
// repeat byte for byte, or len(prev) when next extends prev unchanged.
func firstChangedInput(prev, next []json.RawMessage) int {
	for i := range prev {
		if i >= len(next) || !bytes.Equal(prev[i], next[i]) {
			return i
		}
	}
	return len(prev)
}

func liveInputKind(item json.RawMessage) string {
	var head struct{ Type, Role string }
	_ = json.Unmarshal(item, &head)
	return strings.TrimSpace(head.Type + " " + head.Role)
}

func liveRequestImageCount(messages []Message) int {
	n := 0
	for _, m := range messages {
		for _, r := range m.ToolResults {
			n += len(r.Images)
		}
		for _, part := range m.Parts {
			if part.ImageURL != nil {
				n++
			}
		}
	}
	return n
}

// TestIntegration_CodexToolImageRetention checks kept tool images with a real
// model, through Core's own request projection (prepareToolResultRequest and
// the cache epoch scope, as thinkWithProvider uses them):
//   - labelled video frames stay usable across later turns, clicks and
//     screenshots, without calling the tool again;
//   - past the high mark the oldest frames retire in one batch; the model reads
//     the removal note and fetches the frames again instead of guessing;
//   - between batches Core resends the previous request unchanged (checked on
//     the request bytes), and the provider serves most requests from cache.
//
// Paid: set RUN_CODEX_TOOL_IMAGE_RETENTION=1 and run without -short. It uses
// the local Codex credential, like the other Sol live tests.
func TestIntegration_CodexToolImageRetention(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_TOOL_IMAGE_RETENTION") != "1" {
		t.Skip("set RUN_CODEX_TOOL_IMAGE_RETENTION=1 without -short for paid image retention verification")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live Codex test requested without a valid local credential")
	}
	p := toolReasonLiveProvider(t, token)
	sentInputs := captureLLMInputs(t)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	videos := map[string][]liveFrame{"A": liveVideoFrames(t, rng, 6), "B": liveVideoFrames(t, rng, 12)}
	var describe []string
	for i, f := range videos["A"] {
		describe = append(describe, fmt.Sprintf("%ds=%s/%s", i*5, f.color, f.corner))
	}
	t.Logf("video A: %s", strings.Join(describe, " "))

	var mu sync.Mutex
	frameCalls := map[string]int{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ID     int64 `json:"id"`
			Params struct {
				Arguments map[string]any `json:"arguments"`
			} `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid test request", http.StatusBadRequest)
			return
		}
		action, _ := req.Params.Arguments["action"].(string)
		video, _ := req.Params.Arguments["video"].(string)
		video = strings.TrimPrefix(strings.ToUpper(strings.TrimSpace(video)), "VIDEO ")
		var payload map[string]any
		switch {
		case action == "video_frames" && videos[video] != nil:
			mu.Lock()
			frameCalls[video]++
			mu.Unlock()
			payload = liveFramesPayload(video, videos[video])
		case action == "click":
			target, _ := req.Params.Arguments["target"].(string)
			payload = mediaResultPayload(
				map[string]any{"type": "text", "text": fmt.Sprintf("Clicked %q. The page settled.", target)},
				mediaImageBlock("image/jpeg", liveScreenshotJPEG(t, uint8(200+rng.Intn(40)))),
			)
		default:
			payload = map[string]any{"isError": true, "content": []map[string]any{{"type": "text", "text": "unknown action or video"}}}
		}
		raw, _ := json.Marshal(payload)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jsonRPCResponse{JSONRPC: "2.0", ID: req.ID, Result: raw})
	}))
	t.Cleanup(server.Close)
	conn := &MCPHTTPServer{Name: "computer", url: server.URL, client: server.Client()}
	t.Cleanup(conn.Close)

	th := newTestThinker()
	defer th.Stop()
	th.session = NewSession(t.TempDir(), "main")
	th.registry = NewToolRegistry("")
	schema := map[string]any{"type": "object", "properties": map[string]any{
		"action": map[string]any{"type": "string", "enum": []string{"video_frames", "click"}, "description": "video_frames returns the frames of a video, one every 5 s, as labelled images. click clicks a button and returns a screenshot."},
		"video":  map[string]any{"type": "string", "enum": []string{"A", "B"}, "description": "For video_frames: which video."},
		"target": map[string]any{"type": "string", "description": "For click: the button text."},
	}, "required": []string{"action"}}
	th.registry.Register(&ToolDef{Name: "computer_use", Description: "Operate the test browser: read video frames or click buttons.", MCP: true, MCPServer: "computer", InputSchema: schema,
		Handler: mcpProxyHandler(conn, "computer_use"), HandlerContext: mcpProxyHandlerContext(conn, "computer_use")})
	tools := []NativeTool{th.registry.Get("computer_use").native}
	th.messages = []Message{{Role: "system", Content: "You operate a test browser through computer_use. Follow each user instruction exactly. When asked for JSON, reply with the JSON object only."}}

	type sample struct {
		label          string
		prompt, cached int
		images         int
		batch          bool
		// changed is the first input item of the previous request that this
		// one did not repeat; prevItems is how many items that request had.
		changed, prevItems int
	}
	var samples []sample
	call := func(label string) (ChatResponse, []Message) {
		t.Helper()
		epoch := th.promptCacheEpoch
		request := th.prepareToolResultRequest(th.messages)
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()
		// As thinkWithProvider: the cache key follows the thread's cache epoch.
		ctx = withOpenAIPromptCacheScope(ctx, openAIPromptCacheScope{Identity: "tool-image-retention-live", Epoch: th.promptCacheEpoch})
		resp, err := p.Chat(ctx, request, toolReasonLiveModel, tools, nil, nil, nil)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		th.markToolResultsConsumed(request)
		s := sample{label: label, prompt: resp.Usage.PromptTokens, cached: resp.Usage.CachedTokens, images: liveRequestImageCount(request), batch: th.promptCacheEpoch != epoch}
		changedAt := ""
		if sent := sentInputs(); len(sent) >= 2 {
			prev, next := sent[len(sent)-2], sent[len(sent)-1]
			s.changed, s.prevItems = firstChangedInput(prev, next), len(prev)
			changedAt = fmt.Sprintf(" prefix=%d/%d", s.changed, s.prevItems)
			if s.changed < len(prev) {
				changedAt += " (" + liveInputKind(prev[s.changed]) + ")"
			}
		}
		samples = append(samples, s)
		t.Logf("%-26s prompt=%6d cached=%6d images=%2d batch=%v%s calls=%d text=%q", label, s.prompt, s.cached, s.images, s.batch, changedAt, len(resp.ToolCalls), resp.Text)
		return resp, request
	}
	user := func(text string) { th.messages = append(th.messages, Message{Role: "user", Content: text}) }
	assistant := func(resp ChatResponse) {
		th.messages = append(th.messages, Message{Role: "assistant", Content: resp.Text, ToolCalls: resp.ToolCalls, Reasoning: resp.Reasoning, ProviderState: resp.ProviderState})
	}
	runTool := func(label string, resp ChatResponse, action string) string {
		t.Helper()
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "computer_use" || resp.ToolCalls[0].Args["action"] != action {
			t.Fatalf("%s: want one computer_use %s call, got calls=%+v text=%q", label, action, resp.ToolCalls, resp.Text)
		}
		assistant(resp)
		event := dispatchMediaFixtureTool(t, th, resp.ToolCalls[0])
		th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{*event.ToolResult}}))
		return resp.ToolCalls[0].ID
	}
	plainReply := func(label string, resp ChatResponse) {
		t.Helper()
		if len(resp.ToolCalls) != 0 {
			t.Fatalf("%s: want a plain reply, got calls=%+v", label, resp.ToolCalls)
		}
		lower := strings.ToLower(resp.Text)
		for _, paint := range liveFramePalette {
			if strings.Contains(lower, paint.name) {
				t.Fatalf("%s: the model described the frames (%q); the recall checks would prove nothing", label, resp.Text)
			}
		}
		assistant(resp)
	}

	// 1. Video A's six labelled frames arrive.
	user("Use computer_use with action=video_frames and video=A to look at video A. After the frames arrive, reply with the single word: ready. Do not describe the frames.")
	resp, _ := call("1 ask for video A")
	callA := runTool("1 ask for video A", resp, "video_frames")
	resp, _ = call("2 video A arrives")
	plainReply("2 video A arrives", resp)

	// 2. Two clicks, each returning a screenshot that replaces the last.
	for i, target := range []string{"Accept cookies", "Continue"} {
		user(fmt.Sprintf("Click the %q button with computer_use (action=click, target=%q). After the screenshot arrives, reply with the single word: done.", target, target))
		label := fmt.Sprintf("%d click %s", 3+2*i, target)
		resp, _ = call(label)
		runTool(label, resp, "click")
		label = fmt.Sprintf("%d screenshot %d", 4+2*i, i+1)
		resp, _ = call(label)
		plainReply(label, resp)
	}

	// 3. The frames are still there, and their labels pick the right one.
	user(`Without calling any tool, look at the frames of video A you already received. Reply only with JSON {"color_at_5s": "...", "square_corner_at_20s": "..."}. Colours: red, green, blue, yellow, purple, orange. Corners: top-left, top-right, bottom-left, bottom-right.`)
	resp, _ = call("7 recall video A")
	if len(resp.ToolCalls) != 0 {
		t.Fatalf("7 recall video A: the model called a tool instead of using the kept frames: %+v", resp.ToolCalls)
	}
	recall := liveJSONReply(t, "7 recall video A", resp.Text)
	if got := recall["color_at_5s"]; got != videos["A"][1].color {
		t.Errorf("colour at 5 s = %q, want %q", got, videos["A"][1].color)
	}
	if got := recall["square_corner_at_20s"]; got != videos["A"][4].corner {
		t.Errorf("square corner at 20 s = %q, want %q", got, videos["A"][4].corner)
	}
	assistant(resp)

	// 4. Video B's twelve frames pass the high mark: video A's frames retire
	// in one batch, with a note naming them.
	user("Use computer_use with action=video_frames and video=B. After the frames arrive, reply with the single word: ready. Do not describe the frames.")
	resp, _ = call("8 ask for video B")
	runTool("8 ask for video B", resp, "video_frames")
	resp, request := call("9 video B arrives")
	if !samples[len(samples)-1].batch {
		t.Fatal("9 video B arrives: expected a retirement batch past the high mark")
	}
	if n := len(requestImages(request, callA)); n != 0 {
		t.Fatalf("video A still sent %d frames after the batch", n)
	}
	if note := requestContent(request, callA); !strings.Contains(note, "removed to save context") || !strings.Contains(note, "Video A, frame at 10 s") {
		t.Fatalf("video A note = %q", note)
	}
	plainReply("9 video B arrives", resp)

	// 5. The model knows which frames it lost and how to get them back.
	user(`Without calling any tool, answer from this conversation as it is now. Reply only with JSON {"video_a_frames_visible": true or false, "video_b_frames_visible": true or false, "how_to_see_video_a_again": "..."}.`)
	resp, _ = call("10 what is visible")
	if len(resp.ToolCalls) != 0 {
		t.Fatalf("10 what is visible: unexpected tool call %+v", resp.ToolCalls)
	}
	visible := liveJSONReply(t, "10 what is visible", resp.Text)
	if visible["video_a_frames_visible"] != "false" || visible["video_b_frames_visible"] != "true" {
		t.Errorf("visibility = %q, want video A false and video B true", resp.Text)
	}
	if how := visible["how_to_see_video_a_again"]; !strings.Contains(how, "video-frames") && !strings.Contains(how, "computer-use") {
		t.Errorf("how to see video A again = %q, want it to name the tool", how)
	}
	assistant(resp)

	// 6. Asked about a frame it has not described, it fetches video A again.
	user(`What colour is video A's frame at 10 s, and in which corner is its white square? Do not guess: if you cannot see that frame now, fetch it again with computer_use first. Then reply only with JSON {"color_at_10s": "...", "square_corner_at_10s": "..."}.`)
	resp, _ = call("11 ask about 10 s")
	if len(resp.ToolCalls) > 0 {
		runTool("11 ask about 10 s", resp, "video_frames")
		resp, _ = call("12 video A again")
	} else {
		t.Errorf("11 ask about 10 s: the model answered %q without fetching the removed frames again", resp.Text)
	}
	answer := liveJSONReply(t, "final answer", resp.Text)
	if answer["color_at_10s"] != videos["A"][2].color || answer["square_corner_at_10s"] != videos["A"][2].corner {
		t.Errorf("frame at 10 s = %q, want %s / %s", resp.Text, videos["A"][2].color, videos["A"][2].corner)
	}
	mu.Lock()
	t.Logf("video_frames calls: A=%d B=%d", frameCalls["A"], frameCalls["B"])
	mu.Unlock()

	// Between batches Core must resend the previous request unchanged; only
	// its last item, the current-screenshot tail, may move. That is what
	// keeps the prompt cache usable, and it is checked on the bytes sent.
	// Whether the provider then serves a request from cache varies from run
	// to run, so that part only has to hold for most requests. The first two
	// requests are too short to cache.
	eligible, hits := 0, 0
	for i := 1; i < len(samples); i++ {
		s := samples[i]
		if s.batch {
			continue
		}
		if s.changed < s.prevItems-1 {
			t.Errorf("%q changed input item %d of %d between batches", s.label, s.changed, s.prevItems)
		}
		if i < 2 {
			continue
		}
		eligible++
		if s.cached*2 >= samples[i-1].prompt {
			hits++
		} else {
			t.Logf("provider cache miss at %q: cached %d of the previous request's %d prompt tokens", s.label, s.cached, samples[i-1].prompt)
		}
	}
	t.Logf("provider cache hits between batches: %d of %d", hits, eligible)
	if hits*2 < eligible {
		t.Errorf("only %d of %d requests between batches were served from the prompt cache", hits, eligible)
	}
}
