package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Exercise the complete request projection, rather than only the screenshot
// helper: a consumed current frame must survive historical-result aging.
func TestContextEfficiencyComputerScreenshotSurvivesProjection(t *testing.T) {
	for _, textPressure := range []bool{false, true} {
		name := "images_only"
		if textPressure {
			name = "text_pressure"
		}
		t.Run(name, func(t *testing.T) {
			th := &Thinker{threadID: "worker", session: NewSession(t.TempDir(), "worker")}
			th.messages = []Message{{Role: "system", Content: "Navigate using the current screen."}, {Role: "user", Content: "Inspect the current page."}}
			var latest []byte
			for _, id := range []string{"old-screen", "current-screen"} {
				pixels := computerScreenshotJPEG(t, 180_000)
				if id == "current-screen" {
					latest = pixels
				}
				th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "computer_computer_use"}}})
				observation := th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: "Navigation succeeded; screen metadata " + id + strings.Repeat(" element coordinates ", 1000), Image: pixels}}})
				th.messages = append(th.messages, observation)
			}
			if textPressure {
				th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "large-text", Name: "read_document"}}})
				observation := th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "large-text", Content: strings.Repeat("historical document ", 20_000)}}})
				th.messages = append(th.messages, observation)
			}
			for i := 0; i < toolResultFullRetentionCalls; i++ {
				th.markToolResultsConsumed(th.messages)
			}
			before := cloneMessages(th.messages)
			projected := th.prepareToolResultRequest(th.messages)
			if !reflect.DeepEqual(th.messages, before) {
				t.Fatal("projection mutated live image history")
			}
			if len(projected) != len(before)+1 {
				t.Fatal("current screenshot request tail missing after retention", len(projected))
			}
			tail := projected[len(projected)-1]
			want := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(latest)
			if !tail.RequestContext || len(tail.Parts) != 2 || tail.Parts[1].ImageURL == nil || tail.Parts[1].ImageURL.URL != want || !strings.Contains(tail.Parts[0].Text, "current-screen") {
				t.Fatal("current frame or source ID changed")
			}
			wire := string(mustJSON(t, (&OpenAINativeProvider{}).buildInput(projected)))
			if strings.Count(wire, want) != 1 {
				t.Fatal("native request did not include exactly one current screenshot")
			}
			if projected[5].ToolResults[0].Content != before[5].ToolResults[0].Content || projected[5].ToolResults[0].ContentIsPreview {
				t.Fatal("current navigation metadata shortened")
			}
			if !textPressure && th.promptCacheEpoch != 0 {
				t.Fatal("compressed image bytes triggered text-retention checkpoint")
			}
			if textPressure && !projected[len(before)-1].ToolResults[0].ContentIsPreview {
				t.Fatal("mature oversized text was not shortened")
			}
			// The previously current frame may age only when a newer frame
			// replaces it. Scheduled checkpoints must preserve that new one.
			th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "next-screen", Name: "computer_computer_use"}}})
			th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "next-screen", Content: "Next navigation state", Image: latest}}}))
			th.commitMatureToolResults(th.messages)
			if !th.toolResultIsHistorical(th.messages[5].ToolResults[0]) || th.toolResultIsHistorical(th.messages[len(th.messages)-1].ToolResults[0]) {
				t.Fatal("screenshot replacement failed to move current-frame protection")
			}
		})
	}
}

// Read actual pixels after Core's full historical-result projection. The
// fixture's colors are absent from tool text, so text-only success is impossible.
func TestIntegration_CodexGPT61SolComputerVisionRetention(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_COMPUTER_VISION_LIVE") != "1" {
		t.Skip("set RUN_CODEX_COMPUTER_VISION_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live vision test requested without valid Codex credential")
	}
	p := toolReasonLiveProvider(t, token)
	th := retryTestThinker(p)
	th.session = NewSession(t.TempDir(), "main")
	_, data, _ := strings.Cut(makeQuadrantPNG(t), ",")
	pixels, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		t.Fatal(err)
	}
	th.messages = []Message{
		{Role: "system", Content: "Inspect only the current screenshot's pixels. Reply as a JSON object with keys top_left, top_right, bottom_left, bottom_right and the lowercase basic color in each quadrant. No tools or commentary."},
		{Role: "user", Content: "What colors are in the four quadrants of the current screen?"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screen", Name: "computer_computer_use", Args: map[string]string{"action": "screenshot"}}}},
	}
	th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "screen", Content: "Capture successful", Image: pixels}}}))
	th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "document", Name: "read_document"}}})
	th.messages = append(th.messages, th.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "document", Content: strings.Repeat("Historical document unrelated to the current screen. ", 7000)}}}))
	for i := 0; i < toolResultFullRetentionCalls; i++ {
		th.markToolResultsConsumed(th.messages)
	}
	projected := th.prepareToolResultRequest(th.messages)
	if !projected[5].ToolResults[0].ContentIsPreview {
		t.Fatal("fixture did not exercise pressure compaction")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	response, err := th.thinkWithProviderMessages(ctx, p, projected)
	if err != nil {
		t.Fatal(err)
	}
	var colors map[string]string
	if err := json.Unmarshal([]byte(strings.TrimSpace(response.Text)), &colors); err != nil {
		t.Fatalf("invalid visual report: %v; text=%q", err, response.Text)
	}
	for position, want := range map[string]string{"top_left": "red", "top_right": "green", "bottom_left": "blue", "bottom_right": "yellow"} {
		if colors[position] != want {
			t.Fatalf("current pixels lost: %s=%q want %q", position, colors[position], want)
		}
	}
	t.Logf("model=%s correctly read all four screenshot quadrants after compaction; input_tokens=%d", response.Model, response.Usage.PromptTokens)
}
