package core

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

// TestIntegration_Kimi_VisionInput checks whether kimi-k2.6 served via
// Fireworks vs OpenCode Go accepts OpenAI-format image inputs and
// actually reasons about the pixels. The probe is a synthetic 200×200
// PNG split into four solid-color quadrants (red / green / blue /
// yellow); we ask the model which colors it sees and assert at least
// three are mentioned in the reply.
//
// Why this matters: instance config can route through either provider
// today, so if a future agent gets a screenshot tool result and the
// gateway silently strips multimodal content, the agent thinks
// nothing came through and either hallucinates or stalls. This test
// catches that class of regression cheaply.
//
// Per-endpoint subtest, skips automatically when the matching key is
// unset. Free helpers are reused from provider_kimi_stream_test.go
// (firstNonEmpty, trunc, sortStrings, walkFields) so this file stays
// focused on the vision-specific bits.
func TestIntegration_Kimi_VisionInput(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test, skipping in -short")
	}
	if os.Getenv("RUN_LLM_INTEGRATION_TESTS") != "1" {
		t.Skip("set RUN_LLM_INTEGRATION_TESTS=1 to run paid live-provider integration tests")
	}

	endpoints := []struct {
		name   string
		envVar string
		url    string
		model  string
	}{
		{"fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1/chat/completions", "accounts/fireworks/models/kimi-k2p6"},
		{"opencode-go", "OPENCODE_GO_API_KEY", "https://opencode.ai/zen/go/v1/chat/completions", "kimi-k2.6"},
	}

	// .env loader — same shape as the stream-shape test.
	loadDotEnv(t)

	imgDataURL := makeQuadrantPNG(t)
	expectedColors := []string{"red", "green", "blue", "yellow"}

	for _, ep := range endpoints {
		key := os.Getenv(ep.envVar)
		if key == "" {
			t.Logf("[%s] %s not set, skipping", ep.name, ep.envVar)
			continue
		}

		t.Run(ep.name, func(t *testing.T) {
			body := map[string]any{
				"model":  ep.model,
				"stream": true,
				"messages": []map[string]any{{
					"role": "user",
					"content": []map[string]any{
						{"type": "text", "text": "This image is a 2x2 grid of solid-colored squares. List the four colors you see, one per line, lowercase. No commentary."},
						{"type": "image_url", "image_url": map[string]any{"url": imgDataURL}},
					},
				}},
			}
			raw, _ := json.Marshal(body)

			req, err := http.NewRequest("POST", ep.url, bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Authorization", "Bearer "+key)

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != 200 {
				b, _ := io.ReadAll(resp.Body)
				t.Fatalf("status=%d body=%s", resp.StatusCode, string(b))
			}

			var (
				content   strings.Builder
				reasoning strings.Builder
				promptT   int
				complT    int
			)

			scanner := bufio.NewScanner(resp.Body)
			scanner.Buffer(make([]byte, 1<<20), 1<<20)
			for scanner.Scan() {
				line := scanner.Text()
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}
				var event struct {
					Choices []struct {
						Delta struct {
							Content          string `json:"content"`
							ReasoningContent string `json:"reasoning_content,omitempty"`
							Reasoning        string `json:"reasoning,omitempty"`
						} `json:"delta"`
					} `json:"choices"`
					Usage *struct {
						PromptTokens     int `json:"prompt_tokens"`
						CompletionTokens int `json:"completion_tokens"`
					} `json:"usage,omitempty"`
				}
				if err := json.Unmarshal([]byte(data), &event); err != nil {
					continue
				}
				if len(event.Choices) > 0 {
					d := event.Choices[0].Delta
					content.WriteString(d.Content)
					reasoning.WriteString(firstNonEmpty(d.ReasoningContent, d.Reasoning))
				}
				if event.Usage != nil {
					promptT = event.Usage.PromptTokens
					complT = event.Usage.CompletionTokens
				}
			}

			reply := strings.TrimSpace(content.String())
			thinkPreview := trunc(reasoning.String(), 120)
			t.Logf("==== %s / vision (model=%s) ====", ep.name, ep.model)
			t.Logf("reply        : %q", reply)
			if thinkPreview != "" {
				t.Logf("reasoning    : %q (total %d bytes)", thinkPreview, reasoning.Len())
			}
			t.Logf("usage tokens : prompt=%d completion=%d", promptT, complT)

			lower := strings.ToLower(reply)
			var hits []string
			for _, c := range expectedColors {
				if strings.Contains(lower, c) {
					hits = append(hits, c)
				}
			}
			t.Logf("colors hit   : %v / %v", hits, expectedColors)

			// Cheap canary: any "I can't see images" / "no image was
			// provided" text means the gateway either stripped or didn't
			// support the multimodal content. Surface that loudly.
			for _, blind := range []string{
				"can't see", "cannot see", "no image", "couldn't see",
				"unable to view", "i don't see any image", "i don't have the ability",
			} {
				if strings.Contains(lower, blind) {
					t.Fatalf("model reports it cannot see the image — endpoint may have stripped multimodal content. reply=%q", reply)
				}
			}

			if len(hits) < 3 {
				t.Errorf("expected at least 3 of %v in reply, got %v (reply=%q)", expectedColors, hits, reply)
			}

			fmt.Fprintf(os.Stdout, "[VISION] endpoint=%s model=%s colors_hit=%d/%d completion_tokens=%d\n",
				ep.name, ep.model, len(hits), len(expectedColors), complT)
		})
	}
}

// makeQuadrantPNG generates a 200×200 PNG with four solid-color
// quadrants (TL=red, TR=green, BL=blue, BR=yellow) and returns it as
// a data: URL. The colors are unambiguous primaries so a model that
// can see at all should name them.
func makeQuadrantPNG(t *testing.T) string {
	t.Helper()
	const side = 200
	img := image.NewRGBA(image.Rect(0, 0, side, side))
	red := color.RGBA{255, 0, 0, 255}
	green := color.RGBA{0, 200, 0, 255}
	blue := color.RGBA{0, 64, 255, 255}
	yellow := color.RGBA{255, 230, 0, 255}
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			var c color.RGBA
			switch {
			case x < side/2 && y < side/2:
				c = red
			case x >= side/2 && y < side/2:
				c = green
			case x < side/2 && y >= side/2:
				c = blue
			default:
				c = yellow
			}
			img.SetRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// loadDotEnv is a tiny .env loader local to this test file so we
// don't depend on godotenv being initialized elsewhere.
func loadDotEnv(t *testing.T) {
	t.Helper()
	wd, _ := os.Getwd()
	f, err := os.Open(wd + "/.env")
	if err != nil {
		return
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		k := strings.TrimSpace(line[:eq])
		v := strings.Trim(strings.TrimSpace(line[eq+1:]), `"'`)
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}
}
