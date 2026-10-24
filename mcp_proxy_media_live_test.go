package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image/jpeg"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"
)

// Paid opt-in. The real model must request the tool, and read actual pixels
// after HTTP MCP decoding, proxying and asynchronous Core dispatch. Colors
// appear only in the image. The companion document body appears only in the
// resource content, rather than in the prompt or the tool's description.
func TestIntegration_CodexGPT61SolMCPNativeVision(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_MCP_NATIVE_VISION") != "1" {
		t.Skip("set RUN_CODEX_MCP_NATIVE_VISION=1 without -short for paid native MCP vision verification")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live Codex test requested without a valid local credential")
	}
	p := toolReasonLiveProvider(t, token)
	_, encoded, _ := strings.Cut(makeQuadrantPNG(t), ",")
	pngData, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := png.Decode(bytes.NewReader(pngData))
	if err != nil {
		t.Fatal(err)
	}
	var jpegData bytes.Buffer
	if err := jpeg.Encode(&jpegData, decoded, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, mime string
		data       []byte
	}{{"png_tool_image", "image/png", pngData}, {"jpeg_content_part", "image/jpeg", jpegData.Bytes()}} {
		t.Run(tc.name, func(t *testing.T) {
			const appointment = "Tuesday at 10:00"
			conn := mediaResultMCP(t, mediaResultPayload(map[string]any{"type": "text", "text": "Content retrieved successfully."}, mediaImageBlock(tc.mime, tc.data), map[string]any{"type": "resource", "resource": map[string]any{"uri": "document:appointment", "mimeType": "text/plain", "text": "The agreed appointment is " + appointment + "."}}))
			th := newTestThinker()
			defer th.Stop()
			def := registerMediaFixtureTool(th, conn)
			messages := []Message{{Role: "system", Content: "Call media_read exactly once to retrieve the source image and document. Inspect the returned pixels; never guess if content is absent. Then reply only with a JSON object containing top_left, top_right, bottom_left, bottom_right as lowercase basic color names, and appointment as the agreed day/time exactly as stated in the document. Do not use markdown."}, {Role: "user", Content: "Read the image's four quadrants and the appointment in its accompanying document."}}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			first, err := p.Chat(ctx, messages, toolReasonLiveModel, []NativeTool{def.native}, nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(first.ToolCalls) != 1 || first.ToolCalls[0].Name != def.Name {
				t.Fatalf("expected a real media tool call, got calls=%+v text=%q", first.ToolCalls, first.Text)
			}
			event := dispatchMediaFixtureTool(t, th, first.ToolCalls[0])
			messages = append(messages, Message{Role: "assistant", Content: first.Text, ToolCalls: first.ToolCalls, Reasoning: first.Reasoning, ProviderState: first.ProviderState})
			messages = append(messages, mediaEventMessages(event)...)
			result, err := p.Chat(ctx, messages, toolReasonLiveModel, nil, nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			var report map[string]string
			if err := json.Unmarshal([]byte(strings.TrimSpace(result.Text)), &report); err != nil {
				t.Fatalf("invalid visual report: %v; text=%q", err, result.Text)
			}
			for key, want := range map[string]string{"top_left": "red", "top_right": "green", "bottom_left": "blue", "bottom_right": "yellow", "appointment": appointment} {
				if report[key] != want {
					t.Fatalf("%s=%q want %q; report=%q", key, report[key], want, result.Text)
				}
			}
			t.Logf("model=%s source=%s verified real tool call, four image quadrants and accompanying document; image_bytes=%d parts=%d", toolReasonLiveModel, tc.mime, len(event.ToolResult.Image), len(event.Parts))
		})
	}
}
