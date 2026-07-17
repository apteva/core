package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestOpenAIRealtimeLiveToolContinuation is an opt-in paid smoke for the
// actual GA WebSocket endpoint. It validates the two places a static fixture
// cannot: provider acceptance of session.update and tool-result continuation.
//
// RUN_OPENAI_REALTIME_SMOKE=1 OPENAI_API_KEY=... go test -run TestOpenAIRealtimeLiveToolContinuation -timeout 2m .
func TestOpenAIRealtimeLiveToolContinuation(t *testing.T) {
	if os.Getenv("RUN_OPENAI_REALTIME_SMOKE") != "1" {
		t.Skip("set RUN_OPENAI_REALTIME_SMOKE=1 to run the paid OpenAI Realtime smoke")
	}
	key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	provider := NewOpenAIRealtimeProvider(key)
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: provider.DefaultVoice(),
		Instructions: "Call probe exactly once, then say the returned value exactly and nothing else.",
		Tools: []NativeTool{{
			Name: "probe", Description: "Return the deterministic test value.",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000, TranscribeInput: true,
		SafetyIdentifier: "apt_realtime_smoke",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.SendText("user", "Run the probe now."); err != nil {
		t.Fatal(err)
	}
	toolSeen := false
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("timed out; tool_seen=%v", toolSeen)
		case event, ok := <-session.Events():
			if !ok {
				t.Fatalf("session ended; tool_seen=%v", toolSeen)
			}
			if event.Type == RealtimeEventError {
				t.Fatal(event.Err)
			}
			if event.Type == RealtimeEventToolCall && event.ToolName == "probe" {
				toolSeen = true
				if err := session.SendToolResult(event.ToolCallID, "PONG", false); err != nil {
					t.Fatal(err)
				}
				if err := session.RequestResponse(); err != nil {
					t.Fatal(err)
				}
			}
			if event.Type == RealtimeEventTranscriptOutput && event.Final && strings.Contains(strings.ToUpper(event.Transcript), "PONG") {
				if !toolSeen {
					t.Fatal("model answered without calling probe")
				}
				return
			}
		}
	}
}
