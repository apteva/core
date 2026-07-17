package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestXAIRealtimeLiveToolContinuation is an opt-in paid smoke for xAI's
// production Voice Agent endpoint.
//
// RUN_XAI_REALTIME_SMOKE=1 XAI_API_KEY=... go test -run TestXAIRealtimeLiveToolContinuation -timeout 2m .
func TestXAIRealtimeLiveToolContinuation(t *testing.T) {
	if os.Getenv("RUN_XAI_REALTIME_SMOKE") != "1" {
		t.Skip("set RUN_XAI_REALTIME_SMOKE=1 to run the paid xAI realtime smoke")
	}
	key := strings.TrimSpace(os.Getenv("XAI_API_KEY"))
	if key == "" {
		t.Skip("XAI_API_KEY not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	provider := NewXAIRealtimeProvider(key)
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: provider.DefaultVoice(),
		Instructions: "Call probe exactly once, then say the returned value exactly and nothing else.",
		Tools: []NativeTool{{
			Name: "probe", Description: "Return the deterministic test value.",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		}},
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		TranscribeInput: true, TranscriptionModel: provider.DefaultTranscriptionModel(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	if err := session.SendText("user", "Run the probe now."); err != nil {
		t.Fatal(err)
	}
	toolSeen, resultPending := false, false
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
				toolSeen, resultPending = true, true
				if err := session.SendToolResult(event.ToolCallID, "PONG", false); err != nil {
					t.Fatal(err)
				}
			}
			if event.Type == RealtimeEventResponseDone && resultPending {
				resultPending = false
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
