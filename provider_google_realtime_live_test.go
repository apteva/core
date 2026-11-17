package core

import (
	"context"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestGoogleRealtimeLiveToolContinuation is an opt-in paid smoke against
// Gemini Live. It verifies the production WebSocket setup, history unlock,
// function call, batched function response, continuation, and transcript with
// the full registered Core tool set plus representative external JSON Schemas.
// Only the deterministic probe is executed; no real tool side effects occur.
//
// RUN_GOOGLE_REALTIME_SMOKE=1 GOOGLE_API_KEY=... go test -run TestGoogleRealtimeLiveToolContinuation -timeout 2m .
func TestGoogleRealtimeLiveToolContinuation(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_REALTIME_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_SMOKE=1 to run the paid Google realtime smoke")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY must be set when requesting the live schema smoke")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	provider := NewGoogleRealtimeProvider(key)
	tools := googleFullSchemaSmokeTools(t)
	t.Logf("opening Gemini Live with %d tools (all Core built-ins, including search_tools, plus schema fixtures and probe)", len(tools))
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: provider.Models()[ModelSmall], Voice: provider.DefaultVoice(),
		Instructions: "Call probe exactly once, then say the returned value exactly and nothing else. Never announce or name the tool.",
		Tools:        tools,
		AudioInFmt:   AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		TranscribeInput: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	t.Log("Google acknowledged setupComplete with the full tool set")
	if err := session.RestoreConversation(nil); err != nil {
		t.Fatal(err)
	}
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
			if event.Type == RealtimeEventToolCall && event.ToolName != "probe" {
				t.Fatalf("unexpected tool %q; refusing to execute anything except probe", event.ToolName)
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
				t.Log("received probe call, returned its result, and received spoken PONG transcript")
				return
			}
		}
	}
}

func googleFullSchemaSmokeTools(t *testing.T) []NativeTool {
	t.Helper()
	tools := NewToolRegistry("google-full-schema-smoke").NativeTools(nil, nil, true)
	if !nativeToolSet(tools)["search_tools"] {
		t.Fatal("full-tool smoke must include the production search_tools schema")
	}
	fixtures := googleSchemaRegressionFixtures(t)
	names := make([]string, 0, len(fixtures))
	for name := range fixtures {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tools = append(tools, NativeTool{Name: "schema_" + name, Description: "Schema compatibility fixture; do not call.", Parameters: fixtures[name]})
	}
	return append(tools, NativeTool{
		Name: "probe", Description: "Return the deterministic test value.",
		Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
	})
}

// The standard Gemini path shares the JSON Schema adapter with Live. Verify
// Google's actual GenerateContent validator accepts the same full tool set.
func TestGoogleLiveSchemaStandardGeminiSmoke(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_REALTIME_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_SMOKE=1 to run the paid Google schema smoke")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY must be set when requesting the live schema smoke")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	response, err := NewGoogleProvider(key).Chat(ctx, []Message{
		{Role: "user", Content: "Call probe exactly once. Do not call any other tool."},
	}, "gemini-2.5-flash", googleFullSchemaSmokeTools(t), nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "probe" {
		t.Fatalf("expected probe call, got %#v", response.ToolCalls)
	}
	t.Log("standard Gemini accepted the full tool set and returned probe")
}
