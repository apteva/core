package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// RUN_GOOGLE_38_LIVE_SMOKE=1 GOOGLE_API_KEY=... go test -run TestGoogle38Live -v .
// Paid, real-provider verification. All registry tools are declared, but only
// the two deterministic local probes can execute (no real external actions).
func TestGoogle38LiveParallelTools(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_38_LIVE_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_38_LIVE_SMOKE=1 for paid Gemini 3.8 verification")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY is required for the live smoke")
	}
	for _, model := range []string{"gemini-3.8-live", "gemini-3.8-live-extended-thinking"} {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			provider := NewGoogleRealtimeProvider(key)
			tools := googleFullSchemaSmokeTools(t)
			for _, name := range []string{"probe_fast", "probe_slow"} {
				tools = append(tools, NativeTool{Name: name, Description: "Return a secret test marker after a short delay.", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}})
			}
			session, err := provider.Open(ctx, RealtimeSessionOpts{
				Model: model, Reasoning: "low", Voice: "Kore", Tools: tools,
				Instructions: "Say Checking now briefly, then call probe_fast and probe_slow exactly once each, in parallel. You may give brief progress speech while waiting. After BOTH return, say both returned markers. Call no other tools and never invent their results.",
				AudioInFmt:   AudioPCM16, AudioOutFmt: AudioPCM16, AudioInRate: 24000, AudioOutRate: 24000, TranscribeInput: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer session.Close()
			t.Logf("setupComplete: model=%s tools=%d async=%v", model, len(tools), realtimeToolsAreAsync(session))
			if err := session.SendText("user", "Run both probes and tell me their returned markers."); err != nil {
				t.Fatal(err)
			}
			type result struct{ id, value string }
			results := make(chan result, 2)
			calls := map[string]bool{}
			pending, returned, audioBytes, utterances := 0, 0, 0, 0
			batchDone, needsFlush := false, false
			transcript := ""
			for {
				select {
				case <-ctx.Done():
					t.Fatalf("timeout: calls=%v returned=%d utterances=%d audio=%d transcript=%q", calls, returned, utterances, audioBytes, transcript)
				case result := <-results:
					if err := session.SendToolResult(result.id, result.value, false); err != nil {
						t.Fatal(err)
					}
					pending--
					returned++
					needsFlush = true
				case event, ok := <-session.Events():
					if !ok {
						t.Fatal("session ended before completion")
					}
					switch event.Type {
					case RealtimeEventError:
						t.Fatal(event.Err)
					case RealtimeEventToolCall:
						if (event.ToolName != "probe_fast" && event.ToolName != "probe_slow") || calls[event.ToolName] {
							t.Fatalf("unexpected/repeated tool: %q", event.ToolName)
						}
						calls[event.ToolName] = true
						pending++
						value, delay := "ALPHA", 200*time.Millisecond
						if event.ToolName == "probe_slow" {
							value, delay = "BRAVO", 2*time.Second
						}
						go func(id, value string, delay time.Duration) {
							timer := time.NewTimer(delay)
							defer timer.Stop()
							select {
							case <-ctx.Done():
								return
							case <-timer.C:
							}
							select {
							case results <- result{id, value}:
							case <-ctx.Done():
							}
						}(event.ToolCallID, value, delay)
					case RealtimeEventAudioOut:
						audioBytes += len(event.Audio)
					case RealtimeEventTranscriptOutput:
						if event.Final {
							transcript += " " + event.Transcript
							t.Logf("utterance: %s", event.Transcript)
						}
					case RealtimeEventUtteranceDone:
						utterances++
					case RealtimeEventResponseDone:
						batchDone = true
						if returned == 2 && strings.Contains(strings.ToUpper(transcript), "ALPHA") && strings.Contains(strings.ToUpper(transcript), "BRAVO") && audioBytes > 0 {
							t.Logf("completed after both delayed results: utterances=%d audio_bytes=%d", utterances, audioBytes)
							return
						}
					}
				}
				if !realtimeToolsAreAsync(session) && needsFlush && pending == 0 && batchDone {
					if err := session.RequestResponse(); err != nil {
						t.Fatal(err)
					}
					needsFlush, batchDone = false, false
				}
			}
		})
	}
}

func TestGoogle38LiveMCPThreads(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_38_LIVE_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_38_LIVE_SMOKE=1 for paid Gemini 3.8 verification")
	}
	if strings.TrimSpace(os.Getenv("GOOGLE_API_KEY")) == "" {
		t.Fatal("GOOGLE_API_KEY is required")
	}
	for _, model := range []string{"gemini-3.8-live", "gemini-3.8-live-extended-thinking"} {
		t.Run(model, func(t *testing.T) {
			t.Setenv("RUN_GOOGLE_REALTIME_MCP_SMOKE", "1")
			runGoogleRealtimeLiveMCPThread(t, model)
		})
	}
}
