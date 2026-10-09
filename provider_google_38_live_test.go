package core

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"
)

var liveProbeBothCompleted = regexp.MustCompile(`(?i)\bboth(?: probes| tools| calls| results| markers)?(?: have| are| now| just| already| successfully)* (?:returned|completed|finished|done)\b|\bboth(?: returned)? markers are\b`)
var liveProbeMarkerPair = regexp.MustCompile(`(?i)\b(ALPHA|BRAVO)\s+(?:and|&)\s+([a-z]+)\b`)

// This evaluator is deliberately specific to the known probe data. Unlike the
// old final-substring check, a correction cannot erase an earlier wrong claim.
func liveProbeTranscriptError(text string, delivered map[string]bool) error {
	upper := strings.ToUpper(text)
	if strings.Contains(upper, "BETA") {
		return fmt.Errorf("invented marker in speech: %q", text)
	}
	for _, pair := range liveProbeMarkerPair.FindAllStringSubmatch(upper, -1) {
		// Accept ordinary progress constructions such as "ALPHA and I am
		// waiting"; code-like claims must contain the actual other marker.
		switch pair[2] {
		case "I", "WE", "THE", "AM", "ARE", "WAITING", "STILL":
			continue
		}
		if pair[2] != "ALPHA" && pair[2] != "BRAVO" || pair[1] == pair[2] {
			return fmt.Errorf("incorrect marker pair in speech: %q", text)
		}
	}
	for tool, marker := range map[string]string{"probe_fast": "ALPHA", "probe_slow": "BRAVO"} {
		if strings.Contains(upper, marker) && !delivered[tool] {
			return fmt.Errorf("spoke %s before its tool result was delivered: %q", marker, text)
		}
	}
	if liveProbeBothCompleted.MatchString(text) && (!delivered["probe_fast"] || !delivered["probe_slow"]) {
		return fmt.Errorf("claimed both results before delivery: %q", text)
	}
	return nil
}

// RUN_GOOGLE_38_LIVE_SMOKE=1 GOOGLE_API_KEY=... go test -run TestGoogle38Live -v .
// Paid, real-provider verification. All registry tools are declared, but only
// the two deterministic local probes can execute (no real external actions).
func TestGoogle38LiveParallelTools(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_38_LIVE_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_38_LIVE_SMOKE=1 for paid Gemini 3.8 verification")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY is required for the live smoke")
	}
	for _, test := range []struct {
		name, model, mode string
		docControl        bool
	}{
		{"standard_blocking", "gemini-3.8-live", "", false}, {"standard_async", "gemini-3.8-live", "async", false}, {"extended_async", "gemini-3.8-live-extended-thinking", "", false},
		{"standard_async_doc_control", "gemini-3.8-live", "async", true},
	} {
		model := test.model
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			provider := NewGoogleRealtimeProvider(key)
			var tools []NativeTool
			if !test.docControl {
				tools = googleFullSchemaSmokeTools(t)
			}
			for _, name := range []string{"probe_fast", "probe_slow"} {
				tools = append(tools, NativeTool{Name: name, Description: "Return a secret test marker after a short delay.", Parameters: map[string]any{"type": "object", "properties": map[string]any{}}})
			}
			session, err := provider.Open(ctx, RealtimeSessionOpts{
				Model: model, Reasoning: "low", Voice: "Kore", Tools: tools, OutputConfig: RealtimeOutputConfig{ToolMode: test.mode},
				Instructions: "Say Checking now briefly, then call probe_fast and probe_slow exactly once each, in parallel. You may give brief progress speech while waiting. After BOTH return, say both returned markers. Call no other tools and never invent their results.",
				AudioInFmt:   AudioPCM16, AudioOutFmt: AudioPCM16, AudioInRate: 24000, AudioOutRate: 24000, TranscribeInput: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer session.Close()
			t.Logf("setupComplete: model=%s tools=%d async=%v", model, len(tools), realtimeToolsAreAsync(session))
			var sendErr error
			if test.docControl {
				// Match the guide's finite text-turn flow, with only the two
				// relevant functions, to separate Core's broader tool surface
				// and live text input from the model's async coordination.
				sendErr = session.(*googleRealtimeSession).enqueue(map[string]any{"clientContent": map[string]any{
					"turns":        []map[string]any{{"role": "user", "parts": []map[string]any{{"text": "Run both probes and tell me their returned markers."}}}},
					"turnComplete": true,
				}})
			} else {
				sendErr = session.SendText("user", "Run both probes and tell me their returned markers.")
			}
			if sendErr != nil {
				t.Fatal(sendErr)
			}
			type result struct{ id, name, value string }
			results := make(chan result, 2)
			calls := map[string]bool{}
			delivered := map[string]bool{}
			pending, returned, audioBytes, utterances := 0, 0, 0, 0
			batchDone, needsFlush := false, false
			transcript := ""
			var violations []string
			violatingItems := map[string]bool{}
			for {
				select {
				case <-ctx.Done():
					t.Fatalf("timeout: calls=%v returned=%d utterances=%d audio=%d transcript=%q", calls, returned, utterances, audioBytes, transcript)
				case result := <-results:
					if err := session.SendToolResult(result.id, result.value, false); err != nil {
						t.Fatal(err)
					}
					delivered[result.name] = true
					t.Logf("result delivered: %s = %s", result.name, result.value)
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
						go func(id, name, value string, delay time.Duration) {
							timer := time.NewTimer(delay)
							defer timer.Stop()
							select {
							case <-ctx.Done():
								return
							case <-timer.C:
							}
							select {
							case results <- result{id, name, value}:
							case <-ctx.Done():
							}
						}(event.ToolCallID, event.ToolName, value, delay)
					case RealtimeEventAudioOut:
						audioBytes += len(event.Audio)
					case RealtimeEventTranscriptOutput:
						if err := liveProbeTranscriptError(event.Transcript, delivered); err != nil {
							key := event.ResponseID + "/" + event.ItemID
							if !violatingItems[key] {
								violatingItems[key] = true
								violations = append(violations, err.Error())
								t.Logf("output violation: %v", err)
							}
						}
						if event.Final {
							transcript += " " + event.Transcript
							t.Logf("utterance: %s", event.Transcript)
						}
					case RealtimeEventUtteranceDone:
						utterances++
					case RealtimeEventResponseDone:
						batchDone = true
						if returned == 2 && strings.Contains(strings.ToUpper(transcript), "ALPHA") && strings.Contains(strings.ToUpper(transcript), "BRAVO") && audioBytes > 0 {
							if len(violations) > 0 {
								t.Fatalf("incorrect intermediate speech cannot be repaired by a correction: %v; transcript=%q", violations, transcript)
							}
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
	loadIntegrationEnv()
	if strings.TrimSpace(os.Getenv("GOOGLE_API_KEY")) == "" {
		t.Fatal("GOOGLE_API_KEY is required")
	}
	for _, test := range []struct{ name, model, mode string }{
		{"standard_blocking", "gemini-3.8-live", ""}, {"standard_async", "gemini-3.8-live", "async"}, {"extended_async", "gemini-3.8-live-extended-thinking", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("RUN_GOOGLE_REALTIME_MCP_SMOKE", "1")
			runGoogleRealtimeLiveMCPThread(t, test.model, RealtimeOutputConfig{ToolMode: test.mode})
		})
	}
}
