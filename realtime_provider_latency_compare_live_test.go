package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Explicit paid replay using identical PCM, instructions and paced silence.
// This isolates provider transport + endpointing; it excludes PSTN and Telephony.
func TestRealtimeProviderLatencyComparison(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_REALTIME_PROVIDER_COMPARE") != "1" {
		t.Skip("paid opt-in provider comparison")
	}
	phrase, err := os.ReadFile(os.Getenv("REALTIME_MEDIA_REPLAY_PCM"))
	if err != nil {
		t.Fatal(err)
	}
	if len(phrase) == 0 || len(phrase)%2 != 0 {
		t.Fatal("expected mono PCM16 at 24kHz")
	}
	variants := []string{"openai_websocket", "openai_webrtc", "gemini_750", "gemini_500"}
	if selected := os.Getenv("REALTIME_COMPARE_VARIANTS"); selected != "" {
		variants = strings.Split(selected, ",")
	}
	for trial := 1; trial <= 3; trial++ {
		for _, variant := range variants {
			t.Run(fmt.Sprintf("%s_%d", variant, trial), func(t *testing.T) {
				keyName := "GOOGLE_API_KEY"
				if strings.HasPrefix(variant, "openai_") {
					keyName = "OPENAI_API_KEY"
				}
				if os.Getenv(keyName) == "" {
					t.Fatalf("%s required", keyName)
				}
				switch variant {
				case "openai_websocket", "openai_webrtc", "gemini_750", "gemini_500":
				default:
					t.Fatalf("unknown variant %q", variant)
				}
				var p RealtimeProvider = NewGoogleRealtimeProvider(os.Getenv("GOOGLE_API_KEY"))
				instruction := "Vous êtes un assistant francophone. Attendez la question audio et répondez directement en une seule phrase courte. Aucune recherche ni outil pour les questions simples de culture générale. Ne saluez pas et ne posez pas de question."
				opts := RealtimeSessionOpts{Model: "gemini-3.8-live", Voice: "Kore", Instructions: instruction, ConversationInstructions: instruction, AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16, AudioInRate: 24000, AudioOutRate: 24000, TranscribeInput: true, TurnDetection: RealtimeTurnDetectionConfig{Profile: "telephony"}}
				if strings.HasPrefix(variant, "openai_") {
					p = NewOpenAIRealtimeProvider(os.Getenv("OPENAI_API_KEY"))
					opts.Model = "gpt-live-1"
					opts.Voice = "brise"
					opts.OutputConfig.Media = RealtimeMediaConfig{Transport: strings.TrimPrefix(variant, "openai_"), Format: AudioPCM16}
				}
				if variant == "gemini_500" {
					opts.TurnDetection.SilenceDurationMS = 500
				}
				ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
				defer cancel()
				start := time.Now()
				s, err := p.Open(ctx, opts)
				if err != nil {
					t.Fatal(err)
				}
				defer s.Close()
				setup := time.Since(start)
				var mu sync.Mutex
				input := make([]byte, 48000+len(phrase)+6*48000)
				copy(input[48000:], phrase)
				endpoint := replayAcousticEndpoint(input)
				if endpoint == 0 {
					t.Fatal("no input speech above documented acoustic threshold")
				}
				origin := time.Now()
				observation := realtimeReplayAcousticTiming{endpoint: origin.Add(endpoint)}
				var output []byte
				var captions []string
				var failures, shutdownErrors, assistantText []string
				var closing bool
				done := make(chan struct{})
				go func() {
					defer close(done)
					for ev := range s.Events() {
						mu.Lock()
						if ev.Type == RealtimeEventError && ev.Err != nil {
							if closing {
								shutdownErrors = append(shutdownErrors, ev.Err.Error())
							} else {
								failures = append(failures, ev.Err.Error())
							}
						}
						if ev.Transcript != "" {
							captions = append(captions, ev.Transcript)
							if ev.Type == RealtimeEventTranscriptOutput {
								assistantText = append(assistantText, ev.Transcript)
							}
						}
						if len(ev.Audio) > 0 {
							output = append(output, ev.Audio...)
							observation.audio(ev.Audio, time.Now())
						}
						mu.Unlock()
					}
				}()
				for off := 0; off < len(input); off += 960 {
					delay := time.Until(origin.Add(time.Duration(off) * time.Second / 48000))
					if delay > 0 {
						select {
						case <-ctx.Done():
							t.Fatal(ctx.Err())
						case <-time.After(delay):
						}
					}
					frame := input[off:min(off+960, len(input))]
					if err := s.SendAudio(frame); err != nil {
						t.Fatal(err)
					}
				}
				mu.Lock()
				closing = true
				mu.Unlock()
				_ = s.Close()
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Fatal("events did not close")
				}
				mu.Lock()
				defer mu.Unlock()
				report := map[string]any{"variant": variant, "trial": trial, "setup_ms": setup.Milliseconds(), "transcript": captions, "errors": failures, "shutdown_errors": shutdownErrors, "assistant_transcript": assistantText, "output_bytes": len(output)}
				report["first_acoustic_ms"] = timingMS(observation.endpoint, observation.firstAcoustic)
				report["first_packet_ms"] = timingMS(observation.endpoint, observation.firstPacket)
				report["overlap_or_continuation_possible"] = observation.overlap
				report["eligible_latency_sample"] = !observation.overlap && !observation.firstAcoustic.IsZero()
				report["missed_acoustic_response"] = observation.firstAcoustic.IsZero()
				report["quality_gate"] = "diagnostic_only_requires_semantic_and_playback_review"
				report["speech_end_source"] = "recording_acoustic_endpoint"
				report["acoustic_evidence"] = "two_consecutive_10ms_rms_windows"
				data, _ := json.MarshalIndent(report, "", "  ")
				t.Log(string(data))
				if dir := os.Getenv("REALTIME_MEDIA_REPORT_DIR"); dir != "" {
					if err := os.MkdirAll(dir, 0700); err != nil {
						t.Fatal(err)
					}
					base := filepath.Join(dir, fmt.Sprintf("%s_%d", variant, trial))
					if err := os.WriteFile(base+".json", data, 0600); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(base+".pcm16", output, 0600); err != nil {
						t.Fatal(err)
					}
				}
				if expected := os.Getenv("REALTIME_MEDIA_EXPECTED_TEXT"); expected != "" && !strings.Contains(strings.ToLower(strings.Join(assistantText, "")), strings.ToLower(expected)) {
					t.Error("response did not include expected answer")
				}
				if observation.firstAcoustic.IsZero() {
					t.Error("no audible response")
				}
				if len(failures) > 0 {
					t.Error("provider returned errors")
				}
			})
		}
	}
}
