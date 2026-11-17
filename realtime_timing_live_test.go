package core

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

// Paid opt-in, real Gemini + Core + local websocket bridge. The same generated
// caller recording is streamed in 20ms frames for both settings. These small
// samples validate instrumentation/behavior, not statistically prove latency.
// RUN_GOOGLE_REALTIME_TIMING_LIVE=1 go test -run TestGoogleRealtimeTimingLive -v .
func TestGoogleRealtimeTimingLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_GOOGLE_REALTIME_TIMING_LIVE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_TIMING_LIVE=1 for paid Gemini timing verification")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY required")
	}
	t.Setenv("TELEMETRY_URL", "")
	t.Setenv("TELEMETRY_LIVE_URL", "")
	t.Setenv("SERVER_URL", "")
	provider := NewGoogleRealtimeProvider(key)
	provider.models = map[ModelTier]string{ModelLarge: "gemini-3.8-live", ModelMedium: "gemini-3.8-live", ModelSmall: "gemini-3.8-live"}
	caller := synthesizeGoogleRealtimeSpeech(t, provider, "Aoede", "Please say the word orange.")
	for sample := 1; sample <= 2; sample++ {
		for _, silence := range []int{750, 500} {
			t.Run(fmt.Sprintf("silence_%d_sample_%d", silence, sample), func(t *testing.T) { runGoogleRealtimeBridgeTiming(t, caller, silence, sample == 2) })
		}
	}
}

func runGoogleRealtimeBridgeTiming(t *testing.T, caller []int16, silence int, reportSpeechEnd bool) {
	t.Helper()
	t.Chdir(t.TempDir())
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Coordinate the local timing test.", RealtimeEnabled: true,
		Providers: []ProviderConfig{{Name: "google-realtime", Default: true, Models: map[string]string{"large": "gemini-3.8-live", "medium": "gemini-3.8-live", "small": "gemini-3.8-live"}}}}
	parent := NewThinker("", &inertRealtimeTextProvider{}, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()
	input := make(chan []byte, 64)
	output := make(chan RealtimeAudioFrame, 64)
	control := make(chan string, 8)
	const id = "timing-live"
	if err := parent.threads.SpawnWithOpts(id, "Stay silent until the caller speaks. When asked to say a word, answer only that word. Keep the conversation open. Never call tools.", []string{"send"}, SpawnOpts{Realtime: true, Ephemeral: true, ProviderName: "google-realtime", AudioIn: input, AudioOut: output, AudioControl: control, TurnDetection: RealtimeTurnDetectionConfig{Profile: "telephony", SilenceDurationMS: silence}}); err != nil {
		t.Fatal(err)
	}
	api := &APIServer{thinker: parent}
	server := httptest.NewServer(http.HandlerFunc(api.realtimeAudioHandler))
	defer server.Close()
	token := registerAudioBridge(id, input, output, control)
	defer unregisterAudioBridge(id)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	conn, reader, _, err := ws.Dial(ctx, "ws"+strings.TrimPrefix(server.URL, "http")+"/realtime/audio?thread="+id+"&token="+token)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if reader != nil {
		ws.PutReader(reader)
	}
	writeDone := make(chan error, 1)
	go func() {
		const frameSamples = 480
		write := func(pcm []byte) error {
			if err := wsutil.WriteClientBinary(conn, pcm); err != nil {
				return err
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(20 * time.Millisecond):
				return nil
			}
		}
		for offset := 0; offset < len(caller); offset += frameSamples {
			if err := write(pcm16SamplesToBytes(caller[offset:min(offset+frameSamples, len(caller))])); err != nil {
				writeDone <- err
				return
			}
		}
		// The optional report marks this finite recording's end. It is a client
		// observation, not a Gemini VAD measurement, and does not change model input.
		if reportSpeechEnd {
			if err := wsutil.WriteClientText(conn, []byte(`{"type":"input.speech_stopped"}`)); err != nil {
				writeDone <- err
				return
			}
		}
		for range 100 {
			if err := write(make([]byte, frameSamples*2)); err != nil {
				writeDone <- err
				return
			}
		}
		writeDone <- nil
	}()
	audioBytes := 0
	readDone := make(chan error, 1)
	go func() {
		for {
			data, op, err := wsutil.ReadServerData(conn)
			if err != nil {
				readDone <- err
				return
			}
			if op == ws.OpBinary {
				audioBytes += len(data)
			}
		}
	}()
	// audioBytes is read only after stopping/joining the reader.
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	var summary, ready map[string]any
	transcript := ""
	cursor := 0
	for summary == nil || !strings.Contains(strings.ToLower(transcript), "orange") {
		select {
		case <-ctx.Done():
			t.Fatalf("timeout: transcript=%q summary=%v", transcript, summary)
		case err := <-writeDone:
			if err != nil {
				t.Fatal(err)
			}
			writeDone = nil
		case err := <-readDone:
			t.Fatalf("bridge ended early: %v", err)
		case <-ticker.C:
			events, next := parent.telemetry.StoredEvents(cursor)
			cursor = next
			for _, event := range events {
				if event.ThreadID != id {
					continue
				}
				switch event.Type {
				case "realtime.session_ready":
					_ = json.Unmarshal(event.Data, &ready)
				case "realtime.assistant":
					var d struct {
						Text string `json:"text"`
					}
					_ = json.Unmarshal(event.Data, &d)
					transcript += " " + d.Text
				case "realtime.utterance_timing":
					var d map[string]any
					_ = json.Unmarshal(event.Data, &d)
					if d["completion_status"] == "completed" {
						summary = d
					}
				case "realtime.error", "realtime.output_blocked":
					t.Fatalf("unexpected event: %s %s", event.Type, event.Data)
				}
			}
		}
	}
	_ = conn.Close()
	<-readDone
	if audioBytes == 0 {
		t.Fatal("no audio arrived through the actual bridge")
	}
	for _, field := range []string{"provider_audio_to_guard_release_ms", "guard_release_to_output_enqueue_ms", "output_enqueue_to_bridge_write_ms", "provider_audio_to_bridge_write_ms"} {
		if summary[field] == nil {
			t.Fatalf("missing %s: %v", field, summary)
		}
	}
	if !reportSpeechEnd && summary["speech_end_to_provider_audio_ms"] != nil {
		t.Fatalf("invented Gemini speech-end timing: %v", summary)
	}
	if reportSpeechEnd && (summary["speech_end_to_provider_audio_ms"] == nil || summary["speech_end_source"] != "client_reported_signal_received") {
		t.Fatalf("client signal missing/mislabeled: %v", summary)
	}
	if ready == nil || ready["connection_to_ready_ms"] == nil {
		t.Fatalf("missing session setup timing: %v", ready)
	}
	applied := ready["applied"].(map[string]any)
	vad := applied["turn_detection"].(map[string]any)["automaticActivityDetection"].(map[string]any)
	if vad["silenceDurationMs"] != float64(silence) {
		t.Fatalf("incorrect applied VAD: %v", applied)
	}
	if summary["dropped_audio_bytes"] != float64(0) || summary["guard_outcome"] != "transcript_prefix" {
		t.Fatalf("audio loss/guard regression: %v", summary)
	}
	t.Logf("silence_ms=%d transcript=%q audio_bytes=%d timing=%s", silence, transcript, audioBytes, mustJSON(t, summary))
}
