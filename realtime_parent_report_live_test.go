package core

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/joho/godotenv"
)

// Paid, isolated real voice session. No carrier, external tools or real booking.
// RUN_REALTIME_PARENT_REPORT_LIVE=1 GOWORK=off go test -run '^TestRealtimeParentReportOpenAILive$' -v -timeout 2m .
func TestRealtimeParentReportOpenAILive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_REALTIME_PARENT_REPORT_LIVE") != "1" {
		t.Skip("set RUN_REALTIME_PARENT_REPORT_LIVE=1 for a paid real voice session")
	}
	loadIntegrationEnv()
	_ = godotenv.Load("../.env")
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY is required")
	}
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	parent.config.path = filepath.Join(t.TempDir(), "config.json")
	parent.config.RealtimeEnabled = true
	parent.registry = NewToolRegistry("test")
	parent.execution = NewExecutionController(rootPausedExecutionConfig())
	rootGate, _ := startScopedGate(t, parent.execution, "main")
	waitForActiveThread(t, parent.execution, "main")
	voice := NewOpenAIRealtimeProvider(key)
	voice.models = map[ModelTier]string{ModelLarge: "gpt-live-1", ModelMedium: "gpt-live-1", ModelSmall: "gpt-live-1"}
	parent.provider = NewOpenAINativeProvider(key)
	parent.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": parent.provider}, default_: "openai", realtimeProviders: map[string]RealtimeProvider{"openai-realtime": voice}, realtimeDefault: "openai-realtime"}
	input := make(chan []byte, 8)
	output := make(chan RealtimeAudioFrame, 128)
	defer parent.Stop()
	started := time.Now()
	if err := parent.threads.SpawnWithOpts("live-parent-report", "Parlez en français. Dites exactement : Bonjour, comment puis-je vous aider ? Puis attendez. Aucun outil, aucune réservation, ne terminez pas la conversation.", []string{}, SpawnOpts{Realtime: true, ProviderName: "openai-realtime", Voice: "brise", AudioIn: input, AudioOut: output, InitialMessage: "Dites bonjour maintenant.", Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	parent.threads.mu.RLock()
	thread := parent.threads.threads["live-parent-report"]
	parent.threads.mu.RUnlock()
	defer parent.threads.KillWithReason(thread.ID, "stopped")
	// An actual bridge supplies continuous PCM. Silence is kept intact; observing
	// voiced output below is a measurement, never an audio gate.
	thread.Realtime.audioBridgeConnected()
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	deadline := time.After(50 * time.Second)
	var detector realtimeAcousticDetector
	voiced := false
	var bytes int
	var firstVoice time.Duration
	for {
		select {
		case <-deadline:
			t.Fatalf("no voiced greeting and transcript; audio_bytes=%d voiced=%v", bytes, voiced)
		case <-ticker.C:
			select {
			case input <- make([]byte, 960):
			default:
			}
		case frame := <-output:
			bytes += len(frame.Audio)
			if detector.observe(frame.Audio, AudioPCM16, 24000, 180) && !voiced {
				voiced = true
				firstVoice = time.Since(started)
			}
			frame.written(true)
		}
		transcript := false
		for _, m := range thread.Realtime.executionSnapshot() {
			if m.RealtimeTranscript != nil && m.Role == "assistant" && strings.Contains(strings.ToLower(m.Content), "bonjour") {
				transcript = true
			}
		}
		if voiced && transcript {
			break
		}
	}
	ended := time.Now()
	parent.threads.KillWithReason(thread.ID, "caller_done")
	event, report := readParentReport(t, parent)
	if report.Reason != "caller_done" || report.Outcome != "ended" || len(report.Recent) == 0 {
		t.Fatalf("incomplete terminal report: %+v", report)
	}
	if parent.threads.Count() != 0 {
		t.Fatal("live thread was not cleaned up")
	}
	if !parent.execution.ShouldGateThread("main", ExecutionPhaseIterationStart) || parent.execution.Status().WaitingCount != 1 {
		t.Fatal("report bypassed parent pause")
	}
	thread.reportRealtimeTermination()
	again, _ := readParentReport(t, parent)
	if again.ID != event.ID {
		t.Fatal("duplicate final report")
	}
	restored := &Config{path: parent.config.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	if len(restored.getMainEvents()) != 1 {
		t.Fatal("real live parent report did not survive reload")
	}
	parent.execution.CancelThread("main")
	requireGateResult(t, rootGate, false)
	t.Logf("model=gpt-live-1 voice=brise audio_bytes=%d first_voiced_ms=%d hangup_to_durable_parent_ms=%d reports=1 parent_paused=true", bytes, firstVoice.Milliseconds(), time.Since(ended).Milliseconds())
}
