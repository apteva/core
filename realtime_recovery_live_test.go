package core

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Paid opt-in acceptance: real PCM speech, silence, planned renewal,
// fresh-history fallback, MCP booking, and a separate naturally idle thread.
// Provider renewals near the beginning are deliberately injected and labeled;
// subsequent closes/GoAway notices during the soak are natural provider events.
func TestGoogleRealtimeLiveRecoverySoak(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_GOOGLE_REALTIME_RECOVERY_SOAK") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_RECOVERY_SOAK=1 for the paid 12-minute recovery soak")
	}
	key := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	if key == "" {
		t.Fatal("GOOGLE_API_KEY is required")
	}
	duration := 12 * time.Minute
	if raw := os.Getenv("REALTIME_RECOVERY_SOAK_DURATION"); raw != "" {
		var err error
		duration, err = time.ParseDuration(raw)
		if err != nil || duration < time.Minute {
			t.Fatal("invalid diagnostic soak duration")
		}
	}
	t.Chdir(t.TempDir())
	google := NewGoogleRealtimeProvider(key)
	scenario, _ := newReceptionistScenario("fr")
	recall := "Pouvez-vous me rappeler le jour et l'heure de mon rendez-vous déjà réservé ?"
	texts := []string{scenario.CallerSteps[0].Text, scenario.CallerSteps[1].Text, recall, "Répétez lentement tous les nombres de un à trente, puis attendez.", "Pour repère, souvenez-vous du mot abricot. Dites seulement compris.", "Quel est le mot repère que je vous ai demandé de retenir ?"}
	clips := make([][]byte, len(texts))
	for i, text := range texts {
		fixture := ""
		if dir := os.Getenv("REALTIME_RECOVERY_FIXTURE_DIR"); dir != "" {
			if err := os.MkdirAll(dir, 0700); err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256([]byte(google.Models()[ModelLarge] + ":Aoede:" + text))
			fixture = filepath.Join(dir, fmt.Sprintf("%x.s16le", sum))
			if pcm, err := os.ReadFile(fixture); err == nil && len(pcm) > 0 && len(pcm)%2 == 0 {
				clips[i] = pcm
				continue
			}
		}
		clips[i] = pcm16SamplesToBytes(synthesizeGoogleRealtimeSpeech(t, google, "Aoede", text))
		if fixture != "" {
			if err := os.WriteFile(fixture, clips[i], 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	recorder := &receptionistMCPRecorder{}
	mcp := newReceptionistMCPServer(t, recorder, scenario)
	defer mcp.Close()
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Coordinate an isolated voice recovery test.", RealtimeEnabled: true,
		Providers: []ProviderConfig{{Name: "google-realtime", Default: true}}, MCPServers: []MCPServerConfig{{Name: "reception", Transport: "http", URL: mcp.URL + "/mcp"}}}
	parent := NewThinker("", &inertRealtimeTextProvider{}, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()
	defer func() {
		for _, s := range parent.mcpServers {
			s.Close()
		}
	}()
	const id = "google-recovery-soak"
	audioIn := make(chan []byte, 64)
	audioOut := make(chan RealtimeAudioFrame, 256)
	control := make(chan string, 16)
	directive := scenario.Directive + `
Cette conversation comprend aussi deux exercices de validation autorisés, qui font partie de votre mission : mémoriser puis restituer un mot repère, et compter lentement de un à trente à la demande de la personne. Acceptez ces demandes, sans rediriger vers la prise de rendez-vous. Quand on vous demande de retenir le mot, répondez seulement « Compris ». Quand on vous le redemande, dites ce mot.
Si la personne redemande le jour ou l'heure d'un rappel déjà réservé, répondez avec les résultats déjà connus, sans nouvelle vérification ni réservation.`
	if err := parent.threads.SpawnWithOpts(id, directive, []string{receptionAvailabilityTool, receptionBookingTool}, SpawnOpts{
		Realtime: true, Ephemeral: true, ProviderName: "google-realtime", Voice: "Kore", MCPNames: []string{"reception"}, AudioIn: audioIn, AudioOut: audioOut, AudioControl: control, InitialMessage: scenario.InitialMessage,
	}); err != nil {
		t.Fatal(err)
	}
	parent.threads.mu.RLock()
	rt := parent.threads.threads[id].Realtime
	parent.threads.mu.RUnlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	capture := newReceptionistAudioCapture()
	audioDone := make(chan struct{})
	go func() {
		defer close(audioDone)
		for {
			select {
			case frame := <-audioOut:
				capture.add(frame)
				parent.threads.realtimePlaybackProgress(id, frame.ItemID, frame.AudioEndMS)
			case <-control:
			case <-ctx.Done():
				return
			}
		}
	}()
	defer func() { cancel(); <-audioDone }()
	var paused atomic.Bool
	feed := make(chan []byte, 1)
	feederDone := make(chan struct{})
	go func() {
		defer close(feederDone)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		var pcm []byte
		for {
			select {
			case pcm = <-feed:
			case <-ctx.Done():
				return
			case <-ticker.C:
				if paused.Load() {
					continue
				}
				chunk := make([]byte, 4800)
				n := copy(chunk, pcm)
				pcm = pcm[n:]
				select {
				case audioIn <- chunk:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	defer func() { cancel(); <-feederDone }()
	trace := &receptionistLiveTrace{}
	started := time.Now()
	parent.threads.realtimeBridgeConnected(id)
	waitForReceptionistStage(t, parent, id, trace, "greeting", func() bool { return len(trace.assistantTurns) == 1 })
	speak := func(index int, wantAvailability, wantBookings int) {
		before := len(trace.assistantTurns)
		trace.lines = append(trace.lines, "CALLER PCM: "+texts[index])
		sent := time.Now()
		feed <- clips[index]
		waitForReceptionistStage(t, parent, id, trace, "PCM caller response", func() bool {
			spoken := canonicalSpokenText(strings.Join(trace.assistantTurns[before:], " "))
			contentReady := index != 2 || (strings.Contains(spoken, "LUNDI") && containsAny(spoken, "16", "SEIZE"))
			if index == 5 {
				contentReady = strings.Contains(spoken, "ABRICOT")
			}
			if index == 4 {
				contentReady = strings.Contains(spoken, "COMPRIS")
			}
			return contentReady && len(trace.assistantTurns) > before && countReceptionToolResults(trace.toolResults, receptionAvailabilityTool) >= wantAvailability && countReceptionToolResults(trace.toolResults, receptionBookingTool) >= wantBookings
		})
		spoken := canonicalSpokenText(strings.Join(trace.assistantTurns[before:], " "))
		if index == 2 && (!strings.Contains(spoken, "LUNDI") || !containsAny(spoken, "16", "SEIZE")) {
			t.Fatalf("reservation context was lost: %s", spoken)
		}
		if index == 5 && !strings.Contains(spoken, "ABRICOT") {
			t.Fatalf("renewal lost caller context: %s", spoken)
		}
		if countReceptionToolResults(trace.toolResults, receptionAvailabilityTool) != wantAvailability || countReceptionToolResults(trace.toolResults, receptionBookingTool) != wantBookings {
			t.Fatal("unexpected duplicate tool result")
		}
		t.Logf("PCM turn %d answered after %.2fs", index, time.Since(sent).Seconds())
	}
	speak(4, 0, 0)
	// A provider GoAway triggers planned renewal after the current turn.
	paused.Store(true)
	old := rt.currentSession().(*googleRealtimeSession)
	waitRecoveryLive(t, 10*time.Second, func() bool { return !rt.responseInProgress() && rt.playbackSettled() })
	old.emitControl(RealtimeEvent{Type: RealtimeEventSessionExpiring, TimeLeft: 5 * time.Second})
	waitRecoveryLive(t, 15*time.Second, func() bool { return rt.currentSession() != nil && rt.currentSession() != old })
	restored := false
	events, _ := parent.telemetry.StoredEvents(0)
	for _, e := range events {
		if e.ThreadID == id && e.Type == "realtime.session_opened" {
			var d map[string]any
			_ = json.Unmarshal(e.Data, &d)
			if d["restored"] == true {
				restored = true
			}
		}
	}
	if !restored {
		t.Fatal("planned renewal did not restore the conversation")
	}
	trace.lines = append(trace.lines, "TEST: forced planned renewal; verified history restoration succeeded")
	paused.Store(false)
	speak(5, 0, 0)
	speak(0, 1, 0)
	speak(1, 1, 1)
	paused.Store(true)
	waitRecoveryLive(t, 10*time.Second, func() bool { return !rt.responseInProgress() && rt.playbackSettled() })
	old = rt.currentSession().(*googleRealtimeSession)
	_ = old.Close()
	waitRecoveryLive(t, 15*time.Second, func() bool { return rt.currentSession() != nil && rt.currentSession() != old })
	trace.lines = append(trace.lines, "TEST: forced provider disconnect; verified history and tool results restored")
	paused.Store(false)
	speak(2, 1, 1)
	// Interrupt a deliberately long spoken answer with a new PCM question.
	waitRecoveryLive(t, 10*time.Second, func() bool { return !rt.responseInProgress() && rt.playbackSettled() })
	_, beforeAudio := capture.snapshot()
	feed <- clips[3]
	waitRecoveryLive(t, 30*time.Second, func() bool { _, n := capture.snapshot(); return n > beforeAudio+32000 })
	speak(2, 1, 1)
	interrupted := false
	interruptionEvents, _ := parent.telemetry.StoredEvents(0)
	for _, event := range interruptionEvents {
		if event.ThreadID == id && event.Type == "realtime.playback_interrupted" {
			interrupted = true
		}
	}
	if !interrupted {
		t.Fatal("PCM barge-in did not interrupt playback")
	}
	trace.lines = append(trace.lines, "TEST: PCM caller interrupted a spoken answer; booking context retained")
	// A second thread receives no input until after two observed idle windows.
	idleThinker := newTestThinker()
	idleThinker.messages = []Message{{Role: "system", Content: "Answer briefly in French. When asked which number follows 41, answer 42. Wait for input; do not greet."}}
	idle, err := startRealtimeThinker(context.Background(), idleThinker, google, "Kore", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	idleDone := make(chan struct{})
	go func() { defer close(idleDone); idle.Run() }()
	defer func() { idle.cancel(); <-idleDone }()
	idleStarted := time.Now()
	idleWoken := false
	nextTurn := time.Now().Add(time.Minute)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for time.Since(started) < duration {
		<-ticker.C
		trace.collect(parent, id)
		if trace.err != nil {
			t.Fatalf("soak error: %v", trace.err)
		}
		if time.Now().After(nextTurn) {
			speak(2, 1, 1)
			nextTurn = time.Now().Add(time.Minute)
			t.Logf("soak %.0fs: conversation and booking context intact", time.Since(started).Seconds())
		}
		if !idleWoken && time.Since(idleStarted) >= 6*time.Minute {
			idle.lifecycleMu.Lock()
			generation := idle.sessionGeneration
			idle.lifecycleMu.Unlock()
			idle.transcriptMu.Lock()
			messages := len(idle.messages)
			idle.transcriptMu.Unlock()
			if generation != 1 || idle.currentSession() != nil || messages != 1 {
				t.Fatal("idle thread reopened or spoke without demand")
			}
			idle.bus.Publish(Event{Type: EventInbox, To: "main", Text: "Quel nombre vient après quarante et un ?"})
			waitRecoveryLive(t, 30*time.Second, func() bool {
				idle.transcriptMu.Lock()
				defer idle.transcriptMu.Unlock()
				for _, m := range idle.messages {
					if m.Role == "assistant" && (strings.Contains(m.Content, "42") || strings.Contains(strings.ToLower(m.Content), "quarante-deux")) {
						return true
					}
				}
				return false
			})
			idleWoken = true
			t.Log("idle thread waited over six minutes, then woke and answered 42")
		}
	}
	trace.collect(parent, id)
	if len(recorder.snapshot()) != 2 {
		t.Fatalf("external tool executions=%d, want one availability and one booking", len(recorder.snapshot()))
	}
	greetings := 0
	for _, line := range trace.assistantTurns {
		if strings.Contains(strings.ToLower(line), "commerciaux") {
			greetings++
		}
	}
	if greetings != 1 {
		t.Fatalf("greeting count=%d, want 1", greetings)
	}
	if duration >= 12*time.Minute && !idleWoken {
		t.Fatal("idle wake was not validated")
	}
	userTurns := 0
	var lifecycle []map[string]any
	events, _ = parent.telemetry.StoredEvents(0)
	for _, e := range events {
		if e.ThreadID != id {
			continue
		}
		if e.Type == "realtime.user" {
			userTurns++
		}
		if e.Type == "realtime.session_opened" || e.Type == "realtime.session_closed" || e.Type == "realtime.reconnect_planned" {
			var d map[string]any
			_ = json.Unmarshal(e.Data, &d)
			d["type"] = e.Type
			lifecycle = append(lifecycle, d)
		}
	}
	if userTurns < 3 {
		t.Fatalf("only %d caller speech transcripts", userTurns)
	}
	segments, audioBytes := capture.snapshot()
	if dir := os.Getenv("REALTIME_RECEPTIONIST_ARTIFACT_DIR"); dir != "" {
		if _, err := writeReceptionistArtifacts(dir, "google-recovery-soak", "fr", "Kore", trace.lines, segments); err != nil {
			t.Fatal(err)
		}
		summary, _ := json.MarshalIndent(map[string]any{"seconds": time.Since(started).Seconds(), "input": "synthetic spoken PCM and silence", "user_transcripts": userTurns, "assistant_turns": len(trace.assistantTurns), "greetings": greetings, "external_tool_calls": len(recorder.snapshot()), "audio_bytes": audioBytes, "idle_wake_verified": idleWoken, "lifecycle": lifecycle}, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, "recovery-summary.json"), summary, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("recovery soak completed: %.1fs, %d caller transcripts, %d audio bytes, exactly two external tool calls", time.Since(started).Seconds(), userTurns, audioBytes)
}

func waitRecoveryLive(t *testing.T, timeout time.Duration, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !ready() {
		if time.Now().After(deadline) {
			t.Fatal(fmt.Sprintf("live recovery condition timed out after %v", timeout))
		}
		time.Sleep(20 * time.Millisecond)
	}
}
