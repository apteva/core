package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Paid, opt-in end-to-end receptionist conversation. Caller speech is generated
// separately, streamed as audio, and never supplied as text to the receptionist.
// Both appointment operations are local fixtures; no real booking is created.
// RUN_OPENAI_LIVE_RECEPTIONIST=1 OPENAI_LIVE_AUDIO_DIR=... go test -run '^TestOpenAILiveFrenchReceptionistLive$' -v -timeout 6m .
func TestOpenAILiveFrenchReceptionistLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_OPENAI_LIVE_RECEPTIONIST") != "1" {
		t.Skip("set RUN_OPENAI_LIVE_RECEPTIONIST=1 for paid receptionist audio verification")
	}
	loadIntegrationEnv()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not configured")
	}
	callerTexts := []string{
		"Bonjour, je voudrais prendre rendez-vous jeudi après-midi, s'il vous plaît.",
		"Oui, jeudi à quinze heures trente, c'est parfait. Vous pouvez réserver, je m'appelle Camille Martin.",
		"Merci beaucoup, au revoir !",
	}
	callerAudio := make([][]byte, len(callerTexts))
	for i, text := range callerTexts {
		callerAudio[i] = renderLiveCallerText(t, key, text)
	}
	mode := os.Getenv("OPENAI_LIVE_RECEPTIONIST_MODE")
	if mode == "" {
		mode = "provider"
	}
	for _, voice := range []string{"brise", "sillage"} {
		t.Run(voice, func(t *testing.T) {
			rt, _ := completionFixture(t)
			rt.Thinker.threadID = "main"
			rt.Thinker.directive = `Vous êtes une standardiste qui accueille les appels et aide à prendre rendez-vous. Restez entièrement générique : parlez simplement de rendez-vous, sans inventer de motif, de secteur d'activité, de poste, d'entretien, de consultation ou de prestation. Ne mentionnez aucune marque, aucun nom d'entreprise ni aucun nom de cabinet. Parlez en français de France avec une voix chaleureuse, souriante, professionnelle et naturelle. Vouvoyez. Soyez concise, une ou deux phrases par tour, et posez une seule question à la fois.
À l'ouverture, dites : « Bonjour, tous nos conseillers sont actuellement occupés. Je peux vous aider à prendre rendez-vous. Quel jour vous conviendrait ? »
Après la demande de rendez-vous, consultez appointment_slots une seule fois, puis proposez le créneau réellement disponible et demandez si cela convient ainsi que le nom de la personne. Ne donnez aucune disponibilité inventée.
Quand la personne confirme le créneau et donne son nom, appelez appointment_book exactement une fois avec le slot_id retourné et le nom entendu. Ne dites jamais que c'est confirmé avant le succès de cette opération. Après le succès, confirmez brièvement le rendez-vous en rappelant le nom, le jour et l'heure, puis demandez si vous pouvez aider pour autre chose.
Quand la personne vous remercie et dit au revoir, souhaitez-lui une bonne journée et dites au revoir. Ne terminez pas le thread vous-même.
Ne prononcez jamais les noms des outils, identifiants, JSON, API, threads ou mécanismes internes. Ne jouez jamais le rôle de l'appelant. Aucun téléphone, courriel ou renseignement supplémentaire n'est requis pour cette démonstration.`
			rt.messages[0].Content = rt.Thinker.directive
			rt.setInitialMessage("Bonjour, tous nos conseillers sont actuellement occupés. Je peux vous aider à prendre rendez-vous. Quel jour vous conviendrait ?")
			backend := NewOpenAINativeProvider(key)
			applyModelOverrides(backend, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
			rt.Thinker.provider = backend
			rt.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": backend}, default_: "openai", order: []string{"openai"}}
			rt.provider = NewOpenAIRealtimeProvider(key)
			rt.opts.Model, rt.opts.Voice = "gpt-live-1", voice
			rt.capabilities = capabilitiesForRealtime(rt.provider, rt.opts.Model)
			rt.config.Providers = []ProviderConfig{{Name: "openai", RealtimeExecution: RealtimeExecutionConfig{Mode: mode}}}
			rt.registry = NewToolRegistry("")
			rt.toolAllowlist = map[string]bool{"appointment_slots": true, "appointment_book": true}
			var lookups, bookings, phase atomic.Int32
			rt.registry.Register(&ToolDef{Name: "appointment_slots", Description: "Read the available appointment slot for the caller's preferred day. Use once; never invent availability.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"day": map[string]any{"type": "string"}}, "required": []string{"day"}}, Handler: func(args map[string]string) ToolResponse {
				lookups.Add(1)
				return ToolResponse{Text: `{"slots":[{"slot_id":"demo-thursday-1530","day":"jeudi","time":"15:30"}],"booking_created":false}`}
			}})
			rt.registry.Register(&ToolDef{Name: "appointment_book", Description: "Book the selected slot only after the caller confirms it and provides their name. This is a local demonstration fixture.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"slot_id": map[string]any{"type": "string"}, "name": map[string]any{"type": "string"}}, "required": []string{"slot_id", "name"}}, Handler: func(args map[string]string) ToolResponse {
				if phase.Load() < 2 || args["slot_id"] != "demo-thursday-1530" || !strings.EqualFold(strings.TrimSpace(args["name"]), "Camille Martin") {
					t.Errorf("invalid booking: phase=%d args=%v", phase.Load(), args)
					return ToolResponse{Text: `{"status":"rejected"}`}
				}
				bookings.Add(1)
				return ToolResponse{Text: `{"status":"confirmed","name":"Camille Martin","day":"jeudi","time":"15:30","appointment_id":"demo-booking-1"}`}
			}})
			ctx, cancel := context.WithTimeout(context.Background(), 140*time.Second)
			defer cancel()
			rt.cancel()
			rt.ctx, rt.cancel = context.WithCancel(ctx)
			if err := rt.openSession(false); err != nil {
				t.Fatal(err)
			}
			done := runCompletionFixture(rt)
			defer func() {
				rt.Thinker.Stop()
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Error("receptionist failed to stop")
				}
			}()
			// The simulated microphone remains active, including silence while
			// the receptionist speaks. Caller clips are paced in real time.
			clips := make(chan []byte, 1)
			micErrors := make(chan error, 1)
			micCtx, stopMic := context.WithCancel(ctx)
			defer stopMic()
			go func() {
				tick := time.NewTicker(20 * time.Millisecond)
				defer tick.Stop()
				var clip []byte
				position := 0
				for {
					select {
					case <-micCtx.Done():
						return
					case clip = <-clips:
						position = 0
					case <-tick.C:
						frame := make([]byte, 960)
						if position < len(clip) {
							copy(frame, clip[position:min(position+len(frame), len(clip))])
							position += len(frame)
						}
						if session := rt.currentSession(); session != nil {
							if err := session.SendAudio(frame); err != nil {
								micErrors <- err
								return
							}
						}
					}
				}
			}()
			var dialogue, assistantPCM []byte
			var transcript strings.Builder
			cursor := 0
			spoken := ""
			stageBytes := 0
			lastAudio := time.Now()
			readyAt := time.Time{}
			reasons := map[string]bool{}
			callerTurns := 0
			tick := time.NewTicker(20 * time.Millisecond)
			defer tick.Stop()
			for {
				select {
				case <-ctx.Done():
					t.Fatalf("receptionist timeout: phase=%d lookups=%d bookings=%d spoken=%s", phase.Load(), lookups.Load(), bookings.Load(), spoken)
				case err := <-micErrors:
					t.Fatal(err)
				case frame := <-rt.audioOut:
					dialogue = append(dialogue, frame.Audio...)
					assistantPCM = append(assistantPCM, frame.Audio...)
					stageBytes += len(frame.Audio)
					lastAudio = time.Now()
					frame.written(true)
				case <-tick.C:
					events, next := rt.telemetry.StoredEvents(cursor)
					cursor = next
					for _, e := range events {
						if e.Type == "realtime.error" || e.Type == "realtime.execution_error" || e.Type == "llm.error" {
							t.Fatalf("%s: %s", e.Type, e.Data)
						}
						if e.Type == "realtime.assistant" || e.Type == "realtime.user" || e.Type == "tool.call" || e.Type == "tool.result" {
							transcript.WriteString(e.Type + " " + string(e.Data) + "\n")
						}
						if e.Type == "realtime.assistant" {
							var data struct {
								Text string `json:"text"`
							}
							if err := json.Unmarshal(e.Data, &data); err != nil {
								t.Fatal(err)
							}
							spoken += " " + strings.ToLower(data.Text)
						}
						if e.Type == "realtime.user" {
							callerTurns++
						}
						if e.Type == "tool.call" {
							var data map[string]any
							if json.Unmarshal(e.Data, &data) == nil && data["reason_source"] == "model" && data["reason"] != "" {
								reasons[data["name"].(string)] = true
							}
						}
						if strings.HasPrefix(e.Type, "realtime.execution") || e.Type == "tool.call" {
							t.Logf("%s %s", e.Type, e.Data)
						}
					}
					ready := false
					switch phase.Load() {
					case 0:
						ready = strings.Contains(spoken, "occup") && strings.Contains(spoken, "rendez")
					case 1:
						if bookings.Load() != 0 {
							t.Fatal("booked before caller consent")
						}
						// A natural offer can be a statement or a question. The caller
						// volunteers both consent and their name in the following turn.
						ready = lookups.Load() > 0 && (strings.Contains(spoken, "trente") || strings.Contains(spoken, "30"))
					case 2:
						ready = bookings.Load() > 0 && (strings.Contains(spoken, "confirm") || strings.Contains(spoken, "réservé")) && strings.Contains(spoken, "jeudi") && (strings.Contains(spoken, "trente") || strings.Contains(spoken, "30"))
					case 3:
						ready = strings.Contains(spoken, "au revoir") || strings.Contains(spoken, "bonne journée")
					}
					if !ready || stageBytes == 0 {
						continue
					}
					if readyAt.IsZero() {
						readyAt = time.Now()
					}
					// Continuous speech has no provider-final turn event. Leave a
					// generous quiet window before the simulated caller responds.
					if time.Since(readyAt) < 4*time.Second || time.Since(lastAudio) < 4*time.Second {
						continue
					}
					t.Logf("voice=%s phase=%d heard=%s", voice, phase.Load(), spoken)
					for _, technical := range []string{"appointment_", "slot_id", "callto:", "horizon", "entretien", "consultation", "recrutement"} {
						if strings.Contains(spoken, technical) {
							t.Fatalf("technical narration: %s", spoken)
						}
					}
					if phase.Load() == 3 {
						if lookups.Load() != 1 || bookings.Load() != 1 || !reasons["appointment_slots"] || !reasons["appointment_book"] || callerTurns == 0 {
							t.Fatalf("invalid flow: lookups=%d bookings=%d reasons=%v caller_transcripts=%d", lookups.Load(), bookings.Load(), reasons, callerTurns)
						}
						if dir := os.Getenv("OPENAI_LIVE_AUDIO_DIR"); dir != "" {
							if err := os.MkdirAll(dir, 0700); err != nil {
								t.Fatal(err)
							}
							base := filepath.Join(dir, "receptionist-"+voice)
							for path, content := range map[string][]byte{base + ".pcm": dialogue, base + "-assistant.pcm": assistantPCM, base + ".transcript.txt": []byte(transcript.String())} {
								if err := os.WriteFile(path, content, 0600); err != nil {
									t.Fatal(err)
								}
							}
							t.Logf("saved receptionist dialogue: %s.pcm (%d bytes); mode=%s voice=%s", base, len(dialogue), mode, voice)
						}
						return
					}
					clip := callerAudio[int(phase.Load())]
					dialogue = append(dialogue, make([]byte, 12000)...)
					dialogue = append(dialogue, clip...)
					dialogue = append(dialogue, make([]byte, 12000)...)
					phase.Add(1)
					clips <- clip
					spoken, stageBytes, readyAt = "", 0, time.Time{}
				}
			}
		})
	}
}
