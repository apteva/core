package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/joho/godotenv"
)

func voiceExecutionTestKey(t *testing.T) string {
	t.Helper()
	if testing.Short() || os.Getenv("RUN_VOICE_EXECUTION_LIVE") != "1" {
		t.Skip("set RUN_VOICE_EXECUTION_LIVE=1 for paid OpenAI calls")
	}
	loadIntegrationEnv()
	if os.Getenv("OPENAI_API_KEY") == "" {
		_ = godotenv.Load("../.env")
	}
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY required")
	}
	return key
}

const bookingInstructions = `Parlez uniquement français. Aidez à prendre un rendez-vous. Utilisez book pour réserver, uniquement avec une date/heure explicite et le nom complet. Ne choisissez jamais une heure à la place du client. Si l'heure manque, demandez-la sans réserver. Pour une réservation confirmée, annoncez les faits vérifiés : date, heure et référence. Ne déclarez jamais une réservation réussie si l'outil échoue ou si son statut n'est pas confirmed. Si le client demande aussi une confirmation par courriel, appelez email après book avant de terminer. N'inventez aucun résultat et ne répétez pas une opération déjà réussie. N'annoncez aucun nom d'outil. Ne terminez pas la conversation. Les outils déjà présentés peuvent être appelés directement.`
const bookingQuestion = `Réservez pour Camille Durand le mardi 3 novembre 2026 à 14 heures, fuseau Europe/Paris. Je confirme ce choix. Annoncez la date, l'heure et la référence de réservation.`

func registerBookingFixture(t *testing.T, th *Thinker, fail bool, calls, emails *atomic.Int32) {
	t.Helper()
	th.registry = NewToolRegistry("")
	th.registry.Register(&ToolDef{Name: "book", Description: "Create the requested appointment. Requires full name and an ISO8601 date with timezone offset. The result contains status, date and reference.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}, "date": map[string]any{"type": "string"}}, "required": []string{"name", "date"}}, Handler: func(args map[string]string) ToolResponse {
		calls.Add(1)
		if args["name"] != "Camille Durand" || args["date"] != "2026-11-03T14:00:00+01:00" {
			return ToolResponse{Text: `{"status":"invalid","message":"incorrect booking details"}`, IsError: true}
		}
		if fail {
			return ToolResponse{Text: `{"status":"unavailable","message":"Ce créneau est indisponible. Demandez une autre date, sans réserver."}`, IsError: true}
		}
		return ToolResponse{Text: `{"status":"confirmed","date":"2026-11-03T14:00:00+01:00","reference":"abricot","private":"internal test secret, never disclose"}`}
	}})
	th.registry.Register(&ToolDef{Name: "email", Description: "Send the confirmed appointment by email after a successful booking. Only call if the client requested email and supplied the address.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"address": map[string]any{"type": "string"}, "reference": map[string]any{"type": "string"}}, "required": []string{"address", "reference"}}, Handler: func(args map[string]string) ToolResponse {
		emails.Add(1)
		if args["address"] != "camille@example.test" || args["reference"] != "abricot" || calls.Load() != 1 {
			return ToolResponse{Text: "invalid email request", IsError: true}
		}
		return ToolResponse{Text: `{"status":"sent","reference":"abricot"}`}
	}})
}

// Real Luna runs the ordinary managed worker and dispatches only local fixtures.
// Cover accuracy before comparing latency; no real appointment/email is made.
func TestVoiceExecutionLunaBookingAccuracyLive(t *testing.T) {
	key := voiceExecutionTestKey(t)
	for _, name := range []string{"confirmed", "missing_time", "unavailable", "followup_email"} {
		t.Run(name, func(t *testing.T) {
			parent := newTestThinkerFull()
			parent.session = NewSession(t.TempDir(), "main")
			defer parent.Stop()
			provider := NewOpenAINativeProvider(key).(*OpenAINativeProvider).WithExecutionModel("gpt-6-luna")
			parent.provider = provider
			var calls, emails atomic.Int32
			registerBookingFixture(t, parent, name == "unavailable", &calls, &emails)
			if err := parent.threads.SpawnWithOpts("worker", bookingInstructions, nil, SpawnOpts{WorkMode: true, DeferRun: true, Reasoning: ReasoningNone}); err != nil {
				t.Fatal(err)
			}
			worker := parent.threads.threads["worker"].Thinker
			worker.executionPolicy = &RealtimeExecutionConfig{PreloadTools: []string{"book", "email"}, ResultTools: []DirectToolResultPolicy{bookingResultPolicy()}}
			if err := parent.threads.Start("worker"); err != nil {
				t.Fatal(err)
			}
			question := bookingQuestion
			if name == "missing_time" {
				question = "Je suis Camille Durand. Réservez le 3 novembre 2026. Je n'ai pas encore choisi l'heure."
			}
			if name == "followup_email" {
				question += " Puis envoyez-moi la confirmation à camille@example.test. Faites les deux opérations."
			}
			if err := parent.threads.QueueWork("worker", ThreadWork{ID: "booking", Revision: 1, Origin: "main", Kind: "request"}, question, nil); err != nil {
				t.Fatal(err)
			}
			deadline := time.After(70 * time.Second)
			for {
				select {
				case <-deadline:
					t.Fatal("booking verification timed out")
				case <-parent.sub.Wake:
					for _, e := range parent.sub.DrainTargeted() {
						if e.Work != nil && e.Work.Kind == "result" {
							parent.threads.KillAll()
							if e.Work.Status != "completed" {
								t.Fatalf("result status=%s text=%s", e.Work.Status, e.Text)
							}
							want := int32(1)
							if name == "missing_time" {
								want = 0
							}
							if calls.Load() != want {
								t.Fatalf("booking calls=%d want=%d result=%s", calls.Load(), want, e.Text)
							}
							wantMail := int32(0)
							if name == "followup_email" {
								wantMail = 1
							}
							if emails.Load() != wantMail {
								t.Fatalf("email calls=%d want=%d", emails.Load(), wantMail)
							}
							if (name == "confirmed" || name == "followup_email") && !strings.Contains(strings.ToLower(e.Text), "abricot") {
								t.Fatalf("missing verified booking reference: %s", e.Text)
							}
							events, _ := parent.telemetry.StoredEvents(0)
							direct := 0
							requests := 0
							for _, event := range events {
								if event.Type == "thread.work_direct_result" {
									direct++
								}
								if event.Type == "llm.request.started" {
									requests++
								}
							}
							if name == "confirmed" && direct != 1 {
								t.Fatal("confirmed booking did not use shortcut")
							}
							if name != "confirmed" && direct != 0 {
								t.Fatal("unsafe shortcut for " + name)
							}
							t.Logf("case=%s model=gpt-6-luna reasoning=none booking_calls=%d email_calls=%d requests=%d direct_receipts=%d result=%s", name, calls.Load(), emails.Load(), requests, direct, e.Text)
							return
						}
					}
				}
			}
		})
	}
}

// Matched real GPT Live/Brise sessions compare backend delivery and verified
// confirmation-caption arrival. Input is the same typed caller request with a
// paced PCM silence clock. This does not measure PSTN/handset playback latency.
func TestVoiceExecutionLiveLatencyComparison(t *testing.T) {
	key := voiceExecutionTestKey(t)
	trials := 3
	providerOnly := os.Getenv("VOICE_EXECUTION_PROVIDER") == "1"
	if os.Getenv("VOICE_EXECUTION_SMOKE") == "1" || providerOnly {
		trials = 1
	}
	for trial := 0; trial < trials; trial++ {
		variants := []string{"sol_baseline", "luna_continuation", "luna_receipt"}
		if trial%2 == 1 {
			variants = []string{"luna_receipt", "luna_continuation", "sol_baseline"}
		}
		if providerOnly {
			variants = []string{"luna_provider"}
		}
		for _, variant := range variants {
			t.Run(fmt.Sprintf("%s_%d", variant, trial+1), func(t *testing.T) {
				rt, _ := completionFixture(t)
				rt.Thinker.threadID = "main"
				rt.Thinker.directive = bookingInstructions
				rt.messages[0].Content = bookingInstructions
				backend := NewOpenAINativeProvider(key)
				applyModelOverrides(backend, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
				rt.Thinker.provider = backend
				rt.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": backend}, default_: "openai", order: []string{"openai"}}
				rt.provider = NewOpenAIRealtimeProvider(key)
				rt.opts.Model = "gpt-live-1"
				rt.opts.Voice = "brise"
				rt.capabilities = capabilitiesForRealtime(rt.provider, rt.opts.Model)
				var calls, emails atomic.Int32
				registerBookingFixture(t, rt.Thinker, false, &calls, &emails)
				policy := RealtimeExecutionConfig{PreloadTools: []string{"book", "email"}}
				if variant == "luna_provider" {
					policy.Mode = "provider"
				}
				if variant == "sol_baseline" {
					policy.Model = "large"
					policy.Reasoning = "auto"
				}
				if variant == "luna_receipt" {
					policy.ResultTools = []DirectToolResultPolicy{bookingResultPolicy()}
				}
				rt.config.Providers = []ProviderConfig{{Name: "openai", RealtimeExecution: policy}}
				ctx, cancel := context.WithTimeout(context.Background(), 70*time.Second)
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
					case <-time.After(3 * time.Second):
						t.Error("live worker failed to stop")
					}
				}()
				audioCtx, stopAudio := context.WithCancel(ctx)
				defer stopAudio()
				go func() {
					tick := time.NewTicker(20 * time.Millisecond)
					defer tick.Stop()
					for {
						select {
						case <-audioCtx.Done():
							return
						case <-tick.C:
							if s := rt.currentSession(); s != nil {
								_ = s.SendAudio(make([]byte, 960))
							}
						}
					}
				}()
				start := time.Now()
				rt.bus.Publish(Event{Type: EventInbox, To: rt.sub.ID, From: "user", Text: bookingQuestion})
				var delivered, confirmed, firstAudio, firstActiveAudio time.Time
				requestStages := map[string]bool{}
				requests, direct, audioBytes := 0, 0, 0
				cursor := 0
				spoken := ""
				tick := time.NewTicker(10 * time.Millisecond)
				defer tick.Stop()
				for {
					select {
					case <-ctx.Done():
						t.Fatalf("timeout variant=%s calls=%d requests=%d spoken=%s", variant, calls.Load(), requests, spoken)
					case frame := <-rt.audioOut:
						audioBytes += len(frame.Audio)
						if firstAudio.IsZero() {
							firstAudio = time.Now()
						}
						if firstActiveAudio.IsZero() && mediaVoiced(frame.Audio, AudioPCM16) {
							firstActiveAudio = time.Now()
						}
						frame.timing.writtenBytes(frame.ResponseID, frame.ItemID, time.Now(), len(frame.Audio), len(frame.Audio))
						frame.written(true)
					case <-tick.C:
						events, next := rt.telemetry.StoredEvents(cursor)
						cursor = next
						for _, e := range events {
							switch e.Type {
							case "realtime.request_timing":
								var data map[string]any
								if json.Unmarshal(e.Data, &data) == nil {
									if stage, ok := data["stage"].(string); ok {
										requestStages[stage] = true
									}
								}
							case "realtime.error", "realtime.execution_error", "llm.error":
								t.Fatalf("%s: %s", e.Type, e.Data)
							case "llm.request.started":
								requests++
							case "thread.work_direct_result":
								direct++
							case "realtime.execution_completed":
								if delivered.IsZero() {
									delivered = time.Now()
								}
							case "realtime.assistant":
								spoken += " " + string(e.Data)
								if confirmed.IsZero() && strings.Contains(strings.ToLower(spoken), "abricot") {
									confirmed = time.Now()
								}
							}
						}
						if !delivered.IsZero() && !confirmed.IsZero() && audioBytes > 0 {
							if calls.Load() != 1 || emails.Load() != 0 {
								t.Fatalf("incorrect actions bookings=%d emails=%d", calls.Load(), emails.Load())
							}
							if variant == "luna_receipt" && (direct != 1 || requests != 1) {
								t.Fatalf("shortcut not taken requests=%d direct=%d", requests, direct)
							}
							if strings.Contains(spoken, "internal test secret") {
								t.Fatal("private tool data disclosed")
							}
							if variant == "luna_receipt" && (!requestStages["result_available"] || !requestStages["submitted"] || !requestStages["context_acknowledged"]) {
								t.Fatalf("missing request timing: %v", requestStages)
							}
							result := map[string]any{"first_output_audio_ms": timingMS(start, firstAudio), "first_non_silent_audio_ms": timingMS(start, firstActiveAudio), "playback_measurement": "unavailable_test_sink", "useful_audio_association": "unknown", "request_stages": requestStages, "variant": variant, "trial": trial + 1, "backend_ms": delivered.Sub(start).Milliseconds(), "confirmation_caption_ms": confirmed.Sub(start).Milliseconds(), "requests": requests, "direct_receipts": direct, "audio_bytes": audioBytes, "booking_calls": calls.Load()}
							data, _ := json.Marshal(result)
							t.Logf("VOICE_BENCH %s", data)
							return
						}
					}
				}
			})
		}
	}
}

// Reuses one ordinary managed worker across availability and booking, with
// independent assignments like a phone conversation. No completion hints are
// added to the user's request: the native tool schema must carry the contract.
func TestVoiceExecutionLunaMultiTurnReceiptsLive(t *testing.T) {
	key := voiceExecutionTestKey(t)
	for trial := 0; trial < 3; trial++ {
		t.Run(fmt.Sprint(trial+1), func(t *testing.T) {
			parent := newTestThinkerFull()
			parent.session = NewSession(t.TempDir(), "main")
			defer parent.Stop()
			parent.provider = NewOpenAINativeProvider(key).(*OpenAINativeProvider).WithExecutionModel("gpt-6-luna")
			var calls, emails, lookups atomic.Int32
			registerBookingFixture(t, parent, false, &calls, &emails)
			parent.registry.Register(&ToolDef{Name: "availability", Description: "Check the requested appointment time without booking. Returns status available, date and reference.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"date": map[string]any{"type": "string"}}, "required": []string{"date"}}, Handler: func(args map[string]string) ToolResponse {
				lookups.Add(1)
				if args["date"] != "2026-11-03T14:00:00+01:00" {
					return ToolResponse{Text: `{"status":"invalid"}`, IsError: true}
				}
				return ToolResponse{Text: `{"status":"available","date":"2026-11-03T14:00:00+01:00","reference":"slot42"}`}
			}})
			if err := parent.threads.SpawnWithOpts("worker", bookingInstructions+" Utilisez availability pour vérifier un créneau sans réserver. Attendez la confirmation du client avant book.", nil, SpawnOpts{WorkMode: true, DeferRun: true, Reasoning: ReasoningNone}); err != nil {
				t.Fatal(err)
			}
			worker := parent.threads.threads["worker"].Thinker
			available := bookingResultPolicy()
			available.Tool = "availability"
			available.SuccessValue = "available"
			worker.executionPolicy = &RealtimeExecutionConfig{PreloadTools: []string{"book", "email", "availability"}, ResultTools: []DirectToolResultPolicy{available, bookingResultPolicy()}}
			if err := parent.threads.Start("worker"); err != nil {
				t.Fatal(err)
			}
			questions := []string{"Bonjour, je suis Camille Durand. Pouvez-vous juste vérifier la disponibilité le 3 novembre 2026 à 14 heures, fuseau Europe/Paris ? Ne réservez pas encore.", "Merci. Je confirme : réservez ce créneau pour Camille Durand, le mardi 3 novembre 2026 à 14 heures, fuseau Europe/Paris. Annoncez la référence."}
			for turn, question := range questions {
				start := time.Now()
				w := ThreadWork{ID: fmt.Sprintf("turn-%d", turn), Revision: 1, Origin: "main", Kind: "request"}
				if err := parent.threads.QueueWork("worker", w, question, nil); err != nil {
					t.Fatal(err)
				}
				deadline := time.NewTimer(60 * time.Second)
				completed := false
				for !completed {
					select {
					case <-deadline.C:
						t.Fatal("multi-turn worker timed out")
					case <-parent.sub.Wake:
						for _, e := range parent.sub.DrainTargeted() {
							if e.Work != nil && e.Work.ID == w.ID && e.Work.Kind == "result" {
								completed = true
								if !strings.Contains(e.Text, []string{"slot42", "abricot"}[turn]) {
									t.Fatalf("incorrect result: %s", e.Text)
								}
								if calls.Load() != int32(turn) || lookups.Load() != 1 || emails.Load() != 0 {
									t.Fatalf("wrong actions: booking=%d availability=%d email=%d", calls.Load(), lookups.Load(), emails.Load())
								}
								t.Logf("trial=%d turn=%d elapsed_ms=%d result=%s", trial+1, turn+1, time.Since(start).Milliseconds(), e.Text)
							}
						}
					}
				}
				deadline.Stop()
			}
			parent.threads.KillAll()
			events, _ := parent.telemetry.StoredEvents(0)
			direct, requests, missing := 0, 0, 0
			for _, e := range events {
				switch e.Type {
				case "thread.work_direct_result":
					direct++
				case "llm.request.started":
					requests++
				case "thread.work_direct_result_evaluated":
					if strings.Contains(string(e.Data), "completion_decision_missing") {
						missing++
					}
				}
			}
			if direct != 2 || requests != 2 || missing != 0 {
				t.Fatalf("direct=%d requests=%d missing=%d", direct, requests, missing)
			}
			t.Logf("trial=%d direct_receipts=%d requests=%d missing_decisions=%d", trial+1, direct, requests, missing)
		})
	}
}
