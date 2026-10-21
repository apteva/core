package core

import (
	"context"
	"encoding/json"
	"github.com/joho/godotenv"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Paid, opt-in verification of both the real OpenAI Live voice transport and
// Core's real Sol inference/tool/result path. The only tool is a local fixture.
// RUN_OPENAI_LIVE_SMOKE=1 go test -run '^TestOpenAILiveFrenchCoreExecutionLive$' -v -timeout 4m .
func TestOpenAILiveFrenchCoreExecutionLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_OPENAI_LIVE_SMOKE") != "1" {
		t.Skip("set RUN_OPENAI_LIVE_SMOKE=1 without -short for paid Live + Sol verification")
	}
	loadIntegrationEnv()
	if os.Getenv("OPENAI_API_KEY") == "" {
		_ = godotenv.Load("../.env")
	}
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not configured")
	}
	var callerPCM []byte
	if os.Getenv("RUN_OPENAI_LIVE_AUDIO_INPUT") == "1" {
		callerPCM = renderLiveCaller(t, key)
	}
	for _, mode := range []string{"thread", "provider"} {
		for _, voice := range []string{"brise", "sillage"} {
			t.Run(mode+"/"+voice, func(t *testing.T) {
				audioExportDir := os.Getenv("OPENAI_LIVE_AUDIO_DIR")
				var recordedPCM []byte
				var recordedTranscript strings.Builder
				var verifiedAt, lastAudioAt time.Time
				rt, _ := completionFixture(t)
				rt.Thinker.threadID = "main"
				rt.Thinker.directive = "Parlez uniquement en français de France. Pour valider un code, consultez account_balance une seule fois et dites le mot réellement retourné. N'inventez aucun résultat. Ne mentionnez jamais les outils ou leur fonctionnement. Ne terminez pas la conversation."
				rt.messages[0].Content = rt.Thinker.directive
				backend := NewOpenAINativeProvider(key)
				applyModelOverrides(backend, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
				rt.Thinker.provider = backend
				rt.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": backend}, default_: "openai", order: []string{"openai"}}
				rt.provider = NewOpenAIRealtimeProvider(key)
				rt.opts.Model = "gpt-live-1"
				rt.opts.Voice = voice
				rt.capabilities = capabilitiesForRealtime(rt.provider, rt.opts.Model)
				rt.registry = NewToolRegistry("")
				rt.toolAllowlist = map[string]bool{"account_balance": true}
				var calls atomic.Int32
				rt.registry.Register(&ToolDef{Name: "account_balance", Description: "Look up the secret validation word. Retrieve it; do not guess.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}, Handler: func(map[string]string) ToolResponse {
					calls.Add(1)
					return ToolResponse{Text: `{"verified_word":"abricot"}`}
				}})
				rt.config.Providers = []ProviderConfig{{Name: "openai", RealtimeExecution: RealtimeExecutionConfig{Mode: mode}}}
				ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
				defer cancel()
				rt.cancel()
				rt.ctx, rt.cancel = context.WithCancel(ctx)
				if err := rt.openSession(false); err != nil {
					t.Fatal(err)
				}
				session := rt.currentSession()
				done := runCompletionFixture(rt)
				defer func() {
					rt.Thinker.Stop()
					select {
					case <-done:
					case <-time.After(2 * time.Second):
						t.Error("live worker failed to stop")
					}
				}()
				stopAudio := make(chan struct{})
				defer close(stopAudio)
				go func() {
					ticker := time.NewTicker(20 * time.Millisecond)
					defer ticker.Stop()
					position := -24000 * 2 // one second of paced silence before the simulated caller
					for {
						select {
						case <-stopAudio:
							return
						case <-ctx.Done():
							return
						case <-ticker.C:
							frame := make([]byte, 960)
							if position >= 0 && position < len(callerPCM) {
								copy(frame, callerPCM[position:min(position+960, len(callerPCM))])
							}
							position += 960
							if active := rt.currentSession(); active != nil {
								_ = active.SendAudio(frame)
							}
						}
					}
				}()
				// The fixture subscribes as main; deliver using its actual subscription.
				if len(callerPCM) == 0 {
					rt.bus.Publish(Event{Type: EventInbox, To: rt.sub.ID, From: "user", Text: "Validez mon code et dites-moi le mot confirmé."})
				}
				audioBytes := 0
				cursor := 0
				spoken := ""
				finished, modelReason, heardCaller := false, false, false
				reconnectStage := 0
				tick := time.NewTicker(20 * time.Millisecond)
				defer tick.Stop()
				for {
					select {
					case <-ctx.Done():
						events, _ := rt.telemetry.StoredEvents(0)
						for _, e := range events {
							if strings.HasPrefix(e.Type, "llm.") || strings.HasPrefix(e.Type, "tool.") || strings.HasPrefix(e.Type, "realtime.execution") {
								t.Logf("%s %s %s", e.Type, e.ThreadID, e.Data)
							}
						}
						t.Fatalf("Live verification timed out; tool_calls=%d audio_bytes=%d spoken=%q", calls.Load(), audioBytes, spoken)
					case frame := <-rt.audioOut:
						audioBytes += len(frame.Audio)
						if audioExportDir != "" {
							recordedPCM = append(recordedPCM, frame.Audio...)
							lastAudioAt = time.Now()
						}
						frame.written(true)
					case <-tick.C:
						events, next := rt.telemetry.StoredEvents(cursor)
						cursor = next
						for _, e := range events {
							if e.Type == "realtime.error" || e.Type == "realtime.execution_error" {
								if reconnectStage == 1 && e.Type == "realtime.error" && strings.Contains(string(e.Data), "closed network connection") {
									continue
								}
								t.Fatalf("live error: %s", e.Data)
							}
							if mode == "provider" && (e.Type == "realtime.backend_event" || e.Type == "tool.call" || e.Type == "realtime.execution_started" || e.Type == "llm.error") {
								t.Logf("%s %s", e.Type, e.Data)
							}
							if reconnectStage == 1 && e.Type == "realtime.session_opened" {
								reconnectStage = 2
								rt.bus.Publish(Event{Type: EventInbox, To: rt.sub.ID, From: "user", Text: "Répétez simplement le mot déjà confirmé. Ne refaites pas la validation."})
							}
							if e.Type == "realtime.execution_completed" {
								finished = true
							}
							if e.Type == "realtime.user" {
								heardCaller = true
							}
							if e.Type == "tool.call" {
								var data map[string]any
								if json.Unmarshal(e.Data, &data) == nil && data["name"] == "account_balance" && data["reason_source"] == "model" && data["reason"] != "" {
									modelReason = true
								}
							}
							if e.Type == "realtime.assistant" {
								spoken += " " + string(e.Data)
								if audioExportDir != "" {
									recordedTranscript.Write(e.Data)
									recordedTranscript.WriteByte('\n')
								}
							}
						}
						if audioBytes > 0 && finished && modelReason && (len(callerPCM) == 0 || heardCaller) && strings.Contains(strings.ToLower(spoken), "abricot") {
							// Recording is opt-in. Allow trailing speech to arrive before
							// stopping or forcing the reconnect used by the Sillage case.
							if audioExportDir != "" {
								if verifiedAt.IsZero() {
									verifiedAt = time.Now()
								}
								if time.Since(verifiedAt) < 8*time.Second && (time.Since(verifiedAt) < 2*time.Second || time.Since(lastAudioAt) < 2*time.Second) {
									continue
								}
							}
							if voice == "sillage" && reconnectStage == 0 {
								reconnectStage = 1
								finished = false
								spoken = ""
								audioBytes = 0
								verifiedAt = time.Time{}
								if err := session.Close(); err != nil {
									t.Fatal(err)
								}
								continue
							}
							rt.Thinker.Stop()
							select {
							case <-done:
							case <-time.After(2 * time.Second):
								t.Fatal("worker stop timed out")
							}
							if calls.Load() != 1 {
								t.Fatalf("tool executions=%d", calls.Load())
							}
							for _, bad := range []string{"callto:", "account_balance", "the word is"} {
								if strings.Contains(strings.ToLower(spoken), bad) {
									t.Fatalf("unexpected speech: %q", spoken)
								}
							}
							if audioExportDir != "" {
								if err := os.MkdirAll(audioExportDir, 0700); err != nil {
									t.Fatal(err)
								}
								base := filepath.Join(audioExportDir, mode+"-"+voice)
								if err := os.WriteFile(base+".pcm", recordedPCM, 0600); err != nil {
									t.Fatal(err)
								}
								if err := os.WriteFile(base+".transcript.jsonl", []byte(recordedTranscript.String()), 0600); err != nil {
									t.Fatal(err)
								}
								t.Logf("saved PCM16 mono 24000 Hz: %s.pcm (%d bytes)", base, len(recordedPCM))
							}
							t.Logf("mode=%s voice=%s worker=%s backend=gpt-6.1-sol tool_calls=%d pcm_bytes=%d reconnected=%t", mode, voice, rt.backend.worker.ID, calls.Load(), audioBytes, reconnectStage == 2)
							return
						}
					}
				}
			})
		}
	}
}

// Uses an independent real voice model to render caller audio. It never
// supplies task text or expected results to the Live execution worker.
func renderLiveCaller(t *testing.T, key string) []byte {
	t.Helper()
	return renderLiveCallerText(t, key, "Pouvez-vous valider mon code et me dire le mot confirmé ?")
}

func renderLiveCallerText(t *testing.T, key, text string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	p := NewOpenAIRealtimeProvider(key)
	s, err := p.Open(ctx, RealtimeSessionOpts{Model: "gpt-realtime-2.1-mini", Voice: "marin", Instructions: "You simulate a French caller. Read the user's sentence exactly once in French, without introductions or extra text."})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.SendText("user", text); err != nil {
		t.Fatal(err)
	}
	if err := s.RequestResponse(); err != nil {
		t.Fatal(err)
	}
	var pcm []byte
	for {
		select {
		case <-ctx.Done():
			t.Fatal("caller audio generation timed out")
		case e, ok := <-s.Events():
			if !ok {
				t.Fatal("caller audio session closed")
			}
			switch e.Type {
			case RealtimeEventAudioOut:
				pcm = append(pcm, e.Audio...)
			case RealtimeEventError:
				t.Fatal(e.Err)
			case RealtimeEventResponseDone:
				if len(pcm) == 0 {
					t.Fatal("caller generated no audio")
				}
				return pcm
			}
		}
	}
}
