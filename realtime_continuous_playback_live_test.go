package core

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// Paid, opt-in: real Live audio and an independently rendered caller, with
// a paced renderer reporting actual simulated consumption (not socket writes).
// This exercises Core; it is not an assertion about browser hardware playback.
func TestOpenAILivePacedInterruptionPlaybackLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_OPENAI_LIVE_PLAYBACK") != "1" {
		t.Skip("set RUN_OPENAI_LIVE_PLAYBACK=1 for paid Live playback verification")
	}
	loadIntegrationEnv()
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not configured")
	}
	caller := renderLiveCallerText(t, key, "Attendez, je voudrais simplement prendre rendez-vous mardi. Quel jour ai-je demandé ?")
	rt, _ := completionFixture(t)
	rt.Thinker.threadID = "main"
	rt.Thinker.directive = "Vous êtes un accueil téléphonique générique. Parlez uniquement en français. Si la personne vous interrompt, écoutez sa demande et répondez brièvement. N'inventez aucune réservation. Ne terminez pas la conversation."
	rt.messages[0].Content = rt.Thinker.directive
	backend := NewOpenAINativeProvider(key)
	applyModelOverrides(backend, map[string]string{"large": "gpt-6.1-sol", "medium": "gpt-6.1-sol", "small": "gpt-6.1-sol"})
	rt.Thinker.provider = backend
	rt.pool = &ProviderPool{providers: map[string]LLMProvider{"openai": backend}, default_: "openai", order: []string{"openai"}}
	rt.provider = NewOpenAIRealtimeProvider(key)
	model := os.Getenv("OPENAI_LIVE_PLAYBACK_MODEL")
	if model == "" {
		model = "gpt-live-1"
	}
	rt.opts.Model, rt.opts.Voice = model, "brise"
	rt.capabilities = capabilitiesForRealtime(rt.provider, model)
	rt.registry = NewToolRegistry("")
	rt.initialMessage = "Commencez maintenant en français : Bonjour, tous nos conseillers sont actuellement occupés. Je peux vous aider à prendre rendez-vous. Expliquez ensuite calmement en plusieurs phrases les renseignements à préparer pour un rendez-vous : le jour souhaité, l'heure, le nom et un moyen de contact. Puis écoutez."
	ctx, cancel := context.WithTimeout(context.Background(), 70*time.Second)
	defer cancel()
	rt.cancel()
	rt.ctx, rt.cancel = context.WithCancel(ctx)
	control := make(chan string, 16)
	rt.audioControl = control
	rt.delivery.playbackReady()
	if err := rt.openSession(false); err != nil {
		t.Fatal(err)
	}
	done := runCompletionFixture(rt)
	defer func() {
		rt.Thinker.Stop()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("worker stop timed out")
		}
	}()
	stopMic := make(chan struct{})
	defer close(stopMic)
	startCaller := make(chan struct{})
	go func() {
		tick := time.NewTicker(20 * time.Millisecond)
		defer tick.Stop()
		position := -1
		callerSignal := startCaller
		for {
			select {
			case <-stopMic:
				return
			case <-ctx.Done():
				return
			case <-callerSignal:
				position = 0
				callerSignal = nil
			case <-tick.C:
				frame := make([]byte, 960)
				if position >= 0 && position < len(caller) {
					copy(frame, caller[position:min(position+960, len(caller))])
					position += 960
				}
				if s := rt.currentSession(); s != nil {
					_ = s.SendAudio(frame)
				}
			}
		}
	}()
	type pending struct {
		frame    RealtimeAudioFrame
		consumed int
	}
	var queue []pending
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	started := time.Now()
	var firstAudio, interruptedAt time.Time
	voicedPlayed := 0
	cursor := 0
	cancelledID := ""
	clearSeen := false
	cancelledIDs := map[string]bool{}
	var clearLatency time.Duration
	heardCaller := false
	confirmedDay := false
	resumedMS := 0
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("playback verification timeout: clear=%v caller=%v day=%v resumed_ms=%d", clearSeen, heardCaller, confirmedDay, resumedMS)
		case frame := <-rt.audioOut:
			if firstAudio.IsZero() && !frame.Silent {
				firstAudio = time.Now()
			}
			if frame.cancelled != nil && frame.cancelled() {
				frame.written(false)
				continue
			}
			frame.written(true)
			queue = append(queue, pending{frame: frame})
		case raw := <-control:
			var c struct {
				Type   string `json:"type"`
				ItemID string `json:"item_id"`
			}
			if json.Unmarshal([]byte(raw), &c) != nil || c.Type != "interrupt" {
				continue
			}
			if !cancelledIDs[c.ItemID] {
				t.Fatalf("clear targeted an unrelated playback %q", c.ItemID)
			}
			kept := queue[:0]
			for _, p := range queue {
				if p.frame.ItemID != c.ItemID {
					kept = append(kept, p)
				}
			}
			queue = kept
			rt.acknowledgePlaybackClear(c.ItemID)
			clearSeen = true
			clearLatency = time.Since(interruptedAt)
		case <-tick.C:
			if !firstAudio.IsZero() && time.Since(firstAudio) > 500*time.Millisecond && len(queue) > 0 {
				p := &queue[0]
				count := min(960, len(p.frame.Audio)-p.consumed)
				p.consumed += count
				if !p.frame.Silent {
					voicedPlayed += count
				}
				end := p.frame.AudioEndMS - (len(p.frame.Audio)-p.consumed)*1000/realtimePCMBytesPerSecond
				rt.acknowledgePlayback(p.frame.ItemID, end)
				if !interruptedAt.IsZero() && !cancelledIDs[p.frame.ItemID] && !p.frame.Silent {
					resumedMS += count * 1000 / realtimePCMBytesPerSecond
				}
				if p.consumed == len(p.frame.Audio) {
					queue = queue[1:]
				}
			}
			if interruptedAt.IsZero() && voicedPlayed >= 4800 && len(queue) > 0 {
				rt.outputMu.Lock()
				cancelledID = rt.outputItemID
				for _, id := range rt.delivery.pendingPlaybackIDs() {
					cancelledIDs[id] = true
				}
				cancelledIDs[cancelledID] = true
				rt.outputMu.Unlock()
				if cancelledID != "" {
					interruptedAt = time.Now()
					rt.rendererSpeechStarted()
					close(startCaller)
				}
			}
			events, next := rt.telemetry.StoredEvents(cursor)
			cursor = next
			for _, e := range events {
				if e.Type == "realtime.error" || e.Type == "realtime.execution_error" {
					t.Fatalf("live error: %s", e.Data)
				}
				if e.Type == "realtime.user" && strings.Contains(strings.ToLower(string(e.Data)), "mardi") {
					heardCaller = true
				}
				if e.Type == "realtime.assistant" && !interruptedAt.IsZero() && strings.Contains(strings.ToLower(string(e.Data)), "mardi") {
					confirmedDay = true
				}
				if e.Type == "realtime.tool_markup_leaked" {
					t.Fatalf("tool syntax in speech: %s", e.Data)
				}
				if e.Type == "realtime.output_blocked" && strings.Contains(string(e.Data), "missing_output_transcript") {
					t.Fatalf("default playback rejected missing captions: %s", e.Data)
				}
			}
			if clearSeen && heardCaller && confirmedDay && resumedMS >= 200 {
				timing := rt.currentTiming()
				maxGuardWait := time.Duration(0)
				timing.mu.Lock()
				for _, items := range []map[string]*realtimeUtteranceTiming{timing.items, timing.completed} {
					for _, item := range items {
						if !item.firstAudio.IsZero() && !item.guardRelease.IsZero() {
							wait := item.guardRelease.Sub(item.firstAudio)
							if wait > maxGuardWait {
								maxGuardWait = wait
							}
							if item.guard != "monitor" {
								timing.mu.Unlock()
								t.Fatalf("unexpected guard on default Live output: %s", item.guard)
							}
						}
					}
				}
				timing.mu.Unlock()
				if maxGuardWait > 100*time.Millisecond {
					t.Fatalf("monitor delayed audio release by %s", maxGuardWait)
				}
				t.Logf("model=%s first_voiced_audio_ms=%d max_guard_wait_ms=%.3f interrupted_playback=%s simulated_clear_ack_ms=%.3f caller_mardi=true assistant_mardi=true resumed_voiced_playback_ms=%d total_ms=%d", model, firstAudio.Sub(started).Milliseconds(), float64(maxGuardWait)/float64(time.Millisecond), cancelledID, float64(clearLatency)/float64(time.Millisecond), resumedMS, time.Since(started).Milliseconds())
				return
			}
		}
	}
}
