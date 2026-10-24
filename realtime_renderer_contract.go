package core

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

const realtimeAudioSessionHeader = "X-Apteva-Audio-Session"
const realtimeLiveBridgeWriteTimeout = 250 * time.Millisecond

// Renderer behavior is distinct from the tool-capability readiness snapshot.
// The HTTP handshake conveys it before any greeting is produced. New renderers
// opt in; older clients retain their existing framing and timeout contract.
func (a *APIServer) realtimeRendererContract(thread string, r *http.Request) string {
	if r.URL.Query().Get("audio_protocol") != "1" {
		if a.thinker != nil && a.thinker.threads != nil {
			_, t := a.thinker.threads.findManagedThread(thread)
			if t != nil && t.Realtime != nil {
				t.Realtime.rendererScoped.Store(false)
			}
		}
		return ""
	}
	continuous, inputSpeech, localInterruption, providerInterruption := false, false, false, false
	media := RealtimeMediaConfig{}
	providerState := "recovering"
	if a.thinker != nil && a.thinker.threads != nil {
		_, t := a.thinker.threads.findManagedThread(thread)
		if t != nil && t.Realtime != nil {
			t.Realtime.rendererScoped.Store(true)
			media = t.Realtime.opts.OutputConfig.Media
			providerState = t.Realtime.providerStateSnapshot()
			continuous = t.Realtime.capabilities.ContinuousConversation
			inputSpeech = t.Realtime.capabilities.InputSpeechEvents
			providerInterruption = t.Realtime.capabilities.ProviderInterruption
			localInterruption = t.Realtime.capabilities.RendererInterruption && !providerInterruption
		}
	}
	raw, _ := json.Marshal(map[string]any{"version": 1, "audio_format": media.format(), "sample_rate": media.rate(), "channels": 1, "provider_state": providerState, "provider_state_events": true, "continuous_conversation": continuous, "input_speech_events": inputSpeech, "local_interruption_required": localInterruption, "provider_interruption": providerInterruption, "scoped_cancellation": true, "receipt_evidence": true, "playback_flow_control": true, "max_input_buffer_ms": 5000, "max_output_buffer_ms": 15000})
	return string(raw)
}

func validPlaybackEvidence(value string) bool {
	return value == "carrier_mark" || value == "estimated" || value == "rtp_transmitted" || value == "legacy"
}

func generationForAudioFrame(rt *RealtimeThinker) int64 {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	return int64(rt.sessionGeneration)
}

// Compatibility admission for standalone bridges without a managed runtime.
// Managed threads admit directly into the shared duration/age budget.
func enqueueLiveRendererInput(ctx context.Context, input chan []byte, pcm []byte) error {
	if len(pcm)%2 != 0 || len(pcm) > 5760 {
		return newRealtimeBridgeEnd("client", "read", 1007, "invalid live PCM frame", "protocol", false)
	}
	for offset := 0; offset < len(pcm); offset += 960 {
		end := min(offset+960, len(pcm))

		select {
		case <-ctx.Done():
			return context.Cause(ctx)
		default:
		}
		frame := pcm[offset:end]
		// Normal delivery needs neither a timer nor an allocation.
		select {
		case input <- frame:
			continue
		default:
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case input <- frame:
			timer.Stop()
		case <-ctx.Done():
			timer.Stop()
			return context.Cause(ctx)
		case <-timer.C:
			return newRealtimeBridgeEnd("core", "read", 1011, "live audio input backpressure", "timeout", false)
		}

	}
	return nil
}

func (rt *RealtimeThinker) rendererPlaybackOverflowForGeneration(item string, generation int64) {
	if generation > 0 && generation != generationForAudioFrame(rt) {
		rt.delivery.mu.Lock()
		if entry := rt.delivery.items[strconv.FormatInt(generation, 10)+"/"+item]; entry != nil {
			entry.cancelled = true
			entry.waitPlayback = false
		}
		rt.delivery.mu.Unlock()
		rt.emit("realtime.renderer_stale_overflow", map[string]any{"item_id": item, "session_generation": generation})
		return
	}
	rt.rendererPlaybackOverflow(item)
}
