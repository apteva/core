package core

import (
	"context"
	"encoding/json"
	"time"
)

// Opening is split into an actor-owned snapshot and connection-only work.
type realtimeOpenAttempt struct {
	ctx         context.Context
	cancel      context.CancelFunc
	opts        RealtimeSessionOpts
	previous    RealtimeSession
	restore     bool
	timing      *realtimeTiming
	connectedAt time.Time
	generation  int
	directive   string
}
type realtimeOpenResult struct {
	attempt        *realtimeOpenAttempt
	session        RealtimeSession
	resumed        bool
	err, resumeErr error
}

func (rt *RealtimeThinker) cancelOpening() {
	if rt.opening != nil {
		rt.opening.cancel()
		rt.opening = nil
	}
}
func (rt *RealtimeThinker) startOpening() {
	a, err := rt.prepareSession(true)
	if err != nil {
		rt.openFailed(err)
		return
	}
	rt.opening = a
	rt.providerState("recovering", nil)
	go func() {
		r := connectRealtimeSession(rt.provider, a)
		// Unbuffered ownership handoff: a late/ignored cancellation cannot leak a
		// candidate or install it into a stopped/newer generation.
		select {
		case rt.openResults <- r:
		case <-a.ctx.Done():
			if r.session != nil {
				go func() { _ = r.session.Close() }()
			}
		}
	}()
}
func (rt *RealtimeThinker) openFailed(err error) {
	rt.recovery.Lock()
	delay := rt.recovery.retry.failed()
	rt.recovery.retryAt = time.Now().Add(delay)
	rt.recovery.Unlock()
	rt.emit("realtime.reconnect", map[string]any{"success": false, "error": err.Error(), "error_class": realtimeErrorClass(err), "delay_ms": delay.Milliseconds(), "input_queue_ms": rt.input.backlog().Milliseconds()})
	rt.providerState("failed", err)
}
func (rt *RealtimeThinker) finishOpening(r realtimeOpenResult) {
	a := r.attempt
	if rt.opening != a || a.ctx.Err() != nil || int64(a.generation) != generationForAudioFrame(rt) || rt.currentSession() != nil {
		a.cancel()
		if r.session != nil {
			go func() { _ = r.session.Close() }()
		}
		return
	}
	rt.opening = nil
	if r.err != nil {
		a.cancel()
		rt.openFailed(r.err)
		return
	}
	if err := rt.installSession(r); err != nil {
		rt.openFailed(err)
		return
	}
	rt.lifecycleMu.Lock()
	generation, reason, planned := rt.sessionGeneration, rt.pendingReconnectReason, rt.pendingReconnectPlanned
	rt.pendingReconnectReason, rt.pendingReconnectPlanned = "", false
	rt.lifecycleMu.Unlock()
	rt.recovery.Lock()
	started := rt.recovery.started
	rt.recovery.started = time.Time{}
	rt.recovery.Unlock()
	if started.IsZero() {
		started = a.connectedAt
	}
	rt.emit("realtime.reconnect", map[string]any{"success": true, "generation": generation, "reason": reason, "planned": planned, "recovery_ms": time.Since(started).Milliseconds(), "connection_ms": time.Since(a.connectedAt).Milliseconds(), "input_queue_ms": rt.input.backlog().Milliseconds()})
	rt.input.nextSend = time.Time{}
	rt.flushRecoveryInput()
}
func (rt *RealtimeThinker) providerState(state string, err error) {
	rt.providerStateMu.Lock()
	rt.providerStatus = state
	rt.providerStateMu.Unlock()
	data := map[string]any{"state": state, "generation": generationForAudioFrame(rt), "input_queue_ms": rt.input.backlog().Milliseconds(), "external_transport_independent": true}
	if err != nil {
		data["error"] = err.Error()
		data["error_class"] = realtimeErrorClass(err)
	}
	rt.emit("realtime.provider_state", data)
	rt.wakeProviderState()
}

func (rt *RealtimeThinker) wakeProviderState() {
	select {
	case rt.providerStateWake <- struct{}{}:
	default:
	}
}

// State is a coalescing latest-value lane, independent of playback controls.
// A full clear/control queue cannot permanently hide the next ready signal.
func (rt *RealtimeThinker) providerStateControl() string {
	raw, _ := json.Marshal(map[string]any{"type": "session.state", "state": rt.providerStateSnapshot(), "session_generation": generationForAudioFrame(rt)})
	return string(raw)
}

func (rt *RealtimeThinker) providerStateSnapshot() string {
	rt.providerStateMu.Lock()
	defer rt.providerStateMu.Unlock()
	if rt.providerStatus == "" {
		return "recovering"
	}
	return rt.providerStatus
}
