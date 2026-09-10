package core

import (
	"sync"
	"time"
)

// This state belongs to the logical thread, not to a provider socket.
type realtimeRecovery struct {
	sync.Mutex
	wake                          chan struct{}
	resume                        RealtimeSession
	want                          bool
	continueTurn                  bool
	input                         []Event
	audio                         []byte
	deadline                      time.Time
	retryAt                       time.Time
	retry                         realtimeRecoveryRetry
	opened, lastInput, lastOutput time.Time
	inputBytes, outputBytes       uint64
	hadInput                      bool
	inputItem                     string
	toolNames, blockedTools       map[string]bool
}

type realtimeRecoveryRetry struct{ failures int }

func (r *realtimeRecoveryRetry) failed() time.Duration {
	r.failures++
	// A circuit cooldown bounds repeated established-session failures too.
	if r.failures >= 6 {
		return time.Minute
	}
	return min(realtimeReconnectMaxDelay, realtimeReconnectMinDelay<<(r.failures-1))
}

func (rt *RealtimeThinker) wakeRecovery() {
	select {
	case rt.recovery.wake <- struct{}{}:
	default:
	}
}

func (rt *RealtimeThinker) pendingToolWork() bool {
	rt.toolBatchMu.Lock()
	defer rt.toolBatchMu.Unlock()
	return len(rt.toolCallBatches) > 0
}

func (rt *RealtimeThinker) needsSession() bool {
	rt.lifecycleMu.Lock()
	bridge, planned := rt.bridgeConnected, rt.pendingReconnectPlanned && rt.pendingReconnectReason != "provider_goaway"
	rt.lifecycleMu.Unlock()
	rt.recovery.Lock()
	defer rt.recovery.Unlock()
	return bridge || planned || rt.recovery.want || len(rt.recovery.input) > 0 || len(rt.recovery.audio) > 0
}

func (rt *RealtimeThinker) playbackSettled() bool {
	rt.outputMu.Lock()
	defer rt.outputMu.Unlock()
	return !rt.playbackTracked || rt.playedMS >= rt.outputBytes*1000/realtimePCMBytesPerSecond
}

func (rt *RealtimeThinker) closeForRecovery(session RealtimeSession, planned bool, reason string) {
	if session == nil || session != rt.currentSession() {
		return
	}
	busy, tools := rt.responseInProgress(), rt.pendingToolWork()
	rt.toolBatchMu.Lock()
	var pendingNames []string
	for _, batch := range rt.toolBatches {
		for name := range batch.names {
			pendingNames = append(pendingNames, name)
		}
	}
	rt.toolBatchMu.Unlock()
	rt.lifecycleMu.Lock()
	rt.pendingReconnectReason, rt.pendingReconnectPlanned = reason, planned
	bridge, generation := rt.bridgeConnected, rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	now := time.Now()
	rt.recovery.Lock()
	r := &rt.recovery
	r.resume = nil
	if !busy && !tools {
		r.resume = session
	}
	r.want = r.want || busy || tools || bridge
	r.continueTurn = r.continueTurn || (busy && r.hadInput) || tools
	if reason == "tool_result_delivery_failed" || reason == "response_request_failed" {
		r.continueTurn, r.want, r.resume = true, true, nil
	}
	r.deadline = time.Time{}
	for name := range r.toolNames {
		r.blockedTools[name] = true
	}
	for _, name := range pendingNames {
		r.blockedTools[name] = true
	}
	if !planned && (r.want || len(r.input) > 0 || len(r.audio) > 0) {
		if r.hadInput && now.Sub(r.opened) >= 2*time.Minute {
			r.retry.failures = 0
		}
		// First unexpected close recovers immediately; successive failures back off.
		delay := time.Duration(0)
		if r.retry.failures > 0 {
			delay = r.retry.failed()
		} else {
			r.retry.failures = 1
		}
		r.retryAt = now.Add(delay)
	}
	data := map[string]any{"reason": reason, "planned": planned, "generation": generation,
		"session_age_ms": now.Sub(r.opened).Milliseconds(), "bridge_connected": bridge,
		"input_audio_bytes": r.inputBytes, "output_audio_bytes": r.outputBytes, "recovery_failures": r.retry.failures}
	if !r.lastInput.IsZero() {
		data["since_input_ms"] = now.Sub(r.lastInput).Milliseconds()
	}
	if !r.lastOutput.IsZero() {
		data["since_output_ms"] = now.Sub(r.lastOutput).Milliseconds()
	}
	rt.recovery.Unlock()
	rt.emit("realtime.session_closed", data)
	rt.replaceSession(nil)
}

func (rt *RealtimeThinker) recordRealtimeInput(audioBytes int) {
	rt.recovery.Lock()
	rt.recovery.lastInput = time.Now()
	rt.recovery.inputBytes += uint64(audioBytes)
	rt.recovery.hadInput = true
	rt.recovery.Unlock()
}

func (rt *RealtimeThinker) newRealtimeCallerTurn() {
	rt.recovery.Lock()
	rt.recovery.blockedTools = map[string]bool{}
	rt.recovery.toolNames = map[string]bool{}
	rt.recovery.Unlock()
}

func (rt *RealtimeThinker) queueRecoveryAudio(audio []byte) {
	// Bound buffered audio to five seconds. Preserve the beginning of the
	// utterance and report overflow instead of silently growing memory forever.
	const limit = 5 * realtimePCMBytesPerSecond
	rt.recovery.Lock()
	keep := min(len(audio), limit-len(rt.recovery.audio))
	rt.recovery.audio = append(rt.recovery.audio, audio[:keep]...)
	rt.recovery.want = true
	rt.recovery.Unlock()
	if keep < len(audio) {
		rt.emit("realtime.audio_drop", map[string]any{"direction": "input", "reason": "recovery_buffer_full", "bytes": len(audio) - keep})
	}
}

func (rt *RealtimeThinker) flushRecoveryInput() {
	rt.recovery.Lock()
	input, audio := rt.recovery.input, rt.recovery.audio
	rt.recovery.input, rt.recovery.audio = nil, nil
	rt.recovery.Unlock()
	for _, event := range input {
		rt.handleBusEvent(event)
	}
	if len(audio) > 0 {
		session := rt.currentSession()
		if session == nil {
			rt.queueRecoveryAudio(audio)
			return
		}
		if err := session.SendAudio(audio); err != nil {
			rt.queueRecoveryAudio(audio)
			rt.closeForRecovery(session, false, "audio_delivery_failed")
		} else {
			rt.recordRealtimeInput(len(audio))
		}
	}
}
