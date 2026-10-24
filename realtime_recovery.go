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
	started                       time.Time
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
	return bridge || planned || rt.recovery.want || len(rt.recovery.input) > 0 || rt.input.pending()
}

func (rt *RealtimeThinker) playbackSettled() bool {
	pending, awaiting, _ := rt.delivery.snapshot()
	if rt.capabilities.ContinuousConversation {
		pending, awaiting = rt.delivery.pendingSpeechPlayback()
	}
	if pending > 0 || awaiting > 0 {
		return false
	}
	rt.outputMu.Lock()
	defer rt.outputMu.Unlock()
	return !rt.playbackTracked || rt.playedMS >= rt.outputBytes*1000/realtimePCMBytesPerSecond
}

func (rt *RealtimeThinker) closeForRecovery(session RealtimeSession, planned bool, reason string, cause ...error) {
	if session == nil || session != rt.currentSession() {
		return
	}
	rt.flushPendingTranscripts(reason)
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
	if r.started.IsZero() {
		r.started = now
	}
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
	if !planned && (r.want || len(r.input) > 0 || rt.input.pending()) {
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
	if len(cause) > 0 && cause[0] != nil {
		data["error"] = cause[0].Error()
		data["error_class"] = realtimeErrorClass(cause[0])
	}
	data["input_queue_ms"] = rt.input.backlog().Milliseconds()
	if !r.lastInput.IsZero() {
		data["since_input_ms"] = now.Sub(r.lastInput).Milliseconds()
	}
	if !r.lastOutput.IsZero() {
		data["since_output_ms"] = now.Sub(r.lastOutput).Milliseconds()
	}
	rt.recovery.Unlock()
	rt.currentTiming().finish("", "", "session_closed")
	rt.recordSessionClosure(reason, planned, data)
	rt.replaceSession(nil)
	rt.providerState("recovering", nil)
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
	rt.input.push(audio, time.Now())
	rt.wakeRecovery()
}

func (rt *RealtimeThinker) flushRecoveryInput() {
	rt.recovery.Lock()
	input := rt.recovery.input
	rt.recovery.input = nil
	rt.recovery.Unlock()
	for _, event := range input {
		rt.handleBusEvent(event)
	}
	if session := rt.currentSession(); session != nil {
		rt.drainRealtimeInput(session)
	}
}
