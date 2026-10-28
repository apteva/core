package core

import (
	"strconv"
	"strings"
	"sync"
	"time"
)

const realtimeCompletionTimeout = 5 * time.Second
const realtimeCompletionQuietPeriod = 100 * time.Millisecond

type realtimeCompletion struct {
	requested          bool
	requestedAt        time.Time
	timeout            time.Duration
	initiator, callID  string
	responseID, itemID string
	utteranceDone      bool
	awaitOutput        bool
	outcome            string
}

type realtimeDeliveredItem struct {
	speechPending, speechEnd                  int
	item                                      string
	queued, pending, written, played, dropped int
	waitPlayback                              bool
	cancelled                                 bool
}

// Tracks every queued frame, including one removed from the channel while the
// bridge is still writing it. Written audio and renderer acknowledgements are
// distinct evidence. No wall-clock estimate is substituted for playback.
type realtimeDelivery struct {
	bytesPerSecond                            int
	mu                                        sync.Mutex
	items                                     map[string]*realtimeDeliveredItem
	evidence                                  string
	supportsPlayback                          bool
	queued, pending, written, played, dropped int
}

func (d *realtimeDelivery) enqueue(key, item string, bytes int, silent ...bool) func(bool) {
	settle := d.enqueueBytes(key, item, bytes, silent...)
	return func(success bool) {
		written := 0
		if success {
			written = bytes
		}
		settle(written)
	}
}
func (d *realtimeDelivery) enqueueBytes(key, item string, bytes int, silent ...bool) func(int) {
	speech := len(silent) == 0 || !silent[0]
	d.mu.Lock()
	if d.items == nil {
		d.items = map[string]*realtimeDeliveredItem{}
	}
	// Bound completed receipts; outstanding frames are never evicted.
	if len(d.items) >= 128 {
		for k, old := range d.items {
			if old.pending == 0 && (!old.waitPlayback || old.played+old.dropped >= old.queued) {
				delete(d.items, k)
			}
		}
	}
	entry := d.items[key]
	if entry == nil {
		entry = &realtimeDeliveredItem{item: item, waitPlayback: d.supportsPlayback}
		d.items[key] = entry
	}
	entry.queued += bytes
	if speech {
		entry.speechPending += bytes
		entry.speechEnd = entry.queued
	}
	entry.pending += bytes
	d.queued += bytes
	d.pending += bytes
	d.mu.Unlock()
	var once sync.Once
	return func(written int) {
		once.Do(func() {
			d.mu.Lock()
			defer d.mu.Unlock()
			entry.pending -= bytes
			if speech {
				entry.speechPending -= bytes
			}
			d.pending -= bytes
			if written < 0 {
				written = 0
			}
			if written > bytes {
				written = bytes
			}
			entry.written += written
			d.written += written
			entry.dropped += bytes - written
			d.dropped += bytes - written
		})
	}
}

func (d *realtimeDelivery) acknowledge(key string, endMS int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	entry := d.items[key]
	if entry == nil || entry.cancelled {
		return
	}
	d.supportsPlayback = true
	entry.waitPlayback = true
	played := min(entry.queued, endMS*defaultMediaRate(d.bytesPerSecond)/1000)
	if played > entry.played {
		d.played += played - entry.played
		entry.played = played
	}
}

func (d *realtimeDelivery) playbackReady() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.supportsPlayback = true
	for _, entry := range d.items {
		if !entry.cancelled {
			entry.waitPlayback = true
		}
	}
}

// An interruption asks the renderer to discard an old utterance. It must not
// remain a playback requirement for a later goodbye.
func (d *realtimeDelivery) abandonPlayback(item string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, entry := range d.items {
		if item == "" || entry.item == item {
			entry.waitPlayback = false
			entry.cancelled = true
		}
	}
}

func (d *realtimeDelivery) snapshot() (pending, awaiting int, data map[string]any) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, item := range d.items {
		if item.waitPlayback && !item.cancelled {
			awaiting += max(0, item.queued-item.dropped-item.played)
		}
	}
	return d.pending, awaiting, map[string]any{
		"queued_audio_ms":            float64(d.queued) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
		"pending_audio_ms":           float64(d.pending) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
		"bridge_written_audio_ms":    float64(d.written) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
		"acknowledged_audio_ms":      float64(d.played) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
		"dropped_audio_ms":           float64(d.dropped) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
		"playback_ack_supported":     d.supportsPlayback,
		"playback_evidence":          d.evidence,
		"audible_playback_confirmed": d.evidence == "carrier_mark" && d.played > 0 && d.played >= d.written && d.pending == 0 && awaiting == 0 && d.dropped == 0,
		"awaiting_playback_ms":       float64(awaiting) * 1000 / float64(defaultMediaRate(d.bytesPerSecond)),
	}
}

func (rt *RealtimeThinker) completionRequested() bool {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	return rt.completion.requested
}

func (rt *RealtimeThinker) requestCompletion(callID string) {
	rt.requestCompletionWithClosing(callID, true)
}

func (rt *RealtimeThinker) requestCompletionWithClosing(callID string, closingNeeded bool) {
	rt.lifecycleMu.Lock()
	if rt.completion.requested {
		rt.lifecycleMu.Unlock()
		return
	}
	rt.completion.requested = true
	rt.completion.requestedAt = time.Now()
	rt.completion.initiator, rt.completion.callID = "model", callID
	rt.completion.awaitOutput = rt.capabilities.ContinuousConversation && closingNeeded
	rt.terminalReason = "done_tool"
	if !closingNeeded {
		rt.completion.initiator = "application_review"
		rt.terminalReason = "completion_review"
	}
	rt.lifecycleMu.Unlock()
	if rt.capabilities.ContinuousConversation && closingNeeded {
		_ = rt.appendRealtimeContext(RealtimeContextUpdate{Kind: "instructions", Content: "The conversation is ending. Give a brief closing in the caller's language, using only verified facts, then stop speaking and listen. Do not start new backend work."})
	}
	rt.emit("realtime.completion_requested", rt.shutdownData())
	rt.setConversationState("finishing", RealtimeEvent{})
	rt.wakeRecovery()
}

func (rt *RealtimeThinker) noteCompletionEvent(event RealtimeEvent) {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	if event.Type == RealtimeEventAudioOut || event.Type == RealtimeEventTranscriptOutput || event.Type == RealtimeEventResponseStarted {
		if rt.completion.requested && (event.Type == RealtimeEventAudioOut || rt.audioOut == nil) {
			rt.completion.awaitOutput = false
		}
		rt.completion.responseID, rt.completion.itemID = event.ResponseID, event.ItemID
		if !event.Final {
			rt.completion.utteranceDone = false
		}
	}
	if event.Type == RealtimeEventResponseDone || event.Type == RealtimeEventUtteranceDone {
		rt.completion.utteranceDone = true
	}
}

// Finishing consumes provider evidence and drains existing output without
// accepting new input, launching tools, reconnecting, or requesting responses.
func (rt *RealtimeThinker) finishRealtimeCompletion(session RealtimeSession) {
	rt.lifecycleMu.Lock()
	timeout := rt.completion.timeout
	if timeout <= 0 {
		timeout = realtimeCompletionTimeout
	}
	deadline := rt.completion.requestedAt.Add(timeout)
	rt.lifecycleMu.Unlock()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	var settledAt time.Time
	closing, finalized := false, false
	var events <-chan RealtimeEvent
	if session != nil && !rt.audioLane.active.Load() {
		events = session.Events()
	}
	for {
		rt.flushAudioTelemetry()
		select {
		case <-rt.quit:
			rt.setCompletionOutcome("forced_stop")
			return
		case <-rt.ctx.Done():
			rt.setCompletionOutcome("forced_stop")
			return
		default:
		}
		rt.lifecycleMu.Lock()
		connected, done, awaitOutput := rt.bridgeConnected, rt.completion.utteranceDone, rt.completion.awaitOutput
		rt.lifecycleMu.Unlock()
		// A missing bridge is also valid for text-only workers. Never drain audio
		// to a peer that has disconnected; retain the evidence before closing.
		if !connected && rt.audioOut != nil {
			rt.setCompletionOutcome("disconnected")
			return
		}
		pending, awaiting, _ := rt.delivery.snapshot()
		if rt.capabilities.ContinuousConversation {
			pending, awaiting = rt.delivery.pendingSpeechPlayback()
			done = rt.continuous.lastAudio.IsZero() || time.Since(rt.continuous.lastAudio) >= realtimeObservedQuiet
		}
		ready := !awaitOutput && (done || !rt.responseInProgress() || session == nil) && pending == 0 && awaiting == 0 && !rt.backend.active
		if ready {
			if settledAt.IsZero() {
				settledAt = time.Now()
			}
			if time.Since(settledAt) >= realtimeCompletionQuietPeriod {
				if graceful, ok := session.(RealtimeGracefulSession); ok && !finalized {
					if !closing {
						if err := graceful.BeginClose(); err != nil {
							rt.setCompletionOutcome("timed_out")
							return
						}
						closing = true
					}
				} else {
					rt.setCompletionOutcome("completed")
					return
				}
			}
		} else {
			settledAt = time.Time{}
		}
		if !time.Now().Before(deadline) {
			rt.setCompletionOutcome("timed_out")
			return
		}
		select {
		case <-rt.quit:
			rt.setCompletionOutcome("forced_stop")
			return
		case <-rt.ctx.Done():
			rt.setCompletionOutcome("forced_stop")
			return
		case <-rt.recovery.wake:
		case <-ticker.C:
		case <-rt.audioLane.wake:
			rt.observeAudioLane()
		case delivered := <-rt.audioLane.events:
			if delivered.session != rt.currentSession() {
				continue
			}
			if delivered.closed {
				if rt.capabilities.ContinuousConversation && !finalized {
					rt.setCompletionOutcome("disconnected")
					return
				}
				session = nil
				continue
			}
			rt.observeAudioLane()
			if delivered.event.Type == RealtimeEventSessionEnded {
				finalized = true
			}
			rt.handleSessionEvent(delivered.event)
			settledAt = time.Time{}
		case event, ok := <-events:
			if !ok {
				events = nil
				if rt.capabilities.ContinuousConversation && !finalized {
					rt.setCompletionOutcome("disconnected")
					return
				}
				if closing && !finalized {
					rt.setCompletionOutcome("timed_out")
					return
				}
				session = nil
				continue
			}
			if event.Type == RealtimeEventSessionEnded {
				finalized = true
			}
			rt.handleSessionEvent(event)
			settledAt = time.Time{}
		}
	}
}

func (rt *RealtimeThinker) setCompletionOutcome(outcome string) {
	rt.lifecycleMu.Lock()
	rt.completion.outcome = outcome
	rt.lifecycleMu.Unlock()
}

func (rt *RealtimeThinker) shutdownData() map[string]any {
	rt.lifecycleMu.Lock()
	reason, generation, connected, completion := rt.terminalReason, rt.sessionGeneration, rt.bridgeConnected, rt.completion
	pendingWork := append([]ThreadWork(nil), rt.terminalWork...)
	pendingTools := rt.terminalToolsPending
	rt.lifecycleMu.Unlock()
	_, _, data := rt.delivery.snapshot()
	initiator := completion.initiator
	if reason != "done_tool" && reason != "completion_review" {
		initiator = "core"
	}
	if initiator == "" {
		initiator = "core"
	}
	data["trigger"], data["reason"], data["initiator"] = reason, reason, initiator
	data["generation"], data["bridge_connected"] = generation, connected
	data["tool_call_id"] = completion.callID
	data["response_id"], data["item_id"] = completion.responseID, completion.itemID
	data["pending_transcripts"] = rt.pendingTranscriptCount()
	data["pending_assignments"], data["pending_tool_work"] = pendingWork, pendingTools
	data["graceful_completion_requested"] = completion.requested
	data["graceful_completion_outcome"] = completion.outcome
	if completion.requested {
		data["finishing_ms"] = time.Since(completion.requestedAt).Milliseconds()
	}
	return data
}

func (rt *RealtimeThinker) finishShutdown() {
	rt.shutdownOnce.Do(func() {
		data := rt.shutdownData()
		reason, _ := data["reason"].(string)
		rt.flushPendingTranscripts(reason)
		// Emit before the bridge is unregistered. This is a stored lifecycle event.
		rt.emitDurableLifecycle("realtime.thread_ended", data)
	})
}

func (rt *RealtimeThinker) emitDurableLifecycle(eventType string, data map[string]any) {
	if rt.telemetry != nil {
		if err := rt.telemetry.EmitDurable(eventType, rt.threadID, data); err != nil {
			logMsg("REALTIME", "persist realtime lifecycle receipt: "+err.Error())
			rt.emit(eventType, data)
		}
	}
}

func (rt *RealtimeThinker) recordSessionClosure(reason string, planned bool, extra map[string]any) {
	data := rt.shutdownData()
	data["trigger"], data["reason"], data["planned"], data["initiator"] = reason, reason, planned, "core"
	if reason == "provider_goaway" || reason == "provider_session_closed" || reason == "provider_session_ended" {
		data["initiator"] = "provider"
	}
	for key, value := range extra {
		data[key] = value
	}
	rt.emitDurableLifecycle("realtime.session_closed", data)
}

func (d *realtimeDelivery) isCancelled(key string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.items[key] != nil && d.items[key].cancelled
}

// Used for presentation only. Socket delivery without a client receipt never
// becomes a claim that the caller heard the speech.
func (d *realtimeDelivery) hasPendingPlayback() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, item := range d.items {
		if !item.cancelled && (item.speechPending > 0 || item.waitPlayback && item.played+item.dropped < item.speechEnd) {
			return true
		}
	}
	return false
}

func (d *realtimeDelivery) pendingSpeechPlayback() (pending, awaiting int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, item := range d.items {
		if item.cancelled {
			continue
		}
		pending += item.speechPending
		if item.waitPlayback {
			awaiting += max(0, item.speechEnd-item.played-item.dropped)
		}
	}
	return
}

func (d *realtimeDelivery) playbackPosition(item string) (queuedMS, playedMS int, active bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, entry := range d.items {
		if entry.item != item || entry.cancelled {
			continue
		}
		queuedMS += entry.queued * 1000 / defaultMediaRate(d.bytesPerSecond)
		playedMS += entry.played * 1000 / defaultMediaRate(d.bytesPerSecond)
		active = active || entry.pending > 0 || entry.waitPlayback && entry.played+entry.dropped < entry.queued
	}
	return
}

func (d *realtimeDelivery) pendingPlaybackIDs() []string {
	d.mu.Lock()
	defer d.mu.Unlock()
	seen := map[string]bool{}
	var ids []string
	for _, entry := range d.items {
		if !entry.cancelled && entry.item != "" && (entry.pending > 0 || entry.waitPlayback && entry.played+entry.dropped < entry.queued) && !seen[entry.item] {
			seen[entry.item] = true
			ids = append(ids, entry.item)
		}
	}
	return ids
}

func (d *realtimeDelivery) setEvidence(e string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if e == "" {
		e = "legacy"
	}
	d.evidence = e
}

func (d *realtimeDelivery) generationFor(item string, fallback int64) int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	if entry := d.items[strconv.FormatInt(fallback, 10)+"/"+item]; entry != nil {
		return fallback
	}
	for key, e := range d.items {
		if e.item == item {
			prefix, _, _ := strings.Cut(key, "/")
			if generation, err := strconv.ParseInt(prefix, 10, 64); err == nil {
				return generation
			}
		}
	}
	return fallback
}

// An explicit lifecycle signal on the authenticated bridge, distinct from an
// ordinary normal close (which might be replacement or operator cleanup).
func (rt *RealtimeThinker) completionCloseReason() string {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	if !rt.completion.requested || (rt.terminalReason != "done_tool" && rt.terminalReason != "completion_review") {
		return ""
	}
	switch rt.completion.outcome {
	case "completed":
		return "realtime conversation completed"
	case "timed_out":
		return "realtime completion timed out"
	default:
		return ""
	}
}
