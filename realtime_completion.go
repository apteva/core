package core

import (
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
	outcome            string
}

type realtimeDeliveredItem struct {
	item                                      string
	queued, pending, written, played, dropped int
	waitPlayback                              bool
}

// Tracks every queued frame, including one removed from the channel while the
// bridge is still writing it. Written audio and renderer acknowledgements are
// distinct evidence. No wall-clock estimate is substituted for playback.
type realtimeDelivery struct {
	mu                                        sync.Mutex
	items                                     map[string]*realtimeDeliveredItem
	supportsPlayback                          bool
	queued, pending, written, played, dropped int
}

func (d *realtimeDelivery) enqueue(key, item string, bytes int) func(bool) {
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
	entry.pending += bytes
	d.queued += bytes
	d.pending += bytes
	d.mu.Unlock()
	var once sync.Once
	return func(success bool) {
		once.Do(func() {
			d.mu.Lock()
			defer d.mu.Unlock()
			entry.pending -= bytes
			d.pending -= bytes
			if success {
				entry.written += bytes
				d.written += bytes
			} else {
				entry.dropped += bytes
				d.dropped += bytes
			}
		})
	}
}

func (d *realtimeDelivery) acknowledge(key string, endMS int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	entry := d.items[key]
	if entry == nil {
		return
	}
	d.supportsPlayback = true
	entry.waitPlayback = true
	played := min(entry.queued, endMS*realtimePCMBytesPerSecond/1000)
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
		entry.waitPlayback = true
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
		}
	}
}

func (d *realtimeDelivery) snapshot() (pending, awaiting int, data map[string]any) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, item := range d.items {
		if item.waitPlayback {
			awaiting += max(0, item.queued-item.dropped-item.played)
		}
	}
	return d.pending, awaiting, map[string]any{
		"queued_audio_ms":         float64(d.queued) * 1000 / realtimePCMBytesPerSecond,
		"pending_audio_ms":        float64(d.pending) * 1000 / realtimePCMBytesPerSecond,
		"bridge_written_audio_ms": float64(d.written) * 1000 / realtimePCMBytesPerSecond,
		"acknowledged_audio_ms":   float64(d.played) * 1000 / realtimePCMBytesPerSecond,
		"dropped_audio_ms":        float64(d.dropped) * 1000 / realtimePCMBytesPerSecond,
		"playback_ack_supported":  d.supportsPlayback,
	}
}

func (rt *RealtimeThinker) completionRequested() bool {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	return rt.completion.requested
}

func (rt *RealtimeThinker) requestCompletion(callID string) {
	rt.lifecycleMu.Lock()
	if rt.completion.requested {
		rt.lifecycleMu.Unlock()
		return
	}
	rt.completion.requested = true
	rt.completion.requestedAt = time.Now()
	rt.completion.initiator, rt.completion.callID = "model", callID
	rt.terminalReason = "done_tool"
	rt.lifecycleMu.Unlock()
	rt.emit("realtime.completion_requested", rt.shutdownData())
	rt.setConversationState("finishing", RealtimeEvent{})
	rt.wakeRecovery()
}

func (rt *RealtimeThinker) noteCompletionEvent(event RealtimeEvent) {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	if event.Type == RealtimeEventAudioOut || event.Type == RealtimeEventTranscriptOutput || event.Type == RealtimeEventResponseStarted {
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
	var events <-chan RealtimeEvent
	if session != nil {
		events = session.Events()
	}
	for {
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
		connected, done := rt.bridgeConnected, rt.completion.utteranceDone
		rt.lifecycleMu.Unlock()
		// A missing bridge is also valid for text-only workers. Never drain audio
		// to a peer that has disconnected; retain the evidence before closing.
		if !connected && rt.audioOut != nil {
			rt.setCompletionOutcome("disconnected")
			return
		}
		pending, awaiting, _ := rt.delivery.snapshot()
		ready := (done || !rt.responseInProgress() || session == nil) && pending == 0 && awaiting == 0
		if ready {
			if settledAt.IsZero() {
				settledAt = time.Now()
			}
			if time.Since(settledAt) >= realtimeCompletionQuietPeriod {
				rt.setCompletionOutcome("completed")
				return
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
		case event, ok := <-events:
			if !ok {
				events = nil
				session = nil
				continue
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
	rt.lifecycleMu.Unlock()
	_, _, data := rt.delivery.snapshot()
	initiator := completion.initiator
	if reason != "done_tool" {
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
