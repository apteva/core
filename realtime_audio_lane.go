package core

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

type realtimeLaneEvent struct {
	session RealtimeSession
	event   RealtimeEvent
	closed  bool
}
type realtimeLaneFailure struct {
	session RealtimeSession
	reason  string
	err     error
}

// The audio lane never persists transcripts, starts backend work or emits telemetry.
// Non-audio events stay ordered on the actor, with a bounded failure policy.
type realtimeLaneTelemetry struct {
	kind string
	data map[string]any
}
type realtimeAudioLane struct {
	active           atomic.Bool
	telemetry        chan realtimeLaneTelemetry
	telemetryDropped atomic.Uint64
	failed           RealtimeSession
	tools            atomic.Value // immutable []NativeTool snapshots
	blocked          map[string]bool
	tails            map[string]string
	events           chan realtimeLaneEvent
	failures         chan realtimeLaneFailure
	wake             chan struct{}
	done             chan struct{}
	mu               sync.Mutex
	latest           *RealtimeEvent
	session          RealtimeSession
}

func (rt *RealtimeThinker) initAudioLane() {
	rt.audioLane.telemetry = make(chan realtimeLaneTelemetry, 1024)
	rt.audioLane.events = make(chan realtimeLaneEvent, 256)
	rt.audioLane.failures = make(chan realtimeLaneFailure, 8)
	rt.audioLane.wake = make(chan struct{}, 1)
	rt.audioLane.done = make(chan struct{})
}
func (rt *RealtimeThinker) startAudioLane(ctx context.Context) {
	rt.audioLane.tools.Store(rt.realtimeToolSchemas())
	rt.audioLane.active.Store(true)
	go func() { defer close(rt.audioLane.done); rt.runAudioLane(ctx) }()
}
func (rt *RealtimeThinker) stopAudioLane() { <-rt.audioLane.done }
func (rt *RealtimeThinker) failAudioLane(s RealtimeSession, reason string, err error) {
	if !rt.audioLane.active.Load() {
		rt.emit("realtime.audio_overflow", map[string]any{"reason": reason, "error": err.Error()})
		return
	}
	rt.audioLane.mu.Lock()
	if rt.audioLane.failed == s {
		rt.audioLane.mu.Unlock()
		return
	}
	rt.audioLane.failed = s
	rt.audioLane.mu.Unlock()
	select {
	case rt.audioLane.failures <- realtimeLaneFailure{s, reason, err}:
	default:
	}
}
func (rt *RealtimeThinker) runAudioLane(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Millisecond)
	defer ticker.Stop()
	var source RealtimeSession
	var generation int64
	var events <-chan RealtimeEvent
	closed := false
	for {
		current := rt.currentSession()
		if current != source {
			source = current
			generation = generationForAudioFrame(rt)
			events = nil
			closed = false
			if source != nil {
				events = source.Events()
			}
			rt.audioLane.mu.Lock()
			rt.audioLane.blocked = map[string]bool{}
			rt.audioLane.tails = map[string]string{}
			rt.audioLane.mu.Unlock()
			rt.input.resetSender()
		}
		rt.audioLane.mu.Lock()
		failed := source != nil && rt.audioLane.failed == source
		rt.audioLane.mu.Unlock()
		receive := func(e RealtimeEvent, ok bool) {
			if source != rt.currentSession() {
				return
			}
			if !ok {
				closed = true
				events = nil
				select {
				case rt.audioLane.events <- realtimeLaneEvent{session: source, closed: true}:
				default:
					rt.failAudioLane(source, "control_event_backlog", fmt.Errorf("realtime actor event backlog exceeded"))
				}
				return
			}
			rt.acceptAudioLaneEvent(source, generation, e)
		}
		select {
		case <-ctx.Done():
			return
		case e, ok := <-events:
			receive(e, ok)
		case <-ticker.C:
			// Process already-received session closure/cancellation before admitting
			// more input. A closed old session must not consume replacement input.
			for i := 0; i < 32; i++ {
				select {
				case e, ok := <-events:
					receive(e, ok)
				default:
					i = 32
				}
			}
			if source != nil && !closed && !failed {
				rt.drainRealtimeInput(source)
			}
			rt.pumpRealtimeOutput()
		}
	}
}
func (rt *RealtimeThinker) acceptAudioLaneEvent(source RealtimeSession, generation int64, e RealtimeEvent) {
	if e.Type == RealtimeEventAudioOut {
		rt.noteCompletionEvent(e)
		rt.responseStarted()
		rt.recovery.Lock()
		rt.recovery.lastOutput = time.Now()
		rt.recovery.outputBytes += uint64(len(e.Audio))
		rt.recovery.Unlock()
		if e.ItemID == "" && rt.capabilities.TranscriptFragments {
			e.ItemID = fmt.Sprintf("stream-%d", generation)
		}
		if e.ObservedAt.IsZero() {
			e.ObservedAt = time.Now()
		}
		accepted := rt.forwardRealtimeAudio(e, generation)
		if !accepted || e.AudioSilence {
			return
		}
		rt.markAssistantAudioEmitted()
		e.Audio = nil
		e.audioForwarded = true
		rt.audioLane.mu.Lock()
		rt.audioLane.latest = &e
		rt.audioLane.session = source
		rt.audioLane.mu.Unlock()
		select {
		case rt.audioLane.wake <- struct{}{}:
		default:
		}
		return
	}
	if e.Type == RealtimeEventTranscriptOutput {
		rt.preflightAudioTranscript(e)
	}
	if e.Type == RealtimeEventSpeechStarted {
		rt.clearCallerInterruptedPlayback("provider_speech_started", false, false)
		e.audioForwarded = true
	}
	select {
	case rt.audioLane.events <- realtimeLaneEvent{session: source, event: e}:
	default:
		rt.failAudioLane(source, "control_event_backlog", fmt.Errorf("realtime actor event backlog exceeded"))
	}
}
func (rt *RealtimeThinker) observeAudioLane() {
	rt.audioLane.mu.Lock()
	e, s := rt.audioLane.latest, rt.audioLane.session
	rt.audioLane.latest = nil
	rt.audioLane.mu.Unlock()
	if e != nil && s == rt.currentSession() {
		rt.handleSessionEvent(*e)
	}
}

func (rt *RealtimeThinker) forwardRealtimeAudio(event RealtimeEvent, generation int64) bool {
	if rt.outputPaused.Load() || !rt.audioLane.active.Load() && rt.paused || generation != generationForAudioFrame(rt) {
		rt.currentTiming().dropped(event.ResponseID, event.ItemID, len(event.Audio))
		return false
	}
	rt.audioLane.mu.Lock()
	blocked := rt.audioLane.blocked[realtimeToolMarkupResponseKey(event)]
	rt.audioLane.mu.Unlock()
	if blocked || rt.toolMarkupResponseSuppressed(event) {
		rt.currentTiming().dropped(event.ResponseID, event.ItemID, len(event.Audio))
		rt.emit("realtime.audio_drop", map[string]any{
			"direction": "output", "bytes": len(event.Audio), "reason": "provider_tool_markup_leaked",
			"response_id": event.ResponseID, "item_id": event.ItemID,
		})
		return false
	}
	if rt.audioOut != nil {
		rt.outputMu.Lock()
		if rt.cancelledPlaybacks[event.ItemID] || rt.suppressOutput && (event.ItemID == "" || event.ItemID == rt.interruptedItem) {
			rt.outputMu.Unlock()
			rt.currentTiming().dropped(event.ResponseID, event.ItemID, len(event.Audio))
			rt.emit("realtime.audio_drop", map[string]any{"direction": "output", "bytes": len(event.Audio), "reason": "interrupted_item"})
			return false
		}
		if rt.suppressOutput && event.ItemID != "" && event.ItemID != rt.interruptedItem {
			rt.suppressOutput = false
			rt.interruptedItem = ""
		}
		if event.ItemID != "" && event.ItemID != rt.outputItemID {
			rt.outputItemID, rt.outputBytes = event.ItemID, 0
			rt.playedItemID, rt.playedMS = event.ItemID, 0
			rt.playbackTracked = false
		}
		endMS := (rt.outputBytes + len(event.Audio)) * 1000 / realtimePCMBytesPerSecond
		frame := RealtimeAudioFrame{timing: rt.currentTiming(), Audio: event.Audio, SessionGeneration: generation, queuedAt: time.Now(), ResponseID: event.ResponseID, ItemID: event.ItemID, AudioEndMS: endMS, Silent: event.AudioSilence}
		itemKey := realtimeTimingKey(event.ResponseID, event.ItemID)
		deliveryKey := fmt.Sprintf("%d/%s", generation, itemKey)
		frame.settle = rt.delivery.enqueueBytes(deliveryKey, event.ItemID, len(event.Audio), event.AudioSilence)
		frame.cancelled = func() bool { return generation != generationForAudioFrame(rt) || rt.delivery.isCancelled(deliveryKey) }
		rt.outputBytes += len(event.Audio)
		frame.timing.enqueued(event.ResponseID, event.ItemID, time.Now(), min(cap(rt.audioOut), len(rt.audioOut)+1), len(event.Audio))
		accepted := rt.queueRealtimeOutputLocked(frame)
		rt.outputMu.Unlock()
		if !accepted {
			frame.timing.written(event.ResponseID, event.ItemID, time.Time{}, len(event.Audio), false)
			frame.written(false)
			rt.failAudioLane(rt.currentSession(), "output_duration_limit", fmt.Errorf("bounded output backlog exceeded"))
			return false
		}
	}
	return true
}

const realtimeOutputBufferDuration = 15 * time.Second

// Includes the public channel and any frame currently being written by the bridge.
func (rt *RealtimeThinker) queueRealtimeOutputLocked(frame RealtimeAudioFrame) bool {
	pending, _, _ := rt.delivery.snapshot()
	if pending > int(realtimeOutputBufferDuration/time.Second)*realtimePCMBytesPerSecond {
		return false
	}
	if len(rt.outputPending) == 0 {
		select {
		case rt.audioOut <- frame:
			return true
		default:
		}
	}
	rt.outputPending = append(rt.outputPending, frame)
	rt.outputPendingBytes += len(frame.Audio)
	return true
}
func (rt *RealtimeThinker) pumpRealtimeOutput() {
	rt.outputMu.Lock()
	defer rt.outputMu.Unlock()
	for len(rt.outputPending) > 0 {
		f := rt.outputPending[0]
		if f.cancelled != nil && f.cancelled() {
			f.timing.written(f.ResponseID, f.ItemID, time.Time{}, len(f.Audio), false)
			f.written(false)
		} else {
			if time.Since(f.queuedAt) > realtimeOutputBufferDuration {
				rt.failAudioLane(rt.currentSession(), "output_age_limit", fmt.Errorf("output exceeded maximum queue age"))
				return
			}
			select {
			case rt.audioOut <- f:
			default:
				return
			}
		}
		rt.outputPending[0] = RealtimeAudioFrame{}
		rt.outputPending = rt.outputPending[1:]
		rt.outputPendingBytes -= len(f.Audio)
	}
}

func (rt *RealtimeThinker) flushAudioTelemetry() {
	if n := rt.audioLane.telemetryDropped.Swap(0); n > 0 && rt.telemetry != nil {
		rt.telemetry.Emit("realtime.telemetry_drop", rt.threadID, map[string]any{"events": n, "reason": "audio_telemetry_backlog"})
	}
	for i := 0; i < 1024; i++ {
		select {
		case e := <-rt.audioLane.telemetry:
			if rt.telemetry != nil {
				rt.telemetry.Emit(e.kind, rt.threadID, e.data)
			}
		default:
			return
		}
	}
}

// Fast safety preflight preserves transcript-before-audio ordering even while
// the actor is waiting on storage. Durable transcript updates and corrective
// model work remain actor-owned. No transcript or tool registry lock is taken.
func (rt *RealtimeThinker) preflightAudioTranscript(e RealtimeEvent) {
	key := realtimeToolMarkupResponseKey(e)
	rt.audioLane.mu.Lock()
	if rt.audioLane.blocked[key] {
		rt.audioLane.mu.Unlock()
		return
	}
	text := rt.audioLane.tails[key] + e.Transcript
	if len(rt.audioLane.tails) >= 128 {
		rt.audioLane.tails = map[string]string{}
	}
	rt.audioLane.tails[key] = boundedRealtimeToolMarkupTail(text)
	rt.audioLane.mu.Unlock()
	var tools []NativeTool
	if v := rt.audioLane.tools.Load(); v != nil {
		tools = v.([]NativeTool)
	}
	_, _, leaked := detectRealtimeToolMarkup(text, tools)
	if !leaked && detectRealtimeInternalNarration(text) == "" {
		return
	}
	rt.audioLane.mu.Lock()
	if len(rt.audioLane.blocked) >= 128 {
		rt.audioLane.blocked = map[string]bool{}
	}
	rt.audioLane.blocked[key] = true
	rt.audioLane.mu.Unlock()
	item := e.PlaybackID
	if item == "" {
		item = e.ItemID
	}
	rt.interruptPlayback("provider_tool_markup_leaked", item, true, true)
}
