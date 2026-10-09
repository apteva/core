package core

import (
	"sync"
	"time"
)

// A session-local, bounded timeline. Audio callbacks only mutate small metadata
// under a mutex; they never emit telemetry, fetch data or wait on channels.
// Bridges retain their session's tracker so a reconnect cannot misattribute a
// frame to the new session generation.
type realtimeTiming struct {
	mu              sync.Mutex
	generation      uint64
	connectedAt     time.Time
	readyEmitted    bool
	emit            func(string, map[string]any)
	items           map[string]*realtimeUtteranceTiming
	closed          map[string]bool
	closedOrder     []string
	completed       map[string]*realtimeUtteranceTiming
	speechEnd       time.Time
	speechEndSource string
}

type realtimeUtteranceTiming struct {
	response, item                                            string
	firstAudio, guardRelease, enqueue, bridgeWrite, speechEnd time.Time
	speechSource, guard, status                               string
	bufferedBytes, droppedBytes, queueMax, pending            int
	pendingBytes                                              int
	done                                                      bool
	acknowledged                                              bool
}

func newRealtimeTiming(generation uint64, emit func(string, map[string]any)) *realtimeTiming {
	return &realtimeTiming{generation: generation, emit: emit, items: map[string]*realtimeUtteranceTiming{}, closed: map[string]bool{}, completed: map[string]*realtimeUtteranceTiming{}}
}

func realtimeTimingKey(response, item string) string {
	if item != "" {
		return item
	}
	return response
}

func (t *realtimeTiming) entry(response, item string) *realtimeUtteranceTiming {
	key := realtimeTimingKey(response, item)
	if key == "" || t.closed[key] {
		return nil
	}
	u := t.items[key]
	if u == nil && len(t.items) < 64 {
		u = &realtimeUtteranceTiming{response: response, item: item, guard: "unavailable", status: "completed"}
		t.items[key] = u
	}
	return u
}

// Text-only and rejected utterances still receive a summary with unavailable
// audio boundaries. Observing a transcript must never invent first-audio time.
func (t *realtimeTiming) observed(response, item string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.entry(response, item)
	t.mu.Unlock()
}

func (t *realtimeTiming) speechStopped(at time.Time, source string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.speechEnd, t.speechEndSource = at, source
	t.mu.Unlock()
}

func (t *realtimeTiming) audio(response, item string, at time.Time) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if u := t.entry(response, item); u != nil && u.firstAudio.IsZero() {
		u.firstAudio = at
		u.speechEnd, u.speechSource = t.speechEnd, t.speechEndSource
		t.speechEnd, t.speechEndSource = time.Time{}, ""
	}
}

func (t *realtimeTiming) released(response, item string, at time.Time, buffered int, guard string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if u := t.entry(response, item); u != nil {
		if u.guardRelease.IsZero() {
			u.guardRelease = at
		}
		if buffered > u.bufferedBytes {
			u.bufferedBytes = buffered
		}
		u.guard = guard
	}
}

func (t *realtimeTiming) dropped(response, item string, bytes int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if u := t.entry(response, item); u != nil {
		u.droppedBytes += bytes
	}
}

func (t *realtimeTiming) enqueued(response, item string, at time.Time, depth int, frameBytes ...int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if u := t.entry(response, item); u != nil {
		if u.enqueue.IsZero() {
			u.enqueue = at
		}
		u.pending++
		if len(frameBytes) > 0 {
			u.pendingBytes += frameBytes[0]
		}
		if depth > u.queueMax {
			u.queueMax = depth
		}
	}
}

func timingMS(start, end time.Time) any {
	if start.IsZero() || end.IsZero() || end.Before(start) {
		return nil
	}
	return float64(end.Sub(start)) / float64(time.Millisecond)
}

func (t *realtimeTiming) summary(key string, u *realtimeUtteranceTiming) map[string]any {
	delete(t.items, key)
	t.closed[key] = true
	t.completed[key] = u
	t.closedOrder = append(t.closedOrder, key)
	if len(t.closedOrder) > 64 {
		delete(t.closed, t.closedOrder[0])
		delete(t.completed, t.closedOrder[0])
		t.closedOrder = t.closedOrder[1:]
	}
	return map[string]any{
		"generation": t.generation, "response_id": u.response, "item_id": u.item,
		"speech_end_to_provider_audio_ms": timingMS(u.speechEnd, u.firstAudio), "speech_end_source": u.speechSource,
		"provider_audio_to_guard_release_ms": timingMS(u.firstAudio, u.guardRelease),
		"guard_release_to_output_enqueue_ms": timingMS(u.guardRelease, u.enqueue),
		"output_enqueue_to_bridge_write_ms":  timingMS(u.enqueue, u.bridgeWrite),
		"provider_audio_to_bridge_write_ms":  timingMS(u.firstAudio, u.bridgeWrite),
		"buffered_audio_ms":                  float64(u.bufferedBytes) * 1000 / realtimePCMBytesPerSecond,
		"max_output_queue_depth":             u.queueMax, "dropped_audio_bytes": u.droppedBytes,
		"guard_outcome": u.guard, "completion_status": u.status,
	}
}

func (t *realtimeTiming) publish(data map[string]any) {
	if data != nil && t.emit != nil {
		t.emit("realtime.utterance_timing", data)
	}
}

func (t *realtimeTiming) written(response, item string, at time.Time, bytes int, success bool) {
	if t == nil {
		return
	}
	var data map[string]any
	t.mu.Lock()
	key := realtimeTimingKey(response, item)
	if u := t.items[key]; u != nil {
		if success && u.bridgeWrite.IsZero() {
			u.bridgeWrite = at
		}
		if !success {
			u.droppedBytes += bytes
			if u.status == "completed" {
				u.status = "output_dropped"
			}
		}
		u.pendingBytes = max(0, u.pendingBytes-bytes)
		if u.pending > 0 {
			u.pending--
		}
		if u.done && u.pending == 0 {
			data = t.summary(key, u)
		}
	}
	t.mu.Unlock()
	t.publish(data)
}

func (t *realtimeTiming) finish(response, item, status string) {
	if t == nil {
		return
	}
	var results []map[string]any
	t.mu.Lock()
	for key, u := range t.items {
		if (item != "" && key != item) || (item == "" && response != "" && u.response != response) {
			continue
		}
		u.done = true
		if status != "completed" || u.status == "completed" {
			u.status = status
		}
		if status == "blocked" {
			u.guard = "blocked"
		}
		if status != "completed" {
			u.droppedBytes += u.pendingBytes
			u.pendingBytes = 0
		}
		if u.pending == 0 || status != "completed" {
			results = append(results, t.summary(key, u))
		}
	}
	t.mu.Unlock()
	for _, data := range results {
		t.publish(data)
	}
}

// Optional client receipt, allowed after the utterance summary. Core's receive
// timestamp is local; this must never be described as carrier playback.
func (t *realtimeTiming) playback(item string, endMS int, at time.Time) {
	if t == nil || endMS <= 0 {
		return
	}
	var data map[string]any
	t.mu.Lock()
	u := t.items[item]
	if u == nil {
		u = t.completed[item]
	}
	if u != nil && !u.acknowledged {
		u.acknowledged = true
		data = map[string]any{"generation": t.generation, "response_id": u.response, "item_id": item,
			"source": "client_reported_progress_received", "audio_end_ms": endMS,
			"bridge_write_to_playback_report_ms": timingMS(u.bridgeWrite, at)}
	}
	t.mu.Unlock()
	if data != nil && t.emit != nil {
		t.emit("realtime.playback_timing", data)
	}
}

func (t *realtimeTiming) buffered(response, item string, bytes int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if u := t.entry(response, item); u != nil && bytes > u.bufferedBytes {
		u.bufferedBytes = bytes
	}
}

func (t *realtimeTiming) ready(at time.Time, settings map[string]any) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if t.readyEmitted {
		t.mu.Unlock()
		return
	}
	t.readyEmitted = true
	data := map[string]any{"generation": t.generation, "connection_to_ready_ms": timingMS(t.connectedAt, at), "applied": settings}
	t.mu.Unlock()
	if t.emit != nil {
		t.emit("realtime.session_ready", data)
	}
}
