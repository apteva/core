package core

import (
	"fmt"
	"time"
)

// Bounded, session-scoped observations, not task state. No transcripts, tool
// arguments or audio are retained here. All timestamps are Core observations.
type realtimeRequestTiming struct {
	work                                                              ThreadWork
	requested, started, available, submitted, acknowledged, speechEnd time.Time
	speechSource, status                                              string
}

func requestTimingKey(w ThreadWork) string { return fmt.Sprintf("%s/%d", w.ID, w.Revision) }

func (t *realtimeTiming) requestData(r *realtimeRequestTiming, stage string) map[string]any {
	return map[string]any{
		"generation": t.generation, "request_id": r.work.ID, "revision": r.work.Revision, "delegation_id": r.work.DelegationID,
		"result_observation": "core_result_received", "stage": stage, "status": r.status, "clock": "core_observed", "speech_end_source": r.speechSource,
		"speech_end_to_delegation_ms":   timingMS(r.speechEnd, r.requested),
		"delegation_to_execution_ms":    timingMS(r.requested, r.started),
		"execution_to_result_ms":        timingMS(r.started, r.available),
		"delegation_to_result_ms":       timingMS(r.requested, r.available),
		"result_to_submission_ms":       timingMS(r.available, r.submitted),
		"submission_to_context_ack_ms":  timingMS(r.submitted, r.acknowledged),
		"speech_end_to_useful_audio_ms": nil, "speech_end_to_useful_playback_report_ms": nil,
		"output_association": "unknown", "playback_confirmed": false,
	}
}

func (t *realtimeTiming) requestStage(w ThreadWork, stage string, at time.Time) {
	if t == nil || w.ID == "" {
		return
	}
	if at.IsZero() {
		at = time.Now()
	}
	var evicted map[string]any
	t.mu.Lock()
	if t.trackingClosed {
		t.mu.Unlock()
		return
	}
	if t.requests == nil {
		t.requests = map[string]*realtimeRequestTiming{}
	}
	key := requestTimingKey(w)
	r := t.requests[key]
	if r == nil {
		if len(t.requestOrder) >= 64 {
			old := t.requestOrder[0]
			t.requestOrder = t.requestOrder[1:]
			if r := t.requests[old]; r != nil {
				evicted = t.requestData(r, "tracking_evicted")
			}
			delete(t.requests, old)
		}
		r = &realtimeRequestTiming{work: w, status: "pending", speechEnd: t.lastSpeechEnd, speechSource: t.lastSpeechSource}
		t.requests[key] = r
		t.requestOrder = append(t.requestOrder, key)
	}
	if w.DelegationID != "" {
		r.work.DelegationID = w.DelegationID
	}
	changed := true
	switch stage {
	case "requested":
		if r.requested.IsZero() {
			r.requested = at
		} else {
			changed = false
		}
	case "started":
		if r.started.IsZero() {
			r.started = at
		} else {
			changed = false
		}
	case "result_available":
		if r.available.IsZero() {
			r.available = at
		} else {
			changed = false
		}
	case "submitted":
		if r.submitted.IsZero() {
			r.submitted = at
		} else {
			changed = false
		}
	case "backend_completed":
		r.status = "backend_completed"
	case "delivery_failed": // Failed admission is not a provider acknowledgement.
	default:
		r.status = stage
	}
	data := t.requestData(r, stage)
	t.mu.Unlock()
	if t.emit != nil {
		if evicted != nil {
			t.emit("realtime.request_timing", evicted)
		}
		if changed {
			t.emit("realtime.request_timing", data)
		}
	}
}

func (t *realtimeTiming) registerContext(id string, w ThreadWork) {
	if t == nil || id == "" || w.ID == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.trackingClosed {
		return
	}
	if t.contextRequests == nil {
		t.contextRequests = map[string]string{}
	}
	if len(t.contextOrder) >= 128 {
		delete(t.contextRequests, t.contextOrder[0])
		t.contextOrder = t.contextOrder[1:]
	}
	t.contextRequests[id] = requestTimingKey(w)
	t.contextOrder = append(t.contextOrder, id)
}
func (t *realtimeTiming) contextAcknowledged(id string, at time.Time) {
	if t == nil {
		return
	}
	if at.IsZero() {
		at = time.Now()
	}
	var data map[string]any
	t.mu.Lock()
	if r := t.requests[t.contextRequests[id]]; r != nil && r.acknowledged.IsZero() {
		r.acknowledged = at
		data = t.requestData(r, "context_acknowledged")
		data["client_event_id"] = id
	}
	delete(t.contextRequests, id)
	t.mu.Unlock()
	if data != nil && t.emit != nil {
		t.emit("realtime.request_timing", data)
	}
}
func (t *realtimeTiming) finishRequests(status string) {
	if t == nil {
		return
	}
	var data []map[string]any
	t.mu.Lock()
	t.trackingClosed = true
	for _, r := range t.requests {
		d := t.requestData(r, "tracking_closed")
		d["close_reason"] = status
		if r.status == "pending" {
			d["status"] = status
		}
		data = append(data, d)
	}
	t.requests = nil
	t.requestOrder = nil
	t.contextRequests = nil
	t.contextOrder = nil
	t.mu.Unlock()
	if t.emit != nil {
		for _, d := range data {
			t.emit("realtime.request_timing", d)
		}
	}
}

// Output identity must be supplied explicitly by an adapter, never inferred
// from whichever task happened to finish most recently. Without that evidence,
// non-silent audio is measured but must not be counted as a useful answer.
func (t *realtimeTiming) outputObservation(e RealtimeEvent, voiced bool, at time.Time) {
	if t == nil {
		return
	}
	if at.IsZero() {
		at = time.Now()
	}
	item := e.PlaybackID
	if item == "" {
		item = e.ItemID
	}
	var caption map[string]any
	t.mu.Lock()
	defer func() {
		t.mu.Unlock()
		if caption != nil && t.emit != nil {
			t.emit("realtime.caption_timing", caption)
		}
	}()
	u := t.entry(e.ResponseID, item)
	if u == nil && e.Type == RealtimeEventTranscriptOutput {
		u = t.completed[realtimeTimingKey(e.ResponseID, item)]
	}
	if u == nil {
		return
	}
	if e.OutputWork != nil && (e.OutputPurpose == "answer" || e.OutputPurpose == "backchannel") {
		if r := t.requests[requestTimingKey(*e.OutputWork)]; r != nil && r.status != "superseded" && (e.OutputWork.DelegationID == "" || e.OutputWork.DelegationID == r.work.DelegationID) {
			u.work = r.work
			u.purpose = e.OutputPurpose
			u.requestSpeechEnd = r.speechEnd
			u.resultSubmitted = r.submitted
		}
	}
	if e.Type == RealtimeEventTranscriptOutput && u.firstCaption.IsZero() {
		u.firstCaption = at
		caption = map[string]any{"generation": t.generation, "response_id": u.response, "item_id": item, "playback_id": item, "speech_end_to_caption_ms": timingMS(u.speechEnd, at), "playback_confirmed": false}
		t.outputData(u, caption)
	}
	if e.Type == RealtimeEventAudioOut && voiced && u.firstNonSilent.IsZero() {
		u.firstNonSilent = at
		u.nonSilentEndMS = e.AudioEndMS
	}
}

func (t *realtimeTiming) outputData(u *realtimeUtteranceTiming, d map[string]any) {
	association := "unknown"
	purpose := u.purpose
	if purpose == "" {
		purpose = "unknown"
	}
	if u.work.ID != "" {
		association = "explicit"
		r := t.requests[requestTimingKey(u.work)]
		if r == nil {
			association = "unavailable"
		} else if r.status == "superseded" {
			association = "superseded"
		}
	}
	d["output_association"] = association
	d["output_purpose"] = purpose
	d["request_id"] = u.work.ID
	d["revision"] = u.work.Revision
	d["delegation_id"] = u.work.DelegationID
	d["first_audio_to_non_silent_ms"] = timingMS(u.firstAudio, u.firstNonSilent)
	d["speech_end_to_non_silent_audio_ms"] = timingMS(u.speechEnd, u.firstNonSilent)
	d["provider_audio_to_caption_ms"] = timingMS(u.firstAudio, u.firstCaption)
	d["submission_to_related_audio_ms"] = timingMS(u.resultSubmitted, u.firstAudio)
	d["speech_end_to_useful_audio_ms"] = nil
	if t.usefulOutput(u) {
		d["speech_end_to_useful_audio_ms"] = timingMS(u.requestSpeechEnd, u.firstNonSilent)
	}
	d["audio_activity_evidence"] = "decoded_audio_amplitude_threshold" // Activity, not semantic speech recognition.
}

// Called with the tracker lock. A corrected assignment cannot inherit an old
// answer's latency even if its audio or renderer receipt arrives late.
func (t *realtimeTiming) usefulOutput(u *realtimeUtteranceTiming) bool {
	if u.purpose != "answer" || u.work.ID == "" {
		return false
	}
	r := t.requests[requestTimingKey(u.work)]
	return r != nil && r.status != "superseded" && u.status != "interrupted" && u.status != "blocked"
}
