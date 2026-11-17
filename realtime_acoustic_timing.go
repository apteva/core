package core

import (
	"encoding/binary"
	"time"
)

// Measurement only. Never use this detector to gate audio, commit a turn, or
// interrupt playback. Two consecutive 10ms RMS windows reject isolated peaks
// and make evidence independent of provider packet boundaries.
type realtimeAcousticDetector struct {
	squares              float64
	samples, consecutive int
	format               AudioFormat
	rate                 int
}

func (d *realtimeAcousticDetector) observe(audio []byte, format AudioFormat, rate int, threshold float64) bool {
	if format == "" {
		format = AudioPCM16
	}
	if rate <= 0 {
		rate = mediaBytesPerSecond(format)
		if format == AudioPCM16 {
			rate /= 2
		}
	}
	if format != AudioPCM16 && format != AudioG711ULaw && format != AudioG711ALaw {
		return false
	}
	if d.format != format || d.rate != rate {
		*d = realtimeAcousticDetector{format: format, rate: rate}
	}
	window := max(1, rate/100)
	active := false
	step := 1
	if format == AudioPCM16 {
		step = 2
	}
	for i := 0; i+step <= len(audio); i += step {
		var n int
		if step == 2 {
			n = int(int16(binary.LittleEndian.Uint16(audio[i:])))
		} else {
			n = mediaSample(audio[i], format)
		}
		d.squares += float64(n) * float64(n)
		d.samples++
		if d.samples == window {
			if d.squares/float64(window) > threshold*threshold {
				d.consecutive++
			} else {
				d.consecutive = 0
			}
			if d.consecutive >= 2 {
				active = true
				d.consecutive = 2
			}
			d.samples, d.squares = 0, 0
		}
	}
	return active
}

type realtimeInputObservation struct {
	tracker                      *realtimeTiming
	sequence                     uint64
	received, submitted, written time.Time
	writeEvidence                string
}

// Caller observations deliberately outlive application playback segmentation.
// An acoustic endpoint is an estimate at Core ingress, not provider VAD or a
// semantic turn. A packet after it is not necessarily an answer.
type realtimeCallerTiming struct {
	id                                                          uint64
	end, signal, firstPacket, firstAcoustic, lastPacket, bridge time.Time
	source, status, acousticItem                                string
	finalInput                                                  *realtimeInputObservation
	ready, dirty, overlap                                       bool
	packets, silentPackets, acousticEndBytes                    int
	silentDuration, cadenceTotal, cadenceMax                    time.Duration
	detector                                                    realtimeAcousticDetector
}

func (t *realtimeTiming) newCallerLocked(at time.Time, source string) *realtimeCallerTiming {
	if t.caller != nil && t.caller.ready {
		t.queueCallerLocked("caller_resumed")
	}
	t.callerSequence++
	t.caller = &realtimeCallerTiming{id: t.callerSequence, end: at, source: source, status: "waiting", dirty: true}
	return t.caller
}

func (t *realtimeTiming) callerSignalLocked(at time.Time, source string) {
	if t.callerClosed {
		return
	}
	c := t.caller
	if c == nil || c.ready && c.source != "core_input_acoustic_estimate" && !at.Equal(c.signal) {
		c = t.newCallerLocked(at, source)
	}
	// A signal-receipt timestamp must not replace an acoustic endpoint with a
	// different meaning. Keep both boundaries and label their evidence.
	if c.source != "core_input_acoustic_estimate" {
		c.end, c.source = at, source
	}
	c.signal, c.ready, c.dirty = at, true, true
}

func (t *realtimeTiming) inputObserved(audio []byte, at time.Time, format AudioFormat, rate int) *realtimeInputObservation {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.callerClosed {
		return nil
	}
	t.inputSequence++
	f := &realtimeInputObservation{tracker: t, sequence: t.inputSequence, received: at}
	active := t.inputAcoustic.observe(audio, format, rate, 20)
	if active {
		c := t.caller
		if c == nil || c.ready {
			c = t.newCallerLocked(at, "core_input_acoustic_estimate")
		}
		// Later speech invalidates tentative measurements from an internal pause.
		id := c.id
		*c = realtimeCallerTiming{id: id, end: at, source: "core_input_acoustic_estimate", status: "waiting", finalInput: f,
			overlap: !t.lastOutputAcoustic.IsZero() && at.Sub(t.lastOutputAcoustic) >= 0 && at.Sub(t.lastOutputAcoustic) < 300*time.Millisecond}
		t.inputQuiet = 0
	} else if c := t.caller; c != nil && !c.ready && c.source == "core_input_acoustic_estimate" {
		bps := rate
		if bps <= 0 {
			bps = mediaBytesPerSecond(format)
			if format == "" || format == AudioPCM16 {
				bps /= 2
			}
		}
		if format == "" || format == AudioPCM16 {
			bps *= 2
		}
		t.inputQuiet += time.Duration(len(audio)) * time.Second / time.Duration(max(1, bps))
		if t.inputQuiet >= 300*time.Millisecond {
			c.ready, c.dirty = true, true
			t.speechEnd, t.speechEndSource = c.end, c.source
			t.lastSpeechEnd, t.lastSpeechSource = c.end, c.source
		}
	}
	return f
}

func (f *realtimeInputObservation) submittedAt(at time.Time) {
	if f == nil {
		return
	}
	f.tracker.mu.Lock()
	defer f.tracker.mu.Unlock()
	if f.submitted.IsZero() {
		f.submitted = at
	}
}
func (f *realtimeInputObservation) settled(at time.Time, success, writeKnown bool) {
	if f == nil {
		return
	}
	t := f.tracker
	t.mu.Lock()
	defer t.mu.Unlock()
	if success && writeKnown {
		f.written, f.writeEvidence = at, "local_transport_write_completed"
	} else if success {
		f.writeEvidence = "local_admission_only"
	} else {
		f.writeEvidence = "failed_or_discarded"
	}
	if t.caller != nil && t.caller.finalInput == f {
		t.caller.dirty = true
	}
}

// Called at the adapter's decoded-audio boundary, before speech guards. This
// records metadata only; publication happens on the actor's reporting tick.
func (t *realtimeTiming) providerAudio(response, item string, at time.Time, audio []byte, format AudioFormat, rate int) {
	if t == nil {
		return
	}
	t.audio(response, item, at)
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.callerClosed {
		return
	}
	u := t.items[realtimeTimingKey(response, item)]
	if u != nil {
		u.providerBytes += len(audio)
	}
	active := t.outputAcoustic.observe(audio, format, rate, 100)
	if active {
		t.lastOutputAcoustic = at
	}
	c := t.caller
	if c == nil || at.Before(c.end) {
		return
	}
	// Correlate each application segment, but packet-cadence counters describe
	// only the wait through first sustained activity, not the rest of the reply.
	if u != nil && u.speechEnd.IsZero() {
		u.speechEnd, u.speechSource = c.end, c.source
	}
	if !c.firstAcoustic.IsZero() {
		return
	}
	if c.firstPacket.IsZero() {
		c.firstPacket = at
	}
	if !c.lastPacket.IsZero() && !at.Before(c.lastPacket) {
		gap := at.Sub(c.lastPacket)
		c.cadenceTotal += gap
		if gap > c.cadenceMax {
			c.cadenceMax = gap
		}
	}
	c.lastPacket = at
	c.packets++
	// Resetting this detector when caller speech extends prevents an earlier
	// greeting/backchannel from satisfying the new endpoint's acoustic window.
	active = c.detector.observe(audio, format, rate, 100)
	if c.firstAcoustic.IsZero() {
		if active {
			c.firstAcoustic, c.acousticItem, c.status = at, realtimeTimingKey(response, item), "acoustic_output_observed"
			if u != nil {
				c.acousticEndBytes = u.providerBytes
			}
		} else {
			c.silentPackets++
			bps := rate
			if bps <= 0 {
				bps = mediaBytesPerSecond(format)
				if format == "" || format == AudioPCM16 {
					bps /= 2
				}
			}
			if format == "" || format == AudioPCM16 {
				bps *= 2
			}
			c.silentDuration += time.Duration(len(audio)) * time.Second / time.Duration(max(1, bps))
		}
	}
	c.dirty = true
}

func (t *realtimeTiming) callerDataLocked(c *realtimeCallerTiming, stage string) map[string]any {
	d := map[string]any{"generation": t.generation, "caller_observation_id": c.id, "stage": stage, "status": c.status,
		"speech_end_source": c.source, "endpoint_observed": c.ready, "clock": "core_observed", "output_association": "temporal_only", "useful_answer_confirmed": false, "provider_receipt_confirmed": false,
		"speech_end_to_signal_received_ms": timingMS(c.end, c.signal), "speech_end_to_first_audio_ms": timingMS(c.end, c.firstPacket), "speech_end_to_acoustic_output_ms": timingMS(c.end, c.firstAcoustic),
		"acoustic_output_to_bridge_write_ms": timingMS(c.firstAcoustic, c.bridge), "speech_end_to_bridge_write_ms": timingMS(c.end, c.bridge),
		"overlap_or_continuation_possible": c.overlap, "audio_packets": c.packets, "pre_acoustic_packets": c.silentPackets, "pre_acoustic_audio_ms": float64(c.silentDuration) / float64(time.Millisecond),
		"packet_interval_max_ms": float64(c.cadenceMax) / float64(time.Millisecond), "packet_interval_mean_ms": float64(c.cadenceTotal) / float64(time.Millisecond) / float64(max(1, c.packets-1)),
		"acoustic_evidence": "two_consecutive_10ms_rms_windows", "output_rms_threshold": 100, "input_rms_threshold": 20, "endpoint_quiet_ms": 300,
		"final_speech_frame_sequence": nil, "final_speech_frame_to_submission_ms": nil, "final_speech_frame_to_local_write_ms": nil, "input_write_evidence": "unavailable", "playback_confirmed": false, "eligible_acoustic_latency_sample": c.ready && !c.overlap && !c.firstAcoustic.IsZero(), "observation_window_closed": stage != "observation", "missing_acoustic_output": c.firstAcoustic.IsZero()}
	if !c.ready {
		d["status"] = "input_interrupted"
		d["speech_end_to_first_audio_ms"], d["speech_end_to_acoustic_output_ms"], d["speech_end_to_bridge_write_ms"] = nil, nil, nil
	}
	if f := c.finalInput; f != nil {
		d["final_speech_frame_sequence"] = f.sequence
		d["final_speech_frame_to_submission_ms"] = timingMS(f.received, f.submitted)
		d["final_speech_frame_to_local_write_ms"] = timingMS(f.received, f.written)
		d["input_write_evidence"] = f.writeEvidence
	}
	return d
}
func (t *realtimeTiming) queueCallerLocked(stage string) {
	if c := t.caller; c != nil {
		if len(t.callerReports) >= 64 {
			t.callerReports = t.callerReports[1:]
			t.callerReportsDropped++
		}
		t.callerReports = append(t.callerReports, t.callerDataLocked(c, stage))
	}
}
func (t *realtimeTiming) reportCallerTiming(closeReason string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if closeReason == "" && time.Since(t.callerLastReport) < time.Second {
		t.mu.Unlock()
		return
	}
	t.callerLastReport = time.Now()
	if closeReason != "" && !t.callerClosed {
		t.callerClosed = true
		if t.caller != nil {
			t.queueCallerLocked(closeReason)
			t.caller = nil
		}
	} else if closeReason == "" && t.caller != nil && t.caller.ready && t.caller.dirty {
		t.queueCallerLocked("observation")
		t.caller.dirty = false
	}
	reports := t.callerReports
	t.callerReports = nil
	dropped := t.callerReportsDropped
	t.callerReportsDropped = 0
	t.mu.Unlock()
	if t.emit != nil {
		for _, d := range reports {
			d["tracking_reports_dropped"] = dropped
			t.emit("realtime.caller_timing", d)
		}
	}
}
