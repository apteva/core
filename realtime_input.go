package core

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type realtimeInputFrame struct {
	audio    []byte
	received time.Time
}
type realtimeInputQueue struct {
	recoveryStarted            time.Time     // mu
	recoveryLast, recoveryPeak time.Duration // mu
	recoveryCount              int           // mu
	mu                         sync.Mutex
	sendMu                     sync.Mutex
	inflight                   int
	flow                       realtimeFlowMetrics
	lastFlowReport             time.Time
	frames                     []realtimeInputFrame
	bytes                      int
	bytesPerSecond             int
	sampleBytes                int
	policy                     RealtimeInputPolicy
	drops                      map[string]int
	nextSend, pressureSince    time.Time // sendMu
}

func (q *realtimeInputQueue) duration(n int) time.Duration {
	if q.bytesPerSecond == 0 {
		return 0
	}
	return time.Duration(n) * time.Second / time.Duration(q.bytesPerSecond)
}
func (q *realtimeInputQueue) pending() bool { q.mu.Lock(); defer q.mu.Unlock(); return q.bytes > 0 }
func (q *realtimeInputQueue) backlog() time.Duration {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.duration(q.bytes + q.inflight)
}
func (q *realtimeInputQueue) dropLocked(reason string, n int) {
	if n == 0 {
		return
	}
	if q.drops == nil {
		q.drops = map[string]int{}
	}
	q.drops[reason] += n
}
func (q *realtimeInputQueue) expireLocked(now time.Time) {
	for len(q.frames) > 0 && now.Sub(q.frames[0].received) > q.policy.MaxAge {
		n := len(q.frames[0].audio)
		q.frames[0] = realtimeInputFrame{}
		q.frames = q.frames[1:]
		q.bytes -= n
		q.dropLocked("stale_input", n)
	}
}
func (q *realtimeInputQueue) push(audio []byte, now time.Time) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.expireLocked(now)
	limit := int(q.policy.MaxBuffered * time.Duration(q.bytesPerSecond) / time.Second)
	keep := min(len(audio), max(0, limit-q.bytes-q.inflight))
	sampleBytes := q.sampleBytes
	if sampleBytes == 0 {
		sampleBytes = 2
	}
	keep -= keep % sampleBytes
	frameBytes := max(2, int(q.policy.FrameDuration*time.Duration(q.bytesPerSecond)/time.Second))
	frameBytes -= frameBytes % sampleBytes
	for offset := 0; offset < keep; offset += frameBytes {
		end := min(offset+frameBytes, keep)
		q.frames = append(q.frames, realtimeInputFrame{append([]byte(nil), audio[offset:end]...), now})
	}
	q.bytes += keep
	q.dropLocked("input_buffer_full", len(audio)-keep)
}
func (q *realtimeInputQueue) discard(reason string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.dropLocked(reason, q.bytes)
	q.frames = nil
	q.bytes = 0
}
func (rt *RealtimeThinker) configureInputQueue() {
	p := defaultRealtimeInputPolicy(rt.capabilities.ContinuousConversation)
	if adapter, ok := rt.provider.(RealtimeInputPolicyProvider); ok {
		v := adapter.RealtimeInputPolicy()
		if v.MaxBuffered > 0 {
			p.MaxBuffered = v.MaxBuffered
		}
		if v.MaxAge > 0 {
			p.MaxAge = v.MaxAge
		}
		if v.FrameDuration > 0 {
			p.FrameDuration = v.FrameDuration
		}
		if v.PressureTimeout > 0 {
			p.PressureTimeout = v.PressureTimeout
		}
		p.Pace = p.Pace || v.Pace
		if v.CatchUpRate > 0 {
			p.CatchUpRate = min(1.25, v.CatchUpRate)
			if p.CatchUpRate < 1 {
				p.CatchUpRate = 1
			}
		}
	}
	rt.input.mu.Lock()
	rt.input.policy = p
	rt.input.bytesPerSecond = int(audioBytesPerSecond(rt.opts.AudioInFmt, rt.opts.AudioInRate))
	rt.input.sampleBytes = 2
	if rt.opts.AudioInFmt == AudioG711ULaw || rt.opts.AudioInFmt == AudioG711ALaw {
		rt.input.sampleBytes = 1
	}
	rt.input.mu.Unlock()
}

// Intake does no provider, persistence or telemetry work. It continues even
// while opening/restoring/closing a session or persisting a transcript stalls.
func (rt *RealtimeThinker) ingestRealtimeAudio(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case audio, ok := <-rt.audioIn:
			if !ok {
				return
			}
			if len(audio) == 0 || rt.inputPaused.Load() {
				continue
			}
			rt.admitRealtimeAudio(audio, time.Now())
		}
	}
}
func (rt *RealtimeThinker) reportInputDrops() {
	q := &rt.input
	q.mu.Lock()
	drops := q.drops
	q.drops = nil
	var flow map[string]any
	if q.flow.frames > 0 && time.Since(q.lastFlowReport) >= time.Second {
		m := q.flow
		q.flow = realtimeFlowMetrics{}
		q.lastFlowReport = time.Now()
		oldestAge := time.Duration(0)
		if len(q.frames) > 0 {
			oldestAge = time.Since(q.frames[0].received)
		}
		flow = map[string]any{"frames": m.frames, "oldest_input_age_ms": oldestAge.Milliseconds(), "catch_up_rate_limit": q.policy.CatchUpRate, "recovery_active": !q.recoveryStarted.IsZero(), "last_recovery_ms": q.recoveryLast.Milliseconds(), "recovery_peak_age_ms": q.recoveryPeak.Milliseconds(), "recoveries": q.recoveryCount, "input_wait_mean_ms": float64(m.inputWait) / float64(time.Millisecond) / float64(m.frames), "input_wait_max_ms": float64(m.inputMax) / float64(time.Millisecond), "adapter_wait_mean_ms": float64(m.adapterWait) / float64(time.Millisecond) / float64(m.frames), "adapter_wait_max_ms": float64(m.adapterMax) / float64(time.Millisecond), "transport_write_mean_ms": float64(m.write) / float64(time.Millisecond) / float64(m.frames), "transport_write_max_ms": float64(m.writeMax) / float64(time.Millisecond), "queued_audio_ms": q.duration(q.bytes + q.inflight).Milliseconds(), "budget_ms": q.policy.MaxBuffered.Milliseconds(), "remote_delivery_confirmed": false}
	}
	q.mu.Unlock()
	if flow != nil {
		rt.emit("realtime.input_flow", flow)
	}
	for reason, n := range drops {
		rt.emit("realtime.audio_drop", map[string]any{"direction": "input", "reason": reason, "bytes": n, "duration_ms": q.duration(n).Milliseconds(), "discontinuity": true, "generation": generationForAudioFrame(rt)})
	}
}
func (rt *RealtimeThinker) recordInputFailure(err error, operation string) {
	rt.emit("realtime.error", map[string]any{"error": fmt.Sprint(err), "error_class": realtimeErrorClass(err), "operation": operation, "generation": generationForAudioFrame(rt), "input_queue_ms": rt.input.backlog().Milliseconds(), "remote_delivery_confirmed": false})
}
func (q *realtimeInputQueue) resetSender() {
	q.sendMu.Lock()
	q.nextSend = time.Time{}
	q.pressureSince = time.Time{}
	q.sendMu.Unlock()
}
func (rt *RealtimeThinker) drainRealtimeInput(session RealtimeSession) {
	rt.drainRealtimeInputAt(session, time.Time{})
}
func (rt *RealtimeThinker) drainRealtimeInputAt(session RealtimeSession, now time.Time) {
	rt.input.sendMu.Lock()
	defer rt.input.sendMu.Unlock()
	if rt.inputPaused.Load() || !rt.audioLane.active.Load() && rt.paused {
		rt.input.discard("paused")
		return
	}
	q := &rt.input
	realtimeClock := now.IsZero()
	for batch := 0; batch < 8; batch++ {
		if realtimeClock {
			now = time.Now()
		}
		if now.Before(q.nextSend) {
			return
		}
		q.mu.Lock()
		q.expireLocked(now)
		if len(q.frames) == 0 {
			q.mu.Unlock()
			q.pressureSince = time.Time{}
			return
		}
		frame := q.frames[0]
		q.mu.Unlock()
		q.mu.Lock()
		q.inflight += len(frame.audio)
		q.mu.Unlock()
		var once sync.Once
		done := func(wait, write time.Duration, delivered bool) {
			once.Do(func() {
				q.mu.Lock()
				defer q.mu.Unlock()
				q.inflight -= len(frame.audio)
				q.flow.observe(now.Sub(frame.received), wait, write)
				if !delivered {
					q.dropLocked("adapter_delivery_failed", len(frame.audio))
				}
			})
		}
		var err error
		if adapter, ok := session.(RealtimeAudioInputSession); ok {
			err = adapter.SendAudioFrame(RealtimeInputAudio{Audio: frame.audio, ReceivedAt: frame.received, MaxAge: q.policy.MaxAge, Complete: done})
		} else {
			err = session.SendAudio(frame.audio)
			if err == nil {
				done(0, 0, true)
			}
		}
		if err != nil {
			once.Do(func() { q.mu.Lock(); q.inflight -= len(frame.audio); q.mu.Unlock() })
		}
		if err != nil && realtimeErrorClass(err) == RealtimeErrorPressure {
			if q.pressureSince.IsZero() {
				q.pressureSince = now
				rt.noteInputSendFailure(session, err, "send_audio", false)
			}
			q.nextSend = now.Add(10 * time.Millisecond)
			if now.Sub(q.pressureSince) >= q.policy.PressureTimeout {
				rt.noteInputSendFailure(session, err, "input_pressure_timeout", true)
				q.pressureSince = time.Time{}
			}
			return
		}
		// Only pressure/closed explicitly guarantee no admission. Unclassified
		// transport failures can be ambiguous, so do not replay that frame.
		if err == nil || realtimeErrorClass(err) != RealtimeErrorClosed {
			q.mu.Lock()
			// Intake may expire the front while SendAudio runs. Never remove a newer frame.
			if len(q.frames) > 0 && &q.frames[0].audio[0] == &frame.audio[0] {
				q.bytes -= len(frame.audio)
				q.frames[0] = realtimeInputFrame{}
				q.frames = q.frames[1:]
				if err != nil {
					q.dropLocked("delivery_uncertain", len(frame.audio))
				}
			}
			q.mu.Unlock()
		}
		if err != nil {
			rt.noteInputSendFailure(session, err, "send_audio", false)
			rt.noteInputSendFailure(session, err, "audio_delivery_failed", true)
			return
		}
		q.pressureSince = time.Time{}
		if q.policy.Pace {
			sentAt := now
			if realtimeClock {
				sentAt = time.Now()
			}
			q.nextSend = q.sendDeadline(sentAt, frame)
			return
		}
	}
}

func (rt *RealtimeThinker) noteInputSendFailure(s RealtimeSession, err error, reason string, recover bool) {
	if rt.audioLane.active.Load() {
		if recover {
			rt.failAudioLane(s, reason, err)
		} else {
			rt.recordInputFailure(err, reason)
		}
		return
	}
	operation := reason
	if reason == "input_pressure_timeout" {
		operation = "sustained_input_pressure"
	}
	rt.recordInputFailure(err, operation)
	if recover {
		rt.closeForRecovery(s, false, reason, err)
	}
}

// The bridge admits directly here so transport, recovery and adapter queues
// share one limit and the original arrival timestamp. No I/O is performed.
func (rt *RealtimeThinker) admitRealtimeAudio(audio []byte, at time.Time) {
	if rt.inputPaused.Load() {
		return
	}
	if rt.capabilities.ContinuousConversation && mediaVoiced(audio, rt.opts.AudioInFmt) {
		rt.lastCallerVoice.Store(at.UnixNano())
	}
	rt.recordRealtimeInput(len(audio))
	rt.input.push(audio, at)
	rt.wakeRecovery()
}

// Never borrow future microphone audio or release an unbounded burst. A modest
// recovery rate applies only to old captured frames. Each send establishes a
// bounded deadline, so scheduler stalls cannot accumulate burst credit.
func (q *realtimeInputQueue) sendDeadline(now time.Time, frame realtimeInputFrame) time.Time {
	interval := q.duration(len(frame.audio))
	age := now.Sub(frame.received)
	q.mu.Lock()
	if age > 100*time.Millisecond && q.recoveryStarted.IsZero() {
		q.recoveryStarted = now
		q.recoveryPeak = age
	}
	if !q.recoveryStarted.IsZero() {
		if age > q.recoveryPeak {
			q.recoveryPeak = age
		}
		if age <= 3*q.policy.FrameDuration {
			q.recoveryLast = now.Sub(q.recoveryStarted)
			q.recoveryStarted = time.Time{}
			q.recoveryCount++
		}
	}
	q.mu.Unlock()
	minimumInterval := interval
	if q.policy.CatchUpRate > 1 {
		minimumInterval = time.Duration(float64(interval) / min(q.policy.CatchUpRate, 1.25))
		if age > 3*q.policy.FrameDuration {
			interval = minimumInterval
		}
	}
	deadline := q.nextSend.Add(interval)
	if q.nextSend.IsZero() {
		return now.Add(interval)
	}
	// Retain cadence across ordinary scheduler jitter, but never spend old
	// deadline debt as a burst after a stall.
	if earliest := now.Add(minimumInterval); deadline.Before(earliest) {
		deadline = earliest
	}
	return deadline
}
