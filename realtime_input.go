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
	mu                      sync.Mutex
	frames                  []realtimeInputFrame
	bytes                   int
	bytesPerSecond          int
	sampleBytes             int
	policy                  RealtimeInputPolicy
	drops                   map[string]int
	nextSend, pressureSince time.Time // runtime loop only
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
	return q.duration(q.bytes)
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
	keep := min(len(audio), max(0, limit-q.bytes))
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
			rt.recordRealtimeInput(len(audio))
			rt.queueRecoveryAudio(audio)
		}
	}
}
func (rt *RealtimeThinker) reportInputDrops() {
	q := &rt.input
	q.mu.Lock()
	drops := q.drops
	q.drops = nil
	q.mu.Unlock()
	for reason, n := range drops {
		rt.emit("realtime.audio_drop", map[string]any{"direction": "input", "reason": reason, "bytes": n, "duration_ms": q.duration(n).Milliseconds(), "discontinuity": true, "generation": generationForAudioFrame(rt)})
	}
}
func (rt *RealtimeThinker) recordInputFailure(err error, operation string) {
	rt.emit("realtime.error", map[string]any{"error": fmt.Sprint(err), "error_class": realtimeErrorClass(err), "operation": operation, "generation": generationForAudioFrame(rt), "input_queue_ms": rt.input.backlog().Milliseconds(), "remote_delivery_confirmed": false})
}
func (rt *RealtimeThinker) drainRealtimeInput(session RealtimeSession) {
	if rt.paused {
		rt.input.discard("paused")
		return
	}
	q := &rt.input
	for batch := 0; batch < 8; batch++ {
		now := time.Now()
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
		err := session.SendAudio(frame.audio)
		if err != nil && realtimeErrorClass(err) == RealtimeErrorPressure {
			if q.pressureSince.IsZero() {
				q.pressureSince = now
				rt.recordInputFailure(err, "send_audio")
			}
			q.nextSend = now.Add(10 * time.Millisecond)
			if now.Sub(q.pressureSince) >= q.policy.PressureTimeout {
				rt.recordInputFailure(err, "sustained_input_pressure")
				rt.closeForRecovery(session, false, "input_pressure_timeout", err)
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
			rt.recordInputFailure(err, "send_audio")
			rt.closeForRecovery(session, false, "audio_delivery_failed", err)
			return
		}
		q.pressureSince = time.Time{}
		if q.policy.Pace {
			duration := q.duration(len(frame.audio))
			base := q.nextSend
			if base.IsZero() || now.Sub(base) > duration {
				base = now
			}
			q.nextSend = base.Add(duration)
			return
		}
	}
}
