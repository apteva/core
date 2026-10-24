package core

import (
	"errors"
	"sync"
	"time"
)

// RealtimeErrorClass distinguishes retry-safe local admission pressure from a
// failed session. A nil SendAudio error acknowledges local admission only.
type RealtimeErrorClass string

const (
	RealtimeErrorPressure  RealtimeErrorClass = "pressure"
	RealtimeErrorClosed    RealtimeErrorClass = "closed"
	RealtimeErrorTransport RealtimeErrorClass = "transport"
	RealtimeErrorProtocol  RealtimeErrorClass = "protocol"
)

type RealtimeAdapterError struct {
	Class RealtimeErrorClass
	Err   error
}

func (e *RealtimeAdapterError) Error() string { return e.Err.Error() }
func (e *RealtimeAdapterError) Unwrap() error { return e.Err }
func realtimeAdapterError(class RealtimeErrorClass, err error) error {
	return &RealtimeAdapterError{Class: class, Err: err}
}
func realtimeErrorClass(err error) RealtimeErrorClass {
	var classified *RealtimeAdapterError
	if errors.As(err, &classified) {
		return classified.Class
	}
	return RealtimeErrorTransport
}

// RealtimeInputPolicy is optional adapter policy, shared by all wire protocols.
// Pace limits catch-up to the recorded sample rate for continuous models.
// PressureTimeout bounds retry-safe local saturation before session recovery.
type RealtimeInputPolicy struct {
	MaxBuffered, MaxAge, FrameDuration, PressureTimeout time.Duration
	Pace                                                bool
}
type RealtimeInputPolicyProvider interface{ RealtimeInputPolicy() RealtimeInputPolicy }

func defaultRealtimeInputPolicy(continuous bool) RealtimeInputPolicy {
	return RealtimeInputPolicy{MaxBuffered: 5 * time.Second, MaxAge: 5 * time.Second, FrameDuration: 20 * time.Millisecond, PressureTimeout: time.Second, Pace: continuous}
}

// Complete releases the shared Core reservation only after transport write or
// discard. Admission is not evidence of provider processing or caller hearing.
type RealtimeInputAudio struct {
	Audio      []byte
	ReceivedAt time.Time
	MaxAge     time.Duration
	Complete   func(queueWait, transportWrite time.Duration, delivered bool)
}
type RealtimeAudioInputSession interface {
	SendAudioFrame(RealtimeInputAudio) error
}

type realtimeFlowMetrics struct {
	frames                         int64
	inputWait, adapterWait, write  time.Duration
	inputMax, adapterMax, writeMax time.Duration
}

func (m *realtimeFlowMetrics) observe(input, adapter, write time.Duration) {
	m.frames++
	m.inputWait += input
	m.adapterWait += adapter
	m.write += write
	if input > m.inputMax {
		m.inputMax = input
	}
	if adapter > m.adapterMax {
		m.adapterMax = adapter
	}
	if write > m.writeMax {
		m.writeMax = write
	}
}

// Each reservation is settled exactly once even if Close races a writer.
type realtimeInputReservations struct {
	mu      sync.Mutex
	pending map[*realtimeInputReservation]bool
	closed  bool
}
type realtimeInputReservation struct {
	owner    *realtimeInputReservations
	frame    RealtimeInputAudio
	admitted time.Time
	once     sync.Once
}

func (r *realtimeInputReservations) admit(f RealtimeInputAudio, enqueue func(*realtimeInputReservation) error) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return realtimeAdapterError(RealtimeErrorClosed, errors.New("realtime session closed"))
	}
	if r.pending == nil {
		r.pending = map[*realtimeInputReservation]bool{}
	}
	p := &realtimeInputReservation{owner: r, frame: f, admitted: time.Now()}
	if err := enqueue(p); err != nil {
		return err
	}
	r.pending[p] = true
	return nil
}
func (p *realtimeInputReservation) settle(success bool, write time.Duration) {
	if p == nil {
		return
	}
	p.once.Do(func() {
		p.owner.mu.Lock()
		delete(p.owner.pending, p)
		p.owner.mu.Unlock()
		if p.frame.Complete != nil {
			wait := time.Since(p.admitted) - write
			if wait < 0 {
				wait = 0
			}
			p.frame.Complete(wait, write, success)
		}
	})
}
func (r *realtimeInputReservations) close() {
	r.mu.Lock()
	r.closed = true
	pending := make([]*realtimeInputReservation, 0, len(r.pending))
	for p := range r.pending {
		pending = append(pending, p)
	}
	r.mu.Unlock()
	for _, p := range pending {
		p.settle(false, 0)
	}
}
func (p *realtimeInputReservation) stale() bool {
	return p != nil && p.frame.MaxAge > 0 && time.Since(p.frame.ReceivedAt) > p.frame.MaxAge
}
