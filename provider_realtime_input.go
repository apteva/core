package core

import (
	"errors"
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
