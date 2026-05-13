package core

import (
	"context"
	"errors"
)

// RealtimeProvider is the parallel of LLMProvider for bidirectional,
// streaming, audio-capable models (e.g. OpenAI Realtime, Gemini Live).
//
// Where LLMProvider.Chat() is request → response over HTTP+SSE, a
// RealtimeProvider holds a persistent WebSocket session for the
// lifetime of a realtime thread. Audio in / audio out flow over
// channels owned by the returned RealtimeSession; the thinker layer
// never touches the audio path itself.
//
// Two interfaces, not one, because the request/response and
// streaming-session models are fundamentally different shapes —
// fusing them would either lossy-shim realtime into Chat() or force
// every text provider to implement methods it has no notion of.
// Common metadata (Name, Models, CostPer1M) lives on both for
// uniformity at registration time.
type RealtimeProvider interface {
	Name() string
	Models() map[ModelTier]string
	// CostPer1M returns (input_text, cached_text, output_text, audio)
	// pricing per 1M tokens / characters. Audio is a separate rate
	// because realtime APIs bill audio in/out very differently from text.
	CostPer1M() (in, cached, out, audio float64)

	// DefaultVoice returns the provider's preferred voice when the
	// caller doesn't specify one. Empty string is acceptable for
	// providers that pick a voice server-side.
	DefaultVoice() string

	// Open establishes a session. ctx propagates to the underlying
	// connection; cancelling ctx closes the session cleanly. The
	// returned session is bound to ctx for its lifetime.
	Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error)
}

// AudioFormat names a wire encoding for realtime audio. Providers map
// these to their native config values (e.g. OpenAI "pcm16",
// "g711_ulaw").
type AudioFormat string

const (
	AudioPCM16     AudioFormat = "pcm16"
	AudioG711ULaw  AudioFormat = "g711_ulaw"
	AudioG711ALaw  AudioFormat = "g711_alaw"
)

// RealtimeSessionOpts is the connect-time configuration for a
// realtime session. Once the session is open, mutable fields can be
// updated via UpdateInstructions / similar — opts itself is consumed
// only at Open.
type RealtimeSessionOpts struct {
	Model        string       // provider-specific model id
	Voice        string       // empty = provider default
	Instructions string       // the thread's directive
	Tools        []NativeTool // tool schemas to expose to the model
	AudioInFmt   AudioFormat
	AudioOutFmt  AudioFormat
	Temperature  float64 // 0 = provider default
}

// RealtimeSession is the live, bidirectional handle to a single
// model session. All methods are safe to call concurrently with each
// other and with Events() consumption.
type RealtimeSession interface {
	// SendAudio pushes a chunk of input audio in the configured
	// AudioInFmt to the model. Non-blocking on the model side — the
	// session buffers and the network does its own back-pressure.
	SendAudio(pcm []byte) error

	// SendText injects a text message into the conversation. role is
	// one of "user", "system", "assistant". Used by main to deliver
	// out-of-band context to the realtime worker without speaking it
	// into the call (e.g. "[from:main] caller is VIP, be warm").
	SendText(role, text string) error

	// SendToolResult delivers the result of a tool call back to the
	// model. callID matches the id from the corresponding
	// RealtimeToolCallEvent.
	SendToolResult(callID, result string, isError bool) error

	// UpdateInstructions replaces the session's system instructions
	// in place (provider-side session.update). The conversation
	// history is preserved; only the directive shifts.
	UpdateInstructions(directive string) error

	// Interrupt cancels the model's current utterance. Used when new
	// user audio arrives during model speech, or when main sends a
	// course-correction the worker decides to act on immediately.
	Interrupt() error

	// Events returns the unified event stream from the session:
	// audio out, transcript fragments, tool calls, lifecycle events,
	// errors. Channel closes when the session ends.
	Events() <-chan RealtimeEvent

	// Close terminates the session and releases resources. Safe to
	// call multiple times.
	Close() error
}

// RealtimeEventType discriminates the union of events a session can
// emit. Receivers should switch on Type before reading fields.
type RealtimeEventType string

const (
	RealtimeEventAudioOut          RealtimeEventType = "audio_out"           // PCM chunk
	RealtimeEventTranscriptInput   RealtimeEventType = "transcript_input"    // user said
	RealtimeEventTranscriptOutput  RealtimeEventType = "transcript_output"   // model said
	RealtimeEventToolCall          RealtimeEventType = "tool_call"
	RealtimeEventResponseDone      RealtimeEventType = "response_done"
	RealtimeEventSessionEnded      RealtimeEventType = "session_ended"
	RealtimeEventError             RealtimeEventType = "error"
)

// RealtimeEvent is a single event from a session. Only the fields
// relevant to the Type are populated.
type RealtimeEvent struct {
	Type RealtimeEventType

	// Audio (RealtimeEventAudioOut)
	Audio []byte

	// Transcript (RealtimeEventTranscript*)
	Transcript string
	Final      bool // true = end-of-utterance, false = partial

	// Tool call (RealtimeEventToolCall)
	ToolCallID string
	ToolName   string
	ToolArgs   string // raw JSON string

	// Error (RealtimeEventError)
	Err error
}

// ErrRealtimeNotImplemented is returned by stub realtime providers
// that have been registered but whose Open() implementation hasn't
// landed yet. Callers should propagate this as a normal spawn
// error — the gate in the spawn handler turns it into a tool result
// the LLM can read.
var ErrRealtimeNotImplemented = errors.New("realtime provider not implemented yet")
