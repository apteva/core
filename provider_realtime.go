package core

import (
	"context"
	"fmt"
	"strings"
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
// Common metadata (Name and Models) lives on both for
// uniformity at registration time.
type RealtimeProvider interface {
	Name() string
	Models() map[ModelTier]string
	// Pricing returns the model-specific text and audio rates. Realtime
	// providers report input/output audio separately, and cached audio has
	// its own rate, so the ordinary three-number LLM price tuple is not
	// expressive enough here.
	Pricing(model string) RealtimePricing

	// DefaultVoice returns the provider's preferred voice when the
	// caller doesn't specify one. Empty string is acceptable for
	// providers that pick a voice server-side.
	DefaultVoice() string

	// DefaultTranscriptionModel returns the provider-native streaming
	// transcription model to request for input captions. Providers that
	// do not support a separate transcription model may return empty.
	DefaultTranscriptionModel() string

	// Open establishes a session. ctx propagates to the underlying
	// connection; cancelling ctx closes the session cleanly. The
	// returned session is bound to ctx for its lifetime.
	Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error)
}

// configurableRealtimeProvider is an optional registration-time extension.
// It keeps provider model/voice overrides out of ProviderPool type switches,
// so adding another realtime adapter does not change thread orchestration.
type configurableRealtimeProvider interface {
	applyRealtimeConfig(ProviderConfig)
}

// realtimeReasoningDefaultProvider optionally supplies a voice-optimized
// reasoning effort when the thread profile leaves reasoning on auto. It is
// deliberately separate from RealtimeProvider because compatible providers
// expose different effort scales and defaults.
type realtimeReasoningDefaultProvider interface {
	DefaultRealtimeReasoning() string
}

func applyRealtimeModelAndVoiceConfig(models map[ModelTier]string, voice *string, config ProviderConfig) {
	for tierName, model := range config.Models {
		if tier, ok := modelNames[strings.ToLower(strings.TrimSpace(tierName))]; ok && strings.TrimSpace(model) != "" {
			models[tier] = strings.TrimSpace(model)
		}
	}
	if voice != nil && strings.TrimSpace(config.RealtimeVoice) != "" {
		*voice = strings.TrimSpace(config.RealtimeVoice)
	}
}

// AudioFormat names a wire encoding for realtime audio. Providers map
// these to their native config values (e.g. OpenAI "pcm16",
// "g711_ulaw").
type AudioFormat string

const (
	AudioPCM16    AudioFormat = "pcm16"
	AudioG711ULaw AudioFormat = "g711_ulaw"
	AudioG711ALaw AudioFormat = "g711_alaw"
)

const (
	RealtimeTurnProfileDefault   = "default"
	RealtimeTurnProfileTelephony = "telephony"

	RealtimeSensitivityDefault = "default"
	RealtimeSensitivityHigh    = "high"
	RealtimeSensitivityLow     = "low"

	RealtimeInterruptionDefault = "default"
	RealtimeInterruptionAllow   = "interrupt"
	RealtimeInterruptionDisable = "no_interruption"
)

// RealtimeTurnDetectionConfig is the provider-neutral turn-taking contract
// carried by a realtime thread. Providers map the subset they support to
// their native session fields. Profile values are durable intent; explicit
// non-zero/non-default fields override the selected profile.
//
// The zero value preserves each provider's existing behavior.
type RealtimeTurnDetectionConfig struct {
	Profile           string `json:"profile,omitempty"`
	StartSensitivity  string `json:"start_sensitivity,omitempty"`
	PrefixPaddingMS   int    `json:"prefix_padding_ms,omitempty"`
	EndSensitivity    string `json:"end_sensitivity,omitempty"`
	SilenceDurationMS int    `json:"silence_duration_ms,omitempty"`
	Interruption      string `json:"interruption,omitempty"`
}

func (c RealtimeTurnDetectionConfig) isZero() bool {
	profile := strings.ToLower(strings.TrimSpace(c.Profile))
	startSensitivity := strings.ToLower(strings.TrimSpace(c.StartSensitivity))
	endSensitivity := strings.ToLower(strings.TrimSpace(c.EndSensitivity))
	interruption := strings.ToLower(strings.TrimSpace(c.Interruption))

	// Explicit provider defaults are semantically the same as omitting turn
	// detection altogether. Treating them as empty keeps clients and models
	// that serialize optional defaults from accidentally opting a normal
	// thread into realtime-only validation.
	return (profile == "" || profile == RealtimeTurnProfileDefault) &&
		(startSensitivity == "" || startSensitivity == RealtimeSensitivityDefault) &&
		c.PrefixPaddingMS == 0 &&
		(endSensitivity == "" || endSensitivity == RealtimeSensitivityDefault) &&
		c.SilenceDurationMS == 0 &&
		(interruption == "" || interruption == RealtimeInterruptionDefault)
}

func realtimeTurnDetectionValue(config *RealtimeTurnDetectionConfig) RealtimeTurnDetectionConfig {
	if config == nil {
		return RealtimeTurnDetectionConfig{}
	}
	return *config
}

func (c RealtimeTurnDetectionConfig) normalized() (RealtimeTurnDetectionConfig, error) {
	c.Profile = strings.ToLower(strings.TrimSpace(c.Profile))
	c.StartSensitivity = strings.ToLower(strings.TrimSpace(c.StartSensitivity))
	c.EndSensitivity = strings.ToLower(strings.TrimSpace(c.EndSensitivity))
	c.Interruption = strings.ToLower(strings.TrimSpace(c.Interruption))
	if c.Profile == "" {
		c.Profile = RealtimeTurnProfileDefault
	}
	switch c.Profile {
	case RealtimeTurnProfileDefault, RealtimeTurnProfileTelephony:
	default:
		return RealtimeTurnDetectionConfig{}, fmt.Errorf("unsupported realtime turn-detection profile %q (use default or telephony)", c.Profile)
	}
	for label, value := range map[string]string{
		"start_sensitivity": c.StartSensitivity,
		"end_sensitivity":   c.EndSensitivity,
	} {
		switch value {
		case "", RealtimeSensitivityDefault, RealtimeSensitivityHigh, RealtimeSensitivityLow:
		default:
			return RealtimeTurnDetectionConfig{}, fmt.Errorf("%s must be default, high, or low", label)
		}
	}
	switch c.Interruption {
	case "", RealtimeInterruptionDefault, RealtimeInterruptionAllow, RealtimeInterruptionDisable:
	default:
		return RealtimeTurnDetectionConfig{}, fmt.Errorf("interruption must be default, interrupt, or no_interruption")
	}
	if c.PrefixPaddingMS < 0 || c.PrefixPaddingMS > 60_000 {
		return RealtimeTurnDetectionConfig{}, fmt.Errorf("prefix_padding_ms must be between 0 and 60000")
	}
	if c.SilenceDurationMS < 0 || c.SilenceDurationMS > 60_000 {
		return RealtimeTurnDetectionConfig{}, fmt.Errorf("silence_duration_ms must be between 0 and 60000")
	}
	return c, nil
}

func (c RealtimeTurnDetectionConfig) resolved() RealtimeTurnDetectionConfig {
	normalized, err := c.normalized()
	if err != nil {
		// Callers validate at the spawn/session boundary. Keeping this helper
		// total makes telemetry and provider setup deterministic if a legacy
		// in-memory caller bypasses that boundary.
		normalized = RealtimeTurnDetectionConfig{Profile: RealtimeTurnProfileDefault}
	}
	if normalized.Profile == RealtimeTurnProfileTelephony {
		if normalized.StartSensitivity == "" || normalized.StartSensitivity == RealtimeSensitivityDefault {
			normalized.StartSensitivity = RealtimeSensitivityLow
		}
		if normalized.PrefixPaddingMS == 0 {
			normalized.PrefixPaddingMS = 300
		}
		if normalized.EndSensitivity == "" || normalized.EndSensitivity == RealtimeSensitivityDefault {
			normalized.EndSensitivity = RealtimeSensitivityLow
		}
		if normalized.SilenceDurationMS == 0 {
			normalized.SilenceDurationMS = 750
		}
		if normalized.Interruption == "" || normalized.Interruption == RealtimeInterruptionDefault {
			normalized.Interruption = RealtimeInterruptionAllow
		}
	}
	return normalized
}

func (c RealtimeTurnDetectionConfig) telemetryData() map[string]any {
	resolved := c.resolved()
	valueOrDefault := func(value string) string {
		if value == "" || value == RealtimeSensitivityDefault || value == RealtimeInterruptionDefault {
			return "provider_default"
		}
		return value
	}
	return map[string]any{
		"profile":             resolved.Profile,
		"start_sensitivity":   valueOrDefault(resolved.StartSensitivity),
		"prefix_padding_ms":   resolved.PrefixPaddingMS,
		"end_sensitivity":     valueOrDefault(resolved.EndSensitivity),
		"silence_duration_ms": resolved.SilenceDurationMS,
		"interruption":        valueOrDefault(resolved.Interruption),
	}
}

// RealtimeSessionOpts is the connect-time configuration for a
// realtime session. Once the session is open, mutable fields can be
// updated via UpdateInstructions / similar — opts itself is consumed
// only at Open.
type RealtimeSessionOpts struct {
	Model              string       // provider-specific model id
	Voice              string       // empty = provider default
	Instructions       string       // complete thread system prompt
	Tools              []NativeTool // tool schemas to expose to the model
	AudioInFmt         AudioFormat
	AudioOutFmt        AudioFormat
	AudioInRate        int    // Hz; 0 = provider default
	AudioOutRate       int    // Hz; 0 = provider default
	Reasoning          string // provider-supported reasoning effort; empty/auto = default
	SafetyIdentifier   string // stable privacy-preserving end-user/session identifier
	TranscribeInput    bool
	TranscriptionModel string // empty = provider default
	TurnDetection      RealtimeTurnDetectionConfig
}

// RealtimePricing supports dollars per one million tokens plus optional
// duration/message rates. Providers leave unsupported dimensions at zero.
type RealtimePricing struct {
	TextInput        float64
	TextCachedInput  float64
	TextOutput       float64
	AudioInput       float64
	AudioCachedInput float64
	AudioOutput      float64

	// Some realtime vendors bill connection media by duration and text
	// conversation items by count rather than by tokens. These optional
	// dimensions coexist with token pricing so provider adapters do not
	// leak billing rules into the generic thread runtime.
	AudioInputPerMinute  float64
	AudioOutputPerMinute float64
	TextInputPerMessage  float64
}

// RealtimeUsage is the detailed token accounting carried by a completed
// realtime response. Keeping text and audio separate is necessary for
// correct cost calculation and cache telemetry.
type RealtimeUsage struct {
	TotalTokens       int
	InputTokens       int
	OutputTokens      int
	TextInputTokens   int
	TextCachedTokens  int
	TextOutputTokens  int
	AudioInputTokens  int
	AudioCachedTokens int
	AudioOutputTokens int

	AudioInputSeconds  float64
	AudioOutputSeconds float64
	TextInputMessages  int
}

// RealtimeAudioFrame is the provider-neutral unit sent to an external audio
// renderer. Audio is always encoded in the canonical bridge format (PCM16,
// 24 kHz, mono). ItemID and AudioEndMS let the renderer report how much of an
// assistant item was actually played without knowing the provider protocol.
type RealtimeAudioFrame struct {
	Audio      []byte
	ResponseID string
	ItemID     string
	AudioEndMS int
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

	// SendToolResult delivers one tool result without starting the next model
	// response. This separation lets the generic thinker submit every result
	// from a parallel tool batch before requesting one continuation.
	SendToolResult(callID, result string, isError bool) error

	// RequestResponse asks the provider to continue after text or tool items
	// have been added to its conversation.
	RequestResponse() error

	// UpdateConfiguration replaces mutable instructions and tools while
	// preserving the provider-side conversation.
	UpdateConfiguration(instructions string, tools []NativeTool) error

	// RestoreConversation appends bounded transcript history without
	// triggering a response. It is used after a provider-enforced session
	// renewal so the worker can continue rather than becoming a blank agent.
	RestoreConversation(messages []Message) error

	// Interrupt cancels the model's current utterance. Used when new
	// user audio arrives during model speech, or when main sends a
	// course-correction the worker decides to act on immediately.
	Interrupt() error

	// Truncate removes the unplayed suffix of an assistant audio item after
	// client-managed WebSocket playback is interrupted.
	Truncate(itemID string, audioEndMS int) error

	// Events returns the unified event stream from the session:
	// audio out, transcript fragments, tool calls, lifecycle events,
	// errors. Channel closes when the session ends.
	Events() <-chan RealtimeEvent

	// Close terminates the session and releases resources. Safe to
	// call multiple times.
	Close() error
}

// RealtimeConfigurationDisposition describes how a live provider can apply a
// changed system prompt or tool set. Sessions that do not implement
// RealtimeConfigurationPreviewer retain the existing live-update behavior.
// Providers with immutable setup (Gemini Live today) use RestartRequired so
// Core can reject an accidental parent update before mutating the thread.
type RealtimeConfigurationDisposition string

const (
	RealtimeConfigurationUnchanged       RealtimeConfigurationDisposition = "unchanged"
	RealtimeConfigurationAppliedLive     RealtimeConfigurationDisposition = "applied_live"
	RealtimeConfigurationRestartRequired RealtimeConfigurationDisposition = "restart_required"
)

// RealtimeConfigurationPreviewer is an optional provider capability. The
// method must be side-effect free.
type RealtimeConfigurationPreviewer interface {
	PreviewConfigurationUpdate(instructions string, tools []NativeTool) RealtimeConfigurationDisposition
}

// RealtimeEventType discriminates the union of events a session can
// emit. Receivers should switch on Type before reading fields.
type RealtimeEventType string

const (
	RealtimeEventAudioOut         RealtimeEventType = "audio_out"         // PCM chunk
	RealtimeEventTranscriptInput  RealtimeEventType = "transcript_input"  // user said
	RealtimeEventTranscriptOutput RealtimeEventType = "transcript_output" // model said
	RealtimeEventToolCall         RealtimeEventType = "tool_call"
	RealtimeEventResponseStarted  RealtimeEventType = "response_started"
	RealtimeEventResponseDone     RealtimeEventType = "response_done"
	RealtimeEventSpeechStarted    RealtimeEventType = "speech_started"
	RealtimeEventRateLimits       RealtimeEventType = "rate_limits"
	RealtimeEventSessionEnded     RealtimeEventType = "session_ended"
	RealtimeEventError            RealtimeEventType = "error"
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

	// Response/audio item metadata. ItemID + AudioEndMS let the caller
	// truncate unplayed audio correctly on barge-in.
	ResponseID   string
	ItemID       string
	Phase        string // provider output phase, e.g. commentary or final_answer
	AudioStartMS int
	AudioEndMS   int
	Usage        RealtimeUsage
	DroppedAudio uint64

	// Error (RealtimeEventError)
	Err error
}
