package core

import (
	"context"
)

// OpenAIRealtimeProvider implements OpenAI's GA Realtime WebSocket API.
type OpenAIRealtimeProvider struct {
	apiKey       string
	models       map[ModelTier]string
	endpoint     string
	defaultVoice string
}

// NewOpenAIRealtimeProvider constructs a provider bound to the given
// API key. Models map can be overridden via config; defaults pick
// OpenAI's current GA realtime models.
func NewOpenAIRealtimeProvider(apiKey string) *OpenAIRealtimeProvider {
	return &OpenAIRealtimeProvider{
		apiKey:       apiKey,
		endpoint:     "wss://api.openai.com/v1/realtime",
		defaultVoice: "marin",
		models: map[ModelTier]string{
			ModelLarge:  "gpt-realtime-2.1",
			ModelMedium: "gpt-realtime-2.1-mini",
			ModelSmall:  "gpt-realtime-2.1-mini",
		},
	}
}

func (p *OpenAIRealtimeProvider) Name() string { return "openai-realtime" }

func (p *OpenAIRealtimeProvider) Models() map[ModelTier]string {
	out := make(map[ModelTier]string, len(p.models))
	for k, v := range p.models {
		out[k] = v
	}
	return out
}

func (p *OpenAIRealtimeProvider) Pricing(model string) RealtimePricing {
	if model == "gpt-realtime-2.1-mini" {
		return RealtimePricing{
			TextInput: 0.60, TextCachedInput: 0.06, TextOutput: 2.40,
			AudioInput: 10.0, AudioCachedInput: 0.30, AudioOutput: 20.0,
		}
	}
	return RealtimePricing{
		TextInput: 4.0, TextCachedInput: 0.40, TextOutput: 24.0,
		AudioInput: 32.0, AudioCachedInput: 0.40, AudioOutput: 64.0,
	}
}

func (p *OpenAIRealtimeProvider) DefaultVoice() string {
	if p.defaultVoice == "" {
		return "marin"
	}
	return p.defaultVoice
}

func (p *OpenAIRealtimeProvider) DefaultTranscriptionModel() string {
	return "gpt-4o-mini-transcribe"
}

// DefaultRealtimeReasoning follows OpenAI's voice-agent guidance: low is the
// responsive production starting point when the operator did not explicitly
// select an effort. Explicit thread profiles still win.
func (p *OpenAIRealtimeProvider) DefaultRealtimeReasoning() string { return "low" }

func (p *OpenAIRealtimeProvider) applyRealtimeConfig(config ProviderConfig) {
	applyRealtimeModelAndVoiceConfig(p.models, &p.defaultVoice, config)
}

// Open dials the OpenAI Realtime WebSocket, sends the initial
// session.update, and returns a live session. ctx governs the dial
// + handshake; the returned session manages its own goroutines for
// reads/writes and will surface a RealtimeEventSessionEnded event
// when the connection drops.
func (p *OpenAIRealtimeProvider) Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	return p.openSession(ctx, opts)
}

var _ RealtimeProvider = (*OpenAIRealtimeProvider)(nil)
var _ configurableRealtimeProvider = (*OpenAIRealtimeProvider)(nil)
var _ realtimeReasoningDefaultProvider = (*OpenAIRealtimeProvider)(nil)
