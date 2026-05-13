package core

import (
	"context"
)

// OpenAIRealtimeProvider is the stub for OpenAI's Realtime API
// (wss://api.openai.com/v1/realtime). This stage of the realtime
// rollout registers the provider in the pool so the rest of the
// surface (config, spawn gate, prompt bullet) can be exercised
// end-to-end; the WebSocket client lives in a follow-up change.
//
// Open() returns ErrRealtimeNotImplemented today. When that lands
// it'll dial the WebSocket, send a session.update with the directive
// + tools, fan events out on the RealtimeEvent channel, and forward
// audio/text/tool-result calls. The interface boundary is final;
// adding the impl won't change the surface.
type OpenAIRealtimeProvider struct {
	apiKey string
	models map[ModelTier]string
}

// NewOpenAIRealtimeProvider constructs a provider bound to the given
// API key. Models map can be overridden via config; defaults pick
// OpenAI's current realtime preview.
func NewOpenAIRealtimeProvider(apiKey string) *OpenAIRealtimeProvider {
	return &OpenAIRealtimeProvider{
		apiKey: apiKey,
		models: map[ModelTier]string{
			ModelLarge:  "gpt-4o-realtime-preview",
			ModelMedium: "gpt-4o-mini-realtime-preview",
			ModelSmall:  "gpt-4o-mini-realtime-preview",
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

// CostPer1M returns realtime pricing. Numbers reflect OpenAI's
// published rates for gpt-4o-realtime-preview at the time of
// scaffolding; verify before relying on cost accounting in
// production. Audio rate is the per-1M-token rate for audio I/O,
// which is billed separately from text.
func (p *OpenAIRealtimeProvider) CostPer1M() (in, cached, out, audio float64) {
	return 5.0, 2.5, 20.0, 100.0
}

func (p *OpenAIRealtimeProvider) DefaultVoice() string { return "alloy" }

// Open dials the OpenAI Realtime WebSocket, sends the initial
// session.update, and returns a live session. ctx governs the dial
// + handshake; the returned session manages its own goroutines for
// reads/writes and will surface a RealtimeEventSessionEnded event
// when the connection drops.
func (p *OpenAIRealtimeProvider) Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	return p.openSession(ctx, opts)
}
