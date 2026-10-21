package core

import "context"

// Optional execution override on a provider's voice configuration. Empty
// values inherit the thread's existing Core text provider and model tier.
type RealtimeExecutionConfig struct {
	Mode     string `json:"mode,omitempty"` // thread (default) or provider
	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
}

// RealtimeCapabilities describes behavior, not a vendor. The zero/legacy
// profile preserves the existing turn-based direct-tool contract.
type RealtimeCapabilities struct {
	ContinuousConversation bool `json:"continuous_conversation"`
	ProviderExecution      bool `json:"provider_execution"`
	CoreExecution          bool `json:"core_execution"`
	TranscriptFragments    bool `json:"transcript_fragments"`
	ProviderInterruption   bool `json:"provider_interruption"`
	AudioTruncation        bool `json:"audio_truncation"`
}

type RealtimeProviderInfo struct {
	Integration  string                          `json:"integration"`
	Provider     string                          `json:"provider"`
	Models       map[string]string               `json:"models"`
	Capabilities map[string]RealtimeCapabilities `json:"capabilities"`
}

type realtimeCapabilityProvider interface {
	RealtimeCapabilities(model string) RealtimeCapabilities
}

func capabilitiesForRealtime(p RealtimeProvider, model string) RealtimeCapabilities {
	if p, ok := p.(realtimeCapabilityProvider); ok {
		return p.RealtimeCapabilities(model)
	}
	return RealtimeCapabilities{AudioTruncation: true}
}

// A context update is deliberately distinct from a user message or a tool
// result. Adapters own protocol limits and cannot promote tool data to policy.
type RealtimeContextUpdate struct {
	Kind, Content, DelegationID string // instructions, context, or commentary
}

type RealtimeContextSession interface {
	AppendContext(RealtimeContextUpdate) error
}

// BeginClose starts provider finalization without blocking event consumption.
// Close remains the immediate, idempotent resource-release path.
type RealtimeGracefulSession interface {
	BeginClose() error
}

// Implementations must not launch work from the audio transport. Core owns
// inference and dispatch, and distinguishes a delegation from a tool call.
type realtimeExecutionResult struct {
	response   ChatResponse
	err        error
	revision   uint64
	inputCount int
}

// Optional provider-owned inference transport. Core still owns the managed
// child, tool admission, history, and lifecycle.
type RealtimeExecutionSession interface {
	Execute(context.Context, []Message, string, []NativeTool) (ChatResponse, error)
}

func cloneRealtimeExecutionConfig(in *RealtimeExecutionConfig) *RealtimeExecutionConfig {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}
