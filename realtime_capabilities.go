package core

import "context"

// Optional execution override on a provider's voice configuration. Empty
// values use adapter voice defaults, or inherit the thread when none exist.
type RealtimeExecutionConfig struct {
	Mode         string                   `json:"mode,omitempty"` // thread (default) or provider
	Provider     string                   `json:"provider,omitempty"`
	Model        string                   `json:"model,omitempty"`    // existing Core tier override
	ModelID      string                   `json:"model_id,omitempty"` // exact model for this worker only
	Reasoning    string                   `json:"reasoning,omitempty"`
	PreloadTools []string                 `json:"preload_tools,omitempty"`
	ResultTools  []DirectToolResultPolicy `json:"result_tools,omitempty"`
}

// RealtimeCapabilities describes behavior, not a vendor. The zero/legacy
// profile preserves the existing turn-based direct-tool contract.
type RealtimeCapabilities struct {
	InputSpeechEvents      bool `json:"input_speech_events"`
	RendererInterruption   bool `json:"renderer_interruption_required"`
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
	return RealtimeCapabilities{AudioTruncation: true, InputSpeechEvents: true}
}

// A context update is deliberately distinct from a user message or a tool
// result. Adapters own protocol limits and cannot promote tool data to policy.
type RealtimeContextUpdate struct {
	ClientEventID               string      // optional correlation for provider acknowledgement
	Work                        *ThreadWork // internal metadata; never serialized as provider input
	Kind, Content, DelegationID string      // instructions, context, or commentary
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
	out.PreloadTools = append([]string(nil), in.PreloadTools...)
	out.ResultTools = append([]DirectToolResultPolicy(nil), in.ResultTools...)
	for i := range out.ResultTools {
		out.ResultTools[i].PublicFields = append([]string(nil), in.ResultTools[i].PublicFields...)
	}
	return &out
}

func (c RealtimeExecutionConfig) isZero() bool {
	return c.Mode == "" && c.Provider == "" && c.Model == "" && c.ModelID == "" && c.Reasoning == "" && len(c.PreloadTools) == 0 && len(c.ResultTools) == 0
}

// Optional adapter defaults; providers without delegated execution are unchanged.
type realtimeExecutionDefaultProvider interface {
	DefaultRealtimeExecution(model string) RealtimeExecutionConfig
}
