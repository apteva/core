package core

import (
	"context"
	"strings"
	"sync"
)

// OpenCodeGoProvider is a thin policy wrapper over the shared OpenAI-
// compatible transport. Auto uses a model-aware effort chosen for reliable
// tool workflows while an explicit agent choice still wins. Models that reject
// the field are remembered and use their provider default on subsequent
// requests.
type OpenCodeGoProvider struct {
	compat    *OpenAICompatProvider
	reasoning ReasoningSettings
	support   *openCodeGoReasoningSupport
}

type openCodeGoReasoningSupport struct {
	mu          sync.RWMutex
	unsupported map[string]struct{}
}

func newOpenCodeGoReasoningSupport() *openCodeGoReasoningSupport {
	return &openCodeGoReasoningSupport{unsupported: map[string]struct{}{}}
}

func (s *openCodeGoReasoningSupport) accepts(model string) bool {
	if s == nil {
		return true
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, rejected := s.unsupported[normalizeOpenCodeGoModel(model)]
	return !rejected
}

func (s *openCodeGoReasoningSupport) markUnsupported(model string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.unsupported[normalizeOpenCodeGoModel(model)] = struct{}{}
	s.mu.Unlock()
}

func normalizeOpenCodeGoModel(model string) string {
	return strings.ToLower(strings.TrimSpace(model))
}

func (p *OpenCodeGoProvider) Name() string { return p.compat.Name() }

func (p *OpenCodeGoProvider) Models() map[ModelTier]string { return p.compat.Models() }

func (p *OpenCodeGoProvider) CostPer1M() (float64, float64, float64) {
	return p.compat.CostPer1M()
}

func (p *OpenCodeGoProvider) SupportsNativeTools() bool { return p.compat.SupportsNativeTools() }

func (p *OpenCodeGoProvider) AvailableBuiltinTools() []BuiltinTool {
	return p.compat.AvailableBuiltinTools()
}

func (p *OpenCodeGoProvider) SetBuiltinTools(tools []string) {
	p.compat.SetBuiltinTools(tools)
}

func (p *OpenCodeGoProvider) WithBuiltins(builtins []string) LLMProvider {
	clone := *p
	clone.compat = p.compat.WithBuiltins(builtins).(*OpenAICompatProvider)
	return &clone
}

func (p *OpenCodeGoProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	clone := *p
	clone.reasoning = settings
	return &clone
}

func (p *OpenCodeGoProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	requested := normalizeReasoningLevel(p.reasoning.Level).String()
	effort := openCodeGoReasoningEffort(model, p.reasoning.Level)

	if p.support.accepts(model) {
		ctx = withOpenAICompatRequestOptions(ctx, openAICompatRequestOptions{
			OptionalFields: map[string]openAICompatOptionalField{
				"reasoning_effort": {
					Value: effort,
					OnUnsupported: func() {
						p.support.markUnsupported(model)
					},
				},
			},
		})
	}

	resp, err := p.compat.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
	resp.RequestedReasoningEffort = requested
	if p.support.accepts(model) {
		resp.EffectiveReasoningEffort = effort
	} else {
		resp.EffectiveReasoningEffort = "provider-default"
	}
	return resp, err
}

func openCodeGoReasoningEffort(model string, level ReasoningLevel) string {
	switch normalizeReasoningLevel(level) {
	case ReasoningAuto:
		if normalizeOpenCodeGoModel(model) == "glm-5.2" {
			return "medium"
		}
		return "minimal"
	case ReasoningNone:
		return "none"
	case ReasoningMinimal:
		return "minimal"
	case ReasoningLow:
		return "low"
	case ReasoningMedium:
		return "medium"
	case ReasoningHigh:
		return "high"
	case ReasoningXHigh:
		return "xhigh"
	default:
		return "minimal"
	}
}

var _ LLMProvider = (*OpenCodeGoProvider)(nil)
var _ ReasoningProvider = (*OpenCodeGoProvider)(nil)
