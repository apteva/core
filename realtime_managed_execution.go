package core

import (
	"context"
	"fmt"
)

// Transport adapter only: the ordinary managed child's Thinker remains the
// sole owner of history, tools, permissions, budgets, and assignment results.
type realtimeManagedProvider struct {
	LLMProvider
	session func() RealtimeSession
}

func (p *realtimeManagedProvider) Chat(ctx context.Context, m []Message, model string, tools []NativeTool, onChunk func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	s, ok := p.session().(RealtimeExecutionSession)
	if !ok {
		return ChatResponse{}, fmt.Errorf("provider-managed execution session unavailable")
	}
	r, err := s.Execute(ctx, m, model, tools)
	if err == nil && onChunk != nil && r.Text != "" {
		onChunk(r.Text)
	}
	return r, err
}
func (p *realtimeManagedProvider) WithBuiltins(_ []string) LLMProvider { return p }
func (p *realtimeManagedProvider) ManagesInferenceLifecycle() bool     { return true }

func managesInferenceLifecycle(p LLMProvider) bool {
	v, ok := p.(interface{ ManagesInferenceLifecycle() bool })
	return ok && v.ManagesInferenceLifecycle()
}
