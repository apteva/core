package core

// Select by configured order, but do not route an oversized request to a
// smaller provider or compact live history just to attempt an unsuitable one.
func (t *Thinker) compatibleFallback(primary string, messages []Message) LLMProvider {
	for _, name := range t.pool.order {
		if name == primary {
			continue
		}
		provider := t.pool.Get(name)
		if provider == nil {
			continue
		}
		var tools []NativeTool
		if provider.SupportsNativeTools() && t.registry != nil {
			tools = t.prepareNativeTools(name)
		}
		budget := estimatePreparedRequest(name, modelIDForProvider(provider, t.model), fileRefMessagesForModel(messages), tools)
		if p, ok := provider.(*AnthropicProvider); ok {
			budget.finish(p.outputTokenLimit(budget.Model))
		}
		if !budget.OverBudget {
			return provider
		}
		if t.telemetry != nil {
			t.telemetry.Emit("llm.fallback.skipped", t.threadID, map[string]any{
				"reason": "input_budget_exceeded", "budget": budget,
				"primary_provider": primary,
			})
		}
	}
	return nil
}
