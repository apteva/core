package core

// fallbackFits reports whether a fallback can take the prepared request. An
// oversized request is not routed to a smaller provider, and live history is
// not compacted just to attempt an unsuitable one. A skip is reported once
// per provider and request attempt.
func (t *Thinker) fallbackFits(primary string, provider LLMProvider, messages []Message) (requestBudget, bool) {
	name := provider.Name()
	var tools []NativeTool
	if provider.SupportsNativeTools() && t.registry != nil {
		tools = t.prepareNativeTools(name)
	}
	budget := estimatePreparedRequest(name, modelIDForProvider(provider, t.model), fileRefMessagesForModel(messages), tools)
	if p, ok := provider.(*AnthropicProvider); ok {
		budget.finish(p.outputTokenLimit(budget.Model))
	}
	if !budget.OverBudget {
		return budget, true
	}
	if t.telemetry != nil {
		t.telemetry.Emit("llm.fallback.skipped", t.threadID, map[string]any{
			"reason": "input_budget_exceeded", "budget": budget,
			"primary_provider": primary,
		})
	}
	return budget, false
}
