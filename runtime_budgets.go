package core

import (
	"context"
	"strings"
)

// Separate main lanes reserve control progress even when all worker capacity
// is occupied. Channel waiters are bounded by worker/tool admission limits.
type runtimeBudgets struct{ llm, mainLLM, tools, mainTools chan struct{} }

func (b *EventBus) limits() *runtimeBudgets {
	b.budgetOnce.Do(func() {
		b.budget = &runtimeBudgets{make(chan struct{}, 16), make(chan struct{}, 2), make(chan struct{}, 64), make(chan struct{}, 16)}
	})
	return b.budget
}
func acquireBudget(ctx context.Context, slots chan struct{}) (func(), error) {
	select {
	case slots <- struct{}{}:
		return func() { <-slots }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
func (t *Thinker) acquireLLMBudget(ctx context.Context) (func(), error) {
	if t.bus == nil {
		return func() {}, nil
	}
	b := t.bus.limits()
	slots := b.llm
	if t.threadID == "main" {
		slots = b.mainLLM
	}
	return acquireBudget(ctx, slots)
}
func (t *Thinker) acquireExecutionBudget(ctx context.Context) (func(), error) {
	if t.bus == nil {
		return func() {}, nil
	}
	b := t.bus.limits()
	slots := b.tools
	if t.threadID == "main" {
		slots = b.mainTools
	}
	return acquireBudget(ctx, slots)
}

func permanentProviderError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, code := range []string{"api error 400:", "api error 401:", "api error 403:", "api error 404:", "api error 422:", "invalid_api_key", "invalid api key"} {
		if strings.Contains(msg, code) {
			return true
		}
	}
	return false
}
