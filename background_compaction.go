package core

import (
	"context"
	"strings"
)

var compactionSlots = make(chan struct{}, 2)

// Called only by the owner. Capture immutable request inputs before starting
// background work; Session rejects commits after a reset or another rewrite.
func (t *Thinker) startBackgroundCompaction() {
	if t.session == nil || t.provider == nil || !t.session.NeedsCompaction() || !t.compactionActive.CompareAndSwap(false, true) {
		return
	}
	select {
	case compactionSlots <- struct{}{}:
	default:
		t.compactionActive.Store(false)
		return
	}
	session, provider, model := t.session, t.provider, t.provider.Models()[ModelSmall]
	if model == "" {
		model = t.modelID()
	}
	parent := t.toolContext()
	go func() {
		defer t.compactionActive.Store(false)
		defer func() { <-compactionSlots }()
		session.Compact(func(text string) string {
			ctx, cancel := context.WithTimeout(parent, semanticCompactionTimeout)
			defer cancel()
			response, err := provider.Chat(ctx, []Message{
				{Role: "system", Content: "Summarize agent history without inventing facts. Preserve exact identifiers, dates, constraints, decisions, completed and open work, failures, and important tool results."},
				{Role: "user", Content: text},
			}, model, nil, nil, nil, nil)
			if err != nil || ctx.Err() != nil {
				return ""
			}
			return strings.TrimSpace(response.Text)
		})
	}()
}
