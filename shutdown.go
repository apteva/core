package core

import (
	"context"
	"sync"
)

// Shutdown stops work while preserving durable worker definitions and inboxes
// for restart. User-requested kill/done retains its existing removal semantics.
func (t *Thinker) Shutdown(ctx context.Context) error {
	var all []*Thinker
	var collect func(*Thinker)
	collect = func(current *Thinker) {
		current.shuttingDown.Store(true)
		all = append(all, current)
		if current.threads == nil {
			return
		}
		current.threads.mu.RLock()
		children := make([]*Thinker, 0, len(current.threads.threads))
		for _, thread := range current.threads.threads {
			children = append(children, thread.Thinker)
		}
		current.threads.mu.RUnlock()
		for _, child := range children {
			collect(child)
		}
	}
	collect(t)
	for _, current := range all {
		current.Stop()
	}
	done := make(chan struct{})
	go func() {
		var wg sync.WaitGroup
		for _, current := range all {
			wg.Add(1)
			go func(th *Thinker) { defer wg.Done(); th.runMu.Lock(); th.runMu.Unlock() }(current)
		}
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		return ctx.Err()
	}
	for _, srv := range t.mcpServers {
		srv.Close()
	}
	if t.blobs != nil {
		t.blobs.Close()
	}
	if t.telemetry != nil {
		t.telemetry.Stop()
	}
	return nil
}
