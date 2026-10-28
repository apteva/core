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
		current.runLaunchMu.Lock()
		current.shuttingDown.Store(true)
		current.runLaunchMu.Unlock()
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
			go func(th *Thinker) {
				defer wg.Done()
				th.runScheduled.Wait()
				th.runMu.Lock()
				th.runMu.Unlock()
			}(current)
		}
		wg.Wait()
		// A child can stage a parent report from its Run cleanup. Join report
		// workers only after all Run loops finish, so none can be created after
		// we inspect their completion channels. Shutdown must not return while
		// a report worker can still write configuration or inbox files.
		for _, current := range all {
			current.parentReportMu.Lock()
			reportDone := current.parentReportDone
			current.parentReportMu.Unlock()
			if reportDone != nil {
				<-reportDone
			}
		}
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

// Register before launching: Run may not acquire runMu until after Shutdown
// starts. The launch gate prevents WaitGroup additions after shutdown starts.
func (t *Thinker) startRunLoop(run func()) bool {
	t.runLaunchMu.Lock()
	defer t.runLaunchMu.Unlock()
	if t.shuttingDown.Load() {
		return false
	}
	t.runScheduled.Add(1)
	go func() { defer t.runScheduled.Done(); run() }()
	return true
}
