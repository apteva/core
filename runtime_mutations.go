package core

import (
	"context"
	"fmt"
)

type runtimeMutation struct {
	apply func() error
	done  chan error
}

// The loop owns mutable conversation/provider state. External mutations cancel
// only the current request, then run at a loop boundary before it can dispatch
// stale tool calls. Unstarted fixture/restoration instances apply synchronously.
func (t *Thinker) mutateRuntime(fn func() error) error {
	t.mutationMu.Lock()
	if t.mutationWake == nil {
		t.mutationWake = make(chan struct{}, 1)
	}
	if !t.runtimeRunning {
		defer t.mutationMu.Unlock()
		return fn()
	}
	command := runtimeMutation{apply: fn, done: make(chan error, 1)}
	t.mutations = append(t.mutations, command)
	if t.requestCancel != nil {
		t.requestCancel()
	}
	select {
	case t.mutationWake <- struct{}{}:
	default:
	}
	t.mutationMu.Unlock()
	select {
	case err := <-command.done:
		return err
	case <-t.quit:
		return fmt.Errorf("thread stopped during update")
	}
}

func (t *Thinker) beginRuntime() {
	t.mutationMu.Lock()
	defer t.mutationMu.Unlock()
	if t.mutationWake == nil {
		t.mutationWake = make(chan struct{}, 1)
	}
	t.runtimeRunning = true
}
func (t *Thinker) endRuntime() {
	t.mutationMu.Lock()
	defer t.mutationMu.Unlock()
	t.runtimeRunning = false
	t.requestCancel = nil
	for _, cmd := range t.mutations {
		cmd.done <- fmt.Errorf("thread stopped during update")
	}
	t.mutations = nil
}
func (t *Thinker) applyRuntimeMutations() bool {
	t.mutationMu.Lock()
	batch := t.mutations
	t.mutations = nil
	select {
	case <-t.mutationWake:
	default:
	}
	t.mutationMu.Unlock()
	for _, cmd := range batch {
		cmd.done <- cmd.apply()
	}
	return len(batch) > 0
}
func (t *Thinker) runtimeRequest(ctx context.Context) (context.Context, func()) {
	request, cancel := context.WithCancel(ctx)
	t.mutationMu.Lock()
	t.requestCancel = cancel
	if len(t.mutations) > 0 {
		cancel()
	}
	t.mutationMu.Unlock()
	return request, func() { cancel(); t.mutationMu.Lock(); t.requestCancel = nil; t.mutationMu.Unlock() }
}

func (t *Thinker) recordPersistenceFailure(err error) {
	logMsg("SESSION", fmt.Sprintf("[%s] persistence failed; stopping before further side effects: %v", t.threadID, err))
	if t.telemetry != nil {
		t.telemetry.Emit("session.persistence_failed", t.threadID, map[string]string{"error": err.Error()})
	}
	t.shuttingDown.Store(true) // retain durable worker definition for recovery
	t.Stop()
}
