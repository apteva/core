package core

import (
	"context"
	"fmt"
)

type runtimeMutation struct {
	apply              func() error
	done               chan error
	invalidatesRequest bool
}

// The loop owns mutable conversation/provider state. External mutations cancel
// only the current request, then run at a loop boundary before it can dispatch
// stale tool calls. Unstarted fixture/restoration instances apply synchronously.
func (t *Thinker) mutateRuntime(fn func() error) error {
	return t.enqueueRuntimeMutation(fn, true)
}

// createRuntimeChild queues child construction on the owning loop without
// invalidating its inference. The callback may construct a child and publish
// inbox events, but must not edit the parent's prompt, grants, or provider.
func (t *Thinker) createRuntimeChild(fn func() error) error {
	return t.enqueueRuntimeMutation(fn, false)
}

func (t *Thinker) enqueueRuntimeMutation(fn func() error, invalidatesRequest bool) error {
	t.mutationMu.Lock()
	if t.mutationWake == nil {
		t.mutationWake = make(chan struct{}, 1)
	}
	if !t.runtimeRunning {
		defer t.mutationMu.Unlock()
		return fn()
	}
	command := runtimeMutation{apply: fn, done: make(chan error, 1), invalidatesRequest: invalidatesRequest}
	t.mutations = append(t.mutations, command)
	if invalidatesRequest && t.requestCancel != nil {
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
	return t.applyQueuedRuntimeMutations(false)
}

// While inference is running, only child construction is safe. Keep commands
// behind an invalidating update queued until inference returns, preserving FIFO.
// The result says whether the model response became stale, not whether any
// command ran: child creation must not discard a completed response.
func (t *Thinker) applyQueuedRuntimeMutations(inferenceRunning bool) bool {
	t.mutationMu.Lock()
	count := len(t.mutations)
	if inferenceRunning {
		count = 0
		for count < len(t.mutations) && !t.mutations[count].invalidatesRequest {
			count++
		}
	}
	batch := t.mutations[:count]
	if count == len(t.mutations) {
		t.mutations = nil
	} else {
		t.mutations = t.mutations[count:]
	}
	select {
	case <-t.mutationWake:
	default:
	}
	t.mutationMu.Unlock()
	invalidated := false
	for _, cmd := range batch {
		invalidated = invalidated || cmd.invalidatesRequest
		cmd.done <- cmd.apply()
	}
	return invalidated
}
func (t *Thinker) runtimeRequest(ctx context.Context) (context.Context, func()) {
	request, cancel := context.WithCancel(ctx)
	t.mutationMu.Lock()
	t.requestCancel = cancel
	for _, cmd := range t.mutations {
		if cmd.invalidatesRequest {
			cancel()
			break
		}
	}
	t.mutationMu.Unlock()
	return request, func() { cancel(); t.mutationMu.Lock(); t.requestCancel = nil; t.mutationMu.Unlock() }
}

// The owner services child creation while the provider runs. Child construction
// reads stable parent dependencies and uses the registry/index/manager locks;
// request-invalidating edits wait until inference exits. Parent inbox events
// remain queued for the next turn, and the current response stays usable.
func (t *Thinker) callLLMWithRuntimeMutations(ctx context.Context, messages []Message) (ChatResponse, error) {
	requestCtx, finish := t.runtimeRequest(ctx)
	defer finish()
	type result struct {
		response ChatResponse
		err      error
		panic    any
	}
	completed := make(chan result, 1)
	go func() {
		var out result
		defer func() {
			out.panic = recover()
			completed <- out
		}()
		out.response, out.err = t.callLLMWithRetryMessages(requestCtx, messages)
	}()
	for {
		select {
		case out := <-completed:
			if out.panic != nil {
				panic(out.panic) // preserve Run's existing panic recovery
			}
			return out.response, out.err
		case <-t.mutationWake:
			t.applyQueuedRuntimeMutations(true)
		}
	}
}

func (t *Thinker) recordPersistenceFailure(err error) {
	logMsg("SESSION", fmt.Sprintf("[%s] persistence failed; stopping before further side effects: %v", t.threadID, err))
	if t.telemetry != nil {
		t.telemetry.Emit("session.persistence_failed", t.threadID, map[string]string{"error": err.Error()})
	}
	t.shuttingDown.Store(true) // retain durable worker definition for recovery
	t.Stop()
}
