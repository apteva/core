package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// A pin is a thread's explicit provider choice: from spawn, the HTTP API, or a
// pace switch. It is a preference, not a requirement. When the pinned provider
// isn't configured the thread keeps working on what it would otherwise inherit,
// says so through telemetry and GET /threads, and returns to its pin as soon as
// the provider is configured again.

// invalidProviderError reports a request that named a provider the agent
// doesn't have. Callers see it as a bad request, not a server failure.
type invalidProviderError struct {
	name      string
	realtime  bool
	available []string
}

func (e *invalidProviderError) Error() string {
	kind := "provider"
	if e.realtime {
		kind = "realtime provider"
	}
	if len(e.available) == 0 {
		return fmt.Sprintf("unknown %s %q (none configured)", kind, e.name)
	}
	return fmt.Sprintf("unknown %s %q (available: %s)", kind, e.name, strings.Join(e.available, ", "))
}

// selectableProvider returns the named provider, or an error listing the
// names a caller can use instead.
func selectableProvider(pool *ProviderPool, current LLMProvider, name string) (LLMProvider, error) {
	if p := pool.Get(name); p != nil {
		return p, nil
	}
	if pool == nil && current != nil && current.Name() == name {
		return current, nil
	}
	return nil, &invalidProviderError{name: name, available: pool.Names()}
}

// validateSpawnProvider checks a new thread's requested provider. It belongs at
// the entry points only: restoring a saved thread must not refuse, or a worker
// would be lost because its provider was removed while the agent was down.
func (tm *ThreadManager) validateSpawnProvider(name string, realtime bool) error {
	if name == "" {
		return nil
	}
	if realtime {
		if tm.parent.pool.RealtimeByName(name) == nil {
			return &invalidProviderError{name: name, realtime: true, available: tm.parent.pool.RealtimeNames()}
		}
		return nil
	}
	_, err := selectableProvider(tm.parent.pool, tm.parent.provider, name)
	return err
}

// selectThreadProvider picks what a thread runs on: its pin when the pool has
// it, otherwise what it inherits. Its builtin scope applies either way.
func selectThreadProvider(pool *ProviderPool, pin string, inherited LLMProvider, scope []string) LLMProvider {
	selected := inherited
	if pin != "" {
		if p := pool.Get(pin); p != nil {
			selected = p
		}
	}
	if scope != nil && selected != nil {
		selected = selected.WithBuiltins(scope)
	}
	return selected
}

func cloneBuiltinScope(scope []string) []string {
	if scope == nil {
		return nil
	}
	return append([]string{}, scope...)
}

// restoredProviderName returns the provider to request for a saved thread.
//
// Records saved before pins were stored hold whichever provider the worker last
// ran on, so a real pin and an inherited provider look the same. A record that
// matches its parent can safely follow it. One that differs keeps that provider,
// which is what restoring has always done.
func (tm *ThreadManager) restoredProviderName(pt PersistentThread) string {
	if pt.Realtime || pt.WorkMode {
		return pt.Provider
	}
	if pt.ProviderPin != nil {
		return *pt.ProviderPin
	}
	if tm.parent.provider != nil && pt.Provider == tm.parent.provider.Name() {
		return ""
	}
	return pt.Provider
}

// reportThreadRestoreFailure records a saved thread that didn't come back.
func (t *Thinker) reportThreadRestoreFailure(id string, err error) {
	logMsg("RESPAWN", fmt.Sprintf("thread %q was not restored: %v", id, err))
	if t.telemetry != nil {
		t.telemetry.Emit("thread.restore_failed", id, map[string]any{"error": err.Error()})
	}
}

func (t *Thinker) providerPinMissing() bool {
	return t.providerPin != "" && (t.provider == nil || t.provider.Name() != t.providerPin)
}

// reportProviderPinState announces changes in whether this thread is running on
// its pin, once per change.
func (t *Thinker) reportProviderPinState(source string) {
	missing := t.providerPinMissing()
	if missing == t.pinMissingReported {
		return
	}
	t.pinMissingReported = missing
	runningOn := ""
	if t.provider != nil {
		runningOn = t.provider.Name()
	}
	event := "thread.provider_restored"
	if missing {
		event = "thread.provider_missing"
		logMsg("PROVIDER", fmt.Sprintf("[%s] pinned provider %q is not configured; running on %q", t.threadID, t.providerPin, runningOn))
	} else {
		logMsg("PROVIDER", fmt.Sprintf("[%s] running on %q", t.threadID, runningOn))
	}
	if t.telemetry != nil {
		t.telemetry.Emit(event, t.threadID, map[string]any{"pinned_provider": t.providerPin, "running_on": runningOn, "source": source})
	}
}

// syncChildProviders moves this thinker's children onto a new provider pool,
// leaving each where a restart would put it. It runs on the owner's loop and
// never waits: each child applies the change on its own loop and then repeats
// this for its children. A busy worker can't hold up its owner, and a child
// spawned during the change is either listed here or inherits the new pool.
//
// Applying the change cancels a worker's in-flight LLM request, which is then
// retried on the new provider.
func (t *Thinker) syncChildProviders(pool *ProviderPool) {
	if t.threads == nil {
		return
	}
	t.threads.mu.RLock()
	children := make([]*Thread, 0, len(t.threads.threads))
	for _, thread := range t.threads.threads {
		children = append(children, thread)
	}
	t.threads.mu.RUnlock()

	inherited := t.provider
	for _, thread := range children {
		// A voice backend worker belongs to one live call and ends with it. The
		// next call builds a new worker from the updated realtime thread.
		if thread.WorkMode {
			continue
		}
		// An invalidating update would restart a realtime thread's backend work
		// mid-call, so realtime threads take the change without one.
		_, _ = thread.Thinker.queueRuntimeMutation(func() error {
			thread.applyProviderPool(pool, inherited)
			return nil
		}, !thread.IsRealtime)
	}
}

// applyProviderPool runs on the thread's own loop.
func (thread *Thread) applyProviderPool(pool *ProviderPool, inherited LLMProvider) {
	t := thread.Thinker
	t.pool = pool
	if thread.IsRealtime {
		// The live session keeps its own providers. This only affects the next one.
		t.provider = selectThreadProvider(pool, thread.ProviderName, inherited, t.builtinScope)
		thread.reportRealtimeProviderState(pool)
	} else {
		t.provider = selectThreadProvider(pool, t.providerPin, inherited, t.builtinScope)
		thread.nativeToolDocs.Store(poolSupportsNativeTools(pool))
		if t.rebuildPrompt != nil && len(t.messages) > 0 {
			if prompt := t.rebuildPrompt(""); prompt != t.messages[0].Content {
				t.messages[0] = Message{Role: "system", Content: prompt}
			}
		}
		t.resetPromptCache("provider_changed")
		t.publishContextStatus()
	}
	t.reportProviderPinState("config_update")
	t.publishRuntimeStatus()
	if !thread.IsRealtime && t.config != nil {
		if err := t.config.RefreshThread(persistentThreadState(thread)); err != nil {
			logMsg("PROVIDER", fmt.Sprintf("[%s] save thread after provider change: %v", t.threadID, err))
		}
	}
	t.syncChildProviders(pool)
}

func (thread *Thread) reportRealtimeProviderState(pool *ProviderPool) {
	rt := thread.Realtime
	if rt == nil || rt.provider == nil || pool.RealtimeByName(rt.provider.Name()) != nil {
		return
	}
	name := rt.provider.Name()
	logMsg("PROVIDER", fmt.Sprintf("[%s] realtime provider %q is not configured; the live session continues until it ends", thread.ID, name))
	if thread.Thinker.telemetry != nil {
		thread.Thinker.telemetry.Emit("thread.provider_missing", thread.ID, map[string]any{"pinned_provider": name, "running_on": name, "realtime": true, "source": "config_update"})
	}
}

// providerConfigsEqual compares what a provider list means, not how it was
// decoded: unset and empty optional fields are the same thing.
func providerConfigsEqual(a, b []ProviderConfig) bool {
	if len(a) == 0 && len(b) == 0 {
		return true
	}
	left, leftErr := json.Marshal(a)
	right, rightErr := json.Marshal(b)
	return leftErr == nil && rightErr == nil && bytes.Equal(left, right)
}
