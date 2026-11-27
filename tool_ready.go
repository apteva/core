package core

import (
	"context"
	"encoding/json"
)

// toolReadyObserver is told once per tool call, at the moment the provider
// stream has delivered that call's complete arguments. A response that holds
// several calls reports each one as it finishes, not when the stream ends.
type toolReadyObserver func(toolName, callID string, argBytes int)

type toolReadyObserverKey struct{}

func withToolReadyObserver(ctx context.Context, observer toolReadyObserver) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, toolReadyObserverKey{}, observer)
}

func toolReadyObserverFromContext(ctx context.Context) toolReadyObserver {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(toolReadyObserverKey{}).(toolReadyObserver)
	return observer
}

// notifyToolReady reports a call whose identity and arguments are complete.
// Calls without an identity or with arguments that are not a JSON object are
// skipped: the provider rejects those when it assembles the response.
func notifyToolReady(observer toolReadyObserver, toolName, callID, argsJSON string) bool {
	if observer == nil || toolName == "" || callID == "" {
		return false
	}
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(argsJSON), &object) != nil || object == nil {
		return false
	}
	observer(toolName, callID, len(argsJSON))
	return true
}
