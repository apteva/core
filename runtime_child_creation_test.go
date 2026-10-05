package core

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

type runtimeChildCall struct {
	thread   string
	ctx      context.Context
	messages []Message
	reply    chan ChatResponse
}

type runtimeChildProvider struct {
	LLMProvider
	calls chan runtimeChildCall
}

func (p *runtimeChildProvider) Chat(ctx context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	thread, _ := ctx.Value(blobCallerThreadKey{}).(string)
	call := runtimeChildCall{thread: thread, ctx: ctx, messages: cloneMessages(messages), reply: make(chan ChatResponse, 1)}
	select {
	case p.calls <- call:
	case <-ctx.Done():
		return ChatResponse{}, ctx.Err()
	}
	select {
	case response := <-call.reply:
		return response, nil
	case <-ctx.Done():
		return ChatResponse{}, ctx.Err()
	}
}

func (p *runtimeChildProvider) WithBuiltins([]string) LLMProvider { return p }

func awaitRuntimeChildCall(t *testing.T, provider *runtimeChildProvider) runtimeChildCall {
	t.Helper()
	select {
	case call := <-provider.calls:
		return call
	case <-time.After(3 * time.Second):
		t.Fatal("provider did not receive a request")
		return runtimeChildCall{}
	}
}

func awaitRuntimeHTTP(t *testing.T, invoke func() *httptest.ResponseRecorder) *httptest.ResponseRecorder {
	t.Helper()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- invoke() }()
	select {
	case w := <-done:
		return w
	case <-time.After(3 * time.Second):
		t.Fatal("thread creation waited for the unrelated model response")
		return nil
	}
}

func TestRuntimeChildCreationPreservesRequestsAndParentEvents(t *testing.T) {
	api, thinker, parked := newPersistentThreadTestAPI(t)
	provider := &runtimeChildProvider{LLMProvider: parked, calls: make(chan runtimeChildCall, 16)}
	thinker.provider = provider
	thinker.pool = &ProviderPool{providers: map[string]LLMProvider{provider.Name(): provider}, order: []string{provider.Name()}, default_: provider.Name()}
	runDone := make(chan struct{})
	go func() { defer close(runDone); thinker.Run() }()
	t.Cleanup(func() {
		thinker.Stop()
		thinker.threads.KillAll()
		select {
		case <-runDone:
		case <-time.After(3 * time.Second):
			t.Error("main did not stop")
		}
		thinker.telemetry.Stop()
	})
	main := awaitRuntimeChildCall(t, provider)
	if main.thread != "main" {
		t.Fatalf("first request came from %q", main.thread)
	}
	var workers []runtimeChildCall
	ids := []string{"chat-overlap-a", "chat-overlap-b", "chat-overlap-c", "chat-overlap-lazy"}
	for i, id := range ids {
		w := awaitRuntimeHTTP(t, func() *httptest.ResponseRecorder {
			if i == len(ids)-1 {
				recorder := httptest.NewRecorder()
				api.postEvent(recorder, httptest.NewRequest(http.MethodPost, "/event", strings.NewReader(`{"thread_id":"chat-overlap-lazy","message":"Synthetic conversation event"}`)))
				return recorder
			}
			body := map[string]any{
				"directive": "Handle this independent synthetic conversation.", "tools": []string{}, "mcp": []string{},
				"events": []map[string]string{{"id": "initial-" + id, "message": "Synthetic conversation event"}},
			}
			if i != 0 {
				return postThreadForTest(t, api, id, body)
			}
			// Concurrent first messages may all try to lazily create the same
			// room. Only one constructor and one parent notification may win.
			start := make(chan struct{})
			responses := make(chan *httptest.ResponseRecorder, 4)
			for range 4 {
				go func() { <-start; responses <- postThreadForTest(t, api, id, body) }()
			}
			close(start)
			created := 0
			var result *httptest.ResponseRecorder
			for range 4 {
				result = <-responses
				if result.Code != http.StatusOK {
					t.Errorf("concurrent duplicate spawn: HTTP %d %s", result.Code, result.Body.String())
				}
				if strings.Contains(result.Body.String(), `"created"`) {
					created++
				}
			}
			if created != 1 {
				t.Errorf("concurrent requests created %d children, want one", created)
			}
			return result
		})
		if w.Code != http.StatusOK {
			t.Fatalf("create %s: HTTP %d %s", id, w.Code, w.Body.String())
		}
		if main.ctx.Err() != nil {
			t.Fatalf("child %s canceled main: %v", id, main.ctx.Err())
		}
		for _, earlier := range workers {
			if earlier.ctx.Err() != nil {
				t.Fatalf("child %s canceled earlier worker %s", id, earlier.thread)
			}
		}
		worker := awaitRuntimeChildCall(t, provider)
		if worker.thread != id {
			t.Fatalf("create %s produced request for %s", id, worker.thread)
		}
		workers = append(workers, worker)
		if _, ok := persistentThreadByID(thinker.config.GetThreads(), id); !ok {
			t.Fatalf("child %s returned success without its durable definition", id)
		}
	}
	w := awaitRuntimeHTTP(t, func() *httptest.ResponseRecorder { return postThreadForTest(t, api, ids[0], map[string]any{}) })
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"exists"`) || main.ctx.Err() != nil {
		t.Fatalf("duplicate creation changed main or idempotency: HTTP %d ctx=%v", w.Code, main.ctx.Err())
	}
	// Complete the original response. Run must persist it exactly once and
	// include the queued notifications in the next request, without reissuing it.
	const retained = "Original main response completed successfully."
	main.reply <- ChatResponse{Text: retained}
	next := awaitRuntimeChildCall(t, provider)
	if next.thread != "main" {
		t.Fatalf("next request belongs to %s", next.thread)
	}
	assistantCount := 0
	for _, message := range next.messages {
		if message.Role == "assistant" && message.Content == retained {
			assistantCount++
		}
	}
	if assistantCount != 1 {
		t.Fatalf("original response was discarded or duplicated: count=%d", assistantCount)
	}
	for _, id := range ids {
		count := 0
		for _, message := range next.messages {
			count += strings.Count(message.Content, "[thread:"+id+"] started")
		}
		if count != 1 {
			t.Errorf("parent notification for %s count=%d, want exactly once", id, count)
		}
	}
	if len(traceEvents(t, thinker.telemetry, "llm.cancelled")) != 0 {
		t.Fatal("thread creation emitted cancellation telemetry")
	}
}

func TestRuntimeChildCommandsRespectInvalidatingUpdateOrder(t *testing.T) {
	thinker := retryTestThinker(&scriptedRetryProvider{name: "test"})
	thinker.beginRuntime()
	defer thinker.endRuntime()
	ctx, finish := thinker.runtimeRequest(context.Background())
	defer finish()
	var order []string
	done := make(chan error, 3)
	enqueue := func(name string, invalidates bool, count int) {
		go func() {
			done <- thinker.enqueueRuntimeMutation(func() error { order = append(order, name); return nil }, invalidates)
		}()
		deadline := time.Now().Add(time.Second)
		for {
			thinker.mutationMu.Lock()
			queued := len(thinker.mutations)
			thinker.mutationMu.Unlock()
			if queued == count {
				return
			}
			if time.Now().After(deadline) {
				t.Fatal("command did not queue")
			}
			runtime.Gosched()
		}
	}
	enqueue("first child", false, 1)
	if ctx.Err() != nil {
		t.Fatal("child construction canceled inference")
	}
	enqueue("configuration", true, 2)
	enqueue("second child", false, 3)
	if !errors.Is(ctx.Err(), context.Canceled) {
		t.Fatal("request-invalidating edit did not cancel inference")
	}
	if thinker.applyQueuedRuntimeMutations(true) || !reflect.DeepEqual(order, []string{"first child"}) {
		t.Fatalf("applied an unsafe command during inference: %v", order)
	}
	if !thinker.applyRuntimeMutations() || !reflect.DeepEqual(order, []string{"first child", "configuration", "second child"}) {
		t.Fatalf("lost invalidation or FIFO order: %v", order)
	}
	for range 3 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
}

func TestRuntimeChildPendingBeforeRequestDoesNotInvalidate(t *testing.T) {
	thinker := retryTestThinker(&scriptedRetryProvider{name: "test", response: ChatResponse{Text: "retained"}})
	thinker.beginRuntime()
	defer thinker.endRuntime()
	done := make(chan error, 1)
	go func() { done <- thinker.createRuntimeChild(func() error { return nil }) }()
	select {
	case <-thinker.mutationWake:
	case <-time.After(time.Second):
		t.Fatal("child creation did not queue")
	}
	ctx, finish := thinker.runtimeRequest(context.Background())
	defer finish()
	if ctx.Err() != nil || thinker.applyRuntimeMutations() {
		t.Fatal("pending child command invalidated a new request or its response")
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

type runtimePanicProvider struct{ LLMProvider }

func (p *runtimePanicProvider) Chat(context.Context, []Message, string, []NativeTool, func(string), func(string), func(string, string, string)) (ChatResponse, error) {
	panic("runtime inference panic")
}

func TestRuntimeInferencePanicReturnsToOwner(t *testing.T) {
	thinker := retryTestThinker(&runtimePanicProvider{LLMProvider: &scriptedRetryProvider{name: "panic"}})
	thinker.beginRuntime()
	defer thinker.endRuntime()
	defer func() {
		if value := recover(); value != "runtime inference panic" {
			t.Errorf("inference panic was lost: %v", value)
		}
	}()
	_, _ = thinker.callLLMWithRuntimeMutations(context.Background(), thinker.messages)
	t.Fatal("inference panic was swallowed")
}
