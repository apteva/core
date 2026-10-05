package core

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

type runtimeChildLiveResult struct {
	thread   string
	response ChatResponse
	err      error
}

type runtimeChildLiveProvider struct {
	LLMProvider
	mu       sync.Mutex
	calls    map[string]int
	stream   chan context.Context
	release  chan struct{}
	released sync.Once
	results  chan runtimeChildLiveResult
}

func (p *runtimeChildLiveProvider) WithBuiltins([]string) LLMProvider { return p }
func (p *runtimeChildLiveProvider) releaseStream()                    { p.released.Do(func() { close(p.release) }) }

func (p *runtimeChildLiveProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk, onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	thread, _ := ctx.Value(blobCallerThreadKey{}).(string)
	p.mu.Lock()
	p.calls[thread]++
	turn := p.calls[thread]
	p.mu.Unlock()
	// Every child performs one real inference and dispatches its local report.
	// Park later continuations to bound this test's external model calls.
	if thread != "main" && turn > 1 {
		<-ctx.Done()
		return ChatResponse{}, ctx.Err()
	}
	var first sync.Once
	gate := func() {
		if thread == "main" && turn == 1 {
			first.Do(func() {
				p.stream <- ctx
				// Pause only after genuine provider output has arrived, so all
				// child POSTs deterministically overlap the live parent stream.
				select {
				case <-p.release:
				case <-ctx.Done():
				}
			})
		}
	}
	response, err := p.LLMProvider.Chat(ctx, messages, model, tools,
		func(chunk string) {
			if chunk != "" {
				gate()
			}
			if onChunk != nil {
				onChunk(chunk)
			}
		},
		func(chunk string) {
			if chunk != "" {
				gate()
			}
			if onThinking != nil {
				onThinking(chunk)
			}
		},
		func(name, id, chunk string) {
			gate()
			if onToolChunk != nil {
				onToolChunk(name, id, chunk)
			}
		})
	p.results <- runtimeChildLiveResult{thread: thread, response: response, err: err}
	return response, err
}

//	RUN_CODEX_RUNTIME_CHILD_LIVE=1 go test -v -count=1 \
//	  -run '^TestIntegration_CodexGPT61SolChildCreationPreservesInference$' -timeout 6m .
//
// Uses real Sol 6.1 requests with local-only probe tools and temporary sessions.
func TestIntegration_CodexGPT61SolChildCreationPreservesInference(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_RUNTIME_CHILD_LIVE") != "1" {
		t.Skip("set RUN_CODEX_RUNTIME_CHILD_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live child-creation test requested without a valid Codex credential")
	}
	api, thinker, _ := newPersistentThreadTestAPI(t)
	provider := &runtimeChildLiveProvider{LLMProvider: toolReasonLiveProvider(t, token), calls: map[string]int{}, stream: make(chan context.Context, 1), release: make(chan struct{}), results: make(chan runtimeChildLiveResult, 8)}
	thinker.provider = provider
	thinker.pool = &ProviderPool{providers: map[string]LLMProvider{provider.Name(): provider}, order: []string{provider.Name()}, default_: provider.Name()}
	reports := make(chan string, 3)
	thinker.registry.Register(&ToolDef{Name: "probe_report", Description: "Record the exact synthetic marker requested in your directive. This is the only operation to perform in this turn.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"marker": map[string]any{"type": "string"}}, "required": []string{"marker"}}, Handler: func(args map[string]string) ToolResponse {
		reports <- args["marker"]
		return ToolResponse{Text: `{"accepted":true}`}
	}})
	thinker.registry.Register(&ToolDef{Name: "probe_observe", Description: "Report the thread IDs from the supplied parent startup events and the marker from the preceding parent report.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"thread_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}}, "parent_marker": map[string]any{"type": "string"}}, "required": []string{"thread_ids", "parent_marker"}}})
	thinker.messages = []Message{{Role: "system", Content: "This is an isolated integration test. Follow the latest user instruction using the specified probe tool exactly once. Do not delegate or perform additional work."}, {Role: "user", Content: "Call probe_report once with marker parent-live-overlap. Do not call probe_observe yet."}}
	thinker.beginRuntime()
	t.Cleanup(func() {
		provider.releaseStream()
		thinker.threads.KillAll()
		thinker.endRuntime()
		thinker.telemetry.Stop()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	type parentResult struct {
		response ChatResponse
		err      error
	}
	parentDone := make(chan parentResult, 1)
	go func() {
		response, err := thinker.callLLMWithRuntimeMutations(ctx, thinker.messages)
		parentDone <- parentResult{response, err}
	}()
	var parentCtx context.Context
	select {
	case parentCtx = <-provider.stream:
	case result := <-parentDone:
		t.Fatalf("parent ended before live overlap: %v", result.err)
	case <-ctx.Done():
		t.Fatal("real parent stream did not produce output")
	}
	ids := []string{"chat-live-child-a", "chat-live-child-b", "chat-live-child-c"}
	for _, id := range ids {
		w := awaitRuntimeHTTP(t, func() *httptest.ResponseRecorder {
			return postThreadForTest(t, api, id, map[string]any{"directive": fmt.Sprintf("Call probe_report exactly once with marker %s using the available tool. Do not call any other tool and do not delegate.", id), "tools": []string{"probe_report"}, "mcp": []string{}})
		})
		if w.Code != http.StatusOK {
			t.Fatalf("create %s: HTTP %d %s", id, w.Code, w.Body.String())
		}
		if parentCtx.Err() != nil {
			t.Fatalf("create %s canceled real parent stream: %v", id, parentCtx.Err())
		}
	}
	provider.releaseStream()
	var parent parentResult
	select {
	case parent = <-parentDone:
	case <-ctx.Done():
		t.Fatal("parent request did not finish")
	}
	if parent.err != nil {
		t.Fatalf("real parent request: %v", parent.err)
	}
	if thinker.applyRuntimeMutations() {
		t.Fatal("child creation invalidated the completed real response")
	}
	if len(parent.response.ToolCalls) != 1 || parent.response.ToolCalls[0].Name != "probe_report" || parent.response.ToolCalls[0].Args["marker"] != "parent-live-overlap" {
		t.Fatalf("original parent report was lost: %+v", parent.response.ToolCalls)
	}
	assertLiveModelToolReason(t, thinker.registry, parent.response.ToolCalls[0])
	workerResults := map[string]bool{}
	for len(workerResults) < len(ids) {
		select {
		case result := <-provider.results:
			if result.err != nil {
				t.Fatalf("real %s inference failed: %v", result.thread, result.err)
			}
			if result.thread == "main" {
				continue
			}
			if workerResults[result.thread] || len(result.response.ToolCalls) != 1 || result.response.ToolCalls[0].Name != "probe_report" || result.response.ToolCalls[0].Args["marker"] != result.thread {
				t.Fatalf("invalid child report for %s: %+v", result.thread, result.response.ToolCalls)
			}
			assertLiveModelToolReason(t, thinker.registry, result.response.ToolCalls[0])
			workerResults[result.thread] = true
		case <-ctx.Done():
			t.Fatal("real workers did not finish")
		}
	}
	var dispatched []string
	for range ids {
		select {
		case marker := <-reports:
			dispatched = append(dispatched, marker)
		case <-ctx.Done():
			t.Fatal("real child tool calls were not dispatched")
		}
	}
	sort.Strings(dispatched)
	if !reflect.DeepEqual(dispatched, ids) {
		t.Fatalf("child tool dispatch=%v, want %v", dispatched, ids)
	}
	events := thinker.sub.DrainTargeted()
	var notifications []string
	for _, id := range ids {
		count := 0
		for _, event := range events {
			if event.Type == EventInbox && strings.Contains(event.Text, "[thread:"+id+"] started") {
				count++
				notifications = append(notifications, event.Text)
			}
		}
		if count != 1 {
			t.Fatalf("parent startup notification %s count=%d", id, count)
		}
	}
	call := parent.response.ToolCalls[0]
	continuation := append(cloneMessages(thinker.messages), Message{Role: "assistant", Content: parent.response.Text, Reasoning: parent.response.Reasoning, ToolCalls: parent.response.ToolCalls, ProviderState: parent.response.ProviderState}, Message{Role: "tool", ToolResults: []ToolResult{{CallID: call.ID, ToolName: call.Name, Content: `{"accepted":true}`}}}, Message{Role: "user", Content: "Parent startup events:\n" + strings.Join(notifications, "\n") + "\nNow call probe_observe once with exactly those three thread IDs and the marker from your original probe_report."})
	response, err := thinker.callLLMWithRuntimeMutations(ctx, continuation)
	if err != nil {
		t.Fatalf("real parent continuation: %v", err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "probe_observe" || response.ToolCalls[0].Args["parent_marker"] != "parent-live-overlap" {
		t.Fatalf("parent lost original result or event continuation: %+v", response.ToolCalls)
	}
	assertLiveModelToolReason(t, thinker.registry, response.ToolCalls[0])
	var observed []string
	if err := json.Unmarshal([]byte(response.ToolCalls[0].Args["thread_ids"]), &observed); err != nil {
		t.Fatal(err)
	}
	sort.Strings(observed)
	if !reflect.DeepEqual(observed, ids) {
		t.Fatalf("parent observed %v, want %v", observed, ids)
	}
	if len(traceEvents(t, thinker.telemetry, "llm.cancelled")) != 0 {
		t.Fatal("overlapping real requests emitted cancellation telemetry")
	}
	provider.mu.Lock()
	parentCalls := provider.calls["main"]
	provider.mu.Unlock()
	if parentCalls != 2 {
		t.Fatalf("parent requests=%d, want original plus explicit continuation", parentCalls)
	}
	t.Log("gpt-6.1-sol: original parent response preserved, three child requests and local tool dispatches succeeded, and parent consumed all three startup notifications in one continuation; zero request cancellations")
}
