package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type profileEventRequest struct {
	ctx      context.Context
	messages []Message
	tools    []NativeTool
	finish   chan ChatResponse
}

type profileEventProvider struct {
	scriptedRetryProvider
	requests chan profileEventRequest
}

func (p *profileEventProvider) SupportsNativeTools() bool { return true }
func (p *profileEventProvider) Chat(ctx context.Context, messages []Message, _ string, tools []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	request := profileEventRequest{ctx: ctx, messages: cloneMessages(messages), tools: append([]NativeTool(nil), tools...), finish: make(chan ChatResponse, 1)}
	p.requests <- request
	select {
	case <-ctx.Done():
		return ChatResponse{}, ctx.Err()
	case response := <-request.finish:
		return response, nil
	}
}
func takeProfileEventRequest(t *testing.T, p *profileEventProvider) profileEventRequest {
	t.Helper()
	select {
	case request := <-p.requests:
		return request
	case <-time.After(3 * time.Second):
		t.Fatal("expected provider request")
		return profileEventRequest{}
	}
}
func assertNoProfileEventRequest(t *testing.T, p *profileEventProvider) {
	t.Helper()
	select {
	case <-p.requests:
		t.Fatal("unexpected inference")
	case <-time.After(30 * time.Millisecond):
	}
}
func putProfileEvents(api *APIServer, id, directive string, tools []string, events []any) *httptest.ResponseRecorder {
	payload, _ := json.Marshal(map[string]any{"directive": directive, "tools": tools, "events": events})
	rec := httptest.NewRecorder()
	api.updateThread(rec, httptest.NewRequest(http.MethodPut, "/threads/"+id, bytes.NewReader(payload)), id)
	return rec
}
func profileEventPayload(id, text string) []any {
	return []any{map[string]any{"id": id, "message": text, "track_lifecycle": true}}
}
func newProfileEventFixture(t *testing.T) (*APIServer, *Thinker, *Thread, *profileEventProvider) {
	t.Helper()
	api, parent, _ := newPersistentThreadTestAPI(t)
	provider := &profileEventProvider{scriptedRetryProvider: scriptedRetryProvider{name: "profile-events"}, requests: make(chan profileEventRequest, 16)}
	useThreadEventProvider(parent, provider)
	parent.registry.Register(&ToolDef{Name: "new_profile_probe", Description: "Probe for the new profile", InputSchema: map[string]any{"type": "object"}, Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "ok"} }})
	rec := postThreadForTest(t, api, "profile-events", map[string]any{"directive": "OLD_PROFILE", "tools": []string{}})
	if rec.Code != 200 {
		t.Fatalf("spawn: %d %s", rec.Code, rec.Body.String())
	}
	_, thread := parent.threads.findManagedThread("profile-events")
	return api, parent, thread, provider
}
func requireProfileHTTP(t *testing.T, rec *httptest.ResponseRecorder, status int) {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("HTTP=%d want=%d body=%s", rec.Code, status, rec.Body.String())
	}
}

func TestThreadProfileUnchangedEventDoesNotCancelInference(t *testing.T) {
	api, _, thread, provider := newProfileEventFixture(t)
	first := takeProfileEventRequest(t, provider)
	unlockInbox := lockProfileEventInbox(thread)
	defer unlockInbox()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		done <- putProfileEvents(api, thread.ID, "OLD_PROFILE", []string{}, profileEventPayload("break", "BREAK_MARKER"))
	}()
	select {
	case <-first.ctx.Done():
		unlockInbox()
		t.Fatal("no-op profile canceled active inference")
	default:
	}
	assertNoProfileEventRequest(t, provider)
	unlockInbox()
	requireProfileHTTP(t, <-done, 200)
	select {
	case <-first.ctx.Done():
		t.Fatal("ordinary event canceled active inference")
	default:
	}
	first.finish <- ChatResponse{Text: "prior request finished"}
	next := takeProfileEventRequest(t, provider)
	if !strings.Contains(messagesText(next.messages), "BREAK_MARKER") {
		t.Fatal("next inference did not receive accepted event")
	}
}

func TestThreadProfileChangedEventReachesReplacementTogether(t *testing.T) {
	api, parent, thread, provider := newProfileEventFixture(t)
	first := takeProfileEventRequest(t, provider)
	unlockInbox := lockProfileEventInbox(thread)
	defer unlockInbox()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		done <- putProfileEvents(api, thread.ID, "NEW_PROFILE", []string{"new_profile_probe"}, profileEventPayload("break", "BREAK_MARKER"))
	}()
	// Event validation cannot be bypassed to start a premature replacement.
	assertNoProfileEventRequest(t, provider)
	select {
	case <-first.ctx.Done():
		unlockInbox()
		t.Fatal("canceled before event preflight")
	default:
	}
	unlockInbox()
	replacement := takeProfileEventRequest(t, provider)
	requireProfileHTTP(t, <-done, 200)
	select {
	case <-first.ctx.Done():
	default:
		t.Fatal("changed profile did not cancel obsolete request")
	}
	text := messagesText(replacement.messages)
	if !strings.Contains(text, "NEW_PROFILE") || !strings.Contains(text, "BREAK_MARKER") {
		t.Fatalf("replacement missing profile/event: %s", text)
	}
	found := false
	for _, tool := range replacement.tools {
		found = found || tool.Name == "new_profile_probe"
	}
	if !found {
		t.Fatal("replacement lacks new profile's tool")
	}
	stored, ok := persistentThreadByID(parent.config.GetThreads(), thread.ID)
	if !ok || stored.Directive != "NEW_PROFILE" || len(stored.Events) != 1 || stored.Events[0].ID != "break" {
		t.Fatalf("durable profile/events=%+v", stored)
	}
}

func TestThreadProfileIdleNoopWaitsForEvent(t *testing.T) {
	api, parent, thread, provider := newProfileEventFixture(t)
	first := takeProfileEventRequest(t, provider)
	first.finish <- ChatResponse{ToolCalls: []NativeToolCall{{ID: "wait", Name: "pace", Args: map[string]string{"clear_wake": "true"}}}}
	assertNoProfileEventRequest(t, provider)
	if !thread.Thinker.status().WaitForEvents {
		t.Fatal("fixture did not enter event-only wait")
	}
	result, err := parent.threads.UpdateWithOpts(thread.ID, "", "OLD_PROFILE", []string{}, ThreadUpdateOptions{ReplaceTools: true})
	if err != nil || result.Changed {
		t.Fatalf("no-op result=%+v err=%v", result, err)
	}
	assertNoProfileEventRequest(t, provider)
	unlockInbox := lockProfileEventInbox(thread)
	defer unlockInbox()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		done <- putProfileEvents(api, thread.ID, "OLD_PROFILE", []string{}, profileEventPayload("idle", "IDLE_EVENT"))
	}()
	assertNoProfileEventRequest(t, provider)
	unlockInbox()
	request := takeProfileEventRequest(t, provider)
	requireProfileHTTP(t, <-done, 200)
	if !strings.Contains(messagesText(request.messages), "IDLE_EVENT") {
		t.Fatal("idle thread started empty inference")
	}
}

func TestThreadProfileConcurrentDuplicateRetries(t *testing.T) {
	api, _, thread, provider := newProfileEventFixture(t)
	takeProfileEventRequest(t, provider)
	done := make(chan *httptest.ResponseRecorder, 2)
	for i := 0; i < 2; i++ {
		go func() {
			done <- putProfileEvents(api, thread.ID, "NEW_PROFILE", []string{}, profileEventPayload("same", "ONCE_EVENT"))
		}()
	}
	replacement := takeProfileEventRequest(t, provider)
	accepted, duplicates := 0, 0
	executionID := ""
	for i := 0; i < 2; i++ {
		rec := <-done
		requireProfileHTTP(t, rec, 200)
		response := decodeThreadEventResponse(t, rec.Body.Bytes())
		accepted += len(responseEventIDs(t, response, "accepted"))
		duplicates += len(responseEventIDs(t, response, "duplicates"))
		id := response["events"].(map[string]any)["executions"].(map[string]any)["same"].(string)
		if executionID != "" && id != executionID {
			t.Fatal("retry changed execution id")
		}
		executionID = id
	}
	if accepted != 1 || duplicates != 1 {
		t.Fatalf("accepted=%d duplicates=%d", accepted, duplicates)
	}
	select {
	case <-replacement.ctx.Done():
		t.Fatal("duplicate reconciliation canceled replacement")
	default:
	}
	if strings.Count(messagesText(replacement.messages), "ONCE_EVENT") != 1 {
		t.Fatal("duplicate event delivery")
	}
	assertNoProfileEventRequest(t, provider)
}

func TestThreadProfileEventsPersistenceFailureIsAtomic(t *testing.T) {
	api, parent, thread, provider := newProfileEventFixture(t)
	takeProfileEventRequest(t, provider)
	originalPath := parent.config.path
	blocker := filepath.Join(t.TempDir(), "directory")
	if err := os.Mkdir(blocker, 0700); err != nil {
		t.Fatal(err)
	}
	parent.config.path = blocker
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		done <- putProfileEvents(api, thread.ID, "NEW_PROFILE", []string{}, profileEventPayload("retryable", "RETRYABLE_EVENT"))
	}()
	rec := <-done
	requireProfileHTTP(t, rec, 500)
	stored, _ := persistentThreadByID(parent.config.GetThreads(), thread.ID)
	if thread.Directive != "OLD_PROFILE" || stored.Directive != "OLD_PROFILE" || len(stored.Events) != 0 || len(thread.inboxEvents) != 0 {
		t.Fatalf("failed commit changed profile or events: %+v", stored)
	}
	if len(parent.config.EventExecutions) != 0 {
		t.Fatal("failed commit registered execution")
	}
	parent.config.path = originalPath
	go func() {
		done <- putProfileEvents(api, thread.ID, "NEW_PROFILE", []string{}, profileEventPayload("retryable", "RETRYABLE_EVENT"))
	}()
	rec = <-done
	requireProfileHTTP(t, rec, 200)
	response := decodeThreadEventResponse(t, rec.Body.Bytes())
	if len(responseEventIDs(t, response, "accepted")) != 1 {
		t.Fatal("failed delivery was not retryable")
	}
	// A failed request may already have been replaced; consume until the
	// successful reconciliation's replacement arrives.
	for {
		request := takeProfileEventRequest(t, provider)
		if strings.Contains(messagesText(request.messages), "NEW_PROFILE") {
			if !strings.Contains(messagesText(request.messages), "RETRYABLE_EVENT") {
				t.Fatal("replacement lacks retry event")
			}
			break
		}
	}

	reloaded := &Config{path: originalPath}
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	persisted, _ := persistentThreadByID(reloaded.GetThreads(), thread.ID)
	if persisted.Directive != "NEW_PROFILE" || len(persisted.Events) != 1 {
		t.Fatalf("restart lost atomic commit: %+v", persisted)
	}
}

func TestThreadProfileConflictingEventDoesNotChangeOrCancel(t *testing.T) {
	api, _, thread, provider := newProfileEventFixture(t)
	first := takeProfileEventRequest(t, provider)
	rec := putProfileEvents(api, thread.ID, "OLD_PROFILE", []string{}, profileEventPayload("same", "original"))
	requireProfileHTTP(t, rec, 200)
	rec = putProfileEvents(api, thread.ID, "NEW_PROFILE", []string{}, profileEventPayload("same", "different"))
	requireProfileHTTP(t, rec, 409)
	if thread.Directive != "OLD_PROFILE" {
		t.Fatal("conflicting event changed profile")
	}
	select {
	case <-first.ctx.Done():
		t.Fatal("conflicting event canceled inference")
	default:
	}
	assertNoProfileEventRequest(t, provider)
}

func lockProfileEventInbox(thread *Thread) func() {
	thread.inboxMu.Lock()
	var once sync.Once
	return func() { once.Do(thread.inboxMu.Unlock) }
}
