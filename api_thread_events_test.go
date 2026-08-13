package core

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type recordingThreadEventProvider struct {
	requests chan []Message
	calls    atomic.Int32
	onChat   func()
}

func newRecordingThreadEventProvider() *recordingThreadEventProvider {
	return &recordingThreadEventProvider{requests: make(chan []Message, 32)}
}

func (p *recordingThreadEventProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.calls.Add(1)
	if p.onChat != nil {
		p.onChat()
	}
	p.requests <- cloneMessages(messages)
	return ChatResponse{Text: "event handled"}, nil
}

func (p *recordingThreadEventProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "event-large", ModelMedium: "event-medium", ModelSmall: "event-small"}
}
func (p *recordingThreadEventProvider) Name() string                           { return "event-provider" }
func (p *recordingThreadEventProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *recordingThreadEventProvider) SupportsNativeTools() bool              { return true }
func (p *recordingThreadEventProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *recordingThreadEventProvider) SetBuiltinTools([]string)               {}
func (p *recordingThreadEventProvider) WithBuiltins([]string) LLMProvider      { return p }

func useThreadEventProvider(thinker *Thinker, provider LLMProvider) {
	thinker.provider = provider
	thinker.pool = &ProviderPool{
		providers: map[string]LLMProvider{provider.Name(): provider},
		order:     []string{provider.Name()}, default_: provider.Name(),
	}
	thinker.model = ModelLarge
	thinker.agentModel = ModelLarge
	thinker.agentReasoning = ReasoningAuto
	thinker.publishRuntimeStatus()
}

func waitThreadEventRequest(t *testing.T, provider *recordingThreadEventProvider) []Message {
	t.Helper()
	select {
	case messages := <-provider.requests:
		return messages
	case <-time.After(3 * time.Second):
		t.Fatal("thread event never reached provider")
		return nil
	}
}

func decodeThreadEventResponse(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var response map[string]any
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("decode response: %v: %s", err, body)
	}
	return response
}

func responseEventIDs(t *testing.T, response map[string]any, field string) []any {
	t.Helper()
	events, ok := response["events"].(map[string]any)
	if !ok {
		t.Fatalf("response events missing: %#v", response)
	}
	ids, ok := events[field].([]any)
	if !ok {
		t.Fatalf("response events.%s missing: %#v", field, events)
	}
	return ids
}

func messageContaining(messages []Message, needle string) (Message, bool) {
	for _, message := range messages {
		if strings.Contains(message.Content, needle) {
			return message, true
		}
	}
	return Message{}, false
}

func TestAPISpawnThreadEventsFirstTurnMultimodalAndIdempotent(t *testing.T) {
	api, thinker, parked := newPersistentThreadTestAPI(t)
	parked.Release()
	provider := newRecordingThreadEventProvider()
	useThreadEventProvider(thinker, provider)
	var startedAfterPersistence atomic.Bool
	provider.onChat = func() {
		stored, ok := persistentThreadByID(thinker.config.GetThreads(), "crm-chat-events")
		if ok && len(stored.Events) == 1 && stored.Events[0].Consumed {
			startedAfterPersistence.Store(true)
		}
	}

	w := postThreadForTest(t, api, "crm-chat-events", map[string]any{
		"directive": "Handle CRM messages.",
		"events": []any{
			map[string]any{
				"id": "chat-message:2244:agent:286",
				"message": []any{
					map[string]any{"type": "text", "text": "Hi from CRM"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/contact.png"}},
				},
			},
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status=%d body=%s", w.Code, w.Body.String())
	}
	response := decodeThreadEventResponse(t, w.Body.Bytes())
	if accepted := responseEventIDs(t, response, "accepted"); len(accepted) != 1 || accepted[0] != "chat-message:2244:agent:286" {
		t.Fatalf("accepted=%v", accepted)
	}

	messages := waitThreadEventRequest(t, provider)
	if !startedAfterPersistence.Load() {
		t.Fatal("thread reached the provider before its specification and event were durable")
	}
	eventMessage, found := messageContaining(messages, "Hi from CRM")
	if !found || eventMessage.Role != "user" {
		t.Fatalf("first request lost event text: %#v", messages)
	}
	if len(eventMessage.Parts) != 3 || eventMessage.Parts[2].Type != "image_url" || eventMessage.Parts[2].ImageURL == nil || eventMessage.Parts[2].ImageURL.URL != "https://example.com/contact.png" {
		t.Fatalf("first request lost event media: %#v", eventMessage.Parts)
	}
	stored, ok := persistentThreadByID(thinker.config.GetThreads(), "crm-chat-events")
	if !ok || len(stored.Events) != 1 || !stored.Events[0].Consumed || stored.Events[0].Text != "" || len(stored.Events[0].Parts) != 0 {
		t.Fatalf("consumed event ledger=%#v", stored.Events)
	}

	duplicate := postThreadForTest(t, api, "crm-chat-events", map[string]any{
		"events": []any{map[string]any{
			"id": "chat-message:2244:agent:286",
			"message": []any{
				map[string]any{"type": "text", "text": "Hi from CRM"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/contact.png"}},
			},
		}},
	})
	if duplicate.Code != http.StatusOK {
		t.Fatalf("duplicate status=%d body=%s", duplicate.Code, duplicate.Body.String())
	}
	dupResponse := decodeThreadEventResponse(t, duplicate.Body.Bytes())
	if ids := responseEventIDs(t, dupResponse, "duplicates"); len(ids) != 1 || ids[0] != "chat-message:2244:agent:286" {
		t.Fatalf("duplicates=%v", ids)
	}
	select {
	case repeated := <-provider.requests:
		t.Fatalf("duplicate event woke a second model turn: %#v", repeated)
	case <-time.After(250 * time.Millisecond):
	}

	conflict := postThreadForTest(t, api, "crm-chat-events", map[string]any{
		"events": []any{map[string]any{"id": "chat-message:2244:agent:286", "message": "different payload"}},
	})
	if conflict.Code != http.StatusConflict {
		t.Fatalf("conflict status=%d want 409 body=%s", conflict.Code, conflict.Body.String())
	}
}

func TestAPIThreadEventsPersistenceFailureDoesNotStartOrDeliver(t *testing.T) {
	api, thinker, parked := newPersistentThreadTestAPI(t)
	parked.Release()
	provider := newRecordingThreadEventProvider()
	useThreadEventProvider(thinker, provider)
	blocker := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	thinker.config.path = filepath.Join(blocker, "config.json")

	w := postThreadForTest(t, api, "events-persist-failure", map[string]any{
		"events": []any{map[string]any{"id": "never-delivered", "message": "must not run"}},
	})
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status=%d want 500 body=%s", w.Code, w.Body.String())
	}
	if findThinkerByID(thinker, "events-persist-failure") != nil {
		t.Fatal("failed event transaction left a live thread")
	}
	if provider.calls.Load() != 0 {
		t.Fatalf("provider started despite persistence failure: calls=%d", provider.calls.Load())
	}
	select {
	case request := <-provider.requests:
		t.Fatalf("event delivered despite persistence failure: %#v", request)
	default:
	}
}

func TestAPIExistingThreadAcceptsOnlyUnseenEventsInOrder(t *testing.T) {
	api, thinker, parked := newPersistentThreadTestAPI(t)
	parked.Release()
	provider := newRecordingThreadEventProvider()
	useThreadEventProvider(thinker, provider)

	w := postThreadForTest(t, api, "ordered-events", map[string]any{
		"events": []any{
			map[string]any{"id": "event-1", "message": "first message"},
			map[string]any{"id": "event-2", "message": "second message"},
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status=%d body=%s", w.Code, w.Body.String())
	}
	firstRequest := waitThreadEventRequest(t, provider)
	orderedMessage, found := messageContaining(firstRequest, "first message")
	if !found {
		t.Fatalf("first request lost ordered events: %#v", firstRequest)
	}
	content := orderedMessage.Content
	if first, second := strings.Index(content, "first message"), strings.Index(content, "second message"); first < 0 || second <= first {
		t.Fatalf("events lost request order: %q", content)
	}

	w = postThreadForTest(t, api, "ordered-events", map[string]any{
		"events": []any{
			map[string]any{"id": "event-1", "message": "first message"},
			map[string]any{"id": "event-3", "message": "third message"},
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("existing status=%d body=%s", w.Code, w.Body.String())
	}
	response := decodeThreadEventResponse(t, w.Body.Bytes())
	if len(responseEventIDs(t, response, "accepted")) != 1 || len(responseEventIDs(t, response, "duplicates")) != 1 {
		t.Fatalf("existing event response=%#v", response)
	}
	secondRequest := waitThreadEventRequest(t, provider)
	thirdMessage, found := messageContaining(secondRequest, "third message")
	if !found {
		t.Fatalf("existing request lost unseen event: %#v", secondRequest)
	}
	last := thirdMessage.Content
	if !strings.Contains(last, "third message") || strings.Contains(last, "first message") {
		t.Fatalf("existing thread received wrong events: %q", last)
	}
}

func TestAPIThreadEventsRejectInternalFieldsAndInvalidContent(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	for _, tc := range []struct {
		name  string
		event map[string]any
	}{
		{"missing id", map[string]any{"message": "hello"}},
		{"internal type", map[string]any{"id": "one", "type": "tool_result", "message": "hello"}},
		{"internal sender", map[string]any{"id": "one", "from": "main", "message": "hello"}},
		{"unsupported part", map[string]any{"id": "one", "message": []any{map[string]any{"type": "tool_result", "text": "bad"}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := postThreadForTest(t, api, "invalid-"+strings.ReplaceAll(tc.name, " ", "-"), map[string]any{"events": []any{tc.event}})
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status=%d want 400 body=%s", w.Code, w.Body.String())
			}
		})
	}
	if len(thinker.config.GetThreads()) != 0 {
		t.Fatalf("invalid events created threads: %#v", thinker.config.GetThreads())
	}
}

func TestThreadEventsConcurrentRetryQueuesOnce(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts("concurrent-events", "Handle events.", nil, SpawnOpts{DeferRun: true, Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	defer thinker.threads.Kill("concurrent-events")
	event := PersistentThreadEvent{ID: "same-id", Text: "same payload"}
	event.Hash = threadEventHash(event.Text, nil)

	const callers = 12
	results := make(chan ThreadEventQueueResult, callers)
	errs := make(chan error, callers)
	for i := 0; i < callers; i++ {
		go func() {
			result, err := thinker.threads.QueueEvents("concurrent-events", []PersistentThreadEvent{event})
			results <- result
			errs <- err
		}()
	}
	accepted, duplicates := 0, 0
	for i := 0; i < callers; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
		result := <-results
		accepted += len(result.Accepted)
		duplicates += len(result.Duplicate)
	}
	if accepted != 1 || duplicates != callers-1 {
		t.Fatalf("accepted=%d duplicates=%d", accepted, duplicates)
	}
	thread := thinker.threads.threads["concurrent-events"]
	items := thread.Thinker.drainEvents()
	if len(items) != 1 || items[0].ID != "same-id" || items[0].Text != "same payload" {
		t.Fatalf("queued events=%#v", items)
	}
}

func TestPersistentPendingThreadEventRestoresAndRunsOnce(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := newRecordingThreadEventProvider()
	cfg := &Config{path: configFile, Directive: "Coordinate events.", Mode: ModeAutonomous}
	parent := NewThinker("", provider, cfg)
	if err := parent.threads.SpawnWithOpts("restart-event", "Handle the event.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	api := &APIServer{thinker: parent}
	if err := api.persistAPIThread("restart-event", false); err != nil {
		t.Fatal(err)
	}
	event := PersistentThreadEvent{ID: "restart-id", Text: "survive restart"}
	event.Hash = threadEventHash(event.Text, nil)
	if result, err := parent.threads.QueueEvents("restart-event", []PersistentThreadEvent{event}); err != nil || len(result.Accepted) != 1 {
		t.Fatalf("queue result=%#v err=%v", result, err)
	}

	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatal(err)
	}
	parent.threads.KillAll()
	parent.Stop()

	restartedProvider := newRecordingThreadEventProvider()
	restarted := NewThinker("", restartedProvider, reloaded)
	defer func() {
		restarted.threads.KillAll()
		restarted.Stop()
	}()
	messages := waitThreadEventRequest(t, restartedProvider)
	if _, found := messageContaining(messages, "survive restart"); !found {
		t.Fatalf("restored request lost event: %#v", messages)
	}
	result, err := restarted.threads.QueueEvents("restart-event", []PersistentThreadEvent{event})
	if err != nil || len(result.Accepted) != 0 || len(result.Duplicate) != 1 {
		t.Fatalf("restart retry result=%#v err=%v", result, err)
	}
	select {
	case repeated := <-restartedProvider.requests:
		t.Fatalf("restart retry redelivered event: %#v", repeated)
	case <-time.After(250 * time.Millisecond):
	}
}
