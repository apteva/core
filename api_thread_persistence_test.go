package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"
)

type parkedAPIProvider struct {
	started     chan struct{}
	release     chan struct{}
	once        sync.Once
	releaseOnce sync.Once
}

func newParkedAPIProvider() *parkedAPIProvider {
	return &parkedAPIProvider{started: make(chan struct{}), release: make(chan struct{})}
}

func (p *parkedAPIProvider) Chat(ctx context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.once.Do(func() { close(p.started) })
	select {
	case <-p.release:
	case <-ctx.Done():
		return ChatResponse{}, ctx.Err()
	}
	return ChatResponse{Text: "Waiting for work."}, nil
}

func (p *parkedAPIProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "api-large", ModelMedium: "api-medium", ModelSmall: "api-small"}
}
func (p *parkedAPIProvider) Name() string                           { return "api-test-provider" }
func (p *parkedAPIProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *parkedAPIProvider) SupportsNativeTools() bool              { return false }
func (p *parkedAPIProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *parkedAPIProvider) SetBuiltinTools([]string)               {}
func (p *parkedAPIProvider) WithBuiltins([]string) LLMProvider      { return p }
func (p *parkedAPIProvider) Release() {
	p.releaseOnce.Do(func() { close(p.release) })
}

func waitForParkedAPIProvider(t *testing.T, provider *parkedAPIProvider) {
	t.Helper()
	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("thread did not reach the parked provider")
	}
}

func newPersistentThreadTestAPI(t *testing.T) (*APIServer, *Thinker, *parkedAPIProvider) {
	t.Helper()
	t.Chdir(t.TempDir())
	api, thinker := newTestAPI()
	provider := newParkedAPIProvider()
	thinker.provider = provider
	thinker.pool = &ProviderPool{
		providers: map[string]LLMProvider{provider.Name(): provider},
		order:     []string{provider.Name()},
		default_:  provider.Name(),
	}
	thinker.registry = NewToolRegistry("")
	thinker.toolIndex = NewToolIndex()
	thinker.activeTools = map[string]bool{}
	thinker.config.path = configFile
	thinker.model = ModelLarge
	thinker.agentModel = ModelLarge
	thinker.agentReasoning = ReasoningAuto
	thinker.publishRuntimeStatus()
	t.Cleanup(func() {
		provider.Release()
		thinker.threads.KillAll()
		thinker.Stop()
	})
	return api, thinker, provider
}

func postThreadForTest(t *testing.T, api *APIServer, id string, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/threads/"+id, bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.spawnThread(w, req, id)
	return w
}

func persistentThreadByID(threads []PersistentThread, id string) (PersistentThread, bool) {
	for _, thread := range threads {
		if thread.ID == id {
			return thread, true
		}
	}
	return PersistentThread{}, false
}

func TestAPICreatedThreadPersistsEffectiveStateAndRestores(t *testing.T) {
	api, thinker, provider := newPersistentThreadTestAPI(t)
	w := postThreadForTest(t, api, "chat-persisted", map[string]any{
		"directive": "Handle this durable CRM conversation.",
		"tools":     []string{"crm_lookup_contact"},
		"mcp":       []string{"crm"},
		"model":     "medium",
		"reasoning": "low",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status = %d, want 200: %s", w.Code, w.Body.String())
	}

	stored, ok := persistentThreadByID(thinker.config.GetThreads(), "chat-persisted")
	if !ok {
		t.Fatalf("API-created thread missing from config: %#v", thinker.config.GetThreads())
	}
	if stored.Directive != "Handle this durable CRM conversation." || stored.ParentID != "main" || stored.Depth != 0 {
		t.Fatalf("stored identity/directive mismatch: %#v", stored)
	}
	if stored.Provider != provider.Name() || stored.Model != "medium" || stored.Reasoning != "low" {
		t.Fatalf("stored effective provider profile mismatch: %#v", stored)
	}
	if !slices.Equal(stored.MCPNames, []string{"crm"}) {
		t.Fatalf("stored MCP state mismatch: %#v", stored)
	}
	for _, want := range []string{"crm_lookup_contact", "send", "done", "pace", "evolve", "search_tools"} {
		if !slices.Contains(stored.Tools, want) {
			t.Fatalf("stored effective tools missing %q: %v", want, stored.Tools)
		}
	}

	// Load a fresh Config from disk and construct a new thinker, matching a
	// process restart rather than merely reusing the in-memory config object.
	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	restarted := NewThinker("", provider, reloaded)
	t.Cleanup(func() {
		provider.Release()
		restarted.threads.KillAll()
		restarted.Stop()
	})
	if findThinkerByID(restarted, "chat-persisted") == nil {
		t.Fatal("persisted API thread was not restored after restart")
	}
	restored, err := restarted.threads.PersistentState("chat-persisted")
	if err != nil {
		t.Fatalf("restored state: %v", err)
	}
	if restored.Directive != stored.Directive || restored.Provider != stored.Provider || restored.Model != stored.Model || restored.Reasoning != stored.Reasoning {
		t.Fatalf("restored effective state changed:\n stored=%#v\nrestored=%#v", stored, restored)
	}
	if !slices.Equal(restored.MCPNames, stored.MCPNames) || !slices.Equal(restored.Tools, stored.Tools) {
		t.Fatalf("restored tool scope changed:\n stored=%#v\nrestored=%#v", stored, restored)
	}
}

func TestAPIRealtimeTurnDetectionPersistsAndRestoresIntoSession(t *testing.T) {
	api, thinker, textProvider := newPersistentThreadTestAPI(t)
	realtimeProvider := &fakeRealtimeProvider{}
	thinker.config.RealtimeEnabled = true
	thinker.pool.realtimeProviders = map[string]RealtimeProvider{realtimeProvider.Name(): realtimeProvider}
	thinker.pool.realtimeOrder = []string{realtimeProvider.Name()}
	thinker.pool.realtimeDefault = realtimeProvider.Name()

	w := postThreadForTest(t, api, "voice-persisted", map[string]any{
		"directive": "Handle telephone calls safely.",
		"realtime":  true,
		"provider":  realtimeProvider.Name(),
		"turn_detection": map[string]any{
			"profile": "telephony",
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status = %d, want 200: %s", w.Code, w.Body.String())
	}

	stored, ok := persistentThreadByID(thinker.config.GetThreads(), "voice-persisted")
	if !ok || stored.TurnDetection == nil {
		t.Fatalf("realtime turn detection was not persisted: %#v", stored)
	}
	if stored.TurnDetection.Profile != RealtimeTurnProfileTelephony {
		t.Fatalf("stored turn detection = %#v", stored.TurnDetection)
	}
	realtimeProvider.mu.Lock()
	if len(realtimeProvider.opens) != 1 ||
		realtimeProvider.opens[0].TurnDetection.Profile != RealtimeTurnProfileTelephony {
		t.Fatalf("live session did not receive profile: %#v", realtimeProvider.opens)
	}
	realtimeProvider.mu.Unlock()

	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	reloadedState, ok := persistentThreadByID(reloaded.GetThreads(), "voice-persisted")
	if !ok || reloadedState.TurnDetection == nil ||
		reloadedState.TurnDetection.Profile != RealtimeTurnProfileTelephony {
		t.Fatalf("turn detection did not survive config reload: %#v", reloadedState)
	}

	restoredParent := newTestThinker()
	defer restoredParent.Stop()
	restoredParent.config = reloaded
	restoredRealtimeProvider := &fakeRealtimeProvider{}
	restoredParent.pool = &ProviderPool{
		providers: map[string]LLMProvider{textProvider.Name(): textProvider},
		order:     []string{textProvider.Name()}, default_: textProvider.Name(),
		realtimeProviders: map[string]RealtimeProvider{
			restoredRealtimeProvider.Name(): restoredRealtimeProvider,
		},
		realtimeOrder:   []string{restoredRealtimeProvider.Name()},
		realtimeDefault: restoredRealtimeProvider.Name(),
	}
	if err := restoredParent.threads.SpawnWithOpts(
		reloadedState.ID, reloadedState.Directive, reloadedState.Tools,
		SpawnOpts{
			Realtime: true, DeferRun: true,
			ProviderName: restoredRealtimeProvider.Name(), Voice: reloadedState.Voice,
			TurnDetection: realtimeTurnDetectionValue(reloadedState.TurnDetection),
		},
	); err != nil {
		t.Fatalf("restore realtime thread: %v", err)
	}
	restoredThread := restoredParent.threads.threads[reloadedState.ID]
	if restoredThread == nil || restoredThread.Realtime == nil ||
		restoredThread.Realtime.opts.TurnDetection.Profile != RealtimeTurnProfileTelephony {
		t.Fatalf("restored runtime lost turn detection: %#v", restoredThread)
	}
}

func TestAPIRealtimeTurnDetectionRejectsInvalidOrNonRealtimeUse(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	realtimeProvider := &fakeRealtimeProvider{}
	thinker.config.RealtimeEnabled = true
	thinker.pool.realtimeProviders = map[string]RealtimeProvider{realtimeProvider.Name(): realtimeProvider}
	thinker.pool.realtimeOrder = []string{realtimeProvider.Name()}
	thinker.pool.realtimeDefault = realtimeProvider.Name()

	invalid := postThreadForTest(t, api, "voice-invalid", map[string]any{
		"realtime":       true,
		"provider":       realtimeProvider.Name(),
		"turn_detection": map[string]any{"profile": "concert"},
	})
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid profile status = %d, want 400: %s", invalid.Code, invalid.Body.String())
	}
	if findThinkerByID(thinker, "voice-invalid") != nil {
		t.Fatal("invalid profile created a live thread")
	}

	nonRealtime := postThreadForTest(t, api, "text-invalid", map[string]any{
		"turn_detection": map[string]any{"profile": "telephony"},
	})
	if nonRealtime.Code != http.StatusBadRequest {
		t.Fatalf("non-realtime profile status = %d, want 400: %s", nonRealtime.Code, nonRealtime.Body.String())
	}
}

func TestAPIDefaultRealtimeFieldsCreateAndPersistOrdinaryThreadWithoutRealtimeProvider(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	if thinker.pool.HasRealtimeProvider() {
		t.Fatal("test requires no realtime provider")
	}

	w := postThreadForTest(t, api, "ordinary-defaults", map[string]any{
		"directive": "Process this ordinary background job.",
		"realtime":  false,
		"turn_detection": map[string]any{
			"profile":           "default",
			"start_sensitivity": "default",
			"end_sensitivity":   "default",
			"interruption":      "default",
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status = %d, want 200: %s", w.Code, w.Body.String())
	}

	thread := thinker.threads.threads["ordinary-defaults"]
	if thread == nil {
		t.Fatal("ordinary API thread was not created")
	}
	if thread.IsRealtime || thread.Realtime != nil ||
		thread.TurnDetection != (RealtimeTurnDetectionConfig{}) {
		t.Fatalf("ordinary API thread retained realtime state: %#v", thread)
	}
	stored, ok := persistentThreadByID(thinker.config.GetThreads(), "ordinary-defaults")
	if !ok {
		t.Fatal("ordinary API thread was not persisted")
	}
	if stored.Realtime || stored.TurnDetection != nil {
		t.Fatalf("ordinary API thread persisted realtime state: %#v", stored)
	}

	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	reloadedState, ok := persistentThreadByID(reloaded.GetThreads(), "ordinary-defaults")
	if !ok || reloadedState.Realtime || reloadedState.TurnDetection != nil {
		t.Fatalf("reloaded ordinary thread gained realtime state: %#v", reloadedState)
	}
}

func TestAPIEphemeralThreadNeverEntersPersistentConfig(t *testing.T) {
	api, thinker, provider := newPersistentThreadTestAPI(t)
	w := postThreadForTest(t, api, "chat-ephemeral", map[string]any{
		"directive": "Temporary event-driven thread.",
		"ephemeral": true,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("spawn status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("ephemeral spawn was persisted: %#v", threads)
	}
	waitForParkedAPIProvider(t, provider)

	update, _ := json.Marshal(map[string]any{
		"directive": "Updated temporary event-driven thread.",
	})
	req := httptest.NewRequest(http.MethodPut, "/threads/chat-ephemeral", bytes.NewReader(update))
	updated := httptest.NewRecorder()
	api.updateThread(updated, req, "chat-ephemeral")
	if updated.Code != http.StatusOK {
		t.Fatalf("update status = %d, want 200: %s", updated.Code, updated.Body.String())
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("ephemeral update leaked into config: %#v", threads)
	}
	threadThinker := findThinkerByID(thinker, "chat-ephemeral")
	if threadThinker == nil {
		t.Fatal("ephemeral thread disappeared before evolve check")
	}
	_, _, results := threadThinker.handleTools(threadThinker, []toolCall{{
		Name:     "evolve",
		Args:     map[string]string{"directive": "Evolved temporary conversation."},
		Raw:      "evolve",
		NativeID: "ephemeral-evolve",
	}}, nil)
	if len(results) != 1 || results[0].IsError {
		t.Fatalf("ephemeral evolve failed: %#v", results)
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("ephemeral evolve leaked into config: %#v", threads)
	}

	w = postThreadForTest(t, api, "chat-ephemeral", map[string]any{
		"ephemeral": true,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("idempotent repost status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("ephemeral repost leaked into config: %#v", threads)
	}

	if err := thinker.threads.Rename("chat-ephemeral", "chat-ephemeral-renamed"); err != nil {
		t.Fatalf("rename ephemeral thread: %v", err)
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("ephemeral rename leaked into config: %#v", threads)
	}
}

func TestAPIExistingUnpersistedThreadIsBackfilled(t *testing.T) {
	api, thinker, provider := newPersistentThreadTestAPI(t)
	if err := thinker.threads.SpawnWithOpts("legacy-live", "Legacy live conversation.", []string{"crm_lookup_contact"}, SpawnOpts{}); err != nil {
		t.Fatalf("spawn legacy fixture: %v", err)
	}
	if len(thinker.config.GetThreads()) != 0 {
		t.Fatal("SpawnWithOpts unexpectedly persisted the legacy fixture")
	}
	waitForParkedAPIProvider(t, provider)

	// Older server versions may still send the removed field during a rolling
	// upgrade. It is an inert unknown field and must not block normal backfill.
	w := postThreadForTest(t, api, "legacy-live", map[string]any{"conversation": true})
	if w.Code != http.StatusOK {
		t.Fatalf("backfill status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response["status"] != "exists" {
		t.Fatalf("status = %v, want exists", response["status"])
	}
	_, ok := persistentThreadByID(thinker.config.GetThreads(), "legacy-live")
	if !ok {
		t.Fatalf("existing live thread was not backfilled: %#v", thinker.config.GetThreads())
	}
}

func TestAPIPersistentRepostPromotesEphemeralLiveThread(t *testing.T) {
	api, thinker, provider := newPersistentThreadTestAPI(t)
	if err := thinker.threads.SpawnWithOpts("promote-live", "Temporary until claimed.", nil, SpawnOpts{Ephemeral: true}); err != nil {
		t.Fatalf("spawn ephemeral fixture: %v", err)
	}
	waitForParkedAPIProvider(t, provider)
	w := postThreadForTest(t, api, "promote-live", map[string]any{})
	if w.Code != http.StatusOK {
		t.Fatalf("promotion status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := persistentThreadByID(thinker.config.GetThreads(), "promote-live"); !ok {
		t.Fatal("persistent repost did not backfill the promoted thread")
	}
	if ephemeral, err := thinker.threads.EphemeralState("promote-live"); err != nil || ephemeral {
		t.Fatalf("live thread was not promoted: ephemeral=%v err=%v", ephemeral, err)
	}
}

func TestAPISpawnPersistenceFailureRollsBackLiveThreadAndRealtimeBridge(t *testing.T) {
	api, thinker, provider := newPersistentThreadTestAPI(t)
	blocker := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	thinker.config.path = filepath.Join(blocker, "config.json")
	thinker.config.RealtimeEnabled = true
	realtimeSession := newFakeRealtimeSession()
	realtimeProvider := &fakeRealtimeProvider{sessions: []*fakeRealtimeSession{realtimeSession}}
	thinker.pool = &ProviderPool{
		providers:         map[string]LLMProvider{provider.Name(): provider},
		order:             []string{provider.Name()},
		default_:          provider.Name(),
		realtimeProviders: map[string]RealtimeProvider{realtimeProvider.Name(): realtimeProvider},
		realtimeOrder:     []string{realtimeProvider.Name()},
		realtimeDefault:   realtimeProvider.Name(),
	}

	w := postThreadForTest(t, api, "voice-persist-failure", map[string]any{
		"directive": "Answer the caller.",
		"realtime":  true,
		"provider":  realtimeProvider.Name(),
	})
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("spawn status = %d, want 500: %s", w.Code, w.Body.String())
	}
	if findThinkerByID(thinker, "voice-persist-failure") != nil {
		t.Fatal("live thread survived failed persistence")
	}
	if threads := thinker.config.GetThreads(); len(threads) != 0 {
		t.Fatalf("failed persistence changed config: %#v", threads)
	}
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	for _, registration := range audioBridgeByTok {
		if registration.threadID == "voice-persist-failure" {
			t.Fatal("audio bridge token survived failed persistence")
		}
	}
	if audioBridgeLive["voice-persist-failure"] != nil {
		t.Fatal("live audio bridge survived failed persistence")
	}
}

func TestAPILazyEventSpawnIsPersistedAndDeleteRemovesRecord(t *testing.T) {
	api, thinker, _ := newPersistentThreadTestAPI(t)
	payload, _ := json.Marshal(map[string]any{
		"thread_id": "event-created",
		"message":   "New inbound CRM message.",
	})
	req := httptest.NewRequest(http.MethodPost, "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("event status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := persistentThreadByID(thinker.config.GetThreads(), "event-created"); !ok {
		t.Fatal("lazy /event spawn was not persisted")
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/threads/event-created", nil)
	deleted := httptest.NewRecorder()
	api.threadAction(deleted, deleteReq)
	if deleted.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200: %s", deleted.Code, deleted.Body.String())
	}
	if _, ok := persistentThreadByID(thinker.config.GetThreads(), "event-created"); ok {
		t.Fatal("delete left the persistent record behind")
	}
}
