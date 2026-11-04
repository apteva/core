package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// pinTestProvider parks every request until it is cancelled, so a worker stays
// mid-request without ever reaching a real LLM.
type pinTestProvider struct {
	name      string
	native    bool
	scope     []string
	root      *pinTestProvider
	started   chan struct{}
	calls     atomic.Int32
	cancelled atomic.Int32
}

func newPinTestProvider(name string) *pinTestProvider {
	return &pinTestProvider{name: name, started: make(chan struct{}, 64)}
}

func (p *pinTestProvider) origin() *pinTestProvider {
	if p.root != nil {
		return p.root
	}
	return p
}

func (p *pinTestProvider) Chat(ctx context.Context, _ []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	o := p.origin()
	o.calls.Add(1)
	select {
	case o.started <- struct{}{}:
	default:
	}
	<-ctx.Done()
	o.cancelled.Add(1)
	return ChatResponse{}, ctx.Err()
}

func (p *pinTestProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: p.name + "-large", ModelMedium: p.name + "-medium", ModelSmall: p.name + "-small"}
}
func (p *pinTestProvider) Name() string                           { return p.name }
func (p *pinTestProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *pinTestProvider) SupportsNativeTools() bool              { return p.native }
func (p *pinTestProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *pinTestProvider) SetBuiltinTools([]string)               {}
func (p *pinTestProvider) WithBuiltins(builtins []string) LLMProvider {
	return &pinTestProvider{name: p.name, native: p.native, scope: append([]string{}, builtins...), root: p.origin()}
}

func pinTestPool(defaultName string, providers ...*pinTestProvider) *ProviderPool {
	pool := &ProviderPool{providers: map[string]LLMProvider{}, default_: defaultName}
	for _, p := range providers {
		pool.providers[p.name] = p
		pool.order = append(pool.order, p.name)
	}
	return pool
}

func withFakeRealtime(pool *ProviderPool, session *fakeRealtimeProvider) *ProviderPool {
	pool.realtimeProviders = map[string]RealtimeProvider{session.Name(): session}
	pool.realtimeOrder = []string{session.Name()}
	pool.realtimeDefault = session.Name()
	return pool
}

// The first provider is main's.
func newPinTestAPI(t *testing.T, providers ...*pinTestProvider) (*APIServer, *Thinker) {
	t.Helper()
	api, thinker, _ := newPersistentThreadTestAPI(t)
	thinker.provider = providers[0]
	thinker.pool = pinTestPool(providers[0].name, providers...)
	thinker.publishRuntimeStatus()
	return api, thinker
}

// spawnPinWorker creates a saved worker whose loop hasn't started.
func spawnPinWorker(t *testing.T, tm *ThreadManager, id string, tools []string, opts SpawnOpts) *Thread {
	t.Helper()
	opts.DeferRun = true
	if err := tm.SpawnWithOpts(id, "Wait for instructions.", tools, opts); err != nil {
		t.Fatalf("spawn %s: %v", id, err)
	}
	state, err := tm.PersistentState(id)
	if err != nil {
		t.Fatal(err)
	}
	if err := tm.parent.config.SaveThread(state); err != nil {
		t.Fatal(err)
	}
	return tm.threads[id]
}

func pinWaitFor(t *testing.T, what string, done func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !done() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func pinWaitStarted(t *testing.T, p *pinTestProvider) {
	t.Helper()
	select {
	case <-p.started:
	case <-time.After(3 * time.Second):
		t.Fatalf("provider %s received no request", p.name)
	}
}

func assertRunsOn(t *testing.T, thread *Thread, want *pinTestProvider) {
	t.Helper()
	got, ok := thread.Thinker.provider.(*pinTestProvider)
	if !ok || got.origin() != want {
		t.Fatalf("%s runs on %#v, want %s", thread.ID, thread.Thinker.provider, want.name)
	}
}

func savedThread(t *testing.T, thinker *Thinker, id string) PersistentThread {
	t.Helper()
	state, ok := persistentThreadByID(thinker.config.GetThreads(), id)
	if !ok {
		t.Fatalf("thread %s is not saved", id)
	}
	return state
}

func savedPin(t *testing.T, state PersistentThread) string {
	t.Helper()
	if state.ProviderPin == nil {
		t.Fatalf("thread %s saved without a pin field", state.ID)
	}
	return *state.ProviderPin
}

func putProviderConfig(t *testing.T, api *APIServer, payload string) {
	t.Helper()
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest(http.MethodPut, "/config", bytes.NewBufferString(payload)))
	if w.Code != http.StatusOK {
		t.Fatalf("PUT /config status=%d: %s", w.Code, w.Body.String())
	}
}

type threadProviderRow struct {
	ID              string `json:"id"`
	Provider        string `json:"provider"`
	ProviderPin     string `json:"provider_pin"`
	ProviderMissing bool   `json:"provider_missing"`
}

func threadProviderRowFor(t *testing.T, api *APIServer, id string) threadProviderRow {
	t.Helper()
	w := httptest.NewRecorder()
	api.threads(w, httptest.NewRequest(http.MethodGet, "/threads", nil))
	var rows []threadProviderRow
	if err := json.Unmarshal(w.Body.Bytes(), &rows); err != nil {
		t.Fatalf("decode /threads: %v: %s", err, w.Body.String())
	}
	for _, row := range rows {
		if row.ID == id {
			return row
		}
	}
	t.Fatalf("thread %s not in /threads: %s", id, w.Body.String())
	return threadProviderRow{}
}

// Spawning: an unknown provider is refused at every entry point.

func TestAPISpawnRefusesUnknownProvider(t *testing.T) {
	api, thinker := newPinTestAPI(t, newPinTestProvider("alpha"), newPinTestProvider("beta"))
	w := postThreadForTest(t, api, "worker", map[string]any{"directive": "Work.", "provider": "missing"})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d, want 400: %s", w.Code, w.Body.String())
	}
	for _, want := range []string{`unknown provider "missing"`, "alpha", "beta"} {
		if !strings.Contains(w.Body.String(), want) {
			t.Fatalf("error %q does not mention %q", w.Body.String(), want)
		}
	}
	if findThinkerByID(thinker, "worker") != nil {
		t.Fatal("a refused spawn created a thread")
	}
	if len(thinker.config.GetThreads()) != 0 {
		t.Fatalf("a refused spawn saved a thread: %#v", thinker.config.GetThreads())
	}
}

func TestAPISpawnRefusesUnknownRealtimeProvider(t *testing.T) {
	api, thinker := newPinTestAPI(t, newPinTestProvider("alpha"))
	thinker.config.RealtimeEnabled = true
	session := &fakeRealtimeProvider{}
	withFakeRealtime(thinker.pool, session)
	// A text provider's name isn't a realtime provider's, even when it exists.
	for _, name := range []string{"missing-realtime", "alpha"} {
		w := postThreadForTest(t, api, "voice", map[string]any{"realtime": true, "provider": name})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s: status=%d, want 400: %s", name, w.Code, w.Body.String())
		}
		for _, want := range []string{`unknown realtime provider "` + name + `"`, session.Name()} {
			if !strings.Contains(w.Body.String(), want) {
				t.Fatalf("error %q does not mention %q", w.Body.String(), want)
			}
		}
	}
}

func TestAPISpawnPinsARequestedProvider(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	api, thinker := newPinTestAPI(t, alpha, beta)

	if w := postThreadForTest(t, api, "pinned", map[string]any{"directive": "Work.", "provider": " beta "}); w.Code != http.StatusOK {
		t.Fatalf("pinned spawn status=%d: %s", w.Code, w.Body.String())
	}
	if w := postThreadForTest(t, api, "follower", map[string]any{"directive": "Work."}); w.Code != http.StatusOK {
		t.Fatalf("follower spawn status=%d: %s", w.Code, w.Body.String())
	}
	pinned, follower := savedThread(t, thinker, "pinned"), savedThread(t, thinker, "follower")
	if savedPin(t, pinned) != "beta" || pinned.Provider != "beta" {
		t.Fatalf("pinned record = %#v", pinned)
	}
	if savedPin(t, follower) != "" || follower.Provider != "alpha" {
		t.Fatalf("follower record = %#v", follower)
	}
	row := threadProviderRowFor(t, api, "pinned")
	if row.Provider != "beta" || row.ProviderPin != "beta" || row.ProviderMissing {
		t.Fatalf("/threads row = %#v", row)
	}
}

func TestLLMSpawnRefusesUnknownProvider(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"), newPinTestProvider("beta"))
	call := func(args map[string]string) ToolResult {
		t.Helper()
		_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{Name: "spawn", NativeID: "spawn-" + args["id"], Raw: "spawn", Args: args}}, nil)
		if len(results) != 1 {
			t.Fatalf("results = %#v", results)
		}
		return results[0]
	}
	refused := call(map[string]string{"id": "worker", "directive": "Work.", "provider": "missing"})
	if !refused.IsError || !strings.Contains(refused.Content, `unknown provider "missing"`) {
		t.Fatalf("result = %#v", refused)
	}
	if findThinkerByID(thinker, "worker") != nil {
		t.Fatal("a refused spawn created a thread")
	}
	accepted := call(map[string]string{"id": "pinned", "directive": "Work.", "provider": "beta", "paused": "true"})
	if accepted.IsError {
		t.Fatalf("result = %#v", accepted)
	}
	if savedPin(t, savedThread(t, thinker, "pinned")) != "beta" {
		t.Fatalf("record = %#v", savedThread(t, thinker, "pinned"))
	}
}

func TestLLMSpawnRefusesUnknownRealtimeProvider(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"))
	thinker.config.RealtimeEnabled = true
	withFakeRealtime(thinker.pool, &fakeRealtimeProvider{})
	_, _, results := mainToolHandler(thinker)(thinker, []toolCall{{
		Name: "spawn", NativeID: "spawn-voice", Raw: "spawn",
		Args: map[string]string{"id": "voice", "directive": "Talk.", "realtime": "true", "provider": "alpha"},
	}}, nil)
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, `unknown realtime provider "alpha"`) {
		t.Fatalf("results = %#v", results)
	}
	if findThinkerByID(thinker, "voice") != nil {
		t.Fatal("a refused spawn created a thread")
	}
}

func TestNestedLLMSpawnRefusesUnknownProvider(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"))
	lead := spawnPinWorker(t, thinker.threads, "lead", []string{"spawn"}, SpawnOpts{})
	_, _, results := lead.Thinker.handleTools(lead.Thinker, []toolCall{{
		Name: "spawn", NativeID: "nested", Raw: "spawn",
		Args: map[string]string{"id": "kid", "directive": "Work.", "provider": "missing"},
	}}, nil)
	if len(results) != 1 || !strings.Contains(results[0].Content, `unknown provider "missing"`) {
		t.Fatalf("results = %#v", results)
	}
	if lead.Children.threads["kid"] != nil {
		t.Fatal("a refused nested spawn created a thread")
	}
}

// pace: same rule, and a switch is a pin.

func TestPaceRefusesUnknownProviderWithoutChangingAnything(t *testing.T) {
	alpha := newPinTestProvider("alpha")
	_, thinker := newPinTestAPI(t, alpha)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	before := worker.Thinker.agentSleep
	_, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "missing", "sleep": "1h"})
	var invalid *invalidProviderError
	if !errors.As(err, &invalid) {
		t.Fatalf("err = %v", err)
	}
	if worker.Thinker.agentSleep != before {
		t.Fatal("a refused pace still changed the sleep")
	}
	assertRunsOn(t, worker, alpha)
}

func TestPaceSwitchPinsAWorkerAcrossRestarts(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})

	// Naming what it already follows isn't a choice to leave its parent.
	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "alpha"}); err != nil {
		t.Fatal(err)
	}
	if worker.Thinker.providerPin != "" {
		t.Fatalf("pin = %q", worker.Thinker.providerPin)
	}

	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": " beta "}); err != nil {
		t.Fatal(err)
	}
	assertRunsOn(t, worker, beta)
	if saved := savedThread(t, thinker, "worker"); savedPin(t, saved) != "beta" || saved.Provider != "beta" {
		t.Fatalf("record = %#v", saved)
	}
}

func TestPaceOnMainDoesNotPin(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"), newPinTestProvider("beta"))
	if _, err := applyPaceArgs(thinker, map[string]string{"provider": "beta"}); err != nil {
		t.Fatal(err)
	}
	if thinker.provider.Name() != "beta" || thinker.providerPin != "" {
		t.Fatalf("main provider=%s pin=%q", thinker.provider.Name(), thinker.providerPin)
	}
}

func TestPaceSwitchKeepsBuiltinScope(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"), newPinTestProvider("beta"))
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{BuiltinTools: []string{}})
	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "beta"}); err != nil {
		t.Fatal(err)
	}
	got := worker.Thinker.provider.(*pinTestProvider)
	if got.name != "beta" || got.scope == nil || len(got.scope) != 0 {
		t.Fatalf("provider = %#v", got)
	}
}

func TestPacePinMovesAWorkerOffAMissingPin(t *testing.T) {
	alpha := newPinTestProvider("alpha")
	_, thinker := newPinTestAPI(t, alpha)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{ProviderName: "gone"})
	if !worker.Thinker.status().ProviderMissing {
		t.Fatal("worker should report its missing provider")
	}
	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "alpha"}); err != nil {
		t.Fatal(err)
	}
	if status := worker.Thinker.status(); status.ProviderMissing || status.ProviderPin != "alpha" {
		t.Fatalf("status = %#v", status)
	}
	pinWaitFor(t, "restored event", func() bool { return len(traceEvents(t, thinker.telemetry, "thread.provider_restored")) == 1 })
	if event := traceEvents(t, thinker.telemetry, "thread.provider_restored")[0]; event["source"] != "pace" || event["running_on"] != "alpha" {
		t.Fatalf("event = %#v", event)
	}
}

func TestPaceRollbackRestoresTheProviderAndPinWhenSavingFails(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	worker.Thinker.persistPace = func(PersistentPaceState) error { return errors.New("disk full") }

	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "beta"}); err == nil {
		t.Fatal("expected the save failure to be returned")
	}
	assertRunsOn(t, worker, alpha)
	if status := worker.Thinker.status(); status.Provider != "alpha" || status.ProviderPin != "" {
		t.Fatalf("status = %#v", status)
	}
}

func TestPaceCannotMoveProviderManagedInference(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	managed := &realtimeManagedProvider{LLMProvider: alpha}
	worker.Thinker.provider = managed

	_, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "beta"})
	if err == nil || !strings.Contains(err.Error(), "provider-managed inference") {
		t.Fatalf("err = %v", err)
	}
	if worker.Thinker.provider != LLMProvider(managed) || worker.Thinker.providerPin != "" {
		t.Fatalf("provider=%#v pin=%q", worker.Thinker.provider, worker.Thinker.providerPin)
	}
	// Naming the provider it's already on is allowed, and doesn't unwrap it.
	if _, err := applyPaceArgs(worker.Thinker, map[string]string{"provider": "alpha"}); err != nil {
		t.Fatal(err)
	}
	if worker.Thinker.provider != LLMProvider(managed) {
		t.Fatalf("provider = %#v", worker.Thinker.provider)
	}
}

func TestSelectableProviderWithoutAPool(t *testing.T) {
	alpha := newPinTestProvider("alpha")
	got, err := selectableProvider(nil, alpha, "alpha")
	if err != nil || got != LLMProvider(alpha) {
		t.Fatalf("got=%#v err=%v", got, err)
	}
	_, err = selectableProvider(nil, alpha, "beta")
	var invalid *invalidProviderError
	if !errors.As(err, &invalid) || !strings.Contains(err.Error(), "none configured") {
		t.Fatalf("err = %v", err)
	}
}

// Restoring: a saved thread is never refused, and old records keep working.

func TestRestoredProviderName(t *testing.T) {
	pin := func(name string) *string { return &name }
	parent := &Thinker{provider: newPinTestProvider("parent")}
	tm := &ThreadManager{parent: parent}
	for name, tc := range map[string]struct {
		record PersistentThread
		want   string
	}{
		"explicit pin":                {PersistentThread{Provider: "parent", ProviderPin: pin("other")}, "other"},
		"explicit pin to parent":      {PersistentThread{Provider: "parent", ProviderPin: pin("parent")}, "parent"},
		"pin to a removed provider":   {PersistentThread{Provider: "parent", ProviderPin: pin("gone")}, "gone"},
		"follows parent":              {PersistentThread{Provider: "other", ProviderPin: pin("")}, ""},
		"old record matching parent":  {PersistentThread{Provider: "parent"}, ""},
		"old record on another":       {PersistentThread{Provider: "other"}, "other"},
		"old record with no provider": {PersistentThread{}, ""},
		"realtime":                    {PersistentThread{Provider: "fake-realtime", Realtime: true}, "fake-realtime"},
		"realtime named like parent":  {PersistentThread{Provider: "parent", Realtime: true}, "parent"},
		"voice worker":                {PersistentThread{Provider: "parent", WorkMode: true, ProviderPin: pin("")}, "parent"},
	} {
		t.Run(name, func(t *testing.T) {
			if got := tm.restoredProviderName(tc.record); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPersistentThreadProviderFieldsSurviveJSON(t *testing.T) {
	empty := ""
	for name, tc := range map[string]struct {
		record    PersistentThread
		wantPin   string
		noPin     bool
		wantScope bool
	}{
		"old record":      {record: PersistentThread{ID: "x"}, noPin: true},
		"follows parent":  {record: PersistentThread{ID: "x", ProviderPin: &empty}},
		"no builtins":     {record: PersistentThread{ID: "x", ProviderPin: &empty, BuiltinTools: []string{}}, wantScope: true},
		"limited builtin": {record: PersistentThread{ID: "x", ProviderPin: &empty, BuiltinTools: []string{"web"}}, wantScope: true},
	} {
		t.Run(name, func(t *testing.T) {
			raw, err := json.Marshal(tc.record)
			if err != nil {
				t.Fatal(err)
			}
			var got PersistentThread
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			if (got.ProviderPin == nil) != tc.noPin {
				t.Fatalf("pin = %v in %s", got.ProviderPin, raw)
			}
			if (got.BuiltinTools != nil) != tc.wantScope {
				t.Fatalf("builtin scope = %#v in %s", got.BuiltinTools, raw)
			}
		})
	}
}

func restartWithThreads(t *testing.T, provider *pinTestProvider, realtime bool, threads ...PersistentThread) *Thinker {
	t.Helper()
	t.Chdir(t.TempDir())
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cfg := NewConfig()
	cfg.path = filepath.Join(cwd, configFile)
	if err := cfg.update(func() { cfg.Threads = threads; cfg.RealtimeEnabled = realtime }); err != nil {
		t.Fatal(err)
	}
	reloaded := NewConfig()
	if err := reloaded.LoadError(); err != nil {
		t.Fatal(err)
	}
	thinker := NewThinker("", provider, reloaded)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := thinker.Shutdown(ctx); err != nil {
			t.Errorf("shutdown: %v", err)
		}
		thinker.threads.KillAll()
	})
	return thinker
}

func TestRestartKeepsAWorkerWhosePinnedProviderIsGone(t *testing.T) {
	parent := newPinTestProvider("parent")
	gone := "gone"
	thinker := restartWithThreads(t, parent, false, PersistentThread{ID: "worker", Directive: "Work.", Provider: "gone", ProviderPin: &gone})

	worker := thinker.threads.threads["worker"]
	if worker == nil {
		t.Fatal("a worker with a removed provider was lost on restart")
	}
	pinWaitStarted(t, parent) // it keeps working, on its parent's provider
	status := worker.Thinker.status()
	if status.Provider != "parent" || status.ProviderPin != "gone" || !status.ProviderMissing {
		t.Fatalf("status = %#v", status)
	}
	state, err := thinker.threads.PersistentState("worker")
	if err != nil {
		t.Fatal(err)
	}
	if savedPin(t, state) != "gone" {
		t.Fatalf("the pin was lost, so the worker can't return to it: %#v", state)
	}
	pinWaitFor(t, "missing-provider event", func() bool { return len(traceEvents(t, thinker.telemetry, "thread.provider_missing")) == 1 })
	event := traceEvents(t, thinker.telemetry, "thread.provider_missing")[0]
	if event["pinned_provider"] != "gone" || event["running_on"] != "parent" || event["event_thread_id"] != "worker" {
		t.Fatalf("event = %#v", event)
	}
}

func TestRestartMovesAnUnpinnedWorkerToItsParentsProvider(t *testing.T) {
	parent := newPinTestProvider("parent")
	empty := ""
	thinker := restartWithThreads(t, parent, false, PersistentThread{ID: "worker", Directive: "Work.", Provider: "old-default", ProviderPin: &empty})
	worker := thinker.threads.threads["worker"]
	if status := worker.Thinker.status(); status.Provider != "parent" || status.ProviderPin != "" || status.ProviderMissing {
		t.Fatalf("status = %#v", status)
	}
}

func TestRestartRestoresBuiltinScope(t *testing.T) {
	parent := newPinTestProvider("parent")
	empty := ""
	thinker := restartWithThreads(t, parent, false,
		PersistentThread{ID: "restricted", Directive: "Work.", ProviderPin: &empty, BuiltinTools: []string{}},
		PersistentThread{ID: "inherits", Directive: "Work.", ProviderPin: &empty},
	)
	restricted := thinker.threads.threads["restricted"].Thinker.provider.(*pinTestProvider)
	if restricted.scope == nil || len(restricted.scope) != 0 {
		t.Fatalf("a worker with no builtins got them back: %#v", restricted.scope)
	}
	if inherits := thinker.threads.threads["inherits"].Thinker.provider.(*pinTestProvider); inherits.scope != nil {
		t.Fatalf("scope appeared from nowhere: %#v", inherits.scope)
	}
}

func TestRestartReportsAThreadItCouldNotRestore(t *testing.T) {
	thinker := restartWithThreads(t, newPinTestProvider("parent"), true, PersistentThread{ID: "voice", Directive: "Talk.", Realtime: true, Provider: "gone-realtime"})
	if thinker.threads.threads["voice"] != nil {
		t.Fatal("expected the realtime thread to be refused")
	}
	pinWaitFor(t, "restore failure event", func() bool { return len(traceEvents(t, thinker.telemetry, "thread.restore_failed")) == 1 })
	if _, ok := persistentThreadByID(thinker.config.GetThreads(), "voice"); !ok {
		t.Fatal("the saved record should remain so the thread can return with its provider")
	}
}

func TestRestartAppliesTheSameRulesToNestedWorkers(t *testing.T) {
	parent := newPinTestProvider("parent")
	empty := ""
	thinker := restartWithThreads(t, parent, true,
		PersistentThread{ID: "owner", Directive: "Own.", Tools: []string{"spawn"}, ProviderPin: &empty},
		PersistentThread{ID: "follower", ParentID: "owner", Depth: 1, Directive: "Work.", Provider: "old-default", ProviderPin: &empty},
		PersistentThread{ID: "restricted", ParentID: "owner", Depth: 1, Directive: "Work.", ProviderPin: &empty, BuiltinTools: []string{}},
		PersistentThread{ID: "voice", ParentID: "owner", Depth: 1, Directive: "Talk.", Realtime: true, Provider: "gone-realtime"},
	)
	children := thinker.threads.threads["owner"].Children
	if status := children.threads["follower"].Thinker.status(); status.Provider != "parent" || status.ProviderPin != "" || status.ProviderMissing {
		t.Fatalf("follower status = %#v", status)
	}
	restricted := children.threads["restricted"].Thinker.provider.(*pinTestProvider)
	if restricted.scope == nil || len(restricted.scope) != 0 {
		t.Fatalf("a nested worker with no builtins got them back: %#v", restricted.scope)
	}
	if children.threads["voice"] != nil {
		t.Fatal("expected the nested realtime thread to be refused")
	}
	pinWaitFor(t, "restore failure event", func() bool { return len(traceEvents(t, thinker.telemetry, "thread.restore_failed")) == 1 })
}

// Threads whose provider isn't a pin.

func TestRealtimeThreadIsNotPinnedAndSavesItsRealtimeProvider(t *testing.T) {
	_, thinker := newPinTestAPI(t, newPinTestProvider("alpha"))
	session := &fakeRealtimeProvider{}
	withFakeRealtime(thinker.pool, session)
	thinker.config.RealtimeEnabled = true
	for id, requested := range map[string]string{"named": session.Name(), "default": ""} {
		if err := thinker.threads.SpawnWithOpts(id, "Talk.", nil, SpawnOpts{Realtime: true, ProviderName: requested, DeferRun: true}); err != nil {
			t.Fatal(err)
		}
		if status := thinker.threads.threads[id].Thinker.status(); status.ProviderPin != "" || status.ProviderMissing {
			t.Fatalf("%s status = %#v", id, status)
		}
		state, err := thinker.threads.PersistentState(id)
		if err != nil {
			t.Fatal(err)
		}
		if state.Provider != session.Name() || state.ProviderPin != nil {
			t.Fatalf("%s record = %#v", id, state)
		}
	}
}

func TestVoiceWorkerRunsOnItsCallsProviderAfterThatProviderIsRemoved(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha)
	session := &fakeRealtimeProvider{}
	withFakeRealtime(thinker.pool, session)
	thinker.config.RealtimeEnabled = true
	if err := thinker.threads.SpawnWithOpts("voice", "Talk.", nil, SpawnOpts{Realtime: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	rt := thinker.threads.threads["voice"].Realtime
	defer rt.cancel()
	rt.capabilities = RealtimeCapabilities{CoreExecution: true}
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}

	// The provider goes away during the call, before its first assignment.
	pool := withFakeRealtime(pinTestPool("beta", beta), session)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)

	worker, err := rt.ensureExecutionThread()
	if err != nil {
		t.Fatal(err)
	}
	assertRunsOn(t, worker, alpha)
	if status := worker.Thinker.status(); status.ProviderPin != "" || status.ProviderMissing {
		t.Fatalf("status = %#v", status)
	}
	if events := traceEvents(t, thinker.telemetry, "thread.provider_missing"); len(events) != 0 {
		t.Fatalf("the worker never lost its provider: %#v", events)
	}
}

// Config changes reach every running thread.

func TestProviderSyncMovesEachThreadWhereARestartWould(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)

	follower := spawnPinWorker(t, thinker.threads, "follower", []string{"spawn"}, SpawnOpts{})
	followerKid := spawnPinWorker(t, follower.Children, "follower-kid", nil, SpawnOpts{ParentID: "follower", Depth: 1})
	pinned := spawnPinWorker(t, thinker.threads, "pinned", []string{"spawn"}, SpawnOpts{ProviderName: "beta"})
	pinnedKid := spawnPinWorker(t, pinned.Children, "pinned-kid", nil, SpawnOpts{ParentID: "pinned", Depth: 1})
	pinnedToAlpha := spawnPinWorker(t, follower.Children, "alpha-kid", nil, SpawnOpts{ParentID: "follower", Depth: 1, ProviderName: "alpha"})

	// Same names, new settings: the pool holds new provider objects.
	newAlpha, newBeta, gamma := newPinTestProvider("alpha"), newPinTestProvider("beta"), newPinTestProvider("gamma")
	pool := pinTestPool("gamma", newAlpha, newBeta, gamma)
	thinker.pool, thinker.provider = pool, gamma
	thinker.syncChildProviders(pool)

	assertRunsOn(t, follower, gamma)
	assertRunsOn(t, followerKid, gamma)
	assertRunsOn(t, pinned, newBeta)
	assertRunsOn(t, pinnedKid, newBeta)
	assertRunsOn(t, pinnedToAlpha, newAlpha)
	for _, thread := range []*Thread{follower, followerKid, pinned, pinnedKid, pinnedToAlpha} {
		if thread.Thinker.pool != pool {
			t.Fatalf("%s kept the old pool", thread.ID)
		}
		if thread.Thinker.status().ProviderMissing {
			t.Fatalf("%s reports a missing provider", thread.ID)
		}
	}
	if saved := savedThread(t, thinker, "follower"); saved.Provider != "gamma" || savedPin(t, saved) != "" {
		t.Fatalf("follower record = %#v", saved)
	}
	if saved := savedThread(t, thinker, "pinned"); saved.Provider != "beta" || savedPin(t, saved) != "beta" {
		t.Fatalf("pinned record = %#v", saved)
	}
	// A leader's next spawn checks against the new pool too.
	if err := follower.Children.validateSpawnProvider("gamma", false); err != nil {
		t.Fatalf("new provider refused: %v", err)
	}
}

func TestProviderSyncUpdatesNestedSpawnChecksAfterARemoval(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)
	lead := spawnPinWorker(t, thinker.threads, "lead", []string{"spawn"}, SpawnOpts{})
	pool := pinTestPool("alpha", alpha)
	thinker.pool = pool
	thinker.syncChildProviders(pool)
	var invalid *invalidProviderError
	if err := lead.Children.validateSpawnProvider("beta", false); !errors.As(err, &invalid) {
		t.Fatalf("removed provider accepted: %v", err)
	}
}

func TestProviderSyncKeepsBuiltinScope(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{BuiltinTools: []string{"web"}})
	pool := pinTestPool("beta", beta)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)
	got := worker.Thinker.provider.(*pinTestProvider)
	if got.origin() != beta || len(got.scope) != 1 || got.scope[0] != "web" {
		t.Fatalf("provider = %#v", got)
	}
	saved := savedThread(t, thinker, "worker")
	if len(saved.BuiltinTools) != 1 || saved.BuiltinTools[0] != "web" {
		t.Fatalf("record = %#v", saved)
	}
}

func TestProviderSyncKeepsWorkerRunningWhenItsPinIsRemovedAndReturnsIt(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	api, thinker := newPinTestAPI(t, alpha, beta)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{ProviderName: "beta"})

	without := pinTestPool("alpha", alpha)
	thinker.pool = without
	thinker.syncChildProviders(without)
	assertRunsOn(t, worker, alpha)
	if status := worker.Thinker.status(); status.Paused || !status.ProviderMissing || status.ProviderPin != "beta" {
		t.Fatalf("status = %#v", status)
	}
	if row := threadProviderRowFor(t, api, "worker"); row.Provider != "alpha" || row.ProviderPin != "beta" || !row.ProviderMissing {
		t.Fatalf("/threads row = %#v", row)
	}
	if saved := savedThread(t, thinker, "worker"); savedPin(t, saved) != "beta" {
		t.Fatalf("record = %#v", saved)
	}

	// A second change that leaves it missing doesn't repeat the alert.
	thinker.syncChildProviders(without)

	again := newPinTestProvider("beta")
	with := pinTestPool("alpha", alpha, again)
	thinker.pool = with
	thinker.syncChildProviders(with)
	assertRunsOn(t, worker, again)
	if status := worker.Thinker.status(); status.ProviderMissing {
		t.Fatalf("status = %#v", status)
	}

	pinWaitFor(t, "provider events", func() bool {
		return len(traceEvents(t, thinker.telemetry, "thread.provider_restored")) == 1
	})
	if missing := traceEvents(t, thinker.telemetry, "thread.provider_missing"); len(missing) != 1 {
		t.Fatalf("missing events = %#v", missing)
	}
}

func TestProviderSyncCancelsAnInFlightRequestAndRetriesOnTheNewProvider(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha, beta)
	if err := thinker.threads.SpawnWithOpts("worker", "Work.", nil, SpawnOpts{}); err != nil {
		t.Fatal(err)
	}
	pinWaitStarted(t, alpha)

	pool := pinTestPool("beta", alpha, beta)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)

	pinWaitStarted(t, beta)
	if alpha.cancelled.Load() == 0 {
		t.Fatal("the old provider's request was not cancelled")
	}
	if alpha.calls.Load() != 1 {
		t.Fatalf("old provider received %d requests after the change", alpha.calls.Load())
	}
}

func TestProviderSyncDoesNotWaitForBusyThreads(t *testing.T) {
	alpha, beta, gamma := newPinTestProvider("alpha"), newPinTestProvider("beta"), newPinTestProvider("gamma")
	_, thinker := newPinTestAPI(t, alpha)
	lead := spawnPinWorker(t, thinker.threads, "lead", []string{"spawn"}, SpawnOpts{})
	// Marked running, but no loop services it: it is as busy as a thread can be.
	lead.Thinker.beginRuntime()
	defer lead.Thinker.endRuntime()

	returned := make(chan struct{})
	first, second := pinTestPool("beta", beta), pinTestPool("gamma", gamma)
	go func() {
		thinker.provider = beta
		thinker.syncChildProviders(first)
		thinker.provider = gamma
		thinker.syncChildProviders(second)
		close(returned)
	}()
	select {
	case <-returned:
	case <-time.After(3 * time.Second):
		t.Fatal("the owner waited on a busy worker")
	}
	assertRunsOn(t, lead, alpha)

	// A child spawned before the worker takes the change is still moved.
	kid := spawnPinWorker(t, lead.Children, "kid", nil, SpawnOpts{ParentID: "lead", Depth: 1})
	assertRunsOn(t, kid, alpha)

	if !lead.Thinker.applyRuntimeMutations() {
		t.Fatal("moving a worker should invalidate its in-flight request")
	}
	// Changes apply in order, so the newest wins.
	assertRunsOn(t, lead, gamma)
	assertRunsOn(t, kid, gamma)
}

func TestProviderSyncLeavesALiveVoiceCallAlone(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha)
	session := &fakeRealtimeProvider{}
	withFakeRealtime(thinker.pool, session)
	thinker.config.RealtimeEnabled = true

	if err := thinker.threads.SpawnWithOpts("voice", "Talk.", nil, SpawnOpts{Realtime: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	voice := thinker.threads.threads["voice"]
	backend := spawnPinWorker(t, voice.Children, "execution", nil, SpawnOpts{ParentID: "voice", Depth: 1, WorkMode: true, ProviderName: "alpha", CapabilityMode: SpawnCapabilitiesInherit})
	backendPool := backend.Thinker.pool
	voice.Thinker.beginRuntime()
	defer voice.Thinker.endRuntime()
	backend.Thinker.beginRuntime()
	defer backend.Thinker.endRuntime()

	pool := withFakeRealtime(pinTestPool("beta", beta), session)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)

	if voice.Thinker.applyRuntimeMutations() {
		t.Fatal("the change would restart the realtime backend mid-call")
	}
	if voice.Realtime.provider != RealtimeProvider(session) {
		t.Fatal("the live session lost its realtime provider")
	}
	// The next call builds its backend from these.
	assertRunsOn(t, voice, beta)
	if voice.Thinker.pool != pool {
		t.Fatal("realtime thread kept the old pool")
	}
	// The worker serving this call is untouched, and nothing was queued for it.
	if backend.Thinker.applyRuntimeMutations() {
		t.Fatal("the backend worker's request was invalidated mid-call")
	}
	assertRunsOn(t, backend, alpha)
	if backend.Thinker.pool != backendPool {
		t.Fatal("the backend worker's pool changed mid-call")
	}
}

func TestProviderSyncReportsARemovedRealtimeProviderWithoutTouchingTheCall(t *testing.T) {
	alpha := newPinTestProvider("alpha")
	_, thinker := newPinTestAPI(t, alpha)
	session := &fakeRealtimeProvider{}
	withFakeRealtime(thinker.pool, session)
	thinker.config.RealtimeEnabled = true
	if err := thinker.threads.SpawnWithOpts("voice", "Talk.", nil, SpawnOpts{Realtime: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	voice := thinker.threads.threads["voice"]

	pool := pinTestPool("alpha", alpha)
	thinker.pool = pool
	thinker.syncChildProviders(pool)

	if voice.Realtime.provider != RealtimeProvider(session) {
		t.Fatal("the live session lost its realtime provider")
	}
	pinWaitFor(t, "missing-provider event", func() bool { return len(traceEvents(t, thinker.telemetry, "thread.provider_missing")) == 1 })
	event := traceEvents(t, thinker.telemetry, "thread.provider_missing")[0]
	if event["realtime"] != true || event["pinned_provider"] != session.Name() {
		t.Fatalf("event = %#v", event)
	}
}

func TestProviderSyncRefreshesWorkerToolDocs(t *testing.T) {
	textTools, nativeTools := newPinTestProvider("text"), newPinTestProvider("native")
	nativeTools.native = true
	_, thinker := newPinTestAPI(t, textTools, nativeTools)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	before := worker.Thinker.messages[0].Content
	epoch := worker.Thinker.promptCacheEpoch

	pool := pinTestPool("native", textTools, nativeTools)
	thinker.pool, thinker.provider = pool, nativeTools
	thinker.syncChildProviders(pool)

	after := worker.Thinker.messages[0].Content
	if after == before {
		t.Fatal("worker prompt still describes tools for the old provider")
	}
	if after != worker.Thinker.rebuildPrompt("") {
		t.Fatal("a later prompt rebuild would undo the change")
	}
	if worker.Thinker.promptCacheEpoch == epoch || worker.Thinker.promptCacheResetReason != "provider_changed" {
		t.Fatalf("prompt cache not reset: epoch=%d reason=%q", worker.Thinker.promptCacheEpoch, worker.Thinker.promptCacheResetReason)
	}
	if shown := worker.Thinker.contextSnapshot().Messages; len(shown) == 0 || shown[0].Content != after {
		t.Fatal("the context view still shows the old prompt")
	}
}

func TestProviderSyncDoesNotBringBackARemovedThread(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha)
	spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	if err := thinker.config.RemoveThread("worker"); err != nil {
		t.Fatal(err)
	}
	pool := pinTestPool("beta", beta)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)
	if _, ok := persistentThreadByID(thinker.config.GetThreads(), "worker"); ok {
		t.Fatal("a late provider change brought a removed thread back")
	}
}

func TestProviderSyncDoesNotSaveEphemeralThreads(t *testing.T) {
	alpha, beta := newPinTestProvider("alpha"), newPinTestProvider("beta")
	_, thinker := newPinTestAPI(t, alpha)
	if err := thinker.threads.SpawnWithOpts("temp", "Work.", nil, SpawnOpts{DeferRun: true, Ephemeral: true}); err != nil {
		t.Fatal(err)
	}
	pool := pinTestPool("beta", beta)
	thinker.pool, thinker.provider = pool, beta
	thinker.syncChildProviders(pool)
	assertRunsOn(t, thinker.threads.threads["temp"], beta)
	if len(thinker.config.GetThreads()) != 0 {
		t.Fatalf("ephemeral thread saved: %#v", thinker.config.GetThreads())
	}
}

// PUT /config, end to end.

func TestPutConfigMovesRunningWorkersAndReportsARemovedPin(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	t.Setenv("ANTHROPIC_API_KEY", "test")
	api, thinker, _ := newPersistentThreadTestAPI(t)
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true},{"name":"anthropic"}]}`)

	// Running loops, but paused, so nothing calls a real provider.
	spawn := func(id, provider string) *Thread {
		t.Helper()
		if err := thinker.threads.SpawnWithOpts(id, "Wait.", nil, SpawnOpts{ProviderName: provider, Paused: true}); err != nil {
			t.Fatal(err)
		}
		state, err := thinker.threads.PersistentState(id)
		if err != nil {
			t.Fatal(err)
		}
		if err := thinker.config.SaveThread(state); err != nil {
			t.Fatal(err)
		}
		return thinker.threads.threads[id]
	}
	follower, pinned := spawn("follower", ""), spawn("pinned", "anthropic")
	if follower.Thinker.status().Provider != "openai" || pinned.Thinker.status().Provider != "anthropic" {
		t.Fatalf("start: follower=%s pinned=%s", follower.Thinker.status().Provider, pinned.Thinker.status().Provider)
	}

	// A new default reaches the follower. The pin holds.
	putProviderConfig(t, api, `{"providers":[{"name":"openai"},{"name":"anthropic","default":true}]}`)
	pinWaitFor(t, "follower to move", func() bool { return follower.Thinker.status().Provider == "anthropic" })
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true},{"name":"anthropic"}]}`)
	pinWaitFor(t, "follower to move back", func() bool { return follower.Thinker.status().Provider == "openai" })
	if got := pinned.Thinker.status().Provider; got != "anthropic" {
		t.Fatalf("pinned moved to %s", got)
	}

	// Removing the pinned provider keeps the worker going and says so.
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true}]}`)
	pinWaitFor(t, "pinned to fall back", func() bool { return pinned.Thinker.status().Provider == "openai" })
	if row := threadProviderRowFor(t, api, "pinned"); row.Provider != "openai" || row.ProviderPin != "anthropic" || !row.ProviderMissing {
		t.Fatalf("/threads row = %#v", row)
	}
	pinWaitFor(t, "saved pin", func() bool {
		saved := savedThread(t, thinker, "pinned")
		return saved.Provider == "openai" && savedPin(t, saved) == "anthropic"
	})
	if !pinned.Thinker.status().Paused {
		t.Fatal("a provider change unpaused a paused worker")
	}

	// Adding it back returns the worker to its pin.
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true},{"name":"anthropic"}]}`)
	pinWaitFor(t, "pinned to return", func() bool { return pinned.Thinker.status().Provider == "anthropic" })
	if row := threadProviderRowFor(t, api, "pinned"); row.ProviderMissing {
		t.Fatalf("/threads row = %#v", row)
	}
}

func TestPutConfigResendingTheSameProvidersLeavesWorkersAlone(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	fake := newPinTestProvider("fake")
	api, thinker := newPinTestAPI(t, fake)
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true}]}`)

	// Keep the worker on the fake, so a wrong move would show and not call out.
	thinker.provider, thinker.pool = fake, pinTestPool("fake", fake)
	if err := thinker.threads.SpawnWithOpts("worker", "Work.", nil, SpawnOpts{}); err != nil {
		t.Fatal(err)
	}
	pinWaitStarted(t, fake)

	// What the dashboard sends when someone edits only the directive.
	putProviderConfig(t, api, `{"directive":"A new directive.","providers":[{"name":"openai","default":true}],"realtime_enabled":false}`)
	if thinker.config.GetDirective() != "A new directive." {
		t.Fatal("the directive edit did not apply")
	}

	time.Sleep(100 * time.Millisecond)
	if got := fake.cancelled.Load(); got != 0 {
		t.Fatalf("a settings save cancelled the worker's request %d time(s)", got)
	}
	assertRunsOn(t, thinker.threads.threads["worker"], fake)
}

func TestPutConfigRealtimeToggleReachesRunningThreads(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	api, thinker := newPinTestAPI(t, newPinTestProvider("fake"))
	putProviderConfig(t, api, `{"providers":[{"name":"openai","default":true}]}`)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	before := worker.Thinker.pool

	putProviderConfig(t, api, `{"realtime_enabled":true}`)

	if worker.Thinker.pool == before || worker.Thinker.pool != thinker.pool {
		t.Fatal("the worker kept the pool from before the toggle")
	}
}

func TestPutConfigComparesEveryProviderField(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test")
	api, thinker := newPinTestAPI(t, newPinTestProvider("fake"))
	const limited = `{"providers":[{"name":"anthropic","default":true,"max_output_tokens":8000}]}`
	putProviderConfig(t, api, limited)
	worker := spawnPinWorker(t, thinker.threads, "worker", nil, SpawnOpts{})
	before := worker.Thinker.pool

	putProviderConfig(t, api, limited)
	if worker.Thinker.pool != before {
		t.Fatal("resending the same providers moved the worker")
	}
	putProviderConfig(t, api, `{"realtime_enabled":false}`)
	if worker.Thinker.pool != before {
		t.Fatal("repeating the realtime flag moved the worker")
	}
	putProviderConfig(t, api, `{"providers":[{"name":"anthropic","default":true,"max_output_tokens":4000}]}`)
	if worker.Thinker.pool == before || worker.Thinker.pool != thinker.pool {
		t.Fatal("a real provider edit did not reach the worker")
	}
}

func TestProviderConfigsEqual(t *testing.T) {
	base := func() []ProviderConfig {
		return []ProviderConfig{{Name: "openai", Default: true}, {Name: "anthropic", Models: map[string]string{"large": "a"}}}
	}
	if !providerConfigsEqual(nil, []ProviderConfig{}) {
		t.Fatal("no providers should equal no providers")
	}
	if !providerConfigsEqual(base(), base()) {
		t.Fatal("identical lists should be equal")
	}
	unsetVersusEmpty := base()
	unsetVersusEmpty[0].BuiltinTools = []string{}
	unsetVersusEmpty[0].Models = map[string]string{}
	if !providerConfigsEqual(base(), unsetVersusEmpty) {
		t.Fatal("unset and empty optional fields mean the same thing")
	}
	for name, edit := range map[string]func([]ProviderConfig){
		"default":      func(p []ProviderConfig) { p[0].Default, p[1].Default = false, true },
		"model":        func(p []ProviderConfig) { p[1].Models["large"] = "b" },
		"order":        func(p []ProviderConfig) { p[0], p[1] = p[1], p[0] },
		"max tokens":   func(p []ProviderConfig) { p[1].MaxOutputTokens = 8000 },
		"service tier": func(p []ProviderConfig) { p[0].ServiceTier = "priority" },
	} {
		changed := base()
		edit(changed)
		if providerConfigsEqual(base(), changed) {
			t.Fatalf("a %s change was not detected", name)
		}
	}
	if providerConfigsEqual(base(), base()[:1]) {
		t.Fatal("a removed provider was not detected")
	}
}

func TestRefreshThreadOnlyUpdatesSavedThreads(t *testing.T) {
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json")}
	if err := cfg.RefreshThread(PersistentThread{ID: "ghost"}); err != nil {
		t.Fatal(err)
	}
	if len(cfg.GetThreads()) != 0 {
		t.Fatal("refresh saved an unknown thread")
	}
	// Ephemeral threads take this path on every provider change.
	if _, err := os.Stat(cfg.path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("refreshing an unknown thread wrote the config: %v", err)
	}
	if err := cfg.SaveThread(PersistentThread{ID: "real", Directive: "old"}); err != nil {
		t.Fatal(err)
	}
	if err := cfg.RefreshThread(PersistentThread{ID: "real", Directive: "new"}); err != nil {
		t.Fatal(err)
	}
	if got := cfg.GetThreads(); len(got) != 1 || got[0].Directive != "new" {
		t.Fatalf("threads = %#v", got)
	}
}
