package core

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func rootPausedExecutionConfig() ExecutionControlConfig {
	return ExecutionControlConfig{Mode: ExecutionAuto, Breakpoints: []string{"iteration.start"}, Threads: map[string]ExecutionThreadControl{"main": {Mode: ExecutionPaused, Breakpoints: []string{"iteration.start"}}}}
}
func startScopedGate(t *testing.T, c *ExecutionController, id string) (<-chan bool, chan struct{}) {
	t.Helper()
	done := make(chan bool, 1)
	quit := make(chan struct{})
	go func() { done <- c.Wait(ExecutionGate{ThreadID: id, Phase: ExecutionPhaseIterationStart}, quit, nil) }()
	t.Cleanup(func() { close(quit) })
	return done, quit
}
func requireGateResult(t *testing.T, done <-chan bool, want bool) {
	t.Helper()
	select {
	case got := <-done:
		if got != want {
			t.Fatalf("proceed=%v want %v", got, want)
		}
	case <-time.After(time.Second):
		t.Fatal("gate did not finish")
	}
}
func TestExecutionThreadScopeAndGlobalPrecedence(t *testing.T) {
	c := NewExecutionController(rootPausedExecutionConfig())
	root, _ := startScopedGate(t, c, "main")
	waitForActiveThread(t, c, "main")
	worker, _ := startScopedGate(t, c, "voice-worker")
	requireGateResult(t, worker, true)
	if !c.ShouldGateThread("main", ExecutionPhaseIterationStart) || c.ShouldGateThread("voice-worker", ExecutionPhaseIterationStart) {
		t.Fatal("checkpoint scope differs from gate scope")
	}
	if _, err := c.Control(ExecutionControlAction{Action: "pause"}); err != nil {
		t.Fatal(err)
	}
	worker, _ = startScopedGate(t, c, "voice-worker")
	waitForWaitingCount(t, c, 2)
	if _, err := c.Control(ExecutionControlAction{Action: "run", Scope: "thread", ThreadID: "voice-worker"}); err != nil {
		t.Fatal(err)
	}
	if c.Status().WaitingCount != 2 {
		t.Fatal("thread run bypassed global pause")
	}
	for _, w := range c.Status().Waiters {
		if w.Scope != "instance" {
			t.Fatal("wrong blocking scope", w)
		}
	}
	if _, err := c.Control(ExecutionControlAction{Action: "run"}); err != nil {
		t.Fatal(err)
	}
	requireGateResult(t, worker, true)
	if st := c.Status(); st.WaitingCount != 1 || st.Waiters[0].Scope != "thread" {
		t.Fatal("global run erased root-specific pause", st)
	}
	if _, err := c.Control(ExecutionControlAction{Action: "run", Scope: "thread", ThreadID: "main"}); err != nil {
		t.Fatal(err)
	}
	requireGateResult(t, root, true)
}
func TestExecutionScopedStepCreditsCannotLeak(t *testing.T) {
	for _, scope := range []string{"instance", "thread"} {
		t.Run(scope, func(t *testing.T) {
			cfg := rootPausedExecutionConfig()
			if scope == "instance" {
				cfg.Mode = ExecutionPaused
				cfg.Threads = nil
			}
			c := NewExecutionController(cfg)
			if _, err := c.Control(ExecutionControlAction{Action: "step", Scope: scope, ThreadID: "main"}); err != nil {
				t.Fatal(err)
			}
			worker, _ := startScopedGate(t, c, "worker")
			if scope == "thread" {
				requireGateResult(t, worker, true)
			} else {
				waitForActiveThread(t, c, "worker")
			}
			root, _ := startScopedGate(t, c, "main")
			requireGateResult(t, root, true)
			root, _ = startScopedGate(t, c, "main")
			waitForActiveThread(t, c, "main")
			if !c.CancelThread("main") {
				t.Fatal("cancel failed")
			}
			requireGateResult(t, root, false)
		})
	}
}
func TestExecutionThreadStepWaitsForGlobalResume(t *testing.T) {
	config := rootPausedExecutionConfig()
	config.Mode = ExecutionPaused
	c := NewExecutionController(config)
	done, _ := startScopedGate(t, c, "main")
	waitForActiveThread(t, c, "main")
	c.Control(ExecutionControlAction{Action: "step", Scope: "thread", ThreadID: "main"})
	if c.Status().WaitingCount != 1 {
		t.Fatal("thread step bypassed global pause")
	}
	c.Control(ExecutionControlAction{Action: "run"})
	requireGateResult(t, done, true)
	done, _ = startScopedGate(t, c, "main")
	waitForActiveThread(t, c, "main")
	c.CancelThread("main")
	requireGateResult(t, done, false)
}
func TestExecutionScopedConfigValidationAndIsolation(t *testing.T) {
	config := rootPausedExecutionConfig()
	c := NewExecutionController(config)
	config.Threads["main"].Breakpoints[0] = "tool.before"
	copy := c.Config()
	copy.Threads["main"].Breakpoints[0] = "llm.done"
	if !c.ShouldGateThread("main", ExecutionPhaseIterationStart) {
		t.Fatal("configuration aliases caller memory")
	}
	for _, bad := range []ExecutionControlConfig{{Scope: "thread"}, {Mode: "bogus"}, {Threads: map[string]ExecutionThreadControl{"../worker": {Mode: ExecutionPaused}}}, {Threads: map[string]ExecutionThreadControl{"main": {Mode: "bogus"}}}, {Threads: map[string]ExecutionThreadControl{"main": {Mode: ExecutionPaused, Breakpoints: []string{"typo"}}}}} {
		if err := c.ApplyConfig(bad); err == nil {
			t.Fatal("accepted invalid scope/config", bad)
		}
		if !c.ShouldGateThread("main", ExecutionPhaseIterationStart) {
			t.Fatal("invalid config changed live gate")
		}
	}
	cfg := &Config{}
	if err := cfg.SetExecutionControl(rootPausedExecutionConfig()); err != nil {
		t.Fatal(err)
	}
	out := cfg.GetExecutionControl()
	out.Threads["main"].Breakpoints[0] = "invalid"
	if cfg.GetExecutionControl().Threads["main"].Breakpoints[0] != "iteration.start" {
		t.Fatal("stored config aliases read snapshot")
	}
}
func TestExecutionScopedAPIAndPersistence(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.execution = NewExecutionController(ExecutionControlConfig{})
	t.Cleanup(thinker.Stop)
	for _, body := range []string{`{"action":"pause","scope":"thread","thread_id":"main","breakpoints":["iteration.start"]}`, `{"action":"step","scope":"thread","thread_id":"main"}`} {
		r := httptest.NewRequest(http.MethodPost, "/control", strings.NewReader(body))
		w := httptest.NewRecorder()
		api.control(w, r)
		if w.Code != 200 {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	if !thinker.execution.ShouldGateThread("main", ExecutionPhaseIterationStart) || thinker.execution.ShouldGateThread("worker", ExecutionPhaseIterationStart) {
		t.Fatal("API changed instance scope")
	}
	config := thinker.config.GetExecutionControl()
	data, _ := json.Marshal(config)
	var restored ExecutionControlConfig
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	c := NewExecutionController(restored)
	root, _ := startScopedGate(t, c, "main")
	waitForActiveThread(t, c, "main") // step credits are never persisted
	c.CancelThread("main")
	requireGateResult(t, root, false)
	w := httptest.NewRecorder()
	api.control(w, httptest.NewRequest(http.MethodPost, "/control", strings.NewReader(`{"action":"run","scope":"thread"}`)))
	if w.Code != 400 {
		t.Fatal("missing exact target accepted", w.Code)
	}
}

func TestExecutionScopedQuitAndTelemetry(t *testing.T) {
	c := NewExecutionController(rootPausedExecutionConfig())
	quit := make(chan struct{})
	events := make(chan ExecutionPhaseData, 2)
	done := make(chan bool, 1)
	go func() {
		done <- c.Wait(ExecutionGate{ThreadID: "main", Phase: ExecutionPhaseIterationStart}, quit, func(_ string, d ExecutionPhaseData) { events <- d })
	}()
	select {
	case d := <-events:
		if d.Scope != "thread" || d.State != "blocked" || d.Phase != "iteration.start" {
			t.Fatal(d)
		}
	case <-time.After(time.Second):
		t.Fatal("missing blocked event")
	}
	close(quit)
	requireGateResult(t, done, false)
	if c.Status().WaitingCount != 0 {
		t.Fatal("cancel left waiter behind")
	}
}
