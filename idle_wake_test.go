package core

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

// Fake time exercises the real loop through hours of silence in milliseconds.
func TestIdleWakeRepeatedCyclesWithoutTiming(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		t.Chdir(t.TempDir())
		cfg := &Config{path: "config.json", Directive: "Maintain your standing responsibilities."}
		provider := newPacingTestProvider(3)
		thinker := NewThinker("", provider, cfg)
		defer thinker.blobs.Close()
		done := make(chan struct{})
		go func() { defer close(done); thinker.Run() }()
		defer func() {
			thinker.Stop()
			for i := 1; i <= 3; i++ {
				provider.release(i)
			}
			<-done
		}()
		for cycle := 1; cycle <= 3; cycle++ {
			synctest.Wait()
			select {
			case got := <-provider.started:
				if got != cycle {
					t.Fatalf("call = %d, want %d", got, cycle)
				}
			default:
				t.Fatalf("cycle %d never woke", cycle)
			}
			if cycle > 1 && !strings.Contains(messagesText(provider.request(cycle)), "reason: timer") {
				t.Fatal("autonomous cycle did not identify its timer wake")
			}
			provider.release(cycle)
			synctest.Wait()
			state := cfg.GetMainPace()
			want := time.Now().Add(idleWakeFallback)
			if state == nil || state.WaitForEvents || !state.NextWakeAt.Equal(want) || !thinker.status().NextWakeAt.Equal(want) {
				t.Fatalf("cycle %d missing durable fallback: %#v", cycle, state)
			}
			if cycle < 3 {
				time.Sleep(idleWakeFallback)
			}
		}
	})
}

func TestIdleWakeExplicitEventWaitReassessesAfterInput(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		t.Chdir(t.TempDir())
		cfg := &Config{path: "config.json", Directive: "Be purely reactive; no autonomous initiative."}
		provider := newPacingTestProvider(2)
		provider.responses = []ChatResponse{{Text: "I will wait for a request.", ToolCalls: []NativeToolCall{{ID: "wait", Name: "pace", Args: map[string]string{"clear_wake": "true"}}}}}
		thinker := NewThinker("", provider, cfg)
		defer thinker.blobs.Close()
		done := make(chan struct{})
		go func() { defer close(done); thinker.Run() }()
		defer func() { thinker.Stop(); provider.release(1); provider.release(2); <-done }()
		synctest.Wait()
		<-provider.started
		provider.release(1)
		synctest.Wait()
		state := cfg.GetMainPace()
		if state == nil || !state.WaitForEvents || !state.NextWakeAt.IsZero() {
			t.Fatalf("explicit wait lost: %#v", state)
		}
		time.Sleep(2 * idleWakeFallback)
		synctest.Wait()
		select {
		case <-provider.started:
			t.Fatal("explicit event wait woke automatically")
		default:
		}
		thinker.InjectConsole("Reassess your responsibilities.")
		synctest.Wait()
		if got := <-provider.started; got != 2 {
			t.Fatalf("event call = %d", got)
		}
		if state := cfg.GetMainPace(); state.WaitForEvents {
			t.Fatal("old event-only decision survived new input")
		}
		provider.release(2)
		synctest.Wait()
		if state := cfg.GetMainPace(); state.WaitForEvents || !state.NextWakeAt.Equal(time.Now().Add(idleWakeFallback)) {
			t.Fatalf("omitted new timing did not fall back: %#v", state)
		}
	})
}

func TestIdleWakeChoicesAndPersistenceFailure(t *testing.T) {
	for _, tc := range []struct {
		name      string
		args      map[string]string
		eventOnly bool
		delay     time.Duration
	}{
		{"omitted", nil, false, idleWakeFallback},
		{"inspect", map[string]string{}, false, idleWakeFallback},
		{"profile", map[string]string{"model": "small"}, false, idleWakeFallback},
		{"finite", map[string]string{"sleep": "2h"}, false, 2 * time.Hour},
		{"explicit events", map[string]string{"clear_wake": "true"}, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var saved PersistentPaceState
			th := &Thinker{persistPace: func(s PersistentPaceState) error { saved = s; return nil }}
			before := time.Now()
			if tc.args != nil {
				if _, err := applyPaceArgs(th, tc.args); err != nil {
					t.Fatal(err)
				}
			}
			if err := th.ensureIdleWake(time.Now()); err != nil {
				t.Fatal(err)
			}
			if th.waitForEvents != tc.eventOnly || saved.WaitForEvents != tc.eventOnly {
				t.Fatalf("decision mismatch: %#v", saved)
			}
			if tc.eventOnly {
				if !th.nextWakeAt.IsZero() {
					t.Fatal("event-only decision acquired a timer")
				}
			} else if th.nextWakeAt.Before(before.Add(tc.delay)) || th.nextWakeAt.After(time.Now().Add(tc.delay)) {
				t.Fatalf("wrong wake: %v", th.nextWakeAt)
			}
			if !saved.NextWakeAt.Equal(th.nextWakeAt) {
				t.Fatal("runtime/durable mismatch")
			}
		})
	}
	th := &Thinker{persistPace: func(PersistentPaceState) error { return errors.New("disk unavailable") }}
	if err := th.ensureIdleWake(time.Now()); err == nil || !th.nextWakeAt.IsZero() || th.paceDurable {
		t.Fatal("failed persistence reported a durable fallback")
	}
	if _, err := applyPaceArgs(th, map[string]string{"clear_wake": "true"}); err == nil || th.waitForEvents {
		t.Fatal("failed pace retained event-only intent")
	}
}

func TestIdleWakeDoesNotPostponeOrResume(t *testing.T) {
	for _, state := range []string{"future", "due", "paused", "stopped"} {
		t.Run(state, func(t *testing.T) {
			th := &Thinker{quit: make(chan struct{})}
			switch state {
			case "future":
				th.nextWakeAt = time.Now().Add(time.Hour)
			case "due":
				th.nextWakeAt = time.Now().Add(-time.Second)
			case "paused":
				th.paused = true
			case "stopped":
				close(th.quit)
			}
			original := th.nextWakeAt
			th.persistPace = func(PersistentPaceState) error {
				t.Fatal("unchanged/paused/stopped thread wrote a fallback")
				return nil
			}
			if err := th.ensureIdleWake(time.Now()); err != nil || !th.nextWakeAt.Equal(original) {
				t.Fatal("idle guard changed a protected state")
			}
		})
	}
}

func TestIdleWakeRestoresLegacyMissingDecision(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		t.Chdir(t.TempDir())
		cfg := &Config{path: "config.json", Directive: "Continue maintaining the assigned domain.", MainPace: &PersistentPaceState{Sleep: "30s"}}
		if err := cfg.Save(); err != nil {
			t.Fatal(err)
		}
		provider := newPacingTestProvider(1)
		thinker := NewThinker("", provider, cfg)
		defer thinker.blobs.Close()
		done := make(chan struct{})
		go func() { defer close(done); thinker.Run() }()
		defer func() { thinker.Stop(); provider.release(1); <-done }()
		synctest.Wait()
		state := cfg.GetMainPace()
		if state == nil || !state.NextWakeAt.Equal(time.Now().Add(idleWakeFallback)) {
			t.Fatalf("legacy wait not repaired: %#v", state)
		}
		reloaded := &Config{path: "config.json"}
		if err := reloaded.load(); err != nil {
			t.Fatal(err)
		}
		if !reloaded.GetMainPace().NextWakeAt.Equal(state.NextWakeAt) {
			t.Fatal("repair was not saved to disk")
		}
		time.Sleep(idleWakeFallback)
		synctest.Wait()
		select {
		case <-provider.started:
		default:
			t.Fatal("repaired legacy agent never woke")
		}
	})
}

func TestIdleWakeWorkerExplicitWaitPersists(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Coordinate the assigned domain."}
	provider := newParkedAPIProvider()
	parent := NewThinker("", provider, cfg)
	defer func() { provider.Release(); parent.Stop() }()
	if err := parent.threads.SpawnWithOpts("reactive", "React to requests only; no autonomous initiative.", nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["reactive"].Thinker
	if _, err := applyPaceArgs(worker, map[string]string{"clear_wake": "true"}); err != nil {
		t.Fatal(err)
	}
	state, err := parent.threads.PersistentState("reactive")
	if err != nil || state.Pace == nil || !state.Pace.WaitForEvents {
		t.Fatalf("worker snapshot lost decision: %#v %v", state.Pace, err)
	}
	if err := cfg.SaveThread(state); err != nil {
		t.Fatal(err)
	}
	reloaded := &Config{path: cfg.path}
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	restartProvider := newParkedAPIProvider()
	restarted := NewThinker("", restartProvider, reloaded)
	defer func() { restartProvider.Release(); restarted.Stop() }()
	select {
	case <-restartProvider.started:
		t.Fatal("restored reactive worker called the model")
	case <-time.After(50 * time.Millisecond):
	}
	if s, err := restarted.threads.PersistentState("reactive"); err != nil || s.Pace == nil || !s.Pace.WaitForEvents {
		t.Fatalf("restored worker lost explicit wait: %#v %v", s.Pace, err)
	}
}

func TestIdleWakeDirectiveContractIsShared(t *testing.T) {
	for name, prompt := range map[string]string{
		"main":   buildSystemPrompt("Own the domain.", NewToolRegistry("test"), "", nil, nil, nil, nil),
		"worker": formatThreadBasePrompt(false, false, "worker", "main"),
		"leader": formatThreadBasePrompt(true, false, "leader", "main"),
	} {
		if !strings.Contains(prompt, idlePacingContract) {
			t.Errorf("%s lacks shared timing contract", name)
		}
		if strings.Contains(prompt, "idle_wake_policy") {
			t.Errorf("%s still exposes removed config", name)
		}
	}
}

func TestIdleWakeDirectiveChangeReassessesExplicitWait(t *testing.T) {
	for _, role := range []string{"main", "worker"} {
		t.Run(role, func(t *testing.T) {
			t.Chdir(t.TempDir())
			cfg := &Config{path: "config.json", Directive: "React only to requests."}
			provider := newParkedAPIProvider()
			parent := NewThinker("", provider, cfg)
			defer func() { provider.Release(); parent.Stop(); parent.blobs.Close() }()
			th := parent
			if role == "worker" {
				if err := parent.threads.SpawnWithOpts("owner", cfg.Directive, nil, SpawnOpts{DeferRun: true}); err != nil {
					t.Fatal(err)
				}
				th = parent.threads.threads["owner"].Thinker
			}
			if _, err := applyPaceArgs(th, map[string]string{"clear_wake": "true"}); err != nil {
				t.Fatal(err)
			}
			directive := "Maintain the domain and reassess responsibilities autonomously."
			if role == "main" {
				if err := cfg.SetDirective(directive); err != nil {
					t.Fatal(err)
				}
				th.reloadDirectiveNow()
			} else if err := parent.threads.Update("owner", "", directive, nil); err != nil {
				t.Fatal(err)
			}
			if th.waitForEvents {
				t.Fatal("changed directive retained obsolete event-only decision")
			}
			if err := th.ensureIdleWake(time.Now()); err != nil {
				t.Fatal(err)
			}
			if !th.status().NextWakeAt.After(time.Now()) {
				t.Fatal("changed directive still has no finite wake")
			}
		})
	}
}
