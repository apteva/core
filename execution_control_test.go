package core

import (
	"testing"
	"time"
)

func TestExecutionControllerStepReleasesActiveThread(t *testing.T) {
	c := NewExecutionController(ExecutionControlConfig{
		Mode:        ExecutionStep,
		Breakpoints: []string{string(ExecutionPhaseToolBefore)},
	})
	quit := make(chan struct{})
	released := make(chan string, 2)

	go func() {
		c.Wait(ExecutionGate{ThreadID: "main", Phase: ExecutionPhaseToolBefore, Iteration: 1, Tool: "spawn"}, quit, nil)
		released <- "main"
	}()
	waitForActiveThread(t, c, "main")

	go func() {
		c.Wait(ExecutionGate{ThreadID: "worker", Phase: ExecutionPhaseToolBefore, Iteration: 1, Tool: "search"}, quit, nil)
		released <- "worker"
	}()
	waitForActiveThread(t, c, "worker")

	st, err := c.Control(ExecutionControlAction{Action: "step"})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if st.ActiveThreadID != "main" {
		t.Fatalf("active after releasing worker = %q, want main", st.ActiveThreadID)
	}
	select {
	case got := <-released:
		if got != "worker" {
			t.Fatalf("released %q, want worker", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for worker release")
	}

	if _, err := c.Control(ExecutionControlAction{Action: "step"}); err != nil {
		t.Fatalf("step main: %v", err)
	}
	select {
	case got := <-released:
		if got != "main" {
			t.Fatalf("released %q, want main", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for main release")
	}
}

func TestExecutionControllerRunReleasesAll(t *testing.T) {
	c := NewExecutionController(ExecutionControlConfig{
		Mode:        ExecutionPaused,
		Breakpoints: []string{string(ExecutionPhaseLLMDone)},
	})
	quit := make(chan struct{})
	released := make(chan string, 2)

	for _, id := range []string{"main", "worker"} {
		id := id
		go func() {
			c.Wait(ExecutionGate{ThreadID: id, Phase: ExecutionPhaseLLMDone, Iteration: 1}, quit, nil)
			released <- id
		}()
	}
	waitForWaitingCount(t, c, 2)
	if _, err := c.Control(ExecutionControlAction{Action: "run"}); err != nil {
		t.Fatalf("run: %v", err)
	}
	got := map[string]bool{}
	for len(got) < 2 {
		select {
		case id := <-released:
			got[id] = true
		case <-time.After(time.Second):
			t.Fatalf("timeout waiting for releases, got=%v", got)
		}
	}
	if st := c.Status(); st.Mode != ExecutionAuto || st.Waiting {
		t.Fatalf("status after run = %+v, want auto/no waiting", st)
	}
}

func waitForActiveThread(t *testing.T, c *ExecutionController, id string) {
	t.Helper()
	deadline := time.After(time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-deadline:
			t.Fatalf("timeout waiting for active thread %q; status=%+v", id, c.Status())
		case <-tick.C:
			if st := c.Status(); st.Waiting && st.ActiveThreadID == id {
				return
			}
		}
	}
}

func waitForWaitingCount(t *testing.T, c *ExecutionController, n int) {
	t.Helper()
	deadline := time.After(time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-deadline:
			t.Fatalf("timeout waiting for waiting_count=%d; status=%+v", n, c.Status())
		case <-tick.C:
			if st := c.Status(); st.WaitingCount == n {
				return
			}
		}
	}
}
