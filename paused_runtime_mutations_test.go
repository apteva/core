package core

import (
	"testing"
	"time"
)

func TestPausedExecutionServicesRuntimeMutationWithoutReleasing(t *testing.T) {
	_, thinker := newTestAPI()
	thinker.execution = NewExecutionController(ExecutionControlConfig{Mode: ExecutionPaused, Breakpoints: []string{string(ExecutionPhaseIterationStart)}})
	thinker.beginRuntime()
	finished := make(chan bool, 1)
	go func() {
		finished <- thinker.executionGate(ExecutionPhaseIterationStart, ExecutionGate{Summary: "paused"})
		thinker.endRuntime()
	}()
	waitForActiveThread(t, thinker.execution, thinker.threadID)
	t.Cleanup(func() {
		select {
		case <-thinker.quit:
		default:
			close(thinker.quit)
		}
		select {
		case <-finished:
		case <-time.After(time.Second):
			t.Error("gate did not stop")
		}
	})
	updated := make(chan error, 1)
	go func() {
		updated <- thinker.mutateRuntime(func() error { return thinker.config.SetDirective("new server behavior") })
	}()
	select {
	case err := <-updated:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("configuration update blocked behind pause")
	}
	if thinker.config.GetDirective() != "new server behavior" {
		t.Fatal("directive not applied")
	}
	status := thinker.execution.Status()
	if status.Mode != ExecutionPaused || !status.Waiting {
		t.Fatalf("update released pause: %+v", status)
	}
	select {
	case <-finished:
		t.Fatal("pause unexpectedly completed")
	default:
	}
}
