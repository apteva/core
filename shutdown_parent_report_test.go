package core

import (
	"context"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestShutdownJoinsScheduledLoopsBeforeCleanup(t *testing.T) {
	for _, realtime := range []bool{false, true} {
		t.Run(map[bool]string{false: "ordinary", true: "realtime"}[realtime], func(t *testing.T) {
			th := newTestThinker()
			var cleanup atomic.Int32
			th.onStop = func() { cleanup.Add(1) }
			rt := newRealtimeThinker(context.Background(), th, &fakeRealtimeProvider{}, "", nil, nil, nil)
			rt.replaceSession(newFakeRealtimeSession())
			defer rt.cancel()
			release := make(chan struct{})
			if !th.startRunLoop(func() {
				<-release
				if realtime {
					rt.Run()
				} else {
					th.Run()
				}
			}) {
				t.Fatal("run not scheduled")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- th.Shutdown(ctx) }()
			select {
			case err := <-done:
				t.Fatalf("shutdown skipped scheduled loop: %v", err)
			case <-time.After(20 * time.Millisecond):
			}
			close(release)
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-ctx.Done():
				t.Fatal("scheduled loop did not stop")
			}
			want := int32(0)
			if realtime {
				want = 1
			}
			if cleanup.Load() != want {
				t.Fatalf("terminal cleanup count=%d want %d", cleanup.Load(), want)
			}
			if th.startRunLoop(func() { t.Error("started work after shutdown") }) {
				t.Fatal("shutdown accepted another run")
			}
		})
	}
}

func TestShutdownJoinsInFlightParentReportWrite(t *testing.T) {
	th := newTestThinker()
	th.config.path = filepath.Join(t.TempDir(), "config.json")
	th.parentReportPending = map[string]PendingParentReport{
		"pending": {ParentID: "missing", Event: PersistentThreadEvent{ID: "pending", Text: "report"}},
	}
	// Hold persistence so the real retry worker is in an in-flight write.
	th.config.saveMu.Lock()
	locked := true
	defer func() {
		if locked {
			th.config.saveMu.Unlock()
		}
	}()
	th.startParentReportRetry()
	deadline := time.Now().Add(time.Second)
	for th.parentReportMu.TryLock() {
		th.parentReportMu.Unlock()
		if time.Now().After(deadline) {
			t.Fatal("report worker did not start")
		}
		time.Sleep(time.Millisecond)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- th.Shutdown(ctx) }()
	select {
	case err := <-done:
		t.Fatalf("shutdown returned before report persistence finished: %v", err)
	case <-time.After(20 * time.Millisecond):
	}
	th.config.saveMu.Unlock()
	locked = false
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("shutdown did not join report worker")
	}
}
