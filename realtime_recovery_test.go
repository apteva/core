package core

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type recoveryTestSession struct {
	*fakeRealtimeSession
	closed      sync.Once
	audio       []byte
	resumeNext  RealtimeSession
	resumeErr   error
	resumeCalls int
}

func newRecoveryTestSession() *recoveryTestSession {
	return &recoveryTestSession{fakeRealtimeSession: newFakeRealtimeSession()}
}
func (s *recoveryTestSession) Close() error { s.closed.Do(func() { close(s.events) }); return nil }
func (s *recoveryTestSession) SendAudio(pcm []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.audio = append(s.audio, pcm...)
	return nil
}
func (s *recoveryTestSession) Resume(_ context.Context, _ RealtimeSessionOpts) (RealtimeSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resumeCalls++
	if s.resumeErr != nil {
		return nil, s.resumeErr
	}
	if s.resumeNext == nil {
		return nil, ErrRealtimeResumeUnavailable
	}
	return s.resumeNext, nil
}

type recoveryTestProvider struct {
	fakeRealtimeProvider
	next []RealtimeSession
}

type blockingRecoveryProvider struct {
	fakeRealtimeProvider
	first   RealtimeSession
	entered chan struct{}
	count   atomic.Int32
}

func (p *blockingRecoveryProvider) Open(ctx context.Context, _ RealtimeSessionOpts) (RealtimeSession, error) {
	if p.count.Add(1) == 1 {
		return p.first, nil
	}
	close(p.entered)
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestRealtimeRecoveryStopCancelsPendingOpen(t *testing.T) {
	first := newRecoveryTestSession()
	p := &blockingRecoveryProvider{first: first, entered: make(chan struct{})}
	thinker := newTestThinker()
	rt, err := startRealtimeThinker(context.Background(), thinker, p, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.cancel()
	rt.audioBridgeConnected()
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	_ = first.Close()
	select {
	case <-p.entered:
	case <-time.After(time.Second):
		t.Fatal("reopen did not start")
	}
	thinker.Stop()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Stop did not cancel provider Open")
	}
}

func (p *recoveryTestProvider) Open(_ context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.opens = append(p.opens, opts)
	if len(p.next) == 0 {
		return nil, errors.New("test provider unavailable")
	}
	s := p.next[0]
	p.next = p.next[1:]
	return s, nil
}

func waitRecovery(t *testing.T, predicate func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !predicate() {
		if time.Now().After(deadline) {
			t.Fatal("recovery condition not reached")
		}
		time.Sleep(time.Millisecond)
	}
}
func runRecoveryThinker(t *testing.T, rt *RealtimeThinker) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	t.Cleanup(func() {
		rt.cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("recovery did not cancel")
		}
	})
}
func recoveryOpenCount(p *recoveryTestProvider) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.opens)
}

func TestRealtimeRecoveryIdleWaitsAndWakesWithoutLosingInput(t *testing.T) {
	for _, wake := range []string{"text", "audio", "bridge"} {
		t.Run(wake, func(t *testing.T) {
			first, second := newRecoveryTestSession(), newRecoveryTestSession()
			p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
			audio := make(chan []byte, 4)
			rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", audio, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			runRecoveryThinker(t, rt)
			_ = first.Close()
			waitRecovery(t, func() bool { rt.stateMu.Lock(); defer rt.stateMu.Unlock(); return rt.state == "waiting" })
			if recoveryOpenCount(p) != 1 {
				t.Fatal("unused thread reopened")
			}
			switch wake {
			case "text":
				rt.bus.Publish(Event{Type: EventInbox, To: "main", Text: "Remember the caller's appointment."})
			case "audio":
				audio <- []byte{1, 2, 3, 4}
			case "bridge":
				rt.audioBridgeConnected()
			}
			waitRecovery(t, func() bool { return recoveryOpenCount(p) == 2 && rt.currentSession() == second })
			waitRecovery(t, func() bool {
				second.mu.Lock()
				defer second.mu.Unlock()
				switch wake {
				case "text":
					return len(second.texts) == 1 && strings.Contains(second.texts[0].Content, "appointment")
				case "audio":
					return bytes.Equal(second.audio, []byte{1, 2, 3, 4})
				}
				return true
			})
		})
	}
}

func TestRealtimeRecoveryResumesOrRestoresOnce(t *testing.T) {
	for _, reject := range []bool{false, true} {
		t.Run(map[bool]string{false: "resume", true: "rejected_handle"}[reject], func(t *testing.T) {
			first, second := newRecoveryTestSession(), newRecoveryTestSession()
			if reject {
				first.resumeErr = errors.New("checkpoint rejected")
			} else {
				first.resumeNext = second
			}
			p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
			rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			rt.appendTranscript("user", "The verification word is apricot.")
			rt.appendTranscript("assistant", "Understood.")
			rt.audioBridgeConnected()
			runRecoveryThinker(t, rt)
			_ = first.Close()
			waitRecovery(t, func() bool { return rt.currentSession() == second })
			second.mu.Lock()
			defer second.mu.Unlock()
			if reject {
				if len(second.restored) != 1 || len(second.restored[0]) != 2 {
					t.Fatal("fresh fallback lost history")
				}
			} else if len(second.restored) != 0 {
				t.Fatal("native resume replayed history")
			}
			if len(second.texts) != 0 {
				t.Fatal("completed conversation generated a new prompt")
			}
			want := 1
			if reject {
				want = 2
			}
			if recoveryOpenCount(p) != want {
				t.Fatal("unexpected extra fresh open")
			}
		})
	}
}

func TestRealtimeRecoveryToolCompletionSurvivesDisconnectAndCannotRepeat(t *testing.T) {
	for _, point := range []string{"before_execution", "during_execution", "after_external_success", "after_result_delivery"} {
		t.Run(point, func(t *testing.T) {
			thinker := newTestThinker()
			thinker.registry = NewToolRegistry("")
			thinker.registry.Register(&ToolDef{Name: "book", Description: "Book a callback", InputSchema: map[string]any{"type": "object"}})
			thinker.toolAllowlist = map[string]bool{"book": true}
			thinker.recordPresentedTools([]NativeTool{{Name: "book"}})
			var calls atomic.Int32
			var completedActions atomic.Int32
			thinker.handleTools = func(_ *Thinker, _ []toolCall, _ []string) ([]string, []string, []ToolResult) {
				calls.Add(1)
				return nil, nil, nil
			}
			first, second := newRecoveryTestSession(), newRecoveryTestSession()
			p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
			rt, err := startRealtimeThinker(context.Background(), thinker, p, "", nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			rt.audioBridgeConnected()
			if point != "before_execution" {
				runRecoveryThinker(t, rt)
			}
			first.events <- RealtimeEvent{Type: RealtimeEventToolCall, ResponseID: "response", ToolCallID: "booking-1", ToolName: "book", ToolArgs: `{"slot":"Monday"}`}
			first.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "response"}
			if point == "before_execution" {
				_ = first.Close()
				runRecoveryThinker(t, rt)
			}
			waitRecovery(t, func() bool { return calls.Load() == 1 })
			result := Event{Type: EventInbox, To: "main", ToolResult: &ToolResult{CallID: "booking-1", ToolName: "book", Content: `{"booking_id":"confirmed-123"}`}}
			if point == "after_external_success" || point == "after_result_delivery" {
				completedActions.Add(1)
			}
			if point == "after_result_delivery" {
				rt.bus.Publish(result)
				waitRecovery(t, func() bool { first.mu.Lock(); defer first.mu.Unlock(); return len(first.toolResults) == 1 })
			}
			_ = first.Close()
			waitRecovery(t, func() bool { return rt.currentSession() != first })
			if point != "after_result_delivery" {
				if recoveryOpenCount(p) != 1 {
					t.Fatal("opened a new model while an operation was unresolved")
				}
				if point == "before_execution" || point == "during_execution" {
					completedActions.Add(1)
				}
				rt.bus.Publish(result)
			}
			waitRecovery(t, func() bool { return rt.currentSession() == second })
			second.mu.Lock()
			history := ""
			for _, batch := range second.restored {
				for _, m := range batch {
					history += m.Content
				}
			}
			second.mu.Unlock()
			if !strings.Contains(history, "confirmed-123") {
				t.Fatal("completed operation missing from fallback history")
			}
			second.events <- RealtimeEvent{Type: RealtimeEventToolCall, ResponseID: "new-response", ToolCallID: "booking-2", ToolName: "book", ToolArgs: `{"slot":"Monday"}`}
			waitRecovery(t, func() bool { second.mu.Lock(); defer second.mu.Unlock(); return len(second.toolResults) == 1 })
			if calls.Load() != 1 || completedActions.Load() != 1 {
				t.Fatal("operation repeated after recovery")
			}
			second.mu.Lock()
			if !second.toolResults[0].isError {
				t.Error("duplicate operation was not blocked")
			}
			second.mu.Unlock()
			rt.bus.Publish(result) // A delayed duplicate completion must be harmless.
			if recoveryOpenCount(p) != 2 {
				t.Fatal("stale completion triggered another open")
			}
			rt.bus.Publish(Event{Type: EventInbox, To: "main", From: "api", Text: "Please create another booking for Tuesday."})
			waitRecovery(t, func() bool {
				second.mu.Lock()
				defer second.mu.Unlock()
				for _, m := range second.texts {
					if strings.Contains(m.Content, "Tuesday") {
						return true
					}
				}
				return false
			})
			second.events <- RealtimeEvent{Type: RealtimeEventToolCall, ResponseID: "authorized-next-turn", ToolCallID: "booking-3", ToolName: "book", ToolArgs: `{"slot":"Tuesday"}`}
			waitRecovery(t, func() bool { return calls.Load() == 2 })
		})
	}
}

func TestRealtimeRecoveryGoAwayDoesNotReopenAnUnusedThread(t *testing.T) {
	first := newRecoveryTestSession()
	p := &recoveryTestProvider{next: []RealtimeSession{first}}
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	runRecoveryThinker(t, rt)
	first.events <- RealtimeEvent{Type: RealtimeEventSessionExpiring, TimeLeft: time.Second}
	waitRecovery(t, func() bool { rt.stateMu.Lock(); defer rt.stateMu.Unlock(); return rt.state == "waiting" })
	if recoveryOpenCount(p) != 1 {
		t.Fatal("GoAway reopened an unused thread")
	}
}

func TestRealtimeRecoveryGoAwayWaitsForTurn(t *testing.T) {
	first, second := newRecoveryTestSession(), newRecoveryTestSession()
	p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	rt.responseStarted()
	runRecoveryThinker(t, rt)
	first.events <- RealtimeEvent{Type: RealtimeEventSessionExpiring, TimeLeft: time.Hour}
	waitRecovery(t, func() bool { rt.recovery.Lock(); defer rt.recovery.Unlock(); return !rt.recovery.deadline.IsZero() })
	if recoveryOpenCount(p) != 1 {
		t.Fatal("planned renewal interrupted a turn")
	}
	first.events <- RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "finished"}
	waitRecovery(t, func() bool { return rt.currentSession() == second })
}

func TestRealtimeRecoveryBackoffSurvivesSocketOpensAndCancels(t *testing.T) {
	var retry realtimeRecoveryRetry
	for i, want := range []time.Duration{time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, time.Minute, time.Minute} {
		if got := retry.failed(); got != want {
			t.Fatalf("attempt %d delay=%v want=%v", i, got, want)
		}
	}
	first, second := newRecoveryTestSession(), newRecoveryTestSession()
	p := &recoveryTestProvider{next: []RealtimeSession{first, second}}
	rt, err := startRealtimeThinker(context.Background(), newTestThinker(), p, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	rt.audioBridgeConnected()
	runRecoveryThinker(t, rt)
	_ = first.Close()
	waitRecovery(t, func() bool { return rt.currentSession() == second })
	_ = second.Close()
	waitRecovery(t, func() bool { return rt.currentSession() == nil })
	rt.recovery.Lock()
	delay := time.Until(rt.recovery.retryAt)
	failures := rt.recovery.retry.failures
	rt.recovery.Unlock()
	if failures < 2 || delay <= 0 {
		t.Fatal("successful socket open reset the failure budget")
	}
}
