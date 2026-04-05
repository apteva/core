package main

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEventBus_TargetedDelivery(t *testing.T) {
	bus := NewEventBus()
	sub1 := bus.Subscribe("alice", 10)
	sub2 := bus.Subscribe("bob", 10)

	bus.Publish(Event{Type: EventInbox, To: "alice", Text: "hello alice"})
	bus.Publish(Event{Type: EventInbox, To: "bob", Text: "hello bob"})

	time.Sleep(10 * time.Millisecond)

	// Alice should have 1 event
	select {
	case ev := <-sub1.C:
		if ev.Text != "hello alice" {
			t.Errorf("alice got %q", ev.Text)
		}
	default:
		t.Error("alice got nothing")
	}

	// Bob should have 1 event
	select {
	case ev := <-sub2.C:
		if ev.Text != "hello bob" {
			t.Errorf("bob got %q", ev.Text)
		}
	default:
		t.Error("bob got nothing")
	}
}

func TestEventBus_BroadcastOnlyToObservers(t *testing.T) {
	bus := NewEventBus()
	sub1 := bus.Subscribe("alice", 10)
	sub2 := bus.Subscribe("bob", 10)
	obs := bus.SubscribeAll("watcher", 10)

	// Alice broadcasts — regular subscribers should NOT get it, only observers
	bus.Publish(Event{Type: EventThinkDone, From: "alice", To: ""})

	time.Sleep(10 * time.Millisecond)

	// Alice should have nothing
	select {
	case ev := <-sub1.C:
		t.Errorf("alice got broadcast: %+v", ev)
	default:
	}

	// Bob should have nothing (regular subscriber, not observer)
	select {
	case ev := <-sub2.C:
		t.Errorf("bob got broadcast: %+v", ev)
	default:
	}

	// Observer should have it
	select {
	case ev := <-obs.C:
		if ev.Type != EventThinkDone {
			t.Errorf("observer got wrong type: %s", ev.Type)
		}
	default:
		t.Error("observer should have received broadcast")
	}
}

func TestEventBus_WakeOnlyForTargeted(t *testing.T) {
	bus := NewEventBus()
	sub := bus.Subscribe("main", 10)

	// Broadcast should NOT wake
	bus.Publish(Event{Type: EventThinkDone, From: "worker", To: ""})
	time.Sleep(10 * time.Millisecond)

	select {
	case <-sub.Wake:
		t.Error("broadcast should not wake targeted subscriber")
	default:
		// correct
	}

	// Targeted event SHOULD wake
	bus.Publish(Event{Type: EventInbox, From: "worker", To: "main", Text: "hello"})
	time.Sleep(10 * time.Millisecond)

	select {
	case <-sub.Wake:
		// correct
	default:
		t.Error("targeted event should wake subscriber")
	}
}

func TestEventBus_ObserverSeesAll(t *testing.T) {
	bus := NewEventBus()
	bus.Subscribe("main", 10)
	obs := bus.SubscribeAll("tui", 10)

	bus.Publish(Event{Type: EventInbox, To: "main", Text: "targeted"})
	bus.Publish(Event{Type: EventThinkDone, From: "main", To: ""})

	time.Sleep(10 * time.Millisecond)

	count := 0
	for {
		select {
		case <-obs.C:
			count++
		default:
			goto done
		}
	}
done:
	if count != 2 {
		t.Errorf("observer should see 2 events, got %d", count)
	}
}

// This is the critical test: simulates a sub-thread completing and
// verifies that main receives the done message via the bus.
func TestThreadDone_MainReceivesEvent(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	// Spawn a thread
	err := thinker.threads.Spawn("worker", "test worker", nil)
	if err != nil {
		t.Fatal(err)
	}

	// Verify thread exists
	if thinker.threads.Count() != 1 {
		t.Fatalf("expected 1 thread, got %d", thinker.threads.Count())
	}

	// Drain the "started" inbox event from main's subscription
	time.Sleep(50 * time.Millisecond)
	startEvents := thinker.drainEventTexts()
	t.Logf("startup events on main: %v", startEvents)

	foundStarted := false
	for _, ev := range startEvents {
		if strings.HasPrefix(ev, "[thread:worker] started") {
			foundStarted = true
		}
	}
	if !foundStarted {
		t.Errorf("main should have received [thread:worker] started, got: %v", startEvents)
	}

	// Clear stale wake
	select {
	case <-thinker.sub.Wake:
	default:
	}

	// Now simulate the thread calling [[done]] by directly invoking the tool handler
	thinker.threads.mu.RLock()
	thread := thinker.threads.threads["worker"]
	thinker.threads.mu.RUnlock()

	if thread == nil {
		t.Fatal("thread not found")
	}

	// Call the thread's tool handler with a done call
	calls := parseToolCalls(`[[done message="task complete"]]`)
	t.Logf("parsed calls: %v", calls)

	thread.Thinker.handleTools(thread.Thinker, calls, nil)

	// Wait for cleanup — thread's Run() goroutine needs to notice quit and call onStop
	deadline := time.Now().Add(5 * time.Second)
	for thinker.threads.Count() != 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}

	// Check 1: Thread should be removed
	if thinker.threads.Count() != 0 {
		t.Errorf("expected 0 threads after done, got %d", thinker.threads.Count())
	}

	// Check 2: Main's subscription should have the done message
	doneEvents := thinker.drainEventTexts()
	t.Logf("done events on main: %v", doneEvents)

	foundDone := false
	for _, ev := range doneEvents {
		t.Logf("  event: %q", ev)
		if ev == "[thread:worker done] task complete" {
			foundDone = true
		}
	}
	if !foundDone {
		t.Errorf("main should have received [thread:worker done] task complete, got: %v", doneEvents)
	}

	// Wake was consumed by drainEvents — that's correct.
	// The important thing is the event was delivered (checked above).
}

// Test that multiple events don't cause spurious wakes after drain
func TestDrainEventsConsumesWake(t *testing.T) {
	bus := NewEventBus()
	thinker := &Thinker{
		bus:      bus,
		sub:      bus.Subscribe("main", 10),
		threadID: "main",
	}

	// Publish 3 targeted events rapidly
	bus.Publish(Event{Type: EventInbox, To: "main", Text: "msg1"})
	bus.Publish(Event{Type: EventInbox, To: "main", Text: "msg2"})
	bus.Publish(Event{Type: EventInbox, To: "main", Text: "msg3"})

	time.Sleep(10 * time.Millisecond)

	// drainEvents should consume both events AND wake signals
	items := thinker.drainEventTexts()
	if len(items) != 3 {
		t.Fatalf("expected 3 events, got %d", len(items))
	}

	// Wake should be empty — consumed by drainEvents
	select {
	case <-thinker.sub.Wake:
		t.Error("spurious wake after drainEvents")
	default:
		// correct
	}
}

// End-to-end: spawn thread, thread sends to main, main receives
func TestThreadSendToMain(t *testing.T) {
	thinker := newTestThinkerFull()
	defer thinker.Stop()

	thinker.threads.Spawn("reporter", "test", nil)
	defer thinker.threads.Kill("reporter")

	// Drain startup events
	time.Sleep(50 * time.Millisecond)
	thinker.drainEventTexts()
	select {
	case <-thinker.sub.Wake:
	default:
	}

	// Reporter sends to main via bus
	thinker.bus.Publish(Event{Type: EventInbox, To: "main", Text: "[from:reporter] job done"})

	time.Sleep(10 * time.Millisecond)

	events := thinker.drainEventTexts()
	t.Logf("events: %v", events)

	if len(events) != 1 || events[0] != "[from:reporter] job done" {
		t.Errorf("expected [from:reporter] job done, got: %v", events)
	}
	// Wake was consumed by drainEvents — correct behavior.
}

// Simulates the exact race: event arrives while thinker is in think() call,
// then verify drainEvents picks it up on the next iteration.
func TestWakeDuringThink(t *testing.T) {
	bus := NewEventBus()
	sub := bus.Subscribe("main", 100)

	// Simulate: drain events (empty), clear wake
	items := func() []string {
		var out []string
		for {
			select {
			case ev := <-sub.C:
				if ev.Type == EventInbox {
					out = append(out, ev.Text)
				}
			default:
				return out
			}
		}
	}()
	if len(items) != 0 {
		t.Fatal("should start empty")
	}
	select {
	case <-sub.Wake:
	default:
	}

	// Now simulate: event arrives during think() — goes into sub.C + signals Wake
	bus.Publish(Event{Type: EventInbox, To: "main", Text: "[thread:worker done] finished"})
	time.Sleep(10 * time.Millisecond)

	// Wake should be signaled
	select {
	case <-sub.Wake:
		t.Log("Wake fired — correct, thinker would wake from sleep")
	default:
		t.Fatal("Wake should have been signaled")
	}

	// Next iteration: drain events should find the message
	items = func() []string {
		var out []string
		for {
			select {
			case ev := <-sub.C:
				if ev.Type == EventInbox {
					out = append(out, ev.Text)
				}
			default:
				return out
			}
		}
	}()
	if len(items) != 1 || items[0] != "[thread:worker done] finished" {
		t.Errorf("expected done message, got: %v", items)
	}
	t.Log("drain found the event — correct")
}

// Stress test: many concurrent publishes, nothing blocks
func TestEventBus_ConcurrentPublish(t *testing.T) {
	bus := NewEventBus()
	bus.Subscribe("main", 5) // small buffer

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			bus.Publish(Event{Type: EventInbox, To: "main", Text: fmt.Sprintf("msg %d", n)})
		}(i)
	}

	// Must complete without hanging
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// correct — nothing blocked
	case <-time.After(2 * time.Second):
		t.Fatal("concurrent publish blocked — should never happen")
	}
}
