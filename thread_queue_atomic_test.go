package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func newQueueAdmissionFixture(t *testing.T) (*Thinker, *Thread) {
	t.Helper()
	parent := newTestThinker()
	dir := t.TempDir()
	parent.session = NewSession(dir, "main")
	parent.config.path = filepath.Join(dir, "config.json")
	if err := parent.threads.SpawnWithOpts("worker", "Original profile", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	path := parent.config.path
	t.Cleanup(func() {
		parent.config.path = path
		parent.threads.cleanupThread("worker")
		parent.Stop()
	})
	return parent, parent.threads.threads["worker"]
}

func queuePersistenceBytes(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	return data
}

func TestAssignmentAdmissionFailurePreservesAcceptedWork(t *testing.T) {
	for _, failure := range []string{"persistence", "capacity", "conflicting-retry"} {
		t.Run(failure, func(t *testing.T) {
			parent, thread := newQueueAdmissionFixture(t)
			old := ThreadWork{ID: "current", Revision: 1, Origin: "main", Kind: "request"}
			if err := parent.threads.QueueWork("worker", old, "Finish accepted work", nil); err != nil {
				t.Fatal(err)
			}
			thread.Thinker.sub.DrainTargeted()
			thread.Thinker.work.observe(&old) // The accepted assignment is executing.
			newer := ThreadWork{ID: "replacement", Revision: 1, Origin: "main", Kind: "request"}
			text := "Replacement work"
			path := parent.config.path
			switch failure {
			case "persistence":
				blocker := filepath.Join(t.TempDir(), "not-a-directory")
				if err := os.WriteFile(blocker, []byte("block"), 0600); err != nil {
					t.Fatal(err)
				}
				parent.config.path = filepath.Join(blocker, "config.json")
			case "capacity":
				fill := make([]PersistentThreadEvent, 4095)
				for i := range fill {
					fill[i] = PersistentThreadEvent{ID: fmt.Sprintf("fill-%d", i), Text: "queued", Hash: threadEventHash("queued", nil)}
				}
				if _, err := parent.threads.QueueEvents("worker", fill); err != nil {
					t.Fatal(err)
				}
				thread.Thinker.sub.DrainTargeted()
			case "conflicting-retry":
				newer = old
				text = "Different payload for the accepted assignment"
			}
			before := thread.pendingInboxEvents()
			configBefore, journalBefore := queuePersistenceBytes(t, path), queuePersistenceBytes(t, path+".runtime.jsonl")
			err := parent.threads.QueueWork("worker", newer, text, nil)
			parent.config.path = path
			if err == nil {
				t.Fatal("replacement was not rejected")
			}
			if !thread.Thinker.work.current(old) || !thread.Thinker.work.ready(old) || thread.Thinker.work.snapshot() != old {
				t.Error("rejected admission superseded the accepted assignment")
			}
			if !reflect.DeepEqual(before, thread.pendingInboxEvents()) || len(thread.Thinker.sub.DrainTargeted()) != 0 {
				t.Error("rejected admission changed or published inbox events")
			}
			if string(configBefore) != string(queuePersistenceBytes(t, path)) || string(journalBefore) != string(queuePersistenceBytes(t, path+".runtime.jsonl")) {
				t.Error("rejected admission changed durable queue state")
			}
		})
	}
}

func TestAssignmentAdmissionCommitsBeforeDeliveryAndIgnoresOldRetry(t *testing.T) {
	parent, thread := newQueueAdmissionFixture(t)
	old := ThreadWork{ID: "old", Revision: 1, Origin: "main", Kind: "request"}
	newer := ThreadWork{ID: "new", Revision: 1, Origin: "main", Kind: "request"}
	if err := parent.threads.QueueWork("worker", old, "old work", nil); err != nil {
		t.Fatal(err)
	}
	thread.Thinker.sub.DrainTargeted()
	// Work admitted through the generic inbox must have the same semantics as
	// QueueWork; callers must not update revisions outside the queue transaction.
	if _, err := parent.threads.QueueEvents("worker", []PersistentThreadEvent{newThreadWorkEvent(newer, "new work", nil)}); err != nil {
		t.Fatal(err)
	}
	if !thread.Thinker.work.current(newer) || thread.Thinker.work.current(old) {
		t.Fatal("committed assignment was not current before event delivery")
	}
	events := thread.Thinker.sub.DrainTargeted()
	if len(events) != 1 || events[0].Work == nil || *events[0].Work != newer {
		t.Fatal("committed assignment was not delivered exactly once")
	}
	restored := &Config{path: parent.config.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	stored, ok := persistentThreadByID(restored.GetThreads(), "worker")
	if !ok || len(stored.Events) != 2 || stored.Events[1].Work == nil || *stored.Events[1].Work != newer {
		t.Fatal("delivered assignment was not durable")
	}
	if err := parent.threads.QueueWork("worker", old, "old work", nil); err != nil {
		t.Fatal(err)
	}
	if !thread.Thinker.work.current(newer) || len(thread.Thinker.sub.DrainTargeted()) != 0 {
		t.Fatal("duplicate old assignment rewound admission or was delivered again")
	}
}

func TestThreadQueueLifecycleConflictRejectsWholeBatch(t *testing.T) {
	for _, queue := range []string{"main", "child", "profile"} {
		for _, trackedFirst := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/tracked-first=%t", queue, trackedFirst), func(t *testing.T) {
				parent, thread := newQueueAdmissionFixture(t)
				first := PersistentThreadEvent{ID: "same-id", Text: "same payload", Hash: threadEventHash("same payload", nil), TrackLifecycle: trackedFirst}
				conflict := first
				conflict.TrackLifecycle = !trackedFirst
				incoming := []PersistentThreadEvent{{ID: "fresh", Text: "must not be partially accepted", Hash: threadEventHash("must not be partially accepted", nil)}, first, conflict}
				before, _ := json.Marshal(parent.config)
				path := parent.config.path
				configBefore, journalBefore := queuePersistenceBytes(t, path), queuePersistenceBytes(t, path+".runtime.jsonl")
				var err error
				switch queue {
				case "main":
					_, err = parent.QueueMainEvents(incoming)
				case "child":
					_, err = parent.threads.QueueEvents("worker", incoming)
				case "profile":
					_, err = parent.threads.UpdateWithEvents("worker", "", "Changed profile", nil, ThreadUpdateOptions{}, incoming)
				}
				var conflictErr *ThreadEventConflictError
				if !errors.As(err, &conflictErr) || conflictErr.ID != first.ID {
					t.Errorf("expected lifecycle conflict, got %v", err)
				}
				after, _ := json.Marshal(parent.config)
				if string(after) != string(before) || len(parent.mainInboxEvents) != 0 || len(thread.pendingInboxEvents()) != 0 || thread.Directive != "Original profile" {
					t.Error("conflicting batch mutated profile, inbox, or execution state")
				}
				if len(parent.sub.DrainTargeted()) != 0 || len(thread.Thinker.sub.DrainTargeted()) != 0 {
					t.Error("conflicting batch published an event")
				}
				if string(configBefore) != string(queuePersistenceBytes(t, path)) || string(journalBefore) != string(queuePersistenceBytes(t, path+".runtime.jsonl")) {
					t.Error("conflicting batch changed durable state")
				}
			})
		}
	}
}

func TestThreadQueueMatchingLifecycleDuplicatesKeepOneExecution(t *testing.T) {
	for _, queue := range []string{"main", "child"} {
		for _, tracked := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/tracked=%t", queue, tracked), func(t *testing.T) {
				parent, _ := newQueueAdmissionFixture(t)
				enqueue := func(events []PersistentThreadEvent) (ThreadEventQueueResult, error) {
					if queue == "main" {
						return parent.QueueMainEvents(events)
					}
					return parent.threads.QueueEvents("worker", events)
				}
				event := PersistentThreadEvent{ID: "repeat", Text: "same payload", Hash: threadEventHash("same payload", nil), TrackLifecycle: tracked}
				first, err := enqueue([]PersistentThreadEvent{event, event})
				if err != nil || len(first.Accepted) != 1 || len(first.Duplicate) != 1 {
					t.Fatalf("matching duplicates were not accepted once: %+v %v", first, err)
				}
				retry, err := enqueue([]PersistentThreadEvent{event})
				if err != nil || len(retry.Accepted) != 0 || len(retry.Duplicate) != 1 {
					t.Fatalf("retry was not idempotent: %+v %v", retry, err)
				}
				if tracked {
					if first.Executions[event.ID] == "" || first.Executions[event.ID] != retry.Executions[event.ID] || len(parent.config.EventExecutions) != 1 {
						t.Fatal("tracked duplicates did not retain one stable execution ID")
					}
				} else if len(first.Executions) != 0 || len(retry.Executions) != 0 || len(parent.config.EventExecutions) != 0 {
					t.Fatal("untracked duplicates created execution state")
				}
			})
		}
	}
}
