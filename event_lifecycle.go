package core

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Event lifecycle tracking is deliberately separate from conversation history
// and ordinary telemetry. History is bounded model context; telemetry is a
// bounded observability stream. These records are the durable delivery state
// for callers that explicitly request tracking on a stable inbox event.
const (
	eventExecutionPending = "pending"
	eventClaimed          = "event.claimed"
	eventActive           = "event.active"
	eventSettled          = "event.settled"
	eventError            = "event.error"
)

type PersistentEventExecution struct {
	EventID           string          `json:"event_id"`
	ExecutionID       string          `json:"execution_id"`
	ThreadID          string          `json:"thread_id"`
	ParentExecutionID string          `json:"parent_execution_id,omitempty"`
	Status            string          `json:"status"`
	Reason            string          `json:"reason,omitempty"`
	Sequence          uint64          `json:"sequence"`
	Participants      map[string]bool `json:"participants,omitempty"` // true while that thread is processing
	CreatedAt         time.Time       `json:"created_at"`
	UpdatedAt         time.Time       `json:"updated_at"`
}

// EventLifecycleTransition is returned by Core's durable lifecycle outbox.
// ID is stable across retries; consumers acknowledge IDs only after their own
// durable write. Timestamp records the transition time rather than delivery.
type EventLifecycleTransition struct {
	ID                string    `json:"id"`
	Type              string    `json:"type"`
	EventID           string    `json:"event_id"`
	ExecutionID       string    `json:"execution_id"`
	ThreadID          string    `json:"thread_id"`
	ParentExecutionID string    `json:"parent_execution_id,omitempty"`
	Timestamp         time.Time `json:"timestamp"`
	Reason            string    `json:"reason,omitempty"`
	Sequence          uint64    `json:"sequence"`
}

func prepareTrackedEventExecutions(events []PersistentThreadEvent) {
	for i := range events {
		if !events[i].TrackLifecycle || events[i].ExecutionID != "" {
			continue
		}
		events[i].ExecutionID = "exe_" + newULID()
	}
}

func registerEventExecutionsLocked(c *Config, threadID string, events []PersistentThreadEvent) {
	now := time.Now().UTC()
	for _, event := range events {
		if !event.TrackLifecycle || event.ExecutionID == "" {
			continue
		}
		found := false
		for _, existing := range c.EventExecutions {
			if existing.ExecutionID == event.ExecutionID {
				found = true
				break
			}
		}
		if found {
			continue
		}
		c.EventExecutions = append(c.EventExecutions, PersistentEventExecution{
			EventID: event.ID, ExecutionID: event.ExecutionID, ThreadID: threadID,
			Status: eventExecutionPending, Participants: map[string]bool{threadID: false},
			CreatedAt: now, UpdatedAt: now,
		})
	}
}

func upsertPersistentThreadLocked(c *Config, pt PersistentThread) {
	for i := range c.Threads {
		if c.Threads[i].ID == pt.ID {
			c.Threads[i] = pt
			return
		}
	}
	c.Threads = append(c.Threads, pt)
}

func (c *Config) saveThreadAndRegisterEventExecutions(pt PersistentThread, accepted []PersistentThreadEvent) error {
	c.mu.RLock()
	exists := false
	for _, thread := range c.Threads {
		if thread.ID == pt.ID {
			exists = true
			break
		}
	}
	c.mu.RUnlock()
	if exists {
		return c.updateRuntime(func() {
			for i := range c.Threads {
				if c.Threads[i].ID == pt.ID {
					c.Threads[i].Events = clonePersistentThreadEvents(pt.Events)
					break
				}
			}
			registerEventExecutionsLocked(c, pt.ID, accepted)
		})
	}
	return c.update(func() {
		upsertPersistentThreadLocked(c, pt)
		registerEventExecutionsLocked(c, pt.ID, accepted)
	})
}

func (c *Config) saveMainEventsAndRegisterExecutions(events, accepted []PersistentThreadEvent) error {
	return c.updateRuntime(func() {
		c.MainEvents = clonePersistentThreadEvents(events)
		registerEventExecutionsLocked(c, "main", accepted)
	})
}

func (c *Config) saveMainEvents(events []PersistentThreadEvent) error {
	return c.updateRuntime(func() { c.MainEvents = clonePersistentThreadEvents(events) })
}

func (c *Config) getMainEvents() []PersistentThreadEvent {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return clonePersistentThreadEvents(c.MainEvents)
}

// EventLifecycle owns the durable state machine. It observes existing Core
// boundaries; it never schedules a model turn, changes pace, or calls a tool.
type EventLifecycle struct {
	config    *Config
	telemetry *Telemetry
	mu        sync.Mutex
}

func NewEventLifecycle(config *Config, telemetry *Telemetry) *EventLifecycle {
	return &EventLifecycle{config: config, telemetry: telemetry}
}

func (l *EventLifecycle) emit(transitions []EventLifecycleTransition) {
	if l == nil || l.telemetry == nil {
		return
	}
	for _, transition := range transitions {
		l.telemetry.Emit(transition.Type, transition.ThreadID, transition)
	}
}

func appendLifecycleTransition(c *Config, execution *PersistentEventExecution, eventType, reason string, now time.Time) EventLifecycleTransition {
	execution.Sequence++
	execution.Status = eventType
	execution.Reason = reason
	execution.UpdatedAt = now
	transition := EventLifecycleTransition{
		ID:   fmt.Sprintf("%s:%d", execution.ExecutionID, execution.Sequence),
		Type: eventType, EventID: execution.EventID, ExecutionID: execution.ExecutionID,
		ThreadID: execution.ThreadID, ParentExecutionID: execution.ParentExecutionID,
		Timestamp: now, Reason: reason, Sequence: execution.Sequence,
	}
	c.EventLifecycleOutbox = append(c.EventLifecycleOutbox, transition)
	return transition
}

func executionTerminal(status string) bool {
	return status == eventSettled || status == eventError
}

// Claim records the crash-safe handoff from durable inbox history into active
// processing. claimed and active are separate ordered transitions, but are
// persisted in one atomic configuration write.
func (l *EventLifecycle) Claim(threadID string, executionIDs []string) error {
	if l == nil || len(executionIDs) == 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now().UTC()
	var emitted []EventLifecycleTransition
	err := l.config.updateRuntime(func() {
		for i := range l.config.EventExecutions {
			execution := &l.config.EventExecutions[i]
			if !containsString(executionIDs, execution.ExecutionID) || executionTerminal(execution.Status) {
				continue
			}
			if execution.Participants == nil {
				execution.Participants = map[string]bool{}
			}
			execution.Participants[threadID] = true
			if execution.Status == eventExecutionPending {
				emitted = append(emitted, appendLifecycleTransition(l.config, execution, eventClaimed, "event_history_persisted", now))
				emitted = append(emitted, appendLifecycleTransition(l.config, execution, eventActive, "processing_started", now))
			} else {
				execution.UpdatedAt = now
			}
		}
	})
	if err == nil {
		l.emit(emitted)
	}
	return err
}

// Propagate marks a causally messaged or spawned thread as a participant in
// the same execution. No new model work is created by this bookkeeping.
func (l *EventLifecycle) Propagate(executionIDs []string, targetThreadID string) error {
	if l == nil || len(executionIDs) == 0 || targetThreadID == "" {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	// Repeated messages to an already active participant do not change durable state.
	l.config.mu.RLock()
	changed := false
	for _, execution := range l.config.EventExecutions {
		if containsString(executionIDs, execution.ExecutionID) && !executionTerminal(execution.Status) && !execution.Participants[targetThreadID] {
			changed = true
			break
		}
	}
	l.config.mu.RUnlock()
	if !changed {
		return nil
	}
	now := time.Now().UTC()
	return l.config.updateRuntime(func() {
		for i := range l.config.EventExecutions {
			execution := &l.config.EventExecutions[i]
			if !containsString(executionIDs, execution.ExecutionID) || executionTerminal(execution.Status) {
				continue
			}
			if execution.Participants == nil {
				execution.Participants = map[string]bool{}
			}
			execution.Participants[targetThreadID] = true
			execution.UpdatedAt = now
		}
	})
}

// SettleThread marks one participant idle. event.settled is emitted only when
// every causally participating thread has reached its existing wait boundary.
func (l *EventLifecycle) SettleThread(threadID string, executionIDs []string, reason string) error {
	if l == nil || len(executionIDs) == 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now().UTC()
	var emitted []EventLifecycleTransition
	err := l.config.updateRuntime(func() {
		for i := range l.config.EventExecutions {
			execution := &l.config.EventExecutions[i]
			if !containsString(executionIDs, execution.ExecutionID) || executionTerminal(execution.Status) {
				continue
			}
			if execution.Participants != nil {
				execution.Participants[threadID] = false
			}
			active := false
			for _, participantActive := range execution.Participants {
				if participantActive {
					active = true
					break
				}
			}
			if !active && execution.Status != eventExecutionPending {
				emitted = append(emitted, appendLifecycleTransition(l.config, execution, eventSettled, reason, now))
			} else {
				execution.UpdatedAt = now
			}
		}
	})
	if err == nil {
		l.emit(emitted)
	}
	return err
}

func (l *EventLifecycle) Fail(executionIDs []string, threadID, reason string) error {
	if l == nil || len(executionIDs) == 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now().UTC()
	var emitted []EventLifecycleTransition
	err := l.config.updateRuntime(func() {
		for i := range l.config.EventExecutions {
			execution := &l.config.EventExecutions[i]
			if !containsString(executionIDs, execution.ExecutionID) || executionTerminal(execution.Status) {
				continue
			}
			if threadID != "" && execution.Participants != nil {
				execution.Participants[threadID] = false
			}
			emitted = append(emitted, appendLifecycleTransition(l.config, execution, eventError, reason, now))
		}
	})
	if err == nil {
		l.emit(emitted)
	}
	return err
}

func (l *EventLifecycle) ActiveForThread(threadID string) []string {
	if l == nil || l.config == nil {
		return nil
	}
	l.config.mu.RLock()
	defer l.config.mu.RUnlock()
	var ids []string
	for _, execution := range l.config.EventExecutions {
		if executionTerminal(execution.Status) || !execution.Participants[threadID] {
			continue
		}
		ids = append(ids, execution.ExecutionID)
	}
	sort.Strings(ids)
	return ids
}

// ReconcileRestoredParticipants terminates executions whose active causal
// participants could not be reconstructed after a Core restart. Ephemeral
// threads are intentionally absent from the restored runtime, but the same
// protection also covers a persistent thread that could not be restored.
//
// Reconciliation happens after the complete persistent thread tree has been
// assembled and before any restored thread is started, so a missing
// participant cannot leave an execution active forever or race new work.
func (l *EventLifecycle) ReconcileRestoredParticipants(restored map[string]bool) ([]string, error) {
	if l == nil || l.config == nil {
		return nil, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now().UTC()
	var emitted []EventLifecycleTransition
	var failed []string
	err := l.config.updateRuntime(func() {
		for i := range l.config.EventExecutions {
			execution := &l.config.EventExecutions[i]
			if executionTerminal(execution.Status) {
				continue
			}
			missing := false
			for participant, active := range execution.Participants {
				if active && !restored[participant] {
					missing = true
					break
				}
			}
			if !missing {
				continue
			}
			failed = append(failed, execution.ExecutionID)
			emitted = append(emitted, appendLifecycleTransition(
				l.config, execution, eventError, "participant_not_restored", now,
			))
		}
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(failed)
	l.emit(emitted)
	return failed, nil
}

func (l *EventLifecycle) PendingTransitions() []EventLifecycleTransition {
	if l == nil || l.config == nil {
		return []EventLifecycleTransition{}
	}
	l.config.mu.RLock()
	defer l.config.mu.RUnlock()
	out := append([]EventLifecycleTransition(nil), l.config.EventLifecycleOutbox...)
	if out == nil {
		return []EventLifecycleTransition{}
	}
	return out
}

func (l *EventLifecycle) Acknowledge(ids []string) error {
	if l == nil || len(ids) == 0 {
		return nil
	}
	wanted := make(map[string]bool, len(ids))
	for _, id := range ids {
		if id = strings.TrimSpace(id); id != "" {
			wanted[id] = true
		}
	}
	if len(wanted) == 0 {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.config.updateRuntime(func() {
		kept := l.config.EventLifecycleOutbox[:0]
		for _, transition := range l.config.EventLifecycleOutbox {
			if !wanted[transition.ID] {
				kept = append(kept, transition)
			}
		}
		l.config.EventLifecycleOutbox = kept
		// Once a terminal execution has no undelivered transitions, its stable
		// event ledger remains sufficient for duplicate POSTs to return the same
		// execution_id. Dropping the internal state here keeps recurring tracked
		// events from growing config.json without bound.
		pendingExecution := map[string]bool{}
		for _, transition := range kept {
			pendingExecution[transition.ExecutionID] = true
		}
		keptExecutions := l.config.EventExecutions[:0]
		for _, execution := range l.config.EventExecutions {
			if executionTerminal(execution.Status) && !pendingExecution[execution.ExecutionID] {
				continue
			}
			keptExecutions = append(keptExecutions, execution)
		}
		l.config.EventExecutions = keptExecutions
	})
}

func containsString(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func firstExecutionID(ids []string) string {
	if len(ids) == 0 {
		return ""
	}
	copyIDs := append([]string(nil), ids...)
	sort.Strings(copyIDs)
	return copyIDs[0]
}

func (t *Thinker) addEventExecutions(ids []string) {
	if t == nil || len(ids) == 0 {
		return
	}
	t.eventExecutionMu.Lock()
	if t.eventExecutionIDs == nil {
		t.eventExecutionIDs = map[string]bool{}
	}
	for _, id := range ids {
		if strings.TrimSpace(id) != "" {
			t.eventExecutionIDs[id] = true
		}
	}
	t.eventExecutionMu.Unlock()
}

func (t *Thinker) currentEventExecutions() []string {
	if t == nil {
		return nil
	}
	t.eventExecutionMu.Lock()
	defer t.eventExecutionMu.Unlock()
	ids := make([]string, 0, len(t.eventExecutionIDs))
	for id := range t.eventExecutionIDs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (t *Thinker) clearEventExecutions(ids []string) {
	if t == nil || len(ids) == 0 {
		return
	}
	t.eventExecutionMu.Lock()
	for _, id := range ids {
		delete(t.eventExecutionIDs, id)
	}
	t.eventExecutionMu.Unlock()
}

// restoredEventParticipants snapshots the fully assembled runtime tree. It is
// used only during startup, before StartAll, so no model work is introduced.
func restoredEventParticipants(root *Thinker) map[string]bool {
	restored := map[string]bool{}
	var walkThinker func(*Thinker)
	var walkManager func(*ThreadManager)
	walkThinker = func(thinker *Thinker) {
		if thinker == nil {
			return
		}
		restored[thinker.threadID] = true
		if thinker.threads != nil {
			walkManager(thinker.threads)
		}
	}
	walkManager = func(manager *ThreadManager) {
		manager.mu.RLock()
		threads := make([]*Thread, 0, len(manager.threads))
		for _, thread := range manager.threads {
			threads = append(threads, thread)
		}
		manager.mu.RUnlock()
		for _, thread := range threads {
			walkThinker(thread.Thinker)
		}
	}
	walkThinker(root)
	return restored
}

func clearEventExecutionsInTree(root *Thinker, executionIDs []string) {
	if root == nil || len(executionIDs) == 0 {
		return
	}
	root.clearEventExecutions(executionIDs)
	if root.threads == nil {
		return
	}
	root.threads.mu.RLock()
	threads := make([]*Thread, 0, len(root.threads.threads))
	for _, thread := range root.threads.threads {
		threads = append(threads, thread)
	}
	root.threads.mu.RUnlock()
	for _, thread := range threads {
		clearEventExecutionsInTree(thread.Thinker, executionIDs)
	}
}

func (t *Thinker) claimEventExecutions(ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	t.addEventExecutions(ids)
	if t.eventLifecycle == nil {
		return nil
	}
	return t.eventLifecycle.Claim(t.threadID, ids)
}

func (t *Thinker) settleEventExecutions(reason string) {
	if t == nil || t.eventLifecycle == nil || t.pendingToolCount() > 0 || t.asyncToolsActive.Load() > 0 {
		return
	}
	ids := t.currentEventExecutions()
	if len(ids) == 0 {
		return
	}
	if err := t.eventLifecycle.SettleThread(t.threadID, ids, reason); err != nil {
		logMsg("EVENT-LIFECYCLE", fmt.Sprintf("[%s] settle: %v", t.threadID, err))
		return
	}
	t.clearEventExecutions(ids)
}

func (t *Thinker) failEventExecutions(reason string) {
	if t == nil || t.eventLifecycle == nil {
		return
	}
	ids := t.currentEventExecutions()
	if len(ids) == 0 {
		return
	}
	if err := t.eventLifecycle.Fail(ids, t.threadID, reason); err != nil {
		logMsg("EVENT-LIFECYCLE", fmt.Sprintf("[%s] fail: %v", t.threadID, err))
		return
	}
	t.clearEventExecutions(ids)
}
