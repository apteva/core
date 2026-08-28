package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const maxConsumedThreadEventIDs = 1024

// ThreadEventQueueResult describes the atomic outcome of adding API inbox
// events to a live thread.
type ThreadEventQueueResult struct {
	Accepted   []string
	Duplicate  []string
	Executions map[string]string
}

// ThreadEventConflictError means an idempotency key was reused for a
// different payload. Nothing from that enqueue operation is accepted.
type ThreadEventConflictError struct {
	ID string
}

type ThreadEventValidationError struct {
	Message string
}

func (e *ThreadEventValidationError) Error() string { return e.Message }

func (e *ThreadEventConflictError) Error() string {
	return fmt.Sprintf("event %q already exists with different content", e.ID)
}

func threadEventHash(text string, parts []ContentPart) string {
	payload, _ := json.Marshal(struct {
		Text  string        `json:"text"`
		Parts []ContentPart `json:"parts,omitempty"`
	}{Text: text, Parts: parts})
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])
}

func clonePersistentThreadEvents(in []PersistentThreadEvent) []PersistentThreadEvent {
	if len(in) == 0 {
		return nil
	}
	out := make([]PersistentThreadEvent, len(in))
	for i := range in {
		out[i] = in[i]
		out[i].Parts = cloneContentParts(in[i].Parts)
	}
	return out
}

func trimPersistentThreadEvents(events []PersistentThreadEvent) []PersistentThreadEvent {
	consumed := 0
	for _, event := range events {
		if event.Consumed {
			consumed++
		}
	}
	drop := consumed - maxConsumedThreadEventIDs
	if drop <= 0 {
		return events
	}
	out := make([]PersistentThreadEvent, 0, len(events)-drop)
	for _, event := range events {
		if event.Consumed && drop > 0 {
			drop--
			continue
		}
		out = append(out, event)
	}
	return out
}

func pendingPersistentEvents(events []PersistentThreadEvent) []PersistentThreadEvent {
	var pending []PersistentThreadEvent
	for _, event := range events {
		if !event.Consumed {
			pending = append(pending, event)
		}
	}
	return clonePersistentThreadEvents(pending)
}

func executionIDsForPersistentEvents(events []PersistentThreadEvent, ids []string) []string {
	wanted := make(map[string]bool, len(ids))
	for _, id := range ids {
		wanted[id] = true
	}
	var executionIDs []string
	for _, event := range events {
		if wanted[event.ID] && event.ExecutionID != "" {
			executionIDs = append(executionIDs, event.ExecutionID)
		}
	}
	return executionIDs
}

func (t *Thinker) executionIDsForInboxEvents(ids []string) []string {
	if t == nil {
		return nil
	}
	t.mainInboxMu.Lock()
	defer t.mainInboxMu.Unlock()
	return executionIDsForPersistentEvents(t.mainInboxEvents, ids)
}

func (thread *Thread) executionIDsForInboxEvents(ids []string) []string {
	thread.inboxMu.Lock()
	defer thread.inboxMu.Unlock()
	return executionIDsForPersistentEvents(thread.inboxEvents, ids)
}

func (thread *Thread) pendingInboxEvents() []PersistentThreadEvent {
	thread.inboxMu.Lock()
	defer thread.inboxMu.Unlock()
	var pending []PersistentThreadEvent
	for _, event := range thread.inboxEvents {
		if !event.Consumed {
			pending = append(pending, event)
		}
	}
	return clonePersistentThreadEvents(pending)
}

func (thread *Thread) hasPendingInboxEvents() bool {
	thread.inboxMu.Lock()
	defer thread.inboxMu.Unlock()
	for _, event := range thread.inboxEvents {
		if !event.Consumed {
			return true
		}
	}
	return false
}

func (thread *Thread) reconcileInboxEvents(messages []Message) bool {
	seen := map[string]bool{}
	for _, message := range messages {
		for _, id := range message.EventIDs {
			seen[id] = true
		}
	}
	return thread.reconcileInboxEventIDs(seen)
}

func (thread *Thread) reconcileInboxEventIDs(seen map[string]bool) bool {
	if len(seen) == 0 {
		return false
	}
	thread.inboxMu.Lock()
	defer thread.inboxMu.Unlock()
	changed := false
	for i := range thread.inboxEvents {
		if !thread.inboxEvents[i].Consumed && seen[thread.inboxEvents[i].ID] {
			thread.inboxEvents[i].Consumed = true
			thread.inboxEvents[i].Text = ""
			thread.inboxEvents[i].Parts = nil
			changed = true
		}
	}
	if changed {
		thread.inboxEvents = trimPersistentThreadEvents(thread.inboxEvents)
	}
	return changed
}

func publishThreadInboxEvents(bus *EventBus, threadID string, events []PersistentThreadEvent) {
	for _, event := range events {
		bus.Publish(Event{
			ID: event.ID, Type: EventInbox, To: threadID,
			Text: event.Text, Parts: cloneContentParts(event.Parts),
			ExecutionIDs: executionIDsForEvent(event),
		})
	}
}

// QueueEvents atomically deduplicates and durably records inbox events before
// publishing them. Existing IDs with identical payloads are harmless retries;
// reusing an ID for different content rejects the complete batch.
func (tm *ThreadManager) QueueEvents(id string, incoming []PersistentThreadEvent) (ThreadEventQueueResult, error) {
	var result ThreadEventQueueResult
	if len(incoming) == 0 {
		return result, nil
	}
	owner, _ := tm.findManagedThread(id)
	if owner == nil {
		return result, fmt.Errorf("thread %q not found", id)
	}
	owner.mu.RLock()
	thread := owner.threads[id]
	if thread == nil {
		owner.mu.RUnlock()
		return result, fmt.Errorf("thread %q not found", id)
	}
	thread.inboxMu.Lock()
	for _, event := range incoming {
		if event.TrackLifecycle && thread.Ephemeral {
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return result, &ThreadEventValidationError{Message: "lifecycle tracking requires a durable thread; ephemeral threads cannot retain execution state across restart"}
		}
	}
	if thread.IsRealtime {
		for _, event := range incoming {
			if len(event.Parts) > 0 {
				thread.inboxMu.Unlock()
				owner.mu.RUnlock()
				return result, &ThreadEventValidationError{Message: "multimodal events are not supported for realtime threads; send text or use the audio bridge"}
			}
		}
	}

	existing := make(map[string]string, len(thread.inboxEvents)+len(incoming))
	existingEvents := make(map[string]PersistentThreadEvent, len(thread.inboxEvents))
	for _, event := range thread.inboxEvents {
		existing[event.ID] = event.Hash
		existingEvents[event.ID] = event
	}
	batch := map[string]string{}
	for _, event := range incoming {
		if hash, ok := existing[event.ID]; ok && hash != event.Hash {
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return result, &ThreadEventConflictError{ID: event.ID}
		}
		if persisted, ok := existingEvents[event.ID]; ok && persisted.TrackLifecycle != event.TrackLifecycle {
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return result, &ThreadEventConflictError{ID: event.ID}
		}
		if hash, ok := batch[event.ID]; ok && hash != event.Hash {
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return result, &ThreadEventConflictError{ID: event.ID}
		}
		batch[event.ID] = event.Hash
	}

	before := clonePersistentThreadEvents(thread.inboxEvents)
	prepareTrackedEventExecutions(incoming)
	acceptedEvents := make([]PersistentThreadEvent, 0, len(incoming))
	for _, event := range incoming {
		if _, ok := existing[event.ID]; ok {
			result.Duplicate = append(result.Duplicate, event.ID)
			for _, persisted := range thread.inboxEvents {
				if persisted.ID == event.ID && persisted.ExecutionID != "" {
					if result.Executions == nil {
						result.Executions = map[string]string{}
					}
					result.Executions[event.ID] = persisted.ExecutionID
					break
				}
			}
			continue
		}
		event.Consumed = false
		event.Parts = cloneContentParts(event.Parts)
		thread.inboxEvents = append(thread.inboxEvents, event)
		existing[event.ID] = event.Hash
		acceptedEvents = append(acceptedEvents, event)
		result.Accepted = append(result.Accepted, event.ID)
		if event.ExecutionID != "" {
			if result.Executions == nil {
				result.Executions = map[string]string{}
			}
			result.Executions[event.ID] = event.ExecutionID
		}
	}

	if len(acceptedEvents) > 0 && !thread.Ephemeral {
		state := persistentThreadStateBase(thread)
		state.Events = clonePersistentThreadEvents(thread.inboxEvents)
		if err := tm.parent.config.saveThreadAndRegisterEventExecutions(state, acceptedEvents); err != nil {
			thread.inboxEvents = before
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return ThreadEventQueueResult{}, fmt.Errorf("persist thread events: %w", err)
		}
	}
	thread.inboxMu.Unlock()
	owner.mu.RUnlock()

	publishThreadInboxEvents(tm.parent.bus, id, acceptedEvents)
	return result, nil
}

// QueueMainEvents gives main the same durable/idempotent inbox contract as
// API-created threads. The legacy POST /event path remains available for
// untracked fire-and-forget input; stable IDs use this queue.
func (t *Thinker) QueueMainEvents(incoming []PersistentThreadEvent) (ThreadEventQueueResult, error) {
	var result ThreadEventQueueResult
	if t == nil || len(incoming) == 0 {
		return result, nil
	}
	t.mainInboxMu.Lock()
	defer t.mainInboxMu.Unlock()

	existing := make(map[string]PersistentThreadEvent, len(t.mainInboxEvents))
	for _, event := range t.mainInboxEvents {
		existing[event.ID] = event
	}
	batch := map[string]PersistentThreadEvent{}
	for _, event := range incoming {
		if persisted, ok := existing[event.ID]; ok {
			if persisted.Hash != event.Hash || persisted.TrackLifecycle != event.TrackLifecycle {
				return result, &ThreadEventConflictError{ID: event.ID}
			}
		}
		if prior, ok := batch[event.ID]; ok && (prior.Hash != event.Hash || prior.TrackLifecycle != event.TrackLifecycle) {
			return result, &ThreadEventConflictError{ID: event.ID}
		}
		batch[event.ID] = event
	}

	prepareTrackedEventExecutions(incoming)
	before := clonePersistentThreadEvents(t.mainInboxEvents)
	acceptedEvents := make([]PersistentThreadEvent, 0, len(incoming))
	for _, event := range incoming {
		if persisted, ok := existing[event.ID]; ok {
			result.Duplicate = append(result.Duplicate, event.ID)
			if persisted.ExecutionID != "" {
				if result.Executions == nil {
					result.Executions = map[string]string{}
				}
				result.Executions[event.ID] = persisted.ExecutionID
			}
			continue
		}
		event.Consumed = false
		event.Parts = cloneContentParts(event.Parts)
		t.mainInboxEvents = append(t.mainInboxEvents, event)
		existing[event.ID] = event
		acceptedEvents = append(acceptedEvents, event)
		result.Accepted = append(result.Accepted, event.ID)
		if event.ExecutionID != "" {
			if result.Executions == nil {
				result.Executions = map[string]string{}
			}
			result.Executions[event.ID] = event.ExecutionID
		}
	}
	if len(acceptedEvents) > 0 {
		if err := t.config.saveMainEventsAndRegisterExecutions(t.mainInboxEvents, acceptedEvents); err != nil {
			t.mainInboxEvents = before
			return ThreadEventQueueResult{}, fmt.Errorf("persist main events: %w", err)
		}
	}
	publishThreadInboxEvents(t.bus, "main", acceptedEvents)
	return result, nil
}

func (t *Thinker) markMainEventsConsumed(ids []string) error {
	if t == nil || len(ids) == 0 {
		return nil
	}
	t.mainInboxMu.Lock()
	defer t.mainInboxMu.Unlock()
	wanted := map[string]bool{}
	for _, id := range ids {
		wanted[id] = true
	}
	before := clonePersistentThreadEvents(t.mainInboxEvents)
	changed := false
	for i := range t.mainInboxEvents {
		if wanted[t.mainInboxEvents[i].ID] && !t.mainInboxEvents[i].Consumed {
			t.mainInboxEvents[i].Consumed = true
			t.mainInboxEvents[i].Text = ""
			t.mainInboxEvents[i].Parts = nil
			changed = true
		}
	}
	if !changed {
		return nil
	}
	t.mainInboxEvents = trimPersistentThreadEvents(t.mainInboxEvents)
	if err := t.config.saveMainEvents(t.mainInboxEvents); err != nil {
		t.mainInboxEvents = before
		return fmt.Errorf("persist consumed main events: %w", err)
	}
	return nil
}

func executionIDsForEvent(event PersistentThreadEvent) []string {
	if event.ExecutionID == "" {
		return nil
	}
	return []string{event.ExecutionID}
}

// markEventsConsumed advances accepted events only after their user message is
// safely appended to session history. The payload can then be discarded while
// the bounded ID+hash ledger continues to deduplicate retries.
func (tm *ThreadManager) markEventsConsumed(id string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	owner, _ := tm.findManagedThread(id)
	if owner == nil {
		return fmt.Errorf("thread %q not found", id)
	}
	owner.mu.RLock()
	thread := owner.threads[id]
	if thread == nil {
		owner.mu.RUnlock()
		return fmt.Errorf("thread %q not found", id)
	}
	thread.inboxMu.Lock()
	wanted := map[string]bool{}
	for _, id := range ids {
		wanted[id] = true
	}
	before := clonePersistentThreadEvents(thread.inboxEvents)
	changed := false
	for i := range thread.inboxEvents {
		if wanted[thread.inboxEvents[i].ID] && !thread.inboxEvents[i].Consumed {
			thread.inboxEvents[i].Consumed = true
			thread.inboxEvents[i].Text = ""
			thread.inboxEvents[i].Parts = nil
			changed = true
		}
	}
	if !changed {
		thread.inboxMu.Unlock()
		owner.mu.RUnlock()
		return nil
	}
	thread.inboxEvents = trimPersistentThreadEvents(thread.inboxEvents)
	if !thread.Ephemeral {
		state := persistentThreadStateBase(thread)
		state.Events = clonePersistentThreadEvents(thread.inboxEvents)
		if err := tm.parent.config.SaveThread(state); err != nil {
			thread.inboxEvents = before
			thread.inboxMu.Unlock()
			owner.mu.RUnlock()
			return fmt.Errorf("persist consumed thread events: %w", err)
		}
	}
	thread.inboxMu.Unlock()
	owner.mu.RUnlock()
	return nil
}
