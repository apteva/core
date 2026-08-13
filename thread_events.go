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
	Accepted  []string
	Duplicate []string
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
	for _, event := range thread.inboxEvents {
		existing[event.ID] = event.Hash
	}
	batch := map[string]string{}
	for _, event := range incoming {
		if hash, ok := existing[event.ID]; ok && hash != event.Hash {
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
	acceptedEvents := make([]PersistentThreadEvent, 0, len(incoming))
	for _, event := range incoming {
		if _, ok := existing[event.ID]; ok {
			result.Duplicate = append(result.Duplicate, event.ID)
			continue
		}
		event.Consumed = false
		event.Parts = cloneContentParts(event.Parts)
		thread.inboxEvents = append(thread.inboxEvents, event)
		existing[event.ID] = event.Hash
		acceptedEvents = append(acceptedEvents, event)
		result.Accepted = append(result.Accepted, event.ID)
	}

	if len(acceptedEvents) > 0 && !thread.Ephemeral {
		state := persistentThreadStateBase(thread)
		state.Events = clonePersistentThreadEvents(thread.inboxEvents)
		if err := tm.parent.config.SaveThread(state); err != nil {
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
