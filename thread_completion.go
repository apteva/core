package core

import "fmt"

func (thread *Thread) queueCompletion(callID, message string) error {
	parent := thread.Parent
	if parent == nil {
		return fmt.Errorf("missing completion recipient")
	}
	if callID == "" {
		callID = newULID()
	}
	text := fmt.Sprintf("[thread:%s done]", thread.ID)
	if message != "" {
		text += " " + message
	}
	ids := thread.Thinker.currentEventExecutions()
	if parent.eventLifecycle != nil {
		if err := parent.eventLifecycle.Propagate(ids, parent.threadID); err != nil {
			return err
		}
	}
	event := PersistentThreadEvent{ID: "completion/" + thread.ID + "/" + callID, Text: text, Hash: threadEventHash(text, nil), ExecutionIDs: ids}
	if parent.threadID == "main" {
		_, err := parent.QueueMainEvents([]PersistentThreadEvent{event})
		return err
	}
	if parent.owner == nil {
		return fmt.Errorf("missing parent inbox owner")
	}
	_, err := parent.owner.QueueEvents(parent.threadID, []PersistentThreadEvent{event})
	return err
}
