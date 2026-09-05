package core

// External/untracked traffic has explicit admission bounds. Already accepted
// durable events and finite tool completions retain reliable internal delivery.
const maxMailboxEvents = 65536
const maxMailboxBytes = 64 << 20

func eventPayloadBytes(ev Event) int {
	n := len(ev.Text)
	for _, p := range ev.Parts {
		n += len(p.Text)
		if p.ImageURL != nil {
			n += len(p.ImageURL.URL)
		}
		if p.InputAudio != nil {
			n += len(p.InputAudio.Data)
		}
		if p.AudioURL != nil {
			n += len(p.AudioURL.URL)
		}
	}
	return n
}
func (b *EventBus) TryPublish(ev Event) bool { return b.publish(ev, true) }
func validateInboxCapacity(existing, incoming []PersistentThreadEvent) error {
	seen := map[string]bool{}
	count, size := 0, 0
	for _, e := range existing {
		seen[e.ID] = true
		if !e.Consumed {
			count++
			size += eventPayloadBytes(Event{Text: e.Text, Parts: e.Parts})
		}
	}
	for _, e := range incoming {
		if !seen[e.ID] {
			seen[e.ID] = true
			count++
			size += eventPayloadBytes(Event{Text: e.Text, Parts: e.Parts})
		}
	}
	if count > 4096 || size > maxMailboxBytes {
		return &ThreadEventValidationError{Message: "durable inbox capacity reached; wait for consumption before retrying"}
	}
	return nil
}
