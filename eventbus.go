package main

import (
	"sync"
	"time"
)

// Event types
const (
	EventInbox       = "inbox"        // message addressed to a thinker (replaces inbox chan string)
	EventChunk       = "chunk"        // streaming token from LLM
	EventThinkDone   = "think_done"   // completed a think cycle
	EventThinkError  = "think_error"  // error during think
	EventThreadStart = "thread_start" // thread spawned
	EventThreadDone  = "thread_done"  // thread terminated
	EventThreadReply = "thread_reply" // thread sent a visible reply
)

// Event is the single message type flowing through the system.
type Event struct {
	Type string // one of the Event* constants
	From string // source: "main", thread ID, "tui", "api", "tool:name"
	To   string // target subscriber ID; "" = broadcast

	Text  string        // message payload
	Parts []ContentPart // optional media (images, audio) attached to this event

	// Structured fields (populated for ThinkDone events)
	Iteration      int
	Duration       time.Duration
	ConsumedEvents []string
	Usage          TokenUsage
	ToolCalls      []string
	Replies        []string
	Rate           ThinkRate
	SleepDuration  time.Duration
	Model          ModelTier
	MemoryCount    int
	ThreadCount    int
	ContextMsgs    int // number of messages in context window
	ContextChars   int // approximate character count of context
	Error          error
}

// Subscription is a handle returned by Subscribe/SubscribeAll.
type Subscription struct {
	ID   string
	C    chan Event
	Wake chan struct{} // signaled on every new event delivery
	all  bool         // true = receives all events (observer)
}

// EventBus is the central pub/sub hub. All thinkers share one bus.
type EventBus struct {
	mu   sync.RWMutex
	subs map[string]*Subscription
}

func NewEventBus() *EventBus {
	return &EventBus{
		subs: make(map[string]*Subscription),
	}
}

// Subscribe creates a targeted subscription. Receives events where To == id, plus broadcasts (To == "").
func (b *EventBus) Subscribe(id string, buffer int) *Subscription {
	sub := &Subscription{
		ID:   id,
		C:    make(chan Event, buffer),
		Wake: make(chan struct{}, 1),
	}
	b.mu.Lock()
	b.subs[id] = sub
	b.mu.Unlock()
	return sub
}

// SubscribeAll creates an observer subscription that receives ALL events.
// Used by TUI, API SSE, tests.
func (b *EventBus) SubscribeAll(id string, buffer int) *Subscription {
	sub := &Subscription{
		ID:   id,
		C:    make(chan Event, buffer),
		Wake: make(chan struct{}, 1),
		all:  true,
	}
	b.mu.Lock()
	b.subs[id] = sub
	b.mu.Unlock()
	return sub
}

// Unsubscribe removes a subscription.
func (b *EventBus) Unsubscribe(id string) {
	b.mu.Lock()
	delete(b.subs, id)
	b.mu.Unlock()
}

// Publish sends an event. Never blocks — drops if a subscriber's channel is full.
func (b *EventBus) Publish(ev Event) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, sub := range b.subs {
		if sub.all {
			// Observers get everything (TUI, tests, API)
			select {
			case sub.C <- ev:
			default:
			}
		} else if ev.To == sub.ID {
			// Regular subscribers only get events targeted to them
			select {
			case sub.C <- ev:
			default:
			}
			// Signal wake for targeted delivery
			select {
			case sub.Wake <- struct{}{}:
			default:
			}
		}
		// Broadcasts (To=="") to non-observers are silently skipped —
		// they're observational (chunks, think_done) and would flood the channel
	}
}
