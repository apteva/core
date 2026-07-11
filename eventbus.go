package core

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Event types
const (
	EventInbox       = "inbox"        // message addressed to a thinker (replaces inbox chan string)
	EventChunk       = "chunk"        // streaming token from LLM
	EventToolChunk   = "tool_chunk"   // streaming tool argument chunk from LLM
	EventThinkDone   = "think_done"   // completed a think cycle
	EventThinkError  = "think_error"  // error during think
	EventThreadStart = "thread_start" // thread spawned
	EventThreadDone  = "thread_done"  // thread terminated
)

// Event is the single message type flowing through the system.
type Event struct {
	Type string // one of the Event* constants
	From string // source: "main", thread ID, "tui", "api", "tool:name"
	To   string // target subscriber ID; "" = broadcast

	Text       string        // message payload
	ToolName   string        // tool name (for EventToolChunk)
	Parts      []ContentPart // optional media (images, audio) attached to this event
	ToolResult *ToolResult   // optional: structured tool result

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
	Reasoning      ReasoningLevel
	MemoryCount    int
	ThreadCount    int
	ContextMsgs    int // number of messages in context window
	ContextChars   int // approximate character count of context
	Error          error
}

// Subscription is a handle returned by Subscribe/SubscribeAll.
//
// Dropped is an atomic counter of events the bus discarded because this
// subscription's channel was full. Non-zero means the consumer is not
// keeping up with the publish rate and some events were lost (order of
// remaining events is preserved — drops are tail truncations of the
// backlog, not reorderings). Read with atomic.LoadUint64; inspect from
// anywhere safely.
type Subscription struct {
	ID      string
	C       chan Event
	Wake    chan struct{} // signaled on every new event delivery
	Dropped uint64        // atomic
	all     bool          // true = receives all events (observer)

	// Rate-limit drop logging — one line per (sub, ~1s window) is enough
	// to alert operators without flooding the log during sustained spikes.
	lastDropLogNano int64 // atomic

	// Targeted thinker events are reliable. Once C fills, events queue here
	// in publication order until the thinker drains them. Observer streams do
	// not use this queue and remain lossy under backpressure.
	queueMu  sync.Mutex
	overflow []Event
}

// Publishes (on drop) emit a log line at most once per logDropThrottle.
const logDropThrottle = int64(time.Second)

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
	sub, _ := b.subscribe(id, buffer, false, false)
	return sub
}

// SubscribeUnique atomically reserves an ID. Thinker subscriptions use this
// path so two branches cannot silently replace each other's routing entry.
func (b *EventBus) SubscribeUnique(id string, buffer int) (*Subscription, error) {
	return b.subscribe(id, buffer, false, true)
}

func (b *EventBus) subscribe(id string, buffer int, all, unique bool) (*Subscription, error) {
	sub := &Subscription{
		ID:   id,
		C:    make(chan Event, buffer),
		Wake: make(chan struct{}, 1),
		all:  all,
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if unique {
		if _, exists := b.subs[id]; exists {
			return nil, fmt.Errorf("subscription %q already exists", id)
		}
	}
	b.subs[id] = sub
	return sub, nil
}

// SubscribeAll creates an observer subscription that receives ALL events.
// Used by TUI, API SSE, tests.
func (b *EventBus) SubscribeAll(id string, buffer int) *Subscription {
	sub, _ := b.subscribe(id, buffer, true, false)
	return sub
}

// RenameSubscription moves an existing subscription without replacing a
// different subscriber. The channel itself is preserved, including backlog.
func (b *EventBus) RenameSubscription(oldID, newID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	sub, ok := b.subs[oldID]
	if !ok {
		return fmt.Errorf("subscription %q not found", oldID)
	}
	if existing, taken := b.subs[newID]; taken && existing != sub {
		return fmt.Errorf("subscription %q already exists", newID)
	}
	delete(b.subs, oldID)
	sub.ID = newID
	b.subs[newID] = sub
	return nil
}

func (b *EventBus) HasSubscriber(id string) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	_, ok := b.subs[id]
	return ok
}

// Unsubscribe removes a subscription.
func (b *EventBus) Unsubscribe(id string) {
	b.mu.Lock()
	delete(b.subs, id)
	b.mu.Unlock()
}

// Publish delivers an event to all matching subscriptions. It never blocks
// the publisher: when a subscriber's channel is full, the event is
// discarded and the subscription's Dropped counter is incremented. Drops
// are logged (rate-limited per subscription) so silent loss is impossible
// to miss in the logs.
//
// Subscribers are snapshotted under a read lock and delivered to without
// holding the lock, so one slow channel never stalls other subscribers or
// blocks concurrent publishers / (un)subscribes.
func (b *EventBus) Publish(ev Event) {
	b.mu.RLock()
	snapshot := make([]*Subscription, 0, len(b.subs))
	for _, sub := range b.subs {
		snapshot = append(snapshot, sub)
	}
	b.mu.RUnlock()

	for _, sub := range snapshot {
		switch {
		case sub.all:
			// Observers (TUI, API SSE, tests) get everything.
			deliver(sub, ev, false)
		case ev.To == sub.ID:
			// Targeted delivery — also signal Wake so consumers using the
			// wake channel (e.g. the thinker's drain loop) spin even if
			// the buffered channel was already full.
			deliverTargeted(sub, ev)
		}
		// Broadcasts (To=="") to non-observers are silently skipped —
		// they're observational (chunks, think_done) and would flood the
		// channel.
	}
}

func deliverTargeted(sub *Subscription, ev Event) {
	sub.queueMu.Lock()
	if len(sub.overflow) > 0 {
		sub.overflow = append(sub.overflow, ev)
	} else {
		select {
		case sub.C <- ev:
		default:
			sub.overflow = append(sub.overflow, ev)
		}
	}
	sub.queueMu.Unlock()
	select {
	case sub.Wake <- struct{}{}:
	default:
	}
}

// DrainTargeted returns every currently queued event in publication order.
// Holding queueMu while draining C prevents a concurrent publisher from
// placing a newer event in C ahead of an older overflow event.
func (sub *Subscription) DrainTargeted() []Event {
	if sub == nil {
		return nil
	}
	sub.queueMu.Lock()
	defer sub.queueMu.Unlock()
	events := make([]Event, 0, len(sub.C)+len(sub.overflow))
	for {
		select {
		case ev := <-sub.C:
			events = append(events, ev)
		default:
			events = append(events, sub.overflow...)
			sub.overflow = nil
			return events
		}
	}
}

// deliver attempts a non-blocking send; on failure increments the sub's
// drop counter and rate-limits a warning log. If signalWake is true, also
// pulses the Wake channel so a consumer waiting on wake sees the miss and
// can re-examine state.
func deliver(sub *Subscription, ev Event, signalWake bool) {
	select {
	case sub.C <- ev:
	default:
		total := atomic.AddUint64(&sub.Dropped, 1)
		now := time.Now().UnixNano()
		last := atomic.LoadInt64(&sub.lastDropLogNano)
		if now-last >= logDropThrottle && atomic.CompareAndSwapInt64(&sub.lastDropLogNano, last, now) {
			logMsg("EVENTBUS", formatDropLog(sub.ID, ev.Type, cap(sub.C), total))
		}
	}
	if signalWake {
		select {
		case sub.Wake <- struct{}{}:
		default:
		}
	}
}

func formatDropLog(subID, evType string, bufCap int, total uint64) string {
	return "dropped " + evType + " for sub=" + subID +
		" (buffer=" + itoa(bufCap) + " full, dropped_total=" + utoa(total) + ")"
}

// Tiny local int formatters to avoid importing fmt here — this function
// runs on every drop, so keep it allocation-light.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func utoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// DropStats returns a snapshot of dropped-event counts keyed by
// subscription id. Useful for tests and ops dashboards to detect
// backpressure. Order is not guaranteed.
func (b *EventBus) DropStats() map[string]uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	out := make(map[string]uint64, len(b.subs))
	for id, sub := range b.subs {
		if d := atomic.LoadUint64(&sub.Dropped); d > 0 {
			out[id] = d
		}
	}
	return out
}
