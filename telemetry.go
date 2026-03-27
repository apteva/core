package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

// TelemetryEvent is the unified event format — matches backplane schema.
type TelemetryEvent struct {
	ID         string          `json:"id"`
	InstanceID int64           `json:"instance_id,omitempty"`
	ThreadID   string          `json:"thread_id"`
	Type       string          `json:"type"`
	Time       time.Time       `json:"time"`
	Data       json.RawMessage `json:"data"`
}

// Telemetry collects events and forwards them to the backplane.
type Telemetry struct {
	mu         sync.Mutex
	log        []TelemetryEvent // stored events (forwarded to backplane)
	liveLog    []TelemetryEvent // all events including live-only (for SSE)
	notify     chan struct{}
	backplane  string // backplane URL (e.g. "http://localhost:5280")
	instanceID int64
	seq        int64
	quit       chan struct{}
}

func NewTelemetry() *Telemetry {
	t := &Telemetry{
		notify: make(chan struct{}, 1),
		quit:   make(chan struct{}),
	}

	// Read instance ID from env (set by backplane when spawning)
	if id := os.Getenv("INSTANCE_ID"); id != "" {
		fmt.Sscanf(id, "%d", &t.instanceID)
	}

	// Configure backplane URL for fire-and-forget
	if url := os.Getenv("BACKPLANE_URL"); url != "" {
		t.backplane = url
		go t.forwardLoop()
	}

	return t
}

func (t *Telemetry) Stop() {
	close(t.quit)
}

func (t *Telemetry) generateID() string {
	t.seq++
	return fmt.Sprintf("%d-%d", time.Now().UnixMilli(), t.seq)
}

// Emit records a telemetry event (stored + forwarded to backplane).
func (t *Telemetry) Emit(eventType, threadID string, data any) {
	t.emit(eventType, threadID, data, true)
}

// EmitLive records a telemetry event for SSE only (not forwarded to backplane).
func (t *Telemetry) EmitLive(eventType, threadID string, data any) {
	t.emit(eventType, threadID, data, false)
}

func (t *Telemetry) emit(eventType, threadID string, data any, store bool) {
	dataJSON, _ := json.Marshal(data)

	ev := TelemetryEvent{
		ID:         t.generateID(),
		InstanceID: t.instanceID,
		ThreadID:   threadID,
		Type:       eventType,
		Time:       time.Now(),
		Data:       json.RawMessage(dataJSON),
	}

	t.mu.Lock()
	if store {
		t.log = append(t.log, ev)
		if len(t.log) > 2000 {
			t.log = t.log[len(t.log)-1000:]
		}
	}
	t.liveLog = append(t.liveLog, ev)
	if len(t.liveLog) > 2000 {
		t.liveLog = t.liveLog[len(t.liveLog)-1000:]
	}
	t.mu.Unlock()

	// Notify SSE watchers
	select {
	case t.notify <- struct{}{}:
	default:
	}

	// Forward live-only events to backplane immediately for SSE broadcast
	if !store && t.backplane != "" {
		go t.forwardLive(ev)
	}
}

func (t *Telemetry) forwardLive(ev TelemetryEvent) {
	body, err := json.Marshal([]TelemetryEvent{ev})
	if err != nil {
		return
	}
	req, err := http.NewRequest("POST", t.backplane+"/telemetry/live", bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 2 * time.Second}
	if resp, err := client.Do(req); err == nil {
		resp.Body.Close()
	}
}

// Events returns all events (including live-only) since the given index. Used by SSE.
func (t *Telemetry) Events(since int) ([]TelemetryEvent, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if since >= len(t.liveLog) {
		return nil, len(t.liveLog)
	}
	events := make([]TelemetryEvent, len(t.liveLog)-since)
	copy(events, t.liveLog[since:])
	return events, len(t.liveLog)
}

// StoredEvents returns only stored events since the given index. Used by forwardLoop.
func (t *Telemetry) StoredEvents(since int) ([]TelemetryEvent, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if since >= len(t.log) {
		return nil, len(t.log)
	}
	events := make([]TelemetryEvent, len(t.log)-since)
	copy(events, t.log[since:])
	return events, len(t.log)
}

// forwardLoop batches events and POSTs them to the backplane every second.
func (t *Telemetry) forwardLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastSent int
	client := &http.Client{Timeout: 5 * time.Second}

	for {
		select {
		case <-ticker.C:
			events, total := t.StoredEvents(lastSent)
			if len(events) == 0 {
				continue
			}

			body, err := json.Marshal(events)
			if err != nil {
				continue
			}

			req, err := http.NewRequest("POST", t.backplane+"/telemetry", bytes.NewReader(body))
			if err != nil {
				continue
			}
			req.Header.Set("Content-Type", "application/json")

			// Fire and forget — don't block on response
			resp, err := client.Do(req)
			if err == nil {
				resp.Body.Close()
				lastSent = total
			}
			// On error, we'll retry next tick with the same events

		case <-t.quit:
			return
		}
	}
}

// --- Convenience emitters with typed data ---

type LLMDoneData struct {
	Model        string  `json:"model"`
	TokensIn     int     `json:"tokens_in"`
	TokensCached int     `json:"tokens_cached"`
	TokensOut    int     `json:"tokens_out"`
	DurationMs   int64   `json:"duration_ms"`
	CostUSD      float64 `json:"cost_usd"`
	Iteration    int     `json:"iteration"`
	Rate         string  `json:"rate"`
	ContextMsgs  int     `json:"context_msgs"`
	ContextChars int     `json:"context_chars"`
	MemoryCount  int     `json:"memory_count"`
	ThreadCount  int     `json:"thread_count"`
	Message      string  `json:"message,omitempty"`
}

type LLMChunkData struct {
	Text      string `json:"text"`
	Iteration int    `json:"iteration"`
}

type LLMErrorData struct {
	Model     string `json:"model"`
	Error     string `json:"error"`
	Iteration int    `json:"iteration"`
}

type ThreadSpawnData struct {
	ParentID  string   `json:"parent_id"`
	Directive string   `json:"directive"`
	Tools     []string `json:"tools"`
}

type ThreadDoneData struct {
	ParentID string `json:"parent_id"`
	Result   string `json:"result,omitempty"`
}

type ThreadMessageData struct {
	From    string `json:"from"`
	To      string `json:"to"`
	Message string `json:"message"`
}

type ToolCallData struct {
	Name string `json:"name"`
	Args string `json:"args,omitempty"`
}

type ToolResultData struct {
	Name       string `json:"name"`
	DurationMs int64  `json:"duration_ms"`
	Success    bool   `json:"success"`
	Result     string `json:"result,omitempty"`
}

type DirectiveChangeData struct {
	Old string `json:"old,omitempty"`
	New string `json:"new"`
}

// --- Cost calculation (matches TUI pricing) ---

const (
	costInputPerMillion  = 0.60
	costCachedPerMillion = 0.10
	costOutputPerMillion = 3.00
)

func calculateCost(usage TokenUsage) float64 {
	uncached := usage.PromptTokens - usage.CachedTokens
	if uncached < 0 {
		uncached = 0
	}
	return (float64(uncached)*costInputPerMillion +
		float64(usage.CachedTokens)*costCachedPerMillion +
		float64(usage.CompletionTokens)*costOutputPerMillion) / 1_000_000
}
