package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// TelemetryEvent is the unified event format — matches server schema.
type TelemetryEvent struct {
	ID         string          `json:"id"`
	InstanceID int64           `json:"instance_id,omitempty"`
	ThreadID   string          `json:"thread_id"`
	Type       string          `json:"type"`
	Time       time.Time       `json:"time"`
	Data       json.RawMessage `json:"data"`
}

// Telemetry collects events and forwards them to the server.
type Telemetry struct {
	mu               sync.Mutex
	log              []TelemetryEvent // stored events (forwarded to server)
	liveLog          []TelemetryEvent // all events including live-only (for SSE)
	notify           chan struct{}
	notifyMu         sync.Mutex
	notifySubs       map[uint64]chan struct{}
	notifySeq        uint64
	forwardCh        chan TelemetryEvent // serialized queue for live event forwarding
	serverURL        string              // server URL (e.g. "http://localhost:5280")
	telemetryURL     string              // full URL for batched stored events
	telemetryLiveURL string              // full URL for live event forwarding
	instanceSecret   string              // shared secret for telemetry auth
	instanceID       int64
	seq              int64
	quit             chan struct{}
	stopOnce         sync.Once

	// dropCount tracks live-forward events dropped because forwardCh was
	// full. We still drop (blocking Emit from the thinker hot path would
	// be worse), but we count drops and log every 50th one so the data
	// loss is visible rather than invisible.
	dropCount int64
}

func NewTelemetry() *Telemetry {
	t := &Telemetry{
		notify: make(chan struct{}, 1),
		// 5000-slot buffer (was 500) to absorb bursts during heavy tool
		// activity like transcription runs without dropping events.
		forwardCh:  make(chan TelemetryEvent, 5000),
		notifySubs: make(map[uint64]chan struct{}),
		quit:       make(chan struct{}),
	}

	// Read agent ID from env (set by server when spawning).
	// AGENT_ID is the canonical name from the platform's rename
	// (Phase 2); INSTANCE_ID stays accepted so an older server build
	// that hasn't dual-written the new var still launches us cleanly.
	id := os.Getenv("AGENT_ID")
	if id == "" {
		id = os.Getenv("INSTANCE_ID")
	}
	if id != "" {
		fmt.Sscanf(id, "%d", &t.instanceID)
	}

	// Read agent secret from env (for telemetry auth). Same dual-name
	// fallback as AGENT_ID above.
	t.instanceSecret = os.Getenv("AGENT_SECRET")
	if t.instanceSecret == "" {
		t.instanceSecret = os.Getenv("INSTANCE_SECRET")
	}

	// Telemetry endpoints are provided by the server via env vars.
	// Core doesn't need to know the server's route layout — fire-and-forget
	// to whatever URLs it was given.
	t.telemetryURL = os.Getenv("TELEMETRY_URL")
	t.telemetryLiveURL = os.Getenv("TELEMETRY_LIVE_URL")
	t.serverURL = os.Getenv("SERVER_URL")

	if t.telemetryURL != "" || t.telemetryLiveURL != "" {
		logMsg("TELEMETRY", fmt.Sprintf("telemetry URLs configured: stored=%s live=%s instanceID=%d",
			t.telemetryURL, t.telemetryLiveURL, t.instanceID))
		go t.forwardLoop()
		go t.liveForwardLoop()
	} else {
		logMsg("TELEMETRY", "no TELEMETRY_URL/TELEMETRY_LIVE_URL set — forwarding disabled")
	}

	return t
}

func (t *Telemetry) Stop() {
	t.stopOnce.Do(func() {
		if t.quit != nil {
			close(t.quit)
		}
	})
}

// generateID returns a monotonically-increasing unique event id.
// Uses atomic increment so concurrent emit calls never collide on the
// same (ms, seq) tuple — previously this was a plain `t.seq++` outside
// the mutex, which occasionally produced duplicate ids under parallel
// thread dispatch and confused the dashboard's dedup.
func (t *Telemetry) generateID() string {
	seq := atomic.AddInt64(&t.seq, 1)
	return fmt.Sprintf("%d-%d", time.Now().UnixMilli(), seq)
}

// Emit records a telemetry event (stored + forwarded to server).
func (t *Telemetry) Emit(eventType, threadID string, data any) {
	t.emit(eventType, threadID, data, true)
}

// EmitLive records a telemetry event for SSE only (not forwarded to server).
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
		logMsg("TELEMETRY", fmt.Sprintf("emit STORED %s (log=%d, url=%s)", eventType, len(t.log), t.telemetryURL))
		if len(t.log) > 2000 {
			t.log = t.log[len(t.log)-1000:]
		}
	} else {
		logMsg("TELEMETRY", fmt.Sprintf("emit LIVE-ONLY %s", eventType))
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
	t.notifyMu.Lock()
	for _, ch := range t.notifySubs {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
	t.notifyMu.Unlock()

	// Forward ALL events to server for broadcast (live display on dashboard/console)
	if t.telemetryLiveURL != "" {
		select {
		case t.forwardCh <- ev:
		default:
			// Channel full — drop to avoid blocking the thinker hot path,
			// but count drops and log periodically so the loss doesn't
			// hide. Every 50th drop is loud enough to notice in logs
			// without spamming when something goes badly wrong.
			dropped := atomic.AddInt64(&t.dropCount, 1)
			if dropped%50 == 1 {
				logMsg("TELEMETRY", fmt.Sprintf("forwardCh FULL, dropped %s (total drops: %d)", eventType, dropped))
			}
		}
	}
}

// DroppedLiveEvents returns the cumulative count of live-forward events
// that were discarded because the buffer was full. Useful for health
// checks and end-of-run diagnostics.
func (t *Telemetry) DroppedLiveEvents() int64 {
	return atomic.LoadInt64(&t.dropCount)
}

// Subscribe returns a per-listener wake channel. A shared channel is not a
// broadcast primitive: with multiple SSE clients, one notification used to
// wake only one of them.
func (t *Telemetry) Subscribe() (<-chan struct{}, func()) {
	t.notifyMu.Lock()
	t.notifySeq++
	id := t.notifySeq
	ch := make(chan struct{}, 1)
	if t.notifySubs == nil {
		t.notifySubs = make(map[uint64]chan struct{})
	}
	t.notifySubs[id] = ch
	t.notifyMu.Unlock()
	return ch, func() {
		t.notifyMu.Lock()
		delete(t.notifySubs, id)
		t.notifyMu.Unlock()
	}
}

// liveForwardLoop preserves order while coalescing token/chunk bursts into a
// single POST. This removes the previous one-HTTP-request-per-token overhead.
func (t *Telemetry) liveForwardLoop() {
	const maxBatch = 100
	const flushDelay = 25 * time.Millisecond
	for {
		select {
		case ev := <-t.forwardCh:
			batch := []TelemetryEvent{ev}
			timer := time.NewTimer(flushDelay)
		collect:
			for len(batch) < maxBatch {
				select {
				case next := <-t.forwardCh:
					batch = append(batch, next)
				case <-timer.C:
					break collect
				case <-t.quit:
					if !timer.Stop() {
						select {
						case <-timer.C:
						default:
						}
					}
					t.forwardLive(batch)
					return
				}
			}
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			t.forwardLiveWithRetry(batch)
		case <-t.quit:
			return
		}
	}
}

// forwardLiveWithRetry keeps ownership of a batch until the server accepts it.
// That preserves event order across a short server restart: newer events stay
// queued in forwardCh while the current batch retries. The queue remains
// bounded so telemetry can never block the thinker hot path indefinitely.
func (t *Telemetry) forwardLiveWithRetry(events []TelemetryEvent) {
	const (
		baseDelay = 100 * time.Millisecond
		maxDelay  = 5 * time.Second
	)
	delay := baseDelay
	for {
		if t.forwardLive(events) {
			return
		}
		timer := time.NewTimer(delay)
		select {
		case <-timer.C:
			if delay < maxDelay {
				delay *= 2
				if delay > maxDelay {
					delay = maxDelay
				}
			}
		case <-t.quit:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			return
		}
	}
}

func (t *Telemetry) forwardLive(events []TelemetryEvent) bool {
	body, err := json.Marshal(events)
	if err != nil {
		return true // deterministic failure: retrying cannot repair this batch
	}
	req, err := http.NewRequest("POST", t.telemetryLiveURL, bytes.NewReader(body))
	if err != nil {
		return true
	}
	req.Header.Set("Content-Type", "application/json")
	if t.instanceSecret != "" {
		// X-Agent-Secret is the post-rename header name; the server
		// accepts X-Instance-Secret as a fallback during the
		// deprecation window so the order of upgrades doesn't matter.
		req.Header.Set("X-Agent-Secret", t.instanceSecret)
	}
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		logMsg("TELEMETRY", fmt.Sprintf("forwardLive: POST error for %d events: %v", len(events), err))
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		logMsg("TELEMETRY", fmt.Sprintf("forwardLive: HTTP %d for %d events", resp.StatusCode, len(events)))
		return false
	}
	return true
}

// Events returns all events (including live-only) since the given index. Used by SSE.
// If the log was truncated (since > len), reset to return everything available.
func (t *Telemetry) Events(since int) ([]TelemetryEvent, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if since > len(t.liveLog) {
		since = 0
	}
	if since == len(t.liveLog) {
		return nil, len(t.liveLog)
	}
	events := make([]TelemetryEvent, len(t.liveLog)-since)
	copy(events, t.liveLog[since:])
	return events, len(t.liveLog)
}

// StoredEvents returns only stored events since the given index. Used by forwardLoop.
// If the log was truncated (since > len), reset to return everything available.
func (t *Telemetry) StoredEvents(since int) ([]TelemetryEvent, int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if since > len(t.log) {
		// Log was truncated — reset to drain everything remaining
		since = 0
	}
	if since == len(t.log) {
		return nil, len(t.log)
	}
	events := make([]TelemetryEvent, len(t.log)-since)
	copy(events, t.log[since:])
	return events, len(t.log)
}

// forwardLoop batches stored events and POSTs them to the server for DB
// persistence. On transient failures (network error, non-2xx) the cursor
// does NOT advance and the same batch is re-sent after an exponential
// backoff (1s → 30s). The server dedups by event.id PRIMARY KEY (INSERT
// OR IGNORE) so retries are safe and never double-insert.
//
// Backoff resets to the base interval on any successful POST. The stored
// event log is bounded (truncated in emit() past 2000 entries); if the
// server is unreachable long enough for truncation to run, StoredEvents()
// resets the cursor to 0 so we drain whatever remains rather than stalling.
func (t *Telemetry) forwardLoop() {
	const (
		baseInterval = 1 * time.Second
		maxInterval  = 30 * time.Second
	)
	interval := baseInterval
	timer := time.NewTimer(interval)
	defer timer.Stop()

	var lastSent int
	var consecutiveFailures int
	client := &http.Client{Timeout: 5 * time.Second}

	logMsg("TELEMETRY", fmt.Sprintf("forwardLoop started, url=%s", t.telemetryURL))

	for {
		select {
		case <-timer.C:
			// Default: reset to base cadence. Overridden below if this
			// iteration fails.
			next := baseInterval

			if t.telemetryURL == "" {
				timer.Reset(next)
				continue
			}
			events, total := t.StoredEvents(lastSent)
			if len(events) == 0 {
				timer.Reset(next)
				continue
			}

			types := make([]string, len(events))
			for i, e := range events {
				types[i] = e.Type
			}
			logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: sending %d events to %s: %v", len(events), t.telemetryURL, types))

			body, err := json.Marshal(events)
			if err != nil {
				// Marshal failure is deterministic — retrying won't help.
				// Advance past this batch so we don't spin forever.
				logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: marshal error (dropping batch): %v", err))
				lastSent = total
				timer.Reset(next)
				continue
			}

			ok := t.postBatch(client, body)
			if ok {
				lastSent = total
				if consecutiveFailures > 0 {
					logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: recovered after %d failures", consecutiveFailures))
				}
				consecutiveFailures = 0
				interval = baseInterval
			} else {
				consecutiveFailures++
				// Exponential backoff, capped. Log every failure so
				// outages are visible in the logs.
				interval *= 2
				if interval > maxInterval {
					interval = maxInterval
				}
				next = interval
				logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: retry in %s (consecutive failures=%d, queued=%d)",
					next, consecutiveFailures, len(events)))
			}
			timer.Reset(next)

		case <-t.quit:
			return
		}
	}
}

// postBatch POSTs a serialized batch of events. Returns true on 2xx.
// Any network error or non-2xx status is treated as a retryable failure —
// the caller keeps lastSent unchanged so the same batch is re-sent after
// backoff.
func (t *Telemetry) postBatch(client *http.Client, body []byte) bool {
	req, err := http.NewRequest("POST", t.telemetryURL, bytes.NewReader(body))
	if err != nil {
		logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: request error: %v", err))
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	if t.instanceSecret != "" {
		// X-Agent-Secret is the post-rename header name; the server
		// accepts X-Instance-Secret as a fallback during the
		// deprecation window so the order of upgrades doesn't matter.
		req.Header.Set("X-Agent-Secret", t.instanceSecret)
	}

	resp, err := client.Do(req)
	if err != nil {
		logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: POST error: %v", err))
		return false
	}
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		logMsg("TELEMETRY", fmt.Sprintf("forwardLoop: POST %s status=%d body=%s",
			t.telemetryURL, resp.StatusCode, string(respBody)))
		return false
	}
	return true
}

// --- Convenience emitters with typed data ---

type LLMDoneData struct {
	Model            string `json:"model"`
	Reasoning        string `json:"reasoning,omitempty"`
	TokensIn         int    `json:"tokens_in"`
	TokensCached     int    `json:"tokens_cached"`
	CacheWriteTokens int    `json:"cache_write_tokens,omitempty"`
	TokensOut        int    `json:"tokens_out"`
	DurationMs       int64  `json:"duration_ms"`
	// cost_usd is no longer populated by core — pricing lives in the
	// server, which enriches llm.done events with a canonical
	// cost_usd on ingest. Removing the field from the Go type keeps
	// core free of pricing data, but the wire format is still the
	// same map-of-strings consumed by dashboards and persisted by
	// the server.
	Iteration           int    `json:"iteration"`
	Rate                string `json:"rate"`
	ContextMsgs         int    `json:"context_msgs"`
	ContextChars        int    `json:"context_chars"`
	ContextTokensEst    int    `json:"context_tokens_est,omitempty"`
	RequestContextMsgs  int    `json:"request_context_msgs,omitempty"`
	RequestContextChars int    `json:"request_context_chars,omitempty"`
	RequestTokensEst    int    `json:"request_context_tokens_est,omitempty"`
	// MaxContextTokens is the model's advertised input-context window
	// (in tokens). Comes from a static lookup keyed on the model id —
	// see ModelContextWindow. 0 when the model isn't in the table; UI
	// should treat 0 as "unknown" and skip percentage rendering.
	MaxContextTokens             int    `json:"max_context_tokens,omitempty"`
	MemoryCount                  int    `json:"memory_count"`
	ThreadCount                  int    `json:"thread_count"`
	Message                      string `json:"message,omitempty"`
	NativeToolCount              int    `json:"native_tool_count,omitempty"`
	ActiveMCPCount               int    `json:"active_mcp_count,omitempty"`
	AlwaysMCPCount               int    `json:"always_mcp_count,omitempty"`
	DeferredMCPCount             int    `json:"deferred_mcp_count,omitempty"`
	ToolMode                     string `json:"tool_mode,omitempty"`
	PromptCacheEpoch             uint64 `json:"prompt_cache_epoch"`
	PromptCacheResetReason       string `json:"prompt_cache_reset_reason,omitempty"`
	PromptCacheIdentityHash      string `json:"prompt_cache_identity_hash,omitempty"`
	PromptCacheStablePrefixHash  string `json:"prompt_cache_stable_prefix_hash,omitempty"`
	PromptCacheRequestHash       string `json:"prompt_cache_request_hash,omitempty"`
	PromptCacheCommonPrefixBytes int    `json:"prompt_cache_common_prefix_bytes,omitempty"`
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
	MCP       []string `json:"mcp,omitempty"`
	Realtime  bool     `json:"realtime,omitempty"`
	Voice     string   `json:"voice,omitempty"`
	Provider  string   `json:"provider,omitempty"`
}

type ThreadDoneData struct {
	ParentID string `json:"parent_id"`
	Result   string `json:"result,omitempty"`
}

// ThreadRenamedData fires when update changes a thread's display name
// or its immutable id. The dashboard uses old_id to swap its record
// for the new identity (id changes are rare but legal). When only Name
// changed, OldID == NewID.
type ThreadRenamedData struct {
	OldID    string `json:"old_id"`
	NewID    string `json:"new_id"`
	Name     string `json:"name,omitempty"`
	ParentID string `json:"parent_id,omitempty"`
}

type ThreadMessageData struct {
	From    string `json:"from"`
	To      string `json:"to"`
	Message string `json:"message"`
}

type ToolCallData struct {
	ID     string            `json:"id,omitempty"`
	Name   string            `json:"name"`
	Args   map[string]string `json:"args,omitempty"`
	Reason string            `json:"reason,omitempty"`
}

type ToolResultData struct {
	ID                  string `json:"id,omitempty"`
	Name                string `json:"name"`
	DurationMs          int64  `json:"duration_ms"`
	Success             bool   `json:"success"`
	Result              string `json:"result,omitempty"`
	ResultOriginalBytes int    `json:"result_original_bytes"`
	ResultContextBytes  int    `json:"result_context_bytes"`
	ResultPreviewBytes  int    `json:"result_preview_bytes"`
	ResultImageBytes    int    `json:"result_image_bytes,omitempty"`
	ResultTruncated     bool   `json:"result_truncated"`
}

const toolResultTelemetryPreviewBytes = 1000

// newToolResultData records payload sizes without copying the complete result
// into server telemetry. ContextBytes describes the raw text+image bytes made
// available to the next model turn; it can differ from the tool's original
// output when core wraps or transforms a result.
func newToolResultData(id, name string, durationMs int64, success bool, originalText, contextText string, imageBytes int) ToolResultData {
	preview := contextText
	truncated := len(preview) > toolResultTelemetryPreviewBytes
	if truncated {
		end := toolResultTelemetryPreviewBytes
		for end > 0 && !utf8.ValidString(preview[:end]) {
			end--
		}
		preview = preview[:end] + "..."
	}
	return ToolResultData{
		ID:                  id,
		Name:                name,
		DurationMs:          durationMs,
		Success:             success,
		Result:              preview,
		ResultOriginalBytes: len(originalText) + imageBytes,
		ResultContextBytes:  len(contextText) + imageBytes,
		ResultPreviewBytes:  len(preview),
		ResultImageBytes:    imageBytes,
		ResultTruncated:     truncated,
	}
}

type DirectiveChangeData struct {
	Old string `json:"old,omitempty"`
	New string `json:"new"`
}

// ModelContextWindow returns the advertised input-context window (in
// tokens) for a given model id. Pure static lookup — no network, no
// API call, ~hundreds of nanoseconds. Returns 0 if the model isn't in
// the table; the UI treats 0 as "unknown" and skips the % display.
//
// Numbers come from each provider's own model documentation. Update
// when a new model ships or a provider changes a limit. Match is by
// substring so we tolerate the various id forms providers use
// (e.g. "claude-opus-4-7", "claude-opus-4-7-20251119", "claude-opus-4-7[1m]").
//
// Order matters: longer / more specific keys checked first so
// "claude-opus-4-7[1m]" matches the 1M variant before falling through
// to the generic "claude-opus-4-7" 200K entry.
func ModelContextWindow(modelID string) int {
	if modelID == "" {
		return 0
	}
	if caps, ok := capabilitiesForModel(modelID); ok && caps.ContextWindow > 0 {
		return caps.ContextWindow
	}
	// Order longest-prefix first to win over shorter substrings.
	table := []struct {
		match  string
		tokens int
	}{
		// --- Anthropic (Claude) ---
		// 1M-context variants are explicitly tagged.
		{"claude-opus-4-7[1m]", 1_000_000},
		{"claude-opus-4-6[1m]", 1_000_000},
		{"claude-opus-4-5[1m]", 1_000_000},
		{"claude-sonnet-4-6[1m]", 1_000_000},
		{"claude-sonnet-4-5[1m]", 1_000_000},
		// Standard 200K Claude family.
		{"claude-opus-4", 200_000},
		{"claude-sonnet-4", 200_000},
		{"claude-haiku-4", 200_000},
		{"claude-3-5-sonnet", 200_000},
		{"claude-3-5-haiku", 200_000},
		{"claude-3-opus", 200_000},
		{"claude-3-sonnet", 200_000},
		{"claude-3-haiku", 200_000},

		// --- Fireworks + OpenCode Go (Kimi / MiniMax) ---
		// Both providers expose the same Kimi K2.x and MiniMax M2.x
		// base models, just under different id forms: Fireworks uses
		// "kimi-k2p6" / "minimax-m2p7", OpenCode Go uses dotted ids
		// like "kimi-k2.6" / "minimax-m2.7". Match-by-substring means
		// dotted variants MUST come before the bare "kimi-k2" /
		// "minimax-m2" fallbacks — otherwise the bare entry shadows
		// them and we return the wrong (older, smaller) context size.
		// Kimi K2 — 256K from K2.5 onwards; bare K2 was 128K.
		{"kimi-k2.6", 256_000},
		{"kimi-k2.5", 256_000},
		{"kimi-k2p7", 256_000},
		{"kimi-k2p6", 256_000},
		{"kimi-k2p5", 256_000},
		{"kimi-k2", 128_000},
		// MiniMax M2 — 196,608 (192K) tokens per provider docs.
		{"minimax-m2.7", 196_608},
		{"minimax-m2.5", 196_608},
		{"minimax-m2p7", 196_608},
		{"minimax-m2p5", 196_608},
		{"minimax-m2", 196_608},

		// --- OpenCode Go (other model families) ---
		// Qwen3 Plus tier on OpenCode Go — Qwen3 Plus is documented
		// at 128K context window (alibabacloud / qwen docs).
		{"qwen3.6-plus", 128_000},
		{"qwen3.5-plus", 128_000},
		// GLM-5.x (Zhipu) — published 128K context window.
		{"glm-5.2", 128_000},
		{"glm-5.1", 128_000},
		{"glm-5", 128_000},
		// MiMo V2 / V2.5 — published 256K context.
		{"mimo-v2.5-pro", 256_000},
		{"mimo-v2.5", 256_000},
		{"mimo-v2-pro", 256_000},
		{"mimo-v2-omni", 256_000},
		// DeepSeek V4 — published 128K context (Anthropic-style endpoint
		// — listed here so the UI's % indicator works regardless of which
		// provider path the request takes).
		{"deepseek-v4-pro", 128_000},
		{"deepseek-v4-flash", 128_000},

		// --- Venice (https://api.venice.ai/api/v1/models) ---
		// Venice resells a rotating catalog under non-prefixed ids. The
		// authoritative context length comes from the live /models
		// response (see server/model_fetch.go:fetchVeniceModels) and is
		// what the picker shows. The static entries below are kept for
		// telemetry's % indicator on the headline house models so the
		// chart works when a request reports a model id we recognize
		// without having to round-trip a fetch.
		{"qwen3-coder-480b", 256_000},
		{"qwen3-6-27b", 256_000},
		{"qwen3-6-plus", 1_000_000},
		{"qwen3-5-9b", 256_000},
		{"qwen3-next-80b", 256_000},
		{"qwen3-vl-235b", 256_000},
		{"mistral-small-3-2", 256_000},
		{"mistral-small-2603", 256_000},
		{"venice-uncensored-1-2", 128_000},
		{"venice-uncensored-role-play", 128_000},
		{"venice-uncensored", 32_000},
		{"grok-41-fast", 1_000_000},
		{"grok-4-20", 2_000_000},
		// --- xAI direct API ---
		// Live catalog capabilities from /v1/language-models take
		// precedence; these keep telemetry useful during catalog outages.
		{"grok-4.5", 500_000},
		{"grok-4.3", 1_000_000},
		{"grok-build-0.1", 256_000},
		{"grok-4.20-0309-reasoning", 1_000_000},
		{"grok-4.20-0309-non-reasoning", 1_000_000},
		{"grok-4.20-multi-agent-0309", 1_000_000},
		{"hermes-3-llama-3.1-405b", 128_000},

		// --- OpenAI ---
		{"gpt-4.1", 1_000_000},
		{"gpt-4o-mini", 128_000},
		{"gpt-4o", 128_000},
		{"gpt-4-turbo", 128_000},
		{"gpt-4", 8_192},
		{"gpt-3.5", 16_385},
		{"o3-mini", 200_000},
		{"o3", 200_000},
		{"o1-mini", 128_000},
		{"o1", 200_000},

		// --- Google (Gemini) ---
		{"gemini-2.5-pro", 2_000_000},
		{"gemini-2.5-flash", 1_000_000},
		{"gemini-2.0-pro", 2_000_000},
		{"gemini-2.0-flash", 1_000_000},
		{"gemini-1.5-pro", 2_000_000},
		{"gemini-1.5-flash", 1_000_000},

		// --- Local / generic ---
		{"llama3.1", 128_000},
		{"llama-3.1", 128_000},
		{"llama3", 8_192},
	}
	low := strings.ToLower(modelID)
	for _, e := range table {
		if strings.Contains(low, strings.ToLower(e.match)) {
			return e.tokens
		}
	}
	return 0
}
