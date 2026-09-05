package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const maxAPIRequestBytes = 16 << 20

type APIServer struct {
	thinker   *Thinker
	startTime time.Time
	apiKey    string // if set, all endpoints except /health require auth
}

type contextResetResult struct {
	Status         string `json:"status"`
	ID             string `json:"id"`
	BeforeCount    int    `json:"before_count"`
	AfterCount     int    `json:"after_count"`
	RemovedCount   int    `json:"removed_count"`
	BeforeChars    int    `json:"before_chars"`
	AfterChars     int    `json:"after_chars"`
	RemovedChars   int    `json:"removed_chars"`
	ThreadsRemoved int    `json:"threads_removed,omitempty"`
	MemoryRemoved  int    `json:"memory_removed,omitempty"`
}

// resetThinkerContext clears every request-context layer, not only the visible
// messages slice. The old reset path left ephemeral retrieval snapshots and
// prompt-cache identity behind, so a nominally clean context could still be
// assembled with state from before the reset.
func resetThinkerContext(t *Thinker) (contextResetResult, error) {
	var result contextResetResult
	err := t.mutateRuntime(func() error { var err error; result, err = resetThinkerContextNow(t); return err })
	return result, err
}
func resetThinkerContextNow(t *Thinker) (contextResetResult, error) {
	t.snapshotMu.Lock()
	defer t.snapshotMu.Unlock()
	result := contextResetResult{Status: "reset", ID: t.threadID}
	result.BeforeCount = len(t.messages)
	result.BeforeChars = contextChars(t.messages)

	if t.session != nil {
		if err := t.session.Reset(); err != nil {
			return result, fmt.Errorf("remove conversation journal: %w", err)
		}
	}
	t.invalidateTools()
	if len(t.messages) > 1 {
		t.messages = t.messages[:1]
	}
	t.requestContext.reset()
	t.resetPromptCache("manual_context_reset")
	t.toolResultMu.Lock()
	t.toolResultAge = map[string]int{}
	t.toolResultMu.Unlock()
	t.publishRuntimeStatus()
	t.contextStatus.Store(thinkerContextStatus{Messages: cloneMessages(t.messages), Composition: buildComposition(t, t.messages)})

	result.AfterCount = len(t.messages)
	result.AfterChars = contextChars(t.messages)
	result.RemovedCount = max(0, result.BeforeCount-result.AfterCount)
	result.RemovedChars = max(0, result.BeforeChars-result.AfterChars)
	return result, nil
}

// apiAuth wraps a handler with API key authentication.
func (a *APIServer) apiAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if a.apiKey != "" {
			auth := r.Header.Get("Authorization")
			if auth != "Bearer "+a.apiKey {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
		}
		if r.Body != nil {
			if r.ContentLength > maxAPIRequestBytes {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, maxAPIRequestBytes)
		}
		next(w, r)
	}
}

func startAPI(thinker *Thinker, addr string) error {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return serveAPI(thinker, listener)
}

func serveAPI(thinker *Thinker, listener net.Listener) error {
	server, err := newCoreHTTPServer(thinker)
	if err != nil {
		_ = listener.Close()
		return err
	}
	return server.Serve(listener)
}

func newCoreHTTPServer(thinker *Thinker) (*http.Server, error) {
	api := &APIServer{
		thinker:   thinker,
		startTime: time.Now(),
		apiKey:    os.Getenv("APTEVA_API_KEY"),
	}
	if api.apiKey == "" && managedCoreProcess() {
		return nil, fmt.Errorf("APTEVA_API_KEY is required for a managed core process")
	}
	if api.apiKey != "" {
		logMsg("API", "API key auth enabled")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", api.health) // always open
	mux.HandleFunc("/status", api.apiAuth(api.status))
	mux.HandleFunc("/threads", api.apiAuth(api.threads))
	mux.HandleFunc("/threads/", api.apiAuth(api.threadAction))
	// Realtime audio bridge — external callers (telephony sidecar,
	// browser bridge) connect with a single-use token issued at
	// realtime-spawn time. PCM16 binary WebSocket frames in both
	// directions.
	mux.HandleFunc("/realtime/audio", api.apiAuth(api.realtimeAudioHandler))
	mux.HandleFunc("/events", api.apiAuth(api.events))
	mux.HandleFunc("/pause", api.apiAuth(api.pause))
	mux.HandleFunc("/control", api.apiAuth(api.control))
	mux.HandleFunc("/event", api.apiAuth(api.postEvent))
	mux.HandleFunc("/event-lifecycle", api.apiAuth(api.eventLifecycle))
	mux.HandleFunc("/config", api.apiAuth(api.config))
	// Memory inspection/editing.
	//   GET  /memory               — list active memories
	//   POST /memory               — insert / upsert by id (used by the
	//                                 platform to push skill-as-memory)
	//   DELETE /memory/{idx}       — drop by zero-based index (legacy,
	//                                 still used by the dashboard)
	//   PUT    /memory/{idx}       — supersede by index
	//   DELETE /memory/by-id/{id}  — drop by ULID (preferred for
	//                                 platform-driven cleanup)
	mux.HandleFunc("/memory", api.apiAuth(api.memoryRoot))
	mux.HandleFunc("/memory/", api.apiAuth(api.memoryItem))
	mux.HandleFunc("/", api.apiAuth(http.FileServer(http.Dir("web")).ServeHTTP))
	return &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       2 * time.Minute,
		MaxHeaderBytes:    64 << 10,
	}, nil
}

func managedCoreProcess() bool {
	return os.Getenv("AGENT_ID") != "" || os.Getenv("INSTANCE_ID") != "" ||
		os.Getenv("SERVER_URL") != "" || os.Getenv("TELEMETRY_URL") != ""
}

func (a *APIServer) health(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /health")
	writeJSON(w, map[string]bool{"ok": true})
}

func (a *APIServer) status(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /status")
	elapsed := time.Since(a.startTime)
	status := a.thinker.status()

	writeJSON(w, map[string]any{
		"uptime_seconds":        int(elapsed.Seconds()),
		"core_version":          Version,
		"core_build_time":       BuildTime,
		"iteration":             status.Iteration,
		"rate":                  formatSleep(status.Sleep),
		"next_wake_at":          formatNextWakeAt(status.NextWakeAt),
		"model":                 status.Model.String(),
		"reasoning":             status.Reasoning.String(),
		"threads":               a.thinker.threads.Count() + 1, // +1 for main
		"memories":              a.thinker.memory.Count(),
		"paused":                status.Paused,
		"llm_active":            status.LLMActive,
		"mode":                  a.thinker.config.GetMode(),
		"execution_control":     a.thinker.executionStatus(),
		"execution_checkpoints": a.thinker.executionCheckpointMeta(),
	})
}

type threadJSON struct {
	ID         string   `json:"id"`
	Name       string   `json:"name,omitempty"`
	ParentID   string   `json:"parent_id,omitempty"`
	Depth      int      `json:"depth"`
	Directive  string   `json:"directive,omitempty"`
	Tools      []string `json:"tools,omitempty"`
	MCPNames   []string `json:"mcp_names,omitempty"`
	Iteration  int      `json:"iteration"`
	Rate       string   `json:"rate"`
	NextWakeAt string   `json:"next_wake_at,omitempty"`
	Model      string   `json:"model"`
	Reasoning  string   `json:"reasoning,omitempty"`
	Age        string   `json:"age"`
	Realtime   bool     `json:"realtime,omitempty"`
	Ephemeral  bool     `json:"ephemeral,omitempty"`
	Voice      string   `json:"voice,omitempty"`
	Provider   string   `json:"provider,omitempty"`
}

func (a *APIServer) threads(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /threads")
	// Main's MCP list = only the servers whose tools are actually live on
	// main's registry (t.mcpServers). Cataloged servers (t.mcpCatalog) are
	// deliberately excluded — main can't call them directly, so listing
	// them under main would mislead the user into thinking the agent is
	// using them. Sub-threads that spawn with mcp="X" are the ones that
	// actually use catalog entries, and those appear in their own rows
	// via tm.List() below with their own MCPNames populated.
	status := a.thinker.status()
	mainMCPs := append([]string(nil), status.MCPNames...)
	// Always include main
	out := []threadJSON{{
		ID:         "main",
		Directive:  a.thinker.config.GetDirective(),
		MCPNames:   mainMCPs,
		Iteration:  status.Iteration,
		Rate:       status.Rate.String(),
		NextWakeAt: formatNextWakeAt(status.NextWakeAt),
		Model:      status.Model.String(),
		Reasoning:  status.Reasoning.String(),
		Age:        formatAge(time.Since(a.startTime)),
	}}

	// Recursively collect all threads (including sub-threads of leaders)
	var collectThreads func(tm *ThreadManager)
	collectThreads = func(tm *ThreadManager) {
		for _, t := range tm.List() {
			out = append(out, threadJSON{
				ID:         t.ID,
				Name:       t.Name,
				ParentID:   t.ParentID,
				Depth:      t.Depth,
				Directive:  t.Directive,
				Tools:      t.Tools,
				MCPNames:   t.MCPNames,
				Iteration:  t.Iteration,
				Rate:       t.Rate.String(),
				NextWakeAt: formatNextWakeAt(t.NextWakeAt),
				Model:      t.Model.String(),
				Reasoning:  t.Reasoning.String(),
				Age:        formatAge(time.Since(t.Started)),
				Realtime:   t.Realtime,
				Ephemeral:  t.Ephemeral,
				Voice:      t.Voice,
				Provider:   t.Provider,
			})
			// Recurse into children
			if t.SubThreads > 0 {
				tm.mu.RLock()
				if thread, ok := tm.threads[t.ID]; ok && thread.Children != nil {
					tm.mu.RUnlock()
					collectThreads(thread.Children)
				} else {
					tm.mu.RUnlock()
				}
			}
		}
	}
	collectThreads(a.thinker.threads)
	writeJSON(w, out)
}

func formatNextWakeAt(wake time.Time) string {
	if wake.IsZero() {
		return ""
	}
	return wake.UTC().Format(time.RFC3339Nano)
}

func (a *APIServer) threadAction(w http.ResponseWriter, r *http.Request) {
	// Path is /threads/{id}, /threads/{id}/context, or /threads/{id}/history.
	path := strings.TrimPrefix(r.URL.Path, "/threads/")
	if path == "" {
		http.Error(w, "thread ID required", http.StatusBadRequest)
		return
	}
	id, sub := path, ""
	if i := strings.Index(path, "/"); i >= 0 {
		id, sub = path[:i], path[i+1:]
	}
	if id != "main" {
		if err := validateThreadID(id); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	logMsg("API", fmt.Sprintf("%s /threads/%s/%s", r.Method, id, sub))

	if sub == "context" {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}
		t := findThinkerByID(a.thinker, id)
		if t == nil {
			http.Error(w, "thread not found", http.StatusNotFound)
			return
		}
		contextStatus := t.contextSnapshot()
		status := t.status()
		msgs := contextStatus.Messages
		iter := status.Iteration
		model := status.ModelID
		totalChars := contextChars(msgs)
		composition := contextStatus.Composition
		composition.ModelMaxTokens = ModelContextWindow(model)
		writeJSON(w, map[string]any{
			"id":          id,
			"iteration":   iter,
			"model":       model,
			"count":       len(msgs),
			"total_chars": totalChars,
			"messages":    msgs,
			"composition": composition,
		})
		return
	}

	if sub == "history" {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}
		t := findThinkerByID(a.thinker, id)
		if t == nil || t.session == nil {
			http.Error(w, "thread not found", http.StatusNotFound)
			return
		}
		after, err := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
		if err != nil || after < 0 {
			http.Error(w, "after must be a non-negative integer", http.StatusBadRequest)
			return
		}
		limit, err := strconv.Atoi(r.URL.Query().Get("limit"))
		if err != nil || limit < 0 {
			http.Error(w, "limit must be a non-negative integer", http.StatusBadRequest)
			return
		}
		if limit == 0 || limit > 500 {
			limit = 500
		}
		entries, nextCursor, err := t.session.LoadAfter(after, limit)
		if err != nil {
			http.Error(w, "read thread history: "+err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{
			"id":          id,
			"after":       after,
			"next_cursor": nextCursor,
			"entries":     entries,
		})
		return
	}

	if sub == "reset" {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		t := findThinkerByID(a.thinker, id)
		if t == nil {
			http.Error(w, "thread not found", http.StatusNotFound)
			return
		}
		result, err := resetThinkerContext(t)
		if err != nil {
			http.Error(w, "reset thread context: "+err.Error(), http.StatusInternalServerError)
			return
		}
		logMsg("API", fmt.Sprintf("reset thread %s: messages %d→%d, chars %d→%d", id, result.BeforeCount, result.AfterCount, result.BeforeChars, result.AfterChars))
		writeJSON(w, result)
		return
	}

	if sub == "audio-token" {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if id == "main" {
			http.Error(w, "main is not a realtime thread", http.StatusBadRequest)
			return
		}
		token, err := a.thinker.threads.RenewRealtimeAudioToken(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		writeJSON(w, map[string]any{
			"status": "renewed", "id": id, "audio_token": token,
			"format": map[string]any{"encoding": "pcm16", "sample_rate": 24000, "channels": 1},
		})
		return
	}

	if sub != "" {
		http.Error(w, fmt.Sprintf("unknown sub-path %q", sub), http.StatusNotFound)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		if id == "main" {
			http.Error(w, "cannot kill main thread", http.StatusBadRequest)
			return
		}
		terminalReason := strings.TrimSpace(r.URL.Query().Get("reason"))
		if terminalReason == "" {
			terminalReason = "stopped"
		}
		switch terminalReason {
		case "caller_done", "carrier_error", "audio_disconnected", "server_shutdown", "stopped":
		default:
			http.Error(w, "invalid terminal reason", http.StatusBadRequest)
			return
		}
		a.thinker.threads.KillWithReason(id, terminalReason)
		if err := a.thinker.config.RemoveThread(id); err != nil {
			http.Error(w, "persist thread removal: "+err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{"status": "killed", "id": id})
	case http.MethodPost:
		a.spawnThread(w, r, id)
	case http.MethodPut:
		a.updateThread(w, r, id)
	default:
		http.Error(w, "DELETE, POST, or PUT only", http.StatusMethodNotAllowed)
	}
}

// updateThread handles PUT /threads/{id} — change a live sub-thread's
// directive (and optionally tools) WITHOUT killing it, so its session
// (conversation history) survives. This lets an authenticated caller update
// a durable event-driven thread without waiting for a process restart.
//
// Directive precedence mirrors spawnThread exactly so PUT and POST
// produce byte-identical system prompts:
//  1. body.Directive non-empty            → use verbatim
//  2. body.DirectiveSuffix non-empty       → main's directive + suffix
//  3. neither                              → main's directive verbatim
func (a *APIServer) updateThread(w http.ResponseWriter, r *http.Request, id string) {
	if id == "main" {
		http.Error(w, "cannot update main via this endpoint", http.StatusBadRequest)
		return
	}
	var body struct {
		Directive       string                  `json:"directive"`
		DirectiveSuffix string                  `json:"directive_suffix"`
		Tools           *[]string               `json:"tools,omitempty"`
		MCP             *[]string               `json:"mcp,omitempty"`
		Events          []apiThreadEventRequest `json:"events,omitempty"`
		RestartRealtime bool                    `json:"restart_realtime,omitempty"`
	}
	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
	}
	directive := body.Directive
	if directive == "" {
		directive = a.thinker.config.GetDirective()
		if body.DirectiveSuffix != "" {
			directive = directive + body.DirectiveSuffix
		}
	}
	normalizedEvents, err := normalizeAPIThreadEvents(body.Events)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var tools []string
	if body.Tools != nil {
		tools = *body.Tools
	}
	result, err := a.thinker.threads.UpdateWithOpts(id, "", directive, tools, ThreadUpdateOptions{
		RestartRealtime: body.RestartRealtime,
		ReplaceTools:    body.Tools != nil,
		MCPNames:        body.MCP,
	})
	if err != nil {
		// Update returns "thread not found" for unknown ids.
		status := http.StatusNotFound
		if errors.Is(err, ErrRealtimeConfigurationRestartRequired) {
			status = http.StatusConflict
		}
		http.Error(w, err.Error(), status)
		return
	}
	eventResult, err := a.thinker.threads.QueueEvents(id, normalizedEvents)
	if err != nil {
		writeThreadEventQueueError(w, err)
		return
	}
	// Nudge the thread the same way the LLM-driven update tool does when the
	// request did not already queue an event. An ensured event is itself the
	// wake-up; publishing the directive nudge first would reopen the original
	// empty-iteration race this endpoint is designed to close.
	if result.DirectiveChanged && len(normalizedEvents) == 0 {
		a.thinker.threads.Send(id, "[directive updated]")
	}
	response := map[string]any{
		"status": resultStatus(result.Changed, "updated", "unchanged"),
		"id":     id, "realtime_restarted": result.RealtimeRestarted,
	}
	if body.Events != nil {
		response["events"] = threadEventsResponse(eventResult)
	}
	writeJSON(w, response)
}

func resultStatus(changed bool, changedStatus, unchangedStatus string) string {
	if changed {
		return changedStatus
	}
	return unchangedStatus
}

const (
	maxThreadEventsPerRequest  = 100
	maxThreadEventIDBytes      = 256
	maxThreadEventMessageBytes = 16 << 20
)

type apiThreadEventRequest struct {
	ID             string          `json:"id"`
	Message        json.RawMessage `json:"message"`
	Type           json.RawMessage `json:"type,omitempty"`
	From           json.RawMessage `json:"from,omitempty"`
	To             json.RawMessage `json:"to,omitempty"`
	ToolResult     json.RawMessage `json:"tool_result,omitempty"`
	TrackLifecycle bool            `json:"track_lifecycle,omitempty"`
}

func parseAPIEventMessage(raw json.RawMessage) (string, []ContentPart, error) {
	if len(raw) == 0 || len(raw) > maxThreadEventMessageBytes {
		return "", nil, fmt.Errorf("message must be present and no larger than %d bytes", maxThreadEventMessageBytes)
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if strings.TrimSpace(text) == "" {
			return "", nil, fmt.Errorf("message required")
		}
		return text, nil, nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", nil, fmt.Errorf("message must be a string or array of content parts")
	}
	if len(parts) == 0 || len(parts) > 32 {
		return "", nil, fmt.Errorf("message content parts must contain between 1 and 32 items")
	}
	var textParts []string
	for i, part := range parts {
		switch part.Type {
		case "text":
			if strings.TrimSpace(part.Text) == "" {
				return "", nil, fmt.Errorf("message part %d: text required", i)
			}
			textParts = append(textParts, part.Text)
		case "image_url":
			if part.ImageURL == nil || strings.TrimSpace(part.ImageURL.URL) == "" {
				return "", nil, fmt.Errorf("message part %d: image_url.url required", i)
			}
		case "input_audio":
			if part.InputAudio == nil || strings.TrimSpace(part.InputAudio.Data) == "" || strings.TrimSpace(part.InputAudio.Format) == "" {
				return "", nil, fmt.Errorf("message part %d: input_audio data and format required", i)
			}
		case "audio_url":
			if part.AudioURL == nil || strings.TrimSpace(part.AudioURL.URL) == "" {
				return "", nil, fmt.Errorf("message part %d: audio_url.url required", i)
			}
		default:
			return "", nil, fmt.Errorf("message part %d: unsupported type %q", i, part.Type)
		}
	}
	return strings.Join(textParts, "\n"), cloneContentParts(parts), nil
}

func normalizeAPIThreadEvents(requests []apiThreadEventRequest) ([]PersistentThreadEvent, error) {
	if len(requests) > maxThreadEventsPerRequest {
		return nil, fmt.Errorf("events may contain at most %d items", maxThreadEventsPerRequest)
	}
	events := make([]PersistentThreadEvent, 0, len(requests))
	for i, request := range requests {
		if len(request.Type) > 0 || len(request.From) > 0 || len(request.To) > 0 || len(request.ToolResult) > 0 {
			return nil, fmt.Errorf("events[%d] may contain only id and message; event type, routing, sender, and tool results are internal", i)
		}
		id := strings.TrimSpace(request.ID)
		if id == "" {
			return nil, fmt.Errorf("events[%d].id required", i)
		}
		if len(id) > maxThreadEventIDBytes || strings.ContainsAny(id, "\r\n\x00") {
			return nil, fmt.Errorf("events[%d].id must be at most %d bytes and contain no control separators", i, maxThreadEventIDBytes)
		}
		text, parts, err := parseAPIEventMessage(request.Message)
		if err != nil {
			return nil, fmt.Errorf("events[%d]: %w", i, err)
		}
		events = append(events, PersistentThreadEvent{
			ID: id, Hash: threadEventHash(text, parts), Text: text, Parts: parts,
			TrackLifecycle: request.TrackLifecycle,
		})
	}
	return events, nil
}

func threadEventsResponse(result ThreadEventQueueResult) map[string]any {
	accepted := result.Accepted
	duplicates := result.Duplicate
	if accepted == nil {
		accepted = []string{}
	}
	if duplicates == nil {
		duplicates = []string{}
	}
	executions := result.Executions
	if executions == nil {
		executions = map[string]string{}
	}
	return map[string]any{"accepted": accepted, "duplicates": duplicates, "executions": executions}
}

func writeThreadEventQueueError(w http.ResponseWriter, err error) {
	var conflict *ThreadEventConflictError
	var validation *ThreadEventValidationError
	switch {
	case errors.As(err, &conflict):
		http.Error(w, err.Error(), http.StatusConflict)
	case errors.As(err, &validation):
		http.Error(w, err.Error(), http.StatusBadRequest)
	default:
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

// persistAPIThread reconciles an idempotent API spawn request with the live
// thread's durable state. Existing non-ephemeral threads are always backfilled,
// which repairs threads created by older cores that returned "created" without
// saving Config.Threads.
func (a *APIServer) persistAPIThread(id string, ephemeralRequested bool) error {
	wasEphemeral, err := a.thinker.threads.EphemeralState(id)
	if err != nil {
		return err
	}

	// Existing durable threads remain durable even if an idempotent repost
	// happens to include ephemeral=true. A genuinely ephemeral live thread is
	// only promoted when the request asks for normal persistence.
	shouldPersist := !wasEphemeral || !ephemeralRequested
	if !shouldPersist {
		return nil
	}
	state, err := a.thinker.threads.PersistentState(id)
	if err != nil {
		return err
	}
	if err := a.thinker.config.SaveThread(state); err != nil {
		return fmt.Errorf("persist thread: %w", err)
	}
	if wasEphemeral && !ephemeralRequested {
		if err := a.thinker.threads.SetEphemeral(id, false); err != nil {
			removeErr := a.thinker.config.RemoveThread(id)
			if removeErr != nil {
				return fmt.Errorf("promote live thread: %v (rollback persistence: %w)", err, removeErr)
			}
			return fmt.Errorf("promote live thread: %w", err)
		}
	}
	return nil
}

// rollbackAPICreatedThread removes every live artifact of a spawn that could
// not cross the persistence boundary. Marking it ephemeral first also removes
// any session entries it managed to create before the configuration write
// failed. Kill performs audio-bridge cleanup; the explicit unregister is an
// idempotent final guard for partially-started realtime threads.
func (a *APIServer) rollbackAPICreatedThread(id string) {
	_ = a.thinker.threads.SetEphemeral(id, true)
	a.thinker.threads.Kill(id)
	unregisterAudioBridge(id)
}

// spawnThread handles POST /threads/{id}. Idempotent: if the thread
// already exists, returns its current state with status="exists". Otherwise it
// spawns a new one with the given directive + tools + mcp and returns
// status="created". Missing
// fields fall back to inherit-from-main: directive=main's directive,
// tools=[] (spawnInternal supplies the safe baseline: send, done,
// pace, evolve), mcp=[].
func (a *APIServer) spawnThread(w http.ResponseWriter, r *http.Request, id string) {
	if id == "main" {
		http.Error(w, "cannot spawn over main", http.StatusBadRequest)
		return
	}
	if err := validateThreadID(id); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var body struct {
		Directive                  string                      `json:"directive"`
		DirectiveSuffix            string                      `json:"directive_suffix"`
		Tools                      []string                    `json:"tools"`
		MCP                        []string                    `json:"mcp"`
		Realtime                   bool                        `json:"realtime,omitempty"`
		Ephemeral                  bool                        `json:"ephemeral,omitempty"`
		Voice                      string                      `json:"voice,omitempty"`
		ProviderName               string                      `json:"provider,omitempty"`
		Model                      string                      `json:"model,omitempty"`
		Reasoning                  string                      `json:"reasoning,omitempty"`
		InitialMessage             string                      `json:"initial_message,omitempty"`
		Events                     []apiThreadEventRequest     `json:"events,omitempty"`
		BridgeDisconnectTTLSeconds int                         `json:"bridge_disconnect_ttl_seconds,omitempty"`
		TurnDetection              RealtimeTurnDetectionConfig `json:"turn_detection,omitempty"`
	}
	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
	}
	normalizedEvents, err := normalizeAPIThreadEvents(body.Events)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if t := findThinkerByID(a.thinker, id); t != nil {
		// The server may lose its in-memory spawn cache while core keeps the
		// thread alive. Preserve POST idempotency and backfill a missing
		// durable record.
		if err := a.persistAPIThread(id, body.Ephemeral); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		eventResult, err := a.thinker.threads.QueueEvents(id, normalizedEvents)
		if err != nil {
			writeThreadEventQueueError(w, err)
			return
		}
		resp := map[string]any{
			"status":    "exists",
			"id":        id,
			"iteration": t.status().Iteration,
		}
		if body.Events != nil {
			resp["events"] = threadEventsResponse(eventResult)
		}
		writeJSON(w, resp)
		return
	}

	// Directive precedence:
	//   1. body.Directive non-empty → use verbatim
	//   2. body.DirectiveSuffix non-empty → main's directive + suffix
	//      (the inherit-with-tweak path: lets a caller add a role hint
	//      without having to fetch main's directive first)
	//   3. neither → main's directive verbatim (pure inherit)
	directive := body.Directive
	if directive == "" {
		directive = a.thinker.config.GetDirective()
		if body.DirectiveSuffix != "" {
			directive = directive + body.DirectiveSuffix
		}
	}
	// API-initiated spawns are privileged: the caller authenticated
	// with the core API key, so this is the system itself asking for
	// a sub-thread. BypassNoSpawn lets explicitly requested MCPs through even
	// when they are no_spawn-flagged to keep LLM-driven spawn tool calls from
	// grabbing them. The LLM's spawn-tool path doesn't set
	// this, so in-agent workers still can't escalate.
	opts := SpawnOpts{
		MCPNames:      body.MCP,
		BypassNoSpawn: true,
		Realtime:      body.Realtime,
		Ephemeral:     body.Ephemeral,
		Voice:         body.Voice,
		TurnDetection: body.TurnDetection,
		ProviderName:  body.ProviderName,
		Model:         strings.ToLower(strings.TrimSpace(body.Model)),
		DeferRun:      true,
	}
	if body.BridgeDisconnectTTLSeconds < 0 || body.BridgeDisconnectTTLSeconds > 3600 {
		http.Error(w, "bridge_disconnect_ttl_seconds must be between 0 and 3600", http.StatusBadRequest)
		return
	}
	if !body.Realtime && !body.TurnDetection.isZero() {
		http.Error(w, "turn_detection requires realtime=true", http.StatusBadRequest)
		return
	}
	if body.Realtime {
		normalizedTurnDetection, err := body.TurnDetection.normalized()
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		opts.TurnDetection = normalizedTurnDetection
	} else {
		// Some API clients serialize optional defaults. They do not turn an
		// otherwise ordinary thread into a realtime thread.
		opts.TurnDetection = RealtimeTurnDetectionConfig{}
	}
	if body.BridgeDisconnectTTLSeconds > 0 {
		opts.BridgeDisconnectTTL = time.Duration(body.BridgeDisconnectTTLSeconds) * time.Second
	}
	opts.InitialMessage = strings.TrimSpace(body.InitialMessage)
	if rawReasoning := strings.TrimSpace(body.Reasoning); rawReasoning != "" {
		r, ok := parseReasoningLevel(rawReasoning)
		if !ok {
			http.Error(w, "invalid reasoning", http.StatusBadRequest)
			return
		}
		opts.Reasoning = r
	}
	if opts.Model != "" {
		if _, ok := modelNames[opts.Model]; !ok {
			http.Error(w, "invalid model", http.StatusBadRequest)
			return
		}
	}

	// Realtime spawns also need audio channels so an external caller
	// (telephony sidecar, browser bridge) can feed/drain PCM. We
	// create the channels here and stash them by thread id so the
	// audio bridge handler can pick them up when its WebSocket
	// connects. The single-use token returned in the response gates
	// that lookup.
	var audioToken string
	if body.Realtime {
		if a.thinker.config == nil || !a.thinker.config.RealtimeEnabledFlag() {
			http.Error(w, "realtime spawn refused: realtime_enabled=false on this instance", http.StatusForbidden)
			return
		}
		audioIn := make(chan []byte, 64)
		audioOut := make(chan RealtimeAudioFrame, 64)
		audioControl := make(chan string, 8)
		opts.AudioIn = audioIn
		opts.AudioOut = audioOut
		opts.AudioControl = audioControl
	}

	if err := a.thinker.mutateRuntime(func() error { return a.thinker.threads.SpawnWithOpts(id, directive, body.Tools, opts) }); err != nil {
		// Race: another caller spawned the same id between our
		// findThinkerByID check and the lock inside Spawn. Treat as
		// success — the caller's intent (a live thread by this name)
		// is satisfied.
		if strings.Contains(err.Error(), "already exists") {
			if err := a.persistAPIThread(id, body.Ephemeral); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			eventResult, queueErr := a.thinker.threads.QueueEvents(id, normalizedEvents)
			if queueErr != nil {
				writeThreadEventQueueError(w, queueErr)
				return
			}
			resp := map[string]any{"status": "exists", "id": id}
			if body.Events != nil {
				resp["events"] = threadEventsResponse(eventResult)
			}
			writeJSON(w, resp)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := a.persistAPIThread(id, body.Ephemeral); err != nil {
		a.rollbackAPICreatedThread(id)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	eventResult, err := a.thinker.threads.QueueEvents(id, normalizedEvents)
	if err != nil {
		a.rollbackAPICreatedThread(id)
		writeThreadEventQueueError(w, err)
		return
	}
	// Publish a bridge token only after a persistent thread has crossed its
	// durability boundary (or immediately for an explicitly ephemeral one).
	// This avoids leaking a token — or unregistering an existing thread's live
	// bridge — when a concurrent duplicate or persistence failure occurs.
	if body.Realtime {
		audioToken = registerAudioBridge(id, opts.AudioIn, opts.AudioOut, opts.AudioControl)
	}
	if err := a.thinker.threads.Start(id); err != nil {
		a.rollbackAPICreatedThread(id)
		http.Error(w, "start thread: "+err.Error(), http.StatusInternalServerError)
		return
	}
	resp := map[string]any{"status": "created", "id": id}
	if body.Events != nil {
		resp["events"] = threadEventsResponse(eventResult)
	}
	if body.Realtime {
		resp["audio_token"] = audioToken
		resp["format"] = map[string]any{"encoding": "pcm16", "sample_rate": 24000, "channels": 1}
	}
	writeJSON(w, resp)
}

// findThinkerByID returns main or any sub-thread's Thinker by id, or nil.
func findThinkerByID(main *Thinker, id string) *Thinker {
	if id == "main" || id == main.threadID {
		return main
	}
	if main.threads == nil {
		return nil
	}
	return findInManager(main.threads, id)
}

func findInManager(tm *ThreadManager, id string) *Thinker {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	for _, t := range tm.threads {
		if t.ID == id {
			return t.Thinker
		}
		if t.Children != nil {
			if th := findInManager(t.Children, id); th != nil {
				return th
			}
		}
	}
	return nil
}

func (a *APIServer) events(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /events (SSE connect)")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	tel := a.thinker.telemetry
	notify, unsubscribe := tel.Subscribe()
	defer unsubscribe()

	// Skip to current position — only stream new events, no history replay
	_, cursor := tel.Events(0)

	// Stream new events as they arrive
	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-notify:
			newEvents, newCursor := tel.Events(cursor)
			cursor = newCursor
			for _, ev := range newEvents {
				data, _ := json.Marshal(ev)
				fmt.Fprintf(w, "data: %s\n\n", data)
				// Flush each event immediately for real-time streaming
				flusher.Flush()
			}
		}
	}
}

func (a *APIServer) pause(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "POST /pause")
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	a.thinker.TogglePause()
	paused := a.thinker.status().Paused
	if paused {
		a.thinker.telemetry.Emit("instance.paused", "main", map[string]string{"status": "paused"})
	} else {
		a.thinker.telemetry.Emit("instance.resumed", "main", map[string]string{"status": "running"})
	}
	writeJSON(w, map[string]bool{"paused": paused})
}

func (a *APIServer) control(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "POST /control")
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body ExecutionControlAction
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if a.thinker.execution == nil {
		a.thinker.execution = NewExecutionController(a.thinker.config.GetExecutionControl())
	}
	if strings.ToLower(strings.TrimSpace(body.Action)) == "restore_checkpoint" {
		meta, err := a.thinker.restoreExecutionCheckpoint(body.CheckpointID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		cfg := a.thinker.execution.Config()
		cfg.Mode = ExecutionStep
		if body.Mode == ExecutionPaused || body.Mode == ExecutionAuto {
			cfg.Mode = body.Mode
		}
		a.thinker.execution.ApplyConfig(cfg)
		if err := a.thinker.config.SetExecutionControl(cfg); err != nil {
			http.Error(w, "persist execution control: "+err.Error(), http.StatusInternalServerError)
			return
		}
		status := a.thinker.executionStatus()
		if a.thinker.telemetry != nil {
			a.thinker.telemetry.Emit("execution.mode_changed", "main", status)
		}
		writeJSON(w, map[string]any{
			"status":            "restored",
			"checkpoint":        meta,
			"execution_control": status,
		})
		return
	}
	status, err := a.thinker.execution.Control(body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cfg := a.thinker.execution.Config()
	if err := a.thinker.config.SetExecutionControl(cfg); err != nil {
		http.Error(w, "persist execution control: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if a.thinker.telemetry != nil {
		a.thinker.telemetry.Emit("execution.mode_changed", "main", status)
	}
	writeJSON(w, map[string]any{"status": "accepted", "execution_control": status})
}

func (a *APIServer) postEvent(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "POST /event")
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Message        json.RawMessage `json:"message"`
		ThreadID       string          `json:"thread_id"`
		EventID        string          `json:"event_id,omitempty"`
		TrackLifecycle bool            `json:"track_lifecycle,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// Parse exactly the same safe text/multimodal envelope accepted by the
	// thread-creation events field.
	text, parts, err := parseAPIEventMessage(body.Message)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	threadID := body.ThreadID
	if threadID == "" {
		threadID = "main"
	}
	if threadID != "main" {
		if err := validateThreadID(threadID); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	eventID := strings.TrimSpace(body.EventID)
	if body.TrackLifecycle && eventID == "" {
		http.Error(w, "track_lifecycle requires event_id", http.StatusBadRequest)
		return
	}
	if eventID != "" && (len(eventID) > maxThreadEventIDBytes || strings.ContainsAny(eventID, "\r\n\x00")) {
		http.Error(w, fmt.Sprintf("event_id must be at most %d bytes and contain no control separators", maxThreadEventIDBytes), http.StatusBadRequest)
		return
	}

	// Lazy auto-spawn: if the event addresses a non-main thread that
	// doesn't exist (yet), spawn it with inherit-from-main defaults
	// before publishing. Without this the bus silently drops the event
	// because nothing is subscribed under that id. Two cases this
	// handles:
	//   - a caller that addresses threads by name without first calling
	//     POST /threads/{id} (slack/email channels, ad-hoc scripts).
	//   - post-restart recovery: a persisted thread id survives in the caller's
	//     data store but core's
	//     in-memory thread tree is empty after restart.
	// The "already exists" race is handled the same way as in
	// spawnThread — treat as success.
	if threadID != "main" && findThinkerByID(a.thinker, threadID) == nil && !a.thinker.bus.HasSubscriber(threadID) {
		directive := a.thinker.config.GetDirective()
		created := false
		if err := a.thinker.mutateRuntime(func() error { return a.thinker.threads.SpawnWithOpts(threadID, directive, nil, SpawnOpts{}) }); err != nil {
			if !strings.Contains(err.Error(), "already exists") {
				logMsg("API", fmt.Sprintf("lazy spawn %q failed: %v", threadID, err))
				http.Error(w, "failed to spawn thread: "+err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			created = true
			logMsg("API", fmt.Sprintf("lazy-spawned thread %q for inbound event", threadID))
		}
		if err := a.persistAPIThread(threadID, false); err != nil {
			if created {
				a.rollbackAPICreatedThread(threadID)
			}
			http.Error(w, "persist lazy-spawned thread: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if eventID != "" {
		eventText := text
		if threadID == "main" {
			eventText = "[console] " + text
		}
		event := PersistentThreadEvent{
			ID: eventID, Text: eventText, Parts: parts,
			Hash: threadEventHash(eventText, parts), TrackLifecycle: body.TrackLifecycle,
		}
		var result ThreadEventQueueResult
		if threadID == "main" {
			result, err = a.thinker.QueueMainEvents([]PersistentThreadEvent{event})
		} else {
			result, err = a.thinker.threads.QueueEvents(threadID, []PersistentThreadEvent{event})
		}
		if err != nil {
			writeThreadEventQueueError(w, err)
			return
		}
		writeJSON(w, map[string]any{"status": "queued", "thread_id": threadID, "events": threadEventsResponse(result)})
		return
	}

	if threadID == "main" || len(parts) > 0 {
		text = "[console] " + text
	}
	if !a.thinker.bus.TryPublish(Event{Type: EventInbox, To: threadID, Text: text, Parts: parts}) {
		http.Error(w, "inbox full or unavailable; retry later", http.StatusTooManyRequests)
		return
	}

	writeJSON(w, map[string]string{"status": "injected", "thread_id": threadID})
}

func (a *APIServer) eventLifecycle(w http.ResponseWriter, r *http.Request) {
	if a.thinker == nil || a.thinker.eventLifecycle == nil {
		http.Error(w, "event lifecycle unavailable", http.StatusServiceUnavailable)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, map[string]any{"transitions": a.thinker.eventLifecycle.PendingTransitions()})
	case http.MethodPost:
		var body struct {
			AckIDs []string `json:"ack_ids"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if err := a.thinker.eventLifecycle.Acknowledge(body.AckIDs); err != nil {
			http.Error(w, "ack lifecycle transitions: "+err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"status": "acknowledged", "acknowledged": len(body.AckIDs)})
	default:
		http.Error(w, "GET or POST only", http.StatusMethodNotAllowed)
	}
}

func (a *APIServer) config(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPut {
		if err := a.thinker.mutateRuntime(func() error { a.configNow(w, r); return nil }); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
		}
		return
	}
	a.configNow(w, r)
}
func (a *APIServer) configNow(w http.ResponseWriter, r *http.Request) {
	logMsg("API", fmt.Sprintf("%s /config", r.Method))
	switch r.Method {
	case http.MethodGet:
		// Build live provider info
		var providerInfo map[string]any
		status := a.thinker.status()
		if status.Provider != "" {
			models := status.ProviderModels
			providerInfo = map[string]any{
				"name": status.Provider,
				"models": map[string]string{
					"large": models[ModelLarge],
					"small": models[ModelSmall],
				},
			}
		}
		// Build live MCP server info. Every configured MCP is connected
		// up-front (no main/catalog split anymore), so "connected" reflects
		// the actual live state. The legacy `main_access` field is no
		// longer surfaced — the dashboard's filter UI for it should be
		// retired alongside this change.
		liveNames := make(map[string]bool, len(status.MCPNames))
		for _, name := range status.MCPNames {
			liveNames[name] = true
		}
		var mcpInfo []map[string]any
		for _, cfg := range a.thinker.config.GetMCPServers() {
			entry := map[string]any{
				"name":      cfg.Name,
				"connected": liveNames[cfg.Name],
				"no_spawn":  cfg.NoSpawn,
			}
			if cfg.ToolLoading != nil {
				entry["tool_loading"] = cfg.ToolLoading
			}
			if cfg.Transport != "" {
				entry["transport"] = cfg.Transport
			}
			if cfg.URL != "" {
				entry["url"] = cfg.URL
			}
			if cfg.Command != "" {
				entry["command"] = cfg.Command
			}
			mcpInfo = append(mcpInfo, entry)
		}

		writeJSON(w, map[string]any{
			"directive":             a.thinker.config.GetDirective(),
			"mode":                  a.thinker.config.GetMode(),
			"provider":              providerInfo,
			"providers":             a.thinker.config.GetProviders(),
			"mcp_servers":           mcpInfo,
			"execution_control":     a.thinker.executionStatus(),
			"execution_checkpoints": a.thinker.executionCheckpointMeta(),
			"realtime_enabled":      a.thinker.config.RealtimeEnabledFlag(),
			"realtime_voice":        a.thinker.config.GetRealtimeVoice(),
			"realtime_voice_mcp":    a.thinker.config.GetRealtimeVoiceMCP(),
		})
	case http.MethodPut:
		var body struct {
			Directive        string                  `json:"directive,omitempty"`
			Mode             RunMode                 `json:"mode,omitempty"`
			Provider         *ProviderConfig         `json:"provider,omitempty"`
			Providers        []ProviderConfig        `json:"providers,omitempty"`
			Computer         json.RawMessage         `json:"computer,omitempty"`
			MCPServers       []MCPServerConfig       `json:"mcp_servers,omitempty"`
			Execution        *ExecutionControlConfig `json:"execution_control,omitempty"`
			RealtimeEnabled  *bool                   `json:"realtime_enabled,omitempty"`
			RealtimeVoice    *string                 `json:"realtime_voice,omitempty"`
			RealtimeVoiceMCP *[]string               `json:"realtime_voice_mcp,omitempty"`
			Reset            *struct {
				History bool `json:"history,omitempty"`
				Memory  bool `json:"memory,omitempty"`
				Threads bool `json:"threads,omitempty"`
			} `json:"reset,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if len(body.Computer) > 0 {
			http.Error(w, "core computer config has been removed; use the Computer app MCP tools instead", http.StatusGone)
			return
		}
		if body.Mode != "" && body.Mode != ModeAutonomous && body.Mode != ModeCautious && body.Mode != ModeLearn {
			http.Error(w, "invalid mode", http.StatusBadRequest)
			return
		}
		for _, server := range body.MCPServers {
			if err := validateMCPToolLoading(server); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		if body.RealtimeVoice != nil {
			candidate := &Config{}
			if err := candidate.SetRealtimeVoice(*body.RealtimeVoice); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		if len(body.Providers) > 0 || body.Provider != nil || body.RealtimeEnabled != nil {
			providers := a.thinker.config.GetProviders()
			if len(body.Providers) > 0 {
				providers = cloneProviderConfigs(body.Providers)
			}
			if body.Provider != nil {
				providers = mergeProviderConfig(providers, *body.Provider)
			}
			enabled := a.thinker.config.RealtimeEnabledFlag()
			if body.RealtimeEnabled != nil {
				enabled = *body.RealtimeEnabled
			}
			if _, err := buildProviderPool(&Config{Providers: providers, RealtimeEnabled: enabled}); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}

		cfg := a.thinker.config
		providers := cfg.GetProviders()
		providerChanged := len(body.Providers) > 0 || body.Provider != nil || body.RealtimeEnabled != nil
		var pool *ProviderPool
		enabled := cfg.RealtimeEnabledFlag()
		if body.RealtimeEnabled != nil {
			enabled = *body.RealtimeEnabled
		}
		if len(body.Providers) > 0 {
			providers = cloneProviderConfigs(body.Providers)
		}
		if body.Provider != nil {
			providers = mergeProviderConfig(providers, *body.Provider)
		}
		if providerChanged {
			var err error
			pool, err = buildProviderPool(&Config{Providers: providers, RealtimeEnabled: enabled})
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		commit := func(mcp []MCPServerConfig) error {
			return cfg.update(func() {
				if body.Directive != "" {
					cfg.Directive = body.Directive
				}
				if body.Mode != "" {
					cfg.Mode = body.Mode
				}
				if body.Execution != nil {
					cfg.Execution = *body.Execution
				}
				if body.RealtimeVoice != nil {
					cfg.RealtimeVoice = strings.TrimSpace(*body.RealtimeVoice)
				}
				if body.RealtimeVoiceMCP != nil {
					cfg.RealtimeVoiceMCP = compactStringList(*body.RealtimeVoiceMCP)
				}
				if providerChanged {
					cfg.Providers = providers
					cfg.Provider = nil
					cfg.RealtimeEnabled = enabled
				}
				if body.MCPServers != nil {
					cfg.MCPServers = mcp
				}
			})
		}
		var commitErr error
		if body.MCPServers != nil {
			commitErr = a.reconcileMCPTransaction(body.MCPServers, commit)
		} else {
			commitErr = commit(nil)
		}
		if commitErr != nil {
			http.Error(w, "apply configuration: "+commitErr.Error(), http.StatusInternalServerError)
			return
		}
		if providerChanged {
			a.thinker.pool = pool
			a.thinker.provider = pool.Default()
		}
		if body.Execution != nil && a.thinker.execution != nil {
			a.thinker.execution.ApplyConfig(*body.Execution)
		}
		if body.Directive != "" || body.Mode != "" || providerChanged || body.MCPServers != nil {
			a.thinker.reloadDirectiveNow()
		}
		a.thinker.publishRuntimeStatus()
		a.thinker.publishContextStatus()
		if body.Mode != "" && a.thinker.telemetry != nil {
			a.thinker.telemetry.Emit("mode.changed", "main", map[string]string{"mode": string(body.Mode)})
		}

		var resetResult *contextResetResult
		if body.Reset != nil {
			logMsg("API", fmt.Sprintf("PUT /config reset: history=%v memory=%v threads=%v", body.Reset.History, body.Reset.Memory, body.Reset.Threads))
			result := contextResetResult{Status: "reset", ID: "main"}
			if body.Reset.Threads {
				result.ThreadsRemoved = a.thinker.threads.Count()
				if err := a.thinker.config.ClearThreads(); err != nil {
					http.Error(w, "persist thread reset: "+err.Error(), http.StatusInternalServerError)
					return
				}
				a.thinker.threads.KillAll()
			}
			if body.Reset.History {
				// Agent-wide history reset includes every persisted sub-thread
				// journal and archived tool result. Recreate the private directory
				// before the live main Session is reused below.
				if err := os.RemoveAll(historyDir); err != nil {
					http.Error(w, "remove agent history: "+err.Error(), http.StatusInternalServerError)
					return
				}
				if err := os.MkdirAll(historyDir, 0700); err != nil {
					http.Error(w, "recreate agent history: "+err.Error(), http.StatusInternalServerError)
					return
				}
				contextResult, err := resetThinkerContextNow(a.thinker)
				if err != nil {
					http.Error(w, "reset agent context: "+err.Error(), http.StatusInternalServerError)
					return
				}
				result.BeforeCount = contextResult.BeforeCount
				result.AfterCount = contextResult.AfterCount
				result.RemovedCount = contextResult.RemovedCount
				result.BeforeChars = contextResult.BeforeChars
				result.AfterChars = contextResult.AfterChars
				result.RemovedChars = contextResult.RemovedChars
			}
			if body.Reset.Memory && a.thinker.memory != nil {
				result.MemoryRemoved = a.thinker.memory.Count()
				if err := a.thinker.memory.Reset(); err != nil {
					http.Error(w, "reset memory: "+err.Error(), http.StatusInternalServerError)
					return
				}
			}
			resetResult = &result
		}
		response := map[string]any{"status": "updated"}
		if resetResult != nil {
			response["reset"] = resetResult
		}
		writeJSON(w, response)
	default:
		http.Error(w, "GET or PUT only", http.StatusMethodNotAllowed)
	}
}

// reconcileMCP diffs the desired MCP server list against the live
// state, connecting new servers, disconnecting removed ones, and
// replacing servers whose connection details (URL, command, args,
// transport, no_spawn) changed.
//
// Every MCP attached here lives in t.mcpServers with an open
// connection and tools registered as MCP=true (hidden from per-turn
// tool list until activated). The legacy split between "main" servers
// (eagerly visible) and "catalog" servers (connected per-thread on
// demand) is gone — see the proposal note in mcp.go's MCPServerConfig
// comment.
func (a *APIServer) reconcileMCP(desired []MCPServerConfig) error {
	return a.reconcileMCPTransaction(desired, func(next []MCPServerConfig) error {
		return a.thinker.config.update(func() { a.thinker.config.MCPServers = next })
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// memoryListItem is the UI-facing projection of an active memory.
// Embeddings are omitted (~6KB per entry, useless to the UI). Index
// is attached so the dashboard's existing DELETE/PUT-by-index flow
// continues to work; ID is also exposed for callers that want to
// address by id directly.
type memoryListItem struct {
	Index  int       `json:"index"`
	ID     string    `json:"id"`
	Text   string    `json:"text"` // = MemoryRecord.Content (kept as `text` for backward UI compat)
	Tags   []string  `json:"tags,omitempty"`
	Weight float64   `json:"weight,omitempty"`
	Time   time.Time `json:"time"` // = MemoryRecord.TS
}

// memoryRoot dispatches /memory by method:
//
//	GET  → list active memories (legacy memoryList behaviour)
//	POST → insert or upsert by id
//
// The platform uses POST to push skill-as-memory records with
// deterministic ids, so re-pushing the same skill upserts via
// Supersede instead of duplicating.
func (a *APIServer) memoryRoot(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		a.memoryList(w, r)
	case http.MethodPost:
		a.memoryUpsert(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// GET /memory — return every memory entry in store order.
func (a *APIServer) memoryList(w http.ResponseWriter, r *http.Request) {
	if a.thinker.memory == nil {
		writeJSON(w, []memoryListItem{})
		return
	}
	active := a.thinker.memory.Active()
	out := make([]memoryListItem, len(active))
	for i, r := range active {
		out[i] = memoryListItem{
			Index:  i,
			ID:     r.ID,
			Text:   r.Content,
			Tags:   r.Tags,
			Weight: r.Weight,
			Time:   r.TS,
		}
	}
	writeJSON(w, out)
}

// POST /memory — insert or upsert.
//
// Body: {id?, content, tags?, weight?}
//
// If id is present and matches an existing record, supersede it.
// If id is present and unused, insert with that id.
// If id is absent, insert with a freshly-minted ULID.
//
// Returns {id, action: "inserted"|"upserted"}. The action field lets
// the platform tell whether a push created a new record or replaced
// an existing one — useful for logging and drift dashboards.
func (a *APIServer) memoryUpsert(w http.ResponseWriter, r *http.Request) {
	if a.thinker.memory == nil {
		http.Error(w, "memory store not initialized", http.StatusServiceUnavailable)
		return
	}
	var body struct {
		ID      string   `json:"id"`
		Content string   `json:"content"`
		Tags    []string `json:"tags"`
		Weight  float64  `json:"weight"`
		Reason  string   `json:"reason"` // optional supersede reason
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(body.Content) == "" {
		http.Error(w, "content required", http.StatusBadRequest)
		return
	}
	id := strings.TrimSpace(body.ID)
	if id == "" {
		newID, err := a.thinker.memory.Remember(body.Content, body.Tags, body.Weight)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"id": newID, "action": "inserted"})
		return
	}
	if targetID, ok := a.thinker.memory.UpsertTargetID(id); ok {
		reason := body.Reason
		if reason == "" {
			reason = "upsert via POST /memory"
		}
		newID, err := a.thinker.memory.Supersede(targetID, body.Content, body.Tags, body.Weight, reason)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"id": newID, "supersedes": targetID, "source_id": id, "action": "upserted"})
		return
	}
	newID, err := a.thinker.memory.RememberWithID(id, body.Content, body.Tags, body.Weight)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, map[string]any{"id": newID, "action": "inserted"})
}

// /memory/{index}            — DELETE prunes by index, PUT supersedes by index
// /memory/by-id/{id}         — DELETE drops by ULID
func (a *APIServer) memoryItem(w http.ResponseWriter, r *http.Request) {
	if a.thinker.memory == nil {
		http.Error(w, "memory store not initialized", http.StatusServiceUnavailable)
		return
	}
	rest := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/memory/"), "/")
	if rest == "" {
		http.Error(w, "index or id required", http.StatusBadRequest)
		return
	}

	// /memory/by-id/{id} — id-addressed delete. The platform uses this
	// to remove specific skill-as-memory records without racing against
	// concurrent index shifts.
	if strings.HasPrefix(rest, "by-id/") {
		id := strings.TrimPrefix(rest, "by-id/")
		if id == "" {
			http.Error(w, "id required", http.StatusBadRequest)
			return
		}
		if r.Method != http.MethodDelete {
			http.Error(w, "DELETE only", http.StatusMethodNotAllowed)
			return
		}
		if !a.thinker.memory.HasID(id) {
			// Idempotent: deleting a non-existent id is a no-op success.
			// The platform may push DELETEs after a record was already
			// dropped (e.g. by an operator); a 404 here would force
			// callers to track state we already track.
			writeJSON(w, map[string]any{"ok": true, "count": a.thinker.memory.Count(), "noop": true})
			return
		}
		reason := r.URL.Query().Get("reason")
		if reason == "" {
			reason = "deleted via DELETE /memory/by-id"
		}
		if err := a.thinker.memory.Drop(id, reason); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"ok": true, "count": a.thinker.memory.Count()})
		return
	}

	idxStr := rest
	var idx int
	if _, err := fmt.Sscanf(idxStr, "%d", &idx); err != nil {
		http.Error(w, "invalid index", http.StatusBadRequest)
		return
	}

	// Translate the legacy index addressing into the new id-based API.
	// The dashboard still passes indices; we resolve to the active
	// record at that position. Out-of-range indices silent no-op for
	// DELETE (matches old behaviour) and 404 for PUT.
	active := a.thinker.memory.Active()
	if idx < 0 || idx >= len(active) {
		if r.Method == http.MethodDelete {
			writeJSON(w, map[string]any{"ok": true, "count": a.thinker.memory.Count()})
			return
		}
		http.Error(w, "index out of range", http.StatusNotFound)
		return
	}
	target := active[idx]

	switch r.Method {
	case http.MethodDelete:
		_ = a.thinker.memory.Drop(target.ID, "deleted via dashboard")
		writeJSON(w, map[string]any{"ok": true, "count": a.thinker.memory.Count()})

	case http.MethodPut:
		var body struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		text := strings.TrimSpace(body.Text)
		if text == "" {
			http.Error(w, "text required", http.StatusBadRequest)
			return
		}
		if _, err := a.thinker.memory.Supersede(target.ID, text, target.Tags, target.Weight, "edited via dashboard"); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"ok": true})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}
