package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type APIServer struct {
	thinker   *Thinker
	startTime time.Time
	apiKey    string // if set, all endpoints except /health require auth
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
		next(w, r)
	}
}

func startAPI(thinker *Thinker, addr string) error {
	api := &APIServer{
		thinker:   thinker,
		startTime: time.Now(),
		apiKey:    os.Getenv("APTEVA_API_KEY"),
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
	mux.Handle("/", http.FileServer(http.Dir("web")))
	return http.ListenAndServe(addr, mux)
}

func (a *APIServer) health(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /health")
	writeJSON(w, map[string]bool{"ok": true})
}

func (a *APIServer) status(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /status")
	elapsed := time.Since(a.startTime)

	writeJSON(w, map[string]any{
		"uptime_seconds":        int(elapsed.Seconds()),
		"core_version":          Version,
		"core_build_time":       BuildTime,
		"iteration":             a.thinker.iteration,
		"rate":                  formatSleep(a.thinker.agentSleep),
		"model":                 a.thinker.model.String(),
		"reasoning":             a.thinker.agentReasoning.String(),
		"threads":               a.thinker.threads.Count() + 1, // +1 for main
		"memories":              a.thinker.memory.Count(),
		"paused":                a.thinker.paused,
		"mode":                  a.thinker.config.GetMode(),
		"execution_control":     a.thinker.executionStatus(),
		"execution_checkpoints": a.thinker.executionCheckpointMeta(),
	})
}

type threadJSON struct {
	ID        string   `json:"id"`
	Name      string   `json:"name,omitempty"`
	ParentID  string   `json:"parent_id,omitempty"`
	Depth     int      `json:"depth"`
	Directive string   `json:"directive,omitempty"`
	Tools     []string `json:"tools,omitempty"`
	MCPNames  []string `json:"mcp_names,omitempty"`
	Iteration int      `json:"iteration"`
	Rate      string   `json:"rate"`
	Model     string   `json:"model"`
	Reasoning string   `json:"reasoning,omitempty"`
	Age       string   `json:"age"`
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
	var mainMCPs []string
	for _, srv := range a.thinker.mcpServers {
		mainMCPs = append(mainMCPs, srv.GetName())
	}
	// Always include main
	out := []threadJSON{{
		ID:        "main",
		Directive: a.thinker.config.GetDirective(),
		MCPNames:  mainMCPs,
		Iteration: a.thinker.iteration,
		Rate:      a.thinker.rate.String(),
		Model:     a.thinker.model.String(),
		Reasoning: a.thinker.agentReasoning.String(),
		Age:       formatAge(time.Since(a.startTime)),
	}}

	// Recursively collect all threads (including sub-threads of leaders)
	var collectThreads func(tm *ThreadManager)
	collectThreads = func(tm *ThreadManager) {
		for _, t := range tm.List() {
			out = append(out, threadJSON{
				ID:        t.ID,
				Name:      t.Name,
				ParentID:  t.ParentID,
				Depth:     t.Depth,
				Directive: t.Directive,
				Tools:     t.Tools,
				MCPNames:  t.MCPNames,
				Iteration: t.Iteration,
				Rate:      t.Rate.String(),
				Model:     t.Model.String(),
				Reasoning: t.Reasoning.String(),
				Age:       formatAge(time.Since(t.Started)),
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

func (a *APIServer) threadAction(w http.ResponseWriter, r *http.Request) {
	// Path is /threads/{id} or /threads/{id}/context.
	path := strings.TrimPrefix(r.URL.Path, "/threads/")
	if path == "" {
		http.Error(w, "thread ID required", http.StatusBadRequest)
		return
	}
	id, sub := path, ""
	if i := strings.Index(path, "/"); i >= 0 {
		id, sub = path[:i], path[i+1:]
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
		// Copy the slice so the caller doesn't race with Run()
		// appending to it on the next iteration. Individual Message
		// fields remain shared — acceptable for a read-only inspect
		// endpoint, same convention as ThreadManager.List().
		msgs := make([]Message, len(t.messages))
		copy(msgs, t.messages)
		iter := t.iteration
		model := t.modelID()
		totalChars := 0
		for _, m := range msgs {
			totalChars += len(m.Content)
			for _, p := range m.Parts {
				totalChars += len(p.Text)
			}
		}
		composition := buildComposition(t, msgs)
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
		// Drop persisted history (session.jsonl) and rebuild an empty
		// session so subsequent iterations append to a clean file.
		if t.session != nil {
			t.session.Delete()
			t.session = NewSession(".", id)
		}
		// Reset the in-memory message slice to just the system prompt.
		// Iteration counter is preserved — the thread keeps its identity,
		// it just forgets what it was talking about.
		if len(t.messages) > 0 {
			t.messages = t.messages[:1]
		}
		logMsg("API", fmt.Sprintf("reset thread %s: history wiped, messages=[system]", id))
		writeJSON(w, map[string]any{"status": "reset", "id": id, "count": len(t.messages)})
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
		a.thinker.threads.Kill(id)
		a.thinker.config.RemoveThread(id)
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
// (conversation history) survives. The chat-handling thread uses this
// so an edit to the agent's directive reaches the live chat thread on
// the next message rather than waiting for a server restart.
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
		Directive       string   `json:"directive"`
		DirectiveSuffix string   `json:"directive_suffix"`
		Tools           []string `json:"tools,omitempty"`
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
	if err := a.thinker.threads.Update(id, "", directive, body.Tools); err != nil {
		// Update returns "thread not found" for unknown ids.
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	// Nudge the thread the same way the LLM-driven update tool does
	// (thread.go) so it notices the change mid-conversation rather than
	// silently swapping its system prompt on the next turn.
	a.thinker.threads.Send(id, "[directive updated]")
	writeJSON(w, map[string]string{"status": "updated", "id": id})
}

// spawnThread handles POST /threads/{id}. Idempotent: if the thread
// already exists, returns its current state with status="exists"
// without mutating it; if not, spawns a new one with the given
// directive + tools + mcp and returns status="created". Missing
// fields fall back to inherit-from-main: directive=main's directive,
// tools=[] (spawnInternal supplies the safe baseline: send, done,
// pace, evolve), mcp=[].
func (a *APIServer) spawnThread(w http.ResponseWriter, r *http.Request, id string) {
	if id == "main" {
		http.Error(w, "cannot spawn over main", http.StatusBadRequest)
		return
	}
	var body struct {
		Directive       string   `json:"directive"`
		DirectiveSuffix string   `json:"directive_suffix"`
		Tools           []string `json:"tools"`
		MCP             []string `json:"mcp"`
		Realtime        bool     `json:"realtime,omitempty"`
		Voice           string   `json:"voice,omitempty"`
		ProviderName    string   `json:"provider,omitempty"`
		Model           string   `json:"model,omitempty"`
		Reasoning       string   `json:"reasoning,omitempty"`
	}
	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
	}

	if t := findThinkerByID(a.thinker, id); t != nil {
		writeJSON(w, map[string]any{
			"status":    "exists",
			"id":        id,
			"iteration": t.iteration,
		})
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
	// a sub-thread (e.g. channelchat's chat-handling thread needs the
	// `channels` MCP, which is no_spawn-flagged to keep LLM-driven
	// spawn tool calls from grabbing it). BypassNoSpawn lets the
	// requested MCPs through; the LLM's spawn-tool path doesn't set
	// this, so in-agent workers still can't escalate.
	opts := SpawnOpts{
		MCPNames:      body.MCP,
		BypassNoSpawn: true,
		Realtime:      body.Realtime,
		Voice:         body.Voice,
		ProviderName:  body.ProviderName,
		Model:         strings.ToLower(strings.TrimSpace(body.Model)),
	}
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
		audioOut := make(chan []byte, 64)
		opts.AudioIn = audioIn
		opts.AudioOut = audioOut
		audioToken = registerAudioBridge(id, audioIn, audioOut)
	}

	if err := a.thinker.threads.SpawnWithOpts(id, directive, body.Tools, opts); err != nil {
		// Race: another caller spawned the same id between our
		// findThinkerByID check and the lock inside Spawn. Treat as
		// success — the caller's intent (a live thread by this name)
		// is satisfied.
		if strings.Contains(err.Error(), "already exists") {
			if body.Realtime {
				unregisterAudioBridge(id)
			}
			writeJSON(w, map[string]any{"status": "exists", "id": id})
			return
		}
		if body.Realtime {
			unregisterAudioBridge(id)
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp := map[string]any{"status": "created", "id": id}
	if body.Realtime {
		resp["audio_token"] = audioToken
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

	// Skip to current position — only stream new events, no history replay
	_, cursor := tel.Events(0)

	// Stream new events as they arrive
	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tel.notify:
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
	paused := a.thinker.paused
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
		a.thinker.config.SetExecutionControl(cfg)
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
	a.thinker.config.SetExecutionControl(cfg)
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
		Message  json.RawMessage `json:"message"`
		ThreadID string          `json:"thread_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// Parse message: string or []ContentPart
	var text string
	var parts []ContentPart

	if err := json.Unmarshal(body.Message, &text); err != nil {
		// Try array of content parts
		if err := json.Unmarshal(body.Message, &parts); err != nil {
			http.Error(w, "message must be a string or array of content parts", http.StatusBadRequest)
			return
		}
		// Extract text from parts for the event bus
		for _, p := range parts {
			if p.Type == "text" {
				text = p.Text
				break
			}
		}
	}

	if text == "" && len(parts) == 0 {
		http.Error(w, "message required", http.StatusBadRequest)
		return
	}

	threadID := body.ThreadID
	if threadID == "" {
		threadID = "main"
	}

	// Lazy auto-spawn: if the event addresses a non-main thread that
	// doesn't exist (yet), spawn it with inherit-from-main defaults
	// before publishing. Without this the bus silently drops the event
	// because nothing is subscribed under that id. Two cases this
	// handles:
	//   - a caller that addresses threads by name without first calling
	//     POST /threads/{id} (slack/email channels, ad-hoc scripts).
	//   - post-restart recovery: a persisted thread id (e.g. channelchat's
	//     stored chat-<id>) survives in the caller's DB but core's
	//     in-memory thread tree is empty after restart.
	// The "already exists" race is handled the same way as in
	// spawnThread — treat as success.
	if threadID != "main" && findThinkerByID(a.thinker, threadID) == nil && !a.thinker.bus.HasSubscriber(threadID) {
		directive := a.thinker.config.GetDirective()
		if err := a.thinker.threads.SpawnWithOpts(threadID, directive, nil, SpawnOpts{}); err != nil {
			if !strings.Contains(err.Error(), "already exists") {
				logMsg("API", fmt.Sprintf("lazy spawn %q failed: %v", threadID, err))
				http.Error(w, "failed to spawn thread: "+err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			logMsg("API", fmt.Sprintf("lazy-spawned thread %q for inbound event", threadID))
		}
	}

	if len(parts) > 0 {
		// Multimodal: publish event with parts directly on the bus
		a.thinker.bus.Publish(Event{Type: EventInbox, To: threadID, Text: "[console] " + text, Parts: parts})
	} else if threadID != "main" {
		a.thinker.bus.Publish(Event{Type: EventInbox, To: threadID, Text: text})
	} else {
		a.thinker.InjectConsole(text)
	}

	writeJSON(w, map[string]string{"status": "injected", "thread_id": threadID})
}

func (a *APIServer) config(w http.ResponseWriter, r *http.Request) {
	logMsg("API", fmt.Sprintf("%s /config", r.Method))
	switch r.Method {
	case http.MethodGet:
		// Build live provider info
		var providerInfo map[string]any
		if a.thinker.provider != nil {
			models := a.thinker.provider.Models()
			providerInfo = map[string]any{
				"name": a.thinker.provider.Name(),
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
		liveNames := make(map[string]bool, len(a.thinker.mcpServers))
		for _, srv := range a.thinker.mcpServers {
			liveNames[srv.GetName()] = true
		}
		var mcpInfo []map[string]any
		for _, cfg := range a.thinker.config.GetMCPServers() {
			entry := map[string]any{
				"name":      cfg.Name,
				"connected": liveNames[cfg.Name],
				"no_spawn":  cfg.NoSpawn,
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
		})
	case http.MethodPut:
		var body struct {
			Directive  string                  `json:"directive,omitempty"`
			Mode       RunMode                 `json:"mode,omitempty"`
			Provider   *ProviderConfig         `json:"provider,omitempty"`
			Providers  []ProviderConfig        `json:"providers,omitempty"`
			Computer   json.RawMessage         `json:"computer,omitempty"`
			MCPServers []MCPServerConfig       `json:"mcp_servers,omitempty"`
			Execution  *ExecutionControlConfig `json:"execution_control,omitempty"`
			Reset      *struct {
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
		if body.Directive != "" {
			a.thinker.config.SetDirective(body.Directive)
			a.thinker.ReloadDirectiveQuiet()
		}
		if body.Mode == ModeAutonomous || body.Mode == ModeCautious || body.Mode == ModeLearn {
			a.thinker.config.SetMode(body.Mode)
			if a.thinker.telemetry != nil {
				a.thinker.telemetry.Emit("mode.changed", "main", map[string]string{"mode": string(body.Mode)})
			}
		}
		if body.Execution != nil {
			a.thinker.config.SetExecutionControl(*body.Execution)
			if a.thinker.execution != nil {
				a.thinker.execution.ApplyConfig(*body.Execution)
			}
			if a.thinker.telemetry != nil {
				a.thinker.telemetry.Emit("execution.mode_changed", "main", a.thinker.executionStatus())
			}
		}
		logMsg("API", fmt.Sprintf("PUT /config: providers=%d provider=%v", len(body.Providers), body.Provider != nil))
		if len(body.Providers) > 0 {
			// Rebuild provider pool from new config
			logMsg("API", fmt.Sprintf("rebuilding pool with %d providers", len(body.Providers)))
			oldDefault := ""
			if a.thinker.provider != nil {
				oldDefault = a.thinker.provider.Name()
			}
			a.thinker.config.mu.Lock()
			a.thinker.config.Providers = body.Providers
			a.thinker.config.mu.Unlock()
			a.thinker.config.Save()
			pool, err := buildProviderPool(a.thinker.config)
			if err == nil && pool != nil {
				a.thinker.pool = pool
				a.thinker.provider = pool.Default()
				// Clear conversation history if provider changed (tool IDs are incompatible across providers)
				if a.thinker.provider.Name() != oldDefault {
					a.thinker.messages = a.thinker.messages[:1] // keep system prompt only
				}
			}
		}
		if body.Provider != nil {
			// Hot-swap provider if name changed
			if body.Provider.Name != "" {
				newProvider := createProviderByName(body.Provider.Name)
				if newProvider != nil {
					if body.Provider.Models != nil {
						applyModelOverrides(newProvider, body.Provider.Models)
					}
					a.thinker.provider = newProvider
					a.thinker.config.SetProvider(body.Provider)
				} else {
					http.Error(w, fmt.Sprintf("provider %q not available (missing API key?)", body.Provider.Name), http.StatusBadRequest)
					return
				}
			} else if body.Provider.Models != nil {
				// Just update models on current provider
				applyModelOverrides(a.thinker.provider, body.Provider.Models)
				// Merge into config
				for tier, modelID := range body.Provider.Models {
					a.thinker.config.SetProviderModel(tier, modelID)
				}
			}
		}
		if body.MCPServers != nil {
			a.reconcileMCP(body.MCPServers)
			// DO NOT rebuild t.mcpCatalog here — reconcileMCP already
			// manages it correctly (populates for non-main-access
			// servers in the connect pass at reconcileMCP:690, prunes
			// removed entries in the prune pass). The old code here
			// wiped the catalog and rebuilt it ONLY from t.mcpServers
			// (the main-access list), which had the effect of deleting
			// every catalog entry on every PUT /config — meaning an
			// agent whose catalog MCPs were attached at runtime never
			// saw them in its system prompt.
			//
			// Rebuild the system prompt so the updated `[AVAILABLE MCP
			// SERVERS]` block reaches the LLM on its next iteration.
			// Without this, the agent's system prompt stays frozen at
			// the boot-time state and new catalog MCPs attached via
			// dashboard are invisible to main. Use rebuildPrompt (which
			// is set up at thinker init) so all the pieces — directive,
			// core docs, providers, threads, MCPs — are consistent.
			if a.thinker.rebuildPrompt != nil {
				a.thinker.messages[0] = Message{
					Role:    "system",
					Content: a.thinker.rebuildPrompt(""),
				}
			}
		}
		if body.Reset != nil {
			logMsg("API", fmt.Sprintf("PUT /config reset: history=%v memory=%v threads=%v", body.Reset.History, body.Reset.Memory, body.Reset.Threads))
			if body.Reset.Threads {
				a.thinker.threads.KillAll()
				a.thinker.config.ClearThreads()
			}
			if body.Reset.History {
				if a.thinker.session != nil {
					a.thinker.session.Delete()
					a.thinker.session = NewSession(".", "main")
				}
				// Clear thread histories
				os.RemoveAll("history")
				os.MkdirAll("history", 0755)
			}
			if body.Reset.Memory && a.thinker.memory != nil {
				os.Remove(a.thinker.memory.path)
				a.thinker.memory.mu.Lock()
				a.thinker.memory.records = nil
				a.thinker.memory.byID = map[string]int{}
				a.thinker.memory.mu.Unlock()
			}
			// Reset message context to just system prompt
			if body.Reset.History {
				a.thinker.messages = a.thinker.messages[:1]
			}
		}
		writeJSON(w, map[string]string{"status": "updated"})
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
func (a *APIServer) reconcileMCP(desired []MCPServerConfig) {

	names := make([]string, len(desired))
	for i, c := range desired {
		names[i] = c.Name
	}
	logMsg("API", fmt.Sprintf("reconcileMCP: %d desired servers (system entries preserved): %v", len(desired), names))
	t := a.thinker

	// Current config map lets us detect when the URL/command/args/transport
	// or no_spawn changed between reconciles — any change forces a
	// detach-then-reattach.
	currentCfg := make(map[string]MCPServerConfig)
	if t.config != nil {
		for _, c := range t.config.GetMCPServers() {
			currentCfg[c.Name] = c
		}
	}

	// Index desired by name
	want := make(map[string]MCPServerConfig, len(desired))
	for _, cfg := range desired {
		want[cfg.Name] = cfg
	}

	// For each server name currently known to exist, decide whether it
	// stays as-is, gets removed, or gets replaced (close + reconnect).
	// Replacement happens when the desired config differs from the
	// current one in any connection-level field. NoSpawn is included
	// because it gates sub-thread search visibility — flipping it
	// shouldn't require a process restart.
	changed := func(old, new MCPServerConfig) bool {
		if old.URL != new.URL || old.Command != new.Command || old.Transport != new.Transport {
			return true
		}
		if old.NoSpawn != new.NoSpawn {
			return true
		}
		if len(old.Args) != len(new.Args) {
			return true
		}
		for i := range old.Args {
			if old.Args[i] != new.Args[i] {
				return true
			}
		}
		if len(old.Env) != len(new.Env) {
			return true
		}
		for k, v := range old.Env {
			if new.Env[k] != v {
				return true
			}
		}
		return false
	}

	// Disconnect servers that are either absent from desired or whose
	// config changed. System entries are never touched — they're not
	// user-editable and the desired list doesn't include them.
	var kept []MCPConn
	for _, srv := range t.mcpServers {
		name := srv.GetName()
		desiredCfg, stillWant := want[name]
		current, hasCurrent := currentCfg[name]
		if stillWant && desiredCfg.NoSpawn {
			// Host-owned MCPs (for example apteva-server and channels)
			// are injected by the server and are already live. Runtime
			// config updates may round-trip them with slightly different
			// connection details; reconnecting them can deadlock the
			// management request they are serving.
			kept = append(kept, srv)
			continue
		}
		if stillWant && !hasCurrent {
			// Some host-injected system MCPs are live without being
			// persisted in config.json. A PUT /config from the server may
			// round-trip those live entries back in `desired`; do not close
			// the very MCP connection currently serving the request just
			// because there is no persisted baseline to compare against.
			kept = append(kept, srv)
			continue
		}
		if stillWant && !changed(current, desiredCfg) {
			kept = append(kept, srv)
			continue
		}
		// Disconnect: either removed or replaced.
		srv.Close()
		t.config.RemoveMCPServer(name)
		t.registry.RemoveByMCPServer(name)
		if t.toolIndex != nil {
			t.toolIndex.Remove(name)
		}
		if t.telemetry != nil {
			t.telemetry.Emit("mcp.disconnected", "api", map[string]string{"name": name})
		}
	}
	t.mcpServers = kept

	// Index what's now live after the prune pass so the connect loop
	// doesn't reprocess servers that survived untouched.
	live := make(map[string]bool, len(kept))
	for _, srv := range kept {
		live[srv.GetName()] = true
	}

	// Connect new / replaced servers. Single path, single registration
	// shape — connectAndRegisterMCP handles registry + index together
	// so they can't drift.
	var toConnect []MCPServerConfig
	for _, cfg := range desired {
		if live[cfg.Name] {
			continue
		}
		toConnect = append(toConnect, cfg)
	}
	if len(toConnect) > 0 {
		connected := connectAndRegisterMCP(toConnect, t.registry, t.toolIndex, t.blobs)
		t.mcpServers = append(t.mcpServers, connected...)
		for _, cfg := range toConnect {
			t.config.SaveMCPServer(cfg)
		}
		if t.telemetry != nil {
			for _, srv := range connected {
				t.telemetry.Emit("mcp.connected", "api", map[string]string{"name": srv.GetName()})
			}
		}
	}
	// Refresh the prompt catalog snapshot — tool counts may have moved.
	t.mcpCatalog = computeMCPCatalog(t.toolIndex)
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
