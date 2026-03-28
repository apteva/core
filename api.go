package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type APIServer struct {
	thinker   *Thinker
	startTime time.Time
}

func startAPI(thinker *Thinker, addr string) error {
	api := &APIServer{thinker: thinker, startTime: time.Now()}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", api.health)
	mux.HandleFunc("/status", api.status)
	mux.HandleFunc("/threads", api.threads)
	mux.HandleFunc("/events", api.events)
	mux.HandleFunc("/pause", api.pause)
	mux.HandleFunc("/event", api.postEvent)
	mux.HandleFunc("/config", api.config)
	mux.HandleFunc("/approve", api.approve)
	mux.Handle("/", http.FileServer(http.Dir("web")))
	return http.ListenAndServe(addr, mux)
}

func (a *APIServer) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]bool{"ok": true})
}

func (a *APIServer) status(w http.ResponseWriter, r *http.Request) {
	elapsed := time.Since(a.startTime)
	// Check for pending approval
	a.thinker.pendingMu.Lock()
	var pending *ToolCallData
	if a.thinker.pendingTool != nil {
		pending = &ToolCallData{
			Name: a.thinker.pendingTool.Name,
			Args: toolArgsSummary(*a.thinker.pendingTool),
		}
	}
	a.thinker.pendingMu.Unlock()

	writeJSON(w, map[string]any{
		"uptime_seconds":   int(elapsed.Seconds()),
		"iteration":        a.thinker.iteration,
		"rate":             formatSleep(a.thinker.agentSleep),
		"model":            a.thinker.model.String(),
		"threads":          a.thinker.threads.Count() + 1, // +1 for main
		"memories":         a.thinker.memory.Count(),
		"paused":           a.thinker.paused,
		"mode":             a.thinker.config.GetMode(),
		"pending_approval": pending,
	})
}

type threadJSON struct {
	ID        string   `json:"id"`
	Directive string   `json:"directive,omitempty"`
	Tools     []string `json:"tools,omitempty"`
	Iteration int      `json:"iteration"`
	Rate      string   `json:"rate"`
	Model     string   `json:"model"`
	Age       string   `json:"age"`
}

func (a *APIServer) threads(w http.ResponseWriter, r *http.Request) {
	// Always include main
	out := []threadJSON{{
		ID:        "main",
		Iteration: a.thinker.iteration,
		Rate:      a.thinker.rate.String(),
		Model:     a.thinker.model.String(),
		Age:       formatAge(time.Since(a.startTime)),
	}}

	for _, t := range a.thinker.threads.List() {
		out = append(out, threadJSON{
			ID:        t.ID,
			Directive: t.Directive,
			Tools:     t.Tools,
			Iteration: t.Iteration,
			Rate:      t.Rate.String(),
			Model:     t.Model.String(),
			Age:       formatAge(time.Since(t.Started)),
		})
	}
	writeJSON(w, out)
}

func (a *APIServer) events(w http.ResponseWriter, r *http.Request) {
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
			}
			flusher.Flush()
		}
	}
}

func (a *APIServer) pause(w http.ResponseWriter, r *http.Request) {
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

func (a *APIServer) postEvent(w http.ResponseWriter, r *http.Request) {
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
		writeJSON(w, map[string]any{
			"directive":    a.thinker.config.GetDirective(),
			"mode":         a.thinker.config.GetMode(),
			"auto_approve": a.thinker.config.AutoApprove,
			"provider":     providerInfo,
		})
	case http.MethodPut:
		var body struct {
			Directive   string          `json:"directive,omitempty"`
			Mode        RunMode         `json:"mode,omitempty"`
			AutoApprove []string        `json:"auto_approve,omitempty"`
			Provider    *ProviderConfig `json:"provider,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if body.Directive != "" {
			a.thinker.config.SetDirective(body.Directive)
			a.thinker.ReloadDirective()
		}
		if body.Mode == ModeAutonomous || body.Mode == ModeSupervised {
			a.thinker.config.SetMode(body.Mode)
			if a.thinker.telemetry != nil {
				a.thinker.telemetry.Emit("mode.changed", "main", map[string]string{"mode": string(body.Mode)})
			}
		}
		if body.AutoApprove != nil {
			a.thinker.config.mu.Lock()
			a.thinker.config.AutoApprove = body.AutoApprove
			a.thinker.config.mu.Unlock()
			a.thinker.config.Save()
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
		writeJSON(w, map[string]string{"status": "updated"})
	default:
		http.Error(w, "GET or PUT only", http.StatusMethodNotAllowed)
	}
}

func (a *APIServer) approve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Approved bool `json:"approved"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// Check there's actually a pending tool
	a.thinker.pendingMu.Lock()
	hasPending := a.thinker.pendingTool != nil
	a.thinker.pendingMu.Unlock()

	if !hasPending {
		http.Error(w, "no pending approval", http.StatusConflict)
		return
	}

	// Non-blocking send
	select {
	case a.thinker.approvalCh <- body.Approved:
	default:
	}

	action := "rejected"
	if body.Approved {
		action = "approved"
	}
	writeJSON(w, map[string]string{"status": action})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
