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
	mux.Handle("/", http.FileServer(http.Dir("web")))
	return http.ListenAndServe(addr, mux)
}

func (a *APIServer) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]bool{"ok": true})
}

func (a *APIServer) status(w http.ResponseWriter, r *http.Request) {
	elapsed := time.Since(a.startTime)
	writeJSON(w, map[string]any{
		"uptime_seconds": int(elapsed.Seconds()),
		"iteration":      a.thinker.iteration,
		"rate":           a.thinker.rate.String(),
		"model":          a.thinker.model.String(),
		"threads":        a.thinker.threads.Count() + 1, // +1 for main
		"memories":       a.thinker.memory.Count(),
		"paused":         a.thinker.paused,
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
		Message  string `json:"message"`
		ThreadID string `json:"thread_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if body.Message == "" {
		http.Error(w, "message required", http.StatusBadRequest)
		return
	}

	if body.ThreadID != "" && body.ThreadID != "main" {
		// Route to specific thread via EventBus
		a.thinker.bus.Publish(Event{Type: EventInbox, To: body.ThreadID, Text: body.Message})
		writeJSON(w, map[string]string{"status": "injected", "thread_id": body.ThreadID})
	} else {
		a.thinker.InjectConsole(body.Message)
		writeJSON(w, map[string]string{"status": "injected", "thread_id": "main"})
	}
}

func (a *APIServer) config(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, map[string]string{
			"directive": a.thinker.config.GetDirective(),
		})
	case http.MethodPut:
		var body struct {
			Directive string `json:"directive"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		a.thinker.config.SetDirective(body.Directive)
		a.thinker.ReloadDirective()
		writeJSON(w, map[string]string{"status": "updated"})
	default:
		http.Error(w, "GET or PUT only", http.StatusMethodNotAllowed)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
