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
	mux.HandleFunc("/event", api.postEvent)
	mux.HandleFunc("/config", api.config)
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
		"threads":        a.thinker.threads.Count(),
		"memories":       a.thinker.memory.Count(),
	})
}

func (a *APIServer) threads(w http.ResponseWriter, r *http.Request) {
	threads := a.thinker.threads.List()
	type threadJSON struct {
		ID        string   `json:"id"`
		Tools     []string `json:"tools"`
		Thinking  bool     `json:"thinking"`
		Iteration int      `json:"iteration"`
		Rate      string   `json:"rate"`
		Model     string   `json:"model"`
		Age       string   `json:"age"`
	}
	var out []threadJSON
	for _, t := range threads {
		out = append(out, threadJSON{
			ID:        t.ID,
			Tools:     t.Tools,
			Thinking:  t.Thinking,
			Iteration: t.Iteration,
			Rate:      t.Rate.String(),
			Model:     t.Model.String(),
			Age:       formatAge(time.Since(t.Started)),
		})
	}
	if out == nil {
		out = []threadJSON{}
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
	flusher.Flush()

	// Tap into thread manager events
	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case ev := <-a.thinker.threads.events:
			data, _ := json.Marshal(map[string]string{
				"type":    ev.Type,
				"thread":  ev.ThreadID,
				"message": ev.Message,
			})
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
			// Re-broadcast so TUI still gets it
			go func() { a.thinker.threads.events <- ev }()
		default:
			select {
			case <-ctx.Done():
				return
			case <-time.After(100 * time.Millisecond):
			}
		}
	}
}

func (a *APIServer) postEvent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if body.Message == "" {
		http.Error(w, "message required", http.StatusBadRequest)
		return
	}
	a.thinker.InjectConsole(body.Message)
	writeJSON(w, map[string]string{"status": "injected"})
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
