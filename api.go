package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	aptcomputer "github.com/apteva/computer"
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
	mux.HandleFunc("/events", api.apiAuth(api.events))
	mux.HandleFunc("/pause", api.apiAuth(api.pause))
	mux.HandleFunc("/event", api.apiAuth(api.postEvent))
	mux.HandleFunc("/config", api.apiAuth(api.config))
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
		"uptime_seconds": int(elapsed.Seconds()),
		"iteration":      a.thinker.iteration,
		"rate":           formatSleep(a.thinker.agentSleep),
		"model":          a.thinker.model.String(),
		"threads":        a.thinker.threads.Count() + 1, // +1 for main
		"memories":       a.thinker.memory.Count(),
		"paused":         a.thinker.paused,
		"mode":           a.thinker.config.GetMode(),
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
	logMsg("API", "GET /threads")
	// Always include main
	out := []threadJSON{{
		ID:        "main",
		Directive: a.thinker.config.GetDirective(),
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

func (a *APIServer) threadAction(w http.ResponseWriter, r *http.Request) {
	// Extract thread ID from path: /threads/{id}
	id := strings.TrimPrefix(r.URL.Path, "/threads/")
	if id == "" {
		http.Error(w, "thread ID required", http.StatusBadRequest)
		return
	}
	logMsg("API", fmt.Sprintf("%s /threads/%s", r.Method, id))

	switch r.Method {
	case http.MethodDelete:
		if id == "main" {
			http.Error(w, "cannot kill main thread", http.StatusBadRequest)
			return
		}
		a.thinker.threads.Kill(id)
		a.thinker.config.RemoveThread(id)
		writeJSON(w, map[string]string{"status": "killed", "id": id})
	default:
		http.Error(w, "DELETE only", http.StatusMethodNotAllowed)
	}
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
		// Build live computer info
		var computerInfo map[string]any
		if a.thinker.computer != nil {
			d := a.thinker.computer.DisplaySize()
			computerInfo = map[string]any{
				"connected": true,
				"display":   map[string]int{"width": d.Width, "height": d.Height},
			}
			if a.thinker.config.Computer != nil {
				computerInfo["type"] = a.thinker.config.Computer.Type
			}
		}
		// Build live MCP server info
		var mcpInfo []map[string]any
		for _, srv := range a.thinker.mcpServers {
			mcpInfo = append(mcpInfo, map[string]any{
				"name":      srv.GetName(),
				"connected": true,
			})
		}

		writeJSON(w, map[string]any{
			"directive":   a.thinker.config.GetDirective(),
			"mode":        a.thinker.config.GetMode(),
			"provider":    providerInfo,
			"providers":   a.thinker.config.GetProviders(),
			"computer":    computerInfo,
			"mcp_servers": mcpInfo,
		})
	case http.MethodPut:
		var body struct {
			Directive  string            `json:"directive,omitempty"`
			Mode       RunMode           `json:"mode,omitempty"`
			Provider   *ProviderConfig   `json:"provider,omitempty"`
			Providers  []ProviderConfig  `json:"providers,omitempty"`
			Computer    *ComputerConfig   `json:"computer,omitempty"`
			MCPServers  []MCPServerConfig `json:"mcp_servers,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		if body.Directive != "" {
			a.thinker.config.SetDirective(body.Directive)
			a.thinker.ReloadDirective()
		}
		if body.Mode == ModeAutonomous || body.Mode == ModeCautious || body.Mode == ModeLearn {
			a.thinker.config.SetMode(body.Mode)
			if a.thinker.telemetry != nil {
				a.thinker.telemetry.Emit("mode.changed", "main", map[string]string{"mode": string(body.Mode)})
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
		if body.Computer != nil {
			// Hot-connect or disconnect computer environment
			if body.Computer.Type == "" {
				// Disconnect
				if a.thinker.computer != nil {
					a.thinker.computer.Close()
					a.thinker.computer = nil
				}
				a.thinker.config.mu.Lock()
				a.thinker.config.Computer = nil
				a.thinker.config.mu.Unlock()
				a.thinker.config.Save()
			} else {
				// Connect new computer
				comp, err := aptcomputer.New(aptcomputer.Config{
					Type:      body.Computer.Type,
					URL:       body.Computer.URL,
					APIKey:    body.Computer.APIKey,
					ProjectID: body.Computer.ProjectID,
					Width:     body.Computer.Width,
					Height:    body.Computer.Height,
				})
				if err != nil {
					http.Error(w, fmt.Sprintf("computer: %v", err), http.StatusBadRequest)
					return
				}
				// Close old session if any
				if a.thinker.computer != nil {
					a.thinker.computer.Close()
				}
				a.thinker.SetComputer(comp)
				a.thinker.config.mu.Lock()
				a.thinker.config.Computer = body.Computer
				a.thinker.config.mu.Unlock()
				a.thinker.config.Save()
			}
		}
		if body.MCPServers != nil {
			a.reconcileMCP(body.MCPServers)
		}
		writeJSON(w, map[string]string{"status": "updated"})
	default:
		http.Error(w, "GET or PUT only", http.StatusMethodNotAllowed)
	}
}

// reconcileMCP diffs the desired MCP server list against the live state,
// connecting new servers and disconnecting removed ones.
func (a *APIServer) reconcileMCP(desired []MCPServerConfig) {
	logMsg("API", fmt.Sprintf("reconcileMCP: %d desired servers", len(desired)))
	t := a.thinker

	// Index desired by name
	want := make(map[string]MCPServerConfig, len(desired))
	for _, cfg := range desired {
		want[cfg.Name] = cfg
	}

	// Disconnect servers not in the desired list
	var kept []MCPConn
	for _, srv := range t.mcpServers {
		if _, ok := want[srv.GetName()]; ok {
			kept = append(kept, srv)
		} else {
			srv.Close()
			t.config.RemoveMCPServer(srv.GetName())
			t.registry.RemoveByMCPServer(srv.GetName())
			if t.telemetry != nil {
				t.telemetry.Emit("mcp.disconnected", "api", map[string]string{"name": srv.GetName()})
			}
		}
	}
	t.mcpServers = kept

	// Index live by name
	live := make(map[string]bool, len(kept))
	for _, srv := range kept {
		live[srv.GetName()] = true
	}

	// Connect new servers
	for _, cfg := range desired {
		if live[cfg.Name] {
			continue
		}
		srv, err := connectAnyMCP(cfg)
		if err != nil {
			continue
		}
		tools, err := srv.ListTools()
		if err != nil {
			srv.Close()
			continue
		}
		t.mcpServers = append(t.mcpServers, srv)
		for _, tool := range tools {
			fullName := cfg.Name + "_" + tool.Name
			syntax := buildMCPSyntax(fullName, tool.InputSchema)
			t.registry.Register(&ToolDef{
				Name:        fullName,
				Description: fmt.Sprintf("[%s] %s", cfg.Name, tool.Description),
				Syntax:      syntax,
				Rules:       fmt.Sprintf("Provided by MCP server '%s'.", cfg.Name),
				Handler:     mcpProxyHandler(srv, tool.Name),
				InputSchema: tool.InputSchema,
				MCP:         !cfg.MainAccess,
				MCPServer:   cfg.Name,
			})
		}
		if t.memory != nil {
			go func(srvName string, srvTools []mcpToolDef) {
				for _, tl := range srvTools {
					fullName := srvName + "_" + tl.Name
					emb, err := t.memory.embed(fullName + ": " + tl.Description)
					if err == nil {
						td := t.registry.Get(fullName)
						if td != nil {
							td.Embedding = emb
						}
					}
				}
			}(cfg.Name, tools)
		}
		t.config.SaveMCPServer(cfg)
		if t.telemetry != nil {
			t.telemetry.Emit("mcp.connected", "api", map[string]string{"name": cfg.Name, "tools": fmt.Sprintf("%d", len(tools))})
		}
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
