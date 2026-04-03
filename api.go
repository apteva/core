package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	aptcomputer "github.com/apteva/computer"
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
	logMsg("API", "GET /health")
	writeJSON(w, map[string]bool{"ok": true})
}

// findPendingApproval checks main and all child threads for a pending tool.
func (a *APIServer) findPendingApproval() *ToolCallData {
	// Check main
	a.thinker.pendingMu.Lock()
	if a.thinker.pendingTool != nil {
		tc := &ToolCallData{
			ID:   a.thinker.pendingTool.NativeID,
			Name: a.thinker.pendingTool.Name,
			Args: a.thinker.pendingTool.Args,
		}
		a.thinker.pendingMu.Unlock()
		return tc
	}
	a.thinker.pendingMu.Unlock()

	// Check child threads
	if a.thinker.threads != nil {
		if tc := a.thinker.threads.FindPendingApproval(); tc != nil {
			return &ToolCallData{ID: tc.NativeID, Name: tc.Name, Args: tc.Args}
		}
	}
	return nil
}

// findPendingThinker returns the thinker (main or child) that has a pending tool call.
func (a *APIServer) findPendingThinker() *Thinker {
	a.thinker.pendingMu.Lock()
	if a.thinker.pendingTool != nil {
		a.thinker.pendingMu.Unlock()
		return a.thinker
	}
	a.thinker.pendingMu.Unlock()

	if a.thinker.threads != nil {
		return a.thinker.threads.FindPendingThinker()
	}
	return nil
}

func (a *APIServer) status(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "GET /status")
	elapsed := time.Since(a.startTime)
	pending := a.findPendingApproval()

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
			}
			flusher.Flush()
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
			"directive":    a.thinker.config.GetDirective(),
			"mode":         a.thinker.config.GetMode(),
			"auto_approve": a.thinker.config.AutoApprove,
			"provider":     providerInfo,
			"computer":     computerInfo,
			"mcp_servers":  mcpInfo,
		})
	case http.MethodPut:
		var body struct {
			Directive   string            `json:"directive,omitempty"`
			Mode        RunMode           `json:"mode,omitempty"`
			AutoApprove []string          `json:"auto_approve,omitempty"`
			Provider    *ProviderConfig   `json:"provider,omitempty"`
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

func (a *APIServer) approve(w http.ResponseWriter, r *http.Request) {
	logMsg("API", "POST /approve")
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

	// Find the thinker (main or child) that actually has the pending tool
	target := a.findPendingThinker()
	if target == nil {
		http.Error(w, "no pending approval", http.StatusConflict)
		return
	}

	// Non-blocking send to the correct thinker's approval channel
	select {
	case target.approvalCh <- body.Approved:
	default:
	}

	action := "rejected"
	if body.Approved {
		action = "approved"
	}
	writeJSON(w, map[string]string{"status": action})
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
