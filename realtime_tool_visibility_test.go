package core

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRealtimeVisibilityMatrix(t *testing.T) {
	for _, tc := range []struct {
		name                                          string
		policy                                        ToolLoadMode
		search                                        string
		scope, exact, active, noSpawn, platform, want bool
	}{
		{name: "always_scope", policy: ToolLoadAlways, search: "on", scope: true, want: true},
		{name: "auto_eager_scope", policy: ToolLoadAuto, search: "off", scope: true, want: true},
		{name: "auto_discovery_scope", policy: ToolLoadAuto, search: "on", scope: true},
		{name: "deferred_scope", policy: ToolLoadDeferred, search: "off", scope: true},
		{name: "deferred_activated_scope", policy: ToolLoadDeferred, search: "on", scope: true, active: true, want: true},
		{name: "deferred_exact_grant", policy: ToolLoadDeferred, search: "on", exact: true, want: true},
		{name: "unauthorized_always", policy: ToolLoadAlways, search: "off"},
		{name: "unauthorized_stale_active", policy: ToolLoadAlways, search: "off", active: true},
		{name: "no_spawn_scope", policy: ToolLoadAlways, search: "off", scope: true, noSpawn: true},
		{name: "no_spawn_exact", policy: ToolLoadAlways, search: "off", exact: true, noSpawn: true},
		{name: "platform_no_spawn_scope", policy: ToolLoadAlways, search: "on", scope: true, noSpawn: true, platform: true, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("APTEVA_TOOL_SEARCH", tc.search)
			registry, index := NewToolRegistry(""), NewToolIndex()
			defs := []mcpToolDef{{Name: "lookup", Description: "Look up a booking", InputSchema: map[string]any{"type": "object"}}}
			registerTestMCPTools(registry, "booking", defs)
			index.Add("booking", defs, tc.noSpawn, &MCPToolLoadingConfig{Default: tc.policy})
			thinker := &Thinker{threadID: "voice", registry: registry, toolIndex: index, toolAllowlist: map[string]bool{"booking_lookup": tc.exact}, toolMCPScopes: map[string]bool{"booking": tc.scope}, activeTools: map[string]bool{"booking_lookup": tc.active}, allowNoSpawn: tc.platform}
			normal := nativeToolSet(thinker.prepareNativeTools("openai-codex"))["booking_lookup"]
			realtime := nativeToolSet(realtimeNativeTools(thinker))["booking_lookup"]
			t.Logf("normal=%t realtime=%t expected=%t", normal, realtime, tc.want)
			if normal != tc.want {
				t.Errorf("normal visibility = %t, want %t", normal, tc.want)
			}
			if realtime != tc.want {
				t.Errorf("realtime visibility = %t, want %t", realtime, tc.want)
			}
		})
	}
}

func TestRealtimeVisibilityHTTPMCP(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	var calls atomic.Int64
	srv := newGeminiLiveMCPServer(t, &calls, make(chan geminiLiveMCPCall, 2))
	defer srv.Close()
	parent := newTestThinker()
	defer parent.Stop()
	parent.registry = NewToolRegistry("test")
	parent.config = &Config{Directive: "test parent", MCPServers: []MCPServerConfig{{Name: "booking", Transport: "http", URL: srv.URL + "/mcp", ToolLoading: &MCPToolLoadingConfig{Default: ToolLoadAlways}}}}
	parent.mcpServers = connectAndRegisterMCP(parent.config.MCPServers, parent.registry, parent.toolIndex, nil)
	defer func() {
		for _, s := range parent.mcpServers {
			s.Close()
		}
	}()
	if len(parent.mcpServers) != 1 {
		t.Fatal("HTTP MCP failed to connect")
	}
	if err := parent.threads.SpawnWithOpts("voice", "Idle.", []string{"send"}, SpawnOpts{DeferRun: true, MCPNames: []string{"booking"}, ParentID: "main"}); err != nil {
		t.Fatal(err)
	}
	parent.threads.mu.RLock()
	child := parent.threads.threads["voice"].Thinker
	parent.threads.mu.RUnlock()
	const name = "booking_lookup_code"
	if !child.toolMCPScopes["booking"] || !child.toolAuthorized(name) {
		t.Fatal("child MCP scope missing")
	}
	def := child.registry.Get(name)
	if def == nil {
		t.Fatal("connected tool not registered")
	}
	result := def.Handler(map[string]string{"code": "ALPHA-7"})
	if result.IsError || calls.Load() != 1 {
		t.Fatalf("registered tool not callable: %+v calls=%d", result, calls.Load())
	}
	if !nativeToolSet(child.prepareNativeTools("openai-codex"))[name] {
		t.Fatal("normal path omitted tool")
	}
	if len(child.activeTools) != 0 {
		t.Fatalf("test unexpectedly activated tools: %v", child.activeTools)
	}
	rt := newRealtimeThinker(context.Background(), child, &fakeRealtimeProvider{}, "", nil, nil, nil)
	defer rt.cancel()
	t.Logf("HTTP MCP connected; scope authorized; registered handler called successfully; normal has %s; realtime session opts tools=%v", name, nativeToolSet(rt.opts.Tools))
	if !nativeToolSet(rt.opts.Tools)[name] {
		t.Errorf("provider session configuration missing actual registered name %s", name)
	}
}

func TestRealtimeVisibilityPreviewUsesProposedGrants(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	registry, index := NewToolRegistry(""), NewToolIndex()
	defs := []mcpToolDef{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	registerTestMCPTools(registry, "booking", defs)
	index.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	thinker := &Thinker{threadID: "voice", registry: registry, toolIndex: index, toolAllowlist: map[string]bool{}, toolMCPScopes: map[string]bool{}, activeTools: map[string]bool{"booking_lookup": true}}
	for _, scope := range []bool{false, true} {
		grants, scopes := map[string]bool{}, map[string]bool{}
		if scope {
			scopes["booking"] = true
		} else {
			grants["booking_lookup"] = true
		}
		if !nativeToolSet(realtimeNativeToolsFor(thinker, grants, scopes, false))["booking_lookup"] {
			t.Fatal("proposed addition missing")
		}
		if thinker.toolAuthorized("booking_lookup") {
			t.Fatal("preview mutated current grants")
		}
		thinker.toolAllowlist, thinker.toolMCPScopes = grants, scopes
		if nativeToolSet(realtimeNativeToolsFor(thinker, map[string]bool{}, map[string]bool{}, false))["booking_lookup"] {
			t.Fatal("revoked grant survives through stale activation")
		}
		thinker.toolAllowlist, thinker.toolMCPScopes = map[string]bool{}, map[string]bool{}
	}
}

func TestRealtimeVisibilityRefreshReopenAndProviderPayloads(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.registry = NewToolRegistry("")
	thinker.toolAllowlist = map[string]bool{}
	thinker.toolMCPScopes = map[string]bool{"booking": true}
	defs := []mcpToolDef{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	registerTestMCPTools(thinker.registry, "booking", defs)
	thinker.toolIndex.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	provider := &fakeRealtimeProvider{}
	rt, err := startRealtimeThinker(context.Background(), thinker, provider, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer rt.cancel()
	defer rt.replaceSession(nil)
	assertTools := func(tools []NativeTool, want string) {
		t.Helper()
		names := nativeToolSet(tools)
		for _, name := range []string{"booking_lookup", "booking_reconnected"} {
			if names[name] != (name == want) {
				t.Fatalf("tools %v want %q", names, want)
			}
		}
		// Assert the provider serializers preserve the actual callable names.
		google, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "gemini-test", Tools: tools}, "Aoede")
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Setup struct {
				Tools []struct {
					Functions []struct {
						Name string `json:"name"`
					} `json:"functionDeclarations"`
				} `json:"tools"`
			} `json:"setup"`
		}
		if err := json.Unmarshal(google, &body); err != nil {
			t.Fatal(err)
		}
		wire := map[string]bool{}
		for _, group := range body.Setup.Tools {
			for _, f := range group.Functions {
				wire[f.Name] = true
			}
		}
		openai := map[string]bool{}
		for _, f := range sessionTools(tools) {
			openai[f["name"].(string)] = true
		}
		for name := range names {
			if !wire[name] || !openai[name] {
				t.Fatalf("provider serializer omitted %s", name)
			}
		}
	}
	assertTools(provider.opens[0].Tools, "booking_lookup")
	// Simulate an MCP replacement: removed name must disappear; new baseline appears.
	thinker.registry.RemoveByMCPServer("booking")
	defs[0].Name = "reconnected"
	registerTestMCPTools(thinker.registry, "booking", defs)
	thinker.toolIndex.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	session := rt.currentSession().(*fakeRealtimeSession)
	rt.refreshConfiguration()
	assertTools(session.updates[len(session.updates)-1].Tools, "booking_reconnected")
	rt.replaceSession(nil)
	if err := rt.openSession(true); err != nil {
		t.Fatal(err)
	}
	assertTools(provider.opens[1].Tools, "booking_reconnected")
	thinker.toolMCPScopes = map[string]bool{}
	thinker.activeTools = map[string]bool{"booking_reconnected": true}
	rt.refreshConfiguration()
	session = rt.currentSession().(*fakeRealtimeSession)
	assertTools(session.updates[len(session.updates)-1].Tools, "")
	if thinker.modelToolCallable("booking_reconnected", nil) {
		t.Fatal("revoked tool callable")
	}
}

func TestRealtimeVisibilityCatalogChangeRefreshesRunningSession(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.registry = NewToolRegistry("")
	thinker.toolAllowlist = map[string]bool{}
	thinker.toolMCPScopes = map[string]bool{"booking": true}
	provider := &fakeRealtimeProvider{}
	rt, err := startRealtimeThinker(context.Background(), thinker, provider, "", nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { rt.Run(); close(done) }()
	defer func() {
		rt.cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("realtime loop failed to stop")
		}
	}()
	defs := []mcpToolDef{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	registerTestMCPTools(thinker.registry, "booking", defs)
	thinker.toolIndex.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	session := rt.currentSession().(*fakeRealtimeSession)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		session.mu.Lock()
		found := false
		for _, u := range session.updates {
			found = found || nativeToolSet(u.Tools)["booking_lookup"]
		}
		session.mu.Unlock()
		if found {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("running realtime session was not notified of catalog change")
}

func TestRealtimeVisibilityImmutableScopeUpdateRejectsBeforeCommit(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	defer parent.Stop()
	defer parent.threads.KillAll()
	parent.config.RealtimeEnabled = true
	parent.registry = NewToolRegistry("")
	provider := &fakeRealtimeProvider{}
	parent.pool = &ProviderPool{providers: map[string]LLMProvider{"fireworks": parent.provider}, order: []string{"fireworks"}, default_: "fireworks", realtimeProviders: map[string]RealtimeProvider{provider.Name(): provider}, realtimeOrder: []string{provider.Name()}, realtimeDefault: provider.Name()}
	defs := []mcpToolDef{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}}
	registerTestMCPTools(parent.registry, "booking", defs)
	parent.toolIndex.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	if err := parent.threads.SpawnWithOpts("voice", "Idle.", nil, SpawnOpts{Realtime: true, Ephemeral: true, ProviderName: provider.Name(), DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	thread := parent.threads.threads["voice"]
	instructions, tools := thread.Realtime.configurationSnapshot()
	session := &immutableRealtimeSession{fakeRealtimeSession: newFakeRealtimeSession(), fingerprint: googleRealtimeConfigFingerprint(instructions, tools)}
	thread.Realtime.replaceSession(session)
	parent.threads.realtimeBridgeConnected("voice")
	scopes := []string{"booking"}
	if _, err := parent.threads.UpdateWithOpts("voice", "", "", nil, ThreadUpdateOptions{MCPNames: &scopes}); !errors.Is(err, ErrRealtimeConfigurationRestartRequired) {
		t.Fatalf("update error: %v", err)
	}
	if thread.Thinker.toolMCPScopes["booking"] || len(thread.MCPNames) > 0 {
		t.Fatal("rejected preview committed new scope")
	}
	result, err := parent.threads.UpdateWithOpts("voice", "", "", nil, ThreadUpdateOptions{MCPNames: &scopes, RestartRealtime: true})
	if err != nil || !result.RealtimeRestarted {
		t.Fatalf("explicit restart: %+v %v", result, err)
	}
	if !nativeToolSet(realtimeNativeTools(thread.Thinker))["booking_lookup"] {
		t.Fatal("accepted scope did not expose baseline tool")
	}
}

func TestRealtimeVisibilityMainAndPerToolPolicy(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "off")
	registry, index := NewToolRegistry(""), NewToolIndex()
	defs := []mcpToolDef{{Name: "lookup"}, {Name: "deferred"}, {Name: "auto"}}
	registerTestMCPTools(registry, "booking", defs)
	index.Add("booking", defs, true, &MCPToolLoadingConfig{Default: ToolLoadAuto, Tools: map[string]ToolLoadMode{"lookup": ToolLoadAlways, "deferred": ToolLoadDeferred}})
	thinker := &Thinker{threadID: "main", registry: registry, toolIndex: index, activeTools: map[string]bool{}}
	for _, search := range []string{"off", "on"} {
		t.Setenv("APTEVA_TOOL_SEARCH", search)
		names := nativeToolSet(realtimeNativeTools(thinker))
		if !names["booking_lookup"] || names["booking_deferred"] || names["booking_auto"] != (search == "off") {
			t.Fatalf("search=%s names=%v", search, names)
		}
	}
	thinker.discoveredToolUntil = map[string]int{"booking_deferred": 4}
	thinker.iteration = 4
	if !nativeToolSet(realtimeNativeTools(thinker))["booking_deferred"] {
		t.Fatal("retained discovery missing")
	}
	thinker.iteration = 5
	if nativeToolSet(realtimeNativeTools(thinker))["booking_deferred"] {
		t.Fatal("expired discovery retained")
	}
	if len(thinker.activeTools) != 0 {
		t.Fatal("baseline/retention polluted active LRU")
	}
}

func TestRealtimeVisibilityCatalogRefreshReopensIdleImmutableProvider(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	thinker.registry = NewToolRegistry("")
	thinker.toolAllowlist = map[string]bool{}
	thinker.toolMCPScopes = map[string]bool{"booking": true}
	provider := &fakeRealtimeProvider{}
	rt := newRealtimeThinker(context.Background(), thinker, provider, "", nil, nil, nil)
	instructions, tools := rt.configurationSnapshot()
	session := &immutableRealtimeSession{fakeRealtimeSession: newFakeRealtimeSession(), fingerprint: googleRealtimeConfigFingerprint(instructions, tools)}
	rt.replaceSession(session)
	done := make(chan struct{})
	go func() { rt.Run(); close(done) }()
	defer func() {
		rt.cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("realtime loop failed to stop")
		}
	}()
	defs := []mcpToolDef{{Name: "lookup"}}
	registerTestMCPTools(thinker.registry, "booking", defs)
	thinker.toolIndex.Add("booking", defs, false, &MCPToolLoadingConfig{Default: ToolLoadAlways})
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		provider.mu.Lock()
		found := len(provider.opens) > 0 && nativeToolSet(provider.opens[0].Tools)["booking_lookup"]
		provider.mu.Unlock()
		if found {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("idle immutable session was not reopened with catalog change")
}
