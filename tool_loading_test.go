package core

import (
	"encoding/json"
	"strings"
	"testing"
)

func registerTestMCPTools(registry *ToolRegistry, server string, tools []mcpToolDef) {
	for _, tool := range tools {
		registry.Register(&ToolDef{
			Name:        server + "_" + tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
			MCP:         true,
			MCPServer:   server,
		})
	}
}

func nativeToolSet(tools []NativeTool) map[string]bool {
	out := make(map[string]bool, len(tools))
	for _, tool := range tools {
		out[tool.Name] = true
	}
	return out
}

func TestMCPToolLoadingConfigBackwardCompatibilityAndValidation(t *testing.T) {
	var cfg MCPServerConfig
	if err := json.Unmarshal([]byte(`{"name":"legacy","url":"http://example.test"}`), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ToolLoading != nil {
		t.Fatalf("legacy config unexpectedly gained tool_loading: %#v", cfg.ToolLoading)
	}
	if got := cfg.toolLoadMode("anything"); got != ToolLoadAuto {
		t.Fatalf("legacy load mode = %q, want auto", got)
	}
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "tool_loading") {
		t.Fatalf("legacy round-trip should omit tool_loading: %s", encoded)
	}

	invalid := MCPServerConfig{
		Name: "bad",
		ToolLoading: &MCPToolLoadingConfig{
			Default: "sometimes",
		},
	}
	if err := validateMCPToolLoading(invalid); err == nil {
		t.Fatal("invalid loading mode was accepted")
	}
}

func TestToolIndexPolicyDefaultAndPerToolOverride(t *testing.T) {
	ix := NewToolIndex()
	tools := []mcpToolDef{
		{Name: "send", Description: "Send a message"},
		{Name: "publish", Description: "Publish a report"},
	}
	ix.Add("channels", tools, true, &MCPToolLoadingConfig{
		Default: ToolLoadDeferred,
		Tools: map[string]ToolLoadMode{
			"send": ToolLoadAlways,
		},
	})

	baseline := ix.BaselineNames(false, true)
	if len(baseline) != 1 || baseline[0] != "channels_send" {
		t.Fatalf("discovery baseline = %v, want only channels_send", baseline)
	}
	if got := ix.BaselineNames(true, true); len(got) != 1 || got[0] != "channels_send" {
		t.Fatalf("eager baseline = %v; explicit deferred override must remain deferred", got)
	}
	if got := ix.BaselineNames(false, false); len(got) != 0 {
		t.Fatalf("no_spawn baseline leaked to worker: %v", got)
	}

	entry, ok := ix.Get("channels_publish")
	if !ok || entry.LoadMode != ToolLoadDeferred || entry.LocalName != "publish" {
		t.Fatalf("publish index entry = %#v, %v", entry, ok)
	}
}

func TestPrepareNativeToolsAlwaysLoadsChannelsUnderForcedDiscovery(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	registry := NewToolRegistry("")
	index := NewToolIndex()

	channelTools := []mcpToolDef{
		{Name: "send", Description: "Send a visible message", InputSchema: map[string]any{"type": "object"}},
		{Name: "publish", Description: "Publish an Inbox artifact", InputSchema: map[string]any{"type": "object"}},
		{Name: "set_status", Description: "Set current status", InputSchema: map[string]any{"type": "object"}},
		{Name: "list_channels", Description: "List channels", InputSchema: map[string]any{"type": "object"}},
	}
	registerTestMCPTools(registry, "channels", channelTools)
	index.Add("channels", channelTools, true, &MCPToolLoadingConfig{Default: ToolLoadAlways})

	decoys := make([]mcpToolDef, 0, defaultToolSearchAutoThreshold+1)
	for i := 0; i < defaultToolSearchAutoThreshold+1; i++ {
		decoys = append(decoys, mcpToolDef{
			Name:        "decoy_" + string(rune('A'+i%26)) + string(rune('a'+i/26)),
			Description: "Unrelated inventory capability",
			InputSchema: map[string]any{"type": "object"},
		})
	}
	registerTestMCPTools(registry, "large", decoys)
	index.Add("large", decoys, false)

	thinker := &Thinker{
		threadID:      "main",
		registry:      registry,
		toolIndex:     index,
		activeTools:   map[string]bool{},
		activeToolAge: map[string]int{},
		directive:     "Idle.",
	}
	names := nativeToolSet(thinker.prepareNativeTools("openai-codex"))
	for _, want := range []string{
		"channels_send", "channels_publish", "channels_set_status", "channels_list_channels",
	} {
		if !names[want] {
			t.Errorf("always-loaded Channels tool %q missing from provider request", want)
		}
		if thinker.activeTools[want] {
			t.Errorf("always-loaded tool %q polluted the LRU activation set", want)
		}
	}
	if thinker.lastAlwaysMCPCount != 4 {
		t.Fatalf("always_mcp_count = %d, want 4", thinker.lastAlwaysMCPCount)
	}
	if thinker.lastDeferredMCPCount != len(decoys) {
		t.Fatalf("deferred_mcp_count = %d, want %d", thinker.lastDeferredMCPCount, len(decoys))
	}
	if thinker.lastToolMode != "discovery" {
		t.Fatalf("tool mode = %q, want discovery", thinker.lastToolMode)
	}

	worker := &Thinker{
		threadID:      "worker",
		registry:      registry,
		toolIndex:     index,
		toolAllowlist: map[string]bool{"pace": true},
		activeTools:   map[string]bool{},
		activeToolAge: map[string]int{},
		directive:     "Idle.",
	}
	workerNames := nativeToolSet(worker.prepareNativeTools("openai-codex"))
	if workerNames["channels_send"] {
		t.Fatal("always-loaded no_spawn Channels tool leaked to an ordinary worker")
	}
}

func TestAlwaysLoadedToolsSurviveActivationLRU(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	registry := NewToolRegistry("")
	index := NewToolIndex()
	channels := []mcpToolDef{{Name: "send", Description: "Send", InputSchema: map[string]any{"type": "object"}}}
	registerTestMCPTools(registry, "channels", channels)
	index.Add("channels", channels, true, &MCPToolLoadingConfig{Default: ToolLoadAlways})

	thinker := &Thinker{
		threadID:      "main",
		registry:      registry,
		toolIndex:     index,
		activeTools:   map[string]bool{},
		activeToolAge: map[string]int{},
		directive:     "Idle.",
	}
	for i := 0; i < 50; i++ {
		name := "active_" + string(rune('A'+i%26)) + string(rune('a'+i/26))
		thinker.activeTools[name] = true
		thinker.activeToolAge[name] = i
	}
	names := nativeToolSet(thinker.prepareNativeTools("openai-codex"))
	if len(thinker.activeTools) != 28 {
		t.Fatalf("activation LRU retained %d tools, want 28", len(thinker.activeTools))
	}
	if !names["channels_send"] {
		t.Fatal("always-loaded Channels tool disappeared after activation LRU eviction")
	}
}

func TestSystemPromptDescribesMixedLoadingWithoutAssumingDeliveryTool(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	catalog := []MCPServerInfo{{
		Name: "channels", ToolCount: 4, AlwaysCount: 4,
	}, {
		Name: "media", ToolCount: 20, AutoCount: 20,
	}}
	prompt := buildSystemPrompt("Help the operator.", nil, "", nil, nil, nil, catalog)
	if strings.Contains(prompt, "channels_respond") {
		t.Fatalf("prompt still references legacy channels_respond")
	}
	if strings.Contains(prompt, "channels_send") {
		t.Fatalf("prompt invents channels_send from the server catalog")
	}
	if !strings.Contains(prompt, "4 always") || !strings.Contains(prompt, "Always-loaded MCP tools are already") {
		t.Fatalf("prompt missing mixed loading guidance:\n%s", prompt)
	}
}

func TestSystemPromptDeliveryIsAppIndependent(t *testing.T) {
	for _, server := range []string{"", "channels", "messenger"} {
		for _, provider := range []string{"openai-codex", "fireworks"} {
			t.Run(server+"/"+provider, func(t *testing.T) {
				registry := NewToolRegistry("")
				var catalog []MCPServerInfo
				if server != "" {
					registerTestMCPTools(registry, server, []mcpToolDef{{Name: "deliver", Description: "Deliver a message"}})
					catalog = []MCPServerInfo{{Name: server, ToolCount: 1, AlwaysCount: 1}}
				}
				prompt := buildSystemPrompt("Answer directly in this thread.", registry, "", nil, nil, &ProviderPool{default_: provider}, catalog)
				for _, forbidden := range []string{"channels_send", "channels_respond", "Prose between tool calls does not deliver a message to the user"} {
					if strings.Contains(prompt, forbidden) {
						t.Fatalf("generic prompt contains obsolete delivery instruction %q", forbidden)
					}
				}
				for _, required := range []string{
					"Follow the directive and request's delivery instructions.",
					"A progress update is not a completed answer.",
					"Provide the requested result before idling, or retain responsibility for genuinely pending work.",
					"Do not claim delivery without evidence.",
					"Answer directly in this thread.",
				} {
					if !strings.Contains(prompt, required) {
						t.Fatalf("missing delivery contract %q", required)
					}
				}
			})
		}
	}
}

func TestWorkerPromptRetainsCoreDeliveryContract(t *testing.T) {
	for _, canSpawn := range []bool{false, true} {
		prompt := formatThreadBasePrompt(canSpawn, false, "worker", "parent")
		for _, required := range []string{
			"return the complete final result exactly once with done(message)",
			"Ordinary assistant text stays in your private thread transcript.",
			"If this is continuing work, send requested results to your parent and remain active.",
			normalThreadIdlePrompt,
		} {
			if !strings.Contains(prompt, required) {
				t.Fatalf("worker canSpawn=%v missing Core delivery contract %q", canSpawn, required)
			}
		}
		if strings.Contains(prompt, "channels_send") {
			t.Fatalf("worker canSpawn=%v assumes an app delivery tool", canSpawn)
		}
	}
}
