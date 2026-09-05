package core

import (
	"strings"
	"testing"
	"time"
)

// TestMCPToolsExcludedFromMainNativeTools verifies that MCP tools are NOT
// included in the native tool list for the main thread (nil allowlist),
// but ARE included when a sub-thread requests them via allowlist.
func TestMCPToolsExcludedFromMainNativeTools(t *testing.T) {
	tr := &ToolRegistry{tools: make(map[string]*ToolDef)}

	// Register core tools
	tr.Register(&ToolDef{Name: "pace", Description: "Set pace", Syntax: `[[pace sleep="5m"]]`, Core: true})
	tr.Register(&ToolDef{Name: "send", Description: "Send message", Syntax: `[[send id="x" message="y"]]`, Core: true})
	tr.Register(&ToolDef{Name: "spawn", Description: "Spawn thread", Syntax: `[[spawn id="x"]]`, Core: true, MainOnly: true})
	tr.Register(&ToolDef{Name: "review_history", Description: "Review history", Syntax: `[[review_history]]`, Core: true, SystemOnly: true})

	// Register MCP tools (simulating connected servers)
	tr.Register(&ToolDef{
		Name: "socialcast_create_post", Description: "Create a post",
		MCP: true, MCPServer: "socialcast",
		Handler: func(args map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{
			"text": map[string]any{"type": "string"},
		}},
	})
	tr.Register(&ToolDef{
		Name: "socialcast_list_accounts", Description: "List accounts",
		MCP: true, MCPServer: "socialcast",
		Handler: func(args map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
	})
	tr.Register(&ToolDef{
		Name: "github_list_repos", Description: "List repos",
		MCP: true, MCPServer: "github",
		Handler: func(args map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
	})

	// Register a non-MCP discoverable tool
	tr.Register(&ToolDef{Name: "web", Description: "Fetch URL", Syntax: `[[web url="..."]]`,
		Handler: func(args map[string]string) ToolResponse { return ToolResponse{Text: "ok"} },
	})

	// Main thread (nil allowlist) — should NOT include MCP tools
	mainTools := tr.NativeTools(nil, nil)
	mainToolNames := make(map[string]bool)
	for _, nt := range mainTools {
		mainToolNames[nt.Name] = true
	}

	// Core tools should be present
	if !mainToolNames["pace"] {
		t.Error("main should have 'pace'")
	}
	if !mainToolNames["send"] {
		t.Error("main should have 'send'")
	}
	if !mainToolNames["spawn"] {
		t.Error("main should have 'spawn'")
	}
	if mainToolNames["review_history"] {
		t.Error("main should NOT have system-only 'review_history'")
	}
	if !mainToolNames["web"] {
		t.Error("main should have 'web'")
	}

	// MCP tools must NOT be present
	if mainToolNames["socialcast_create_post"] {
		t.Error("main should NOT have 'socialcast_create_post' — MCP tools must be excluded")
	}
	if mainToolNames["socialcast_list_accounts"] {
		t.Error("main should NOT have 'socialcast_list_accounts' — MCP tools must be excluded")
	}
	if mainToolNames["github_list_repos"] {
		t.Error("main should NOT have 'github_list_repos' — MCP tools must be excluded")
	}

	t.Logf("Main thread native tools (%d): %v", len(mainTools), mainToolNames)

	// Sub-thread with allowlist — SHOULD include requested MCP tools
	allowlist := map[string]bool{
		"socialcast_create_post": true,
		"send":                   true,
		"done":                   true,
		"pace":                   true,
	}
	threadTools := tr.NativeTools(allowlist, nil)
	threadToolNames := make(map[string]bool)
	for _, nt := range threadTools {
		threadToolNames[nt.Name] = true
	}

	if !threadToolNames["socialcast_create_post"] {
		t.Error("thread should have 'socialcast_create_post' via allowlist")
	}
	if !threadToolNames["send"] {
		t.Error("thread should have 'send' via allowlist")
	}
	if threadToolNames["github_list_repos"] {
		t.Error("thread should NOT have 'github_list_repos' — not in allowlist")
	}
	if threadToolNames["review_history"] {
		t.Error("ordinary thread should NOT have system-only 'review_history'")
	}

	systemTools := tr.NativeTools(map[string]bool{"review_history": true, "pace": true}, nil, true)
	systemToolNames := make(map[string]bool)
	for _, nt := range systemTools {
		systemToolNames[nt.Name] = true
	}
	if !systemToolNames["review_history"] {
		t.Error("system thread should have system-only 'review_history'")
	}

	t.Logf("Thread native tools (%d): %v", len(threadTools), threadToolNames)
}

// TestActiveThreadsInjectedInDynamicContext verifies that active thread
// info reaches the agent via the per-turn dynamic context block — NOT
// via the system prompt (where it used to live and busted the cache
// every iteration). Both behaviors are checked: the system prompt is
// thread-agnostic, the dynamic context carries the thread list.
func TestActiveThreadsInjectedInDynamicContext(t *testing.T) {
	reg := &ToolRegistry{tools: make(map[string]*ToolDef)}
	reg.Register(&ToolDef{Name: "pace", Description: "Set pace", Core: true, Syntax: `[[pace]]`})

	threads := []ThreadInfo{
		{
			ID:        "price-monitor",
			Directive: "Monitor stock prices every 2 minutes",
			Tools:     []string{"stocks_get_quote", "send"},
			Iteration: 15,
			Rate:      RateSlow,
			Model:     ModelSmall,
			Started:   time.Now().Add(-10 * time.Minute),
		},
		{
			ID:        "social-media-manager",
			Directive: "Manage social media posts",
			Tools:     []string{"socialcast_create_post", "send", "done"},
			Iteration: 3,
			Rate:      RateFast,
			Model:     ModelLarge,
			Started:   time.Now().Add(-2 * time.Minute),
		},
		{
			ID:        "unconscious",
			System:    true,
			Directive: "Consolidate durable memories",
			Tools:     []string{"memory_remember"},
		},
	}

	prompt := buildSystemPrompt("Test directive", ModeAutonomous, reg, "", nil, threads, nil, nil)
	// Note: the literal string "[ACTIVE THREADS]" appears inside the
	// evolve tool's description (it's documented as a section name the
	// agent must avoid touching). The test for "section gone" is
	// therefore "no rendered thread row" — directive bodies of the
	// fake threads must NOT show up in the system prompt.
	if strings.Contains(prompt, "Monitor stock prices") || strings.Contains(prompt, "Manage social media") {
		t.Error("system prompt rendered active-thread bodies — they should only appear in the per-turn dynamic context")
	}

	// The agent still has to see them — just via the dynamic-context path.
	dyn := buildDynamicTurnContext(threads, "")

	if !strings.Contains(dyn, "[ACTIVE THREADS]") {
		t.Error("dynamic context should contain [ACTIVE THREADS] section")
	}
	if !strings.Contains(dyn, "price-monitor") {
		t.Error("dynamic context should mention price-monitor thread")
	}
	if !strings.Contains(dyn, "social-media-manager") {
		t.Error("dynamic context should mention social-media-manager thread")
	}
	if !strings.Contains(dyn, "Monitor stock prices") {
		t.Error("dynamic context should include price-monitor directive")
	}
	if !strings.Contains(dyn, "Manage social media") {
		t.Error("dynamic context should include social-media-manager directive")
	}
	if strings.Contains(dyn, "unconscious") || strings.Contains(dyn, "Consolidate durable memories") {
		t.Error("dynamic context should not expose platform-managed system threads")
	}
	if !strings.Contains(dyn, "stocks_get_quote") {
		t.Error("dynamic context should list price-monitor tools")
	}
	if !strings.Contains(dyn, "socialcast_create_post") {
		t.Error("dynamic context should list social-media-manager tools")
	}

	// The new format omits live-ticking values (age, iter, rate, model)
	// because they busted the cache every second. Confirm they're gone.
	for _, banned := range []string{"running 10m0s", "iter #15", "pace slow", "model small"} {
		if strings.Contains(dyn, banned) {
			t.Errorf("dynamic context should not contain volatile field %q", banned)
		}
	}

	t.Logf("Dynamic context:\n%s", dyn)
}

// TestNoActiveThreadsNoSection verifies no ACTIVE THREADS section is
// emitted in the dynamic-context block when there are no active threads.
// The system prompt naturally never contains a rendered section after
// the cache fix; the meaningful assertion is on buildDynamicTurnContext.
func TestNoActiveThreadsNoSection(t *testing.T) {
	if dyn := buildDynamicTurnContext(nil, ""); strings.Contains(dyn, "[ACTIVE THREADS]") {
		t.Error("dynamic context should NOT contain [ACTIVE THREADS] when no threads")
	}
	if dyn := buildDynamicTurnContext([]ThreadInfo{}, ""); strings.Contains(dyn, "[ACTIVE THREADS]") {
		t.Error("dynamic context should NOT contain [ACTIVE THREADS] when empty slice")
	}
}

func TestCopyAndInjectReasonStripsDefaultsOnlyFromModelFacingSchema(t *testing.T) {
	original := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"timeout": map[string]any{"type": "integer", "minimum": 60, "default": 1800},
			"environment": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"locale":  map[string]any{"type": "string", "default": "en-US"},
					"default": map[string]any{"type": "string", "default": "legal-property-name"},
				},
			},
		},
		"required": []string{"timeout"},
	}

	modelFacing := copyAndInjectReason(original)
	props := modelFacing["properties"].(map[string]any)
	if _, ok := props["timeout"].(map[string]any)["default"]; ok {
		t.Fatal("top-level property default reached the model-facing schema")
	}
	environment := props["environment"].(map[string]any)
	environmentProps := environment["properties"].(map[string]any)
	if _, ok := environmentProps["locale"].(map[string]any)["default"]; ok {
		t.Fatal("nested property default reached the model-facing schema")
	}
	if _, ok := environmentProps["default"]; !ok {
		t.Fatal("legal property named default was removed")
	}
	if _, ok := props["_reason"]; !ok {
		t.Fatal("_reason was not injected")
	}

	originalProps := original["properties"].(map[string]any)
	if got := originalProps["timeout"].(map[string]any)["default"]; got != 1800 {
		t.Fatalf("authoritative schema was mutated: default=%v", got)
	}
	originalEnvironment := originalProps["environment"].(map[string]any)
	if got := originalEnvironment["properties"].(map[string]any)["locale"].(map[string]any)["default"]; got != "en-US" {
		t.Fatalf("nested authoritative schema was mutated: default=%v", got)
	}
}
