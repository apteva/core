package main

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

	// Register MCP tools (simulating connected servers)
	tr.Register(&ToolDef{
		Name: "socialcast_create_post", Description: "Create a post",
		MCP: true, MCPServer: "socialcast",
		Handler: func(args map[string]string) string { return "ok" },
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{
			"text": map[string]any{"type": "string"},
		}},
	})
	tr.Register(&ToolDef{
		Name: "socialcast_list_accounts", Description: "List accounts",
		MCP: true, MCPServer: "socialcast",
		Handler: func(args map[string]string) string { return "ok" },
	})
	tr.Register(&ToolDef{
		Name: "github_list_repos", Description: "List repos",
		MCP: true, MCPServer: "github",
		Handler: func(args map[string]string) string { return "ok" },
	})

	// Register a non-MCP discoverable tool
	tr.Register(&ToolDef{Name: "web", Description: "Fetch URL", Syntax: `[[web url="..."]]`,
		Handler: func(args map[string]string) string { return "ok" },
	})

	// Main thread (nil allowlist) — should NOT include MCP tools
	mainTools := tr.NativeTools(nil)
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
		"send":                  true,
		"done":                  true,
		"pace":                  true,
	}
	threadTools := tr.NativeTools(allowlist)
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

	t.Logf("Thread native tools (%d): %v", len(threadTools), threadToolNames)
}

// TestMCPToolSummaryGenerated verifies that the MCP tool summary is generated
// correctly for the system prompt.
func TestMCPToolSummaryGenerated(t *testing.T) {
	tr := &ToolRegistry{tools: make(map[string]*ToolDef)}

	tr.Register(&ToolDef{Name: "pace", Description: "Set pace", Core: true})
	tr.Register(&ToolDef{
		Name: "socialcast_create_post", Description: "[socialcast] Create a post",
		MCP: true, MCPServer: "socialcast",
	})
	tr.Register(&ToolDef{
		Name: "socialcast_list_accounts", Description: "[socialcast] List accounts",
		MCP: true, MCPServer: "socialcast",
	})
	tr.Register(&ToolDef{
		Name: "github_list_repos", Description: "[github] List repos",
		MCP: true, MCPServer: "github",
	})

	summary := tr.MCPToolSummary()

	if summary == "" {
		t.Fatal("expected non-empty MCP tool summary")
	}

	// Should mention both servers
	if !strings.Contains(summary, "socialcast") {
		t.Error("summary should mention socialcast")
	}
	if !strings.Contains(summary, "github") {
		t.Error("summary should mention github")
	}

	// Should contain tool names
	if !strings.Contains(summary, "create_post") {
		t.Error("summary should mention create_post")
	}
	if !strings.Contains(summary, "list_repos") {
		t.Error("summary should mention list_repos")
	}

	// Should instruct to use full prefixed names
	if !strings.Contains(summary, "servername_toolname") {
		t.Error("summary should mention full prefixed naming convention")
	}

	// Should say tools are NOT directly available
	if !strings.Contains(summary, "NOT available to you directly") {
		t.Error("summary should say tools are not directly callable")
	}

	t.Logf("Summary:\n%s", summary)
}

// TestNoMCPToolsEmptySummary verifies that no summary is generated when there
// are no MCP tools.
// TestActiveThreadsInjectedInSystemPrompt verifies that active thread info
// is included in the system prompt so main never forgets running threads.
func TestActiveThreadsInjectedInSystemPrompt(t *testing.T) {
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
	}

	prompt := buildSystemPrompt("Test directive", reg, "", nil, threads)

	// Should contain ACTIVE THREADS section
	if !strings.Contains(prompt, "[ACTIVE THREADS]") {
		t.Error("prompt should contain [ACTIVE THREADS] section")
	}

	// Should list both threads
	if !strings.Contains(prompt, "price-monitor") {
		t.Error("prompt should mention price-monitor thread")
	}
	if !strings.Contains(prompt, "social-media-manager") {
		t.Error("prompt should mention social-media-manager thread")
	}

	// Should include directives
	if !strings.Contains(prompt, "Monitor stock prices") {
		t.Error("prompt should include price-monitor directive")
	}
	if !strings.Contains(prompt, "Manage social media") {
		t.Error("prompt should include social-media-manager directive")
	}

	// Should include tools
	if !strings.Contains(prompt, "stocks_get_quote") {
		t.Error("prompt should list price-monitor tools")
	}
	if !strings.Contains(prompt, "socialcast_create_post") {
		t.Error("prompt should list social-media-manager tools")
	}

	t.Logf("Prompt excerpt:\n%s", prompt[len(prompt)-500:])
}

// TestNoActiveThreadsNoSection verifies no ACTIVE THREADS section when empty.
func TestNoActiveThreadsNoSection(t *testing.T) {
	reg := &ToolRegistry{tools: make(map[string]*ToolDef)}
	prompt := buildSystemPrompt("Test", reg, "", nil, nil)
	if strings.Contains(prompt, "[ACTIVE THREADS]") {
		t.Error("prompt should NOT contain [ACTIVE THREADS] when no threads")
	}

	prompt2 := buildSystemPrompt("Test", reg, "", nil, []ThreadInfo{})
	if strings.Contains(prompt2, "[ACTIVE THREADS]") {
		t.Error("prompt should NOT contain [ACTIVE THREADS] when empty slice")
	}
}

func TestNoMCPToolsEmptySummary(t *testing.T) {
	tr := &ToolRegistry{tools: make(map[string]*ToolDef)}
	tr.Register(&ToolDef{Name: "pace", Description: "Set pace", Core: true})
	tr.Register(&ToolDef{Name: "web", Description: "Fetch URL"})

	summary := tr.MCPToolSummary()
	if summary != "" {
		t.Errorf("expected empty summary with no MCP tools, got: %q", summary)
	}
}
