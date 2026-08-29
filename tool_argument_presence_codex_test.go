package core

import (
	"context"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexOmitsUnrequestedOptionalToolArguments verifies the
// behavior at the model boundary. In particular, a minimum, enum order, and
// falsy-compatible schema must not make Codex manufacture optional overrides.
//
//	RUN_CODEX_TOOL_ARGUMENT_PRESENCE_SMOKE=1 go test -v -run TestIntegration_CodexOmitsUnrequestedOptionalToolArguments -timeout 4m .
func TestIntegration_CodexOmitsUnrequestedOptionalToolArguments(t *testing.T) {
	if os.Getenv("RUN_CODEX_TOOL_ARGUMENT_PRESENCE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_TOOL_ARGUMENT_PRESENCE_SMOKE=1 to run the Codex optional-argument smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex optional-argument smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	inputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action": map[string]any{
				"type": "string", "enum": []any{"open", "close"},
			},
			"context_name": map[string]any{"type": "string"},
			"url":          map[string]any{"type": "string"},
			"timeout": map[string]any{
				"type": "integer", "minimum": 60,
				"description": "Optional override in seconds. Omit to use the server default.",
			},
			"persist": map[string]any{
				"type": "boolean", "description": "Optional persistence override. Omit to use the server default.",
			},
			"backend": map[string]any{
				"type": "string", "enum": []any{"local", "remote"},
				"description": "Optional backend override. Omit to use the server default.",
			},
			"environment": map[string]any{
				"type": "object", "description": "Optional environment overrides.",
			},
		},
		"required":             []any{"action", "context_name", "url"},
		"additionalProperties": false,
	}
	tool := NativeTool{
		Name:        "browser_session",
		Description: "Open or close a named browser context. Optional properties override server behavior and should be omitted unless the request explicitly needs them.",
		Parameters:  copyAndInjectReason(inputSchema),
	}

	provider := NewOpenAICodexProvider(token)
	messages := []Message{
		{Role: "system", Content: buildSystemPrompt("# Role\nCarry out precise browser operations.", ModeAutonomous, NewToolRegistry("test"), "", nil, nil, nil, nil)},
		{Role: "user", Content: "Open the saved browser context named customer-portal at https://example.com/account now. Use the available tool. I did not request any server-behavior overrides."},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	response, err := provider.Chat(ctx, messages, provider.Models()[ModelLarge], []NativeTool{tool}, nil, nil, nil)
	if err != nil {
		t.Fatalf("Codex optional-argument decision: %v", err)
	}

	var calls []NativeToolCall
	for _, call := range response.ToolCalls {
		if call.Name == tool.Name {
			calls = append(calls, call)
		}
	}
	if len(calls) != 1 {
		t.Fatalf("browser_session calls=%d want 1; text=%q tools=%#v", len(calls), response.Text, response.ToolCalls)
	}
	call := calls[0]
	for _, key := range []string{"action", "context_name", "url", "_reason"} {
		if _, ok := call.Args[key]; !ok {
			t.Errorf("Codex omitted required argument %q: %#v", key, call.Args)
		}
	}
	for _, key := range []string{"timeout", "persist", "backend", "environment"} {
		if value, ok := call.Args[key]; ok {
			t.Errorf("Codex manufactured optional argument %s=%q: %#v", key, value, call.Args)
		}
	}
	if len(call.Args) != 4 {
		keys := make([]string, 0, len(call.Args))
		for key := range call.Args {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		t.Errorf("Codex emitted argument keys %s, want exactly [_reason action context_name url]", strings.Join(keys, ", "))
	}

	// Verify the same emitted presence reaches the typed MCP payload. _reason
	// is a Core UI field and is removed before calling an MCP server.
	dispatchArgs := make(map[string]string, len(call.Args)-1)
	for key, value := range call.Args {
		if key != "_reason" {
			dispatchArgs[key] = value
		}
	}
	typed := mcpArgumentsFromStrings(dispatchArgs, inputSchema)
	if len(typed) != 3 {
		t.Fatalf("typed MCP payload materialized optional properties: %#v", typed)
	}
	for _, key := range []string{"action", "context_name", "url"} {
		if _, ok := typed[key]; !ok {
			t.Errorf("typed MCP payload lost %q: %#v", key, typed)
		}
	}
	t.Logf("Codex emitted exact argument keys: action, context_name, url, _reason")
}
