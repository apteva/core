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

// TestIntegration_DelegatedWorkerComputerArgumentModelMatrix exercises the
// production-shaped failure across providers: a delegated worker sees
// Computer's complete browser_session schema and must not turn optional schema
// annotations or boundaries into arguments for an ordinary saved-context open.
//
//	RUN_TOOL_ARGUMENT_MODEL_MATRIX=1 RUN_LLM_INTEGRATION_TESTS=1 go test -v -run TestIntegration_DelegatedWorkerComputerArgumentModelMatrix -timeout 10m .
func TestIntegration_DelegatedWorkerComputerArgumentModelMatrix(t *testing.T) {
	if os.Getenv("RUN_TOOL_ARGUMENT_MODEL_MATRIX") != "1" {
		t.Skip("set RUN_TOOL_ARGUMENT_MODEL_MATRIX=1 to run the delegated-worker argument model matrix")
	}
	if testing.Short() {
		t.Skip("skipping delegated-worker argument model matrix in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	loadIntegrationEnv()

	inputSchema := computerBrowserSessionSchemaFixture()
	tool := NativeTool{
		Name:        "computer_browser_session",
		Description: "Session lifecycle and tab control for app-owned browsers. For an ordinary open, pass only action=open plus url or context_id/context_name. Omit every other optional field unless the task explicitly requires that override; never populate optional fields with guessed or schema-default values. environment is advanced opt-in QA/device emulation and requires environment_override=true. Omitted timeout uses Computer's 1800-second lifetime; timeout is not a task or action timeout and must be omitted unless explicitly requested.",
		Parameters:  copyAndInjectReason(inputSchema),
	}
	workerPrompt := formatThreadBasePrompt(false, false, "computer-worker", "main") +
		"\n\n[DIRECTIVE]\nPerform the assigned browser check narrowly and report the result to your parent."
	messages := []Message{
		{Role: "system", Content: workerPrompt},
		{Role: "user", Content: "[from:main] Open the saved browser context named customer-portal at https://example.com/account for a normal read-only check. I did not request any browser environment, proxy, viewport, persistence, backend, presentation, or provider-lifetime override."},
	}

	type modelCase struct {
		name     string
		provider LLMProvider
		model    string
	}
	cases := []modelCase{
		{name: "codex-terra", provider: NewOpenAICodexProvider(token), model: "gpt-5.6-terra"},
		{name: "codex-sol", provider: NewOpenAICodexProvider(token), model: "gpt-5.6-sol"},
		{name: "codex-5.5", provider: NewOpenAICodexProvider(token), model: "gpt-5.5"},
	}
	if key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY")); key != "" {
		cases = append(cases,
			modelCase{name: "opencode-kimi-k3", provider: NewOpenCodeGoProvider(key), model: "kimi-k3"},
			modelCase{name: "opencode-glm-5.2", provider: NewOpenCodeGoProvider(key), model: "glm-5.2"},
		)
	} else {
		t.Log("OPENCODE_GO_API_KEY is unavailable; skipping Kimi K3 and GLM 5.2 matrix rows")
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
			defer cancel()
			response, err := tc.provider.Chat(ctx, messages, tc.model, []NativeTool{tool}, nil, nil, nil)
			if err != nil {
				t.Fatalf("%s delegated-worker optional-argument decision: %v", tc.model, err)
			}

			var calls []NativeToolCall
			for _, call := range response.ToolCalls {
				if call.Name == tool.Name {
					calls = append(calls, call)
				}
			}
			if len(calls) != 1 {
				t.Fatalf("computer_browser_session calls=%d want 1; text=%q tools=%#v", len(calls), response.Text, response.ToolCalls)
			}
			call := calls[0]
			for _, key := range []string{"action", "context_name", "url", "_reason"} {
				if _, ok := call.Args[key]; !ok {
					t.Errorf("%s omitted required operation argument %q: %#v", tc.model, key, call.Args)
				}
			}
			wantOnly := map[string]bool{"action": true, "context_name": true, "url": true, "_reason": true}
			for key, value := range call.Args {
				if !wantOnly[key] {
					t.Errorf("%s manufactured optional argument %s=%q: %#v", tc.model, key, value, call.Args)
				}
			}
			if len(call.Args) != len(wantOnly) {
				t.Errorf("%s emitted %d arguments, want exactly four: %#v", tc.model, len(call.Args), call.Args)
			}
			t.Logf("%s delegated worker emitted arguments: %#v", tc.model, call.Args)
		})
	}
}

// computerBrowserSessionSchemaFixture mirrors Computer 0.7.86's complete
// browser_session input surface. Keep it local to Core so this regression does
// not make the otherwise-independent Core repository import an app package.
func computerBrowserSessionSchemaFixture() map[string]any {
	environment := map[string]any{
		"type": "object", "additionalProperties": false,
		"description": "Advanced QA/device-emulation override only. Omit this entire object for ordinary browsing and saved-login contexts.",
		"properties": map[string]any{
			"user_agent": map[string]any{"type": "string"},
			"locale":     map[string]any{"type": "string"},
			"languages":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "maxItems": 10},
			"timezone":   map[string]any{"type": "string"},
			"geolocation": map[string]any{
				"type": "object", "additionalProperties": false,
				"properties": map[string]any{
					"latitude":   map[string]any{"type": "number", "minimum": -90, "maximum": 90},
					"longitude":  map[string]any{"type": "number", "minimum": -180, "maximum": 180},
					"accuracy":   map[string]any{"type": "number", "minimum": 0, "maximum": 100000, "description": "Accuracy radius in meters; defaults to 100 when omitted."},
					"permission": map[string]any{"type": "string", "enum": []string{"grant", "prompt", "deny"}, "description": "Permission state; defaults to grant when omitted."},
				},
				"required": []string{"latitude", "longitude"},
			},
			"device_scale_factor": map[string]any{"type": "number", "minimum": 0.5, "maximum": 4, "description": "Defaults to 1 when omitted."},
			"mobile":              map[string]any{"type": "boolean"},
			"touch":               map[string]any{"type": "boolean"},
			"max_touch_points":    map[string]any{"type": "integer", "minimum": 1, "maximum": 20},
		},
	}
	return map[string]any{
		"type": "object", "additionalProperties": false,
		"properties": map[string]any{
			"action":              map[string]any{"type": "string", "enum": []string{"open", "status", "close", "tabs", "switch_tab", "close_tab"}},
			"session_id":          map[string]any{"type": "string"},
			"tab_id":              map[string]any{"type": "string"},
			"backend":             map[string]any{"type": "string", "enum": []string{"local", "browserbase", "steel", "browser-engine", "service"}},
			"presentation_mode":   map[string]any{"type": "string", "enum": []string{"fast", "demo"}},
			"url":                 map[string]any{"type": "string"},
			"context_id":          map[string]any{"type": "string"},
			"context_name":        map[string]any{"type": "string"},
			"provider_context_id": map[string]any{"type": "string"},
			"auto_create_context": map[string]any{"type": "boolean"},
			"persist":             map[string]any{"type": "boolean"},
			"timeout":             map[string]any{"type": "integer", "minimum": 60, "maximum": 21600, "default": 1800},
			"proxy_mode":          map[string]any{"type": "string", "enum": []string{"auto", "direct", "managed", "profile"}},
			"proxy_profile":       map[string]any{"type": "string"},
			"proxy_country":       map[string]any{"type": "string"},
			"proxy_sticky":        map[string]any{"type": "string", "enum": []string{"rotating", "session", "context"}},
			"environment_override": map[string]any{
				"type": "boolean",
			},
			"environment": environment,
			"viewport": map[string]any{
				"type": "object", "properties": map[string]any{
					"width":  map[string]any{"type": "integer"},
					"height": map[string]any{"type": "integer"},
				},
			},
		},
		"required": []string{"action"},
	}
}
