package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func xAIIntegrationKey(t *testing.T) string {
	t.Helper()
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("XAI_API_KEY"))
	if key == "" {
		t.Skip("XAI_API_KEY not set")
	}
	return key
}

func xAIIntegrationProvider(key string) LLMProvider {
	provider := NewXAIProvider(key)
	if model := strings.TrimSpace(os.Getenv("XAI_TEST_MODEL")); model != "" {
		for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
			provider.Models()[tier] = model
		}
	}
	return provider
}

// TestIntegration_XAIToolResultContinuation verifies the complete provider
// path against xAI: native function selection, tool result replay, final text,
// reasoning streaming, usage, and cache telemetry.
func TestIntegration_XAIToolResultContinuation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping paid xAI integration test in short mode")
	}
	if os.Getenv("RUN_XAI_INTEGRATION_TESTS") != "1" {
		t.Skip("set RUN_XAI_INTEGRATION_TESTS=1 to run paid xAI integration tests")
	}
	provider := xAIIntegrationProvider(xAIIntegrationKey(t))
	model := provider.Models()[ModelMedium]
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = withOpenAIPromptCacheScope(ctx, openAIPromptCacheScope{Identity: "integration/xai/tool-continuation", Epoch: 1})
	tools := []NativeTool{{
		Name:        "fixture_lookup",
		Description: "Look up the required test fixture. Always use this tool when the user requests the fixture code.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"name"},
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		},
	}}
	messages := []Message{
		{Role: "system", Content: "You are a provider integration test. You must use fixture_lookup for fixture questions and must not guess its result."},
		{Role: "user", Content: "Call fixture_lookup with name=weather, then report the exact code it returns."},
	}
	first, err := provider.Chat(ctx, messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("first xAI call: %v", err)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].Name != "fixture_lookup" {
		t.Fatalf("xAI did not issue fixture_lookup: text=%q calls=%+v", first.Text, first.ToolCalls)
	}
	call := first.ToolCalls[0]
	messages = append(messages,
		Message{Role: "assistant", Content: first.Text, Reasoning: first.Reasoning, ToolCalls: first.ToolCalls},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: call.ID, ToolName: call.Name, Content: `{"code":"XAI_TOOL_CONTINUATION_OK"}`}}},
	)
	second, err := provider.Chat(ctx, messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("second xAI call: %v", err)
	}
	if !strings.Contains(second.Text, "XAI_TOOL_CONTINUATION_OK") {
		t.Fatalf("xAI did not consume tool result: %q", second.Text)
	}
	t.Logf("xAI tool continuation passed model=%s first_cached=%d second_cached=%d", model, first.Usage.CachedTokens, second.Usage.CachedTokens)
}

func TestXAIDirectiveBehaviorSuite(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping paid xAI directive suite in short mode")
	}
	if os.Getenv("RUN_XAI_DIRECTIVE_SUITE") != "1" {
		t.Skip("set RUN_XAI_DIRECTIVE_SUITE=1 to run the xAI directive behavior suite")
	}
	key := xAIIntegrationKey(t)
	newProvider := func() LLMProvider { return xAIIntegrationProvider(key) }
	runProviderDirectiveBehaviorSuite(t, newProvider)
	t.Run("bounded one-off stays on main", func(t *testing.T) {
		runBoundedOneOffStaysOnMainSmoke(t, newProvider())
	})
}

func TestXAIRecurringInstructionUsesMainWakeLoop(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping paid xAI recurring evolve smoke in short mode")
	}
	if os.Getenv("RUN_XAI_RECURRING_EVOLVE_SMOKE") != "1" {
		t.Skip("set RUN_XAI_RECURRING_EVOLVE_SMOKE=1 to run the xAI recurring evolve smoke")
	}
	runRecurringInstructionUsesMainWakeLoop(t, xAIIntegrationProvider(xAIIntegrationKey(t)))
}
