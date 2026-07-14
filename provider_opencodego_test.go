package core

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/joho/godotenv"
)

func TestOpenCodeGoDefaultsToGLM52(t *testing.T) {
	models := NewOpenCodeGoProvider("test-key").Models()
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		if got := models[tier]; got != "glm-5.2" {
			t.Errorf("tier %q model = %q, want glm-5.2", tier, got)
		}
	}
}

// TestIntegration_OpenCodeGo_NativeToolCalls verifies the regression we
// hit on instances #108/#109: kimi-k2.6 served via opencode.ai/zen/go
// must accept the OpenAI-format `tools` array and return real
// `tool_calls`, not prose that imitates them. Before the
// SupportsNativeTools() fix the provider whitelist excluded
// "opencode-go", so the agent saw the tool only as a topic and
// answered "channels_respond(channel=\"chat\", text=...)" as text.
//
// Pass OPENCODE_GO_API_KEY in env (or .env). Skips otherwise.
func TestIntegration_OpenCodeGo_NativeToolCalls(t *testing.T) {
	apiKey := getOpenCodeGoKey(t)
	prov := NewOpenCodeGoProvider(apiKey)

	if !prov.SupportsNativeTools() {
		t.Fatal("OpenCode Go provider must report SupportsNativeTools()=true — otherwise the thinker drops the tools array and the model answers in prose")
	}

	// One unambiguous tool. The description is intentionally directive
	// ("call this tool to greet") so a model that supports native tools
	// has no excuse to answer in plain text.
	tools := []NativeTool{{
		Name:        "channels_respond",
		Description: "Send a message to the user on a channel. The ONLY way to deliver text — anything outside this tool is invisible to the user.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"text", "channel"},
			"properties": map[string]any{
				"text":    map[string]any{"type": "string", "description": "Message body"},
				"channel": map[string]any{"type": "string", "description": "Channel id, e.g. \"chat\""},
			},
		},
	}}

	messages := []Message{
		{Role: "system", Content: "You are an agent. The user just connected via the `chat` channel and said hello. Respond by calling the channels_respond tool. Do not output free text."},
		{Role: "user", Content: "[chat] Hello"},
	}

	model := prov.Models()[ModelLarge]
	resp, err := prov.Chat(context.Background(), messages, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	t.Logf("model=%s text=%q tool_calls=%d", model, strings.TrimSpace(resp.Text), len(resp.ToolCalls))
	for i, tc := range resp.ToolCalls {
		t.Logf("  [%d] %s args=%v", i, tc.Name, tc.Args)
	}

	if len(resp.ToolCalls) == 0 {
		// Surface the symptom we saw on instance #109 — text that
		// looks like a tool call but isn't.
		if strings.Contains(resp.Text, "channels_respond(") {
			t.Fatalf("model emitted a fake tool-call as TEXT (%q) — provider is not sending native tools", resp.Text)
		}
		t.Fatalf("expected at least one native tool_call, got 0 (text=%q)", resp.Text)
	}

	tc := resp.ToolCalls[0]
	if tc.Name != "channels_respond" {
		t.Errorf("expected tool name 'channels_respond', got %q", tc.Name)
	}
	if got := tc.Args["channel"]; got == "" {
		t.Errorf("expected non-empty channel arg, got empty")
	}
	if got := tc.Args["text"]; got == "" {
		t.Errorf("expected non-empty text arg, got empty")
	}

	fmt.Fprintf(os.Stdout, "[OPENCODE-GO-TOOLS] model=%s tool_calls=%d args=%v\n",
		model, len(resp.ToolCalls), tc.Args)
}

// TestIntegration_OpenCodeGo_MultiTurnToolCall pins the regression
// where Moonshot Kimi K2.6 (which OpenCode Go proxies for the
// kimi-k2.6 slug) rejects an assistant message that has only
// tool_calls and no `content` field with HTTP 400. OpenAI/Fireworks
// accept the omission per spec; Moonshot does not. The fix is in
// toOpenAIMessages: always include `content`, defaulting to "".
//
// The single-turn TestIntegration_OpenCodeGo_NativeToolCalls above
// can't catch this — the bug only surfaces when we send a transcript
// that already contains an assistant tool_calls turn. So this test
// runs a real two-turn sequence:
//
//  1. user → "what's 17 * 23"  (with a calc tool defined)
//  2. capture the model's tool_call response
//  3. send turn 2: prior assistant message (carrying the tool_call)
//     + a tool result + a follow-up user message
//  4. assert no 400, and that the model can still produce a coherent
//     answer
//
// If toOpenAIMessages omits `content` from the assistant turn in
// step 3, OpenCode Go returns 400 and the test fails on Chat error.
func TestIntegration_OpenCodeGo_MultiTurnToolCall(t *testing.T) {
	apiKey := getOpenCodeGoKey(t)
	prov := NewOpenCodeGoProvider(apiKey)
	const model = "kimi-k2.6"

	tools := []NativeTool{{
		Name:        "calc",
		Description: "Compute an arithmetic expression. Returns the numeric result.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"expression"},
			"properties": map[string]any{
				"expression": map[string]any{"type": "string", "description": "Arithmetic, e.g. \"17 * 23\""},
			},
		},
	}}

	// --- Turn 1: user asks; model should call calc ---
	turn1 := []Message{
		{Role: "system", Content: "You are an agent. Use the calc tool when the user asks for arithmetic."},
		{Role: "user", Content: "What's 17 * 23?"},
	}
	resp1, err := prov.Chat(context.Background(), turn1, model, tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("turn1 Chat error: %v", err)
	}
	if len(resp1.ToolCalls) == 0 {
		t.Fatalf("turn1: expected a calc tool_call, got text=%q", resp1.Text)
	}
	call := resp1.ToolCalls[0]
	if call.Name != "calc" {
		t.Fatalf("turn1: expected calc tool, got %q", call.Name)
	}

	// --- Turn 2: assistant tool_calls + tool result + follow-up user ---
	//
	// The assistant message here is the EXACT shape that triggered
	// the 400 before the fix: Content="" + ToolCalls=[…]. Without
	// the fix toOpenAIMessages would omit `content` from the wire
	// payload and Moonshot would reject the entire request.
	turn2 := []Message{
		{Role: "system", Content: "You are an agent. Use the calc tool when the user asks for arithmetic."},
		{Role: "user", Content: "What's 17 * 23?"},
		{
			Role:      "assistant",
			Content:   "", // ← the empty-content case the bug was about
			ToolCalls: []NativeToolCall{call},
		},
		{
			Role:        "tool",
			ToolResults: []ToolResult{{CallID: call.ID, Content: "391"}},
		},
		{Role: "user", Content: "Thanks. Now what's 391 + 9?"},
	}
	resp2, err := prov.Chat(context.Background(), turn2, model, tools, nil, nil, nil)
	if err != nil {
		// This is the failure mode we're pinning. If we ever see
		// HTTP 400 here, the assistant-content omission has come back.
		t.Fatalf("turn2 Chat error (likely the empty-content regression — verify toOpenAIMessages always sends content): %v", err)
	}

	// Either the model calls calc again (good) or answers directly
	// "400" (also fine). Both prove the wire payload was accepted.
	t.Logf("turn2 ok — text=%q tool_calls=%d", strings.TrimSpace(resp2.Text), len(resp2.ToolCalls))
	if resp2.Text == "" && len(resp2.ToolCalls) == 0 {
		t.Fatalf("turn2: model produced neither text nor tool_calls — empty response")
	}
	fmt.Fprintf(os.Stdout, "[OPENCODE-GO-MULTITURN] passed — content field preserved across tool-call turn\n")
}

// TestIntegration_OpenCodeGo_BasicChat is a smoke test — does the
// endpoint respond at all to a streaming chat request with kimi-k2.6.
// Useful for separating "auth/network broken" from "tool-call format
// broken" when triaging a regression.
func TestIntegration_OpenCodeGo_BasicChat(t *testing.T) {
	apiKey := getOpenCodeGoKey(t)
	prov := NewOpenCodeGoProvider(apiKey)

	messages := []Message{
		{Role: "user", Content: "Reply with exactly one word: pong"},
	}
	resp, err := prov.Chat(context.Background(), messages, prov.Models()[ModelLarge], nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	t.Logf("text=%q tokens_in=%d tokens_out=%d", strings.TrimSpace(resp.Text), resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	if strings.TrimSpace(resp.Text) == "" {
		t.Fatal("empty text reply")
	}
}

func getOpenCodeGoKey(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	if os.Getenv("RUN_LLM_INTEGRATION_TESTS") != "1" {
		t.Skip("set RUN_LLM_INTEGRATION_TESTS=1 to run paid live-provider integration tests")
	}
	godotenv.Load()
	key := os.Getenv("OPENCODE_GO_API_KEY")
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set, skipping integration test")
	}
	return key
}
