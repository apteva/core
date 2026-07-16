package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func TestIntegration_CodexLargeToolResultRetention(t *testing.T) {
	if os.Getenv("RUN_CODEX_TOOL_RESULT_ARCHIVE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_TOOL_RESULT_ARCHIVE_SMOKE=1 to run the Codex large-result retention smoke")
	}
	if testing.Short() {
		t.Skip("skipping live Codex large-result smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runLiveLargeToolResultRetention(t, NewOpenAICodexProvider(token))
}

func TestIntegration_GLM52LargeToolResultRetention(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_TOOL_RESULT_ARCHIVE_SMOKE") != "1" {
		t.Skip("set RUN_OPENCODE_GO_TOOL_RESULT_ARCHIVE_SMOKE=1 to run the GLM 5.2 large-result retention smoke")
	}
	if testing.Short() {
		t.Skip("skipping live GLM large-result smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	provider := NewOpenCodeGoProvider(key)
	if provider.Models()[ModelMedium] != "glm-5.2" {
		t.Fatalf("OpenCode Go medium model = %q, want glm-5.2", provider.Models()[ModelMedium])
	}
	runLiveLargeToolResultRetention(t, provider)
}

func runLiveLargeToolResultRetention(t *testing.T, provider LLMProvider) {
	t.Helper()
	const key = "MIDDLE_RESULT_KEY_7C91F2"
	const value = "VALUE_FROM_FULL_RESULT_B93D"
	payload := strings.Repeat("catalog-before\n", 5_000) + key + "=" + value + "\n" + strings.Repeat("catalog-after\n", 5_000)
	thinker := &Thinker{
		threadID:      "main",
		session:       NewSession(t.TempDir(), "main"),
		toolResultAge: map[string]int{},
	}
	thinker.messages = []Message{
		{Role: "system", Content: "You verify ordinary tool-result context. Follow the requested exact output. No retrieval tools are available."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "catalog-call", Name: "media_search", Args: map[string]string{}}}},
	}
	resultMessage := thinker.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "catalog-call", ToolName: "media_search", Content: payload}}})
	thinker.messages = append(thinker.messages, resultMessage,
		Message{Role: "user", Content: "Read the tool result and reply with only the value after " + key + "=."},
	)

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	fullRequest := thinker.prepareToolResultRequest(thinker.messages)
	if !strings.Contains(fullRequest[2].ToolResults[0].Content, value) {
		t.Fatal("fresh request was truncated before provider consumption")
	}
	first, err := provider.Chat(ctx, fullRequest, provider.Models()[ModelMedium], nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("%s full-result call: %v", provider.Name(), err)
	}
	if strings.TrimSpace(first.Text) != value {
		t.Fatalf("%s did not consume the ordinary full tool result: %q", provider.Name(), first.Text)
	}

	for i := 0; i < toolResultFullRetentionCalls; i++ {
		thinker.markToolResultsConsumed(thinker.messages)
	}
	agedMessages := append(cloneMessages(thinker.messages), Message{
		Role:    "user",
		Content: "If the old tool result is now explicitly marked as truncated by core, reply exactly BOUNDED_HISTORY_OK.",
	})
	agedRequest := thinker.prepareToolResultRequest(agedMessages)
	agedResult := agedRequest[2].ToolResults[0]
	if strings.Contains(agedResult.Content, value) || len(agedResult.Content) > historicalToolResultPerResultChars || !strings.Contains(agedResult.Content, "truncated by core") {
		t.Fatalf("aged request was not bounded: %+v", agedResult)
	}
	second, err := provider.Chat(ctx, agedRequest, provider.Models()[ModelMedium], nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("%s bounded-history call: %v", provider.Name(), err)
	}
	if !strings.Contains(second.Text, "BOUNDED_HISTORY_OK") {
		t.Fatalf("%s did not handle bounded ordinary history: %q", provider.Name(), second.Text)
	}
	t.Logf("%s large-result retention passed: full_in=%d bounded_in=%d", provider.Name(), first.Usage.PromptTokens, second.Usage.PromptTokens)
}
