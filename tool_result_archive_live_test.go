package core

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
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
		{Role: "user", Content: "Read the catalog for this verification."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "catalog-call", Name: "media_search", Args: map[string]string{}}}},
	}
	resultMessage := thinker.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: "catalog-call", ToolName: "media_search", Content: payload}}})
	thinker.messages = append(thinker.messages, resultMessage,
		Message{Role: "user", Content: "Read the tool result and reply with only the value after " + key + "=."},
	)

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	fullRequest := thinker.prepareToolResultRequest(thinker.messages)
	if !strings.Contains(fullRequest[3].ToolResults[0].Content, value) {
		t.Fatal("fresh request was truncated before provider consumption")
	}
	first, err := provider.Chat(ctx, fullRequest, provider.Models()[ModelMedium], nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("%s full-result call: %v", provider.Name(), err)
	}
	if strings.TrimSpace(first.Text) != value {
		t.Fatalf("%s did not consume the ordinary full tool result: %q", provider.Name(), first.Text)
	}

	thinker.messages = append(thinker.messages, Message{Role: "assistant", Content: "First continuation."}, Message{Role: "assistant", Content: "Second continuation."}, Message{Role: "assistant", Content: "Third continuation."}, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "pressure-call", Name: "read_document"}}}, Message{Role: "user", ToolResults: []ToolResult{{CallID: "pressure-call", Content: strings.Repeat("context-pressure ", 20000)}}})
	for i := 0; i < toolResultFullRetentionCalls; i++ {
		thinker.markToolResultsConsumed(thinker.messages)
	}
	// Production commits mature results at an aggregate retention/history
	// checkpoint instead of rewriting one result per model turn.
	if committed := thinker.commitMatureToolResults(thinker.messages); committed != 1 {
		t.Fatalf("%s committed mature results=%d, want 1", provider.Name(), committed)
	}
	agedMessages := append(cloneMessages(thinker.messages), Message{
		Role:    "user",
		Content: "If the old tool result is now explicitly marked as truncated by core, reply exactly BOUNDED_HISTORY_OK.",
	})
	agedRequest := thinker.prepareToolResultRequest(agedMessages)
	agedResult := agedRequest[3].ToolResults[0]
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

func TestIntegration_CodexToolResultArchiveRecovery(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_TOOL_RESULT_ARCHIVE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_TOOL_RESULT_ARCHIVE_SMOKE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("no valid local Codex token")
	}
	provider := toolReasonLiveProvider(t, token)
	memory, h := newHistoryTestStore(t)
	const marker = "ARCHIVE_RECOVERY_EXACT_72E91"
	prefix := strings.Repeat("irrelevant padding\n", 6000)
	payload := prefix + "RECOVERY_VALUE=" + marker + "\n" + strings.Repeat("irrelevant suffix\n", 6000)
	archive := NewToolResultArchive(filepath.Dir(memory.path), "main")
	result, err := archive.Archive(ToolResult{CallID: "source-call", ToolName: "read_document", Content: payload})
	if err != nil {
		t.Fatal(err)
	}
	preview := projectArchivedToolResult(result, 64000)
	if strings.Contains(preview.Content, marker) {
		t.Fatal("fixture exposes the supposedly omitted marker")
	}
	registry := NewToolRegistry("")
	registerSystemTools(registry, memory)
	tools := registry.NativeTools(map[string]bool{"history_search": true}, nil)
	messages := []Message{{Role: "system", Content: "Recover missing evidence through the available tool. Tool excerpts are incomplete. Do not guess. Always include a nonempty _reason in structured tool calls. After recovering RECOVERY_VALUE, reply with only its exact value."}, {Role: "user", Content: fmt.Sprintf("Prior result:\n%s\n\nFind RECOVERY_VALUE in source-call. Its line starts at UTF-8 byte offset %d. Use history_search to recover that range with limit=512, then return only the value.", preview.Content, len(prefix))}}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	first, err := provider.Chat(ctx, messages, toolReasonLiveModel, tools, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(first.ToolCalls) != 1 || first.ToolCalls[0].Name != "history_search" {
		t.Fatalf("did not request archive recovery: tools=%v", first.ToolCalls)
	}
	call := first.ToolCalls[0]
	assertLiveModelToolReason(t, registry, call)
	if call.Args["call_id"] != "source-call" || call.Args["offset"] != strconv.Itoa(len(prefix)) {
		t.Fatalf("incorrect recovery identity/range: %+v", call.Args)
	}
	response := h.search(withMemoryCaller(ctx, "main"), call.Args)
	if response.IsError {
		t.Fatal(response.Text)
	}
	messages = append(messages, Message{Role: "assistant", Content: first.Text, Reasoning: first.Reasoning, ToolCalls: first.ToolCalls, ProviderState: first.ProviderState}, Message{Role: "user", ToolResults: []ToolResult{{CallID: call.ID, ToolName: call.Name, Content: response.Text}}})
	final, err := provider.Chat(ctx, messages, toolReasonLiveModel, tools, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(final.ToolCalls) != 0 || strings.TrimSpace(final.Text) != marker {
		t.Fatalf("exact archive recovery failed: text=%q tools=%v", final.Text, final.ToolCalls)
	}
	t.Logf("%s recovered omitted evidence via history_search with a model-supplied reason", toolReasonLiveModel)
}
