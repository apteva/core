package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexPromptCacheSmoke verifies the ChatGPT Codex endpoint
// accepts the stable cache key and reuses an append-only prompt prefix.
//
// Run:
//
//	RUN_CODEX_CACHE_SMOKE=1 go test -run TestIntegration_CodexPromptCacheSmoke -timeout 5m .
func TestIntegration_CodexPromptCacheSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_CACHE_SMOKE") == "" {
		t.Skip("set RUN_CODEX_CACHE_SMOKE=1 to run the Codex prompt-cache smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex prompt-cache smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("no valid local Codex token")
	}

	provider := NewOpenAICodexProvider(token).(*OpenAINativeProvider)
	model := strings.TrimSpace(os.Getenv("CODEX_CACHE_SMOKE_MODEL"))
	if model == "" {
		model = "gpt-5.6-terra"
	}
	stablePrefix := strings.Repeat(
		"This is stable prompt-cache test context. Preserve it exactly and answer only with the requested token. ",
		350,
	)
	messages := []Message{
		{Role: "system", Content: stablePrefix},
		{Role: "user", Content: "Reply only with CACHE-ONE."},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	var usages []TokenUsage
	for i, token := range []string{"CACHE-ONE", "CACHE-TWO", "CACHE-THREE"} {
		if i > 0 {
			messages = append(messages, Message{Role: "user", Content: "Reply only with " + token + "."})
		}
		resp, err := provider.Chat(ctx, messages, model, nil, nil, nil, nil)
		if err != nil {
			if isExpiredCodexCredentialError(err) {
				t.Skipf("Codex access token is expired: %v", err)
			}
			t.Fatalf("Codex cache call %d: %v", i+1, err)
		}
		if !strings.Contains(resp.Text, token) {
			t.Fatalf("call %d response=%q, want %s", i+1, resp.Text, token)
		}
		usages = append(usages, resp.Usage)
		messages = append(messages, Message{Role: "assistant", Content: resp.Text})
	}

	if !provider.promptCacheState().enabled() {
		t.Fatal("Codex endpoint rejected prompt_cache_key and disabled cache hints")
	}
	if usages[0].PromptTokens < 1024 {
		t.Fatalf("cache smoke prompt too short: %+v", usages[0])
	}
	if usages[1].CachedTokens == 0 && usages[2].CachedTokens == 0 {
		t.Fatalf("append-only follow-ups reported no cache reads: %+v", usages)
	}
	t.Logf("Codex cache usage: %+v", usages)
}
