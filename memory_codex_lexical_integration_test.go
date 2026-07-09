package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexUsesLexicalMemoryWithoutEmbeddings is the live
// release-gate for the no-embedding memory path:
//   - no embedding provider is configured,
//   - a memory is written without an embedding,
//   - lexical recall injects it into the agent turn,
//   - live Codex uses that recalled memory in its answer.
//
// Run:
//
//	RUN_CODEX_LEXICAL_MEMORY_SMOKE=1 go test -run TestIntegration_CodexUsesLexicalMemoryWithoutEmbeddings -timeout 5m .
//
// The test reads OPENAI_CODEX_ACCESS_TOKEN from the environment first. If
// absent, it uses the local Codex auth file at ~/.codex/auth.json without
// printing or persisting the token.
func TestIntegration_CodexUsesLexicalMemoryWithoutEmbeddings(t *testing.T) {
	if os.Getenv("RUN_CODEX_LEXICAL_MEMORY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_LEXICAL_MEMORY_SMOKE=1 to run the Codex lexical memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex lexical memory smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and ~/.codex/auth.json has no access token")
	}

	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	inTempCwd(t)

	ms := NewMemoryStore("")
	if ms.backend != nil {
		t.Fatalf("memory backend = %+v, want nil/no embeddings", ms.backend)
	}
	const sentinel = "ultramarine-blue-742"
	if _, err := ms.Remember(
		"For lexical memory smoke tests, the deployment color is "+sentinel+".",
		[]string{"deployment", "color", "lexical-memory"},
		0.95,
	); err != nil {
		t.Fatalf("remember target memory: %v", err)
	}
	if _, err := ms.Remember(
		"The billing import dry run used invoice batch beta.",
		[]string{"billing", "invoice"},
		0.7,
	); err != nil {
		t.Fatalf("remember distractor: %v", err)
	}
	for _, rec := range ms.Active() {
		if len(rec.Embedding) != 0 {
			t.Fatalf("record %q unexpectedly has embedding len=%d", rec.Content, len(rec.Embedding))
		}
	}

	query := "For the lexical memory deployment color test, what color should I use?"
	recalled := ms.Recall(query, 1)
	if !memoryResultsContain(recalled, sentinel) {
		t.Fatalf("lexical recall did not retrieve target memory; results=%v", memoryContents(recalled))
	}
	dynCtx := buildDynamicTurnContext(nil, ms.BuildContext(recalled))
	if !strings.Contains(dynCtx, sentinel) {
		t.Fatalf("dynamic context did not contain sentinel memory:\n%s", dynCtx)
	}

	provider := NewOpenAICodexProvider(token)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	resp, err := provider.Chat(ctx, []Message{
		{
			Role:    "system",
			Content: "You are an Apteva agent. Use the [memories] block when relevant. Answer the user's question exactly in the form: RESULT: <remembered color>. Do not add any other words.",
		},
		{
			Role:    "user",
			Content: dynCtx + "\n\n[2026-07-09 12:00] Events:\n• [console] " + query,
		},
	}, provider.Models()[ModelMedium], nil, nil, nil, nil)
	if err != nil {
		if isExpiredCodexCredentialError(err) {
			t.Skipf("Codex access token is expired: %v", err)
		}
		t.Fatalf("Codex lexical memory smoke failed: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Text), sentinel) {
		t.Fatalf("Codex did not use recalled lexical memory; text=%q", resp.Text)
	}
}

func codexAccessTokenForMemorySmoke(t *testing.T) string {
	t.Helper()
	loadIntegrationEnv()
	if token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN")); token != "" {
		return token
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(filepath.Join(home, ".codex", "auth.json"))
	if err != nil {
		return ""
	}
	var auth struct {
		Tokens struct {
			AccessToken string `json:"access_token"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(data, &auth); err != nil {
		return ""
	}
	return strings.TrimSpace(auth.Tokens.AccessToken)
}
