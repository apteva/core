package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestIntegration_OllamaEmbeddingMemoryStoreSmoke proves the local Ollama
// embedding backend is usable by the MemoryStore itself: live embed call,
// remembered record gets a vector, and recall can use the store afterward.
//
// Run:
//
//	RUN_OLLAMA_MEMORY_SMOKE=1 go test -run TestIntegration_OllamaEmbeddingMemoryStoreSmoke -timeout 2m .
func TestIntegration_OllamaEmbeddingMemoryStoreSmoke(t *testing.T) {
	if os.Getenv("RUN_OLLAMA_MEMORY_SMOKE") == "" {
		t.Skip("set RUN_OLLAMA_MEMORY_SMOKE=1 to run the live Ollama memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping live Ollama memory smoke in short mode")
	}

	dim := configureOllamaEmbeddingTestEnv(t)
	inTempCwd(t)

	ms := NewMemoryStore("")
	if ms.backend == nil {
		t.Fatal("expected memory store to use Ollama embedding backend")
	}
	if ms.backend.Source != "ollama" {
		t.Fatalf("embedding source = %q, want ollama", ms.backend.Source)
	}
	if ms.backend.Model != os.Getenv("OLLAMA_EMBED_MODEL") {
		t.Fatalf("embedding model = %q, want %q", ms.backend.Model, os.Getenv("OLLAMA_EMBED_MODEL"))
	}
	if ms.backend.Dim != dim {
		t.Fatalf("embedding dim = %d, want %d", ms.backend.Dim, dim)
	}

	emb, err := ms.embed("Apteva local memory embedding smoke test")
	if err != nil {
		t.Fatalf("live Ollama embed failed: %v", err)
	}
	if len(emb) != dim {
		t.Fatalf("embedding length = %d, want %d", len(emb), dim)
	}

	if _, err := ms.Remember("Marco prefers ultramarine blue for visual design experiments.", []string{"preference", "ui"}, 0.9); err != nil {
		t.Fatalf("remember preference: %v", err)
	}
	if _, err := ms.Remember("The billing import task was completed yesterday.", []string{"audit"}, 0.6); err != nil {
		t.Fatalf("remember distractor: %v", err)
	}

	active := ms.Active()
	if len(active) != 2 {
		t.Fatalf("active memory count = %d, want 2", len(active))
	}
	for _, rec := range active {
		if len(rec.Embedding) != dim {
			t.Fatalf("record %q embedding length = %d, want %d", rec.Content, len(rec.Embedding), dim)
		}
	}

	results := ms.Recall("Which color should be used for interface mockups?", 2)
	if !memoryResultsContain(results, "ultramarine") {
		t.Fatalf("semantic recall did not return the color preference; results=%v", memoryContents(results))
	}
}

// TestIntegration_CodexOllamaUnconsciousMemorySmoke proves the full agent
// memory path with Codex as the LLM and Ollama as the embedding backend:
// review_history -> Codex decides memory_remember -> MemoryStore writes a
// 1024-dim Ollama vector -> later recall can retrieve it.
//
// Run:
//
//	RUN_CODEX_OLLAMA_MEMORY_SMOKE=1 OPENAI_CODEX_ACCESS_TOKEN=... go test -run TestIntegration_CodexOllamaUnconsciousMemorySmoke -timeout 10m .
func TestIntegration_CodexOllamaUnconsciousMemorySmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_OLLAMA_MEMORY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_OLLAMA_MEMORY_SMOKE=1 to run the Codex + Ollama memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex + Ollama memory smoke in short mode")
	}
	token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"))
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set")
	}

	dim := configureOllamaEmbeddingTestEnv(t)
	inTempCwd(t)
	writeCodexOllamaMemorySmokeHistory(t)

	cfg := NewConfig()
	cfg.Directive = "Test parent. Do not act; the unconscious thread is responsible for memory consolidation."
	cfg.Save()

	provider := NewOpenAICodexProvider(token)
	if provider.Name() != "openai-codex" {
		t.Fatalf("provider name = %q, want openai-codex", provider.Name())
	}
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()

	if parent.provider == nil || parent.provider.Name() != "openai-codex" {
		t.Fatalf("active provider = %v, want openai-codex", parent.provider)
	}
	if parent.memory == nil || parent.memory.backend == nil {
		t.Fatal("expected memory store with embedding backend")
	}
	if parent.memory.backend.Source != "ollama" {
		t.Fatalf("memory backend source = %q, want ollama", parent.memory.backend.Source)
	}
	if parent.memory.backend.Model != os.Getenv("OLLAMA_EMBED_MODEL") {
		t.Fatalf("memory backend model = %q, want %q", parent.memory.backend.Model, os.Getenv("OLLAMA_EMBED_MODEL"))
	}
	if parent.memory.backend.Dim != dim {
		t.Fatalf("memory backend dim = %d, want %d", parent.memory.backend.Dim, dim)
	}

	tools := []string{
		"review_history", "memory_search", "memory_list",
		"memory_remember", "memory_supersede", "memory_drop",
		"skill_write", "pace",
	}
	if err := parent.threads.SpawnWithOpts(
		"unconscious",
		unconsciousDirectiveV2,
		tools,
		SpawnOpts{ParentID: "main", Depth: 0, System: true},
	); err != nil {
		t.Fatalf("spawn unconscious: %v", err)
	}

	parent.bus.Publish(Event{
		Type: EventInbox,
		To:   "unconscious",
		Text: "[wake] new history available — consolidate memory now",
	})

	waitForMemory(t, parent.memory, 8*time.Minute, func(rec MemoryRecord) bool {
		return len(rec.Embedding) == dim && strings.Contains(strings.ToLower(rec.Content), "ultramarine")
	})

	active := parent.memory.Active()
	if len(active) == 0 {
		t.Fatal("expected at least one active memory")
	}
	for _, rec := range active {
		if len(rec.Embedding) == 0 {
			t.Fatalf("memory %q has no embedding", rec.Content)
		}
		if len(rec.Embedding) != dim {
			t.Fatalf("memory %q embedding length = %d, want %d", rec.Content, len(rec.Embedding), dim)
		}
	}

	results := parent.memory.Recall("Which shade should the assistant pick for future UI experiments?", 5)
	if !memoryResultsContain(results, "ultramarine") {
		t.Fatalf("recall did not retrieve the Codex-written color memory; results=%v", memoryContents(results))
	}
	t.Logf("active memories: %v", memoryContents(active))
	t.Logf("recall results: %v", memoryContents(results))
}

func configureOllamaEmbeddingTestEnv(t *testing.T) int {
	t.Helper()

	// Force the memory backend to Ollama even when developer machines also
	// have Fireworks/OpenAI API keys in their shell.
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	host := strings.TrimSpace(os.Getenv("OLLAMA_HOST"))
	if host == "" {
		host = "http://127.0.0.1:11434"
	}
	model := strings.TrimSpace(os.Getenv("OLLAMA_EMBED_MODEL"))
	if model == "" {
		model = "qwen3-embedding:0.6b"
	}
	dimRaw := strings.TrimSpace(os.Getenv("OLLAMA_EMBED_DIM"))
	if dimRaw == "" {
		dimRaw = "1024"
	}
	dim, err := strconv.Atoi(dimRaw)
	if err != nil || dim <= 0 {
		t.Fatalf("invalid OLLAMA_EMBED_DIM %q", dimRaw)
	}

	t.Setenv("OLLAMA_HOST", host)
	t.Setenv("OLLAMA_EMBED_MODEL", model)
	t.Setenv("OLLAMA_EMBED_DIM", dimRaw)
	return dim
}

func inTempCwd(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
}

func writeCodexOllamaMemorySmokeHistory(t *testing.T) {
	t.Helper()
	if err := os.MkdirAll("history", 0755); err != nil {
		t.Fatal(err)
	}
	lines := []map[string]any{
		{
			"ts":      "2026-06-18T09:00:00Z",
			"role":    "user",
			"content": "[chat] Remember this for future UI work: my preferred test color is ultramarine blue.",
		},
		{
			"ts":      "2026-06-18T09:00:05Z",
			"role":    "assistant",
			"content": "Noted.",
		},
		{
			"ts":      "2026-06-18T09:02:00Z",
			"role":    "user",
			"content": "[chat] For memory smoke tests, use that color preference when I ask about interface experiments.",
		},
	}
	var out []string
	for _, line := range lines {
		raw, err := json.Marshal(line)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, string(raw))
	}
	if err := os.WriteFile(filepath.Join("history", "main.jsonl"), []byte(strings.Join(out, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}
}

func waitForMemory(t *testing.T, memory *MemoryStore, timeout time.Duration, match func(MemoryRecord) bool) {
	t.Helper()
	deadline := time.After(timeout)
	tick := time.NewTicker(750 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for matching memory; active=%v", memoryContents(memory.Active()))
		case <-tick.C:
			for _, rec := range memory.Active() {
				if match(rec) {
					return
				}
			}
		}
	}
}

func memoryResultsContain(records []MemoryRecord, needle string) bool {
	needle = strings.ToLower(needle)
	for _, rec := range records {
		if strings.Contains(strings.ToLower(rec.Content), needle) {
			return true
		}
	}
	return false
}

func memoryContents(records []MemoryRecord) []string {
	out := make([]string, 0, len(records))
	for _, rec := range records {
		out = append(out, rec.Content)
	}
	return out
}
