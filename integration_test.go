package main

import (
	"os"
	"strings"
	"testing"

	"github.com/joho/godotenv"
)

// Integration tests with real API calls.
//
//   Unit tests only:      go test -short
//   All tests:            go test -v -count=1
//   Integration only:     go test -v -run TestIntegration

func init() {
	godotenv.Load() // auto-load .env from project root
}

func getAPIKey(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	key := os.Getenv("FIREWORKS_API_KEY")
	if key == "" {
		t.Skip("FIREWORKS_API_KEY not set, skipping integration test")
	}
	return key
}

// drainEvents continuously drains the events channel in the background.
// Returns a stop function that stops draining and returns chunk count.
func drainEvents(thinker *Thinker) func() int {
	chunks := 0
	done := make(chan struct{})
	go func() {
		for {
			select {
			case ev := <-thinker.events:
				if ev.Chunk != "" {
					chunks++
				}
			case <-done:
				return
			}
		}
	}()
	return func() int {
		close(done)
		return chunks
	}
}

func TestIntegration_Think(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	thinker := NewThinker(apiKey)
	thinker.messages = append(thinker.messages, Message{
		Role:    "user",
		Content: "Reply with exactly one word: hello",
	})

	stop := drainEvents(thinker)
	reply, usage, err := thinker.think()
	chunks := stop()

	if err != nil {
		t.Fatalf("think() error: %v", err)
	}
	if reply == "" {
		t.Error("expected non-empty reply")
	}

	t.Logf("Reply: %q", reply)
	t.Logf("Usage: prompt=%d, completion=%d, cached=%d",
		usage.PromptTokens, usage.CompletionTokens, usage.CachedTokens)
	t.Logf("Chunks streamed: %d", chunks)
}

func TestIntegration_ThinkWithToolCall(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	thinker := NewThinker(apiKey)
	thinker.messages = append(thinker.messages, Message{
		Role:    "user",
		Content: `Reply with exactly this text and nothing else: [[reply message="test"]]`,
	})

	stop := drainEvents(thinker)
	reply, _, err := thinker.think()
	stop()

	if err != nil {
		t.Fatalf("think() error: %v", err)
	}

	calls := parseToolCalls(reply)
	t.Logf("Reply: %q", reply)
	t.Logf("Parsed %d tool calls", len(calls))
	for _, call := range calls {
		t.Logf("  Tool: %s, Args: %v", call.Name, call.Args)
	}
}

func TestIntegration_Embedding(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	ms := NewMemoryStore(apiKey)
	emb, err := ms.embed("Hello world")
	if err != nil {
		t.Fatalf("embed() error: %v", err)
	}
	if len(emb) == 0 {
		t.Fatal("expected non-empty embedding")
	}
	t.Logf("Embedding dimensions: %d", len(emb))
}

func TestIntegration_EmbeddingSimilarity(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	ms := NewMemoryStore(apiKey)

	// Run all 3 embeds concurrently
	type embResult struct {
		emb []float64
		err error
	}
	ch := make(chan embResult, 3)

	texts := []string{
		"Go programming language concurrency goroutines",
		"Golang parallel programming channels",
		"chocolate cake recipe baking",
	}
	for _, text := range texts {
		go func(s string) {
			e, err := ms.embed(s)
			ch <- embResult{e, err}
		}(text)
	}

	var embs [][]float64
	for range 3 {
		r := <-ch
		if r.err != nil {
			t.Fatalf("embed() error: %v", r.err)
		}
		embs = append(embs, r.emb)
	}

	// We sent them in order but received concurrently — re-embed to get deterministic order
	emb1, _ := ms.embed(texts[0])
	emb2, _ := ms.embed(texts[1])
	emb3, _ := ms.embed(texts[2])

	simRelated := cosineSimilarity(emb1, emb2)
	simUnrelated := cosineSimilarity(emb1, emb3)

	t.Logf("Similar topics: %f", simRelated)
	t.Logf("Unrelated topics: %f", simUnrelated)

	if simRelated <= simUnrelated {
		t.Errorf("related (%f) should be more similar than unrelated (%f)", simRelated, simUnrelated)
	}
}

func TestIntegration_MemoryStoreAndRetrieve(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	tmp, err := os.CreateTemp("", "memory_integ_*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmp.Name())
	tmp.Close()

	ms := &MemoryStore{
		apiKey:  apiKey,
		session: "integration-test",
		path:    tmp.Name(),
	}

	memories := []string{
		"User asked about Go concurrency patterns and goroutines",
		"Discussed Python async await and event loops",
		"User wants to bake a chocolate cake for a birthday",
	}
	for _, m := range memories {
		if err := ms.Store(m); err != nil {
			t.Fatalf("Store() error: %v", err)
		}
	}

	if ms.Count() != 3 {
		t.Fatalf("expected 3 memories, got %d", ms.Count())
	}

	results := ms.Retrieve("How do goroutines work in Go?", 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	t.Logf("Top result for Go query: %q", results[0].Text)
	if !strings.Contains(results[0].Text, "Go") {
		t.Errorf("expected Go-related memory as top result, got %q", results[0].Text)
	}

	results2 := ms.Retrieve("baking recipes and ingredients", 2)
	t.Logf("Top result for baking query: %q", results2[0].Text)
	if !strings.Contains(results2[0].Text, "cake") {
		t.Errorf("expected cake memory as top result, got %q", results2[0].Text)
	}

	// Persistence round-trip
	ms2 := &MemoryStore{apiKey: apiKey, path: tmp.Name()}
	ms2.load()
	if ms2.Count() != 3 {
		t.Errorf("expected 3 after reload, got %d", ms2.Count())
	}
}

func TestIntegration_WebTool(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	result := webTool(map[string]string{"url": "https://httpbin.org/get"})
	if strings.Contains(result, "error") {
		t.Fatalf("webTool error: %s", result)
	}
	if !strings.Contains(result, "httpbin") {
		t.Errorf("expected httpbin content, got: %s", truncate(result, 200))
	}
	t.Logf("webTool result length: %d chars", len(result))
}

func TestIntegration_WebTool_BadURL(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	result := webTool(map[string]string{"url": "https://thisdomaindoesnotexist12345.com"})
	if !strings.Contains(result, "error") {
		t.Error("expected error for bad URL")
	}
}

func TestIntegration_WebTool_MissingURL(t *testing.T) {
	result := webTool(map[string]string{})
	if result != "error: missing url argument" {
		t.Errorf("expected missing url error, got: %q", result)
	}
}
