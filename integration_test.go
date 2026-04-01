package main

import (
	"encoding/json"
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

func getAPIKey(t *testing.T) string {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	godotenv.Load() // load .env only when integration tests actually need it
	key := os.Getenv("FIREWORKS_API_KEY")
	if key == "" {
		t.Skip("FIREWORKS_API_KEY not set, skipping integration test")
	}
	return key
}

// drainEvents subscribes to the bus and counts chunks in the background.
// Returns a stop function that unsubscribes and returns chunk count.
func drainEvents(thinker *Thinker) func() int {
	sub := thinker.bus.SubscribeAll("drain", 500)
	chunks := 0
	done := make(chan struct{})
	go func() {
		for {
			select {
			case ev := <-sub.C:
				if ev.Type == EventChunk {
					chunks++
				}
			case <-done:
				return
			}
		}
	}()
	return func() int {
		close(done)
		thinker.bus.Unsubscribe("drain")
		return chunks
	}
}

func TestIntegration_Think(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	thinker := NewThinker(apiKey, NewFireworksProvider(apiKey))
	thinker.messages = append(thinker.messages, Message{
		Role:    "user",
		Content: "Reply with exactly one word: hello",
	})

	stop := drainEvents(thinker)
	resp, err := thinker.think()
	reply, usage := resp.Text, resp.Usage
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

	thinker := NewThinker(apiKey, NewFireworksProvider(apiKey))
	thinker.messages = append(thinker.messages, Message{
		Role:    "user",
		Content: `Reply with exactly this text and nothing else: [[reply message="test"]]`,
	})

	stop := drainEvents(thinker)
	resp2, err := thinker.think()
	reply := resp2.Text
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

// TestIntegration_NativeToolCallArrayArgs verifies that when the LLM returns
// tool calls with array/object arguments, the provider preserves them as valid
// JSON strings (not Go's %v representation). This prevents the bug where
// account_ids=[33] was sent as "33" or "[33]" with broken nested objects.
func TestIntegration_NativeToolCallArrayArgs(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	provider := NewFireworksProvider(apiKey)

	// Define a tool that requires array and object params
	tools := []NativeTool{
		{
			Name:        "create_post",
			Description: "Create a social media post. You MUST call this tool with the exact parameters given.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"account_ids": map[string]any{
						"type":        "array",
						"description": "Array of account IDs",
						"items":       map[string]any{"type": "integer"},
					},
					"text": map[string]any{
						"type":        "string",
						"description": "Post text",
					},
					"tags": map[string]any{
						"type":        "array",
						"description": "Tags for the post",
						"items":       map[string]any{"type": "string"},
					},
				},
				"required": []string{"account_ids", "text", "tags"},
			},
		},
	}

	messages := []Message{
		{Role: "system", Content: "You are a tool-calling assistant. When asked to create a post, call the create_post tool with the exact values provided. Do not add commentary."},
		{Role: "user", Content: `Call create_post with account_ids [33, 44], text "Hello world", and tags ["social", "test"].`},
	}

	resp, err := provider.Chat(messages, provider.Models()[ModelLarge], tools, func(s string) {})
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if len(resp.ToolCalls) == 0 {
		t.Fatalf("expected tool calls, got none. Text: %q", resp.Text)
	}

	tc := resp.ToolCalls[0]
	t.Logf("Tool call: %s, args: %v", tc.Name, tc.Args)

	// account_ids should be valid JSON array
	ids := tc.Args["account_ids"]
	if ids == "" {
		t.Fatal("account_ids arg is empty")
	}
	// Must be valid JSON, not Go's fmt representation like "map[...]"
	var parsedIDs []any
	if err := json.Unmarshal([]byte(ids), &parsedIDs); err != nil {
		t.Errorf("account_ids is not valid JSON: %q (error: %v)", ids, err)
	} else {
		t.Logf("account_ids parsed as JSON: %v", parsedIDs)
		if len(parsedIDs) == 0 {
			t.Error("expected non-empty account_ids array")
		}
	}

	// tags should be valid JSON array
	tags := tc.Args["tags"]
	if tags == "" {
		t.Fatal("tags arg is empty")
	}
	var parsedTags []any
	if err := json.Unmarshal([]byte(tags), &parsedTags); err != nil {
		t.Errorf("tags is not valid JSON: %q (error: %v)", tags, err)
	} else {
		t.Logf("tags parsed as JSON: %v", parsedTags)
	}

	// text should be a plain string (not JSON-wrapped)
	text := tc.Args["text"]
	if text == "" {
		t.Fatal("text arg is empty")
	}
	if strings.HasPrefix(text, `"`) {
		t.Errorf("text should be a plain string, not JSON-quoted: %q", text)
	}
	t.Logf("text: %q", text)
}

// TestIntegration_NativeToolCallNestedArrayArgs verifies that the LLM correctly
// produces nested object arrays (like media_urls with {url, type}) when the schema
// specifies items with object properties. This requires the full JSON schema to be
// sent to the LLM, not a flattened string schema.
func TestIntegration_NativeToolCallNestedArrayArgs(t *testing.T) {
	t.Parallel()
	apiKey := getAPIKey(t)

	provider := NewFireworksProvider(apiKey)

	// Mimics the socialcast create_post schema with nested objects in arrays
	tools := []NativeTool{
		{
			Name:        "create_post",
			Description: "Create a social media post with media attachments. You MUST call this tool with the exact parameters given.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"account_ids": map[string]any{
						"type":        "array",
						"description": "Array of account IDs to post to",
						"items":       map[string]any{"type": "integer"},
					},
					"text": map[string]any{
						"type":        "string",
						"description": "Post text",
					},
					"media_urls": map[string]any{
						"type":        "array",
						"description": "Media attachments",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"url":  map[string]any{"type": "string", "description": "URL of the media file"},
								"type": map[string]any{"type": "string", "enum": []string{"image", "video"}, "description": "Media type"},
							},
							"required": []string{"url", "type"},
						},
					},
				},
				"required": []string{"account_ids", "text", "media_urls"},
			},
		},
	}

	messages := []Message{
		{Role: "system", Content: "You are a tool-calling assistant. Call the tool with the exact values provided. Do not add commentary."},
		{Role: "user", Content: `Call create_post with account_ids [33], text "Check this out!", and media_urls containing one item: url "https://example.com/video.mp4" with type "video".`},
	}

	resp, err := provider.Chat(messages, provider.Models()[ModelLarge], tools, func(s string) {})
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	if len(resp.ToolCalls) == 0 {
		t.Fatalf("expected tool calls, got none. Text: %q", resp.Text)
	}

	tc := resp.ToolCalls[0]
	t.Logf("Tool call: %s", tc.Name)
	t.Logf("Args: %v", tc.Args)

	// account_ids must be valid JSON array
	ids := tc.Args["account_ids"]
	var parsedIDs []any
	if err := json.Unmarshal([]byte(ids), &parsedIDs); err != nil {
		t.Errorf("account_ids not valid JSON array: %q (error: %v)", ids, err)
	} else {
		t.Logf("account_ids: %v", parsedIDs)
	}

	// media_urls must be valid JSON array of objects
	media := tc.Args["media_urls"]
	if media == "" {
		t.Fatal("media_urls arg is empty")
	}
	var parsedMedia []map[string]any
	if err := json.Unmarshal([]byte(media), &parsedMedia); err != nil {
		t.Errorf("media_urls not valid JSON array of objects: %q (error: %v)", media, err)
	} else {
		t.Logf("media_urls: %v", parsedMedia)
		if len(parsedMedia) == 0 {
			t.Error("expected at least 1 media item")
		} else {
			item := parsedMedia[0]
			if _, ok := item["url"]; !ok {
				t.Error("media item missing 'url' field")
			}
			if _, ok := item["type"]; !ok {
				t.Error("media item missing 'type' field")
			}
		}
	}

	// text should be a plain string
	text := tc.Args["text"]
	if text == "" {
		t.Fatal("text arg is empty")
	}
	t.Logf("text: %q", text)
}

func TestIntegration_WebTool_MissingURL(t *testing.T) {
	result := webTool(map[string]string{})
	if result != "error: missing url argument" {
		t.Errorf("expected missing url error, got: %q", result)
	}
}
