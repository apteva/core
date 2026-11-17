package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const solCacheAcceptanceRate = 0.80

type cacheVerifiedProvider struct {
	LLMProvider
	t         *testing.T
	mu        sync.Mutex
	previous  map[string][]byte
	manifests map[string]string
	epochs    map[string]uint64
	nextStart map[string]time.Time
}

func (p *cacheVerifiedProvider) WithBuiltins([]string) LLMProvider { return p }
func (p *cacheVerifiedProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, chunk, thinking func(string), toolChunk func(string, string, string)) (ChatResponse, error) {
	scope := openAIPromptCacheScopeFromContext(ctx)
	identity := scope.Identity
	encodedTools, _ := json.Marshal(tools)
	manifest := promptCacheShortHash(encodedTools)
	wire := stabilityInputBytes(messages)
	p.mu.Lock()
	// Cache-affinity benchmarks stay below the provider's documented overflow
	// guidance. This is test-only pacing, never a runtime agent throttle.
	if p.nextStart == nil {
		p.nextStart = map[string]time.Time{}
	}
	ready := p.nextStart[identity]
	if ready.Before(time.Now()) {
		ready = time.Now()
	}
	p.nextStart[identity] = ready.Add(5 * time.Second)
	previous := p.previous[identity]
	if len(previous) > 0 && p.manifests[identity] == manifest && p.epochs[identity] == scope.Epoch && commonPrefixBytes(previous, wire) != len(previous) {
		p.t.Errorf("Core rewrote same-manifest wire prefix for %s: common=%d previous=%d", identity, commonPrefixBytes(previous, wire), len(previous))
	}
	p.previous[identity] = wire
	p.manifests[identity] = manifest
	if p.epochs == nil {
		p.epochs = map[string]uint64{}
	}
	p.epochs[identity] = scope.Epoch
	p.mu.Unlock()
	if err := waitSolCacheUntil(ctx, ready); err != nil {
		return ChatResponse{}, err
	}
	return p.LLMProvider.Chat(ctx, messages, model, tools, chunk, thinking, toolChunk)
}

func waitSolCacheUntil(ctx context.Context, ready time.Time) error {
	if delay := time.Until(ready); delay > 0 {
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
	}
	return nil
}

// Explicitly paid endpoint control. The Core workflow regression below uses
// normal agent instructions/tools instead of this synthetic fixed prefix.
func TestIntegration_SolCacheEndpointControl(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_SOL_CACHE_STABILITY") != "1" {
		t.Skip("set RUN_SOL_CACHE_STABILITY=1 for real gpt-6.1-sol cache verification")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("real cache test requires a valid Codex credential")
	}
	p := NewOpenAICodexProvider(token)
	th := &Thinker{threadID: "cache-endpoint-control"}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	ctx = th.providerSessionContext(ctx, "inference")
	ctx = withOpenAIPromptCacheScope(ctx, openAIPromptCacheScope{Identity: "cache-endpoint-control"})
	messages := []Message{{Role: "system", Content: "Use the reference exactly. Reply only with the requested token."}, {Role: "user", Content: strings.Repeat("Reference context: preserve the existing request and return only the exact requested label.\n", 1400)}}
	var warmInput, warmCached int
	var next time.Time
	for i, label := range []string{"CONTROL-ONE", "CONTROL-TWO", "CONTROL-THREE", "CONTROL-FOUR", "CONTROL-FIVE"} {
		if err := waitSolCacheUntil(ctx, next); err != nil {
			t.Fatal(err)
		}
		next = time.Now().Add(5 * time.Second)
		messages = append(messages, Message{Role: "user", Content: "Reply only " + label})
		response, err := p.Chat(ctx, messages, "gpt-6.1-sol", nil, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(response.Text) != label {
			t.Fatalf("turn %d response=%q", i+1, response.Text)
		}
		t.Logf("CONTROL turn=%d input=%d cached=%d writes=%d", i+1, response.Usage.PromptTokens, response.Usage.CachedTokens, response.Usage.CacheWriteTokens)
		if i > 0 {
			warmInput += response.Usage.PromptTokens
			warmCached += response.Usage.CachedTokens
		}
		messages = append(messages, Message{Role: "assistant", Content: response.Text, ProviderState: response.ProviderState})
	}
	ratio := float64(warmCached) / float64(warmInput)
	t.Logf("CONTROL WARM CACHE RATE %.2f%% (%d/%d)", 100*ratio, warmCached, warmInput)
	if ratio < solCacheAcceptanceRate {
		t.Fatalf("endpoint control cache rate %.2f%% below %.0f%%", 100*ratio, 100*solCacheAcceptanceRate)
	}
}

// Two actual managed threads run the ordinary Thinker loop, dispatch fixture
// tools, discover a new schema with a full automatic budget, and handle short
// confirmations without repeating operations. No synthetic prefix padding.
func TestIntegration_SolCacheManagedThreads(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_SOL_CACHE_STABILITY") != "1" {
		t.Skip("set RUN_SOL_CACHE_STABILITY=1 for real managed-thread cache verification")
	}
	if os.Getenv("RUN_SOL_CACHE_DIAGNOSTICS") == "1" {
		installSolCacheDiagnostics(t)
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("valid Codex credential required")
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Chdir(t.TempDir())
	provider := &cacheVerifiedProvider{LLMProvider: NewOpenAICodexProvider(token), t: t, previous: map[string][]byte{}, manifests: map[string]string{}}
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Supervise the isolated cache test without doing child work.", AutomaticToolLoading: &AutomaticToolLoadingConfig{Enabled: true, MaxTools: 1, MaxSchemaTokens: 2000}}
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()
	var mu sync.Mutex
	counts := map[string]int{}
	for _, name := range []string{"lookup", "receipt"} {
		catalog := mkTool(name, "Read the verified value for the supplied request_id. Call only for an explicit request; never repeat a completed read on a short confirmation.")
		catalog.InputSchema = map[string]any{"type": "object", "properties": map[string]any{"request_id": map[string]any{"type": "string"}}, "required": []string{"request_id"}}
		// Add merges by server; register both entries together below.
		parent.registry.Register(&ToolDef{Name: "probe_" + name, Description: catalog.Description, MCP: true, MCPServer: "probe", InputSchema: catalog.InputSchema, Handler: func(args map[string]string) ToolResponse {
			id := args["request_id"]
			mu.Lock()
			counts[id]++
			count := counts[id]
			mu.Unlock()
			if count != 1 {
				t.Errorf("duplicate execution for %s", id)
			}
			return ToolResponse{Text: fmt.Sprintf(`{"request_id":%q,"verified_value":%q}`, id, "VERIFIED-"+promptCacheShortHash([]byte("isolated secret "+id)))}
		}})
	}
	var catalog []mcpToolDef
	for _, name := range []string{"lookup", "receipt"} {
		def := parent.registry.Get("probe_" + name)
		catalog = append(catalog, mcpToolDef{Name: name, Description: def.Description, InputSchema: def.InputSchema, Annotations: map[string]any{"readOnlyHint": true}})
	}
	parent.toolIndex.Add("probe", catalog, false)
	parent.mcpCatalog = computeMCPCatalog(parent.toolIndex)
	directive := `This is a continuing isolated read-only service. Wait for user events; never start unrequested work.
On an explicit request naming probe_lookup or probe_receipt and request_id, call that named tool exactly once with that request_id. If its schema is absent, discover it using search_tools first in a separate discovery-only turn. Do not substitute tools.
After its result arrives, call pace alone with sleep="24h". Put the exact request_id and exact verified_value from that result in pace's _reason. This records the verified answer and waits for another event.
On short confirmations such as "yes" or "continue", do NOT repeat any read. Call pace alone with sleep="24h", and include the last completed request_id and verified_value in _reason again.
Do not use send, done, spawn, evolve, update or other tools. Every tool call must have a nonempty _reason. Do not guess verified values.`
	for _, id := range []string{"cache-a", "cache-b"} {
		if err := parent.threads.SpawnWithOpts(id, directive, []string{"pace", "search_tools"}, SpawnOpts{MCPNames: []string{"probe"}, DeferRun: true, ParentID: "main"}); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	var wg sync.WaitGroup
	for _, id := range []string{"cache-a", "cache-b"} {
		worker := parent.threads.threads[id].Thinker
		wg.Add(1)
		go func(id string, worker *Thinker) {
			defer wg.Done()
			cursor := 0
			turns := []struct{ text, requestID string }{
				{"Read probe_lookup once with request_id=" + id + "-lookup", id + "-lookup"},
				{"yes", id + "-lookup"},
				{"Read probe_receipt once with request_id=" + id + "-receipt", id + "-receipt"},
				{"continue", id + "-receipt"},
			}
			for n := 0; n < 6; n++ {
				name := "probe_lookup"
				if n%2 == 1 {
					name = "probe_receipt"
				}
				requestID := fmt.Sprintf("%s-extra-%d", id, n)
				turns = append(turns, struct{ text, requestID string }{"Read " + name + " once with request_id=" + requestID, requestID})
			}
			for turn, input := range turns {
				requestID := input.requestID
				want := "VERIFIED-" + promptCacheShortHash([]byte("isolated secret "+requestID))
				worker.InjectConsole(input.text)
				if turn == 0 {
					go worker.Run()
				}
				waiting := true
				for waiting {
					select {
					case <-ctx.Done():
						t.Errorf("%s turn %d timed out", id, turn)
						return
					case <-time.After(50 * time.Millisecond):
					}
					events, next := parent.telemetry.StoredEvents(cursor)
					cursor = next
					for _, event := range events {
						if event.ThreadID != id {
							continue
						}
						if event.Type == "llm.error" || event.Type == "tool.blocked" {
							t.Errorf("%s %s", event.Type, event.Data)
							return
						}
						if event.Type != "tool.call" {
							continue
						}
						var data ToolCallData
						if json.Unmarshal(event.Data, &data) != nil {
							continue
						}
						if data.ReasonSource != "model" || data.Reason == "" {
							t.Errorf("missing model reason: %s", event.Data)
						}
						if data.Name == "pace" {
							if !strings.Contains(data.Reason, requestID) || !strings.Contains(data.Reason, want) {
								t.Errorf("%s lost verified state: %s", id, event.Data)
								return
							}
							waiting = false
						}
					}
				}
				// Observe the completed yield before publishing another event.
				time.Sleep(100 * time.Millisecond)
			}
		}(id, worker)
	}
	wg.Wait()
	parent.threads.KillAll()
	if t.Failed() {
		return
	}
	mu.Lock()
	if len(counts) != 16 {
		t.Errorf("fixture reads=%v, want sixteen distinct operations", counts)
	}
	for _, count := range counts {
		if count != 1 {
			t.Error("duplicate fixture execution")
		}
	}
	mu.Unlock()
	events, _ := parent.telemetry.StoredEvents(0)
	for _, id := range []string{"cache-a", "cache-b"} {
		var input, cached, warmInput, warmCached, calls, searches int
		previous := ""
		var previousEpoch uint64
		for _, event := range events {
			if event.ThreadID != id {
				continue
			}
			if event.Type == "tool.call" {
				var data ToolCallData
				_ = json.Unmarshal(event.Data, &data)
				if data.Name == "search_tools" {
					searches++
				}
			}
			if event.Type != "llm.done" {
				continue
			}
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) != nil {
				t.Fatal("invalid usage")
			}
			input += data.TokensIn
			cached += data.TokensCached
			calls++
			if previous != "" && previous == data.PromptCacheStablePrefixHash && previousEpoch == data.PromptCacheEpoch {
				warmInput += data.TokensIn
				warmCached += data.TokensCached
			}
			previous = data.PromptCacheStablePrefixHash
			previousEpoch = data.PromptCacheEpoch
			t.Logf("WORKER %s call=%d input=%d cached=%d epoch=%d reason=%s manifest=%s", id, calls, data.TokensIn, data.TokensCached, data.PromptCacheEpoch, data.PromptCacheResetReason, data.PromptCacheStablePrefixHash)
		}
		if searches == 0 {
			t.Error("new tool was never explicitly discovered", id)
		}
		if warmInput == 0 {
			t.Fatal("no warmed requests", id)
		}
		ratio := float64(warmCached) / float64(warmInput)
		t.Logf("WORKER %s warmed=%.2f%% (%d/%d) all_calls=%.2f%% (%d/%d) searches=%d", id, 100*ratio, warmCached, warmInput, 100*float64(cached)/float64(input), cached, input, searches)
		if ratio < solCacheAcceptanceRate {
			t.Errorf("%s warmed cache rate %.2f%% below %.0f%%", id, 100*ratio, 100*solCacheAcceptanceRate)
		}
	}
}

func TestIntegration_SolCacheContextCorrections(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_SOL_CACHE_STABILITY") != "1" {
		t.Skip("set RUN_SOL_CACHE_STABILITY=1")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("valid Codex credential required")
	}
	p := NewOpenAICodexProvider(token)
	th := &Thinker{threadID: "context-corrections"}
	th.messages = []Message{{Role: "system", Content: buildSystemPrompt("Answer the user's reference-code question with only the latest reference section's exact code. Context updates are observations: a later value supersedes earlier values. If the latest reference section is empty, reply exactly NONE. Never perform actions.", nil, "", nil, nil, nil, nil)}}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	for i, code := range []string{"FIRST-CODE-927", "UPDATED-CODE-418", ""} {
		th.messages = append(th.messages, Message{Role: "user", Content: "What is the current reference code? Reply only the code, or NONE when empty."})
		request := th.requestContext.prepareSections(th.messages, []requestContextSection{{"reference", code}}, fmt.Sprint(i), false, true)
		callCtx := th.preparePromptCacheContext(ctx, request, nil)
		callCtx = th.providerSessionContext(callCtx, "inference")
		response, err := p.Chat(callCtx, request, "gpt-6.1-sol", nil, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		want := code
		if want == "" {
			want = "NONE"
		}
		if strings.TrimSpace(response.Text) != want {
			t.Fatalf("context correction %d: got=%q want=%q", i, response.Text, want)
		}
		if len(response.ToolCalls) != 0 {
			t.Fatal("context observation caused an action")
		}
		th.messages = append(th.messages, Message{Role: "assistant", Content: response.Text, ProviderState: response.ProviderState})
		t.Logf("CORRECTION turn=%d input=%d cached=%d answer=%s", i+1, response.Usage.PromptTokens, response.Usage.CachedTokens, want)
	}
}
