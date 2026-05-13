package core

// Scenario harness — exported test infrastructure used by core's scenario
// tests AND by sibling test packages (e.g. core/scenarios/).
//
// This file is intentionally NOT a _test.go file: symbols here must be
// importable from packages outside core. The cost is that the harness
// (~470 lines) is compiled into the apteva-core binary. Net cost is
// trivial — ~100 KB of test machinery the binary never invokes.
//
// Why exported here rather than reimplemented in each test package:
// the harness reaches deep into Thinker internals (bus, threads, provider,
// quit, …) and uses unexported helpers (buildProviderPool, mainToolHandler,
// connectAndRegisterMCP, …). Exposing all of those would balloon the
// public API. Keeping the harness inside package core lets it use those
// unexported deps freely while still presenting a small, clean entrypoint
// to scenario authors: Scenario, Phase, RunScenario, plus a handful of
// helpers.

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/joho/godotenv"
)

// --- Public types ---

// Scenario defines a complete agent behavior test.
type Scenario struct {
	Name       string
	Directive  string
	MCPServers []MCPServerConfig // {{dataDir}} in Env values is replaced at runtime
	Providers  []ProviderConfig  // multi-provider pool config (optional)
	DataSetup  func(t *testing.T, dir string)
	Phases     []Phase
	Timeout    time.Duration // hard cap for entire scenario
	MaxThreads int           // peak thread count limit (0 = no limit)
	// MinPeakThreads asserts the agent autonomously spawned at least this
	// many concurrent threads at some point. Used by scenarios that test
	// whether the agent parallelises work on its own without being told to.
	MinPeakThreads int
}

// Phase is a step in a scenario.
type Phase struct {
	Name    string
	Setup   func(t *testing.T, dir string)                   // optional: inject data before this phase
	Wait    func(t *testing.T, dir string, th *Thinker) bool // poll condition (return true when done)
	Verify  func(t *testing.T, dir string, th *Thinker)      // optional: assertions after Wait succeeds
	Timeout time.Duration
}

// ScenarioAuditEntry is the row shape mock MCPs write into audit.jsonl.
type ScenarioAuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

// ChatReply represents a single chat reply parsed out of an audit log.
type ChatReply struct {
	User    string
	Message string
}

// --- Internals shared by harness + integration tests ---

// getAPIKey returns the Fireworks API key. Kept as a separate helper
// because some integration tests intentionally exercise Fireworks-
// specific behaviour (prompt caching ratio, the MiniMax stall, the
// nomic-embed embeddings endpoint) and can't be satisfied by any
// other provider. Use getTestProvider for everything else.
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

// testProvider bundles a provider with the apiKey to plumb into
// NewThinker for tests. Tests use the apiKey only for memory-store
// embeddings (which always go to Fireworks if configured); when the
// chosen provider is OpenCode Go and no Fireworks key exists, the
// memory store will detect "no embedding backend" and disable itself
// — exactly the same path Bug 1's fix added.
type testProvider struct {
	APIKey   string
	Provider LLMProvider
	Source   string // "opencode-go" | "fireworks" — for log messages
}

// getTestProvider returns whichever real LLM provider is available,
// preferring OpenCode Go (flat-rate subscription, basically free per
// call) over Fireworks (per-token billing). Tests that don't need
// Fireworks-specific features should use this helper to avoid burning
// budget on every CI / dev run.
//
// Skip rules, in order:
//   1. -short → skip (any integration test).
//   2. Neither key set → skip with a message listing both.
//   3. Otherwise: OpenCode Go > Fireworks.
//
// Tests that need a *specific* provider (cache test, MiniMax stall,
// embedding tests) should keep using getAPIKey + NewFireworksProvider
// directly.
func getTestProvider(t *testing.T) testProvider {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}
	godotenv.Load()

	// Wipe persistent agent state BEFORE handing back a provider.
	// NewThinker() loads ./history/<threadID>.jsonl on construction —
	// a previous run that crashed or was killed mid-flight can leave
	// a broken tool_call/tool_result chain on disk, which then makes
	// every subsequent test infinite-retry on a 400 from any
	// tools-strict provider (Moonshot, Anthropic, OpenAI strict mode).
	// We burned hours debugging this once; never again.
	resetPersistentAgentState(t)

	if k := os.Getenv("OPENCODE_GO_API_KEY"); k != "" {
		// memory store still talks to Fireworks for embeddings if a
		// FIREWORKS_API_KEY is also set; otherwise it disables itself.
		return testProvider{
			APIKey:   firstNonEmptyEnv("FIREWORKS_API_KEY", "OPENAI_API_KEY"),
			Provider: NewOpenCodeGoProvider(k),
			Source:   "opencode-go",
		}
	}
	if k := os.Getenv("FIREWORKS_API_KEY"); k != "" {
		return testProvider{
			APIKey:   k,
			Provider: NewFireworksProvider(k),
			Source:   "fireworks",
		}
	}
	t.Skip("no LLM provider key set (OPENCODE_GO_API_KEY or FIREWORKS_API_KEY) — skipping integration test")
	return testProvider{} // unreachable; t.Skip aborts
}

// resetPersistentAgentState wipes the on-disk Thinker state files so
// each test starts from a known-empty conversation history. Called by
// getTestProvider before any LLM provider is handed back. Re-wipes on
// t.Cleanup so the next test process also starts clean (otherwise a
// passing run would leave its own history behind for the next one).
//
// What gets wiped:
//   ./history/         per-thread JSONL conversation logs (main.jsonl etc.)
//   ./memory.jsonl     cross-session user memories surfaced into the prompt
func resetPersistentAgentState(t *testing.T) {
	t.Helper()
	paths := []string{"history", "memory.jsonl"}
	wipe := func() {
		for _, p := range paths {
			if err := os.RemoveAll(p); err != nil && !os.IsNotExist(err) {
				t.Logf("resetPersistentAgentState: could not remove %s: %v", p, err)
			}
		}
	}
	wipe()
	t.Cleanup(wipe)
}

func firstNonEmptyEnv(keys ...string) string {
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}

// moduleRoot walks up from this source file's location until it finds
// a go.mod, returning that directory. Used by BuildMCPBinary so callers
// can pass paths relative to the core module root regardless of where
// `go test` was invoked from.
func moduleRoot() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found above %s", file)
		}
		dir = parent
	}
}

// --- Public funcs ---

// RunScenario executes a scenario end-to-end against a freshly-built
// Thinker. Spawns a goroutine per scenario observer, polls each Phase's
// Wait condition until it returns true (or times out), runs Verify, and
// reports token/cost totals at the end.
func RunScenario(t *testing.T, s Scenario) {
	t.Helper()
	scenarioStart := time.Now()
	apiKey := getAPIKey(t)

	if s.Timeout == 0 {
		s.Timeout = 3 * time.Minute
	}

	// Hard deadline
	deadline := time.AfterFunc(s.Timeout, func() {
		t.Errorf("HARD TIMEOUT after %v — stopping to prevent token burn", s.Timeout)
	})
	defer deadline.Stop()

	// Data directory
	dataDir := t.TempDir()
	if s.DataSetup != nil {
		s.DataSetup(t, dataDir)
	}

	// Replace {{dataDir}} in MCP server env
	mcpServers := make([]MCPServerConfig, len(s.MCPServers))
	for i, cfg := range s.MCPServers {
		mcpServers[i] = cfg
		if mcpServers[i].Env != nil {
			env := make(map[string]string)
			for k, v := range cfg.Env {
				env[k] = strings.ReplaceAll(v, "{{dataDir}}", dataDir)
			}
			mcpServers[i].Env = env
		}
	}

	// Create thinker
	thinker := newScenarioThinker(t, apiKey, s.Directive, mcpServers, s.Providers)

	// Track peak thread count
	var peakThreads atomic.Int32
	var stopped atomic.Bool

	// Token/cost tracking
	var totalPrompt, totalCached, totalCompletion atomic.Int64
	var iterCount atomic.Int64

	// Observer: log events
	obs := thinker.bus.SubscribeAll("test-observer", 500)
	go func() {
		for !stopped.Load() {
			select {
			case ev := <-obs.C:
				switch ev.Type {
				case EventThinkDone:
					totalPrompt.Add(int64(ev.Usage.PromptTokens))
					totalCached.Add(int64(ev.Usage.CachedTokens))
					totalCompletion.Add(int64(ev.Usage.CompletionTokens))
					iterCount.Add(1)
					cost := calculateCostForProvider(thinker.provider, ev.Usage)
					ratio := 0.0
					if ev.Usage.PromptTokens > 0 {
						ratio = float64(ev.Usage.CachedTokens) / float64(ev.Usage.PromptTokens) * 100
					}
					t.Logf("[%s iter %d] threads=%d rate=%s tok=in:%d/cached:%d(%.0f%%)/out:%d $%.4f tools=%v events=%d",
						ev.From, ev.Iteration, ev.ThreadCount, ev.Rate,
						ev.Usage.PromptTokens, ev.Usage.CachedTokens, ratio, ev.Usage.CompletionTokens, cost,
						ev.ToolCalls, len(ev.ConsumedEvents))
				case EventThreadStart, EventThreadDone:
					t.Logf("[%s] %s %s", ev.From, ev.Type, ev.Text)
				case EventInbox:
					text := ev.Text
					if len(text) > 80 {
						text = text[:80] + "..."
					}
					t.Logf("[bus] %s→%s %s", ev.From, ev.To, text)
				}
			case <-thinker.quit:
				return
			}
		}
	}()

	// Track peak threads
	go func() {
		for !stopped.Load() {
			count := int32(thinker.threads.Count())
			for {
				old := peakThreads.Load()
				if count <= old || peakThreads.CompareAndSwap(old, count) {
					break
				}
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()

	go thinker.Run()
	defer func() {
		stopped.Store(true)
		thinker.Stop()
	}()

	// Run phases
	for i, phase := range s.Phases {
		t.Logf("=== Phase %d: %s ===", i+1, phase.Name)

		if phase.Setup != nil {
			phase.Setup(t, dataDir)
		}

		timeout := phase.Timeout
		if timeout == 0 {
			timeout = 60 * time.Second
		}

		if phase.Wait != nil {
			WaitFor(t, timeout, 3*time.Second, phase.Name, func() bool {
				return phase.Wait(t, dataDir, thinker)
			})
		}

		if phase.Verify != nil {
			phase.Verify(t, dataDir, thinker)
		}

		t.Logf("Phase %d PASSED", i+1)
	}

	// Check peak threads
	peak := int(peakThreads.Load())

	// Token/cost summary
	prompt := totalPrompt.Load()
	cached := totalCached.Load()
	completion := totalCompletion.Load()
	iters := iterCount.Load()
	totalTok := prompt + completion
	totalCost := calculateCostForProvider(thinker.provider, TokenUsage{
		PromptTokens: int(prompt), CachedTokens: int(cached), CompletionTokens: int(completion),
	})
	elapsed := time.Since(scenarioStart)

	cacheRatio := 0.0
	if prompt > 0 {
		cacheRatio = float64(cached) / float64(prompt) * 100
	}
	t.Logf("────────────────────────────────────────")
	t.Logf("Scenario: %s", s.Name)
	t.Logf("Duration: %s | Iterations: %d | Peak threads: %d", elapsed.Round(time.Second), iters, peak)
	t.Logf("Tokens:   %d total (in:%d cached:%d out:%d)", totalTok, prompt, cached, completion)
	t.Logf("Cache:    %.1f%% hit ratio (cached / prompt across all iterations)", cacheRatio)
	t.Logf("Cost:     $%.4f | Provider: %s", totalCost, thinker.provider.Name())
	t.Logf("────────────────────────────────────────")

	if s.MaxThreads > 0 && peak > s.MaxThreads {
		t.Errorf("peak thread count %d exceeded limit of %d", peak, s.MaxThreads)
	}
	if s.MinPeakThreads > 0 && peak < s.MinPeakThreads {
		t.Errorf("peak thread count %d below required minimum of %d — agent did not parallelise work via spawn", peak, s.MinPeakThreads)
	}

	t.Logf("=== Scenario %q PASSED ===", s.Name)
}

// BuildMCPBinary compiles the Go MCP server in `dir` and returns the
// binary path. Pass `dir` either as an absolute path, a path relative
// to the current working directory, OR a path relative to the core
// module root (e.g. "mcps/helpdesk") — the helper tries each in turn,
// so callers from sibling packages don't need to thread their cwd.
func BuildMCPBinary(t *testing.T, dir string) string {
	t.Helper()
	resolved := dir
	if !filepath.IsAbs(resolved) {
		if _, err := os.Stat(resolved); err != nil {
			// Fall back to module-root-relative resolution.
			if root, rerr := moduleRoot(); rerr == nil {
				candidate := filepath.Join(root, dir)
				if _, err := os.Stat(candidate); err == nil {
					resolved = candidate
				}
			}
		}
	}
	bin := filepath.Join(t.TempDir(), filepath.Base(resolved))
	cmd := exec.Command("go", "build", "-o", bin, ".")
	cmd.Dir = resolved
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", dir, err, out)
	}
	return bin
}

// newScenarioThinker constructs a Thinker for a single scenario run.
// Kept unexported because it pokes at Thinker's unexported fields —
// callers should go through RunScenario which wraps this.
func newScenarioThinker(t *testing.T, apiKey, directive string, mcpServers []MCPServerConfig, providerConfigs ...[]ProviderConfig) *Thinker {
	t.Helper()

	// Clean up leftover history files from previous runs
	os.RemoveAll("history")

	tmpDir := t.TempDir()

	cfg := &Config{
		path:       filepath.Join(tmpDir, "config.json"),
		Directive:  directive,
		MCPServers: mcpServers,
	}
	// Apply provider configs if provided
	if len(providerConfigs) > 0 && len(providerConfigs[0]) > 0 {
		cfg.Providers = providerConfigs[0]
	}
	cfg.Save()

	memStore := &MemoryStore{
		backend: &embeddingBackend{
			URL:    "https://api.fireworks.ai/inference/v1/embeddings",
			Model:  "nomic-ai/nomic-embed-text-v1.5",
			APIKey: apiKey,
			Header: "Bearer",
			Dim:    768,
			Source: "fireworks (harness)",
		},
		path: filepath.Join(tmpDir, "memory.jsonl"),
		byID: map[string]int{},
	}

	// Build provider pool from config + env vars
	pool, err := buildProviderPool(cfg)
	if err != nil {
		t.Fatalf("no LLM provider: %v", err)
	}
	provider := pool.Default()

	bus := NewEventBus()
	thinker := &Thinker{
		apiKey:   apiKey,
		pool:     pool,
		provider: provider,
		messages: []Message{
			{Role: "system", Content: ""},
		},
		config:    cfg,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateReactive,
		agentRate: RateSlow,
		memory:    memStore,
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
		telemetry: NewTelemetry(),
	}
	thinker.threads = NewThreadManager(thinker)
	thinker.registry = NewToolRegistry(apiKey)
	thinker.toolIndex = NewToolIndex()
	thinker.activeTools = map[string]bool{}

	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, ModeAutonomous, thinker.registry, "", nil, nil, pool, nil)}

	go thinker.registry.EmbedAll(memStore)

	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(cfg.GetDirective(), ModeAutonomous, thinker.registry, toolDocs, thinker.mcpServers, nil, thinker.pool, thinker.mcpCatalog)
	}

	// Mirror production MCP wiring (thinker.go::Run init): every MCP
	// configured for the scenario gets connected and indexed. Tools
	// register as MCP=true so they stay hidden from main's per-turn
	// tool list until activated via search_tools or spawn-time
	// MCPNames preload — same behavior the production path now uses.
	if len(mcpServers) > 0 {
		thinker.mcpServers = connectAndRegisterMCP(mcpServers, thinker.registry, thinker.toolIndex, memStore, thinker.blobs)
		thinker.mcpCatalog = computeMCPCatalog(thinker.toolIndex)
		t.Cleanup(func() {
			for _, s := range thinker.mcpServers {
				s.Close()
			}
		})
		// Rebuild messages[0] now that mcpServers + index are populated.
		thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, ModeAutonomous, thinker.registry, "", thinker.mcpServers, nil, pool, thinker.mcpCatalog)}
	}

	return thinker
}

// WaitFor polls cond at `interval` until it returns true or `timeout`
// elapses (in which case t.Fatalf fires with `desc`).
func WaitFor(t *testing.T, timeout, interval time.Duration, desc string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(interval)
	}
	t.Fatalf("timeout after %v waiting for: %s", timeout, desc)
}

// ReadAuditEntries parses the audit.jsonl file the mock MCPs write into
// dataDir. Returns nil if the file doesn't exist yet.
func ReadAuditEntries(dir string) []ScenarioAuditEntry {
	data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	if err != nil {
		return nil
	}
	var entries []ScenarioAuditEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var e ScenarioAuditEntry
		json.Unmarshal([]byte(line), &e)
		entries = append(entries, e)
	}
	return entries
}

// CountTool returns the number of audit entries with `tool == name`.
func CountTool(entries []ScenarioAuditEntry, tool string) int {
	n := 0
	for _, e := range entries {
		if e.Tool == tool {
			n++
		}
	}
	return n
}

// WriteJSONFile writes v as indented JSON to dir/name. Fatals on error.
func WriteJSONFile(t *testing.T, dir, name string, v any) {
	t.Helper()
	data, _ := json.MarshalIndent(v, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
		t.Fatal(err)
	}
}

// ThreadIDs returns the IDs of the Thinker's currently-live threads.
func ThreadIDs(th *Thinker) []string {
	var ids []string
	for _, info := range th.threads.List() {
		ids = append(ids, info.ID)
	}
	return ids
}

// AllThreadInfos walks tm and all nested ThreadManagers, returning
// flattened ThreadInfo records. Used by scenarios that assert on the
// total tree of agent threads, not just the top-level set.
func AllThreadInfos(tm *ThreadManager) []ThreadInfo {
	if tm == nil {
		return nil
	}
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	var all []ThreadInfo
	for _, t := range tm.threads {
		providerName := ""
		if t.Thinker.provider != nil {
			providerName = t.Thinker.provider.Name()
		}
		all = append(all, ThreadInfo{
			ID:       t.ID,
			ParentID: t.ParentID,
			Depth:    t.Depth,
			Provider: providerName,
		})
		if t.Children != nil {
			all = append(all, AllThreadInfos(t.Children)...)
		}
	}
	return all
}

// ReadChatReplies parses send_reply audit entries into ChatReply records
// for chat-style scenarios.
func ReadChatReplies(dir string) []ChatReply {
	entries := ReadAuditEntries(dir)
	var replies []ChatReply
	for _, e := range entries {
		if e.Tool == "send_reply" {
			replies = append(replies, ChatReply{User: e.Args["user"], Message: e.Args["message"]})
		}
	}
	return replies
}

// ChatContainsAny returns true if any reply's message contains at least
// one of the keywords (case-insensitive).
func ChatContainsAny(replies []ChatReply, keywords ...string) bool {
	for _, r := range replies {
		lower := strings.ToLower(r.Message)
		for _, kw := range keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				return true
			}
		}
	}
	return false
}
