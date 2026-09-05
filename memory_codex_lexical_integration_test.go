package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type codexUnconsciousRouteProvider struct {
	codex LLMProvider
	seq   atomic.Uint64
}

func (p *codexUnconsciousRouteProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	for _, tool := range tools {
		if tool.Name == "review_history" {
			return p.codex.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
		}
	}
	id := p.seq.Add(1)
	return ChatResponse{ToolCalls: []NativeToolCall{{
		ID: "main-threshold-pace-" + fmt.Sprint(id), Name: "pace",
		Args: map[string]string{"sleep": "24h", "model": "small", "reasoning": "minimal"},
	}}}, nil
}

func (p *codexUnconsciousRouteProvider) Models() map[ModelTier]string { return p.codex.Models() }
func (p *codexUnconsciousRouteProvider) Name() string                 { return p.codex.Name() }
func (p *codexUnconsciousRouteProvider) CostPer1M() (float64, float64, float64) {
	return p.codex.CostPer1M()
}
func (p *codexUnconsciousRouteProvider) SupportsNativeTools() bool {
	return p.codex.SupportsNativeTools()
}
func (p *codexUnconsciousRouteProvider) AvailableBuiltinTools() []BuiltinTool {
	return p.codex.AvailableBuiltinTools()
}
func (p *codexUnconsciousRouteProvider) SetBuiltinTools(tools []string) {
	p.codex.SetBuiltinTools(tools)
}
func (p *codexUnconsciousRouteProvider) WithBuiltins([]string) LLMProvider { return p }

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
	runUsesLexicalMemoryWithoutEmbeddings(t, NewOpenAICodexProvider(token))
}

func runUsesLexicalMemoryWithoutEmbeddings(t *testing.T, provider LLMProvider) {
	t.Helper()
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

// TestIntegration_CodexUsesEphemeralMemoryAcrossTurns verifies the complete
// thinker path: uniform lexical recall across free-form tags, live
// Codex use, turn-scoped request injection, relevance filtering, telemetry, and zero
// retrieval-context persistence across repeated turns.
func TestIntegration_CodexUsesEphemeralMemoryAcrossTurns(t *testing.T) {
	if os.Getenv("RUN_CODEX_EPHEMERAL_MEMORY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_EPHEMERAL_MEMORY_SMOKE=1 to run the Codex ephemeral memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex ephemeral memory smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runUsesEphemeralMemoryAcrossTurns(t, NewOpenAICodexProvider(token))
}

func runUsesEphemeralMemoryAcrossTurns(t *testing.T, provider LLMProvider) {
	t.Helper()
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	inTempCwd(t)
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Answer operator questions using relevant memory.",
			"",
			"# Goals",
			"- Resolve heliotrope operation questions accurately.",
			"- Do not spawn workers.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	relevantID, err := thinker.memory.Remember(
		"For operation heliotrope-echo-731, the escalation channel is cobalt-desk-904.",
		[]string{"procedure", "escalation", "heliotrope"},
		0.95,
	)
	if err != nil {
		t.Fatalf("remember relevant procedure: %v", err)
	}
	unrelatedID, err := thinker.memory.Remember(
		"Payroll capability: reconcile zephyr-wage-118 archives and tax ledgers using the finance workflow.",
		[]string{"procedure", "payroll", "finance"},
		0.95,
	)
	if err != nil {
		t.Fatalf("remember unrelated procedure: %v", err)
	}

	go thinker.Run()
	query := "What is the escalation channel for operation heliotrope-echo-731? Include the exact channel token in your response."
	seenDone := 0
	for turn := 1; turn <= 3; turn++ {
		thinker.InjectConsole(query)
		deadline := time.Now().Add(2 * time.Minute)
		turnAnswered := false
		for time.Now().Before(deadline) {
			events, _ := thinker.telemetry.StoredEvents(0)
			count := 0
			answered := false
			for _, event := range events {
				if event.Type != "llm.done" {
					continue
				}
				count++
				if count > seenDone && strings.Contains(strings.ToLower(string(event.Data)), "cobalt-desk-904") {
					answered = true
				}
			}
			if answered {
				seenDone = count
				turnAnswered = true
				break
			}
			time.Sleep(250 * time.Millisecond)
		}
		if !turnAnswered {
			t.Fatalf("turn %d did not use recalled memory", turn)
		}
	}
	thinker.Stop()

	history, err := os.ReadFile(filepath.Join("history", "main.jsonl"))
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if strings.Contains(string(history), "[memories — surfaced") || strings.Contains(string(history), ephemeralContextHeader) {
		t.Fatalf("retrieval context leaked into durable history")
	}
	if strings.Contains(string(history), "zephyr-wage-118") {
		t.Fatalf("unrelated memory leaked into conversation history")
	}

	events, _ := thinker.telemetry.StoredEvents(0)
	recalls := 0
	for _, event := range events {
		if event.Type != "memory.recall" {
			continue
		}
		var data struct {
			Matches []struct {
				ID string `json:"id"`
			} `json:"matches"`
			Ephemeral bool `json:"ephemeral"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatal(err)
		}
		// Rejected candidates are separately recorded under skipped_matches.
		// Only accepted matches were injected into the model's context.
		for _, match := range data.Matches {
			if match.ID == relevantID {
				recalls++
			}
			if match.ID == unrelatedID {
				t.Fatalf("unrelated memory was recalled: %s", event.Data)
			}
		}
		if !data.Ephemeral {
			t.Fatalf("recall telemetry missing ephemeral marker: %s", event.Data)
		}
	}
	if recalls < 3 {
		t.Fatalf("relevant memory recalled %d times, want at least 3", recalls)
	}
	t.Logf("live provider used one relevant memory across three ephemeral turns")
}

// TestIntegration_CodexAutoSpawnedUnconsciousCreatesLexicalMemory is the live
// release gate for the production startup path without embeddings:
// Config.Unconscious -> NewThinker auto-spawn -> StartAll -> Codex calls
// memory_remember -> lexical recall returns the persisted record.
//
// Run:
//
//	RUN_CODEX_UNCONSCIOUS_MEMORY_SMOKE=1 go test -run TestIntegration_CodexAutoSpawnedUnconsciousCreatesLexicalMemory -timeout 10m .
func TestIntegration_CodexAutoSpawnedUnconsciousCreatesLexicalMemory(t *testing.T) {
	if os.Getenv("RUN_CODEX_UNCONSCIOUS_MEMORY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_UNCONSCIOUS_MEMORY_SMOKE=1 to run the Codex unconscious memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex unconscious memory smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and ~/.codex/auth.json has no access token")
	}
	runAutoSpawnedUnconsciousCreatesLexicalMemory(t, NewOpenAICodexProvider(token))
}

func runAutoSpawnedUnconsciousCreatesLexicalMemory(t *testing.T, provider LLMProvider) {
	t.Helper()
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	inTempCwd(t)
	writeCodexOllamaMemorySmokeHistory(t)

	cfg := NewConfig()
	cfg.Directive = "Test parent. The unconscious thread owns memory consolidation."
	cfg.Unconscious = true
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	parent.threads.mu.RLock()
	autoSpawned := parent.threads.threads["unconscious"]
	parent.threads.mu.RUnlock()
	if autoSpawned == nil || !autoSpawned.Thinker.systemThread {
		t.Fatal("Config.Unconscious did not auto-spawn the system thread")
	}

	waitForMemory(t, parent.memory, 8*time.Minute, func(rec MemoryRecord) bool {
		return len(rec.Embedding) == 0 && strings.Contains(strings.ToLower(rec.Content), "ultramarine")
	})

	if got := fileSize("memory.jsonl"); got == 0 {
		t.Fatal("auto-spawned unconscious did not persist memory.jsonl")
	}
	results := parent.memory.Recall("Which color should be used for future interface experiments?", 5)
	if !memoryResultsContain(results, "ultramarine") {
		t.Fatalf("lexical recall missed Codex-created memory; results=%v", memoryContents(results))
	}
	t.Logf("provider-created memories: %v", memoryContents(parent.memory.Active()))
}

// TestIntegration_CodexUnconsciousHistoryGrowthCreatesPersistentMemory is the
// live release gate for the threshold-driven lifecycle:
// enabled unconscious -> initial idle cycle -> real parent events grow history
// by 50KB -> safety-floor wake -> Codex reviews history and writes memory ->
// restart loads and recalls it. Main uses a deterministic pace response so all
// paid/live LLM calls are isolated to the unconscious behavior under test.
//
// Run:
//
//	RUN_CODEX_UNCONSCIOUS_THRESHOLD_SMOKE=1 go test -run TestIntegration_CodexUnconsciousHistoryGrowthCreatesPersistentMemory -timeout 12m .
func TestIntegration_CodexUnconsciousHistoryGrowthCreatesPersistentMemory(t *testing.T) {
	if os.Getenv("RUN_CODEX_UNCONSCIOUS_THRESHOLD_SMOKE") == "" {
		t.Skip("set RUN_CODEX_UNCONSCIOUS_THRESHOLD_SMOKE=1 to run the Codex threshold memory smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex threshold memory smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runUnconsciousHistoryGrowthCreatesPersistentMemory(t, NewOpenAICodexProvider(token))
}

func runUnconsciousHistoryGrowthCreatesPersistentMemory(t *testing.T, provider LLMProvider) {
	t.Helper()
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	inTempCwd(t)

	cfg := NewConfig()
	cfg.Directive = "Process incoming test events normally. Memory consolidation belongs to the unconscious thread."
	cfg.Unconscious = true
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	routedProvider := &codexUnconsciousRouteProvider{codex: provider}
	startedAt := time.Now()
	parent := NewThinker("", routedProvider, cfg)
	stopped := false
	t.Cleanup(func() {
		if !stopped {
			parent.threads.KillAll()
			parent.Stop()
		}
	})

	waitForToolCall(t, parent.telemetry, "unconscious", "pace", startedAt, 4*time.Minute)
	// llm.done/tool.call telemetry precedes the end-of-iteration safety-state
	// update by only a few instructions; allow that baseline write to finish.
	time.Sleep(200 * time.Millisecond)
	baseline := fileSize(filepath.Join("history", "main.jsonl"))
	memoriesBefore := parent.memory.Count()

	go parent.Run()
	const sentinel = "viridian green"
	padding := strings.Repeat("Routine build output from the current deployment review completed successfully. ", 550)
	parent.InjectConsole(padding)
	parent.InjectConsole(padding)
	parent.InjectConsole("Please remember my lasting preference: deployment dashboards should use " + sentinel + " for status accents.")
	parent.InjectConsole("For future deployment dashboard work, keep " + sentinel + " as my preferred status accent color.")
	waitForHistoryGrowth(t, baseline+unconsciousByteThreshold, 15*time.Second)

	thresholdAt := time.Now()
	ticks := make(chan time.Time, 1)
	floorDone := make(chan struct{})
	go func() {
		parent.runUnconsciousSafetyFloors(ticks, func() int64 { return fileSize(filepath.Join("history", "main.jsonl")) })
		close(floorDone)
	}()
	ticks <- thresholdAt

	waitForToolCall(t, parent.telemetry, "unconscious", "review_history", thresholdAt, time.Minute)
	waitForToolCall(t, parent.telemetry, "unconscious", "memory_remember", thresholdAt, 6*time.Minute)
	waitForMemory(t, parent.memory, time.Minute, func(record MemoryRecord) bool {
		return len(record.Embedding) == 0 && strings.Contains(strings.ToLower(record.Content), sentinel)
	})
	if parent.memory.Count() <= memoriesBefore {
		t.Fatalf("memory count did not increase: before=%d after=%d", memoriesBefore, parent.memory.Count())
	}

	parent.threads.KillAll()
	parent.Stop()
	stopped = true
	select {
	case <-floorDone:
	case <-time.After(time.Second):
		t.Fatal("threshold safety-floor loop did not stop")
	}

	if fileSize("memory.jsonl") == 0 {
		t.Fatal("Codex threshold memory was not persisted")
	}
	reloaded := NewMemoryStore("")
	results := reloaded.Recall("Which status accent color does the user prefer for deployment dashboards?", 5)
	if !memoryResultsContain(results, sentinel) {
		t.Fatalf("restarted memory store could not recall %s: %v", sentinel, memoryContents(results))
	}
	t.Logf("threshold wake persisted and reloaded memories: %v", memoryContents(reloaded.Active()))
}

func waitForToolCall(t *testing.T, telemetry *Telemetry, threadID, toolName string, after time.Time, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		events, _ := telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != threadID || event.Type != "tool.call" || event.Time.Before(after) {
				continue
			}
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) == nil && data.Name == toolName {
				return
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s tool call on %s", toolName, threadID)
}

func codexAccessTokenForMemorySmoke(t *testing.T) string {
	t.Helper()
	loadIntegrationEnv()
	if token := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN")); codexTokenValidForSmoke(token) {
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
	token := strings.TrimSpace(auth.Tokens.AccessToken)
	if !codexTokenValidForSmoke(token) {
		return ""
	}
	return token
}

func codexTokenValidForSmoke(token string) bool {
	if token == "" {
		return false
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return true
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return true
	}
	var claims struct {
		ExpiresAt int64 `json:"exp"`
	}
	if json.Unmarshal(payload, &claims) != nil || claims.ExpiresAt == 0 {
		return true
	}
	return time.Until(time.Unix(claims.ExpiresAt, 0)) > time.Minute
}

func TestCodexTokenValidForMemorySmokeRejectsExpiredJWT(t *testing.T) {
	jwt := func(exp time.Time) string {
		payload, err := json.Marshal(map[string]int64{"exp": exp.Unix()})
		if err != nil {
			t.Fatal(err)
		}
		return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
	}
	if codexTokenValidForSmoke(jwt(time.Now().Add(-time.Minute))) {
		t.Fatal("expired JWT was accepted")
	}
	if !codexTokenValidForSmoke(jwt(time.Now().Add(time.Hour))) {
		t.Fatal("fresh JWT was rejected")
	}
}
