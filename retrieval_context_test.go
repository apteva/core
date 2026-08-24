package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestAppendEphemeralTurnContextPreservesDurablePrefix(t *testing.T) {
	base := []Message{
		{Role: "system", Content: "stable system"},
		{Role: "user", Content: "raw event"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "lookup"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: "result"}}},
	}
	request := appendEphemeralTurnContext(base, "[memories — surfaced]\n- relevant fact", "2026-07-10 14:00", false)
	if len(request) != len(base)+1 {
		t.Fatalf("request messages = %d, want %d", len(request), len(base)+1)
	}
	for i := range base {
		if request[i].Role != base[i].Role || request[i].Content != base[i].Content {
			t.Fatalf("durable prefix changed at %d: got=%+v want=%+v", i, request[i], base[i])
		}
	}
	if !strings.HasPrefix(request[len(request)-1].Content, ephemeralContextHeader) {
		t.Fatalf("last request message is not ephemeral context: %q", request[len(request)-1].Content)
	}
	if len(base) != 4 || strings.Contains(base[len(base)-1].Content, "memories") {
		t.Fatal("base conversation was mutated")
	}

	wire := toOpenAIMessages(request)
	if len(wire) < 3 {
		t.Fatalf("OpenAI wire messages too short: %+v", wire)
	}
	nativeInput := (&OpenAINativeProvider{}).buildInput(request)
	if len(nativeInput) < 3 {
		t.Fatalf("OpenAI native input too short: %+v", nativeInput)
	}
	last := nativeInput[len(nativeInput)-1]
	if last.Type != "message" || last.Role != "user" || !strings.HasPrefix(last.Content.(string), ephemeralContextHeader) {
		t.Fatalf("OpenAI native ephemeral context is not last: %+v", last)
	}
	foundToolResult := false
	for _, item := range nativeInput[:len(nativeInput)-1] {
		if item.Type == "function_call_output" && item.CallID == "call-1" {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Fatalf("tool result missing before ephemeral tail: %+v", nativeInput)
	}
}

func TestEphemeralTurnContextAlwaysIncludesCurrentUTC(t *testing.T) {
	content := renderEphemeralTurnContext("", "2026-07-13T12:03:00Z", false)
	for _, want := range []string{ephemeralContextHeader, "[CURRENT TIME]", "UTC: 2026-07-13T12:03:00Z"} {
		if !strings.Contains(content, want) {
			t.Fatalf("context missing %q: %q", want, content)
		}
	}

	state := ephemeralTurnContextState{}
	base := []Message{{Role: "system", Content: "stable"}}
	first := state.prepare(base, "", "2026-07-13T12:03:00Z", true, true)
	second := state.prepare(base, "", "2026-07-14T12:03:00Z", true, true)
	if len(first) != 2 || len(second) != 2 {
		t.Fatalf("timer wake accumulated request context: first=%+v second=%+v", first, second)
	}
	latest := second[len(second)-1].Content
	if first[1].Content == latest || !strings.Contains(latest, "2026-07-14T12:03:00Z") {
		t.Fatalf("timer wake reused stale time: first=%q latest=%q", first[1].Content, latest)
	}
	if strings.Contains(latest, "2026-07-13T12:03:00Z") {
		t.Fatalf("timer wake retained superseded time: %q", latest)
	}
}

func TestEphemeralTurnContextStatePreservesAppendOnlyPrefix(t *testing.T) {
	state := ephemeralTurnContextState{}
	durable := []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "task"}}
	first := state.prepare(durable, "[memories — surfaced]\n- relevant", "2026-07-10 14:00", false, true)
	if len(first) != 3 || !first[2].RequestContext {
		t.Fatalf("first request context = %+v", first)
	}

	durable = append(durable,
		Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "lookup"}}},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: "result"}}},
	)
	second := state.prepare(durable, "[memories — surfaced]\n- relevant", "2026-07-10 14:01", false, false)
	if !reflect.DeepEqual(first, second[:len(first)]) {
		t.Fatalf("prior request is not an exact prefix:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if !second[2].RequestContext || second[2].Content != first[2].Content {
		t.Fatalf("request context moved or changed: first=%+v second=%+v", first[2], second[2])
	}
	firstInput := (&OpenAINativeProvider{}).buildInput(first)
	secondInput := (&OpenAINativeProvider{}).buildInput(second)
	if len(secondInput) < len(firstInput) || !reflect.DeepEqual(firstInput, secondInput[:len(firstInput)]) {
		t.Fatalf("native Responses input lost append-only prefix:\nfirst=%+v\nsecond=%+v", firstInput, secondInput)
	}
}

func TestEphemeralTurnContextNewSemanticTurnReplacesSnapshot(t *testing.T) {
	state := ephemeralTurnContextState{}
	durable := []Message{{Role: "system", Content: "stable"}, {Role: "user", Content: "first task"}}
	first := state.prepare(durable, "[ACTIVE THREADS]\n- worker-a", "2026-07-10T14:00:00Z", false, true)
	if len(first) != len(durable)+1 || strings.Count(first[len(first)-1].Content, ephemeralContextHeader) != 1 {
		t.Fatalf("initial request context = %+v", first)
	}

	durable = append(durable,
		Message{Role: "assistant", Content: "first answer"},
		Message{Role: "user", Content: "second task"},
	)
	second := state.prepare(durable, "[ACTIVE THREADS]\n- worker-b", "2026-07-10T14:05:00Z", false, true)
	if len(second) != len(durable)+1 {
		t.Fatalf("new semantic turn accumulated context: durable=%d second=%+v", len(durable), second)
	}
	latest := second[len(second)-1]
	if !latest.RequestContext || !strings.Contains(latest.Content, "worker-b") || !strings.Contains(latest.Content, "14:05:00Z") {
		t.Fatalf("latest context snapshot = %+v", latest)
	}
	serialized, _ := json.Marshal(second)
	if strings.Contains(string(serialized), "worker-a") || strings.Count(string(serialized), ephemeralContextHeader) != 1 {
		t.Fatalf("superseded request context survived replacement: %s", serialized)
	}
}

func TestRecallQueryPriorityNeverUsesAssistantFiller(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "[2026-07-10 13:00] Events:\n• [console] Investigate storage uploads"},
		{Role: "assistant", Content: "I'll wait for follow-up."},
	}
	query, source := recallQueryForTurn(nil, messages, "Monitor deployments")
	if source != "latest_user_context" || !strings.Contains(query, "storage uploads") {
		t.Fatalf("query=%q source=%q", query, source)
	}
	if strings.Contains(query, "wait for follow-up") {
		t.Fatalf("assistant filler became recall query: %q", query)
	}

	query, source = recallQueryForTurn([]string{"[console] Check invoices"}, messages, "Monitor deployments")
	if source != "event" || query != "[console] Check invoices" {
		t.Fatalf("event query=%q source=%q", query, source)
	}

	query, source = recallQueryForTurn(nil, []Message{{Role: "system", Content: "system"}}, "Monitor deployments")
	if source != "directive" || query != "Monitor deployments" {
		t.Fatalf("directive query=%q source=%q", query, source)
	}
}

func TestRecallQueriesAlwaysIncludeDirectiveWithExternalContext(t *testing.T) {
	queries, source := recallQueriesForTurn(
		[]string{"[from:main] Begin now and use the shared guidance."},
		[]Message{{Role: "system", Content: "worker"}},
		"Inspect Patreon drafts and follow the Patreon scheduling procedure.",
	)
	if source != "event+directive" {
		t.Fatalf("source = %q, want event+directive", source)
	}
	want := []string{
		"[from:main] Begin now and use the shared guidance.",
		"Inspect Patreon drafts and follow the Patreon scheduling procedure.",
	}
	if !reflect.DeepEqual(queries, want) {
		t.Fatalf("queries = %#v, want %#v", queries, want)
	}
}

func TestMemoryRecallCycleRefreshesOnlyForExternalContextChanges(t *testing.T) {
	var state memoryRecallCycleState
	if got := state.refreshReason(0, "directive", false, false, false); got != "thread_start" {
		t.Fatalf("initial refresh reason = %q", got)
	}
	state.set(0, "directive", "memory block", "directive", 1, nil)
	if got := state.refreshReason(0, "directive", false, false, false); got != "" {
		t.Fatalf("internal continuation refreshed memory: %q", got)
	}
	if got := state.refreshReason(0, "directive", true, false, false); got != "external_event" {
		t.Fatalf("external event refresh reason = %q", got)
	}
	if got := state.refreshReason(0, "directive", false, true, false); got != "timer" {
		t.Fatalf("timer refresh reason = %q", got)
	}
	if got := state.refreshReason(0, "directive", false, false, true); got != "resume" {
		t.Fatalf("resume refresh reason = %q", got)
	}
	if got := state.refreshReason(1, "directive", false, false, false); got != "memory_changed" {
		t.Fatalf("memory mutation refresh reason = %q", got)
	}
	if got := state.refreshReason(0, "updated directive", false, false, false); got != "directive_changed" {
		t.Fatalf("directive refresh reason = %q", got)
	}
}

func TestStripLegacyDynamicContextKeepsOnlyRealEvent(t *testing.T) {
	legacy := "[ACTIVE THREADS]\n- worker\n\n[memories — surfaced because they may be relevant]\n- large capability\n\n[2026-07-10 14:00] Events:\n• [console] Real request\n"
	cleaned, synthetic := stripLegacyDynamicContext(legacy)
	if !synthetic {
		t.Fatal("legacy dynamic context was not detected")
	}
	want := "[2026-07-10 14:00] Events:\n• [console] Real request"
	if cleaned != want {
		t.Fatalf("cleaned=%q want=%q", cleaned, want)
	}

	cleaned, synthetic = stripLegacyDynamicContext("[memories — surfaced because they may be relevant]\n- large capability")
	if !synthetic || cleaned != "" {
		t.Fatalf("synthetic-only block cleaned=%q detected=%v", cleaned, synthetic)
	}
}

func TestSessionLoadTailFiltersLegacyDynamicMessages(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "main")
	if err := session.Append(SessionEntry{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- duplicate capability"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- fact\n\n[2026-07-10 14:00] Events:\n• [console] Keep me"}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "assistant", Content: "done"}); err != nil {
		t.Fatal(err)
	}

	messages, _ := session.LoadTail(10)
	if len(messages) != 2 {
		t.Fatalf("loaded %d messages, want 2: %+v", len(messages), messages)
	}
	if strings.Contains(messages[0].Content, "memories") || !strings.Contains(messages[0].Content, "Keep me") {
		t.Fatalf("combined legacy message not cleaned: %q", messages[0].Content)
	}

	combined, _, recent := buildCompactionParts([]SessionEntry{
		{Role: "user", Content: "[memories — surfaced because they may be relevant]\n- duplicate capability"},
		{Role: "user", Content: "real task"},
		{Role: "assistant", Content: "done"},
	}, 1)
	if strings.Contains(combined, "duplicate capability") || strings.Contains(combined, "[memories") {
		t.Fatalf("compaction input retained synthetic memory: %q", combined)
	}
	if len(recent) != 1 || recent[0].Content != "done" {
		t.Fatalf("unexpected recent entries: %+v", recent)
	}
}

func TestSessionLoadTailFiltersLegacySystemThreadStartup(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "main")
	startup := "[2026-07-12 11:26] Events:\n• [thread:unconscious] started (provider: openai-codex, role: worker, tools: memory_search, pace)\n"
	combined := "[2026-07-12 11:27] Events:\n• [thread:unconscious] started (provider: openai-codex, role: worker, tools: memory_search, pace)\n• [console] Keep this request\n"
	if err := session.Append(SessionEntry{Role: "user", Content: startup}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "user", Content: combined}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "assistant", Content: "done"}); err != nil {
		t.Fatal(err)
	}

	messages, _ := session.LoadTail(10)
	if len(messages) != 2 {
		t.Fatalf("loaded %d messages, want combined event + assistant: %+v", len(messages), messages)
	}
	if strings.Contains(messages[0].Content, "unconscious") || strings.Contains(messages[0].Content, "memory_search") {
		t.Fatalf("legacy system startup leaked through: %q", messages[0].Content)
	}
	if !strings.Contains(messages[0].Content, "Keep this request") {
		t.Fatalf("real event was removed with system startup: %q", messages[0].Content)
	}
}

func TestSessionLoadTailFiltersLegacySystemThreadControlAttempt(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "main")
	if err := session.Append(SessionEntry{
		Role:      "assistant",
		Content:   "I’ll delegate to the memory worker.",
		ToolCalls: []NativeToolCall{{ID: "call-system", Name: "send", Args: map[string]string{"id": "unconscious", "message": "search"}}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{
		Role:        "user",
		ToolResults: []ToolResult{{CallID: "call-system", Content: `error: thread "unconscious" is platform-managed`}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := session.Append(SessionEntry{Role: "user", Content: "real request"}); err != nil {
		t.Fatal(err)
	}

	messages, _ := session.LoadTail(10)
	if len(messages) != 1 || messages[0].Content != "real request" {
		t.Fatalf("system control attempt survived reload: %+v", messages)
	}
}

type retrievalCaptureProvider struct {
	mu       sync.Mutex
	requests [][]Message
	called   chan struct{}
	sleep    string
}

type retrievalToolCycleProvider struct {
	mu       sync.Mutex
	requests [][]Message
	called   chan struct{}
	calls    int
}

func (p *retrievalToolCycleProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	p.calls++
	callNumber := p.calls
	p.requests = append(p.requests, cloneMessages(messages))
	p.mu.Unlock()
	select {
	case p.called <- struct{}{}:
	default:
	}
	if callNumber <= 8 {
		return ChatResponse{ToolCalls: []NativeToolCall{{
			ID:   fmt.Sprintf("observe-%d", callNumber),
			Name: "cycle_observe",
			Args: map[string]string{"step": fmt.Sprint(callNumber)},
		}}}, nil
	}
	return ChatResponse{ToolCalls: []NativeToolCall{{
		ID:   fmt.Sprintf("pace-%d", callNumber),
		Name: "pace",
		Args: map[string]string{"sleep": "1h", "_reason": "The observation cycle is complete"},
	}}}, nil
}

func (p *retrievalToolCycleProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "capture", ModelMedium: "capture", ModelSmall: "capture"}
}
func (p *retrievalToolCycleProvider) Name() string { return "retrieval-tool-cycle" }
func (p *retrievalToolCycleProvider) CostPer1M() (float64, float64, float64) {
	return 0, 0, 0
}
func (p *retrievalToolCycleProvider) SupportsNativeTools() bool            { return true }
func (p *retrievalToolCycleProvider) AvailableBuiltinTools() []BuiltinTool { return nil }
func (p *retrievalToolCycleProvider) SetBuiltinTools([]string)             {}
func (p *retrievalToolCycleProvider) WithBuiltins([]string) LLMProvider    { return p }

func (p *retrievalToolCycleProvider) capturedRequests() [][]Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]Message, len(p.requests))
	for i := range p.requests {
		out[i] = cloneMessages(p.requests[i])
	}
	return out
}

func TestThinkerReusesBoundedMemoryAcrossToolResultContinuations(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := &retrievalToolCycleProvider{called: make(chan struct{}, 16)}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Use relevant operating guidance to complete external requests directly.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	thinker.paused = true
	thinker.maxHistory = maxHistoryWorker

	relevant := []struct {
		id      string
		content string
		tags    []string
	}{
		{"skill_patreon", "PATREON_SKILL_SENTINEL\n" + strings.Repeat("Patreon publishing procedure and existing draft guidance. ", 130), []string{"skill", "patreon", "publishing"}},
		{"skill_computer", "COMPUTER_SKILL_SENTINEL\n" + strings.Repeat("Computer browser observation and action procedure. ", 130), []string{"skill", "computer", "browser"}},
	}
	for _, rec := range relevant {
		if _, err := thinker.memory.RememberWithID(rec.id, rec.content, rec.tags, 0.95); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 8; i++ {
		content := fmt.Sprintf("DISTRACTOR_%d\n", i) + strings.Repeat("Payroll tax kitchen inventory unrelated archive. ", 100)
		if _, err := thinker.memory.RememberWithID(fmt.Sprintf("skill_distractor_%d", i), content, []string{"skill", "unrelated"}, 0.9); err != nil {
			t.Fatal(err)
		}
	}

	thinker.registry.Register(&ToolDef{
		Name:        "cycle_observe",
		Description: "Return the next deterministic observation in the current workflow.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"step": map[string]any{"type": "string"}},
			"required":   []string{"step"},
		},
		Handler: func(args map[string]string) ToolResponse {
			return ToolResponse{Text: "observation step " + args["step"] + " complete\n" + strings.Repeat("state ", 3000)}
		},
	})

	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	thinker.InjectConsole("Use the Patreon and Computer browser guidance to process the existing publishing draft through several observation steps.")
	for i := 0; i < 9; i++ {
		select {
		case <-provider.called:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for provider call %d", i+1)
		}
	}

	requests := provider.capturedRequests()
	if len(requests) < 9 {
		t.Fatalf("captured requests = %d", len(requests))
	}
	for i, request := range requests[:9] {
		serialized, _ := json.Marshal(request)
		body := string(serialized)
		if strings.Count(body, "[memories — surfaced") != 1 {
			t.Fatalf("request %d did not retain exactly one memory snapshot", i+1)
		}
		for _, sentinel := range []string{"PATREON_SKILL_SENTINEL", "COMPUTER_SKILL_SENTINEL"} {
			if !strings.Contains(body, sentinel) {
				t.Fatalf("request %d lost %s", i+1, sentinel)
			}
		}
		if strings.Contains(body, "DISTRACTOR_") {
			t.Fatalf("request %d included an unrelated skill", i+1)
		}
	}

	events, _ := thinker.telemetry.StoredEvents(0)
	recalls := 0
	for _, event := range events {
		if event.Type == "memory.recall" {
			recalls++
			if !strings.Contains(string(event.Data), `"refresh_reason":"external_event"`) {
				t.Fatalf("unexpected recall telemetry: %s", event.Data)
			}
		}
		if event.Type == "llm.prompt_cache_reset" && strings.Contains(string(event.Data), "tool_result_retention_expired") {
			t.Fatalf("rolling tool-result cache reset survived: %s", event.Data)
		}
	}
	if recalls != 1 {
		t.Fatalf("automatic recalls = %d, want one for the external wake", recalls)
	}

	thinker.InjectConsole("Start a separate external follow-up using the current memory.")
	select {
	case <-provider.called:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for second external cycle")
	}
	events, _ = thinker.telemetry.StoredEvents(0)
	recalls = 0
	for _, event := range events {
		if event.Type == "memory.recall" {
			recalls++
		}
	}
	if recalls != 2 {
		t.Fatalf("recalls after a second external wake = %d, want 2", recalls)
	}

	thinker.Stop()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("thinker did not stop")
	}
}

func (p *retrievalCaptureProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	copyMessages := append([]Message(nil), messages...)
	p.mu.Lock()
	p.requests = append(p.requests, copyMessages)
	callNumber := len(p.requests)
	p.mu.Unlock()
	select {
	case p.called <- struct{}{}:
	default:
	}
	sleep := p.sleep
	if sleep == "" {
		sleep = "1ms"
		if callNumber >= 8 {
			sleep = "1h"
		}
	}
	return ChatResponse{
		Text: "Continuing standing work.",
		ToolCalls: []NativeToolCall{{
			ID: "pace-" + string(rune('0'+callNumber)), Name: "pace",
			Args: map[string]string{"sleep": sleep, "_reason": "Waiting for work"},
		}},
		Usage: TokenUsage{PromptTokens: contextChars(messages) / 4, CompletionTokens: 8},
	}, nil
}

func (p *retrievalCaptureProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "capture", ModelMedium: "capture", ModelSmall: "capture"}
}
func (p *retrievalCaptureProvider) Name() string                           { return "retrieval-capture" }
func (p *retrievalCaptureProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *retrievalCaptureProvider) SupportsNativeTools() bool              { return true }
func (p *retrievalCaptureProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *retrievalCaptureProvider) SetBuiltinTools([]string)               {}
func (p *retrievalCaptureProvider) WithBuiltins([]string) LLMProvider      { return p }

func TestSpawnedWorkerRecallUsesDirectiveAlongsideVagueParentEvent(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := &retrievalCaptureProvider{called: make(chan struct{}, 1), sleep: "1h"}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Coordinate bounded workers.",
		Mode:      ModeAutonomous,
	}
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()

	const skillID = "skill_53_0"
	const sentinel = "PATREON_ATTACHED_SKILL_SENTINEL"
	skill := "# Patreon scheduling and analytics\n\n" + sentinel + "\n" +
		"Inspect previous Friday Patreon posts, determine the correct membership tiers, open Earnings when MRR is required, edit only the named existing draft, schedule it for the requested time, and never publish it immediately. " +
		strings.Repeat("Follow the authoritative Patreon operating procedure and verify every publishing state before continuing. ", 150)
	if _, err := parent.memory.RememberWithID(skillID, skill, []string{
		"skill", "skill:patreon", "skill-id:53", "skill-source:app", "skill-hash:abc123",
	}, 0.85); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if _, err := parent.memory.RememberWithID(
			fmt.Sprintf("skill_unrelated_%d", i),
			fmt.Sprintf("UNRELATED_SKILL_%d\n", i)+strings.Repeat("CRM contacts opportunities pipelines email reporting. ", 80),
			[]string{"skill", "crm", "reporting"},
			0.85,
		); err != nil {
			t.Fatal(err)
		}
	}

	workerDirective := strings.Join([]string{
		"Inspect previous Friday Patreon posts and determine the correct tiers.",
		"Edit only draft 166563499 and schedule it without publishing immediately.",
		"Follow the named Patreon scheduling and analytics procedure from shared memory.",
	}, " ")
	if err := parent.threads.SpawnWithOpts(
		"patreon-worker", workerDirective, []string{"pace"},
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["patreon-worker"]
	if worker == nil {
		t.Fatal("spawned worker missing")
	}
	worker.Thinker.Inject("[from:main] Begin now.")
	go worker.Thinker.Run()

	select {
	case <-provider.called:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for worker model request")
	}

	provider.mu.Lock()
	requests := append([][]Message(nil), provider.requests...)
	provider.mu.Unlock()
	if len(requests) == 0 {
		t.Fatal("worker model request was not captured")
	}
	encoded, _ := json.Marshal(requests[0])
	body := string(encoded)
	if !strings.Contains(body, sentinel) {
		t.Fatalf("worker request omitted the directive-relevant attached skill: %s", body)
	}
	if strings.Contains(body, "UNRELATED_SKILL_") {
		t.Fatalf("worker request included an unrelated skill: %s", body)
	}

	events, _ := parent.telemetry.StoredEvents(0)
	foundCombinedRecall := false
	for _, event := range events {
		if event.ThreadID == "patreon-worker" && event.Type == "memory.recall" &&
			strings.Contains(string(event.Data), `"query_source":"event+directive"`) &&
			strings.Contains(string(event.Data), skillID) {
			foundCombinedRecall = true
			break
		}
	}
	if !foundCombinedRecall {
		t.Fatal("missing combined event+directive recall telemetry for attached skill")
	}
}

func TestMemoryRecallTelemetryReportsRelevantRecordSkippedBySizeLimit(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	provider := &retrievalCaptureProvider{called: make(chan struct{}, 1), sleep: "1h"}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Coordinate bounded workers.",
		Mode:      ModeAutonomous,
	}
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()

	records := []struct {
		id     string
		marker string
		size   int
		weight float64
	}{
		{id: "a_tasks", marker: "TASKS_SKILL", size: 13_500, weight: 1.00},
		{id: "b_computer", marker: "COMPUTER_SKILL", size: 9_000, weight: 0.95},
		{id: "c_patreon", marker: "PATREON_SKILL", size: 17_000, weight: 0.90},
		{id: "d_overflow", marker: "OVERFLOW_SKILL", size: 12_000, weight: 0.85},
	}
	for _, rec := range records {
		content := rec.marker + "\nAuthoritative shared alpha beta gamma operating procedure.\n" + strings.Repeat("detail ", rec.size/7)
		if _, err := parent.memory.RememberWithID(rec.id, content, []string{"shared", "procedure"}, rec.weight); err != nil {
			t.Fatalf("remember %s: %v", rec.id, err)
		}
	}

	directive := "Follow the authoritative shared alpha beta gamma operating procedures for Tasks, Computer, and Patreon."
	if err := parent.threads.SpawnWithOpts(
		"recall-budget-worker", directive, []string{"pace"},
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatalf("spawn worker: %v", err)
	}
	worker := parent.threads.threads["recall-budget-worker"]
	worker.Thinker.InjectConsole("Begin now.")
	go worker.Thinker.Run()

	select {
	case <-provider.called:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for worker model request")
	}

	events, _ := parent.telemetry.StoredEvents(0)
	for _, event := range events {
		if event.ThreadID != "recall-budget-worker" || event.Type != "memory.recall" {
			continue
		}
		var data struct {
			Accepted       int `json:"accepted"`
			Chars          int `json:"chars"`
			SkippedMatches []struct {
				ID         string `json:"id"`
				SkipReason string `json:"skip_reason"`
				Chars      int    `json:"chars"`
			} `json:"skipped_matches"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatalf("decode memory.recall telemetry: %v", err)
		}
		if data.Accepted != 3 || data.Chars <= 24*1024 || data.Chars > automaticMemoryRecallMaxChars {
			t.Fatalf("accepted=%d chars=%d limit=%d", data.Accepted, data.Chars, automaticMemoryRecallMaxChars)
		}
		if len(data.SkippedMatches) != 1 || data.SkippedMatches[0].ID != "d_overflow" ||
			data.SkippedMatches[0].SkipReason != memoryRecallSkipSizeLimit || data.SkippedMatches[0].Chars == 0 {
			t.Fatalf("skipped_matches=%+v", data.SkippedMatches)
		}
		return
	}
	t.Fatal("missing memory.recall telemetry for worker")
}

func TestThinkerRecallIsEphemeralAndReplacedAcrossManyTurns(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := &retrievalCaptureProvider{called: make(chan struct{}, 8)}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Monitor storage uploads and signed URLs.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	if _, err := thinker.memory.Remember("Use signed URLs for storage uploads.", []string{"procedure", "storage"}, 0.95); err != nil {
		t.Fatalf("remember: %v", err)
	}
	defer thinker.Stop()
	go thinker.Run()

	for i := 0; i < 8; i++ {
		select {
		case <-provider.called:
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for provider call %d", i+1)
		}
	}
	thinker.Stop()
	time.Sleep(50 * time.Millisecond)

	provider.mu.Lock()
	requests := append([][]Message(nil), provider.requests...)
	provider.mu.Unlock()
	if len(requests) < 8 {
		t.Fatalf("captured %d requests", len(requests))
	}
	for i, request := range requests[:8] {
		markers := 0
		requestContexts := 0
		foundIndex := -1
		for j, message := range request {
			markers += strings.Count(message.Content, "[memories — surfaced")
			if message.RequestContext && strings.HasPrefix(message.Content, ephemeralContextHeader) {
				requestContexts++
				foundIndex = j
			}
		}
		if markers != 1 || requestContexts != 1 {
			t.Fatalf("request %d has memory_blocks=%d request_contexts=%d, want exactly one current snapshot", i+1, markers, requestContexts)
		}
		if foundIndex < 0 || !strings.HasPrefix(request[foundIndex].Content, ephemeralContextHeader) {
			t.Fatalf("request %d is missing marked ephemeral context: %+v", i+1, request)
		}
		if !strings.Contains(request[foundIndex].Content, "[CURRENT TIME]\nUTC: ") {
			t.Fatalf("request %d lacks current time: %q", i+1, request[foundIndex].Content)
		}
		if strings.Contains(request[0].Content, "[memories — surfaced") {
			t.Fatalf("request %d rewrote memory into the system prompt: %q", i+1, request[0].Content)
		}
	}
	history, err := os.ReadFile(filepath.Join("history", "main.jsonl"))
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if strings.Contains(string(history), "[memories — surfaced") || strings.Contains(string(history), ephemeralContextHeader) {
		t.Fatalf("session history contains retrieval context: %s", history)
	}
	events, _ := thinker.telemetry.StoredEvents(0)
	foundRecall := false
	foundCacheDiagnostics := false
	for _, event := range events {
		if event.Type == "memory.recall" && strings.Contains(string(event.Data), `"ephemeral":true`) {
			foundRecall = true
		}
		if event.Type == "llm.done" {
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) == nil &&
				data.PromptCacheStablePrefixHash != "" &&
				data.PromptCacheRequestHash != "" &&
				data.PromptCacheIdentityHash != "" &&
				data.PromptCacheCommonPrefixBytes > 0 {
				foundCacheDiagnostics = true
			}
		}
	}
	if !foundRecall {
		t.Fatal("missing ephemeral memory.recall telemetry")
	}
	if !foundCacheDiagnostics {
		t.Fatal("missing append-only prompt-cache diagnostics on llm.done telemetry")
	}
}
