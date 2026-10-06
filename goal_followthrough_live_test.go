package core

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

//	RUN_CODEX_GOAL_LIVE=1 go test -v -count=1 \
//	  -run '^TestIntegration_CodexGPT61SolGoalFollowthrough$' -timeout 5m .
//
// One outcome-based directive drives the real worker loop. The test never
// sends a follow-up, advances a clock, supplies model responses, or adds a goal
// evaluator. Domain tools are local fixtures; all decisions use real Sol 6.1.
func TestIntegration_CodexGPT61SolGoalFollowthrough(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_GOAL_LIVE") != "1" {
		t.Skip("set RUN_CODEX_GOAL_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live goal test requested without a valid Codex credential")
	}
	// Environment loading above can include server routing. Keep execution,
	// tools, sessions, and telemetry isolated from the user's running agents.
	for _, key := range []string{"TELEMETRY_URL", "TELEMETRY_LIVE_URL", "APTEVA_TELEMETRY_OUTBOX_DIR", "SERVER_URL", "APTEVA_MANAGED_LLM_URL", "OPENAI_CODEX_PROVIDER_ID", "APTEVA_API_KEY"} {
		t.Setenv(key, "")
	}
	t.Chdir(t.TempDir())
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	audit := &goalLiveAudit{}
	provider := &goalLiveProvider{LLMProvider: toolReasonLiveProvider(t, token), ctx: ctx, audit: audit}
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Coordinate worker ownership."}
	parent := NewThinker("", provider, cfg)
	parent.pool = &ProviderPool{providers: map[string]LLMProvider{provider.Name(): provider}, order: []string{provider.Name()}, default_: provider.Name()}
	t.Cleanup(func() {
		cancel()
		parent.threads.KillAll()
		parent.Stop()
		parent.telemetry.Stop()
		parent.blobs.Close()
	})
	fixture := newGoalInventoryFixture()
	fixture.register(parent.registry)
	const workerID = "goal-inventory"
	const directive = "This is a one-shot assignment. Reconcile the local warehouse inventory so amber has 9 units and cobalt has 4 units; leave all other stock unchanged. Obtain a completed independent inventory audit proving those quantities, and return the verified quantities and its exact receipt to your parent. Own the assignment through locally recoverable failures and any pending audit; finish only when its outcome is verified."
	if err := parent.threads.SpawnWithOpts(workerID, directive,
		[]string{"inventory_snapshot", "inventory_set", "inventory_audit_start", "inventory_audit_get"},
		SpawnOpts{Reasoning: "medium"}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	var savedPace *PersistentPaceState
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for parent.threads.Count() != 0 {
		state := fixture.snapshot()
		if !state.ReadyAt.IsZero() && !state.Verified {
			for _, thread := range cfg.GetThreads() {
				if thread.ID == workerID && thread.Pace != nil && !thread.Pace.NextWakeAt.Before(state.ReadyAt) {
					savedPace = clonePersistentPaceState(thread.Pace)
				}
			}
		}
		_, providerErr := audit.snapshot()
		if providerErr != "" {
			t.Fatalf("real model request failed: %s; fixture=%+v", providerErr, state)
		}
		select {
		case <-ctx.Done():
			rows, _ := audit.snapshot()
			t.Fatalf("worker did not achieve its goal without follow-ups: %v; fixture=%+v; exchanges=%+v", ctx.Err(), state, rows)
		case <-ticker.C:
		}
	}
	state := fixture.snapshot()
	if state.Amber != 9 || state.Cobalt != 4 || state.Other != 23 || !state.Verified {
		t.Fatalf("worker finished before achieving the actual goal: %+v", state)
	}
	if state.Conflicts != 1 || state.Snapshots < 2 || state.Writes != 2 || state.Audits != 1 || state.EarlyChecks != 0 || state.Verifications != 1 {
		t.Fatalf("worker failed recovery/verification or repeated work: %+v", state)
	}
	if savedPace == nil || savedPace.WaitForEvents || savedPace.Sleep == "" {
		t.Fatalf("pending audit did not produce a durable model-selected timer: %+v", savedPace)
	}
	rows, providerErr := audit.snapshot()
	if providerErr != "" {
		t.Fatal(providerErr)
	}
	var timedPace, timerWake bool
	var doneCalls, sendCalls int
	var final string
	for _, row := range rows {
		if !row.StartedAt.Before(state.ReadyAt) && strings.Contains(row.Wake, "reason: timer") {
			timerWake = true
		}
		for _, call := range row.Calls {
			switch call.Name {
			case "pace":
				if !row.CompletedAt.Before(state.AuditStartedAt) && row.CompletedAt.Before(state.VerifiedAt) && (call.Args["sleep"] != "" || call.Args["rate"] != "") {
					timedPace = true
				}
			case "done":
				doneCalls++
				final = call.Args["message"]
				if row.StartedAt.Before(state.VerifiedAt) {
					t.Fatal("model declared completion before consuming the independent audit")
				}
			case "send":
				sendCalls++
			}
		}
	}
	if !timedPace || !timerWake {
		t.Fatalf("no proof of autonomous timer continuation: model_pace=%v timer_wake=%v", timedPace, timerWake)
	}
	if doneCalls != 1 || sendCalls != 0 || !strings.Contains(final, state.Receipt) || !strings.Contains(final, "amber") || !strings.Contains(final, "cobalt") || !strings.Contains(final, "9") || !strings.Contains(final, "4") {
		t.Fatalf("expected one verified final result: done=%d send=%d final=%q", doneCalls, sendCalls, final)
	}
	var delivered int
	for _, event := range parent.drainEventTexts() {
		if strings.Contains(event, "[thread:"+workerID+" done]") {
			delivered++
			if !strings.Contains(event, state.Receipt) {
				t.Fatalf("parent received a final result without audit evidence: %q", event)
			}
		}
	}
	if delivered != 1 {
		t.Fatalf("parent received %d final results, want one", delivered)
	}
	t.Logf("model=%s achieved one directive without follow-ups in %s: %d real model calls, recovered revision conflict, two writes, persisted pace=%s, timer wake, independent audit, one final receipt", toolReasonLiveModel, time.Since(started).Round(time.Millisecond), len(rows), savedPace.Sleep)
}

type goalLiveExchange struct {
	StartedAt, CompletedAt time.Time
	Wake                   string
	Calls                  []NativeToolCall
	SawInformationalEvent  bool
}

type goalLiveAudit struct {
	mu       sync.Mutex
	started  int
	maxCalls int
	rows     []goalLiveExchange
	err      string
}

func (a *goalLiveAudit) snapshot() ([]goalLiveExchange, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]goalLiveExchange(nil), a.rows...), a.err
}

type goalLiveProvider struct {
	LLMProvider
	ctx   context.Context
	audit *goalLiveAudit
}

func (p *goalLiveProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	return &goalLiveProvider{LLMProvider: providerWithReasoning(p.LLMProvider, settings.Level), ctx: p.ctx, audit: p.audit}
}

func (p *goalLiveProvider) WithBuiltins(names []string) LLMProvider {
	return &goalLiveProvider{LLMProvider: p.LLMProvider.WithBuiltins(names), ctx: p.ctx, audit: p.audit}
}

func (p *goalLiveProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, chunk, thinking func(string), toolChunk func(string, string, string)) (ChatResponse, error) {
	p.audit.mu.Lock()
	p.audit.started++
	limit := p.audit.maxCalls
	if limit == 0 {
		limit = 16
	}
	if p.audit.started > limit {
		p.audit.err = fmt.Sprintf("exceeded %d real model calls without finishing", limit)
		p.audit.mu.Unlock()
		return ChatResponse{}, fmt.Errorf("live goal inference limit exceeded")
	}
	p.audit.mu.Unlock()
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	stop := context.AfterFunc(p.ctx, cancel)
	defer stop()
	defer cancel()
	row := goalLiveExchange{StartedAt: time.Now()}
	for _, msg := range messages {
		if strings.Contains(msg.TextContent(), "[WAKE STATE]") {
			row.Wake = msg.TextContent()
		}
		if strings.Contains(msg.TextContent(), "Scanner maintenance completed") {
			row.SawInformationalEvent = true
		}
	}
	response, err := p.LLMProvider.Chat(ctx, messages, model, tools, chunk, thinking, toolChunk)
	row.CompletedAt, row.Calls = time.Now(), response.ToolCalls
	p.audit.mu.Lock()
	p.audit.rows = append(p.audit.rows, row)
	if err != nil {
		p.audit.err = err.Error()
	}
	p.audit.mu.Unlock()
	return response, err
}

type goalInventoryState struct {
	Amber, Cobalt, Other, Revision                                   int
	Snapshots, Conflicts, Writes, Audits, EarlyChecks, Verifications int
	AuditStartedAt, ReadyAt, VerifiedAt                              time.Time
	Verified                                                         bool
	Job, Receipt                                                     string
}

type goalInventoryFixture struct {
	mu    sync.Mutex
	state goalInventoryState
}

func newGoalInventoryFixture() *goalInventoryFixture {
	return &goalInventoryFixture{state: goalInventoryState{Amber: 7, Cobalt: 11, Other: 23, Revision: 1}}
}

func (f *goalInventoryFixture) snapshot() goalInventoryState {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.state
}

func (f *goalInventoryFixture) register(registry *ToolRegistry) {
	for _, tool := range []struct {
		name, description string
		fields            []string
	}{
		{"inventory_snapshot", "Read warehouse stock quantities and the current revision required by inventory_set.", nil},
		{"inventory_set", "Set one stock quantity using the revision from inventory_snapshot. A stale revision returns a conflict with no write; read a fresh snapshot to recover. Every accepted write advances the revision.", []string{"sku", "quantity", "revision"}},
		{"inventory_audit_start", "Start one independent audit of the current stock. Returns its job_id, pending status, and ready_at UTC deadline. The audit is pull-only and will not send an event. Its result cannot be checked before ready_at.", nil},
		{"inventory_audit_get", "Read an existing audit by job_id at or after its ready_at deadline. Only a completed audit proves the inventory outcome and returns its final receipt.", []string{"job_id"}},
	} {
		properties := map[string]any{}
		for _, field := range tool.fields {
			kind := "string"
			if field == "quantity" || field == "revision" {
				kind = "integer"
			}
			properties[field] = map[string]any{"type": kind}
		}
		schema := map[string]any{"type": "object", "properties": properties}
		if len(tool.fields) > 0 {
			schema["required"] = tool.fields
		}
		registry.Register(&ToolDef{Name: tool.name, Description: tool.description, InputSchema: schema, Handler: func(args map[string]string) ToolResponse {
			return f.call(tool.name, args)
		}})
	}
}

func (f *goalInventoryFixture) call(name string, args map[string]string) ToolResponse {
	f.mu.Lock()
	defer f.mu.Unlock()
	s := &f.state
	result := func(value any, failed bool) ToolResponse {
		encoded, _ := json.Marshal(value)
		return ToolResponse{Text: string(encoded), IsError: failed}
	}
	fail := func(message string) ToolResponse { return result(map[string]any{"error": message}, true) }
	switch name {
	case "inventory_snapshot":
		s.Snapshots++
		return result(map[string]any{"revision": s.Revision, "stock": map[string]int{"amber": s.Amber, "cobalt": s.Cobalt, "other": s.Other}}, false)
	case "inventory_set":
		revision, err := strconv.Atoi(args["revision"])
		quantity, quantityErr := strconv.Atoi(args["quantity"])
		if err != nil || quantityErr != nil || quantity < 0 || (args["sku"] != "amber" && args["sku"] != "cobalt") {
			return fail("invalid stock update")
		}
		// Simulate a concurrent revision bump once, without changing stock.
		if s.Conflicts == 0 {
			s.Revision++
			s.Conflicts++
			return fail("revision conflict; no write applied; reload the warehouse snapshot")
		}
		if revision != s.Revision {
			return fail("revision conflict; no write applied; reload the warehouse snapshot")
		}
		if s.Audits > 0 {
			return fail("stock is frozen for the independent audit")
		}
		if args["sku"] == "amber" {
			s.Amber = quantity
		} else {
			s.Cobalt = quantity
		}
		s.Revision++
		s.Writes++
		return result(map[string]any{"accepted": true, "revision": s.Revision}, false)
	case "inventory_audit_start":
		if s.Audits > 0 {
			return fail("audit already exists; use the existing job_id")
		}
		if s.Amber != 9 || s.Cobalt != 4 || s.Other != 23 {
			return fail("inventory is not reconciled")
		}
		s.Audits++
		s.AuditStartedAt = time.Now().UTC()
		s.ReadyAt = s.AuditStartedAt.Add(20 * time.Second)
		s.Job = fmt.Sprintf("audit-%x", rand.Text())
		return result(map[string]any{"job_id": s.Job, "status": "pending", "ready_at": s.ReadyAt.Format(time.RFC3339Nano)}, false)
	case "inventory_audit_get":
		if s.Job == "" || args["job_id"] != s.Job {
			return fail("unknown audit job")
		}
		if time.Now().Before(s.ReadyAt) {
			s.EarlyChecks++
			return result(map[string]any{"status": "pending", "ready_at": s.ReadyAt.Format(time.RFC3339Nano)}, false)
		}
		s.Verifications++
		s.Verified = true
		s.VerifiedAt = time.Now()
		if s.Receipt == "" {
			s.Receipt = "verified-" + rand.Text()
		}
		return result(map[string]any{"status": "completed", "receipt": s.Receipt, "stock": map[string]int{"amber": s.Amber, "cobalt": s.Cobalt, "other": s.Other}}, false)
	default:
		return fail("unknown inventory operation")
	}
}
