package core

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

//	RUN_CODEX_GOAL_LIVE=1 go test -v -count=1 \
//	  -run '^TestIntegration_CodexGPT61SolLongGoalRecovery$' -timeout 9m .
//
// A longer outcome-based assignment crosses two audit waits, an unrelated
// machine event, a real state change after an apparently successful write,
// and an ambiguous publication response. No human continuation or correction
// is supplied. The fixture changes domain state, never model responses.
func TestIntegration_CodexGPT61SolLongGoalRecovery(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_GOAL_LIVE") != "1" {
		t.Skip("set RUN_CODEX_GOAL_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live goal test requested without a valid Codex credential")
	}
	for _, key := range []string{"TELEMETRY_URL", "TELEMETRY_LIVE_URL", "APTEVA_TELEMETRY_OUTBOX_DIR", "SERVER_URL", "APTEVA_MANAGED_LLM_URL", "OPENAI_CODEX_PROVIDER_ID", "APTEVA_API_KEY"} {
		t.Setenv(key, "")
	}
	t.Chdir(t.TempDir())
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()
	audit := &goalLiveAudit{maxCalls: 40}
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
	fixture := &goalReportFixture{inventory: newGoalInventoryFixture()}
	fixture.register(parent.registry)
	const workerID = "goal-report-recovery"
	const directive = "This is a one-shot assignment. Reconcile the local warehouse to amber=9 and cobalt=4 while preserving all other stock. Produce and publish an independently audited inventory report that agrees with those target quantities and the current warehouse state. Verify the published report by reading its public view. Return the published report ID, verified quantities, audit receipt, and public readback receipt to your parent. Own this goal through recoverable setbacks and waits; do not finish on an accepted write or a stale report, and avoid duplicate publication."
	if err := parent.threads.SpawnWithOpts(workerID, directive, []string{
		"inventory_snapshot", "inventory_set", "inventory_report_build", "inventory_report_audit_start",
		"inventory_report_audit_get", "inventory_report_publish", "inventory_report_public_get",
	}, SpawnOpts{Reasoning: "medium"}); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	savedWakes := map[string]PersistentPaceState{}
	informationalEvent := false
	lastProgress := ""
	for parent.threads.Count() != 0 {
		state := fixture.snapshot()
		progress := fmt.Sprintf("reports=%d failed_audits=%d passed_audits=%d published=%v readback=%v", state.ReportsBuilt, state.FailedAudits, state.PassedAudits, state.Published, !state.ReadbackAt.IsZero())
		if progress != lastProgress {
			t.Logf("progress: %s", progress)
			lastProgress = progress
		}
		if state.PendingJob != "" {
			for _, thread := range cfg.GetThreads() {
				if thread.ID == workerID && thread.Pace != nil && !thread.Pace.NextWakeAt.Before(state.ReadyAt) {
					savedWakes[state.PendingJob] = *thread.Pace
				}
			}
			// A machine notification wakes the owner early without helping it
			// solve the task or changing its goal. Its real audit timer remains.
			if !informationalEvent && len(savedWakes) > 0 && time.Now().Before(state.ReadyAt.Add(-5*time.Second)) {
				informationalEvent = parent.bus.TryPublish(Event{Type: EventInbox, To: workerID, Text: "[warehouse-status] Scanner maintenance completed. This is an informational notification; no action is required."})
			}
		}
		_, providerErr := audit.snapshot()
		if providerErr != "" {
			t.Fatalf("real model request failed: %s; state=%+v", providerErr, state)
		}
		select {
		case <-ctx.Done():
			rows, _ := audit.snapshot()
			t.Fatalf("long goal did not finish without human follow-ups: %v; state=%+v; exchanges=%+v", ctx.Err(), state, rows)
		case <-ticker.C:
		}
	}
	state := fixture.snapshot()
	stock := fixture.inventory.snapshot()
	if stock.Amber != 9 || stock.Cobalt != 4 || stock.Other != 23 || stock.Conflicts != 1 || stock.Writes != 3 {
		t.Fatalf("agent failed to restore the original goal after source drift: %+v", stock)
	}
	if state.FailedAudits != 1 || state.PassedAudits != 1 || state.ReportsBuilt != 2 || state.EarlyChecks != 0 || state.PublishCalls != 1 || !state.Published || state.ReadbackAt.IsZero() {
		t.Fatalf("agent skipped recovery/verification or duplicated publication: %+v", state)
	}
	if !informationalEvent || len(savedWakes) != 2 {
		t.Fatalf("expected an early event and two durable audit waits: event=%v wakes=%v", informationalEvent, savedWakes)
	}
	rows, providerErr := audit.snapshot()
	if providerErr != "" {
		t.Fatal(providerErr)
	}
	var timerWakes, eventWakes, doneCalls, sendCalls int
	var final string
	for i, row := range rows {
		if strings.Contains(row.Wake, "reason: timer") {
			timerWakes++
		}
		if strings.Contains(row.Wake, "reason: event") && row.SawInformationalEvent {
			eventWakes++
		}
		var names []string
		for _, call := range row.Calls {
			names = append(names, call.Name)
			switch call.Name {
			case "done":
				doneCalls++
				final = call.Args["message"]
				if row.StartedAt.Before(state.ReadbackAt) {
					t.Fatal("agent declared completion before observing the published report")
				}
			case "send":
				sendCalls++
			}
		}
		t.Logf("turn=%d tools=%v", i+1, names)
	}
	if timerWakes < 2 || eventWakes < 1 || doneCalls != 1 || sendCalls != 0 {
		t.Fatalf("wrong autonomous lifecycle: timer=%d informational=%d done=%d send=%d", timerWakes, eventWakes, doneCalls, sendCalls)
	}
	for _, evidence := range []string{state.ReportID, state.AuditReceipt, state.ReadbackReceipt, "amber", "cobalt", "9", "4"} {
		if evidence == "" || !strings.Contains(final, evidence) {
			t.Fatalf("final result omitted verified evidence %q: %q", evidence, final)
		}
	}
	var delivered int
	for _, event := range parent.drainEventTexts() {
		if strings.Contains(event, "[thread:"+workerID+" done]") {
			delivered++
			if !strings.Contains(event, state.ReadbackReceipt) {
				t.Fatalf("parent did not receive the verified public outcome: %q", event)
			}
		}
	}
	if delivered != 1 {
		t.Fatalf("parent received %d final results, want one", delivered)
	}
	t.Logf("model=%s achieved long goal in %s with %d real calls: revision conflict, two timed audits, early informational event, failed verification/source drift, repair/rebuild, ambiguous publication recovery, public readback, one final result", toolReasonLiveModel, time.Since(started).Round(time.Millisecond), len(rows))
}

type goalReportState struct {
	ReportID, PendingJob, AuditReceipt, ReadbackReceipt string
	ReportRevision                                      int
	ReportAmber, ReportCobalt, ReportOther              int
	ReportsBuilt, FailedAudits, PassedAudits            int
	EarlyChecks, PublishCalls                           int
	ReadyAt, ReadbackAt                                 time.Time
	Published                                           bool
}

type goalReportFixture struct {
	mu        sync.Mutex
	inventory *goalInventoryFixture
	state     goalReportState
}

func (f *goalReportFixture) snapshot() goalReportState {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.state
}

func (f *goalReportFixture) register(registry *ToolRegistry) {
	f.inventory.register(registry)
	for _, tool := range []struct {
		name, description string
		fields            []string
	}{
		{"inventory_report_build", "Build a report from the current warehouse quantities and revision. Returns report_id and source_revision. Building is not auditing or publishing.", nil},
		{"inventory_report_audit_start", "Start an independent audit of report_id against the current warehouse and required target quantities. Returns job_id and ready_at UTC deadline. This pull-only job sends no completion event; do not check it before ready_at.", []string{"report_id"}},
		{"inventory_report_audit_get", "Read an audit by job_id at or after ready_at. A failed audit includes actual evidence; a passed audit returns an audit_receipt bound to this report and source revision. Failed or stale reports are not publishable.", []string{"job_id"}},
		{"inventory_report_publish", "Publish report_id with its audit_receipt. Only a current, passed report may be published. If the response indicates an unknown outcome, inspect inventory_report_public_get before issuing another write; an error response does not establish that publication failed.", []string{"report_id", "audit_receipt"}},
		{"inventory_report_public_get", "Read the latest publicly published report, its actual stock quantities, audit receipt, and a public readback receipt. This read is the authoritative way to resolve an uncertain publication outcome and verify what was published.", nil},
	} {
		properties := map[string]any{}
		for _, field := range tool.fields {
			properties[field] = map[string]any{"type": "string"}
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

func (f *goalReportFixture) call(name string, args map[string]string) ToolResponse {
	f.mu.Lock()
	defer f.mu.Unlock()
	s := &f.state
	result := func(value any, failed bool) ToolResponse {
		encoded, _ := json.Marshal(value)
		return ToolResponse{Text: string(encoded), IsError: failed}
	}
	fail := func(message string) ToolResponse { return result(map[string]any{"error": message}, true) }
	stock := f.inventory.snapshot()
	current := func() bool {
		return s.ReportRevision == stock.Revision && s.ReportAmber == stock.Amber && s.ReportCobalt == stock.Cobalt && s.ReportOther == stock.Other && stock.Amber == 9 && stock.Cobalt == 4 && stock.Other == 23
	}
	switch name {
	case "inventory_report_build":
		if s.PendingJob != "" || s.Published {
			return fail("an audit is still pending or a report is already published")
		}
		s.ReportsBuilt++
		s.ReportID = "report-" + rand.Text()
		s.ReportRevision = stock.Revision
		s.ReportAmber, s.ReportCobalt, s.ReportOther = stock.Amber, stock.Cobalt, stock.Other
		s.AuditReceipt = ""
		return result(map[string]any{"report_id": s.ReportID, "source_revision": s.ReportRevision, "stock": map[string]int{"amber": s.ReportAmber, "cobalt": s.ReportCobalt, "other": s.ReportOther}}, false)
	case "inventory_report_audit_start":
		if s.ReportID == "" || args["report_id"] != s.ReportID || s.PendingJob != "" {
			return fail("unknown report or an audit is already pending")
		}
		s.PendingJob = "job-" + rand.Text()
		s.ReadyAt = time.Now().UTC().Add(20 * time.Second)
		return result(map[string]any{"job_id": s.PendingJob, "status": "pending", "ready_at": s.ReadyAt.Format(time.RFC3339Nano)}, false)
	case "inventory_report_audit_get":
		if s.PendingJob == "" || args["job_id"] != s.PendingJob {
			return fail("unknown or already consumed audit job")
		}
		if time.Now().Before(s.ReadyAt) {
			s.EarlyChecks++
			return result(map[string]any{"status": "pending", "ready_at": s.ReadyAt.Format(time.RFC3339Nano)}, false)
		}
		s.PendingJob = ""
		if s.FailedAudits == 0 {
			// An actual concurrent stock adjustment invalidates the report.
			// The goal remains unchanged; the agent must restore cobalt=4.
			f.inventory.mu.Lock()
			f.inventory.state.Cobalt = 3
			f.inventory.state.Revision++
			f.inventory.mu.Unlock()
			s.FailedAudits++
			return result(map[string]any{"status": "failed", "report_id": s.ReportID, "issue": "source_changed", "evidence": "A concurrent adjustment reduced warehouse cobalt to 3 after the report was built. The report no longer matches the warehouse, and the warehouse does not satisfy the required target cobalt=4. No audit receipt issued."}, false)
		}
		if !current() {
			s.FailedAudits++
			return result(map[string]any{"status": "failed", "issue": "report and warehouse do not agree with target quantities", "warehouse": stock}, false)
		}
		s.PassedAudits++
		s.AuditReceipt = "audit-ok-" + rand.Text()
		return result(map[string]any{"status": "passed", "report_id": s.ReportID, "audit_receipt": s.AuditReceipt, "source_revision": s.ReportRevision, "stock": map[string]int{"amber": stock.Amber, "cobalt": stock.Cobalt, "other": stock.Other}}, false)
	case "inventory_report_publish":
		s.PublishCalls++
		if !current() || s.AuditReceipt == "" || args["report_id"] != s.ReportID || args["audit_receipt"] != s.AuditReceipt {
			return fail("publication rejected: report is stale, unverified, or has an invalid audit receipt")
		}
		if s.Published {
			return result(map[string]any{"already_published": true, "report_id": s.ReportID}, false)
		}
		s.Published = true
		return result(map[string]any{"error": "connection_lost_after_submission", "outcome": "unknown", "report_id": s.ReportID}, true)
	case "inventory_report_public_get":
		if !s.Published {
			return result(map[string]any{"published": false}, false)
		}
		if s.ReadbackAt.IsZero() {
			s.ReadbackAt = time.Now()
			s.ReadbackReceipt = "public-read-" + rand.Text()
		}
		return result(map[string]any{"published": true, "report_id": s.ReportID, "audit_receipt": s.AuditReceipt, "readback_receipt": s.ReadbackReceipt, "stock": map[string]int{"amber": s.ReportAmber, "cobalt": s.ReportCobalt, "other": s.ReportOther}}, false)
	default:
		return fail(fmt.Sprintf("unknown report operation %q", name))
	}
}
