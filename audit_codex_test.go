package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type auditCountingProvider struct {
	LLMProvider
	active, peak, calls atomic.Int32
}

func stopAuditThinker(t *testing.T, thinker *Thinker) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := thinker.Shutdown(ctx); err != nil {
		t.Errorf("shutdown: %v", err)
	}
}

func (p *auditCountingProvider) Chat(ctx context.Context, m []Message, model string, tools []NativeTool, chunk func(string), thinking func(string), toolChunk func(string, string, string)) (ChatResponse, error) {
	p.calls.Add(1)
	n := p.active.Add(1)
	defer p.active.Add(-1)
	for old := p.peak.Load(); n > old; old = p.peak.Load() {
		if p.peak.CompareAndSwap(old, n) {
			break
		}
	}
	return p.LLMProvider.Chat(ctx, m, model, tools, chunk, thinking, toolChunk)
}

// Explicitly opt in: consumes the user's saved Codex allowance. All tools are
// local fixtures, with no external side effects. The serial control measures
// the same three assignments; latency is reported, never asserted as an SLA.
func TestAuditCodexParallelWorkersAndMain(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_AUDIT") != "1" {
		t.Skip("set RUN_CODEX_AUDIT=1 for real Codex fan-out/fan-in")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live audit requested but Codex authentication unavailable")
	}
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprintf("parallel_%v", parallel), func(t *testing.T) {
			t.Chdir(t.TempDir())
			base := &fixedModelProvider{LLMProvider: NewOpenAICodexProvider(token), model: "gpt-5.6-terra"}
			p := &auditCountingProvider{LLMProvider: base}
			cfg := &Config{path: "config.json", Directive: "Three workers return authoritative markers via done. Once all three worker completions are present, call audit_report exactly once with result containing all three exact markers. Do not spawn, send, or call audit_lookup. After audit_report succeeds, call pace with clear_wake=true and wait.", Mode: ModeAutonomous}
			if err := cfg.Save(); err != nil {
				t.Fatal(err)
			}
			main := NewThinker("", p, cfg)
			defer stopAuditThinker(t, main)
			var mu sync.Mutex
			counts := map[string]int{}
			reported := make(chan string, 4)
			main.registry.Register(&ToolDef{Name: "audit_lookup", Description: "Return the authoritative marker for a worker key.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"key": map[string]any{"type": "string"}}, "required": []string{"key"}}, HandlerContext: func(ctx context.Context, args map[string]string) ToolResponse {
				key := args["key"]
				mu.Lock()
				counts[key]++
				mu.Unlock()
				select {
				case <-time.After(200 * time.Millisecond):
				case <-ctx.Done():
					return ToolResponse{Text: ctx.Err().Error(), IsError: true}
				}
				return ToolResponse{Text: "marker=CODEX-" + strings.ToUpper(key) + "-917"}
			}})
			main.registry.Register(&ToolDef{Name: "audit_report", Description: "Submit the combined final markers.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"result": map[string]any{"type": "string"}}, "required": []string{"result"}}, Handler: func(args map[string]string) ToolResponse {
				reported <- args["result"]
				return ToolResponse{Text: "accepted"}
			}})
			started := time.Now()
			waitWorkers := func() {
				t.Helper()
				deadline := time.Now().Add(2 * time.Minute)
				diagnose := func() {
					t.Logf("unfinished workers: %+v; model calls=%d active=%d", main.threads.List(), p.calls.Load(), p.active.Load())
					events, _ := main.telemetry.StoredEvents(0)
					for _, event := range events {
						if event.Type == "llm.error" || event.Type == "tool.result" || event.Type == "llm.done" {
							t.Logf("worker diagnostic %s %s: %s", event.ThreadID, event.Type, event.Data)
						}
					}
				}
				diagnosed := false
				for (main.threads.Count() != 0 || len(cfg.GetThreads()) != 0) && time.Now().Before(deadline) {
					if !diagnosed && time.Until(deadline) < 105*time.Second {
						diagnose()
						diagnosed = true
					}
					time.Sleep(20 * time.Millisecond)
				}
				if main.threads.Count() != 0 {
					diagnose()
					t.Fatal("workers did not complete")
				}
			}
			for _, key := range []string{"alpha", "beta", "gamma"} {
				if err := main.threads.SpawnWithOpts(key, "Call audit_lookup exactly once with key "+key+". Await the result. Then call done alone with message equal to the exact returned marker value. Do not send or pace.", []string{"audit_lookup"}, SpawnOpts{}); err != nil {
					t.Fatal(err)
				}
				if !parallel {
					waitWorkers()
				}
			}
			waitWorkers()
			workersElapsed := time.Since(started)
			workerCalls, peak := p.calls.Load(), p.peak.Load()
			mu.Lock()
			for _, key := range []string{"alpha", "beta", "gamma"} {
				if counts[key] != 1 {
					t.Errorf("%s lookups=%d want 1", key, counts[key])
				}
			}
			mu.Unlock()
			if parallel && peak < 2 {
				t.Errorf("parallel requests never overlapped: peak=%d", peak)
			}
			pending := cfg.getMainEvents()
			if len(pending) != 3 {
				t.Fatalf("durable completions=%d want 3", len(pending))
			}
			// Reconstruct the parent before it consumes completion: exercises crash
			// recovery, then real Codex reads those messages and joins the results.
			stopAuditThinker(t, main)
			restored := NewConfig()
			if err := restored.LoadError(); err != nil {
				t.Fatal(err)
			}
			parent := NewThinker("", p, restored)
			defer stopAuditThinker(t, parent)
			parent.registry.Register(main.registry.Get("audit_report"))
			go parent.Run()
			select {
			case result := <-reported:
				for _, key := range []string{"ALPHA", "BETA", "GAMMA"} {
					if !strings.Contains(result, "CODEX-"+key+"-917") {
						t.Errorf("report missing %s: %s", key, result)
					}
				}
			case <-time.After(2 * time.Minute):
				t.Fatal("main did not aggregate recovered completions")
			}
			select {
			case duplicate := <-reported:
				t.Errorf("duplicate report: %s", duplicate)
			case <-time.After(300 * time.Millisecond):
			}
			data, _ := json.Marshal(map[string]any{"parallel": parallel, "worker_ms": workersElapsed.Milliseconds(), "total_ms": time.Since(started).Milliseconds(), "worker_llm_calls": workerCalls, "peak_llm_requests": peak, "model": "gpt-5.6-terra", "workers": 3, "lookups": counts, "durable_completions": len(pending)})
			t.Logf("LIVE_CODEX_METRICS %s", data)
		})
	}
}
