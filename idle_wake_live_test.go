package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Opt-in model check: the directive, rather than a numeric Core setting,
// determines whether the agent chooses a finite sleep or explicitly waits.
func TestCodexIdleWakeDirectiveChoices(t *testing.T) {
	token, model := requireInitiativeLive(t)
	for _, tc := range []struct {
		name      string
		policy    string
		eventOnly bool
	}{
		{"reactive", initiativePolicy(t, 0), true},
		{"conservative", initiativePolicy(t, 25), false},
		{"unspecified", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Chdir(dir)
			cfg := &Config{path: filepath.Join(dir, "config.json"), Directive: "# Role\nCoordinate the internal service. There are currently no requests, incidents, scheduled tasks, assigned recurring responsibilities, or pending tool results. No external information source is connected.\n" + tc.policy}
			audit := &initiativeAudit{}
			th := newInitiativeThinker(token, model, cfg, audit)
			done := make(chan struct{})
			go func() { defer close(done); th.Run() }()
			defer func() { th.Stop(); <-done; th.blobs.Close() }()
			if reason := waitInitiativeSettled(th, audit, 45*time.Second); reason != "" {
				t.Fatal(reason)
			}
			th.Stop()
			<-done
			rows, _, _ := audit.snapshot()
			var timing []NativeToolCall
			for _, row := range rows {
				if row.Error != "" {
					t.Fatalf("provider error: %s", row.Error)
				}
				for _, call := range row.Calls {
					if call.Name != "pace" {
						t.Fatalf("idle agent invented work: %s", call.Name)
					}
					if call.Args["sleep"] != "" || call.Args["rate"] != "" || parseTruthy(call.Args["clear_wake"]) {
						timing = append(timing, call)
					}
				}
			}
			if len(timing) == 0 {
				t.Fatal("model omitted its timing decision; runtime fallback alone is insufficient proof")
			}
			last := timing[len(timing)-1]
			if parseTruthy(last.Args["clear_wake"]) != tc.eventOnly {
				t.Fatalf("wrong directive-driven decision: %+v", last.Args)
			}
			state := cfg.GetMainPace()
			if state == nil || state.WaitForEvents != tc.eventOnly || state.NextWakeAt.IsZero() != tc.eventOnly {
				t.Fatalf("decision not reflected in saved state: %#v", state)
			}
			if !tc.eventOnly && !state.NextWakeAt.After(time.Now()) {
				t.Fatal("chosen sleep has no future deadline")
			}
			report := map[string]any{"scenario": tc.name, "model": model, "reasoning": rows[0].Reasoning, "model_calls": len(rows), "timing": timing, "saved_pace": state}
			encoded, _ := json.MarshalIndent(report, "", "  ")
			if out := os.Getenv("PROACTIVITY_REPORT_DIR"); out != "" {
				if err := os.WriteFile(filepath.Join(out, "idle-wake-"+tc.name+".json"), encoded, 0600); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("%s: model=%s reasoning=%s timing=%v saved=%+v", tc.name, model, rows[0].Reasoning, last.Args, state)
		})
	}
}
