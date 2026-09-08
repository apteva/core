package core

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func assertDirectiveBehavior(t *testing.T, prompt, directive string) {
	t.Helper()
	if !strings.Contains(prompt, directive) {
		t.Fatal("behavior instructions missing from system prompt")
	}
	for _, legacy := range []string{"[SAFETY MODE:", "You operate independently and are trusted to act", "Soft gate —", "Before any STATE-CHANGING tool", "Decide yourself."} {
		if strings.Contains(prompt, legacy) {
			t.Errorf("core injected obsolete behavior policy %q", legacy)
		}
	}
}

// Existing server payloads may still include mode. It is now an unknown field:
// it must neither control behavior nor survive a core config round trip.
func TestLegacyBehaviorModeIgnoredOnLoad(t *testing.T) {
	for _, mode := range []string{"autonomous", "cautious", "learn", "custom"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			directive := "Ask the owner before publishing. Read local files freely."
			raw, err := json.Marshal(map[string]any{"directive": directive, "mode": mode, "execution_control": map[string]any{"mode": "paused"}})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			cfg := &Config{path: path}
			if err := cfg.load(); err != nil {
				t.Fatal(err)
			}
			if cfg.GetDirective() != directive {
				t.Fatal("legacy config lost directive")
			}
			if cfg.GetExecutionControl().Mode != ExecutionPaused {
				t.Fatal("execution control changed")
			}
			assertDirectiveBehavior(t, buildSystemPrompt(cfg.GetDirective(), nil, "", nil, nil, nil, nil), directive)
			if err := cfg.Save(); err != nil {
				t.Fatal(err)
			}
			saved, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var body map[string]any
			if err := json.Unmarshal(saved, &body); err != nil {
				t.Fatal(err)
			}
			if _, exists := body["mode"]; exists {
				t.Fatal("obsolete behavior mode persisted")
			}
		})
	}
}

func TestLegacyBehaviorModeAPIAndDirectiveUpdate(t *testing.T) {
	api, thinker := newTestAPI()
	defer thinker.telemetry.Stop()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")
	thinker.ReloadDirectiveQuiet()
	before := thinker.messages[0].Content
	for _, mode := range []string{"autonomous", "cautious", "learn"} {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(`{"mode":"`+mode+`"}`)))
		if w.Code != 200 {
			t.Fatalf("legacy update: %d %s", w.Code, w.Body.String())
		}
		if thinker.messages[0].Content != before {
			t.Fatal("legacy mode changed the prompt")
		}
	}
	directive := "# Behavior\nBefore publishing, ask for approval and wait. Reads need no approval."
	raw, err := json.Marshal(map[string]any{"directive": directive, "mode": "autonomous", "execution_control": map[string]any{"mode": "paused"}})
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(string(raw))))
	if w.Code != 200 {
		t.Fatalf("directive update: %d %s", w.Code, w.Body.String())
	}
	assertDirectiveBehavior(t, thinker.messages[0].Content, directive)
	if thinker.config.GetExecutionControl().Mode != ExecutionPaused {
		t.Fatal("execution pause config was removed")
	}
	for _, endpoint := range []string{"/config", "/status"} {
		w := httptest.NewRecorder()
		req := httptest.NewRequest("GET", endpoint, nil)
		if endpoint == "/config" {
			api.config(w, req)
		} else {
			api.status(w, req)
		}
		if w.Code != 200 {
			t.Fatalf("%s: %d", endpoint, w.Code)
		}
		var body map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if _, exists := body["mode"]; exists {
			t.Fatalf("%s exposes obsolete behavior mode", endpoint)
		}
	}
	loaded := &Config{path: thinker.config.path}
	if err := loaded.load(); err != nil {
		t.Fatal(err)
	}
	assertDirectiveBehavior(t, buildSystemPrompt(loaded.GetDirective(), nil, "", nil, nil, nil, nil), directive)
}

func TestWorkerBehaviorComesFromOwnDirective(t *testing.T) {
	t.Chdir(t.TempDir())
	parent := newTestThinker()
	defer parent.Stop()
	parent.config.path = "config.json"
	if err := parent.config.SetDirective("Main coordinates work and follows the owner's instructions."); err != nil {
		t.Fatal(err)
	}
	directive := "# Behavior\nAsk your parent before external writes, then wait for approval."
	if err := parent.threads.SpawnWithOpts("worker", directive, nil, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["worker"]
	assertDirectiveBehavior(t, worker.Thinker.messages[0].Content, directive)
	// Exercise the prompt rebuild hook used when context is refreshed.
	assertDirectiveBehavior(t, worker.Thinker.rebuildPrompt(""), directive)
	updated := "# Behavior\nRead the assigned files without asking. Report the findings to your parent."
	if err := parent.threads.Update("worker", "", updated, nil); err != nil {
		t.Fatal(err)
	}
	assertDirectiveBehavior(t, worker.Thinker.messages[0].Content, updated)
	if strings.Contains(worker.Thinker.messages[0].Content, directive) {
		t.Fatal("old worker policy remains after update")
	}
	loaded := &Config{path: "config.json"}
	if err := loaded.load(); err != nil {
		t.Fatal(err)
	}
	threads := loaded.GetThreads()
	if len(threads) != 1 || threads[0].Directive != updated {
		t.Fatal("updated worker directive was not persisted")
	}
	// Restore into a fresh manager without running a model or starting workers.
	restored := newTestThinker()
	defer restored.Stop()
	restored.config = loaded
	if err := restored.threads.SpawnWithOpts(threads[0].ID, threads[0].Directive, threads[0].Tools, SpawnOpts{DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	assertDirectiveBehavior(t, restored.threads.threads["worker"].Thinker.rebuildPrompt(""), updated)
}
