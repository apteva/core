package core

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// recordingPaceProvider keeps the real provider behavior while making the
// exact wake-state context and model-selected tool arguments assertable.
type recordingPaceProvider struct {
	LLMProvider
	mu        sync.Mutex
	requests  [][]Message
	responses []ChatResponse
}

func (p *recordingPaceProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	response, err := p.LLMProvider.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
	p.mu.Lock()
	p.requests = append(p.requests, cloneMessages(messages))
	p.responses = append(p.responses, response)
	p.mu.Unlock()
	return response, err
}

func (p *recordingPaceProvider) firstExchange() ([]Message, ChatResponse) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.requests) == 0 || len(p.responses) == 0 {
		return nil, ChatResponse{}
	}
	return cloneMessages(p.requests[0]), p.responses[0]
}

// TestCodexEarlyEventPreservesPlannedWakeSmoke verifies the behavioral half
// of agent-owned pacing against real Codex. Core's deterministic tests verify
// the state machine; this test verifies the prompt leads the model to preserve
// an already-correct deadline while handling an unrelated early event.
func TestCodexEarlyEventPreservesPlannedWakeSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_PACE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_PACE_SMOKE=1 to run the Codex pace smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex pace smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runEarlyEventPreservesPlannedWakeSmoke(t, NewOpenAICodexProvider(token))
}

func runEarlyEventPreservesPlannedWakeSmoke(t *testing.T, liveProvider LLMProvider) {
	t.Helper()
	t.Chdir(t.TempDir())
	pendingWake := time.Now().Add(90 * time.Minute).UTC()
	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Own a small set of closely related CRM follow-up responsibilities.",
			"",
			"# Planned Work",
			"- Review overdue follow-ups during the hourly CRM cycle.",
			"- Check stale opportunity stages during that same cycle.",
			"- Include notable exceptions in the next daily summary.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{Sleep: "90m", NextWakeAt: pendingWake}); err != nil {
		t.Fatal(err)
	}
	provider := &recordingPaceProvider{LLMProvider: liveProvider}
	label := livePaceProviderLabel(liveProvider)
	thinker := NewThinker("", provider, cfg)
	observer := thinker.bus.SubscribeAll("live-early-pace-observer", 64)
	defer thinker.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Errorf("%s thinker did not stop", label)
		}
	}()

	thinker.InjectConsole(strings.Join([]string{
		"An unrelated approval dismissal arrived.",
		"Record that no action is needed. The existing pending automatic wake remains correct for all planned CRM work.",
		"Preserve it and wait; do not change the directive or work ownership.",
	}, " "))
	waitForLivePaceTurn(t, observer, runDone, "main")

	request, response := provider.firstExchange()
	requestText := messagesText(request)
	for _, want := range []string{
		"[WAKE STATE]",
		"reason: event",
		"pending_wake_at: " + pendingWake.Format(time.RFC3339Nano),
	} {
		if !strings.Contains(requestText, want) {
			t.Fatalf("%s early-event request missing %q:\n%s", label, want, requestText)
		}
	}
	for _, call := range response.ToolCalls {
		switch call.Name {
		case "pace":
			if strings.TrimSpace(call.Args["sleep"]) != "" ||
				strings.TrimSpace(call.Args["rate"]) != "" ||
				parseTruthy(call.Args["clear_wake"]) {
				t.Fatalf("%s moved an explicitly unchanged wake: args=%v", label, call.Args)
			}
		case "spawn", "evolve", "update":
			t.Fatalf("%s changed ownership/directive for an unrelated event: call=%s args=%v", label, call.Name, call.Args)
		}
	}
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("early event moved persisted wake: got=%#v want=%s", state, pendingWake.Format(time.RFC3339Nano))
	}
	if !thinker.status().NextWakeAt.Equal(pendingWake) {
		t.Fatalf("early event moved runtime wake: got=%s want=%s",
			thinker.status().NextWakeAt.Format(time.RFC3339Nano), pendingWake.Format(time.RFC3339Nano))
	}
	t.Logf("%s preserved %s while handling an unrelated event; calls=%v", label, pendingWake.Format(time.RFC3339Nano), response.ToolCalls)
}

// TestCodexRecurringOwnerReplansSeveralResponsibilitiesSmoke verifies that a
// persistent owner sees a consumed timer, considers several related planned
// responsibilities, and explicitly chooses its next wake instead of relying
// on a Core-generated recurrence.
func TestCodexRecurringOwnerReplansSeveralResponsibilitiesSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_PACE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_PACE_SMOKE=1 to run the Codex pace smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex pace smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runRecurringOwnerReplansSeveralResponsibilitiesSmoke(t, NewOpenAICodexProvider(token))
}

func runRecurringOwnerReplansSeveralResponsibilitiesSmoke(t *testing.T, liveProvider LLMProvider) {
	t.Helper()
	t.Chdir(t.TempDir())
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nGovern work ownership.",
		Mode:      ModeAutonomous,
	}
	provider := &recordingPaceProvider{LLMProvider: liveProvider}
	label := livePaceProviderLabel(liveProvider)
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	dueWake := time.Now().Add(100 * time.Millisecond).UTC()
	if err := parent.threads.SpawnWithOpts(
		"crm-cycle-owner",
		strings.Join([]string{
			"# Role",
			"Own the related CRM hygiene cycle.",
			"",
			"# Planned Work",
			"- Review overdue follow-ups.",
			"- Review stale opportunity stages.",
			"- Prepare notable exceptions for the daily summary.",
			"",
			"# Timing",
			"When a timer wakes you, assess all planned work against the current time and history.",
			"This smoke fixture provides no domain tools, so do not claim the checks ran. Before waiting, explicitly set another automatic wake for one hour.",
			"Do not send a routine heartbeat to your parent.",
		}, "\n"),
		nil,
		SpawnOpts{
			DeferRun: true,
			Pace:     &PersistentPaceState{Sleep: "1h", NextWakeAt: dueWake},
		},
	); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["crm-cycle-owner"].Thinker
	observer := parent.bus.SubscribeAll("live-recurring-pace-observer", 64)
	defer parent.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		worker.Run()
	}()
	defer func() {
		worker.Stop()
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Errorf("%s recurring worker did not stop", label)
		}
	}()

	waitForLivePaceTurn(t, observer, runDone, "crm-cycle-owner")
	request, response := provider.firstExchange()
	requestText := messagesText(request)
	for _, want := range []string{
		"[WAKE STATE]",
		"reason: timer",
		"pending_wake_at: none (timer fired)",
		"Review overdue follow-ups",
		"Review stale opportunity stages",
		"daily summary",
	} {
		if !strings.Contains(requestText, want) {
			t.Fatalf("%s recurring request missing %q:\n%s", label, want, requestText)
		}
	}

	var timingCall *NativeToolCall
	for i := range response.ToolCalls {
		call := &response.ToolCalls[i]
		switch call.Name {
		case "pace":
			if strings.TrimSpace(call.Args["sleep"]) != "" || strings.TrimSpace(call.Args["rate"]) != "" {
				timingCall = call
			}
		case "spawn", "evolve", "send", "done":
			t.Fatalf("recurring owner performed unrelated coordination: call=%s args=%v", call.Name, call.Args)
		}
	}
	if timingCall == nil {
		t.Fatalf("%s did not explicitly choose another wake after the timer: text=%q calls=%v", label, response.Text, response.ToolCalls)
	}
	nextWake := worker.status().NextWakeAt
	remaining := time.Until(nextWake)
	if remaining < 50*time.Minute || remaining > 70*time.Minute {
		t.Fatalf("Codex next wake remaining=%s at=%s, want approximately one hour; pace=%v",
			remaining, nextWake.Format(time.RFC3339Nano), timingCall.Args)
	}
	state, err := parent.threads.PersistentState("crm-cycle-owner")
	if err != nil {
		t.Fatal(err)
	}
	if state.Pace == nil || !state.Pace.NextWakeAt.Equal(nextWake) {
		t.Fatalf("runtime/persistent wake mismatch: runtime=%s persistent=%#v", nextWake.Format(time.RFC3339Nano), state.Pace)
	}
	t.Logf("%s evaluated three related responsibilities and selected next wake %s with %v",
		label, nextWake.Format(time.RFC3339Nano), timingCall.Args)
}

func livePaceProviderLabel(provider LLMProvider) string {
	if provider == nil {
		return "unknown provider"
	}
	model := provider.Models()[ModelLarge]
	if model == "" {
		return provider.Name()
	}
	return provider.Name() + "/" + model
}

func waitForLivePaceTurn(t *testing.T, observer *Subscription, runDone <-chan struct{}, threadID string) {
	t.Helper()
	timer := time.NewTimer(150 * time.Second)
	defer timer.Stop()
	for {
		select {
		case event := <-observer.C:
			if event.From != threadID {
				continue
			}
			if event.Type == EventThinkError {
				t.Fatalf("live pace turn failed: %v", event.Error)
			}
			if event.Type == EventThinkDone {
				return
			}
		case <-runDone:
			t.Fatalf("live thinker %q stopped before completing a turn", threadID)
		case <-timer.C:
			t.Fatalf("timed out waiting for live pace turn on %q", threadID)
		}
	}
}
