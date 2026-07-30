package core

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type pacingTestProvider struct {
	mu        sync.Mutex
	calls     int
	started   chan int
	releases  []chan struct{}
	requests  [][]Message
	responses []ChatResponse
}

func newPacingTestProvider(callCount int) *pacingTestProvider {
	releases := make([]chan struct{}, callCount)
	for i := range releases {
		releases[i] = make(chan struct{})
	}
	return &pacingTestProvider{
		started:  make(chan int, callCount+4),
		releases: releases,
	}
}

func (p *pacingTestProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	index := p.calls
	p.calls++
	p.requests = append(p.requests, cloneMessages(messages))
	var release chan struct{}
	if index < len(p.releases) {
		release = p.releases[index]
	}
	response := ChatResponse{Text: "Handled the wake."}
	if index < len(p.responses) {
		response = p.responses[index]
	}
	p.mu.Unlock()
	p.started <- index + 1
	if release != nil {
		<-release
	}
	return response, nil
}

func (p *pacingTestProvider) release(call int) {
	if call <= 0 || call > len(p.releases) {
		return
	}
	select {
	case <-p.releases[call-1]:
	default:
		close(p.releases[call-1])
	}
}

func (p *pacingTestProvider) request(call int) []Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	if call <= 0 || call > len(p.requests) {
		return nil
	}
	return cloneMessages(p.requests[call-1])
}

func (p *pacingTestProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "pace-large", ModelMedium: "pace-medium", ModelSmall: "pace-small"}
}
func (p *pacingTestProvider) Name() string                           { return "pace-test-provider" }
func (p *pacingTestProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *pacingTestProvider) SupportsNativeTools() bool              { return true }
func (p *pacingTestProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *pacingTestProvider) SetBuiltinTools([]string)               {}
func (p *pacingTestProvider) WithBuiltins([]string) LLMProvider      { return p }

func TestApplyPaceArgsKeepsTwentyFourHourMaximum(t *testing.T) {
	thinker := &Thinker{agentSleep: time.Hour, agentRate: RateNormal, agentModel: ModelLarge, agentReasoning: ReasoningLow}
	result, err := applyPaceArgs(thinker, map[string]string{"sleep": "48h"})
	if err != nil {
		t.Fatalf("applyPaceArgs: %v", err)
	}
	if thinker.agentSleep != 24*time.Hour {
		t.Fatalf("agentSleep = %v, want 24h", thinker.agentSleep)
	}
	if !strings.HasPrefix(result, "set sleep=24h (requested 48h; capped at 24h) next_wake_at=") {
		t.Fatalf("result = %q", result)
	}
}

func TestApplyPaceArgsRejectsCalendarUnitsAtomically(t *testing.T) {
	thinker := &Thinker{agentSleep: time.Hour, agentRate: RateNormal, agentModel: ModelLarge, agentReasoning: ReasoningLow}
	beforeModel := thinker.agentModel
	beforeReasoning := thinker.agentReasoning
	result, err := applyPaceArgs(thinker, map[string]string{
		"sleep":     "7d",
		"model":     "small",
		"reasoning": "minimal",
	})
	if err == nil || !strings.Contains(err.Error(), `invalid sleep "7d"`) || !strings.Contains(err.Error(), "maximum 24h") {
		t.Fatalf("error = %v", err)
	}
	if result != "" {
		t.Fatalf("result = %q, want empty", result)
	}
	if thinker.agentSleep != time.Hour || thinker.agentRate != RateNormal || thinker.agentModel != beforeModel || thinker.agentReasoning != beforeReasoning {
		t.Fatalf("invalid pace partially mutated thinker: sleep=%v rate=%v model=%v reasoning=%v", thinker.agentSleep, thinker.agentRate, thinker.agentModel, thinker.agentReasoning)
	}
}

func TestApplyPaceArgsReportsExactEffectiveSleep(t *testing.T) {
	thinker := &Thinker{}
	result, err := applyPaceArgs(thinker, map[string]string{"sleep": "24h", "model": "small", "reasoning": "minimal"})
	if err != nil {
		t.Fatalf("applyPaceArgs: %v", err)
	}
	if !strings.HasPrefix(result, "set sleep=24h model=small reasoning=minimal next_wake_at=") {
		t.Fatalf("result = %q", result)
	}
	if thinker.agentSleep != 24*time.Hour || thinker.agentRate != RateSleep || thinker.agentModel != ModelSmall || thinker.agentReasoning != ReasoningMinimal {
		t.Fatalf("unexpected thinker state: sleep=%v rate=%v model=%v reasoning=%v", thinker.agentSleep, thinker.agentRate, thinker.agentModel, thinker.agentReasoning)
	}
}

func TestParseSleepDurationRejectsWeekUnit(t *testing.T) {
	if _, ok := parseSleepDuration("1w"); ok {
		t.Fatal("1w should be rejected")
	}
}

func TestMainPacePersistsCadenceAndPendingWakeAcrossRestart(t *testing.T) {
	t.Chdir(t.TempDir())
	path := filepath.Join(t.TempDir(), "config.json")
	cfg := &Config{path: path, Directive: "# Role\nCoordinate work.", Mode: ModeAutonomous}
	provider := newParkedAPIProvider()
	thinker := NewThinker("", provider, cfg)
	defer func() {
		provider.Release()
		thinker.Stop()
	}()

	before := time.Now()
	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "2h"}); err != nil {
		t.Fatalf("apply pace: %v", err)
	}
	stored := cfg.GetMainPace()
	if stored == nil || stored.Sleep != "2h" {
		t.Fatalf("stored main pace = %#v", stored)
	}
	if stored.NextWakeAt.Before(before.Add(2*time.Hour-time.Second)) || stored.NextWakeAt.After(time.Now().Add(2*time.Hour+time.Second)) {
		t.Fatalf("stored main next wake = %v, want about two hours from now", stored.NextWakeAt)
	}

	reloaded := &Config{path: path}
	if err := reloaded.load(); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	restartProvider := newParkedAPIProvider()
	restarted := NewThinker("", restartProvider, reloaded)
	defer func() {
		restartProvider.Release()
		restarted.Stop()
	}()
	if restarted.agentSleep != 2*time.Hour {
		t.Fatalf("restored main sleep = %v, want 2h", restarted.agentSleep)
	}
	if restarted.resumeWakeAt.IsZero() || !restarted.resumeWakeAt.Equal(stored.NextWakeAt) {
		t.Fatalf("restored main wake = %v, want %v", restarted.resumeWakeAt, stored.NextWakeAt)
	}
}

func TestPersistentThreadPaceSurvivesRestart(t *testing.T) {
	t.Chdir(t.TempDir())
	path := filepath.Join(t.TempDir(), "config.json")
	cfg := &Config{path: path, Directive: "# Role\nCoordinate work.", Mode: ModeAutonomous}
	provider := newParkedAPIProvider()
	parent := NewThinker("", provider, cfg)
	defer func() {
		provider.Release()
		parent.Stop()
	}()

	if err := parent.threads.SpawnWithOpts(
		"domain-owner",
		"# Role\nOwn a continuing domain responsibility.",
		nil,
		SpawnOpts{DeferRun: true},
	); err != nil {
		t.Fatalf("spawn persistent owner: %v", err)
	}
	initial, err := parent.threads.PersistentState("domain-owner")
	if err != nil {
		t.Fatalf("initial persistent state: %v", err)
	}
	if err := cfg.SaveThread(initial); err != nil {
		t.Fatalf("save initial thread: %v", err)
	}
	worker := parent.threads.threads["domain-owner"].Thinker
	if _, err := applyPaceArgs(worker, map[string]string{"sleep": "45m"}); err != nil {
		t.Fatalf("apply worker pace: %v", err)
	}

	var stored PersistentThread
	for _, candidate := range cfg.GetThreads() {
		if candidate.ID == "domain-owner" {
			stored = candidate
			break
		}
	}
	if stored.Pace == nil || stored.Pace.Sleep != "45m" || stored.Pace.NextWakeAt.IsZero() {
		t.Fatalf("stored thread pace = %#v", stored.Pace)
	}

	reloaded := &Config{path: path}
	if err := reloaded.load(); err != nil {
		t.Fatalf("reload config: %v", err)
	}
	restartProvider := newParkedAPIProvider()
	restarted := NewThinker("", restartProvider, reloaded)
	defer func() {
		restartProvider.Release()
		restarted.Stop()
	}()
	live := restarted.threads.threads["domain-owner"]
	if live == nil || live.Thinker == nil {
		t.Fatal("persistent owner was not restored")
	}
	if live.Thinker.agentSleep != 45*time.Minute {
		t.Fatalf("restored worker sleep = %v, want 45m", live.Thinker.agentSleep)
	}
	if !live.Thinker.nextWakeAt.Equal(stored.Pace.NextWakeAt) {
		t.Fatalf("restored worker wake = %v, want %v", live.Thinker.nextWakeAt, stored.Pace.NextWakeAt)
	}
	select {
	case <-restartProvider.started:
		t.Fatal("restored worker called the provider before its pending wake")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestRestoredWakeWaitsButInboxEventResumesEarly(t *testing.T) {
	t.Chdir(t.TempDir())
	pendingWake := time.Now().Add(time.Hour).UTC()
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{
		Sleep:      "1h",
		NextWakeAt: pendingWake,
	}); err != nil {
		t.Fatalf("save main pace: %v", err)
	}
	provider := newParkedAPIProvider()
	thinker := NewThinker("", provider, cfg)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		provider.Release()
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("thinker did not stop before test cleanup")
		}
	}()

	select {
	case <-provider.started:
		t.Fatal("restored thinker called the provider before its pending wake")
	case <-time.After(100 * time.Millisecond):
	}

	thinker.InjectConsole("Handle this event now.")
	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("inbox event did not interrupt restored wake")
	}
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("early event changed pending wake while processing: %#v, want %v", state, pendingWake)
	}
	provider.Release()
	deadline := time.Now().Add(2 * time.Second)
	for thinker.status().LLMActive && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("completed early-event turn changed pending wake: %#v, want %v", state, pendingWake)
	}
	delay, armed := pendingWakeDelay(thinker.status().NextWakeAt, time.Now())
	if !armed || delay < 59*time.Minute || delay > time.Hour {
		t.Fatalf("remaining timer = %v armed=%v, want approximately one hour minus event processing", delay, armed)
	}
}

func TestRestorePersistentPaceRetainsTwentyFourHourMaximum(t *testing.T) {
	now := time.Now()
	sleep, wake := restorePersistentPace(&PersistentPaceState{
		Sleep:      "72h",
		NextWakeAt: now.Add(72 * time.Hour),
	}, time.Minute, now)
	if sleep != maxSleep {
		t.Fatalf("restored sleep = %v, want 24h cap", sleep)
	}
	if wake.Before(now.Add(maxSleep-time.Second)) || wake.After(now.Add(maxSleep+time.Second)) {
		t.Fatalf("restored wake = %v, want capped near %v", wake, now.Add(maxSleep))
	}
}

func TestFiredWakeIsConsumedWithoutAutomaticAdvance(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	provider := newParkedAPIProvider()
	thinker := NewThinker("", provider, cfg)
	defer func() {
		provider.Release()
		thinker.Stop()
	}()
	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "2h"}); err != nil {
		t.Fatalf("apply pace: %v", err)
	}

	// Simulate delivery and completion of that one-shot timer wake without a
	// replacement pace call.
	thinker.nextWakeAt = time.Now().Add(-time.Minute)
	thinker.wakeDeadlineFired = true
	thinker.publishRuntimeStatus()
	revision := thinker.paceRevision
	thinker.completeFiredWake(revision)
	stored := cfg.GetMainPace()
	if stored == nil {
		t.Fatal("consumed wake did not retain durable pace state")
	}
	if !stored.NextWakeAt.IsZero() || !thinker.nextWakeAt.IsZero() {
		t.Fatalf("timer wake was automatically advanced: runtime=%v stored=%v", thinker.nextWakeAt, stored.NextWakeAt)
	}
	if stored.Sleep != "2h" {
		t.Fatalf("last agent timing decision = %q, want 2h", stored.Sleep)
	}
}

func TestFiredWakeKeepsExplicitAgentReplacement(t *testing.T) {
	thinker := &Thinker{
		agentSleep:        time.Hour,
		agentRate:         RateSleep,
		nextWakeAt:        time.Now().Add(-time.Second),
		paceDurable:       true,
		wakeDeadlineFired: true,
	}
	revisionAtStart := thinker.paceRevision
	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "2h"}); err != nil {
		t.Fatal(err)
	}
	replacement := thinker.nextWakeAt
	thinker.completeFiredWake(revisionAtStart)
	if thinker.nextWakeAt.IsZero() || !thinker.nextWakeAt.Equal(replacement) {
		t.Fatalf("timer completion removed explicit replacement: got=%v want=%v", thinker.nextWakeAt, replacement)
	}
	if thinker.wakeDeadlineFired {
		t.Fatal("replacement left the prior timer marked fired")
	}
}

func TestPendingWakeDelayUsesOnlyRemainingDuration(t *testing.T) {
	now := time.Date(2026, time.July, 29, 8, 35, 0, 0, time.UTC)
	wake := time.Date(2026, time.July, 29, 8, 52, 0, 0, time.UTC)
	delay, armed := pendingWakeDelay(wake, now)
	if !armed || delay != 17*time.Minute {
		t.Fatalf("pendingWakeDelay() = %v armed=%v, want 17m", delay, armed)
	}
	if delay, armed := pendingWakeDelay(time.Time{}, now); armed || delay != 0 {
		t.Fatalf("zero pending wake = %v armed=%v", delay, armed)
	}
	if delay, armed := pendingWakeDelay(now.Add(-time.Second), now); !armed || delay != 0 {
		t.Fatalf("overdue pending wake = %v armed=%v", delay, armed)
	}
}

func TestPaceReplacesOnlyWhenAgentChangesWakeTiming(t *testing.T) {
	thinker := &Thinker{
		agentSleep:     time.Hour,
		agentRate:      RateSleep,
		agentModel:     ModelLarge,
		agentReasoning: ReasoningLow,
	}
	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "1h"}); err != nil {
		t.Fatal(err)
	}
	original := thinker.nextWakeAt
	originalRevision := thinker.paceRevision

	result, err := applyPaceArgs(thinker, map[string]string{"model": "small"})
	if err != nil {
		t.Fatal(err)
	}
	if !thinker.nextWakeAt.Equal(original) || thinker.paceRevision != originalRevision {
		t.Fatalf("profile-only pace moved wake: before=%v after=%v revisions=%d→%d", original, thinker.nextWakeAt, originalRevision, thinker.paceRevision)
	}
	if !strings.Contains(result, "pending_wake_at="+pendingWakeDescription(original)) {
		t.Fatalf("profile-only result did not report preserved wake: %q", result)
	}

	result, err = applyPaceArgs(thinker, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if result != "pending_wake_at="+pendingWakeDescription(original) || !thinker.nextWakeAt.Equal(original) {
		t.Fatalf("pace() changed or misreported wake: result=%q wake=%v", result, thinker.nextWakeAt)
	}

	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "2h"}); err != nil {
		t.Fatal(err)
	}
	if !thinker.nextWakeAt.After(original.Add(59 * time.Minute)) {
		t.Fatalf("explicit replacement wake = %v, want roughly one hour after original %v", thinker.nextWakeAt, original)
	}
	if thinker.paceRevision != originalRevision+1 {
		t.Fatalf("replacement revision = %d, want %d", thinker.paceRevision, originalRevision+1)
	}
}

func TestPaceClearWakeIsExplicitAndAtomic(t *testing.T) {
	thinker := &Thinker{agentSleep: time.Hour, agentRate: RateSleep}
	if _, err := applyPaceArgs(thinker, map[string]string{"sleep": "1h"}); err != nil {
		t.Fatal(err)
	}
	result, err := applyPaceArgs(thinker, map[string]string{"clear_wake": "true"})
	if err != nil {
		t.Fatal(err)
	}
	if result != "cleared pending wake" || !thinker.nextWakeAt.IsZero() {
		t.Fatalf("clear result=%q wake=%v", result, thinker.nextWakeAt)
	}

	beforeRevision := thinker.paceRevision
	if _, err := applyPaceArgs(thinker, map[string]string{"clear_wake": "true", "sleep": "5m"}); err == nil {
		t.Fatal("clear_wake combined with sleep was accepted")
	}
	if !thinker.nextWakeAt.IsZero() || thinker.paceRevision != beforeRevision {
		t.Fatal("invalid combined pace mutated wake state")
	}
}

func TestClearedWakeRestartsEventOnlyUntilAnEventArrives(t *testing.T) {
	t.Chdir(t.TempDir())
	path := filepath.Join(t.TempDir(), "config.json")
	cfg := &Config{
		path:      path,
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	setupProvider := newParkedAPIProvider()
	setup := NewThinker("", setupProvider, cfg)
	if _, err := applyPaceArgs(setup, map[string]string{"sleep": "1h"}); err != nil {
		t.Fatal(err)
	}
	if _, err := applyPaceArgs(setup, map[string]string{"clear_wake": "true"}); err != nil {
		t.Fatal(err)
	}
	setupProvider.Release()
	setup.Stop()

	reloaded := &Config{path: path}
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	provider := newParkedAPIProvider()
	restarted := NewThinker("", provider, reloaded)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		restarted.Run()
	}()
	defer func() {
		provider.Release()
		restarted.Stop()
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("restarted thinker did not stop")
		}
	}()

	select {
	case <-provider.started:
		t.Fatal("cleared wake restarted as an automatic model turn")
	case <-time.After(100 * time.Millisecond):
	}
	restarted.InjectConsole("A new event arrived.")
	select {
	case <-provider.started:
	case <-time.After(2 * time.Second):
		t.Fatal("event-only thinker did not wake for an event")
	}
}

func TestToolResultWakePreservesPendingDeadlineAndExposesState(t *testing.T) {
	t.Chdir(t.TempDir())
	pendingWake := time.Now().Add(time.Hour).UTC()
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{Sleep: "1h", NextWakeAt: pendingWake}); err != nil {
		t.Fatal(err)
	}
	provider := newPacingTestProvider(1)
	thinker := NewThinker("", provider, cfg)
	observer := thinker.bus.SubscribeAll("pace-tool-result-observer", 16)
	defer thinker.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		provider.release(1)
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("thinker did not stop")
		}
	}()

	result := ToolResult{CallID: "lookup-1", ToolName: "lookup", Content: "LOOKUP_OK"}
	thinker.bus.Publish(Event{
		Type: EventInbox, From: "tool:lookup", To: "main",
		Text: "[tool:lookup] LOOKUP_OK", ToolResult: &result,
	})
	select {
	case call := <-provider.started:
		if call != 1 {
			t.Fatalf("first call = %d", call)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("tool result did not wake thinker")
	}

	requestText := messagesText(provider.request(1))
	for _, want := range []string{
		"[WAKE STATE]",
		"reason: tool_result",
		"pending_wake_at: " + pendingWake.Format(time.RFC3339Nano),
	} {
		if !strings.Contains(requestText, want) {
			t.Fatalf("tool-result request missing %q:\n%s", want, requestText)
		}
	}
	provider.release(1)
	waitForPacingThinkDone(t, observer, "main")
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("tool-result continuation moved pending wake: %#v", state)
	}
}

func TestTimerWakeIsConsumedAndDoesNotInventRecurrence(t *testing.T) {
	t.Chdir(t.TempDir())
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{
		Sleep:      "1h",
		NextWakeAt: time.Now().Add(80 * time.Millisecond),
	}); err != nil {
		t.Fatal(err)
	}
	provider := newPacingTestProvider(1)
	thinker := NewThinker("", provider, cfg)
	observer := thinker.bus.SubscribeAll("pace-timer-observer", 16)
	defer thinker.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		provider.release(1)
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("thinker did not stop")
		}
	}()

	select {
	case call := <-provider.started:
		if call != 1 {
			t.Fatalf("timer call = %d", call)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pending timer did not wake thinker")
	}
	requestText := messagesText(provider.request(1))
	if !strings.Contains(requestText, "reason: timer") ||
		!strings.Contains(requestText, "pending_wake_at: none (timer fired)") {
		t.Fatalf("timer request lacks consumed wake state:\n%s", requestText)
	}
	provider.release(1)
	waitForPacingThinkDone(t, observer, "main")
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.IsZero() {
		t.Fatalf("timer completion invented another wake: %#v", state)
	}
	select {
	case call := <-provider.started:
		t.Fatalf("unexpected automatic recurrence call %d", call)
	case <-time.After(150 * time.Millisecond):
	}
}

func TestEventProcessingThatCrossesDeadlineImmediatelyDeliversTimerWake(t *testing.T) {
	t.Chdir(t.TempDir())
	pendingWake := time.Now().Add(120 * time.Millisecond).UTC()
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{Sleep: "1h", NextWakeAt: pendingWake}); err != nil {
		t.Fatal(err)
	}
	provider := newPacingTestProvider(2)
	thinker := NewThinker("", provider, cfg)
	observer := thinker.bus.SubscribeAll("pace-cross-deadline-observer", 16)
	defer thinker.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		thinker.Run()
	}()
	defer func() {
		thinker.Stop()
		provider.release(1)
		provider.release(2)
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("thinker did not stop")
		}
	}()

	thinker.InjectConsole("Handle this early event.")
	select {
	case call := <-provider.started:
		if call != 1 {
			t.Fatalf("event call = %d", call)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("event did not wake thinker")
	}
	firstRequest := messagesText(provider.request(1))
	if !strings.Contains(firstRequest, "reason: event") ||
		!strings.Contains(firstRequest, "pending_wake_at: "+pendingWake.Format(time.RFC3339Nano)) {
		t.Fatalf("early-event request did not preserve pending timer:\n%s", firstRequest)
	}
	time.Sleep(time.Until(pendingWake) + 30*time.Millisecond)
	provider.release(1)
	waitForPacingThinkDone(t, observer, "main")

	select {
	case call := <-provider.started:
		if call != 2 {
			t.Fatalf("due call = %d", call)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("deadline crossed during event processing did not trigger due turn")
	}
	secondRequest := messagesText(provider.request(2))
	if !strings.Contains(secondRequest, "reason: timer") ||
		!strings.Contains(secondRequest, "pending_wake_at: none (timer fired)") {
		t.Fatalf("due request lacks timer state:\n%s", secondRequest)
	}
	provider.release(2)
	waitForPacingThinkDone(t, observer, "main")
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.IsZero() {
		t.Fatalf("completed timer wake was not consumed: %#v", state)
	}
}

func TestPersistentWorkerEarlyEventPreservesItsOwnPendingWake(t *testing.T) {
	t.Chdir(t.TempDir())
	pendingWake := time.Now().Add(time.Hour).UTC()
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nGovern work ownership.",
		Mode:      ModeAutonomous,
	}
	provider := newPacingTestProvider(1)
	parent := NewThinker("", provider, cfg)
	defer parent.Stop()
	if err := parent.threads.SpawnWithOpts(
		"crm-owner",
		"# Role\nOwn recurring CRM operations and their schedule.",
		nil,
		SpawnOpts{
			DeferRun: true,
			Pace:     &PersistentPaceState{Sleep: "1h", NextWakeAt: pendingWake},
		},
	); err != nil {
		t.Fatal(err)
	}
	initialState, err := parent.threads.PersistentState("crm-owner")
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.SaveThread(initialState); err != nil {
		t.Fatal(err)
	}
	worker := parent.threads.threads["crm-owner"].Thinker
	observer := parent.bus.SubscribeAll("pace-worker-observer", 16)
	defer parent.bus.Unsubscribe(observer.ID)
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		worker.Run()
	}()
	defer func() {
		provider.release(1)
		worker.Stop()
		select {
		case <-runDone:
		case <-time.After(2 * time.Second):
			t.Error("worker did not stop")
		}
	}()

	worker.InjectConsole("Record this unrelated status event; it does not change the CRM cycle.")
	select {
	case call := <-provider.started:
		if call != 1 {
			t.Fatalf("worker call = %d", call)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("event did not wake persistent worker")
	}
	requestText := messagesText(provider.request(1))
	if !strings.Contains(requestText, "reason: event") ||
		!strings.Contains(requestText, "pending_wake_at: "+pendingWake.Format(time.RFC3339Nano)) {
		t.Fatalf("worker did not receive its preserved wake state:\n%s", requestText)
	}
	provider.release(1)
	waitForPacingThinkDone(t, observer, "crm-owner")

	live := parent.threads.threads["crm-owner"]
	if live == nil || !live.Thinker.status().NextWakeAt.Equal(pendingWake) {
		t.Fatalf("worker runtime wake moved: thread=%v", live)
	}
	state, err := parent.threads.PersistentState("crm-owner")
	if err != nil {
		t.Fatal(err)
	}
	if state.Pace == nil || !state.Pace.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("worker persistent wake moved: %#v", state.Pace)
	}
	var stored *PersistentPaceState
	for _, candidate := range cfg.GetThreads() {
		if candidate.ID == "crm-owner" {
			stored = candidate.Pace
			break
		}
	}
	if stored == nil || !stored.NextWakeAt.Equal(pendingWake) {
		t.Fatalf("worker config wake moved: %#v", stored)
	}
}

func messagesText(messages []Message) string {
	var parts []string
	for _, message := range messages {
		parts = append(parts, message.TextContent())
	}
	return strings.Join(parts, "\n")
}

func waitForPacingThinkDone(t *testing.T, observer *Subscription, threadID string) {
	t.Helper()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	for {
		select {
		case event := <-observer.C:
			if event.Type == EventThinkDone && event.From == threadID {
				return
			}
		case <-timer.C:
			t.Fatalf("thinker %q did not finish its turn", threadID)
		}
	}
}
