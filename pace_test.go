package core

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestApplyPaceArgsKeepsTwentyFourHourMaximum(t *testing.T) {
	thinker := &Thinker{agentSleep: time.Hour, agentRate: RateNormal, agentModel: ModelLarge, agentReasoning: ReasoningLow}
	result, err := applyPaceArgs(thinker, map[string]string{"sleep": "48h"})
	if err != nil {
		t.Fatalf("applyPaceArgs: %v", err)
	}
	if thinker.agentSleep != 24*time.Hour {
		t.Fatalf("agentSleep = %v, want 24h", thinker.agentSleep)
	}
	if result != "set sleep=24h (requested 48h; capped at 24h)" {
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
	if result != "set sleep=24h model=small reasoning=minimal" {
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
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate work.",
		Mode:      ModeAutonomous,
	}
	if err := cfg.SetMainPace(PersistentPaceState{
		Sleep:      "1h",
		NextWakeAt: time.Now().Add(time.Hour),
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
	if state := cfg.GetMainPace(); state == nil || !state.NextWakeAt.IsZero() {
		t.Fatalf("early event did not clear pending wake: %#v", state)
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

func TestDurablePaceAdvancesPendingWakeForLaterCycles(t *testing.T) {
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

	// Simulate a later cycle after the original deadline has elapsed. The
	// model does not need to repeat the unchanged pace call.
	thinker.nextWakeAt = time.Now().Add(-time.Minute)
	thinker.publishRuntimeStatus()
	before := time.Now()
	thinker.refreshPersistentWakeForSleep(2 * time.Hour)
	stored := cfg.GetMainPace()
	if stored == nil {
		t.Fatal("later cycle did not persist pace")
	}
	if stored.NextWakeAt.Before(before.Add(2*time.Hour-time.Second)) || stored.NextWakeAt.After(time.Now().Add(2*time.Hour+time.Second)) {
		t.Fatalf("advanced wake = %v, want about two hours from now", stored.NextWakeAt)
	}
	if !thinker.nextWakeAt.Equal(stored.NextWakeAt) {
		t.Fatalf("runtime wake = %v, stored = %v", thinker.nextWakeAt, stored.NextWakeAt)
	}
}

func TestDefaultCadenceDoesNotPersistEverySleep(t *testing.T) {
	called := false
	thinker := &Thinker{
		agentRate:  RateSlow,
		agentSleep: 30 * time.Second,
		persistPace: func(PersistentPaceState) error {
			called = true
			return nil
		},
	}
	thinker.refreshPersistentWakeForSleep(30 * time.Second)
	if called {
		t.Fatal("unconfigured default cadence was persisted")
	}
}
