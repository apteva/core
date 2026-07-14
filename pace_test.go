package core

import (
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
