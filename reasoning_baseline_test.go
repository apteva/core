package core

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

type capturedQualityRequest struct {
	model     string
	reasoning ReasoningLevel
}

type qualityCaptureProvider struct {
	mu        sync.Mutex
	reasoning ReasoningLevel
	requests  chan capturedQualityRequest
}

func (p *qualityCaptureProvider) Chat(_ context.Context, _ []Message, model string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	reasoning := normalizeReasoningLevel(p.reasoning)
	p.mu.Unlock()
	p.requests <- capturedQualityRequest{model: model, reasoning: reasoning}
	return ChatResponse{Text: "Handled external work."}, nil
}

func (p *qualityCaptureProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	return &qualityCaptureProvider{reasoning: settings.Level, requests: p.requests}
}
func (p *qualityCaptureProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "quality-large", ModelMedium: "quality-medium", ModelSmall: "quality-small"}
}
func (p *qualityCaptureProvider) Name() string                           { return "quality-capture" }
func (p *qualityCaptureProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *qualityCaptureProvider) SupportsNativeTools() bool              { return true }
func (p *qualityCaptureProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *qualityCaptureProvider) SetBuiltinTools([]string)               {}
func (p *qualityCaptureProvider) WithBuiltins([]string) LLMProvider      { return p }

func TestActiveExternalWorkAppliesConfiguredQualityFloor(t *testing.T) {
	thinker := &Thinker{
		model:             ModelSmall,
		agentModel:        ModelSmall,
		agentReasoning:    ReasoningLow,
		baselineModel:     ModelLarge,
		baselineReasoning: ReasoningAuto,
		activeWork:        true,
	}

	thinker.applyActiveModelFloor()
	if thinker.model != ModelMedium {
		t.Fatalf("active model = %s, want medium floor", thinker.model.String())
	}
	if got := thinker.effectiveReasoningLevel(); got != ReasoningAuto {
		t.Fatalf("active reasoning = %s, want configured auto floor", got)
	}

	thinker.activeWork = false
	thinker.model = thinker.agentModel
	thinker.applyActiveModelFloor()
	if thinker.model != ModelSmall || thinker.effectiveReasoningLevel() != ReasoningLow {
		t.Fatalf("timer-only profile = model %s reasoning %s, want small+low", thinker.model.String(), thinker.effectiveReasoningLevel())
	}
}

func TestExplicitLightweightBaselineRemainsLowDuringExternalWork(t *testing.T) {
	thinker := &Thinker{
		model:             ModelSmall,
		agentModel:        ModelSmall,
		agentReasoning:    ReasoningLow,
		baselineModel:     ModelSmall,
		baselineReasoning: ReasoningLow,
		activeWork:        true,
	}

	thinker.applyActiveModelFloor()
	if thinker.model != ModelSmall || thinker.effectiveReasoningLevel() != ReasoningLow {
		t.Fatalf("explicit lightweight baseline changed: model %s reasoning %s", thinker.model.String(), thinker.effectiveReasoningLevel())
	}
}

func TestPersistentWorkerKeepsConfiguredBaselineNotTemporaryPaceProfile(t *testing.T) {
	thinker := &Thinker{
		model:             ModelMedium,
		agentModel:        ModelSmall,
		agentReasoning:    ReasoningMinimal,
		baselineModel:     ModelLarge,
		baselineReasoning: ReasoningHigh,
		activeWork:        true,
	}
	thinker.publishRuntimeStatus()
	thread := &Thread{ID: "heavy-worker", Directive: "Analyze a substantial data set.", Thinker: thinker}

	state := persistentThreadStateBase(thread)
	if state.Model != "large" || state.Reasoning != "high" {
		t.Fatalf("persistent baseline = model %q reasoning %q, want large+high", state.Model, state.Reasoning)
	}
}

func TestReasoningQualityFloorNeverDowngradesHigherEffort(t *testing.T) {
	thinker := &Thinker{
		agentReasoning:    ReasoningHigh,
		baselineReasoning: ReasoningAuto,
		activeWork:        true,
	}
	if got := thinker.effectiveReasoningLevel(); got != ReasoningHigh {
		t.Fatalf("active reasoning = %s, want selected high", got)
	}
}

func TestExternalEventUsesBaselineThroughActualProviderCall(t *testing.T) {
	t.Chdir(t.TempDir())
	provider := &qualityCaptureProvider{requests: make(chan capturedQualityRequest, 2)}
	cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Handle external work accurately.", Mode: ModeAutonomous}
	thinker := NewThinker("", provider, cfg)
	thinker.agentModel = ModelSmall
	thinker.agentReasoning = ReasoningLow
	thinker.model = ModelSmall
	thinker.activeWork = false // represent a settled timer-only low profile
	thinker.InjectConsole("Reconcile several conflicting records and report the result.")
	go thinker.Run()
	defer thinker.Stop()

	select {
	case request := <-provider.requests:
		if request.model != "quality-medium" {
			t.Fatalf("external model = %q, want medium floor", request.model)
		}
		if request.reasoning != ReasoningAuto {
			t.Fatalf("external reasoning = %q, want auto baseline", request.reasoning)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for external provider call")
	}
}
