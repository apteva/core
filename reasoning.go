package core

import "strings"

type ReasoningLevel string

const (
	ReasoningAuto    ReasoningLevel = "auto"
	ReasoningNone    ReasoningLevel = "none"
	ReasoningMinimal ReasoningLevel = "minimal"
	ReasoningLow     ReasoningLevel = "low"
	ReasoningMedium  ReasoningLevel = "medium"
	ReasoningHigh    ReasoningLevel = "high"
	ReasoningXHigh   ReasoningLevel = "xhigh"
)

type ReasoningSettings struct {
	Level ReasoningLevel
}

func (r ReasoningLevel) String() string {
	if r == "" {
		return string(ReasoningAuto)
	}
	return string(r)
}

func parseReasoningLevel(raw string) (ReasoningLevel, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return ReasoningAuto, false
	case "auto":
		return ReasoningAuto, true
	case "none", "off", "disabled":
		return ReasoningNone, true
	case "minimal":
		return ReasoningMinimal, true
	case "low":
		return ReasoningLow, true
	case "medium":
		return ReasoningMedium, true
	case "high":
		return ReasoningHigh, true
	case "xhigh", "x-high", "extra_high", "extra-high":
		return ReasoningXHigh, true
	default:
		return ReasoningAuto, false
	}
}

func reasoningFromArgs(args map[string]string) (ReasoningLevel, bool) {
	if args == nil {
		return ReasoningAuto, false
	}
	if raw, ok := args["reasoning"]; ok && strings.TrimSpace(raw) != "" {
		return parseReasoningLevel(raw)
	}
	if raw, ok := args["thinking"]; ok && strings.TrimSpace(raw) != "" {
		return parseReasoningLevel(raw)
	}
	return ReasoningAuto, false
}

func reasoningArgValue(args map[string]string) string {
	if args == nil {
		return ""
	}
	if raw := strings.TrimSpace(args["reasoning"]); raw != "" {
		return raw
	}
	return strings.TrimSpace(args["thinking"])
}

func normalizeReasoningLevel(level ReasoningLevel) ReasoningLevel {
	if level == "" {
		return ReasoningAuto
	}
	return level
}

func providerWithReasoning(provider LLMProvider, level ReasoningLevel) LLMProvider {
	if provider == nil {
		return nil
	}
	rp, ok := provider.(interface {
		WithReasoning(ReasoningSettings) LLMProvider
	})
	if !ok {
		return provider
	}
	return rp.WithReasoning(ReasoningSettings{Level: normalizeReasoningLevel(level)})
}

func reasoningQualityRank(level ReasoningLevel) int {
	switch normalizeReasoningLevel(level) {
	case ReasoningNone:
		return 0
	case ReasoningMinimal:
		return 1
	case ReasoningLow:
		return 2
	case ReasoningAuto, ReasoningMedium:
		// Auto is Core's normal baseline. Providers may tune it, but a model-
		// selected low/minimal override must not undercut it during newly
		// assigned external work.
		return 3
	case ReasoningHigh:
		return 4
	case ReasoningXHigh:
		return 5
	default:
		return 3
	}
}

func (t *Thinker) effectiveReasoningLevel() ReasoningLevel {
	if t == nil {
		return ReasoningAuto
	}
	selected := normalizeReasoningLevel(t.agentReasoning)
	baseline := normalizeReasoningLevel(t.baselineReasoning)
	if t.activeWork && reasoningQualityRank(selected) < reasoningQualityRank(baseline) {
		return baseline
	}
	return selected
}

func (t *Thinker) applyActiveModelFloor() {
	if t == nil || !t.activeWork {
		return
	}
	// A worker explicitly configured as small remains a small worker. For
	// ordinary large/medium threads, active externally-driven workflows never
	// fall below medium even if a prior idle pace selected small.
	if t.baselineModel != ModelSmall && t.model == ModelSmall {
		t.model = ModelMedium
	}
}
