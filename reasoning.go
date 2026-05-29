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
