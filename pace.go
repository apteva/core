package core

import (
	"fmt"
	"strings"
	"time"
)

const (
	minSleep = 500 * time.Millisecond
	maxSleep = 24 * time.Hour
)

type parsedSleepDuration struct {
	duration time.Duration
	clamped  string
}

// parseSleepDurationDetailed parses the exact duration grammar exposed by the
// pace tool. Long-running responsibilities deliberately keep the 24-hour
// ceiling: the continuous loop wakes again and reassesses whether work is due.
func parseSleepDurationDetailed(raw string) (parsedSleepDuration, error) {
	raw = strings.TrimSpace(raw)
	if d, ok := rateAliases[raw]; ok {
		return parsedSleepDuration{duration: d}, nil
	}

	// time.ParseDuration intentionally rejects calendar units such as d and w.
	// Keep that behavior explicit: a day is 24h, while longer responsibilities
	// are implemented by repeated autonomous wake-ups.
	d, err := time.ParseDuration(raw)
	if err != nil || raw == "" {
		return parsedSleepDuration{}, fmt.Errorf("invalid sleep %q; use Go duration units ms, s, m, or h (maximum 24h)", raw)
	}

	result := parsedSleepDuration{duration: d}
	if d < minSleep {
		result.duration = minSleep
		result.clamped = fmt.Sprintf("requested %s; raised to %s", raw, formatPaceDuration(minSleep))
	}
	if d > maxSleep {
		result.duration = maxSleep
		result.clamped = fmt.Sprintf("requested %s; capped at %s", raw, formatPaceDuration(maxSleep))
	}
	return result, nil
}

// parseSleepDuration is kept as the small compatibility surface used by
// callers and tests that only need the effective duration.
func parseSleepDuration(raw string) (time.Duration, bool) {
	result, err := parseSleepDurationDetailed(raw)
	return result.duration, err == nil
}

func formatPaceDuration(d time.Duration) string {
	if d%time.Hour == 0 {
		return fmt.Sprintf("%dh", int64(d/time.Hour))
	}
	if d%time.Minute == 0 {
		return fmt.Sprintf("%dm", int64(d/time.Minute))
	}
	if d%time.Second == 0 {
		return fmt.Sprintf("%ds", int64(d/time.Second))
	}
	return d.String()
}

// applyPaceArgs validates the complete request before mutating thinker state.
// This prevents a bad sleep such as "7d" from silently changing the model or
// reasoning while leaving the prior cadence active.
func applyPaceArgs(t *Thinker, args map[string]string) (string, error) {
	nextSleep := t.agentSleep
	nextRate := t.agentRate
	nextModel := t.agentModel
	nextReasoning := t.agentReasoning
	nextProvider := t.provider

	var parts []string
	if raw := strings.TrimSpace(args["sleep"]); raw != "" {
		parsed, err := parseSleepDurationDetailed(raw)
		if err != nil {
			return "", err
		}
		nextSleep = parsed.duration
		nextRate = RateSleep
		sleepPart := "sleep=" + formatPaceDuration(parsed.duration)
		if parsed.clamped != "" {
			sleepPart += " (" + parsed.clamped + ")"
		}
		parts = append(parts, sleepPart)
	} else if r, ok := rateNames[args["rate"]]; ok {
		nextRate = r
		if d, ok2 := rateAliases[args["rate"]]; ok2 {
			nextSleep = d
		}
		parts = append(parts, "rate="+args["rate"])
	}

	if m, ok := modelNames[args["model"]]; ok {
		nextModel = m
		parts = append(parts, "model="+args["model"])
	}
	if rawReasoning := reasoningArgValue(args); rawReasoning != "" {
		r, ok := parseReasoningLevel(rawReasoning)
		if !ok {
			return "", fmt.Errorf("invalid reasoning %q (use auto, none, minimal, low, medium, high, or xhigh)", rawReasoning)
		}
		nextReasoning = r
		parts = append(parts, "reasoning="+r.String())
	}
	if name := args["provider"]; name != "" && t.pool != nil {
		if p := t.pool.Get(name); p != nil {
			nextProvider = p
			parts = append(parts, "provider="+name)
		}
	}

	t.agentSleep = nextSleep
	t.agentRate = nextRate
	t.agentModel = nextModel
	t.agentReasoning = nextReasoning
	t.provider = nextProvider
	if len(parts) == 0 {
		return "ok", nil
	}
	return "set " + strings.Join(parts, " "), nil
}
