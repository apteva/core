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

// restorePersistentPace validates persisted scheduling state instead of
// trusting hand-edited config. A corrupted or legacy state falls back to the
// role default, and a future deadline can never exceed the same 24-hour cap
// enforced by the pace tool.
func restorePersistentPace(state *PersistentPaceState, fallback time.Duration, now time.Time) (time.Duration, time.Time) {
	if state == nil {
		return fallback, time.Time{}
	}
	sleep := fallback
	if strings.TrimSpace(state.Sleep) != "" {
		if parsed, err := parseSleepDurationDetailed(state.Sleep); err == nil {
			sleep = parsed.duration
		}
	}
	wake := state.NextWakeAt
	if wake.IsZero() {
		return sleep, time.Time{}
	}
	latest := now.Add(maxSleep)
	if wake.After(latest) {
		wake = latest
	}
	return sleep, wake
}

func paceState(sleep time.Duration, wake time.Time) PersistentPaceState {
	if sleep <= 0 {
		sleep = RateNormal.Delay()
	}
	if sleep > maxSleep {
		sleep = maxSleep
	}
	return PersistentPaceState{
		Sleep:      formatPaceDuration(sleep),
		NextWakeAt: wake,
	}
}

// pendingWakeDelay returns the remaining delay for the single agent-owned
// pending wake. A non-zero past deadline is still armed and returns zero so
// the run loop can deliver the timer wake immediately.
func pendingWakeDelay(wake, now time.Time) (time.Duration, bool) {
	if wake.IsZero() {
		return 0, false
	}
	if !wake.After(now) {
		return 0, true
	}
	return wake.Sub(now), true
}

func pendingWakeDescription(wake time.Time) string {
	if wake.IsZero() {
		return "none"
	}
	return wake.UTC().Format(time.RFC3339Nano)
}

// completeFiredWake consumes a one-shot timer after the model has received the
// wake. If the model called pace with a timing argument during that turn, the
// replacement deadline wins and is left untouched.
func (t *Thinker) completeFiredWake(revisionAtStart uint64) {
	if t == nil || !t.wakeDeadlineFired {
		return
	}
	t.wakeDeadlineFired = false
	if t.paceRevision != revisionAtStart {
		return
	}
	t.nextWakeAt = time.Time{}
	t.publishRuntimeStatus()
	if t.persistPace != nil {
		if err := t.persistPace(paceState(t.agentSleep, time.Time{})); err != nil {
			// Runtime must not repeatedly redeliver a wake the agent already
			// processed. Keeping the persisted expired deadline is safer on a
			// crash (at-least-once delivery) than silently skipping the work.
			logMsg("PACE", fmt.Sprintf("[%s] consume fired wake: %v", t.threadID, err))
		}
	}
}

// applyPaceArgs validates the complete request before mutating thinker state.
// This prevents a bad sleep such as "7d" from silently changing the model or
// reasoning while leaving the prior cadence active.
func applyPaceArgs(t *Thinker, args map[string]string) (string, error) {
	previousSleep := t.agentSleep
	previousRate := t.agentRate
	previousModel := t.agentModel
	previousReasoning := t.agentReasoning
	previousProvider := t.provider
	previousWake := t.nextWakeAt
	previousDurable := t.paceDurable
	previousRevision := t.paceRevision
	previousFired := t.wakeDeadlineFired

	nextSleep := t.agentSleep
	nextRate := t.agentRate
	nextModel := t.agentModel
	nextReasoning := t.agentReasoning
	nextProvider := t.provider
	nextWake := t.nextWakeAt
	nextRevision := t.paceRevision
	nextFired := t.wakeDeadlineFired

	rawSleep := strings.TrimSpace(args["sleep"])
	rawRate := strings.TrimSpace(args["rate"])
	clearWake := parseTruthy(args["clear_wake"])
	if rawSleep != "" && rawRate != "" {
		return "", fmt.Errorf("sleep and rate are mutually exclusive")
	}
	if clearWake && (rawSleep != "" || rawRate != "") {
		return "", fmt.Errorf("clear_wake cannot be combined with sleep or rate")
	}

	var parts []string
	now := time.Now()
	switch {
	case clearWake:
		nextWake = time.Time{}
		nextFired = false
		nextRevision++
		parts = append(parts, "cleared pending wake")
	case rawSleep != "":
		parsed, err := parseSleepDurationDetailed(rawSleep)
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
		nextWake = now.Add(parsed.duration)
		nextFired = false
		nextRevision++
	case rawRate != "":
		r, ok := rateNames[rawRate]
		if !ok {
			return "", fmt.Errorf("invalid rate %q (use fast, normal, slow, or sleep)", rawRate)
		}
		nextRate = r
		if d, ok2 := rateAliases[rawRate]; ok2 {
			nextSleep = d
		}
		parts = append(parts, "rate="+rawRate)
		effectiveSleep := nextSleep
		if effectiveSleep <= 0 {
			effectiveSleep = nextRate.Delay()
		}
		nextWake = now.Add(effectiveSleep)
		nextFired = false
		nextRevision++
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
	t.nextWakeAt = nextWake
	t.paceDurable = true
	t.paceRevision = nextRevision
	t.wakeDeadlineFired = nextFired
	t.publishRuntimeStatus()
	if t.persistPace != nil {
		if err := t.persistPace(paceState(nextSleep, nextWake)); err != nil {
			t.agentSleep = previousSleep
			t.agentRate = previousRate
			t.agentModel = previousModel
			t.agentReasoning = previousReasoning
			t.provider = previousProvider
			t.nextWakeAt = previousWake
			t.paceDurable = previousDurable
			t.paceRevision = previousRevision
			t.wakeDeadlineFired = previousFired
			t.publishRuntimeStatus()
			return "", fmt.Errorf("persist pace: %w", err)
		}
	}
	if clearWake {
		return strings.Join(parts, " "), nil
	}
	if rawSleep != "" || rawRate != "" {
		parts = append(parts, "next_wake_at="+pendingWakeDescription(nextWake))
		return "set " + strings.Join(parts, " "), nil
	}
	if len(parts) == 0 {
		return "pending_wake_at=" + pendingWakeDescription(nextWake), nil
	}
	parts = append(parts, "pending_wake_at="+pendingWakeDescription(nextWake))
	return "set " + strings.Join(parts, " "), nil
}
