package core

import (
	"fmt"
	"time"
)

const idleWakeFallback = 30 * time.Minute

// ensureIdleWake runs on the thinker loop before idle. An explicit agent
// decision wins; missing timing never silently becomes indefinite waiting.
func (t *Thinker) ensureIdleWake(now time.Time) error {
	if t.paused || !t.nextWakeAt.IsZero() || t.waitForEvents {
		return nil
	}
	select {
	case <-t.quit:
		return nil
	default:
	}
	wake := now.Add(idleWakeFallback)
	if t.persistPace != nil {
		if err := t.persistPace(paceState(idleWakeFallback, wake)); err != nil {
			return fmt.Errorf("persist fallback idle wake: %w", err)
		}
	}
	t.agentSleep = idleWakeFallback
	t.agentRate = RateSleep
	t.rate = RateSleep
	t.nextWakeAt = wake
	t.paceDurable = true
	t.paceRevision++
	t.publishRuntimeStatus()
	logMsg("PACE", fmt.Sprintf("[%s] no timing decision; fallback next_wake_at=%s", t.threadID, pendingWakeDescription(wake)))
	return nil
}

// Consume the recorded wait before processing new input so a crash cannot
// restore an obsolete event-only decision after the directive has changed.
func (t *Thinker) consumeEventOnlyWait() error {
	if !t.waitForEvents {
		return nil
	}
	if t.persistPace != nil {
		if err := t.persistPace(paceState(t.agentSleep, t.nextWakeAt)); err != nil {
			return fmt.Errorf("consume event-only wait: %w", err)
		}
	}
	t.waitForEvents = false
	t.publishRuntimeStatus()
	return nil
}
