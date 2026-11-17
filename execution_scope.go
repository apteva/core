package core

import "fmt"

// Exact-thread rules are additive to the instance controller. They are not
// inherited by descendants and cannot bypass an instance breakpoint.
type ExecutionThreadControl struct {
	Mode        ExecutionControlMode `json:"mode"`
	Breakpoints []string             `json:"breakpoints,omitempty"`
}

type ExecutionWaitingStatus struct {
	ThreadID string `json:"thread_id"`
	Scope    string `json:"scope"`
	Phase    string `json:"phase"`
}

func cloneExecutionThreads(in map[string]ExecutionThreadControl) map[string]ExecutionThreadControl {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]ExecutionThreadControl, len(in))
	for id, rule := range in {
		rule.Breakpoints = append([]string(nil), rule.Breakpoints...)
		out[id] = rule
	}
	return out
}
func cloneExecutionControl(in ExecutionControlConfig) ExecutionControlConfig {
	in.Breakpoints = append([]string(nil), in.Breakpoints...)
	in.Threads = cloneExecutionThreads(in.Threads)
	return in
}
func validateExecutionControl(cfg ExecutionControlConfig) error {
	if cfg.Scope != "" && cfg.Scope != "instance" {
		return fmt.Errorf("execution_control.scope must be instance; use threads for exact-thread rules")
	}
	if cfg.Mode != "" && cfg.Mode != ExecutionAuto && cfg.Mode != ExecutionPaused && cfg.Mode != ExecutionStep {
		return fmt.Errorf("invalid execution control mode")
	}
	if len(cfg.Threads) > 128 {
		return fmt.Errorf("execution control exceeds 128 thread rules")
	}
	if err := validateExecutionBreakpoints(cfg.Breakpoints); err != nil {
		return err
	}
	for id, rule := range cfg.Threads {
		if err := validateThreadID(id); err != nil {
			return err
		}
		if rule.Mode != ExecutionAuto && rule.Mode != ExecutionPaused && rule.Mode != ExecutionStep {
			return fmt.Errorf("invalid execution mode for thread %q", id)
		}
		if err := validateExecutionBreakpoints(rule.Breakpoints); err != nil {
			return err
		}
	}
	return nil
}
func validateExecutionBreakpoints(bps []string) error {
	for _, bp := range bps {
		switch ExecutionPhase(bp) {
		case ExecutionPhaseIterationStart, ExecutionPhaseInputReady, ExecutionPhaseLLMStart, ExecutionPhaseLLMDone, ExecutionPhaseToolBefore, ExecutionPhaseToolAfter, ExecutionPhaseIterationDone, ExecutionPhaseSleepBefore:
		default:
			return fmt.Errorf("unknown execution breakpoint %q", bp)
		}
	}
	return nil
}
func executionThreadGates(rule ExecutionThreadControl, phase ExecutionPhase) bool {
	if rule.Mode == "" || rule.Mode == ExecutionAuto {
		return false
	}
	bps := rule.Breakpoints
	if len(bps) == 0 {
		bps = defaultExecutionBreakpoints()
	}
	for _, bp := range bps {
		if ExecutionPhase(bp) == phase {
			return true
		}
	}
	return false
}
func (c *ExecutionController) blockingScopeLocked(gate ExecutionGate) string {
	if c.shouldGateLocked(gate.Phase) {
		return "instance"
	}
	if executionThreadGates(c.threads[gate.ThreadID], gate.Phase) {
		return "thread"
	}
	return ""
}
func (c *ExecutionController) ShouldGateThread(threadID string, phase ExecutionPhase) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.blockingScopeLocked(ExecutionGate{ThreadID: threadID, Phase: phase}) != ""
}
func (c *ExecutionController) releaseUngatedLocked() {
	for id, w := range c.waiters {
		if scope := c.blockingScopeLocked(w.gate); scope == "" || c.consumeStepLocked(w.gate, scope) {
			c.releaseWaiterLocked(id)
		}
	}
}
func (c *ExecutionController) consumeStepLocked(gate ExecutionGate, scope string) bool {
	if c.targetSteps[gate.ThreadID] > 0 {
		c.targetSteps[gate.ThreadID]--
		return true
	}
	if c.pendingSteps > 0 {
		c.pendingSteps--
		return true
	}
	if scope == "thread" && c.threadSteps[gate.ThreadID] > 0 {
		c.threadSteps[gate.ThreadID]--
		return true
	}
	return false
}
func (c *ExecutionController) controlThreadLocked(action ExecutionControlAction) error {
	if action.ThreadID == "" {
		return fmt.Errorf("thread scope requires thread_id")
	}
	if err := validateThreadID(action.ThreadID); err != nil {
		return err
	}
	if err := validateExecutionBreakpoints(action.Breakpoints); err != nil {
		return err
	}
	if c.threads == nil {
		c.threads = map[string]ExecutionThreadControl{}
	}
	rule, exists := c.threads[action.ThreadID]
	if !exists && action.Action != "run" && len(c.threads) >= 128 {
		return fmt.Errorf("execution control exceeds 128 thread rules")
	}
	if action.Breakpoints != nil {
		rule.Breakpoints = append([]string(nil), action.Breakpoints...)
	}
	switch action.Action {
	case "run":
		delete(c.threads, action.ThreadID)
		delete(c.threadSteps, action.ThreadID)
		c.releaseUngatedLocked()
		return nil
	case "pause":
		rule.Mode = ExecutionPaused
		delete(c.threadSteps, action.ThreadID)
	case "step":
		rule.Mode = ExecutionStep
	default:
		return fmt.Errorf("action must be run, pause, or step")
	}
	c.threads[action.ThreadID] = rule
	if action.Action == "step" {
		for _, id := range c.order {
			w := c.waiters[id]
			if w != nil && w.gate.ThreadID == action.ThreadID && !c.shouldGateLocked(w.gate.Phase) {
				c.releaseWaiterLocked(id)
				return nil
			}
		}
		c.threadSteps[action.ThreadID]++
	}
	return nil
}

func (c *ExecutionController) waitingStatusLocked() []ExecutionWaitingStatus {
	var out []ExecutionWaitingStatus
	for _, id := range c.order {
		if w := c.waiters[id]; w != nil {
			out = append(out, ExecutionWaitingStatus{ThreadID: w.gate.ThreadID, Scope: c.blockingScopeLocked(w.gate), Phase: string(w.gate.Phase)})
		}
	}
	return out
}
