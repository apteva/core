package core

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

type ExecutionControlMode string

const (
	ExecutionAuto   ExecutionControlMode = "auto"
	ExecutionPaused ExecutionControlMode = "paused"
	ExecutionStep   ExecutionControlMode = "step"
)

type ExecutionPhase string

const (
	ExecutionPhaseIterationStart ExecutionPhase = "iteration.start"
	ExecutionPhaseInputReady     ExecutionPhase = "input.ready"
	ExecutionPhaseLLMStart       ExecutionPhase = "llm.start"
	ExecutionPhaseLLMDone        ExecutionPhase = "llm.done"
	ExecutionPhaseToolBefore     ExecutionPhase = "tool.before"
	ExecutionPhaseToolAfter      ExecutionPhase = "tool.after"
	ExecutionPhaseIterationDone  ExecutionPhase = "iteration.done"
	ExecutionPhaseSleepBefore    ExecutionPhase = "sleep.before"
)

type ExecutionControlConfig struct {
	Mode        ExecutionControlMode `json:"mode,omitempty"`
	Scope       string               `json:"scope,omitempty"`
	Breakpoints []string             `json:"breakpoints,omitempty"`
	Follow      string               `json:"follow,omitempty"`
}

type ExecutionControlStatus struct {
	Mode                ExecutionControlMode `json:"mode"`
	Scope               string               `json:"scope"`
	Breakpoints         []string             `json:"breakpoints"`
	Follow              string               `json:"follow"`
	Waiting             bool                 `json:"waiting"`
	Phase               string               `json:"phase,omitempty"`
	ActiveThreadID      string               `json:"active_thread_id,omitempty"`
	Iteration           int                  `json:"iteration,omitempty"`
	Tool                string               `json:"tool,omitempty"`
	CallID              string               `json:"call_id,omitempty"`
	Summary             string               `json:"summary,omitempty"`
	Args                map[string]string    `json:"args,omitempty"`
	WaitingCount        int                  `json:"waiting_count,omitempty"`
	CanRestore          bool                 `json:"can_restore,omitempty"`
	RestoreCheckpointID string               `json:"restore_checkpoint_id,omitempty"`
	RestoreSummary      string               `json:"restore_summary,omitempty"`
	RestorePhase        string               `json:"restore_phase,omitempty"`
}

type ExecutionPhaseData struct {
	ThreadID       string            `json:"thread_id"`
	Iteration      int               `json:"iteration"`
	Phase          string            `json:"phase"`
	Tool           string            `json:"tool,omitempty"`
	CallID         string            `json:"call_id,omitempty"`
	Summary        string            `json:"summary,omitempty"`
	Args           map[string]string `json:"args,omitempty"`
	ResultPreview  string            `json:"result_preview,omitempty"`
	ParentThreadID string            `json:"parent_thread_id,omitempty"`
}

type ExecutionGate struct {
	ThreadID  string
	Phase     ExecutionPhase
	Iteration int
	Tool      string
	CallID    string
	Summary   string
	Args      map[string]string
	Result    string
	ParentID  string
}

type ExecutionControlAction struct {
	Action       string               `json:"action"`
	ThreadID     string               `json:"thread_id,omitempty"`
	CheckpointID string               `json:"checkpoint_id,omitempty"`
	Mode         ExecutionControlMode `json:"mode,omitempty"`
}

type executionWaiter struct {
	id      string
	gate    ExecutionGate
	release chan bool
}

type ExecutionController struct {
	mu           sync.Mutex
	mode         ExecutionControlMode
	scope        string
	follow       string
	breakpoints  map[ExecutionPhase]bool
	breakOrder   []string
	waiters      map[string]*executionWaiter
	order        []string
	activeWaitID string
	pendingSteps int
	seq          uint64
}

func NewExecutionController(cfg ExecutionControlConfig) *ExecutionController {
	c := &ExecutionController{
		waiters: map[string]*executionWaiter{},
	}
	c.ApplyConfig(cfg)
	return c
}

func defaultExecutionBreakpoints() []string {
	return []string{
		string(ExecutionPhaseLLMDone),
		string(ExecutionPhaseToolBefore),
		string(ExecutionPhaseToolAfter),
		string(ExecutionPhaseIterationDone),
	}
}

func normalizeExecutionMode(mode ExecutionControlMode) ExecutionControlMode {
	switch mode {
	case ExecutionPaused, ExecutionStep:
		return mode
	default:
		return ExecutionAuto
	}
}

func (c *ExecutionController) ApplyConfig(cfg ExecutionControlConfig) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.mode = normalizeExecutionMode(cfg.Mode)
	c.scope = cfg.Scope
	if c.scope == "" {
		c.scope = "instance"
	}
	c.follow = cfg.Follow
	if c.follow == "" {
		c.follow = "active"
	}
	bps := cfg.Breakpoints
	if len(bps) == 0 {
		bps = defaultExecutionBreakpoints()
	}
	c.breakpoints = map[ExecutionPhase]bool{}
	c.breakOrder = append(c.breakOrder[:0], bps...)
	for _, bp := range bps {
		c.breakpoints[ExecutionPhase(bp)] = true
	}
	if c.mode == ExecutionAuto {
		c.releaseAllLocked()
	}
}

func (c *ExecutionController) Config() ExecutionControlConfig {
	c.mu.Lock()
	defer c.mu.Unlock()
	return ExecutionControlConfig{
		Mode:        c.mode,
		Scope:       c.scope,
		Follow:      c.follow,
		Breakpoints: append([]string(nil), c.breakOrder...),
	}
}

func (c *ExecutionController) Status() ExecutionControlStatus {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := ExecutionControlStatus{
		Mode:         c.mode,
		Scope:        c.scope,
		Follow:       c.follow,
		Breakpoints:  append([]string(nil), c.breakOrder...),
		WaitingCount: len(c.waiters),
	}
	if c.activeWaitID != "" {
		if w := c.waiters[c.activeWaitID]; w != nil {
			st.Waiting = true
			st.ActiveThreadID = w.gate.ThreadID
			st.Phase = string(w.gate.Phase)
			st.Iteration = w.gate.Iteration
			st.Tool = w.gate.Tool
			st.CallID = w.gate.CallID
			st.Summary = w.gate.Summary
			if len(w.gate.Args) > 0 {
				st.Args = copyStringMap(w.gate.Args)
			}
		}
	}
	return st
}

func (c *ExecutionController) Control(action ExecutionControlAction) (ExecutionControlStatus, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch strings.ToLower(strings.TrimSpace(action.Action)) {
	case "run":
		c.mode = ExecutionAuto
		c.pendingSteps = 0
		c.releaseAllLocked()
	case "pause":
		c.mode = ExecutionPaused
	case "step":
		c.mode = ExecutionStep
		if !c.releaseOneLocked(action.ThreadID) {
			c.pendingSteps++
		}
	default:
		return c.statusLocked(), fmt.Errorf("action must be run, pause, or step")
	}
	return c.statusLocked(), nil
}

func (c *ExecutionController) ShouldGate(phase ExecutionPhase) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.shouldGateLocked(phase)
}

func (c *ExecutionController) Wait(gate ExecutionGate, quit <-chan struct{}, emit func(string, ExecutionPhaseData)) bool {
	c.mu.Lock()
	if !c.shouldGateLocked(gate.Phase) {
		c.mu.Unlock()
		return true
	}
	if c.pendingSteps > 0 {
		c.pendingSteps--
		c.mu.Unlock()
		if emit != nil {
			emit("execution.released", gate.phaseData())
		}
		return true
	}

	c.seq++
	id := fmt.Sprintf("gate-%d", c.seq)
	w := &executionWaiter{id: id, gate: gate, release: make(chan bool, 1)}
	c.waiters[id] = w
	c.order = append(c.order, id)
	c.activeWaitID = id
	c.mu.Unlock()

	if emit != nil {
		emit("execution.waiting", gate.phaseData())
	}

	select {
	case proceed := <-w.release:
		if proceed {
			if emit != nil {
				emit("execution.released", gate.phaseData())
			}
			return true
		}
		if emit != nil {
			emit("execution.cancelled", gate.phaseData())
		}
		return false
	case <-quit:
		c.removeWaiter(id)
		if emit != nil {
			emit("execution.cancelled", gate.phaseData())
		}
		return false
	}
}

func (c *ExecutionController) shouldGateLocked(phase ExecutionPhase) bool {
	if c.mode == ExecutionAuto {
		return false
	}
	if len(c.breakpoints) == 0 {
		return true
	}
	return c.breakpoints[phase]
}

func (c *ExecutionController) releaseOneLocked(threadID string) bool {
	if threadID != "" {
		for _, id := range c.order {
			w := c.waiters[id]
			if w != nil && w.gate.ThreadID == threadID {
				c.releaseWaiterLocked(id)
				return true
			}
		}
		return false
	}
	if c.activeWaitID != "" && c.waiters[c.activeWaitID] != nil {
		c.releaseWaiterLocked(c.activeWaitID)
		return true
	}
	for _, id := range c.order {
		if c.waiters[id] != nil {
			c.releaseWaiterLocked(id)
			return true
		}
	}
	return false
}

func (c *ExecutionController) CancelThread(threadID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if threadID == "" {
		threadID = "main"
	}
	for _, id := range c.order {
		w := c.waiters[id]
		if w != nil && w.gate.ThreadID == threadID {
			c.cancelWaiterLocked(id)
			return true
		}
	}
	return false
}

func (c *ExecutionController) releaseAllLocked() {
	for id := range c.waiters {
		c.releaseWaiterLocked(id)
	}
}

func (c *ExecutionController) releaseWaiterLocked(id string) {
	w := c.waiters[id]
	if w == nil {
		return
	}
	delete(c.waiters, id)
	w.release <- true
	c.compactOrderLocked()
	if c.activeWaitID == id {
		c.activeWaitID = ""
		for i := len(c.order) - 1; i >= 0; i-- {
			if c.waiters[c.order[i]] != nil {
				c.activeWaitID = c.order[i]
				break
			}
		}
	}
}

func (c *ExecutionController) cancelWaiterLocked(id string) {
	w := c.waiters[id]
	if w == nil {
		return
	}
	delete(c.waiters, id)
	w.release <- false
	c.compactOrderLocked()
	if c.activeWaitID == id {
		c.activeWaitID = ""
		for i := len(c.order) - 1; i >= 0; i-- {
			if c.waiters[c.order[i]] != nil {
				c.activeWaitID = c.order[i]
				break
			}
		}
	}
}

func (c *ExecutionController) removeWaiter(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.waiters, id)
	if c.activeWaitID == id {
		c.activeWaitID = ""
	}
	c.compactOrderLocked()
	if c.activeWaitID == "" {
		for i := len(c.order) - 1; i >= 0; i-- {
			if c.waiters[c.order[i]] != nil {
				c.activeWaitID = c.order[i]
				break
			}
		}
	}
}

func (c *ExecutionController) compactOrderLocked() {
	if len(c.order) == 0 {
		return
	}
	out := c.order[:0]
	for _, id := range c.order {
		if c.waiters[id] != nil {
			out = append(out, id)
		}
	}
	c.order = out
}

func (c *ExecutionController) statusLocked() ExecutionControlStatus {
	st := ExecutionControlStatus{
		Mode:         c.mode,
		Scope:        c.scope,
		Follow:       c.follow,
		Breakpoints:  append([]string(nil), c.breakOrder...),
		WaitingCount: len(c.waiters),
	}
	if c.activeWaitID != "" {
		if w := c.waiters[c.activeWaitID]; w != nil {
			st.Waiting = true
			st.ActiveThreadID = w.gate.ThreadID
			st.Phase = string(w.gate.Phase)
			st.Iteration = w.gate.Iteration
			st.Tool = w.gate.Tool
			st.CallID = w.gate.CallID
			st.Summary = w.gate.Summary
			if len(w.gate.Args) > 0 {
				st.Args = copyStringMap(w.gate.Args)
			}
		}
	}
	return st
}

func (g ExecutionGate) phaseData() ExecutionPhaseData {
	data := ExecutionPhaseData{
		ThreadID:       g.ThreadID,
		Iteration:      g.Iteration,
		Phase:          string(g.Phase),
		Tool:           g.Tool,
		CallID:         g.CallID,
		Summary:        g.Summary,
		Args:           sanitizeExecutionArgs(g.Args),
		ParentThreadID: g.ParentID,
	}
	if g.Result != "" {
		data.ResultPreview = truncateStr(g.Result, 1000)
	}
	return data
}

func sanitizeExecutionArgs(args map[string]string) map[string]string {
	if len(args) == 0 {
		return nil
	}
	out := make(map[string]string, len(args))
	for k, v := range args {
		if k == "_reason" {
			continue
		}
		out[k] = truncateStr(v, 500)
	}
	return out
}

func copyStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func toolSummary(name string, args map[string]string) string {
	if reason := strings.TrimSpace(args["_reason"]); reason != "" {
		return reason
	}
	if len(args) == 0 {
		return name
	}
	parts := make([]string, 0, len(args))
	for k, v := range args {
		if k == "_reason" {
			continue
		}
		parts = append(parts, k+"="+truncateStr(v, 80))
		if len(parts) == 3 {
			break
		}
	}
	if len(parts) == 0 {
		return name
	}
	return name + " " + strings.Join(parts, ", ")
}

func sleepSummary(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	return formatSleep(d)
}
