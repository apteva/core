package core

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

const maxExecutionCheckpoints = 100

type ExecutionCheckpointMeta struct {
	ID        string            `json:"id"`
	ThreadID  string            `json:"thread_id"`
	Iteration int               `json:"iteration"`
	Phase     string            `json:"phase"`
	Tool      string            `json:"tool,omitempty"`
	CallID    string            `json:"call_id,omitempty"`
	Summary   string            `json:"summary,omitempty"`
	Args      map[string]string `json:"args,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
}

type executionCheckpoint struct {
	ExecutionCheckpointMeta
	messages          []Message
	activeTools       map[string]bool
	activeToolAge     map[string]int
	rate              ThinkRate
	agentRate         ThinkRate
	agentSleep        time.Duration
	model             ModelTier
	agentModel        ModelTier
	agentReasoning    ReasoningLevel
	baselineModel     ModelTier
	baselineReasoning ReasoningLevel
	activeWork        bool
	directive         string
}

type ExecutionCheckpointStore struct {
	mu    sync.Mutex
	seq   uint64
	items []*executionCheckpoint
}

func NewExecutionCheckpointStore() *ExecutionCheckpointStore {
	return &ExecutionCheckpointStore{}
}

func (s *ExecutionCheckpointStore) Capture(t *Thinker, gate ExecutionGate) *ExecutionCheckpointMeta {
	if s == nil || t == nil || !isRestorableCheckpointPhase(gate.Phase) {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	s.seq++
	cp := &executionCheckpoint{
		ExecutionCheckpointMeta: ExecutionCheckpointMeta{
			ID:        fmt.Sprintf("chk-%d", s.seq),
			ThreadID:  t.threadID,
			Iteration: t.iteration,
			Phase:     string(gate.Phase),
			Tool:      gate.Tool,
			CallID:    gate.CallID,
			Summary:   gate.Summary,
			Args:      sanitizeExecutionArgs(gate.Args),
			CreatedAt: time.Now(),
		},
		messages:          cloneMessages(t.messages),
		activeTools:       copyBoolMap(t.activeTools),
		activeToolAge:     copyIntMap(t.activeToolAge),
		rate:              t.rate,
		agentRate:         t.agentRate,
		agentSleep:        t.agentSleep,
		model:             t.model,
		agentModel:        t.agentModel,
		agentReasoning:    t.agentReasoning,
		baselineModel:     t.baselineModel,
		baselineReasoning: t.baselineReasoning,
		activeWork:        t.activeWork,
		directive:         t.directive,
	}
	s.items = append(s.items, cp)
	if len(s.items) > maxExecutionCheckpoints {
		s.items = s.items[len(s.items)-maxExecutionCheckpoints:]
	}
	meta := cp.ExecutionCheckpointMeta
	return &meta
}

func (s *ExecutionCheckpointStore) ListMeta() []ExecutionCheckpointMeta {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ExecutionCheckpointMeta, len(s.items))
	for i, cp := range s.items {
		out[i] = cp.ExecutionCheckpointMeta
		out[i].Args = copyStringMap(cp.Args)
	}
	return out
}

func (s *ExecutionCheckpointStore) RestoreTarget(threadID string) *ExecutionCheckpointMeta {
	if s == nil || threadID == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.items) - 1; i >= 0; i-- {
		cp := s.items[i]
		if cp.ThreadID == threadID {
			meta := cp.ExecutionCheckpointMeta
			meta.Args = copyStringMap(cp.Args)
			return &meta
		}
	}
	return nil
}

func (s *ExecutionCheckpointStore) RestoreTargetBeforeGate(gate ExecutionGate) *ExecutionCheckpointMeta {
	if s == nil || gate.ThreadID == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	latestForThread := -1
	current := -1
	for i := len(s.items) - 1; i >= 0; i-- {
		cp := s.items[i]
		if cp.ThreadID != gate.ThreadID {
			continue
		}
		if latestForThread == -1 {
			latestForThread = i
		}
		if cp.Iteration == gate.Iteration &&
			cp.Phase == string(gate.Phase) &&
			cp.CallID == gate.CallID &&
			cp.Tool == gate.Tool {
			current = i
			break
		}
	}
	target := latestForThread
	if current > 0 {
		if gate.Phase == ExecutionPhaseLLMDone || gate.Phase == ExecutionPhaseToolBefore {
			for i := current - 1; i >= 0; i-- {
				cp := s.items[i]
				if cp.ThreadID == gate.ThreadID &&
					cp.Iteration == gate.Iteration &&
					cp.Phase == string(ExecutionPhaseInputReady) {
					target = i
					meta := s.items[target].ExecutionCheckpointMeta
					meta.Args = copyStringMap(s.items[target].Args)
					return &meta
				}
			}
		}
		target = -1
		for i := current - 1; i >= 0; i-- {
			cp := s.items[i]
			if cp.ThreadID != gate.ThreadID {
				continue
			}
			if gate.Phase == ExecutionPhaseToolBefore &&
				cp.Iteration == gate.Iteration &&
				cp.Phase == string(ExecutionPhaseLLMDone) {
				continue
			}
			if gate.Phase == ExecutionPhaseLLMDone &&
				cp.Iteration == gate.Iteration &&
				cp.Phase == string(ExecutionPhaseLLMDone) {
				continue
			}
			target = i
			break
		}
	}
	if target < 0 {
		return nil
	}
	meta := s.items[target].ExecutionCheckpointMeta
	meta.Args = copyStringMap(s.items[target].Args)
	return &meta
}

func (s *ExecutionCheckpointStore) Get(id string) (*executionCheckpoint, bool) {
	if s == nil || id == "" {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, cp := range s.items {
		if cp.ID == id {
			return cp.clone(), true
		}
	}
	return nil, false
}

func (cp *executionCheckpoint) clone() *executionCheckpoint {
	if cp == nil {
		return nil
	}
	out := *cp
	out.Args = copyStringMap(cp.Args)
	out.messages = cloneMessages(cp.messages)
	out.activeTools = copyBoolMap(cp.activeTools)
	out.activeToolAge = copyIntMap(cp.activeToolAge)
	return &out
}

func isRestorableCheckpointPhase(phase ExecutionPhase) bool {
	switch phase {
	case ExecutionPhaseInputReady, ExecutionPhaseLLMStart, ExecutionPhaseLLMDone, ExecutionPhaseToolBefore, ExecutionPhaseToolAfter, ExecutionPhaseIterationDone:
		return true
	default:
		return false
	}
}

func cloneMessages(in []Message) []Message {
	if len(in) == 0 {
		return nil
	}
	out := make([]Message, len(in))
	for i, m := range in {
		out[i] = m
		out[i].Parts = cloneContentParts(m.Parts)
		out[i].ToolCalls = cloneToolCalls(m.ToolCalls)
		out[i].ToolResults = cloneToolResults(m.ToolResults)
		out[i].EventIDs = append([]string(nil), m.EventIDs...)
	}
	return out
}

func cloneContentParts(in []ContentPart) []ContentPart {
	if len(in) == 0 {
		return nil
	}
	out := make([]ContentPart, len(in))
	copy(out, in)
	return out
}

func cloneToolCalls(in []NativeToolCall) []NativeToolCall {
	if len(in) == 0 {
		return nil
	}
	out := make([]NativeToolCall, len(in))
	for i, tc := range in {
		out[i] = tc
		out[i].Args = copyStringMap(tc.Args)
		out[i].CanonicalArgs = append(json.RawMessage(nil), tc.CanonicalArgs...)
	}
	return out
}

func cloneToolResults(in []ToolResult) []ToolResult {
	if len(in) == 0 {
		return nil
	}
	out := make([]ToolResult, len(in))
	for i, tr := range in {
		out[i] = tr
		if tr.Image != nil {
			out[i].Image = append([]byte(nil), tr.Image...)
		}
	}
	return out
}

func copyBoolMap(in map[string]bool) map[string]bool {
	if len(in) == 0 {
		return map[string]bool{}
	}
	out := make(map[string]bool, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyIntMap(in map[string]int) map[string]int {
	if len(in) == 0 {
		return map[string]int{}
	}
	out := make(map[string]int, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
