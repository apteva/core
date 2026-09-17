package core

import "time"

// InferenceHealth describes completed logical requests, including recovery and
// fallback. Transport retries do not inflate the consecutive failure count.
type InferenceHealth struct {
	ConsecutiveFailures     int        `json:"consecutive_failures"`
	LastSuccessfulInference *time.Time `json:"last_successful_inference,omitempty"`
	BlockedReason           string     `json:"blocked_reason,omitempty"`
}

func (t *Thinker) recordInferenceOutcome(err error) {
	t.inferenceMu.Lock()
	defer t.inferenceMu.Unlock()
	if err != nil {
		t.inferenceHealth.ConsecutiveFailures++
		t.inferenceHealth.BlockedReason = providerExecutionFailureReason(err)
		return
	}
	now := time.Now().UTC()
	t.inferenceHealth = InferenceHealth{LastSuccessfulInference: &now}
}

func (t *Thinker) inferenceSnapshot() InferenceHealth {
	t.inferenceMu.Lock()
	defer t.inferenceMu.Unlock()
	return t.inferenceHealth
}

func (t *Thinker) workflowHealthy() bool {
	if t == nil {
		return true
	}
	if t.inferenceSnapshot().ConsecutiveFailures > 0 {
		return false
	}
	if t.threads != nil {
		t.threads.mu.RLock()
		children := make([]*Thinker, 0, len(t.threads.threads))
		for _, thread := range t.threads.threads {
			children = append(children, thread.Thinker)
		}
		t.threads.mu.RUnlock()
		for _, child := range children {
			if !child.workflowHealthy() {
				return false
			}
		}
	}
	return true
}
