package core

import "testing"

func TestExecutionCheckpointRestoreTargetBeforeCurrentGate(t *testing.T) {
	store := NewExecutionCheckpointStore()
	th := &Thinker{
		threadID:  "main",
		iteration: 7,
		messages:  []Message{{Role: "system", Content: "test"}},
	}
	inputReady := store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseInputReady,
		Iteration: 7,
		Summary:   "Input ready: 1 events, 0 tool results",
	})
	store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseLLMStart,
		Iteration: 7,
		Summary:   "Calling model",
	})
	store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseLLMDone,
		Iteration: 7,
		Summary:   "Returned 1 tool calls: pushover_send_notification",
	})
	store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseToolBefore,
		Iteration: 7,
		Tool:      "pushover_send_notification",
		CallID:    "call-1",
		Summary:   "Sending push",
	})

	target := store.RestoreTargetBeforeGate(ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseToolBefore,
		Iteration: 7,
		Tool:      "pushover_send_notification",
		CallID:    "call-1",
	})
	if target == nil {
		t.Fatal("expected restore target")
	}
	if target.ID != inputReady.ID {
		t.Fatalf("target = %s/%s, want input.ready checkpoint %s", target.ID, target.Phase, inputReady.ID)
	}
}

func TestExecutionCheckpointRestoreTargetFallsBackToLLMStart(t *testing.T) {
	store := NewExecutionCheckpointStore()
	th := &Thinker{
		threadID:  "main",
		iteration: 7,
		messages:  []Message{{Role: "system", Content: "test"}},
	}
	llmStart := store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseLLMStart,
		Iteration: 7,
		Summary:   "Calling model",
	})
	store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseLLMDone,
		Iteration: 7,
		Summary:   "Returned 1 tool calls: pushover_send_notification",
	})
	store.Capture(th, ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseToolBefore,
		Iteration: 7,
		Tool:      "pushover_send_notification",
		CallID:    "call-1",
		Summary:   "Sending push",
	})

	target := store.RestoreTargetBeforeGate(ExecutionGate{
		ThreadID:  "main",
		Phase:     ExecutionPhaseToolBefore,
		Iteration: 7,
		Tool:      "pushover_send_notification",
		CallID:    "call-1",
	})
	if target == nil {
		t.Fatal("expected restore target")
	}
	if target.ID != llmStart.ID {
		t.Fatalf("target = %s/%s, want llm.start checkpoint %s", target.ID, target.Phase, llmStart.ID)
	}
}
