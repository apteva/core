package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	evolveHandoffThreadID = "durable-requester"
	evolveHandoffTool     = "deliver_durable_confirmation"
)

// scriptedEvolveHandoffProvider makes the workflow deterministic for the
// always-on integration test. The same harness is also run against real Codex
// below, so Core mechanics and model behavior are covered separately.
type scriptedEvolveHandoffProvider struct{}

func (p *scriptedEvolveHandoffProvider) Chat(_ context.Context, messages []Message, _ string, _ []NativeTool, _ func(string), _ func(string), _ func(string, string, string)) (ChatResponse, error) {
	serialized, _ := json.Marshal(messages)
	contextText := string(serialized)
	isWorker := messageContentContains(messages, `SUB-THREAD (id="`+evolveHandoffThreadID+`")`)
	if isWorker {
		switch {
		case hasToolResultNamed(messages, evolveHandoffTool):
			return ChatResponse{Text: "The authoritative result was delivered; I can now wait.", ToolCalls: []NativeToolCall{{
				ID: "worker-wait", Name: "pace", Args: map[string]string{"sleep": "1h"},
			}}}, nil
		case strings.Contains(contextText, `[from:main] Durable policy updated`):
			return ChatResponse{Text: "I received main's authoritative completion.", ToolCalls: []NativeToolCall{{
				ID: "deliver-confirmation", Name: evolveHandoffTool,
				Args: map[string]string{"message": "Daily check-in policy saved for 09:00 UTC."},
			}}}, nil
		case hasToolResultNamed(messages, "send"):
			return ChatResponse{Text: "Waiting for main's actual result."}, nil
		default:
			return ChatResponse{Text: "I will ask main to persist the durable policy.", ToolCalls: []NativeToolCall{{
				ID: "request-main-evolve", Name: "send",
				Args: map[string]string{
					"id":      "main",
					"message": `The operator requests a durable daily policy: at 09:00 UTC send exactly "Daily check-in." Please persist it and reply to durable-requester with the authoritative result.`,
				},
			}}}, nil
		}
	}

	switch {
	case hasToolResultNamed(messages, "send"):
		return ChatResponse{Text: "The requester was notified; I can now wait.", ToolCalls: []NativeToolCall{{
			ID: "main-wait", Name: "pace", Args: map[string]string{"sleep": "1h"},
		}}}, nil
	case hasToolResultNamed(messages, "evolve"):
		return ChatResponse{Text: "The durable edit succeeded; I will reply to the requester.", ToolCalls: []NativeToolCall{{
			ID: "confirm-requester", Name: "send",
			Args: map[string]string{
				"id":      evolveHandoffThreadID,
				"message": "Durable policy updated: every day at 09:00 UTC send exactly Daily check-in.",
			},
		}}}, nil
	default:
		return ChatResponse{Text: "I will persist the authenticated operator instruction.", ToolCalls: []NativeToolCall{{
			ID: "persist-daily-policy", Name: "evolve",
			Args: map[string]string{
				"edit_mode": "section_append",
				"section":   "Schedule",
				"content":   `- Every day at 09:00 UTC, send exactly "Daily check-in."`,
			},
		}}}, nil
	}
}

func messageContentContains(messages []Message, fragment string) bool {
	for _, message := range messages {
		if strings.Contains(message.Content, fragment) {
			return true
		}
	}
	return false
}

func hasToolResultNamed(messages []Message, name string) bool {
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if result.ToolName == name {
				return true
			}
		}
	}
	return false
}

func (p *scriptedEvolveHandoffProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: "scripted", ModelMedium: "scripted", ModelSmall: "scripted"}
}
func (p *scriptedEvolveHandoffProvider) Name() string { return "scripted-evolve-handoff" }
func (p *scriptedEvolveHandoffProvider) CostPer1M() (float64, float64, float64) {
	return 0, 0, 0
}
func (p *scriptedEvolveHandoffProvider) SupportsNativeTools() bool            { return true }
func (p *scriptedEvolveHandoffProvider) AvailableBuiltinTools() []BuiltinTool { return nil }
func (p *scriptedEvolveHandoffProvider) SetBuiltinTools([]string)             {}
func (p *scriptedEvolveHandoffProvider) WithBuiltins([]string) LLMProvider    { return p }

func TestIntegration_SubthreadMainEvolveConfirmationWorkflow(t *testing.T) {
	runSubthreadMainEvolveConfirmationWorkflow(t, &scriptedEvolveHandoffProvider{}, 20*time.Second)
}

// TestCodexSubthreadMainEvolveConfirmationWorkflowSmoke validates the same
// complete handoff with real model decisions. The harness simulates the
// trusted server forwarding the authenticated owner instruction to main after
// the requester sends its correlated ordinary thread message. The ordinary
// thread message alone is intentionally not treated as evolution authority.
//
// Run:
//
//	RUN_CODEX_SUBTHREAD_EVOLVE_HANDOFF_SMOKE=1 go test -v -run TestCodexSubthreadMainEvolveConfirmationWorkflowSmoke -timeout 6m .
func TestCodexSubthreadMainEvolveConfirmationWorkflowSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_SUBTHREAD_EVOLVE_HANDOFF_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_SUBTHREAD_EVOLVE_HANDOFF_SMOKE=1 to run the Codex subthread evolve handoff smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex subthread evolve handoff smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runSubthreadMainEvolveConfirmationWorkflow(t, NewOpenAICodexProvider(token), 4*time.Minute)
}

func runSubthreadMainEvolveConfirmationWorkflow(t *testing.T, provider LLMProvider, timeout time.Duration) {
	t.Helper()
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	t.Chdir(t.TempDir())

	cfg := &Config{
		path: filepath.Join(t.TempDir(), "config.json"),
		Directive: strings.Join([]string{
			"# Role",
			"Coordinate durable operator responsibilities.",
			"",
			"# Operating Rules",
			"Reply to a waiting requester after durable work succeeds.",
		}, "\n"),
		Mode: ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	defer thinker.threads.KillAll()

	var deliveryMu sync.Mutex
	deliveryCalls := 0
	deliveryText := ""
	thinker.registry.Register(&ToolDef{
		Name:        evolveHandoffTool,
		Description: "Deliver main's completed durable-policy result to the waiting user. Call exactly once and only after main replies.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"message": map[string]any{"type": "string", "description": "Completed result for the waiting user."},
			},
			"required": []string{"message"},
		},
		Handler: func(args map[string]string) ToolResponse {
			deliveryMu.Lock()
			defer deliveryMu.Unlock()
			deliveryCalls++
			deliveryText = args["message"]
			return ToolResponse{Text: `{"ok":true,"delivered":true}`}
		},
	})

	workerDirective := strings.Join([]string{
		"# Role",
		"Handle the waiting user's request and deliver the authoritative result.",
		"",
		"# Workflow",
		"For a durable policy request, send it to main exactly once. The send receipt is not completion: wait for main's actual reply without pacing or sending again.",
		"Do not evolve your own directive. When main replies that the policy was updated, call deliver_durable_confirmation exactly once with a concise user-facing confirmation.",
	}, "\n")
	if err := thinker.threads.SpawnWithOpts(
		evolveHandoffThreadID,
		workerDirective,
		[]string{"send", evolveHandoffTool, "pace"},
		SpawnOpts{DeferRun: true, ParentID: "main"},
	); err != nil {
		t.Fatalf("spawn durable requester: %v", err)
	}
	thinker.drainEventTexts() // discard the startup notice before the workflow
	worker := thinker.threads.threads[evolveHandoffThreadID]
	worker.Thinker.agentSleep = 6 * time.Hour

	started := time.Now()
	thinker.bus.Publish(Event{
		Type: EventInbox,
		From: "operator",
		To:   evolveHandoffThreadID,
		Text: `[console] From now on, every day at 09:00 UTC send exactly "Daily check-in." Ask main to persist this durable policy and wait for main's actual result before confirming it to me.`,
	})
	go worker.Thinker.Run()

	// Wait until the real requester thread has issued its handoff. Only then
	// simulate the server's trusted provenance path by forwarding the original
	// authenticated owner command to main as a console event.
	if !waitForEvolveHandoffToolCall(t, thinker.telemetry, started, evolveHandoffThreadID, "send", timeout/3) {
		t.Fatal("requester thread did not send durable work to main")
	}
	thinker.InjectConsole(strings.Join([]string{
		"Authenticated operator instruction correlated with requester durable-requester:",
		`From now on, every day at 09:00 UTC send exactly "Daily check-in."`,
		"Main owns this one lightweight recurring responsibility; persist it now, do not spawn, and send the successful result to durable-requester before pacing.",
	}, " "))
	go thinker.Run()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		deliveryMu.Lock()
		delivered := deliveryCalls > 0
		deliveryMu.Unlock()
		if delivered {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Allow receipt-processing continuations to expose accidental duplicates.
	time.Sleep(500 * time.Millisecond)
	events, _ := thinker.telemetry.StoredEvents(0)
	requestSends := 0
	evolveCalls := 0
	evolveSuccesses := 0
	mainReplies := 0
	workerDeliveries := 0
	var evolvedAt, repliedAt, deliveredAt time.Time
	for _, event := range events {
		if event.Time.Before(started) {
			continue
		}
		switch event.Type {
		case "tool.call":
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) != nil {
				continue
			}
			switch {
			case event.ThreadID == evolveHandoffThreadID && data.Name == "send" && data.Args["id"] == "main":
				requestSends++
			case event.ThreadID == "main" && data.Name == "evolve":
				evolveCalls++
			case event.ThreadID == "main" && data.Name == "send" && data.Args["id"] == evolveHandoffThreadID:
				mainReplies++
				repliedAt = event.Time
			case event.ThreadID == evolveHandoffThreadID && data.Name == evolveHandoffTool:
				workerDeliveries++
				deliveredAt = event.Time
			case data.Name == "pace":
				if event.ThreadID == "main" && evolvedAt.IsZero() {
					t.Errorf("main paced before completing evolve: args=%v", data.Args)
				}
				if event.ThreadID == evolveHandoffThreadID && repliedAt.IsZero() {
					t.Errorf("requester paced before main replied: args=%v", data.Args)
				}
			}
		case "tool.result":
			if event.ThreadID != "main" {
				continue
			}
			var data ToolResultData
			if json.Unmarshal(event.Data, &data) == nil && data.Name == "evolve" && data.Success && !strings.HasPrefix(data.Result, "error:") {
				evolveSuccesses++
			}
		case "directive.evolved":
			if event.ThreadID == "main" {
				evolvedAt = event.Time
			}
		}
	}

	deliveryMu.Lock()
	finalDeliveryCalls := deliveryCalls
	finalDeliveryText := deliveryText
	deliveryMu.Unlock()
	if requestSends != 1 || evolveCalls != 1 || evolveSuccesses != 1 || mainReplies != 1 || workerDeliveries != 1 || finalDeliveryCalls != 1 {
		t.Fatalf("handoff counts: requester_send=%d evolve_calls=%d evolve_successes=%d main_replies=%d worker_delivery_calls=%d handler_deliveries=%d directive=\n%s",
			requestSends, evolveCalls, evolveSuccesses, mainReplies, workerDeliveries, finalDeliveryCalls, cfg.GetDirective())
	}
	if evolvedAt.IsZero() || repliedAt.IsZero() || deliveredAt.IsZero() || repliedAt.Before(evolvedAt) || deliveredAt.Before(repliedAt) {
		t.Fatalf("invalid workflow order: evolved=%s replied=%s delivered=%s", evolvedAt, repliedAt, deliveredAt)
	}
	directive := cfg.GetDirective()
	if !strings.Contains(directive, "09:00 UTC") || !strings.Contains(directive, "Daily check-in") {
		t.Fatalf("main directive was not durably updated:\n%s", directive)
	}
	if !strings.Contains(strings.ToLower(finalDeliveryText), "09:00") || !strings.Contains(strings.ToLower(finalDeliveryText), "daily check-in") {
		t.Fatalf("requester delivered an incomplete confirmation: %q", finalDeliveryText)
	}
	t.Logf("complete evolve handoff passed: requester_send=%d evolve=%d main_reply=%d requester_delivery=%d", requestSends, evolveCalls, mainReplies, workerDeliveries)
}

func waitForEvolveHandoffToolCall(t *testing.T, telemetry *Telemetry, after time.Time, threadID, toolName string, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		events, _ := telemetry.StoredEvents(0)
		for _, event := range events {
			if event.Time.Before(after) || event.ThreadID != threadID || event.Type != "tool.call" {
				continue
			}
			var data ToolCallData
			if json.Unmarshal(event.Data, &data) == nil && data.Name == toolName {
				return true
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return false
}

var _ LLMProvider = (*scriptedEvolveHandoffProvider)(nil)
