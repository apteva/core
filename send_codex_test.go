package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestCodexSendReceiptContinuesWithoutDuplicateSmoke forces a conversation
// thread's first turn to hand work to main, then delegates the receipt turn to
// real Codex. It proves the sender sees the delivery receipt immediately and
// does not mistake it for main's completed response or resend the handoff.
func TestCodexSendReceiptContinuesWithoutDuplicateSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_SEND_RECEIPT_SMOKE") == "" {
		t.Skip("set RUN_CODEX_SEND_RECEIPT_SMOKE=1 to run the Codex send receipt smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex send receipt smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Chdir(t.TempDir())
	provider := &forcedFirstResponseProvider{
		LLMProvider: NewOpenAICodexProvider(token),
		response: ChatResponse{ToolCalls: []NativeToolCall{{
			ID:   "forced-send-main",
			Name: "send",
			Args: map[string]string{
				"id":      "main",
				"message": "Please configure the durable daily check-in policy and reply to this thread with the result.",
				"_reason": "Handing off durable schedule",
			},
		}}},
	}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "# Role\nCoordinate durable operator work.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	if err := thinker.threads.SpawnWithOpts(
		"chat-test",
		"# Role\nHandle a user conversation. Hand durable work to main and wait for main's actual reply before confirming completion.",
		[]string{"send", "pace"},
		SpawnOpts{DeferRun: true},
	); err != nil {
		t.Fatalf("spawn user-facing event thread: %v", err)
	}
	thinker.drainEventTexts() // discard the thread-started event on main
	worker := thinker.threads.threads["chat-test"]
	// Make the semantic distinction measurable: without send's completion kick,
	// the receipt-processing turn would not run within this test's deadline.
	worker.Thinker.agentSleep = 6 * time.Hour
	started := time.Now()
	thinker.bus.Publish(Event{
		Type: EventInbox,
		From: "operator",
		To:   "chat-test",
		Text: "[console] Arrange the durable daily check-in through main. Do not tell me it is complete until main replies with the result.",
	})
	go worker.Thinker.Run()

	deadline := time.Now().Add(90 * time.Second)
	seenEventIDs := map[string]bool{}
	sendCalls := 0
	seenReceipt := false
	llmTurns := 0
	var receiptAt, receiptTurnAt time.Time
	var receiptTurnMessage string
	for time.Now().Before(deadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != "chat-test" || event.Time.Before(started) || seenEventIDs[event.ID] {
				continue
			}
			seenEventIDs[event.ID] = true
			switch event.Type {
			case "tool.call":
				var data ToolCallData
				if json.Unmarshal(event.Data, &data) == nil && data.Name == "send" {
					sendCalls++
					if sendCalls > 1 {
						t.Fatalf("Codex resent after a delivery receipt: calls=%d args=%v", sendCalls, data.Args)
					}
				}
			case "tool.result":
				var data ToolResultData
				if json.Unmarshal(event.Data, &data) == nil && data.Name == "send" &&
					strings.Contains(data.Result, "delivery only, not completion") {
					seenReceipt = true
					receiptAt = event.Time
				}
			case "llm.done":
				llmTurns++
				if llmTurns == 2 {
					receiptTurnAt = event.Time
					var data LLMDoneData
					if json.Unmarshal(event.Data, &data) == nil {
						receiptTurnMessage = strings.TrimSpace(data.Message)
					}
				}
			}
		}
		if seenReceipt && llmTurns >= 2 {
			if receiptAt.IsZero() || receiptTurnAt.IsZero() || receiptTurnAt.Before(receiptAt) {
				t.Fatalf("invalid receipt continuation order: receipt=%s second_turn=%s",
					receiptAt.Format(time.RFC3339Nano), receiptTurnAt.Format(time.RFC3339Nano))
			}
			continuationDelay := receiptTurnAt.Sub(receiptAt)
			if continuationDelay > 60*time.Second {
				t.Fatalf("send receipt continuation took %s, want an immediate turn rather than paced sleep", continuationDelay)
			}
			if receiptTurnMessage == "" {
				t.Fatal("Codex receipt-processing turn returned no feedback")
			}
			time.Sleep(1500 * time.Millisecond)
			events, _ = thinker.telemetry.StoredEvents(0)
			for _, event := range events {
				if event.ThreadID != "chat-test" || event.Type != "tool.call" || seenEventIDs[event.ID] {
					continue
				}
				var data ToolCallData
				if json.Unmarshal(event.Data, &data) == nil && data.Name == "send" {
					t.Fatalf("Codex resent after the receipt-processing turn: args=%v", data.Args)
				}
			}
			if sendCalls != 1 {
				t.Fatalf("send calls=%d, want exactly one", sendCalls)
			}
			mainEvents := thinker.drainEventTexts()
			if len(mainEvents) != 1 || !strings.Contains(mainEvents[0], "[from:chat-test]") ||
				!strings.Contains(mainEvents[0], "durable daily check-in") {
				t.Fatalf("main inbox = %v", mainEvents)
			}
			t.Logf("Codex processed the send receipt after %s without duplicating the handoff; feedback=%q",
				continuationDelay.Round(time.Millisecond), receiptTurnMessage)
			return
		}
		time.Sleep(250 * time.Millisecond)
	}
	t.Fatalf("timed out: receipt=%v llm_turns=%d send_calls=%d", seenReceipt, llmTurns, sendCalls)
}
