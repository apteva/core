package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestCodexExplicitUTCWeekdayControlsFridayWorkSmoke verifies the behavioral
// purpose of the explicit calendar fields: Codex performs Friday-only work on
// Friday and does not perform it on the adjacent Saturday.
//
//	RUN_CODEX_WEEKDAY_SMOKE=1 go test -v -run TestCodexExplicitUTCWeekdayControlsFridayWorkSmoke -timeout 5m .
func TestCodexExplicitUTCWeekdayControlsFridayWorkSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_WEEKDAY_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_WEEKDAY_SMOKE=1 to run the Codex weekday smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex weekday smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	provider := NewOpenAICodexProvider(token)
	model := provider.Models()[ModelLarge]
	tool := NativeTool{
		Name:        "deliver_friday_report",
		Description: "Deliver the weekly report. This must be called only when the supplied current UTC weekday is Friday.",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{},
		},
	}

	for _, test := range []struct {
		name, now string
		wantCall  bool
	}{
		{name: "friday due", now: "2026-08-28T12:00:00Z", wantCall: true},
		{name: "saturday not due", now: "2026-08-29T12:00:00Z", wantCall: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			messages := appendEphemeralTurnContext([]Message{
				{Role: "system", Content: strings.Join([]string{
					"You own one weekly reporting responsibility.",
					"When the supplied current UTC weekday is Friday, call deliver_friday_report exactly once.",
					"On every other weekday, do not call it and reply exactly NOT_DUE.",
					"Use the explicit UTC weekday in CURRENT TIME; do not calculate the weekday yourself.",
				}, "\n")},
				{Role: "user", Content: "Assess whether the weekly report is due now."},
			}, "", test.now, false)
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			response, err := provider.Chat(ctx, messages, model, []NativeTool{tool}, nil, nil, nil)
			if err != nil {
				t.Fatalf("Codex weekday decision: %v", err)
			}
			calls := 0
			for _, call := range response.ToolCalls {
				if call.Name == tool.Name {
					calls++
				}
			}
			if test.wantCall && calls != 1 {
				t.Fatalf("Friday calls=%d want 1; text=%q tools=%#v", calls, response.Text, response.ToolCalls)
			}
			if !test.wantCall && calls != 0 {
				t.Fatalf("non-Friday called report tool: %#v", response.ToolCalls)
			}
			if !test.wantCall && !strings.Contains(response.Text, "NOT_DUE") {
				t.Fatalf("non-Friday response=%q want NOT_DUE", response.Text)
			}
		})
	}
}
