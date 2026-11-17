package core

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// RUN_LLM_INTEGRATION_TESTS=1 VENICE_API_KEY=... go test -run
// '^TestIntegration_VeniceLongToolRunRecovery$' -count=1 -timeout 6m -v .
// A deliberately smaller Core budget forces the real recovery path without
// depending on the provider's current maximum context window.
func TestIntegration_VeniceLongToolRunRecovery(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_LLM_INTEGRATION_TESTS") != "1" {
		t.Skip("set RUN_LLM_INTEGRATION_TESTS=1 to run paid live-provider tests")
	}
	key := strings.TrimSpace(os.Getenv("VENICE_API_KEY"))
	if key == "" {
		t.Skip("VENICE_API_KEY not set")
	}
	const model = "deepseek-v4-1-flash"
	p := NewVeniceProvider(key)
	applyModelOverrides(p, map[string]string{"large": model, "medium": model, "small": model})
	previous, hadPrevious := capabilitiesForModel(model)
	registerModelCapabilities(map[string]ModelCapabilities{model: {ContextWindow: 65536}})
	t.Cleanup(func() {
		runtimeModelCapabilities.Lock()
		defer runtimeModelCapabilities.Unlock()
		if hadPrevious {
			runtimeModelCapabilities.models[model] = previous
		} else {
			delete(runtimeModelCapabilities.models, model)
		}
	})
	for _, wide := range []bool{false, true} {
		t.Run(map[bool]string{false: "completed-tool-run", true: "wide-json-result"}[wide], func(t *testing.T) {
			messages := longToolRunFixture()
			if wide {
				rows := make([]map[string]any, 1600)
				for i := range rows {
					rows[i] = map[string]any{"record": i, "observation": strings.Repeat("reviewed record; no external action. ", 8)}
				}
				raw, err := json.Marshal(rows)
				if err != nil {
					t.Fatal(err)
				}
				messages = messages[:5]
				messages[4].ToolResults[0].Content = string(raw)
			} else {
				// Each individual result is below the bulky-payload threshold;
				// only summarizing completed exchanges can make this fit.
				more := cloneMessages(messages[3:])
				for i := range more {
					for j := range more[i].ToolCalls {
						more[i].ToolCalls[j].ID = "more-" + more[i].ToolCalls[j].ID
					}
					for j := range more[i].ToolResults {
						more[i].ToolResults[j].CallID = "more-" + more[i].ToolResults[j].CallID
					}
				}
				messages = append(messages, more...)
			}
			messages[0].Content += " This is a synthetic read-only record review. Do not call tools. Return a brief final status naming the existing post and pending release; do not claim any new external action."
			for i := range messages {
				if len(messages[i].ToolCalls) > 0 {
					messages[i].Reasoning = "Inspect the record before reporting status."
				}
			}
			before := estimatePreparedRequest(p.Name(), model, messages, nil)
			if !before.OverBudget {
				t.Fatal("fixture did not force context recovery")
			}
			th := recoveryTestThinker(t, p, messages)
			ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
			defer cancel()
			response, err := th.callLLMWithRetryMessages(ctx, messages)
			if err != nil {
				t.Fatal(err)
			}
			if len(response.ToolCalls) != 0 || !strings.Contains(response.Text, "POST-881") || !strings.Contains(strings.ToLower(response.Text), "pending") {
				t.Fatal("model did not finish with the preserved release state")
			}
			if err := validateToolHistory(th.messages); err != nil {
				t.Fatal(err)
			}
			after := estimatePreparedRequest(p.Name(), model, th.messages, nil)
			if after.OverBudget || after.InputTokens >= before.InputTokens {
				t.Fatal("recovery did not produce a smaller request within budget")
			}
			loaded, _ := th.session.LoadTail(200)
			if !th.session.EventIDs()["long-run-event"] || !strings.Contains(fmt.Sprint(loaded), messages[2].Content) {
				t.Fatal("checkpoint lost the current instruction or execution correlation")
			}
			events, _ := th.telemetry.StoredEvents(0)
			recoveries := 0
			for _, event := range events {
				if event.Type == "llm.context_recovery" {
					recoveries++
				}
			}
			if recoveries == 0 {
				t.Fatal("live test bypassed recovery")
			}
			t.Logf("%s: recovered %d → %d estimated input tokens; final reply and durable checkpoint verified", model, before.InputTokens, after.InputTokens)
		})
	}
}
