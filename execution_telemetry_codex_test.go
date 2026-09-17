package core

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestCodexExecutionTelemetry(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_EXECUTION_TELEMETRY") != "1" {
		t.Skip("set RUN_CODEX_EXECUTION_TELEMETRY=1 for live request telemetry verification")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live test requires a valid Codex token")
	}
	th := retryTestThinker(NewOpenAICodexProvider(token))
	attachTraceTelemetry(th)
	th.messages = []Message{{Role: "system", Content: "Reply with one short sentence. No tools are needed."}, {Role: "user", Content: "Say hello."}}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	response, err := th.callLLMWithRetry(ctx)
	if err != nil {
		t.Fatal(err)
	}
	finished := traceEvents(t, th.telemetry, "llm.request.finished")
	if len(finished) != 1 || finished[0]["provider"] != "openai-codex" || finished[0]["outcome"] != "success" {
		t.Fatalf("finished: %+v", finished)
	}
	if response.Usage.PromptTokens == 0 || response.Usage.CompletionTokens == 0 {
		t.Fatal("missing usage")
	}
	outputs := traceEvents(t, th.telemetry, "llm.request.first_output")
	if len(outputs) == 0 {
		t.Fatal("missing first output")
	}
	httpDone := traceEvents(t, th.telemetry, "llm.http.finished")
	if len(httpDone) == 0 || httpDone[0]["request_id"] != finished[0]["request_id"] || outputs[0]["request_id"] != finished[0]["request_id"] {
		t.Fatal("HTTP/output correlation missing")
	}
	t.Logf("Real Codex model=%s input=%d output=%d HTTP attempts=%d", response.Model, response.Usage.PromptTokens, response.Usage.CompletionTokens, len(httpDone))
}
