package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestIntegration_CodexHeavyWorkerKeepsStrongQualityBaseline verifies the
// model-facing half of the quality contract. A substantial delegated workload
// must not be assigned to a small/low worker; omitted profile fields are also
// valid because spawn defaults to large+auto.
//
//	RUN_CODEX_REASONING_BASELINE_SMOKE=1 go test -v -run TestIntegration_CodexHeavyWorkerKeepsStrongQualityBaseline -timeout 4m .
func TestIntegration_CodexHeavyWorkerKeepsStrongQualityBaseline(t *testing.T) {
	if os.Getenv("RUN_CODEX_REASONING_BASELINE_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_REASONING_BASELINE_SMOKE=1 to run the Codex quality-baseline smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex quality-baseline smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	registry := NewToolRegistry("test")
	var spawnTool NativeTool
	for _, tool := range registry.NativeTools(nil, nil) {
		if tool.Name == "spawn" {
			spawnTool = tool
			break
		}
	}
	if spawnTool.Name == "" {
		t.Fatal("spawn tool not found")
	}

	provider := NewOpenAICodexProvider(token)
	messages := []Message{
		{Role: "system", Content: buildSystemPrompt("# Role\nCoordinate complex analytical work safely.", ModeAutonomous, registry, "", nil, nil, nil, nil)},
		{Role: "user", Content: strings.Join([]string{
			"Create exactly one worker named revenue-risk-auditor; do not perform the audit on main.",
			"This is substantial work: reconcile 2,000 customer records across billing, CRM, support, and product-usage sources; investigate conflicting identities and amounts; use several read-only tools; preserve an evidence trail; identify financial and compliance risks; and synthesize an operator-facing report with confidence levels and unresolved ambiguities.",
			"The worker must remain active through all tool results and report only after the full cross-source reconciliation is complete.",
		}, " ")},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	response, err := provider.Chat(ctx, messages, provider.Models()[ModelLarge], []NativeTool{spawnTool}, nil, nil, nil)
	if err != nil {
		t.Fatalf("Codex heavy-worker selection: %v", err)
	}

	var calls []NativeToolCall
	for _, call := range response.ToolCalls {
		if call.Name == "spawn" {
			calls = append(calls, call)
		}
	}
	if len(calls) != 1 {
		t.Fatalf("spawn calls=%d want 1; text=%q tools=%#v", len(calls), response.Text, response.ToolCalls)
	}
	call := calls[0]
	if call.Args["id"] != "revenue-risk-auditor" {
		t.Fatalf("spawn id=%q want revenue-risk-auditor; args=%#v", call.Args["id"], call.Args)
	}

	model := strings.ToLower(strings.TrimSpace(call.Args["model"]))
	if model == "" {
		model = "large" // spawn's configured default
	}
	if model == "small" {
		t.Fatalf("Codex assigned substantial worker to small model: %#v", call.Args)
	}
	reasoning := strings.ToLower(reasoningArgValue(call.Args))
	if reasoning == "" {
		reasoning = "auto" // spawn's configured default
	}
	switch reasoning {
	case "none", "minimal", "low":
		t.Fatalf("Codex assigned substantial worker %s reasoning: %#v", reasoning, call.Args)
	}
	t.Logf("Codex heavy worker profile: model=%s reasoning=%s", model, reasoning)
}
