package core

import (
	"testing"
	"time"
)

func TestToolCallTimeoutUsesConnectorDefaultAndOverride(t *testing.T) {
	call := toolCall{definition: &ToolDef{MCP: true, InputSchema: map[string]any{"x-apteva-timeout-ms": float64(240000)}}}
	if got := toolCallTimeout(call); got != 270*time.Second {
		t.Fatalf("connector timeout=%s, want 270s", got)
	}
	call.Args = map[string]string{"_apteva": `{"timeout_ms":300000}`}
	if got := toolCallTimeout(call); got != 330*time.Second {
		t.Fatalf("overridden timeout=%s, want 330s", got)
	}
	call.Args["_apteva"] = `{"timeout_ms":600000}`
	if got := toolCallTimeout(call); got != 10*time.Minute {
		t.Fatalf("capped timeout=%s, want 10m", got)
	}
}

func TestToolCallTimeoutKeepsOrdinaryDefault(t *testing.T) {
	if got := toolCallTimeout(toolCall{}); got != 3*time.Minute {
		t.Fatalf("ordinary timeout=%s", got)
	}
	call := toolCall{definition: &ToolDef{MCP: true, InputSchema: map[string]any{"x-apteva-timeout-ms": 30000}}}
	if got := toolCallTimeout(call); got != 3*time.Minute {
		t.Fatalf("short connector timeout=%s", got)
	}
}
