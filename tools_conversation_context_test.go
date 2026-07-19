package core

import "testing"

func TestToolDispatchArgsInjectsConversationContextOnlyForChannels(t *testing.T) {
	registry := NewToolRegistry("")
	registry.Register(&ToolDef{Name: "channels_send", MCP: true, MCPServer: "channels"})
	registry.Register(&ToolDef{Name: "other_call", MCP: true, MCPServer: "other"})
	thinker := &Thinker{registry: registry, threadID: "chat-conv-123"}
	original := map[string]string{"channel": "current", "text": "hello"}

	got := toolDispatchArgs(thinker, toolCall{Name: "channels_send", Args: original})
	if got["_apteva_caller_context"] != "chat-conv-123" {
		t.Fatalf("context=%q", got["_apteva_caller_context"])
	}
	if _, leaked := original["_apteva_caller_context"]; leaked {
		t.Fatal("runtime context mutated model-visible arguments")
	}
	other := toolDispatchArgs(thinker, toolCall{Name: "other_call", Args: original})
	if _, leaked := other["_apteva_caller_context"]; leaked {
		t.Fatal("runtime context leaked to unrelated MCP")
	}
}
