package core

import "testing"

func TestToolDispatchArgsInjectsOpaqueThreadForAptevaApps(t *testing.T) {
	registry := NewToolRegistry("")
	registry.Register(&ToolDef{Name: "channels_send", MCP: true, MCPServer: "channels"})
	registry.Register(&ToolDef{Name: "tasks_create", MCP: true, MCPServer: "tasks", MCPApp: true})
	registry.Register(&ToolDef{Name: "other_call", MCP: true, MCPServer: "other"})
	thinker := &Thinker{registry: registry, threadID: "chat-conv-123"}
	original := map[string]string{"channel": "current", "text": "hello"}

	got := toolDispatchArgs(thinker, toolCall{Name: "channels_send", Args: original})
	if got["_apteva_caller_context"] != "chat-conv-123" {
		t.Fatalf("context=%q", got["_apteva_caller_context"])
	}
	if got["_apteva_caller_thread"] != "chat-conv-123" {
		t.Fatalf("thread=%q", got["_apteva_caller_thread"])
	}
	if _, leaked := original["_apteva_caller_context"]; leaked {
		t.Fatal("runtime context mutated model-visible arguments")
	}
	app := toolDispatchArgs(thinker, toolCall{Name: "tasks_create", Args: original})
	if app["_apteva_caller_thread"] != "chat-conv-123" {
		t.Fatalf("app thread=%q", app["_apteva_caller_thread"])
	}
	other := toolDispatchArgs(thinker, toolCall{Name: "other_call", Args: original})
	if _, leaked := other["_apteva_caller_thread"]; leaked {
		t.Fatal("runtime context leaked to unrelated MCP")
	}
}
