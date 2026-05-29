package core

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOpenAINativeBuildInput_SystemPromptHandling(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system rules"},
		{Role: "user", Content: "hello"},
	}

	regular := (&OpenAINativeProvider{}).buildInput(messages)
	if len(regular) != 2 {
		t.Fatalf("regular input len = %d, want 2", len(regular))
	}
	if regular[0].Role != "developer" || regular[0].Content != "system rules" {
		t.Fatalf("regular system item = %#v", regular[0])
	}

	codex := (&OpenAINativeProvider{forceStoreFalse: true}).buildInput(messages)
	if len(codex) != 1 {
		t.Fatalf("codex input len = %d, want 1", len(codex))
	}
	if codex[0].Role != "user" || codex[0].Content != "hello" {
		b, _ := json.Marshal(codex)
		t.Fatalf("codex input = %s", b)
	}
}

func TestOpenAINativeChat_BuildsFunctionToolsOnly(t *testing.T) {
	p := &OpenAINativeProvider{apiKey: "test", name: "openai-codex"}
	tools := p.buildAPITools("gpt-5.5", []NativeTool{
		{Name: "app_tool", Description: "tool", Parameters: map[string]any{"type": "object"}},
	})
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	b, _ := json.Marshal(tools[0])
	var decoded map[string]any
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("tool JSON: %v", err)
	}
	if decoded["type"] != "function" || decoded["name"] != "app_tool" {
		t.Fatalf("tool = %#v", decoded)
	}
}

func TestOpenAINativeRequestReasoning(t *testing.T) {
	codex := NewOpenAICodexProvider("token").(*OpenAINativeProvider)
	if got := codex.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "" {
		t.Fatalf("codex auto reasoning = %#v, want summary auto only", got)
	}

	high := codex.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).(*OpenAINativeProvider)
	if got := high.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "high" {
		t.Fatalf("codex high reasoning = %#v, want summary auto effort high", got)
	}
	minimal := codex.WithReasoning(ReasoningSettings{Level: ReasoningMinimal}).(*OpenAINativeProvider)
	if got := minimal.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "low" {
		t.Fatalf("codex minimal reasoning = %#v, want summary auto effort low", got)
	}
	xhigh := codex.WithReasoning(ReasoningSettings{Level: ReasoningXHigh}).(*OpenAINativeProvider)
	if got := xhigh.requestReasoning("gpt-5.5"); got == nil || got.Summary != "auto" || got.Effort != "xhigh" {
		t.Fatalf("codex xhigh reasoning = %#v, want summary auto effort xhigh", got)
	}

	openai := NewOpenAINativeProvider("key").(*OpenAINativeProvider)
	if got := openai.requestReasoning("gpt-5.4-mini"); got != nil {
		t.Fatalf("openai auto reasoning = %#v, want nil", got)
	}
	low := openai.WithReasoning(ReasoningSettings{Level: ReasoningLow}).(*OpenAINativeProvider)
	if got := low.requestReasoning("gpt-5.4-mini"); got == nil || got.Summary != "auto" || got.Effort != "low" {
		t.Fatalf("openai low reasoning = %#v, want summary auto effort low", got)
	}
	none := openai.WithReasoning(ReasoningSettings{Level: ReasoningNone}).(*OpenAINativeProvider)
	if got := none.requestReasoning("gpt-5.4-mini"); got == nil || got.Summary != "" || got.Effort != "none" {
		t.Fatalf("openai none reasoning = %#v, want effort none without summary", got)
	}
}

func TestOpenAINativeBuildInput_FunctionImageOutputs(t *testing.T) {
	messages := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{
			ID:   "call_image",
			Name: "app_tool",
			Args: map[string]string{"action": "capture"},
		}}},
		{Role: "user", ToolResults: []ToolResult{{
			CallID:  "call_image",
			Content: "screenshot attached",
			Image:   []byte{0x89, 0x50, 0x4e, 0x47},
		}}},
	}

	items := (&OpenAINativeProvider{}).buildInput(messages)
	if len(items) != 2 {
		t.Fatalf("items len = %d, want 2: %#v", len(items), items)
	}
	if items[0].Type != "function_call" || items[0].Name != "app_tool" {
		t.Fatalf("replay item = %#v", items[0])
	}
	if items[1].Type != "function_call_output" {
		t.Fatalf("image result type = %q, want function_call_output", items[1].Type)
	}
	blocks, ok := items[1].Output.([]oaiContentBlock)
	if !ok || len(blocks) != 2 || blocks[0].Type != "input_text" || blocks[1].Type != "input_image" {
		t.Fatalf("image output = %#v", items[1].Output)
	}
}

func TestOpenAINativeStreamResponse_ReasoningSummary(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.reasoning_summary_text.delta","delta":"checking "}`,
		`data: {"type":"response.reasoning_summary_text.delta","delta":"state"}`,
		`data: {"type":"response.output_text.delta","delta":"done"}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var thinking strings.Builder
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if resp.Text != "done" {
		t.Fatalf("Text = %q, want done", resp.Text)
	}
	if resp.Reasoning != "checking state" {
		t.Fatalf("Reasoning = %q, want checking state", resp.Reasoning)
	}
	if thinking.String() != resp.Reasoning {
		t.Fatalf("onThinking = %q, want %q", thinking.String(), resp.Reasoning)
	}
	if resp.Usage.PromptTokens != 3 || resp.Usage.CompletionTokens != 2 || resp.Usage.CachedTokens != 1 {
		t.Fatalf("Usage = %+v", resp.Usage)
	}
}

func TestOpenAINativeStreamResponse_ToolChunkUsesCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"id\":\"main\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send","arguments":"{\"id\":\"main\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunkTool, chunkID, chunk string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunkTool = toolName
		chunkID = callID
		chunk = argChunk
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if chunkTool != "send" {
		t.Fatalf("chunk tool = %q, want send", chunkTool)
	}
	if chunkID != "call_final_456" {
		t.Fatalf("chunk id = %q, want final call_id", chunkID)
	}
	if chunk != `{"id":"main"}` {
		t.Fatalf("chunk = %q", chunk)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	if resp.ToolCalls[0].ID != chunkID {
		t.Fatalf("final call id = %q, want chunk id %q", resp.ToolCalls[0].ID, chunkID)
	}
}

func TestOpenAINativeStreamResponse_ReasoningSummaryPartDone(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"summary only"}}`,
		`data: {"type":"response.output_text.delta","delta":"done"}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var thinking strings.Builder
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, func(chunk string) {
		thinking.WriteString(chunk)
	}, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if resp.Reasoning != "summary only" {
		t.Fatalf("Reasoning = %q, want summary only", resp.Reasoning)
	}
	if thinking.String() != resp.Reasoning {
		t.Fatalf("onThinking = %q, want %q", thinking.String(), resp.Reasoning)
	}
}

func TestOpenAINativeStreamResponse_BuffersToolChunkUntilCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call","name":"send"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"id\":\"main\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"send","arguments":"{\"id\":\"main\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks len = %d, want 1: %#v", len(chunks), chunks)
	}
	if chunks[0] != `send|call_final_456|{"id":"main"}` {
		t.Fatalf("chunk = %q", chunks[0])
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_final_456" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponse_BuffersToolChunkUntilNameAndCallID(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call"}}`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_item_123","delta":"{\"text\":\"hello\"}"}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"channels_respond","arguments":"{\"text\":\"hello\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks len = %d, want 1: %#v", len(chunks), chunks)
	}
	if chunks[0] != `channels_respond|call_final_456|{"text":"hello"}` {
		t.Fatalf("chunk = %q", chunks[0])
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "channels_respond" || resp.ToolCalls[0].ID != "call_final_456" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}

func TestOpenAINativeStreamResponse_FinalArgumentsEmitToolChunkWhenNoDeltas(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"fc_item_123","type":"function_call"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"fc_item_123","type":"function_call","call_id":"call_final_456","name":"channels_respond","arguments":"{\"text\":\"hello\"}"}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	var chunks []string
	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, func(toolName, callID, argChunk string) {
		chunks = append(chunks, toolName+"|"+callID+"|"+argChunk)
	})
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(chunks) != 1 || chunks[0] != `channels_respond|call_final_456|{"text":"hello"}` {
		t.Fatalf("chunks = %#v", chunks)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Args["text"] != "hello" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
	}
}
