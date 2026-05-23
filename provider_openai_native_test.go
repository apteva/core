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

func TestOpenAINativeChat_ComputerToolModes(t *testing.T) {
	tests := []struct {
		name             string
		mode             string
		providerName     string
		model            string
		wantComputerTool bool
	}{
		{name: "native gpt 5.5", model: "gpt-5.5", wantComputerTool: true},
		{name: "native gpt 5.4 mini", model: "gpt-5.4-mini", wantComputerTool: true},
		{name: "codex defaults custom", providerName: "openai-codex", model: "gpt-5.5", wantComputerTool: false},
		{name: "codex native override", providerName: "openai-codex", mode: "native", model: "gpt-5.5", wantComputerTool: true},
		{name: "custom fallback", mode: "custom", model: "gpt-5.5", wantComputerTool: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("APTEVA_OPENAI_COMPUTER_MODE", tt.mode)
			p := &OpenAINativeProvider{apiKey: "test", name: tt.providerName}
			tools := p.buildAPITools(tt.model, []NativeTool{
				{Name: "computer_use", Description: "screen", Parameters: map[string]any{"type": "object"}},
				{Name: "browser_session", Description: "session", Parameters: map[string]any{"type": "object"}},
			})

			var sawComputer, sawComputerFunction, sawBrowserSession bool
			for _, tool := range tools {
				b, _ := json.Marshal(tool)
				var decoded map[string]any
				if err := json.Unmarshal(b, &decoded); err != nil {
					t.Fatalf("tool JSON: %v", err)
				}
				if decoded["type"] == "computer" {
					sawComputer = true
				}
				if decoded["type"] == "function" {
					fn, _ := decoded["name"].(string)
					if fn == "computer_use" {
						sawComputerFunction = true
					}
					if fn == "browser_session" {
						sawBrowserSession = true
					}
				}
			}
			if sawComputer != tt.wantComputerTool {
				t.Fatalf("saw native computer tool = %v, want %v; tools=%#v", sawComputer, tt.wantComputerTool, tools)
			}
			if tt.wantComputerTool && sawComputerFunction {
				t.Fatalf("native mode should not also expose computer_use as a function: %#v", tools)
			}
			if !tt.wantComputerTool && !sawComputerFunction {
				t.Fatalf("custom mode should expose computer_use as a function: %#v", tools)
			}
			if !sawBrowserSession {
				t.Fatalf("browser_session should remain available so sessions can be opened: %#v", tools)
			}
		})
	}
}

func TestOpenAINativeBuildInput_ComputerImageOutputs(t *testing.T) {
	messages := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{
			ID:   "call_computer",
			Name: "computer_use",
			Args: map[string]string{"action": "screenshot"},
		}}},
		{Role: "user", ToolResults: []ToolResult{{
			CallID:  "call_computer",
			Content: "screenshot attached",
			Image:   []byte{0x89, 0x50, 0x4e, 0x47},
		}}},
	}

	nativeItems := (&OpenAINativeProvider{}).buildInput(messages)
	if len(nativeItems) != 2 {
		t.Fatalf("native items len = %d, want 2: %#v", len(nativeItems), nativeItems)
	}
	if nativeItems[0].Type != "computer_call" || nativeItems[0].Status != "completed" || nativeItems[0].Actions == nil {
		t.Fatalf("native replay item = %#v", nativeItems[0])
	}
	if nativeItems[1].Type != "computer_call_output" {
		t.Fatalf("native image result type = %q, want computer_call_output", nativeItems[1].Type)
	}
	out, ok := nativeItems[1].Output.(map[string]any)
	if !ok || out["type"] != "computer_screenshot" {
		t.Fatalf("native image output = %#v", nativeItems[1].Output)
	}

	t.Setenv("APTEVA_OPENAI_COMPUTER_MODE", "custom")
	customItems := (&OpenAINativeProvider{}).buildInput(messages)
	if customItems[0].Type != "function_call" || customItems[0].Name != "computer_use" {
		t.Fatalf("custom replay item = %#v", customItems[0])
	}
	if customItems[1].Type != "function_call_output" {
		t.Fatalf("custom image result type = %q, want function_call_output", customItems[1].Type)
	}
	blocks, ok := customItems[1].Output.([]oaiContentBlock)
	if !ok || len(blocks) != 2 || blocks[0].Type != "input_text" || blocks[1].Type != "input_image" {
		t.Fatalf("custom image output = %#v", customItems[1].Output)
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

func TestOpenAINativeStreamResponse_ComputerCallActions(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"cu_item_123","type":"computer_call","call_id":"call_cu_456"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"cu_item_123","type":"computer_call","call_id":"call_cu_456","actions":[{"type":"click","x":10,"y":20,"button":"left"},{"type":"type","text":"penguin"}]}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(resp.ToolCalls) != 2 {
		t.Fatalf("ToolCalls len = %d, want 2: %#v", len(resp.ToolCalls), resp.ToolCalls)
	}
	if resp.ToolCalls[0].ID != "call_cu_456" || resp.ToolCalls[0].Name != "computer_use" || resp.ToolCalls[0].Args["action"] != "click" || resp.ToolCalls[0].Args["coordinate"] != "[10, 20]" {
		t.Fatalf("first tool call = %#v", resp.ToolCalls[0])
	}
	if resp.ToolCalls[1].ID != "call_cu_456_1" || resp.ToolCalls[1].Args["action"] != "type" || resp.ToolCalls[1].Args["text"] != "penguin" {
		t.Fatalf("second tool call = %#v", resp.ToolCalls[1])
	}
}

func TestOpenAINativeStreamResponse_LegacyComputerCallAction(t *testing.T) {
	stream := strings.NewReader(strings.Join([]string{
		`data: {"type":"response.output_item.added","item":{"id":"cu_item_123","type":"computer_call","call_id":"call_cu_456"}}`,
		`data: {"type":"response.output_item.done","item":{"id":"cu_item_123","type":"computer_call","call_id":"call_cu_456","action":{"type":"screenshot"}}}`,
		`data: [DONE]`,
		``,
	}, "\n"))

	resp, err := (&OpenAINativeProvider{}).streamResponse(stream, nil, nil, nil)
	if err != nil {
		t.Fatalf("streamResponse: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Args["action"] != "screenshot" {
		t.Fatalf("ToolCalls = %#v", resp.ToolCalls)
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
