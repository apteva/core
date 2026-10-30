package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAnthropicMaxTokens(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"claude-haiku-4-5-20251001", 64000},
		{"claude-sonnet-4-6", 64000},
		{"claude-opus-4-7", 64000},
		{"claude-fable-5", 64000},
		{"claude-mythos-5-preview", 64000},
		{"claude-opus-5-5", 64000},
		{"claude-sonnet-5-5", 64000},
		{"claude-haiku-5-5", 64000},
		{"claude-3-5-sonnet-20241022", 4096},
		{"claude-3-haiku-20240307", 4096},
		{"unknown-model", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := anthropicMaxTokens(tt.model); got != tt.want {
				t.Fatalf("anthropicMaxTokens(%q) = %d, want %d", tt.model, got, tt.want)
			}
		})
	}
}

func TestAnthropicDefaultsAndPricing(t *testing.T) {
	p := NewAnthropicProvider("test").(*AnthropicProvider)
	if m := p.Models(); m[ModelLarge] != "claude-opus-5-5" || m[ModelMedium] != "claude-sonnet-5-5" || m[ModelSmall] != "claude-haiku-5-5" {
		t.Fatalf("models=%v", m)
	}
	if in, cached, out := p.CostPer1M(); in != 4 || cached != 0.20 || out != 20 {
		t.Fatalf("opus 5.5 price=%v/%v/%v", in, cached, out)
	}
	applyModelOverrides(p, map[string]string{"large": "claude-haiku-4-5-20251001"})
	if in, _, out := p.CostPer1M(); in != 1 || out != 5 {
		t.Fatalf("dated haiku 4.5 price=%v/%v", in, out)
	}
	applyModelOverrides(p, map[string]string{"large": "some-unknown-model"})
	if in, _, out := p.CostPer1M(); in != 3 || out != 15 {
		t.Fatalf("fallback price=%v/%v", in, out)
	}
}

func TestAnthropicEffortMapping(t *testing.T) {
	tests := []struct {
		level ReasoningLevel
		model string
		want  string
	}{
		{ReasoningAuto, "claude-opus-5-5", ""},
		{ReasoningNone, "claude-opus-5-5", "low"},
		{ReasoningMinimal, "claude-sonnet-5-5", "low"},
		{ReasoningMedium, "claude-haiku-5-5", "medium"},
		{ReasoningXHigh, "claude-fable-5-1", "xhigh"},
		{ReasoningXHigh, "claude-opus-4-8", "xhigh"},
		{ReasoningXHigh, "claude-sonnet-4-6", "high"},
		{ReasoningHigh, "claude-haiku-4-5-20251001", ""},
		{ReasoningHigh, "claude-sonnet-4-5", ""},
		{ReasoningHigh, "gpt-5", ""},
	}
	for _, tt := range tests {
		if got := anthropicEffort(tt.level, tt.model); got != tt.want {
			t.Errorf("anthropicEffort(%s, %s)=%q want %q", tt.level, tt.model, got, tt.want)
		}
	}
}

// anthropicCapture is a fake Messages API that records each request and
// streams a scripted list of SSE events.
type anthropicCapture struct {
	bodies  []map[string]any
	headers []http.Header
}

func (c *anthropicCapture) server(t *testing.T, events ...string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body: %v", err)
		}
		c.bodies = append(c.bodies, body)
		c.headers = append(c.headers, r.Header.Clone())
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
		}
	}))
	t.Cleanup(server.Close)
	return server
}

var anthropicThinkingToolTurn = []string{
	`{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}`,
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Plan the lookup."}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-abc"}}`,
	`{"type":"content_block_stop","index":0}`,
	`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking."}}`,
	`{"type":"content_block_stop","index":1}`,
	`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`,
	`{"type":"content_block_stop","index":2}`,
	`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}`,
	`{"type":"message_stop"}`,
}

func TestAnthropicThinkingTurnReplaysUnchanged(t *testing.T) {
	var capture anthropicCapture
	server := capture.server(t, anthropicThinkingToolTurn...)
	p := &AnthropicProvider{apiKey: "test", url: server.URL}
	var thought strings.Builder
	first, err := p.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).Chat(context.Background(), []Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "find x"}}, "claude-opus-5-5", nil, nil, func(s string) { thought.WriteString(s) }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if capture.headers[0].Get("anthropic-beta") != anthropicThinkingBindingBeta {
		t.Fatalf("beta header=%q", capture.headers[0].Get("anthropic-beta"))
	}
	thinking, _ := json.Marshal(capture.bodies[0]["thinking"])
	if string(thinking) != `{"block_binding":{"prefix_mismatch_behavior":"drop_block"},"type":"adaptive"}` {
		t.Fatalf("thinking=%s", thinking)
	}
	if effort, _ := json.Marshal(capture.bodies[0]["output_config"]); string(effort) != `{"effort":"high"}` || first.EffectiveReasoningEffort != "high" {
		t.Fatalf("output_config=%s effective=%s", effort, first.EffectiveReasoningEffort)
	}
	if first.Reasoning != "Plan the lookup." || thought.String() != first.Reasoning || len(first.ToolCalls) != 1 {
		t.Fatalf("reasoning=%q calls=%+v", first.Reasoning, first.ToolCalls)
	}
	if first.ProviderState == nil || first.ProviderState.Provider != "anthropic" || len(first.ProviderState.Items) != 3 {
		t.Fatalf("state=%+v", first.ProviderState)
	}

	history := []Message{
		{Role: "user", Content: "find x"},
		{Role: "assistant", Content: first.Text, ToolCalls: first.ToolCalls, ProviderState: first.ProviderState},
		{Role: "user", ToolResults: []ToolResult{{CallID: "toolu_1", Content: "x=1"}}},
	}
	for _, tc := range []struct {
		name     string
		edit     func(*Message)
		thinking bool
	}{
		{"unchanged", func(*Message) {}, true},
		{"tool_calls_edited", func(m *Message) { m.ToolCalls = nil }, false},
		{"tool_name_edited", func(m *Message) { m.ToolCalls[0].Name = "other_lookup" }, false},
		{"tool_input_edited", func(m *Message) { m.ToolCalls[0].CanonicalArgs = json.RawMessage(`{"q":"changed"}`) }, false},
		{"text_edited", func(m *Message) { m.Content = "something else" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			msgs := cloneMessages(history)
			tc.edit(&msgs[1])
			if tc.name == "tool_calls_edited" {
				msgs = msgs[:2]
			}
			capture.bodies = nil
			if _, err := p.Chat(context.Background(), msgs, "claude-opus-5-5", nil, nil, nil, nil); err != nil {
				t.Fatal(err)
			}
			assistant, _ := json.Marshal(capture.bodies[0]["messages"].([]any)[1])
			hasThinking := strings.Contains(string(assistant), `"signature":"sig-abc"`)
			if hasThinking != tc.thinking {
				t.Fatalf("assistant=%s", assistant)
			}
			if tc.thinking && string(assistant) != `{"content":[{"signature":"sig-abc","thinking":"Plan the lookup.","type":"thinking"},{"text":"Checking.","type":"text"},{"id":"toolu_1","input":{"q":"x"},"name":"lookup","type":"tool_use"}],"role":"assistant"}` {
				t.Fatalf("replayed assistant changed: %s", assistant)
			}
		})
	}
}

func TestAnthropicOlderModelsAndServerToolsUnchanged(t *testing.T) {
	var capture anthropicCapture
	server := capture.server(t, anthropicThinkingToolTurn...)
	p := &AnthropicProvider{apiKey: "test", url: server.URL}
	resp, err := p.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "claude-haiku-4-5-20251001", nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := capture.bodies[0]["thinking"]; ok || capture.headers[0].Get("anthropic-beta") != "" {
		t.Fatal("thinking controls sent to a 4.x model")
	}
	if _, ok := capture.bodies[0]["output_config"]; ok || resp.EffectiveReasoningEffort != "provider-default" {
		t.Fatal("effort sent to a model without effort support")
	}

	serverTool := append([]string{anthropicThinkingToolTurn[0], anthropicThinkingToolTurn[1], anthropicThinkingToolTurn[4],
		`{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{}}}`,
		`{"type":"content_block_stop","index":1}`}, anthropicThinkingToolTurn[11:]...)
	server2 := (&anthropicCapture{}).server(t, serverTool...)
	resp, err = (&AnthropicProvider{apiKey: "test", url: server2.URL}).Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "claude-opus-5-5", nil, nil, nil, nil)
	if err != nil || resp.ProviderState != nil {
		t.Fatalf("server tool turn kept replay state: %+v %v", resp.ProviderState, err)
	}
}
