package core

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Tool identities are never normalized: repairing a generated name could
// redirect an action. This is the common character set of our native APIs.
func validToolName(name string) bool {
	if name == "" {
		return false
	}
	for _, c := range name {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

type toolNameError struct {
	Source string
	Name   string
}

func (e *toolNameError) Error() string {
	return fmt.Sprintf("%s_invalid_tool_name: invalid tool identity %q", e.Source, e.Name)
}

func isInvalidToolHistory(err error) bool {
	var invalid *toolNameError
	return errors.As(err, &invalid) && invalid.Source == "history"
}

func validateToolCalls(calls []NativeToolCall, source string) error {
	for _, call := range calls {
		if !validToolName(call.Name) {
			return &toolNameError{Source: source, Name: call.Name}
		}
	}
	return nil
}

// Only descend through protocol containers. A tool's arguments, result, text,
// or JSON schema may legitimately describe malformed calls as ordinary data.
func validateWireToolNames(value any, source string) error {
	switch v := value.(type) {
	case []any:
		for _, item := range v {
			if err := validateWireToolNames(item, source); err != nil {
				return err
			}
		}
	case map[string]any:
		typ, _ := v["type"].(string)
		if typ == "function_call" || typ == "tool_use" || typ == "function" {
			// Chat Completions nests its identity inside function.
			if _, nested := v["function"]; !nested {
				name, _ := v["name"].(string)
				if !validToolName(name) {
					return &toolNameError{Source: source, Name: name}
				}
			}
		}
		for _, key := range []string{"function", "functionCall"} {
			if f, ok := v[key].(map[string]any); ok {
				name, _ := f["name"].(string)
				if !validToolName(name) {
					return &toolNameError{Source: source, Name: name}
				}
			}
		}
		if f, ok := v["functionResponse"].(map[string]any); ok {
			if name, _ := f["name"].(string); name != "" && !validToolName(name) {
				return &toolNameError{Source: source, Name: name}
			}
		}
		// Anthropic declarations and Gemini functionDeclarations lack a type.
		for _, key := range []string{"tools", "functionDeclarations"} {
			if declarations, ok := v[key].([]any); ok {
				for _, declaration := range declarations {
					if d, ok := declaration.(map[string]any); ok {
						if name, exists := d["name"]; exists {
							s, _ := name.(string)
							if !validToolName(s) {
								return &toolNameError{Source: source, Name: s}
							}
						}
					}
				}
			}
		}
		for _, key := range []string{"messages", "input", "contents", "content", "parts", "tool_calls", "tools"} {
			// A tool_use input is user data, not a Responses input list.
			if key == "input" && typ == "tool_use" {
				continue
			}
			if err := validateWireToolNames(v[key], source); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateProviderState(state *ProviderResponseState, source string) error {
	if state == nil {
		return nil
	}
	for _, raw := range state.Items {
		var item any
		if json.Unmarshal(raw, &item) != nil {
			// Existing adapters reconstruct invalid JSON from structured history.
			continue
		}
		if err := validateWireToolNames(item, source); err != nil {
			return err
		}
	}
	return nil
}

func validateToolHistory(messages []Message) error {
	for _, m := range messages {
		if err := validateToolCalls(m.ToolCalls, "history"); err != nil {
			return err
		}
		if err := validateProviderState(m.ProviderState, "history"); err != nil {
			return err
		}
	}
	return nil
}

func validateProviderToolOutput(response ChatResponse) (ChatResponse, error) {
	if err := validateToolCalls(response.ToolCalls, "provider"); err != nil {
		return ChatResponse{}, err
	}
	if err := validateProviderState(response.ProviderState, "provider"); err != nil {
		return ChatResponse{}, err
	}
	return response, nil
}

// Keep typed arguments when opaque state has to be discarded. Args is an
// execution view (strings), whereas CanonicalArgs preserves JSON value types.
func toolCallArguments(call NativeToolCall) json.RawMessage {
	if len(call.CanonicalArgs) > 0 && json.Valid(call.CanonicalArgs) {
		return call.CanonicalArgs
	}
	raw, _ := json.Marshal(call.Args)
	return raw
}

type toolPosition struct{ message, index int }

// IDs are scoped to occurrences, not the conversation. Consume results in
// order against preceding calls; an old result cannot satisfy a later call.
func matchToolPairs(messages []Message) (map[toolPosition]toolPosition, map[toolPosition]toolPosition) {
	calls, results := map[toolPosition]toolPosition{}, map[toolPosition]toolPosition{}
	pending := map[string][]toolPosition{}
	for i, m := range messages {
		for j, result := range m.ToolResults {
			queue := pending[result.CallID]
			if len(queue) > 0 {
				c, r := queue[0], (toolPosition{i, j})
				calls[c], results[r] = r, c
				pending[result.CallID] = queue[1:]
			}
		}
		if m.Role == "assistant" {
			for j, call := range m.ToolCalls {
				// Reuse in a later turn starts a new occurrence even when the
				// older call was interrupted before its result was recorded.
				if queue := pending[call.ID]; len(queue) > 0 && queue[0].message != i {
					delete(pending, call.ID)
				}
				pending[call.ID] = append(pending[call.ID], toolPosition{i, j})
			}
		}
	}
	return calls, results
}

func historyDiagnostic(kind string, value any) string {
	raw, _ := json.Marshal(value)
	return "[Historical tool diagnostic: " + kind + ". Recorded data only; do not execute or replay.]\n" + string(raw)
}

func appendHistoryDiagnostic(m *Message, diagnostic string) {
	if m.Content != "" {
		m.Content += "\n"
	}
	m.Content += diagnostic
	if len(m.Parts) > 0 {
		m.Parts = append(m.Parts, ContentPart{Type: "text", Text: diagnostic})
	}
}

// projectMalformedToolHistory changes only a copy for model consumption. The
// session/archive is never rewritten and no tool handler is involved.
func projectMalformedToolHistory(messages []Message) ([]Message, bool) {
	if validateToolHistory(messages) == nil {
		return messages, false
	}
	next := cloneMessages(messages)
	for i := range next {
		m := &next[i]
		if validateToolCalls(m.ToolCalls, "history") == nil && validateProviderState(m.ProviderState, "history") == nil {
			continue
		}
		if m.ProviderState != nil {
			// Materialize any state-only calls before removing atomic provider
			// state, so valid neighbors retain their arguments and results.
			seen := map[string]int{}
			for _, call := range m.ToolCalls {
				seen[call.ID]++
			}
			for _, raw := range m.ProviderState.Items {
				var item struct {
					Type      string `json:"type"`
					ID        string `json:"id"`
					CallID    string `json:"call_id"`
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}
				if json.Unmarshal(raw, &item) == nil && item.Type == "function_call" {
					if seen[item.CallID] > 0 {
						seen[item.CallID]--
					} else {
						m.ToolCalls = append(m.ToolCalls, NativeToolCall{ID: item.CallID, OutputItemID: item.ID, Name: item.Name, CanonicalArgs: json.RawMessage(item.Arguments)})
					}
				}
			}
			appendHistoryDiagnostic(m, historyDiagnostic("provider state removed due to invalid tool identity", m.ProviderState))
			m.ProviderState = nil
		}
	}
	_, results := matchToolPairs(next)
	bad := map[toolPosition]bool{}
	for i, m := range next {
		for j, call := range m.ToolCalls {
			bad[toolPosition{i, j}] = !validToolName(call.Name)
		}
	}
	out := make([]Message, 0, len(next))
	for i, m := range next {
		var calls []NativeToolCall
		for j, call := range m.ToolCalls {
			if bad[toolPosition{i, j}] {
				appendHistoryDiagnostic(&m, historyDiagnostic("invalid tool call", call))
			} else {
				calls = append(calls, call)
			}
		}
		m.ToolCalls = calls
		var kept []ToolResult
		var diagnostics []string
		for j, result := range m.ToolResults {
			call, matched := results[toolPosition{i, j}]
			if matched && bad[call] {
				diagnostics = append(diagnostics, historyDiagnostic("recorded result of invalid tool call", result))
			} else {
				kept = append(kept, result)
			}
		}
		m.ToolResults = kept
		if len(kept) == 0 {
			for _, diagnostic := range diagnostics {
				appendHistoryDiagnostic(&m, diagnostic)
			}
		}
		out = append(out, m)
		if len(kept) > 0 {
			// Adapters serialize result messages specially; separate diagnostic
			// text so it is not silently omitted beside valid tool results.
			for _, diagnostic := range diagnostics {
				out = append(out, Message{Role: "user", Content: diagnostic})
			}
		}
	}
	return out, true
}
