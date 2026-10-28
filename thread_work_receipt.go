package core

import (
	"encoding/json"
	"fmt"
	"strings"
)

const completeAssignmentArg = "_complete_assignment"

// DirectToolResultPolicy is host configuration, never model/tool supplied.
// Only these top-level JSON scalar fields may be disclosed directly. The
// success field must equal SuccessValue as a JSON string (e.g. status=confirmed).
// A call must also explicitly declare that it finishes the current assignment.
// Missing declarations never imply completion, even for a successful tool.
type DirectToolResultPolicy struct {
	Tool         string   `json:"tool"`
	SuccessField string   `json:"success_field"`
	SuccessValue string   `json:"success_value"`
	PublicFields []string `json:"public_fields"`
}

// ThreadWorkReceipt binds verified public facts to the admitted assignment and
// native call. It is persisted with the original tool output, not sent as policy.
type ThreadWorkReceipt struct {
	Work    ThreadWork `json:"work"`
	Content string     `json:"content"`
}

func validateRealtimeExecutionConfig(c RealtimeExecutionConfig) error {
	if c.Model != "" && c.ModelID != "" {
		return fmt.Errorf("realtime_execution: choose model tier or model_id, not both")
	}
	if c.Reasoning != "" {
		if _, ok := parseReasoningLevel(c.Reasoning); !ok {
			return fmt.Errorf("invalid realtime execution reasoning %q", c.Reasoning)
		}
	}
	if len(c.PreloadTools)+len(c.ResultTools) > 20 {
		return fmt.Errorf("realtime execution preload/result tools exceed 20 entries")
	}
	for _, n := range c.PreloadTools {
		if !validToolName(n) {
			return fmt.Errorf("invalid execution preload tool %q", n)
		}
	}
	seen := map[string]bool{}
	for _, p := range c.ResultTools {
		if !validToolName(p.Tool) || seen[p.Tool] || p.SuccessField == "" || p.SuccessValue == "" || len(p.PublicFields) == 0 || len(p.PublicFields) > 16 {
			return fmt.Errorf("invalid direct result policy for %q", p.Tool)
		}
		seen[p.Tool] = true
		for _, f := range p.PublicFields {
			if strings.TrimSpace(f) == "" {
				return fmt.Errorf("empty public result field for %q", p.Tool)
			}
		}
	}
	return nil
}

func (t *Thinker) directResultPolicy(name string) *DirectToolResultPolicy {
	if t == nil || t.work == nil || t.executionPolicy == nil {
		return nil
	}
	if t.registry != nil {
		if def := t.registry.Get(name); def != nil {
			if props, ok := def.InputSchema["properties"].(map[string]any); ok {
				if _, exists := props[completeAssignmentArg]; exists {
					return nil
				}
			}
		}
	}
	for i := range t.executionPolicy.ResultTools {
		if t.executionPolicy.ResultTools[i].Tool == name {
			return &t.executionPolicy.ResultTools[i]
		}
	}
	return nil
}

func (t *Thinker) preloadExecutionTools(overlay map[string]bool) map[string]bool {
	if t.executionPolicy == nil {
		return overlay
	}
	names := append([]string(nil), t.executionPolicy.PreloadTools...)
	for _, p := range t.executionPolicy.ResultTools {
		names = append(names, p.Tool)
	}
	bytes := 0
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] || !t.toolAuthorized(name) {
			continue
		}
		seen[name] = true
		def := t.registry.Get(name)
		if def == nil {
			continue
		}
		encoded, _ := json.Marshal(def.InputSchema)
		size := len(encoded) + len(def.Description) + len(def.Name)
		if len(seen) > 20 || bytes+size > 64000 {
			continue
		}
		bytes += size
		if overlay == nil {
			overlay = map[string]bool{}
		}
		overlay[name] = true
	}
	return overlay
}

func (t *Thinker) decorateDirectResultTools(tools []NativeTool) {
	for i := range tools {
		if t.directResultPolicy(tools[i].Name) == nil {
			continue
		}
		// Never mutate registry schemas shared by the main thread/other workers.
		data, _ := json.Marshal(tools[i].Parameters)
		var schema map[string]any
		if json.Unmarshal(data, &schema) != nil || schema == nil {
			continue
		}
		props, ok := schema["properties"].(map[string]any)
		if !ok {
			continue
		}
		if _, exists := props[completeAssignmentArg]; exists {
			continue
		}
		props[completeAssignmentArg] = map[string]any{"type": "boolean", "description": "Required completion decision. Always supply true or false. Set true only when this single tool's successful result completes ALL requested work and all required steps. Core may return its verified public facts without another model turn. Set false for intermediate work, unresolved ambiguity, or any remaining follow-up (such as sending a message after booking)."}
		// JSON round-tripping turns existing required entries into []any. Preserve
		// every app requirement and require an explicit completion decision.
		required, _ := schema["required"].([]any)
		schema["required"] = append(required, completeAssignmentArg)
		tools[i].Description += " Always include _complete_assignment: true only if this call completes the entire assignment; false if any work remains."
		tools[i].Parameters = schema
		tools[i].serializedBytes = 0
	}
}

func (t *Thinker) directToolReceipt(call toolCall, response ToolResponse) (receipt *ThreadWorkReceipt) {
	policy := t.directResultPolicy(call.Name)
	// Set by the result checks below. Never includes result contents.
	reason := "result_not_eligible"
	if policy != nil {
		defer func() {
			if t.telemetry == nil {
				return
			}
			if receipt != nil {
				reason = "eligible"
			} else if _, ok := call.Args[completeAssignmentArg]; !ok {
				reason = "completion_decision_missing"
			} else if call.Args[completeAssignmentArg] != "true" {
				reason = "assignment_not_complete"
			}
			data := map[string]any{"tool": call.Name, "call_id": call.NativeID, "reason": reason}
			if call.work != nil {
				data["request_id"] = call.work.ID
				data["revision"] = call.work.Revision
				data["delegation_id"] = call.work.DelegationID
			}
			t.telemetry.Emit("thread.work_direct_result_evaluated", t.threadID, data)
		}()
	}
	if policy == nil || call.work == nil || call.NativeID == "" || call.Args[completeAssignmentArg] != "true" || !t.work.current(*call.work) || response.IsError || len(response.Image) != 0 || len(response.Parts) != 0 {
		return nil
	}
	// Policy cannot override a tool's own argument with the same reserved name.
	if call.definition != nil {
		if props, ok := call.definition.InputSchema["properties"].(map[string]any); ok {
			if _, exists := props[completeAssignmentArg]; exists {
				return nil
			}
		}
	}
	// The result must be exactly one JSON object. When structured content is
	// present, the model-facing text must be that same object: extra text or a
	// differing mirror could carry an error the receipt would hide.
	var result map[string]json.RawMessage
	_, strictErr := decodeSingleJSON([]byte(response.Text))
	if strictErr != nil || json.Unmarshal([]byte(response.Text), &result) != nil || result == nil {
		reason = "malformed_result"
		if len(response.StructuredContent) > 0 && json.Unmarshal(response.StructuredContent, &result) == nil && result != nil {
			reason = "conflicting_representations"
		}
		return nil
	}
	if len(response.StructuredContent) > 0 && !equivalentJSON([]byte(response.Text), response.StructuredContent) {
		reason = "conflicting_representations"
		return nil
	}
	var status string
	if json.Unmarshal(result[policy.SuccessField], &status) != nil || status != policy.SuccessValue {
		reason = "success_status_failed"
		return nil
	}
	facts := map[string]json.RawMessage{}
	for _, field := range policy.PublicFields {
		value, ok := result[field]
		if !ok {
			return nil
		}
		var scalar any
		if json.Unmarshal(value, &scalar) != nil || scalar == nil {
			return nil
		}
		switch v := scalar.(type) {
		case string:
			if strings.TrimSpace(v) == "" {
				return nil
			}
		case float64, bool:
		default:
			return nil
		}
		facts[field] = value
	}
	data, err := json.Marshal(facts)
	// Conservative byte bound also bounds tokenizer input for Live's 500-token
	// append limit. Oversized receipts take the regular summarization path.
	if err != nil || len(data) > 400 {
		return nil
	}
	return &ThreadWorkReceipt{Work: *call.work, Content: string(data)}
}

func (t *Thinker) completeDirectToolResult(results []ToolResult, external bool) (bool, error) {
	if external || t.work == nil || len(results) != 1 || t.pendingToolCount() != 0 {
		return false, nil
	}
	r := results[0]
	receipt := r.DirectReceipt
	if receipt == nil || r.IsError || r.ContentIsPreview || !t.work.ready(receipt.Work) || t.directResultPolicy(r.ToolName) == nil {
		return false, nil
	}
	// Only a sole native call can finish here. A successful member of a parallel
	// batch cannot silently abandon another operation, even if it finished first.
	matched := false
	for i := len(t.messages) - 1; i >= 0; i-- {
		m := t.messages[i]
		if m.Role != "assistant" {
			continue
		}
		matched = len(m.ToolCalls) == 1 && m.ToolCalls[0].ID == r.CallID && m.ToolCalls[0].Name == r.ToolName && m.ToolCalls[0].Args[completeAssignmentArg] == "true"
		break
	}
	if !matched {
		return false, nil
	}
	w := receipt.Work
	w.Kind, w.Status, w.Worker = "result", "completed", t.threadID
	msg := Message{Role: "assistant", Content: receipt.Content, Work: &w}
	// The tool output is already durable. Publish to the idempotent parent
	// outbox before marking history completed, so a crash cannot suppress an
	// as-yet-undelivered receipt on restart.
	if err := t.publishThreadWork(receipt.Work, receipt.Content, "completed", nil); err != nil {
		return false, err
	}
	// publishThreadWork has persisted the new event-only wait; do not overwrite
	// it with an earlier timer's pacing state.
	t.wakeDeadlineFired = false
	if t.session != nil {
		if err := t.session.AppendMessage(msg, t.iteration, TokenUsage{}); err != nil {
			return false, err
		}
	}
	t.messages = append(t.messages, msg)
	if t.telemetry != nil {
		t.telemetry.Emit("thread.work_direct_result", t.threadID, map[string]any{"request_id": w.ID, "revision": w.Revision, "delegation_id": w.DelegationID, "tool": r.ToolName, "call_id": r.CallID, "narration_requests_skipped": 1})
	}
	return true, nil
}
