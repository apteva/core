package core

import (
	"encoding/json"
	"sort"
)

// A receipt is an observation, not an inferred summary. Preserve small exact
// state fields and source references; explicitly mark collections whose full
// contents must be read again. The immutable archive retains every byte.
func toolResultStateReceipt(content string, budget int) string {
	value, valid := decodeRecoveryJSON(content)
	root, ok := value.(map[string]any)
	if !valid || !ok || budget < 256 {
		return ""
	}
	var bounded func(any, int) any
	bounded = func(v any, depth int) any {
		switch x := v.(type) {
		case string:
			if len(x) > 512 {
				return map[string]any{"omitted_bytes": len(x), "re_read_required": true}
			}
		case []any:
			raw, _ := json.Marshal(x)
			if len(raw) > 512 {
				return map[string]any{"omitted_items": len(x), "re_read_required": true}
			}
		case map[string]any:
			if depth >= 4 {
				return map[string]any{"re_read_required": true}
			}
			out := map[string]any{}
			for _, key := range receiptFieldOrder(x) {
				candidate := bounded(x[key], depth+1)
				out[key] = candidate
				raw, _ := json.Marshal(out)
				if len(raw) > 1024 {
					delete(out, key)
					out["re_read_required"] = true
				}
			}
			return out
		}
		return v
	}
	out := map[string]any{"re_read_required": true}
	for _, key := range receiptFieldOrder(root) {
		out[key] = bounded(root[key], 0)
		raw, _ := json.Marshal(out)
		if len(raw) > budget {
			delete(out, key)
		}
	}
	raw, _ := json.Marshal(out)
	return string(raw)
}

func receiptFieldOrder(value map[string]any) []string {
	priority := []string{"read_reference", "run", "step", "id", "state", "status", "decision", "approval", "approval_state", "pending_release", "published_post", "receipt_id", "artifact", "artifact_ref", "url"}
	seen := map[string]bool{}
	var keys, rest []string
	for _, key := range priority {
		if _, ok := value[key]; ok {
			keys = append(keys, key)
			seen[key] = true
		}
	}
	for key := range value {
		if !seen[key] {
			rest = append(rest, key)
		}
	}
	sort.Strings(rest)
	return append(keys, rest...)
}
