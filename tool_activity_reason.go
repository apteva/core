package core

import (
	"strings"
	"unicode"
)

// prepareToolCallData removes operator-only metadata before dispatch. A model
// can omit _reason even when it is required in the presented schema; that must
// not block the operation or leave its activity label empty. Fallbacks describe
// the declared tool operation, never inferred intent or argument values.
func (t *Thinker) prepareToolCallData(call *toolCall, executionIDs []string) ToolCallData {
	reason := strings.TrimSpace(call.Args["_reason"])
	delete(call.Args, "_reason")
	source := "model"
	if reason == "" {
		def := call.definition
		if def == nil && t.registry != nil {
			def = t.registry.Get(call.Name)
		}
		if def != nil {
			reason = toolDescriptionActivityLabel(def.Description)
		}
		source = "tool_description"
		if reason == "" {
			name := strings.Join(strings.FieldsFunc(call.Name, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r)
			}), " ")
			reason = "Running tool"
			if name != "" {
				reason = "Running " + name
			}
			source = "tool_name"
		}
		reason = boundToolActivityLabel(reason)
	}
	return ToolCallData{
		ID: call.NativeID, Name: call.Name, Args: call.Args,
		Reason: reason, ReasonSource: source, ExecutionIDs: executionIDs,
	}
}

func toolDescriptionActivityLabel(description string) string {
	description = strings.TrimSpace(description)
	// Descriptions commonly append argument documentation or usage guidance.
	// Display just the first sentence/line, without changing its meaning.
	for _, delimiter := range []string{"\n", "\r", ". ", ";", " Args:", " Arguments:"} {
		if i := strings.Index(description, delimiter); i >= 0 {
			description = description[:i]
		}
	}
	return strings.TrimRight(strings.Join(strings.Fields(description), " "), ".")
}

func boundToolActivityLabel(label string) string {
	const limit = 160
	runes := []rune(label)
	if len(runes) <= limit {
		return label
	}
	return strings.TrimSpace(string(runes[:limit-1])) + "…"
}
