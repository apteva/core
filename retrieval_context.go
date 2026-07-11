package core

import (
	"fmt"
	"regexp"
	"strings"
)

const ephemeralContextHeader = "[REQUEST CONTEXT — ephemeral; not conversation history or current user input]"

var legacyEventHeaderRE = regexp.MustCompile(`(?m)^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] Events:\s*$`)

// ephemeralTurnContextState keeps request-only context at a stable position
// for the lifetime of one semantic turn. Tool calls and their results then
// append after the same context, preserving the exact provider prefix without
// leaking retrieval context into durable history.
type ephemeralTurnContextState struct {
	anchor    int
	signature string
	message   Message
	active    bool
}

func (s *ephemeralTurnContextState) reset() {
	*s = ephemeralTurnContextState{}
}

func renderEphemeralTurnContext(dynamicContext, now string, idle bool) string {
	content := strings.TrimSpace(dynamicContext)
	if content != "" {
		content = ephemeralContextHeader + "\n" + content
	}
	if idle {
		idleText := fmt.Sprintf("[%s] (no new events; continue standing work or wait)", now)
		if content != "" {
			content += "\n\n" + idleText
		} else {
			content = idleText
		}
	}
	return content
}

func ephemeralTurnContextSignature(dynamicContext string, idle bool) string {
	return strings.TrimSpace(dynamicContext) + fmt.Sprintf("\x00idle=%t", idle)
}

func (s *ephemeralTurnContextState) prepare(messages []Message, dynamicContext, now string, idle, forceNew bool) []Message {
	content := renderEphemeralTurnContext(dynamicContext, now, idle)
	if content == "" {
		s.reset()
		return messages
	}

	signature := ephemeralTurnContextSignature(dynamicContext, idle)
	if forceNew || !s.active || s.signature != signature || s.anchor < 0 || s.anchor > len(messages) {
		s.anchor = len(messages)
		s.signature = signature
		s.message = Message{Role: "user", Content: content, RequestContext: true}
		s.active = true
	}

	request := make([]Message, 0, len(messages)+1)
	request = append(request, messages[:s.anchor]...)
	request = append(request, s.message)
	request = append(request, messages[s.anchor:]...)
	return request
}

// appendEphemeralTurnContext preserves the complete durable prefix and adds
// volatile retrieval state only at the tail of the provider request.
func appendEphemeralTurnContext(messages []Message, dynamicContext, now string, idle bool) []Message {
	var state ephemeralTurnContextState
	return state.prepare(messages, dynamicContext, now, idle, true)
}

func recallQueryForTurn(consumed []string, messages []Message, directive string) (query, source string) {
	var current []string
	for _, event := range consumed {
		if strings.TrimSpace(event) == "" || strings.HasPrefix(event, "[tool:") {
			continue
		}
		current = append(current, event)
	}
	if len(current) > 0 {
		return strings.Join(current, "\n"), "event"
	}

	for i := len(messages) - 1; i >= 0; i-- {
		message := messages[i]
		if message.Role != "user" || len(message.ToolResults) > 0 || strings.TrimSpace(message.Content) == "" {
			continue
		}
		cleaned, synthetic := stripLegacyDynamicContext(message.Content)
		if synthetic && strings.TrimSpace(cleaned) == "" {
			continue
		}
		if strings.Contains(cleaned, "(no events)") || strings.Contains(cleaned, "(no new events;") {
			continue
		}
		return cleaned, "active_task"
	}

	if strings.TrimSpace(directive) != "" {
		return directive, "directive"
	}
	return "", "none"
}

// stripLegacyDynamicContext removes request-only blocks written by older core
// versions. A combined entry keeps its real Events block; a synthetic-only
// entry is dropped by callers.
func stripLegacyDynamicContext(content string) (string, bool) {
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "[memories ") &&
		!strings.HasPrefix(trimmed, "[ACTIVE THREADS]") &&
		!strings.HasPrefix(trimmed, ephemeralContextHeader) {
		return content, false
	}
	if loc := legacyEventHeaderRE.FindStringIndex(trimmed); loc != nil {
		return strings.TrimSpace(trimmed[loc[0]:]), true
	}
	return "", true
}
