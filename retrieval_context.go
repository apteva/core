package core

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

const ephemeralContextHeader = "[REQUEST CONTEXT SNAPSHOT — ephemeral; not durable conversation history or current user input; later snapshots supersede earlier ones]"

var legacyEventHeaderRE = regexp.MustCompile(`(?m)^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] Events:\s*$`)

type ephemeralTurnContextSnapshot struct {
	anchor  int
	message Message
}

// ephemeralTurnContextState keeps every request-only turn snapshot at the
// position where the provider first saw it. A later semantic turn appends a
// new snapshot instead of removing or repositioning the old one. This makes
// normal provider requests append-only while keeping retrieval context out of
// durable conversation history and the session journal.
type ephemeralTurnContextState struct {
	snapshots []ephemeralTurnContextSnapshot
	signature string
	active    bool
}

func (s *ephemeralTurnContextState) reset() {
	*s = ephemeralTurnContextState{}
}

func renderEphemeralTurnContext(dynamicContext, now string, idle bool) string {
	content := ephemeralContextHeader + "\n[CURRENT TIME]\nUTC: " + now
	if dynamic := strings.TrimSpace(dynamicContext); dynamic != "" {
		content += "\n\n" + dynamic
	}
	if idle {
		idleText := fmt.Sprintf("[%s] (no new events; continue standing work or wait)", now)
		content += "\n\n" + idleText
	}
	return content
}

func appendWakeStateContext(dynamicContext, reason string, wake time.Time, fired bool) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "continuation"
	}
	pending := pendingWakeDescription(wake)
	if fired {
		pending = "none (timer fired)"
	}
	wakeState := "[WAKE STATE]\nreason: " + reason + "\npending_wake_at: " + pending
	if strings.TrimSpace(dynamicContext) == "" {
		return wakeState
	}
	return strings.TrimSpace(dynamicContext) + "\n\n" + wakeState
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
	for _, snapshot := range s.snapshots {
		if snapshot.anchor < 0 || snapshot.anchor > len(messages) {
			s.reset()
			forceNew = true
			break
		}
	}
	if forceNew || !s.active || s.signature != signature {
		s.snapshots = append(s.snapshots, ephemeralTurnContextSnapshot{
			anchor:  len(messages),
			message: Message{Role: "user", Content: content, RequestContext: true},
		})
		s.signature = signature
		s.active = true
	}

	request := make([]Message, 0, len(messages)+len(s.snapshots))
	for i := 0; i <= len(messages); i++ {
		for _, snapshot := range s.snapshots {
			if snapshot.anchor == i {
				request = append(request, snapshot.message)
			}
		}
		if i < len(messages) {
			request = append(request, messages[i])
		}
	}
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
