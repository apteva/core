package core

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

const ephemeralContextHeader = "[REQUEST CONTEXT SNAPSHOT — ephemeral; not durable conversation history or current user input; this current snapshot replaces prior request context]"

var legacyEventHeaderRE = regexp.MustCompile(`(?m)^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] Events:\s*$`)

type ephemeralTurnContextSnapshot struct {
	anchor  int
	message Message
}

// ephemeralTurnContextState keeps exactly one request-only context snapshot.
// It stays anchored while tool results and correction turns continue the same
// wake, then is replaced at the durable tail when a new external/wake context
// begins. This keeps the current retrieval available without accumulating
// superseded memory blocks in provider requests or durable history.
type ephemeralTurnContextState struct {
	snapshots []ephemeralTurnContextSnapshot
	active    bool
}

// memoryRecallCycleState holds one automatic-retrieval result across the
// internal continuations caused by tool results, retries, and correction
// turns. A new external wake (event, timer, or resume) starts a fresh cycle.
// There is intentionally no task abstraction here.
type memoryRecallCycleState struct {
	initialized     bool
	storeGeneration uint64
	directive       string
	context         string
	matches         []MemoryRecallMatch
	candidates      int
	querySource     string
	contextHash     string
}

func (s *memoryRecallCycleState) refreshReason(storeGeneration uint64, directive string, hasExternalEvent, timerWake, resumeWake bool) string {
	switch {
	case hasExternalEvent:
		return "external_event"
	case timerWake:
		return "timer"
	case resumeWake:
		return "resume"
	case !s.initialized:
		return "thread_start"
	case s.storeGeneration != storeGeneration:
		return "memory_changed"
	case s.directive != directive:
		return "directive_changed"
	default:
		return ""
	}
}

func (s *memoryRecallCycleState) set(storeGeneration uint64, directive, context, querySource string, candidates int, matches []MemoryRecallMatch) {
	s.initialized = true
	s.storeGeneration = storeGeneration
	s.directive = directive
	s.context = context
	s.matches = append(s.matches[:0], matches...)
	s.candidates = candidates
	s.querySource = querySource
	s.contextHash = promptCacheShortHash([]byte(context))
}

func (s *ephemeralTurnContextState) reset() {
	*s = ephemeralTurnContextState{}
}

func renderEphemeralTurnContext(dynamicContext, now string, idle bool) string {
	content := ephemeralContextHeader + "\n" + renderCurrentTimeContext(now)
	if dynamic := strings.TrimSpace(dynamicContext); dynamic != "" {
		content += "\n\n" + dynamic
	}
	if idle {
		idleText := fmt.Sprintf("[%s] (no new events; continue standing work or wait)", now)
		content += "\n\n" + idleText
	}
	return content
}

func renderCurrentTimeContext(now string) string {
	content := "[CURRENT TIME]\nUTC: " + now
	if parsed, ok := parsePromptUTC(now); ok {
		content += "\nUTC weekday: " + parsed.Weekday().String()
		content += "\nUTC calendar date: " + parsed.Format("2006-01-02")
	}
	content += "\nTimezone rule: Unless an instruction explicitly specifies another timezone, interpret unqualified calendar days and dates in UTC."
	return content
}

func parsePromptUTC(value string) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return parsed.UTC(), true
	}
	for _, layout := range []string{"2006-01-02 15:04:05", "2006-01-02 15:04"} {
		if parsed, err := time.ParseInLocation(layout, value, time.UTC); err == nil {
			return parsed.UTC(), true
		}
	}
	return time.Time{}, false
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

func (s *ephemeralTurnContextState) prepare(messages []Message, dynamicContext, now string, idle, forceNew bool) []Message {
	content := renderEphemeralTurnContext(dynamicContext, now, idle)
	if content == "" {
		s.reset()
		return messages
	}

	for _, snapshot := range s.snapshots {
		if snapshot.anchor < 0 || snapshot.anchor > len(messages) {
			s.reset()
			forceNew = true
			break
		}
	}
	if forceNew || !s.active {
		s.snapshots = []ephemeralTurnContextSnapshot{{
			anchor:  len(messages),
			message: Message{Role: "user", Content: content, RequestContext: true},
		}}
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
		return cleaned, "latest_user_context"
	}

	if strings.TrimSpace(directive) != "" {
		return directive, "directive"
	}
	return "", "none"
}

// recallQueriesForTurn returns the independent inputs to automatic memory
// relevance. The standing directive always defines the thread's durable
// memory scope; current external context supplements it but never replaces it.
// The strings stay separate so long directives cannot dilute a strong event
// match (or vice versa) inside lexical or embedding retrieval.
func recallQueriesForTurn(consumed []string, messages []Message, directive string) (queries []string, source string) {
	current, currentSource := recallQueryForTurn(consumed, messages, "")
	if strings.TrimSpace(current) != "" {
		queries = append(queries, current)
	}
	directive = strings.TrimSpace(directive)
	if directive != "" && directive != strings.TrimSpace(current) {
		queries = append(queries, directive)
	}

	switch {
	case currentSource != "none" && directive != "":
		return queries, currentSource + "+directive"
	case currentSource != "none":
		return queries, currentSource
	case directive != "":
		return queries, "directive"
	default:
		return nil, "none"
	}
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
