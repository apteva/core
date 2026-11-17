package core

import (
	"sort"
	"strings"
)

const (
	requestContextMaxUpdates = 16
	requestContextMaxBytes   = 128 << 10
)

type requestContextSection struct{ name, content string }

// Preserve already-sent observations in request order. Updates contain only
// changed sections, so a fresh clock or roster does not repeat recalled memory.
// These observations never enter durable user history or grant capabilities.
func (s *ephemeralTurnContextState) prepareSections(messages []Message, sections []requestContextSection, now string, idle, forceNew bool) []Message {
	if !requestStabilityEnabled("APTEVA_APPEND_REQUEST_CONTEXT") {
		var content []string
		for _, section := range sections {
			if section.content != "" {
				content = append(content, section.content)
			}
		}
		return s.prepare(messages, strings.Join(content, "\n\n"), now, idle, forceNew)
	}
	s.checkpointed = false
	if !s.appendOnly && s.active {
		s.reset()
	}
	for _, snapshot := range s.snapshots {
		if snapshot.anchor < 0 || snapshot.anchor > len(messages) {
			s.reset()
			s.checkpointed = true
			break
		}
	}
	if forceNew || !s.active {
		// Missing sections are explicit retractions, not permission to keep
		// their older values active indefinitely.
		wanted := append([]requestContextSection(nil), sections...)
		present := map[string]bool{}
		for _, section := range wanted {
			present[section.name] = true
		}
		var removed []string
		for name := range s.sections {
			if !present[name] {
				removed = append(removed, name)
			}
		}
		sort.Strings(removed)
		sections = append([]requestContextSection(nil), wanted...)
		for _, name := range removed {
			sections = append(sections, requestContextSection{name, ""})
		}
		render := func(full bool) string {
			var updates []string
			for _, section := range sections {
				old, known := s.sections[section.name]
				if !full && known && old == section.content {
					continue
				}
				content := section.content
				if strings.TrimSpace(content) == "" {
					content = "(empty; supersedes this section's previous values)"
				}
				updates = append(updates, "[CONTEXT UPDATE: "+section.name+"]\n"+content)
			}
			if len(updates) == 0 {
				updates = append(updates, "Other context sections are unchanged; retain their latest values.")
			}
			return renderEphemeralTurnContext(strings.Join(updates, "\n\n"), now, idle)
		}
		content := render(!s.active)
		bytes := len(content)
		for _, snapshot := range s.snapshots {
			bytes += len(snapshot.message.Content)
		}
		if s.active && (len(s.snapshots) >= requestContextMaxUpdates || bytes > requestContextMaxBytes) {
			content = render(true)
			s.snapshots = nil
			s.checkpointed = true
		}
		s.snapshots = append(s.snapshots, ephemeralTurnContextSnapshot{anchor: len(messages), message: Message{Role: "user", Content: content, RequestContext: true}})
		s.sections = make(map[string]string, len(sections))
		for _, section := range wanted {
			s.sections[section.name] = section.content
		}
		s.active, s.appendOnly = true, true
	}
	request := make([]Message, 0, len(messages)+len(s.snapshots))
	next := 0
	for i := 0; i <= len(messages); i++ {
		for next < len(s.snapshots) && s.snapshots[next].anchor == i {
			request = append(request, s.snapshots[next].message)
			next++
		}
		if i < len(messages) {
			request = append(request, messages[i])
		}
	}
	return request
}

// A memory write can withdraw or change previously surfaced material. Drop the
// request-only ledger at that explicit boundary rather than retaining withdrawn
// memory for cache reuse. Ordinary retrieval/query changes use small updates.
func (t *Thinker) invalidateRequestContextForMemoryGeneration(generation uint64) {
	if requestStabilityEnabled("APTEVA_APPEND_REQUEST_CONTEXT") && t.memoryRecall.initialized && t.memoryRecall.storeGeneration != generation {
		t.advancePromptCacheEpoch("memory_store_changed", true, nil)
	}
}
