package core

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
)

// Only public answer text backed verbatim by successful tool evidence is eligible
// for early delivery. This is intentionally not a semantic fact checker: arbitrary
// summaries still take the ordinary final-result path. Reasoning/tool-argument
// callbacks never enter this collector.
type threadWorkStream struct {
	thinker  *Thinker
	work     ThreadWork
	evidence []string
	pending  string
	bytes    int
	disabled bool
}

func (t *Thinker) newThreadWorkStream(messages []Message) *threadWorkStream {
	s := &threadWorkStream{thinker: t, work: t.work.snapshot()}
	if !s.work.StreamResults || t.pendingToolCount() != 0 {
		s.disabled = true
		return s
	}
	// Only the most recent result batch can justify this answer. A failed or
	// incomplete operation cannot inherit a success from an earlier assignment.
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && len(messages[i].ToolResults) == 0 {
			break
		}
		if len(messages[i].ToolResults) == 0 {
			continue
		}
		for _, r := range messages[i].ToolResults {
			if r.IsError || r.ContentIsPreview {
				continue
			}
			s.evidence = append(s.evidence, r.Content)
			var value any
			if json.Unmarshal([]byte(r.Content), &value) == nil {
				s.collectStrings(value)
			}
		}
		break
	}
	return s
}
func (s *threadWorkStream) collectStrings(v any) {
	switch v := v.(type) {
	case string:
		s.evidence = append(s.evidence, v)
	case []any:
		for _, x := range v {
			s.collectStrings(x)
		}
	case map[string]any:
		for _, x := range v {
			s.collectStrings(x)
		}
	}
}
func (s *threadWorkStream) append(text string) {
	if s.disabled || !s.thinker.work.current(s.work) {
		return
	}
	s.bytes += len(text)
	if s.bytes > 16384 {
		s.disabled = true
		s.pending = ""
		return
	}
	s.pending += text
	for {
		end := completeWorkSentence(s.pending)
		if end == 0 {
			return
		}
		sentence := strings.TrimSpace(s.pending[:end])
		s.pending = s.pending[end:]
		if len(sentence) < 12 || len(sentence) > 480 || strings.ContainsAny(sentence, "`{}<>") {
			continue
		}
		verified := false
		for _, evidence := range s.evidence {
			if strings.Contains(evidence, sentence) {
				verified = true
				break
			}
		}
		if verified {
			if err := s.thinker.publishThreadWorkChunk(s.work, sentence); err != nil {
				// Preserve normal final delivery if durable incremental publication fails.
				s.disabled = true
				return
			}
		}
	}
}

// Require whitespace after punctuation, so decimals and fragmented words are
// not released as sentences. A final trailing sentence waits for completion.
func completeWorkSentence(text string) int {
	for i := 0; i+1 < len(text); i++ {
		if strings.ContainsRune(".!?", rune(text[i])) && (text[i+1] == ' ' || text[i+1] == '\n') {
			return i + 1
		}
	}
	return 0
}
func (t *Thinker) publishThreadWorkChunk(w ThreadWork, text string) error {
	if t.work == nil || !t.work.current(w) || t.owner == nil || t.owner.parent == nil {
		return nil
	}
	w.Kind, w.Status, w.Worker = "result_chunk", "verified", t.threadID
	w.ChunkID = fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
	e := newThreadWorkEvent(w, text, nil)
	parent := t.owner.parent
	if parent.threadID == "main" {
		_, err := parent.QueueMainEvents([]PersistentThreadEvent{e})
		return err
	}
	if parent.owner != nil {
		_, err := parent.owner.QueueEvents(parent.threadID, []PersistentThreadEvent{e})
		return err
	}
	parent.bus.Publish(Event{ID: e.ID, Type: EventInbox, From: t.threadID, To: parent.threadID, Text: text, Work: &w})
	return nil
}

func removeDeliveredWorkSentences(text, delivered string) string {
	seen := map[string]bool{}
	for _, sentence := range strings.Split(delivered, "\n") {
		seen[strings.TrimSpace(sentence)] = true
	}
	var remaining strings.Builder
	for len(text) > 0 {
		n := completeWorkSentence(text)
		if n == 0 {
			n = len(text)
		}
		part := text[:n]
		text = text[n:]
		if !seen[strings.TrimSpace(part)] {
			remaining.WriteString(part)
		}
	}
	return strings.TrimSpace(remaining.String())
}
