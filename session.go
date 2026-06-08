package core

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	defaultLoadTail   = 50  // messages loaded into context on startup
	compactThreshold  = 500 // trigger compaction when file exceeds this
	compactKeepRecent = 100 // keep this many recent messages after compaction
	historyDir        = "history"
)

// SessionEntry is one line in the JSONL history file.
type SessionEntry struct {
	Timestamp   time.Time        `json:"ts"`
	Role        string           `json:"role"` // "system", "user", "assistant", "tool_result", "_compacted"
	Content     string           `json:"content"`
	Parts       []ContentPart    `json:"parts,omitempty"`
	ToolCalls   []NativeToolCall `json:"tool_calls,omitempty"`
	ToolResults []ToolResult     `json:"tool_results,omitempty"`
	Summary     string           `json:"summary,omitempty"`        // for _compacted entries
	OrigCount   int              `json:"original_count,omitempty"` // how many messages were compacted
	TokensIn    int              `json:"tokens_in,omitempty"`
	TokensOut   int              `json:"tokens_out,omitempty"`
	Iteration   int              `json:"iteration,omitempty"`
}

// Session manages persistent JSONL history for one thread.
type Session struct {
	mu    sync.Mutex
	path  string
	count int // approximate line count
}

// NewSession creates or opens a session file for a thread.
func NewSession(baseDir, threadID string) *Session {
	dir := filepath.Join(baseDir, historyDir)
	os.MkdirAll(dir, 0755)
	path := filepath.Join(dir, threadID+".jsonl")

	s := &Session{path: path}

	// Count existing lines
	if f, err := os.Open(path); err == nil {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			s.count++
		}
		f.Close()
	}

	return s
}

// Append writes one entry to the history file.
func (s *Session) Append(entry SessionEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}

	f, err := os.OpenFile(s.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()

	json.NewEncoder(f).Encode(entry)
	s.count++
}

// AppendMessage is a convenience to append a Message as a SessionEntry.
func (s *Session) AppendMessage(msg Message, iteration int, usage TokenUsage) {
	entry := SessionEntry{
		Timestamp:   time.Now(),
		Role:        msg.Role,
		Content:     msg.Content,
		Parts:       msg.Parts,
		ToolCalls:   msg.ToolCalls,
		ToolResults: msg.ToolResults,
		TokensIn:    usage.PromptTokens,
		TokensOut:   usage.CompletionTokens,
		Iteration:   iteration,
	}
	s.Append(entry)
}

// LoadTail reads the last n messages from the history file and converts them to Messages.
// Skips system messages and _compacted entries (compacted summaries are prepended as context).
func (s *Session) LoadTail(n int) (messages []Message, compactedSummaries []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	f, err := os.Open(s.path)
	if err != nil {
		return nil, nil
	}
	defer f.Close()

	var entries []SessionEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			entries = append(entries, entry)
		}
	}

	// Collect compacted summaries
	for _, e := range entries {
		if e.Role == "_compacted" && e.Summary != "" {
			compactedSummaries = append(compactedSummaries, e.Summary)
		}
	}

	// Filter to real messages only
	var real []SessionEntry
	for _, e := range entries {
		if e.Role == "system" || e.Role == "_compacted" {
			continue
		}
		real = append(real, e)
	}

	// Take tail
	if len(real) > n {
		real = real[len(real)-n:]
	}

	// Convert to Messages
	for _, e := range real {
		msg := Message{
			Role:        e.Role,
			Content:     e.Content,
			Parts:       e.Parts,
			ToolCalls:   e.ToolCalls,
			ToolResults: e.ToolResults,
		}
		// Normalize role: "tool_result" → "user" with ToolResults
		if e.Role == "tool_result" {
			msg.Role = "user"
		}
		messages = append(messages, msg)
	}

	// Sanitize: remove orphaned tool_results that have no matching tool_use
	messages = sanitizeToolPairs(messages)

	return messages, compactedSummaries
}

// sanitizeToolPairs fixes mismatched tool_use/tool_result pairs that
// cause referential-integrity errors on strict providers (Moonshot via
// opencode-go, Anthropic). Every tool_result.CallID must point to an
// assistant tool_calls[].id earlier in the same payload, and every
// assistant tool_call must be followed by a tool result.
//
// Handles both directions:
//   - tool_result without matching tool_use → drop the tool_result.
//   - tool_use without matching tool_result → drop the tool_use from
//     the assistant message.
//
// pendingIDs are tool call IDs with async results still in flight —
// never strip those (the iteration barrier injects placeholders for
// them so the wire payload still pairs cleanly).
//
// Note: a previous version preserved image-bearing tool_results even
// when orphaned, on the theory that screenshots had to survive. That
// produced wire payloads with unpaired tool_call_ids that Moonshot
// 400'd on. Old screenshots are already evicted to text placeholders
// after the 3-image cap (see thinker.go), so dropping orphan-with-
// image entirely is safe and matches the protocol.
func sanitizeToolPairs(messages []Message, pendingIDs ...map[string]bool) []Message {
	pending := map[string]bool{}
	if len(pendingIDs) > 0 && pendingIDs[0] != nil {
		pending = pendingIDs[0]
	}
	toolUseIDs := make(map[string]bool)
	toolResultIDs := make(map[string]bool)
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			toolUseIDs[tc.ID] = true
		}
		for _, tr := range m.ToolResults {
			toolResultIDs[tr.CallID] = true
		}
	}

	var result []Message
	removed := 0
	for _, m := range messages {
		// Drop orphaned tool_results (no matching tool_use anywhere).
		if len(m.ToolResults) > 0 {
			var valid []ToolResult
			for _, tr := range m.ToolResults {
				if toolUseIDs[tr.CallID] {
					valid = append(valid, tr)
				}
			}
			if len(valid) == 0 {
				removed++
				continue
			}
			m.ToolResults = valid
		}

		// Drop orphaned tool_uses (no matching tool_result + not pending).
		if len(m.ToolCalls) > 0 && m.Role == "assistant" {
			var valid []NativeToolCall
			for _, tc := range m.ToolCalls {
				if toolResultIDs[tc.ID] || pending[tc.ID] {
					valid = append(valid, tc)
				}
			}
			if len(valid) != len(m.ToolCalls) {
				removed += len(m.ToolCalls) - len(valid)
				m.ToolCalls = valid
			}
		}

		result = append(result, m)
	}

	if removed > 0 {
		logMsg("SESSION", fmt.Sprintf("sanitized: fixed %d orphaned tool pairs", removed))
	}
	return result
}

// NeedsCompaction returns true if the file is large enough to compact.
func (s *Session) NeedsCompaction() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count > compactThreshold
}

// Compact summarizes old messages and rewrites the file.
// summarize is a function that takes messages text and returns a summary.
func (s *Session) Compact(summarize func(text string) string) {
	s.mu.Lock()
	entries, err := s.readEntriesLocked()
	if err != nil {
		s.mu.Unlock()
		return
	}

	if len(entries) <= compactThreshold {
		s.count = len(entries)
		s.mu.Unlock()
		return
	}

	combined, _, _ := buildCompactionParts(entries)
	s.mu.Unlock()

	// Summarization can be arbitrary caller code. Run it outside the
	// session lock so callbacks can inspect session state without
	// deadlocking, and so a slow summarizer does not block appends.
	summaryText := ""
	if summarize != nil && combined != "" {
		if result := summarize(combined); result != "" {
			summaryText = result
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Re-read after the unlocked summarization window so entries appended
	// concurrently are preserved when the file is rewritten.
	entries, err = s.readEntriesLocked()
	if err != nil {
		return
	}
	if len(entries) <= compactThreshold {
		s.count = len(entries)
		return
	}
	_, realCount, recent := buildCompactionParts(entries)
	if summaryText == "" {
		summaryText = fmt.Sprintf("Compacted %d messages.", realCount)
	}

	compactedEntry := SessionEntry{
		Timestamp: time.Now(),
		Role:      "_compacted",
		Summary:   summaryText,
		OrigCount: realCount,
	}

	// Rewrite file
	newEntries := append([]SessionEntry{compactedEntry}, recent...)
	tmpPath := s.path + ".tmp"
	tf, err := os.Create(tmpPath)
	if err != nil {
		return
	}
	enc := json.NewEncoder(tf)
	for _, e := range newEntries {
		enc.Encode(e)
	}
	tf.Close()

	os.Rename(tmpPath, s.path)
	s.count = len(newEntries)
}

func (s *Session) readEntriesLocked() ([]SessionEntry, error) {
	f, err := os.Open(s.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []SessionEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			entries = append(entries, entry)
		}
	}
	return entries, scanner.Err()
}

func buildCompactionParts(entries []SessionEntry) (combined string, realCount int, recent []SessionEntry) {
	splitAt := len(entries) - compactKeepRecent
	old := entries[:splitAt]
	recent = entries[splitAt:]

	for _, e := range old {
		if e.Role == "_compacted" {
			if len(combined) < 4000 {
				combined += "[previous summary] " + e.Summary + "\n"
			}
		} else if e.Role != "system" {
			realCount++
			if len(combined) < 4000 {
				preview := e.Content
				if len(preview) > 200 {
					preview = preview[:200] + "..."
				}
				combined += fmt.Sprintf("[%s] %s\n", e.Role, preview)
			}
		}
	}
	if len(combined) > 4000 {
		combined = combined[:4000]
	}

	return combined, realCount, recent
}

// Delete removes the history file.
func (s *Session) Delete() {
	s.mu.Lock()
	defer s.mu.Unlock()
	os.Remove(s.path)
}

// Rename moves the on-disk history file when a thread's id changes.
// Best-effort: a missing source file (the thread had no entries yet)
// is treated as success so the caller can rename the in-memory record
// without worrying about whether disk state existed.
func (s *Session) Rename(newThreadID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	dir := filepath.Dir(s.path)
	newPath := filepath.Join(dir, newThreadID+".jsonl")
	if newPath == s.path {
		return nil
	}
	if _, err := os.Stat(s.path); err == nil {
		if err := os.Rename(s.path, newPath); err != nil {
			return err
		}
	}
	s.path = newPath
	return nil
}

// Count returns the approximate number of entries.
func (s *Session) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}
