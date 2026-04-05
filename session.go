package main

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
	defaultLoadTail    = 50  // messages loaded into context on startup
	compactThreshold   = 500 // trigger compaction when file exceeds this
	compactKeepRecent  = 100 // keep this many recent messages after compaction
	historyDir         = "history"
)

// SessionEntry is one line in the JSONL history file.
type SessionEntry struct {
	Timestamp    time.Time        `json:"ts"`
	Role         string           `json:"role"`                    // "system", "user", "assistant", "tool_result", "_compacted"
	Content      string           `json:"content"`
	Parts        []ContentPart    `json:"parts,omitempty"`
	ToolCalls    []NativeToolCall `json:"tool_calls,omitempty"`
	ToolResults  []ToolResult     `json:"tool_results,omitempty"`
	Summary      string           `json:"summary,omitempty"`       // for _compacted entries
	OrigCount    int              `json:"original_count,omitempty"` // how many messages were compacted
	TokensIn     int              `json:"tokens_in,omitempty"`
	TokensOut    int              `json:"tokens_out,omitempty"`
	Iteration    int              `json:"iteration,omitempty"`
}

// Session manages persistent JSONL history for one thread.
type Session struct {
	mu       sync.Mutex
	path     string
	count    int // approximate line count
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

	return messages, compactedSummaries
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
	defer s.mu.Unlock()

	f, err := os.Open(s.path)
	if err != nil {
		return
	}

	var entries []SessionEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			entries = append(entries, entry)
		}
	}
	f.Close()

	if len(entries) <= compactThreshold {
		return
	}

	// Split: old entries to compact, recent to keep
	splitAt := len(entries) - compactKeepRecent
	old := entries[:splitAt]
	recent := entries[splitAt:]

	// Build text from old entries for summarization
	var textParts []string
	realCount := 0
	for _, e := range old {
		if e.Role == "_compacted" {
			textParts = append(textParts, "[previous summary] "+e.Summary)
			continue
		}
		if e.Role == "system" {
			continue
		}
		realCount++
		preview := e.Content
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		textParts = append(textParts, fmt.Sprintf("[%s] %s", e.Role, preview))
	}

	// Summarize
	summaryText := fmt.Sprintf("Compacted %d messages.", realCount)
	if summarize != nil && len(textParts) > 0 {
		combined := ""
		for _, p := range textParts {
			combined += p + "\n"
			if len(combined) > 4000 {
				break
			}
		}
		if result := summarize(combined); result != "" {
			summaryText = result
		}
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

// Delete removes the history file.
func (s *Session) Delete() {
	s.mu.Lock()
	defer s.mu.Unlock()
	os.Remove(s.path)
}

// Count returns the approximate number of entries.
func (s *Session) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}
