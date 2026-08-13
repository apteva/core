package core

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	defaultLoadTail         = 50  // messages loaded into context on startup
	compactThreshold        = 500 // trigger compaction when file exceeds this
	compactKeepRecent       = 100 // keep this many recent messages after compaction
	historyDir              = "history"
	maxSessionEntryBytes    = 32 << 20
	maxCompactionInputBytes = 128 << 10
)

// SessionEntry is one line in the JSONL history file.
type SessionEntry struct {
	// Sequence is a per-session, monotonically increasing cursor. It makes the
	// durable history safe for consumers that need to observe a running thread:
	// unlike the in-memory context window, this cursor does not move when old
	// messages are trimmed from the model prompt.
	Sequence      int64                  `json:"seq,omitempty"`
	Timestamp     time.Time              `json:"ts"`
	Role          string                 `json:"role"` // "system", "user", "assistant", "tool_result", "_compacted"
	Content       string                 `json:"content"`
	Parts         []ContentPart          `json:"parts,omitempty"`
	ToolCalls     []NativeToolCall       `json:"tool_calls,omitempty"`
	ToolResults   []ToolResult           `json:"tool_results,omitempty"`
	Reasoning     string                 `json:"reasoning,omitempty"`
	ProviderState *ProviderResponseState `json:"provider_state,omitempty"`
	EventIDs      []string               `json:"event_ids,omitempty"`
	Summary       string                 `json:"summary,omitempty"`        // for _compacted entries
	OrigCount     int                    `json:"original_count,omitempty"` // how many messages were compacted
	TokensIn      int                    `json:"tokens_in,omitempty"`
	TokensOut     int                    `json:"tokens_out,omitempty"`
	Iteration     int                    `json:"iteration,omitempty"`
}

// Session manages persistent JSONL history for one thread.
type Session struct {
	mu        sync.Mutex
	compactMu sync.Mutex
	path      string
	threadID  string
	archive   *ToolResultArchive
	count     int // approximate line count
	nextSeq   int64
}

// NewSession creates or opens a session file for a thread.
func NewSession(baseDir, threadID string) *Session {
	dir := filepath.Join(baseDir, historyDir)
	_ = os.MkdirAll(dir, 0700)
	_ = os.Chmod(dir, 0700)
	path := safeSessionPath(dir, threadID)

	s := &Session{
		path:     path,
		threadID: threadID,
		archive:  NewToolResultArchive(baseDir, threadID),
	}

	// Count existing lines
	if f, err := os.Open(path); err == nil {
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 64*1024), maxSessionEntryBytes)
		for scanner.Scan() {
			s.count++
			var entry SessionEntry
			if json.Unmarshal(scanner.Bytes(), &entry) == nil && entry.Sequence > s.nextSeq {
				s.nextSeq = entry.Sequence
			}
		}
		f.Close()
	}

	return s
}

func safeSessionPath(dir, threadID string) string {
	candidate := filepath.Join(dir, threadID+".jsonl")
	rel, err := filepath.Rel(dir, candidate)
	if err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return candidate
	}
	sum := sha256.Sum256([]byte(threadID))
	return filepath.Join(dir, fmt.Sprintf("invalid-%x.jsonl", sum[:8]))
}

// Append writes one entry to the history file.
func (s *Session) Append(entry SessionEntry) error {
	var err error
	entry, err = s.prepareEntryForDurability(entry)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	if entry.Sequence <= 0 {
		s.nextSeq++
		entry.Sequence = s.nextSeq
	} else if entry.Sequence > s.nextSeq {
		s.nextSeq = entry.Sequence
	}

	f, err := os.OpenFile(s.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_ = f.Chmod(0600)
	defer f.Close()

	if err := json.NewEncoder(f).Encode(entry); err != nil {
		return err
	}
	s.count++
	return nil
}

// ArchiveToolResults stores each payload before it enters the in-memory
// conversation. The returned live message keeps the full payload through the
// short retention window; persistence stores only internal archive metadata.
func (s *Session) ArchiveToolResults(msg Message) (Message, error) {
	if s == nil || len(msg.ToolResults) == 0 {
		return msg, nil
	}
	archived := msg
	archived.ToolResults = append([]ToolResult(nil), msg.ToolResults...)
	for i := range archived.ToolResults {
		if !shouldArchiveToolResult(archived.ToolResults[i]) {
			continue
		}
		result, err := s.archive.Archive(archived.ToolResults[i])
		if err != nil {
			return msg, err
		}
		archived.ToolResults[i] = result
	}
	return archived, nil
}

func (s *Session) prepareEntryForDurability(entry SessionEntry) (SessionEntry, error) {
	if s == nil || len(entry.ToolResults) == 0 {
		return entry, nil
	}
	entry.ToolResults = append([]ToolResult(nil), entry.ToolResults...)
	for i := range entry.ToolResults {
		if !shouldArchiveToolResult(entry.ToolResults[i]) {
			continue
		}
		result, err := s.archive.Archive(entry.ToolResults[i])
		if err != nil {
			return SessionEntry{}, err
		}
		entry.ToolResults[i] = metadataOnlyArchivedToolResult(result)
	}
	return entry, nil
}

// projectEntriesForModelLoad migrates legacy full-payload entries into the
// archive and always returns bounded projections. If an archive write fails,
// the in-memory fallback is still bounded and contains no image bytes.
func (s *Session) projectEntriesForModelLoad(entries []SessionEntry) []SessionEntry {
	projected := make([]SessionEntry, len(entries))
	for i, entry := range entries {
		projected[i] = entry
		if len(entry.ToolResults) == 0 {
			continue
		}
		projected[i].ToolResults = append([]ToolResult(nil), entry.ToolResults...)
		for j, original := range entry.ToolResults {
			if !shouldArchiveToolResult(original) {
				continue
			}
			archived := original
			var err error
			if isToolResultSHA256(original.SHA256) {
				var object archivedToolResultObject
				ref := original.ArchiveRef
				if ref == "" {
					ref = original.SHA256
				}
				object, err = s.archive.Read(ref)
				if err == nil {
					archived.Content = object.Content
					archived.Image = append([]byte(nil), object.Image...)
					archived.IsError = object.IsError
					archived.OriginalChars = object.OriginalChars
					archived.OriginalBytes = object.OriginalBytes
					archived.OriginalImageBytes = object.OriginalImageBytes
					archived.ContentIsPreview = false
				}
			} else {
				archived, err = s.archive.Archive(original)
			}
			if err != nil {
				archived = original
				archived.Image = nil
				archived.Content = deterministicToolResultExcerpt(original.Content, historicalToolResultPerResultChars)
				archived.ContentIsPreview = true
			}
			projected[i].ToolResults[j] = projectArchivedToolResult(archived, historicalToolResultPerResultChars)
		}
	}
	return projected
}

func (s *Session) archiveEntriesForDurability(entries []SessionEntry) ([]SessionEntry, bool, error) {
	archivedEntries := make([]SessionEntry, len(entries))
	changed := false
	for i, entry := range entries {
		archivedEntries[i] = entry
		if len(entry.ToolResults) == 0 {
			continue
		}
		archivedEntries[i].ToolResults = append([]ToolResult(nil), entry.ToolResults...)
		for j, original := range entry.ToolResults {
			if !shouldArchiveToolResult(original) {
				continue
			}
			if isToolResultSHA256(original.SHA256) && original.Content == "" && len(original.Image) == 0 && original.ContentIsPreview {
				continue
			}
			archived, err := s.archive.Archive(original)
			if err != nil {
				return nil, false, fmt.Errorf("archive legacy tool result %q: %w", original.CallID, err)
			}
			archivedEntries[i].ToolResults[j] = metadataOnlyArchivedToolResult(archived)
			changed = true
		}
	}
	return archivedEntries, changed, nil
}

// AppendMessage is a convenience to append a Message as a SessionEntry.
func (s *Session) AppendMessage(msg Message, iteration int, usage TokenUsage) error {
	entry := SessionEntry{
		Timestamp:     time.Now(),
		Role:          msg.Role,
		Content:       msg.Content,
		Parts:         msg.Parts,
		ToolCalls:     msg.ToolCalls,
		ToolResults:   msg.ToolResults,
		Reasoning:     msg.Reasoning,
		ProviderState: msg.ProviderState,
		EventIDs:      append([]string(nil), msg.EventIDs...),
		TokensIn:      usage.PromptTokens,
		TokensOut:     usage.CompletionTokens,
		Iteration:     iteration,
	}
	return s.Append(entry)
}

// EventIDs returns durable API inbox IDs already represented in the session.
// It is used only during thread restoration to close the crash window between
// appending the user message and marking its inbox record consumed.
func (s *Session) EventIDs() map[string]bool {
	ids := map[string]bool{}
	if s == nil {
		return ids
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.Open(s.path)
	if err != nil {
		return ids
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), maxSessionEntryBytes)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) != nil {
			continue
		}
		for _, id := range entry.EventIDs {
			ids[id] = true
		}
	}
	return ids
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
	migrationSafe := true
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), maxSessionEntryBytes)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			entries = append(entries, entry)
		} else {
			migrationSafe = false
		}
	}
	if scanner.Err() != nil {
		migrationSafe = false
	}
	entries = sanitizeLegacyDynamicEntries(entries)
	durableEntries, migrated, archiveErr := s.archiveEntriesForDurability(entries)
	if archiveErr == nil {
		entries = durableEntries
		if migrated && migrationSafe {
			if err := s.rewriteEntriesLocked(entries); err == nil {
				s.count = len(entries)
			}
		}
	}
	entries = s.projectEntriesForModelLoad(entries)

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
			Role:          e.Role,
			Content:       e.Content,
			Parts:         e.Parts,
			ToolCalls:     e.ToolCalls,
			ToolResults:   e.ToolResults,
			Reasoning:     e.Reasoning,
			ProviderState: e.ProviderState,
			EventIDs:      append([]string(nil), e.EventIDs...),
		}
		// Normalize role: "tool_result" → "user" with ToolResults
		if e.Role == "tool_result" {
			msg.Role = "user"
		}
		messages = append(messages, msg)
	}

	// Sanitize: remove orphaned tool_results that have no matching tool_use
	messages = sanitizeToolPairs(messages)
	// Computer screenshots are transient. Old session files may contain raw
	// frames from before this policy; never reload them into a new process.
	evictStaleComputerScreenshots(messages, 0)

	return messages, compactedSummaries
}

// LoadAfter returns durable session entries strictly newer than after. The
// cursor is the entry Sequence, not the current in-memory context index, so a
// consumer can keep observing a long-running thread even while its prompt is
// compacted or trimmed. Entries written before sequence support return no
// cursor and are intentionally excluded; callers use this endpoint only for
// fresh, live sessions.
func (s *Session) LoadAfter(after int64, limit int) (entries []SessionEntry, nextCursor int64, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	all, err := s.readEntriesLocked()
	if err != nil {
		if os.IsNotExist(err) {
			return []SessionEntry{}, after, nil
		}
		return nil, after, err
	}
	if limit <= 0 {
		limit = 500
	}
	capacity := limit
	if len(all) < capacity {
		capacity = len(all)
	}
	entries = make([]SessionEntry, 0, capacity)
	nextCursor = after
	for _, entry := range all {
		if entry.Sequence <= after {
			continue
		}
		archived, _, archiveErr := s.archiveEntriesForDurability([]SessionEntry{entry})
		if archiveErr != nil {
			return nil, nextCursor, archiveErr
		}
		entries = append(entries, archived[0])
		if entry.Sequence > nextCursor {
			nextCursor = entry.Sequence
		}
		if len(entries) >= limit {
			break
		}
	}
	return entries, nextCursor, nil
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
				// ProviderState is an atomic copy of the original Responses
				// output. Once calls are removed, replaying it would resurrect
				// the orphaned calls; fall back to reconstruction instead.
				m.ProviderState = nil
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
	s.compact(compactKeepRecent, false, summarize)
}

func (s *Session) ForceCompact(keepRecent int, summarize func(text string) string) {
	if keepRecent < 1 {
		keepRecent = 1
	}
	s.compact(keepRecent, true, summarize)
}

func (s *Session) compact(keepRecent int, force bool, summarize func(text string) string) {
	s.compactMu.Lock()
	defer s.compactMu.Unlock()
	s.mu.Lock()
	entries, err := s.readEntriesLocked()
	if err != nil {
		s.mu.Unlock()
		return
	}

	entries = sanitizeLegacyDynamicEntries(entries)
	entries, _, err = s.archiveEntriesForDurability(entries)
	if err != nil {
		s.mu.Unlock()
		return
	}
	if !force && len(entries) <= compactThreshold {
		s.count = len(entries)
		s.mu.Unlock()
		return
	}
	if force && len(entries) <= keepRecent {
		s.count = len(entries)
		s.mu.Unlock()
		return
	}

	compactionEntries := s.projectEntriesForModelLoad(entries)
	combined, realCount, _ := buildCompactionParts(compactionEntries, keepRecent)
	compactPrefix := len(entries) - keepRecent
	if compactPrefix < 0 {
		compactPrefix = 0
	}
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
	if summarize != nil && combined != "" && summaryText == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Re-read after the unlocked summarization window so entries appended
	// concurrently are preserved when the file is rewritten.
	entries, err = s.readEntriesLocked()
	if err != nil {
		return
	}
	entries = sanitizeLegacyDynamicEntries(entries)
	entries, _, err = s.archiveEntriesForDurability(entries)
	if err != nil {
		return
	}
	if !force && len(entries) <= compactThreshold {
		s.count = len(entries)
		return
	}
	if force && len(entries) <= keepRecent {
		s.count = len(entries)
		return
	}
	if compactPrefix > len(entries) {
		return
	}
	recent := append([]SessionEntry(nil), entries[compactPrefix:]...)
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
	if s.rewriteEntriesLocked(newEntries) == nil {
		s.count = len(newEntries)
	}
}

func (s *Session) rewriteEntriesLocked(entries []SessionEntry) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	for _, e := range entries {
		if err := enc.Encode(e); err != nil {
			return err
		}
	}
	return atomicWriteFile(s.path, buf.Bytes(), 0600)
}

func (s *Session) readEntriesLocked() ([]SessionEntry, error) {
	f, err := os.Open(s.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var entries []SessionEntry
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), maxSessionEntryBytes)
	for scanner.Scan() {
		var entry SessionEntry
		if json.Unmarshal(scanner.Bytes(), &entry) == nil {
			entries = append(entries, entry)
		}
	}
	return entries, scanner.Err()
}

func buildCompactionParts(entries []SessionEntry, keepRecent int) (combined string, realCount int, recent []SessionEntry) {
	entries = sanitizeLegacyDynamicEntries(entries)
	if keepRecent < 1 {
		keepRecent = 1
	}
	if keepRecent > len(entries) {
		keepRecent = len(entries)
	}
	splitAt := len(entries) - keepRecent
	old := entries[:splitAt]
	recent = entries[splitAt:]

	for _, e := range old {
		if e.Role == "_compacted" {
			combined += "[previous summary]\n" + excerptForCompaction(e.Summary, compactionInputMessagePreviewChars) + "\n"
		} else if e.Role != "system" {
			realCount++
			combined += fmt.Sprintf("[%s]\n%s\n", e.Role, excerptForCompaction(e.Content, compactionInputMessagePreviewChars))
			if len(e.ToolCalls) > 0 {
				combined += "Tool calls:\n"
				for _, call := range e.ToolCalls {
					combined += fmt.Sprintf("- id=%s name=%s args=%v\n", call.ID, call.Name, call.Args)
				}
			}
			if len(e.ToolResults) > 0 {
				combined += "Tool results:\n"
				for _, result := range e.ToolResults {
					combined += fmt.Sprintf("- call_id=%s is_error=%v content:\n%s\n", result.CallID, result.IsError, excerptForCompaction(result.Content, toolResultContextPreviewChars))
				}
			}
		}
		if len(combined) >= maxCompactionInputBytes {
			break
		}
	}
	if len(combined) > maxCompactionInputBytes {
		combined = combined[:maxCompactionInputBytes]
	}

	return combined, realCount, recent
}

func sanitizeLegacyDynamicEntries(entries []SessionEntry) []SessionEntry {
	cleaned := make([]SessionEntry, 0, len(entries))
	for _, entry := range entries {
		if legacyEntryTargetsSystemThread(entry) {
			continue
		}
		content, systemSynthetic := stripLegacySystemThreadEvents(entry.Content)
		content, dynamicSynthetic := stripLegacyDynamicContext(content)
		synthetic := systemSynthetic || dynamicSynthetic
		if !synthetic {
			cleaned = append(cleaned, entry)
			continue
		}
		entry.Content = content
		if len(entry.Parts) > 0 {
			parts := make([]ContentPart, 0, len(entry.Parts))
			for _, part := range entry.Parts {
				if part.Type == "text" {
					if content != "" {
						part.Text = content
						parts = append(parts, part)
					}
					continue
				}
				parts = append(parts, part)
			}
			entry.Parts = parts
		}
		if content == "" && len(entry.Parts) == 0 && len(entry.ToolCalls) == 0 && len(entry.ToolResults) == 0 {
			continue
		}
		cleaned = append(cleaned, entry)
	}
	return cleaned
}

func legacyEntryTargetsSystemThread(entry SessionEntry) bool {
	for _, call := range entry.ToolCalls {
		if call.Args["id"] == "unconscious" {
			return true
		}
	}
	return false
}

// stripLegacySystemThreadEvents removes unconscious-thread startup notices
// written by cores that predate the System spawn visibility guard. Those
// notices disclosed a platform-managed target and its privileged tools to the
// main model. Keep other events from the same batch intact.
func stripLegacySystemThreadEvents(content string) (string, bool) {
	if !strings.Contains(content, "[thread:unconscious] started") {
		return content, false
	}
	lines := strings.Split(content, "\n")
	kept := make([]string, 0, len(lines))
	removed := false
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "• [thread:unconscious] started") {
			removed = true
			continue
		}
		kept = append(kept, line)
	}
	if !removed {
		return content, false
	}
	result := strings.TrimSpace(strings.Join(kept, "\n"))
	if strings.HasSuffix(result, "Events:") {
		result = ""
	}
	return result, true
}

// Delete removes the history file.
func (s *Session) Delete() {
	s.mu.Lock()
	defer s.mu.Unlock()
	os.Remove(s.path)
}

// Reset removes the durable conversation journal while keeping the Session
// object usable. Reusing the same object matters for live thinkers: replacing
// the pointer during a reset can race an in-flight append and resurrect the
// deleted history through the old Session instance.
func (s *Session) Reset() error {
	s.compactMu.Lock()
	defer s.compactMu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.Remove(s.path); err != nil && !os.IsNotExist(err) {
		return err
	}
	s.count = 0
	s.nextSeq = 0
	return nil
}

// Rename moves the on-disk history file when a thread's id changes.
// Best-effort: a missing source file (the thread had no entries yet)
// is treated as success so the caller can rename the in-memory record
// without worrying about whether disk state existed.
func (s *Session) Rename(newThreadID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	dir := filepath.Dir(s.path)
	newPath := safeSessionPath(dir, newThreadID)
	if newPath == s.path {
		return nil
	}
	if _, err := os.Stat(s.path); err == nil {
		if err := os.Rename(s.path, newPath); err != nil {
			return err
		}
	}
	s.path = newPath
	if s.archive != nil {
		if err := s.archive.RenameThread(newThreadID); err != nil {
			return err
		}
	}
	s.threadID = newThreadID
	return nil
}

// Count returns the approximate number of entries.
func (s *Session) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.count
}
