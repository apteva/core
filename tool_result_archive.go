package core

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	toolResultArchiveVersion           = 1
	toolResultArchiveDir               = "tool-results"
	toolResultArchiveObjectDir         = "objects"
	toolResultArchiveCallsDir          = "calls"
	historicalToolResultPerResultChars = 2_000
	historicalToolResultTotalChars     = 64 * 1024
	maxArchivedToolResultObjectBytes   = 256 << 20
)

type archivedToolResultObject struct {
	Version            int       `json:"version"`
	SHA256             string    `json:"sha256"`
	Content            string    `json:"content"`
	Image              []byte    `json:"image,omitempty"`
	IsError            bool      `json:"is_error,omitempty"`
	OriginalChars      int       `json:"original_chars"`
	OriginalBytes      int       `json:"original_bytes"`
	OriginalImageBytes int       `json:"original_image_bytes,omitempty"`
	CreatedAt          time.Time `json:"created_at"`
}

type toolResultAuditRecord struct {
	Timestamp          time.Time `json:"ts"`
	ThreadID           string    `json:"thread_id"`
	CallID             string    `json:"call_id"`
	ToolName           string    `json:"tool_name,omitempty"`
	OriginalChars      int       `json:"original_chars"`
	OriginalBytes      int       `json:"original_bytes"`
	OriginalImageBytes int       `json:"original_image_bytes,omitempty"`
	SHA256             string    `json:"sha256"`
	ArchiveRef         string    `json:"archive_ref"`
	IsError            bool      `json:"is_error,omitempty"`
}

// ToolResultArchive stores immutable payload objects by content hash and an
// append-only call ledger for one agent thread. Session compaction never
// touches either path.
type ToolResultArchive struct {
	mu         sync.Mutex
	historyDir string
	threadID   string
	callsPath  string
	seenCalls  map[string]bool
	callRefs   map[string]toolResultAuditRecord
}

func NewToolResultArchive(baseDir, threadID string) *ToolResultArchive {
	historyPath := filepath.Join(baseDir, historyDir)
	root := filepath.Join(historyPath, toolResultArchiveDir)
	callsDir := filepath.Join(root, toolResultArchiveCallsDir)
	_ = os.MkdirAll(callsDir, 0700)
	_ = os.Chmod(root, 0700)
	_ = os.Chmod(callsDir, 0700)
	a := &ToolResultArchive{
		historyDir: historyPath,
		threadID:   threadID,
		callsPath:  safeSessionPath(callsDir, threadID),
		seenCalls:  map[string]bool{},
		callRefs:   map[string]toolResultAuditRecord{},
	}
	a.loadCallLedger()
	return a
}

func (a *ToolResultArchive) loadCallLedger() {
	f, err := os.Open(a.callsPath)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 16*1024), 1<<20)
	for scanner.Scan() {
		var record toolResultAuditRecord
		if json.Unmarshal(scanner.Bytes(), &record) != nil || record.CallID == "" || !isToolResultSHA256(record.SHA256) {
			continue
		}
		a.seenCalls[record.CallID+"\x00"+record.SHA256] = true
		a.callRefs[record.CallID] = record
	}
}

func toolResultPayloadSHA256(content string, image []byte, isError bool) string {
	h := sha256.New()
	_, _ = h.Write([]byte("apteva-tool-result-v1\x00"))
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(content)))
	_, _ = h.Write(length[:])
	_, _ = h.Write([]byte(content))
	binary.BigEndian.PutUint64(length[:], uint64(len(image)))
	_, _ = h.Write(length[:])
	_, _ = h.Write(image)
	if isError {
		_, _ = h.Write([]byte{1})
	} else {
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func isToolResultSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func toolResultArchiveRef(hash string) string {
	return filepath.ToSlash(filepath.Join(toolResultArchiveDir, toolResultArchiveObjectDir, "sha256", hash[:2], hash+".json"))
}

func (a *ToolResultArchive) objectPath(hash string) string {
	return filepath.Join(a.historyDir, filepath.FromSlash(toolResultArchiveRef(hash)))
}

func (a *ToolResultArchive) Archive(result ToolResult) (ToolResult, error) {
	if a == nil {
		return result, errors.New("tool result archive is unavailable")
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	if result.SHA256 == "" || result.ArchiveRef == "" {
		hash := toolResultPayloadSHA256(result.Content, result.Image, result.IsError)
		result.SHA256 = hash
		result.ArchiveRef = toolResultArchiveRef(hash)
		result.OriginalChars = utf8.RuneCountInString(result.Content)
		result.OriginalBytes = len(result.Content) + len(result.Image)
		result.OriginalImageBytes = len(result.Image)
		result.ContentIsPreview = false

		object := archivedToolResultObject{
			Version:            toolResultArchiveVersion,
			SHA256:             hash,
			Content:            result.Content,
			Image:              append([]byte(nil), result.Image...),
			IsError:            result.IsError,
			OriginalChars:      result.OriginalChars,
			OriginalBytes:      result.OriginalBytes,
			OriginalImageBytes: result.OriginalImageBytes,
			CreatedAt:          time.Now().UTC(),
		}
		if err := writeImmutableToolResultObject(a.objectPath(hash), object); err != nil {
			return result, err
		}
	} else if !isToolResultSHA256(result.SHA256) {
		return result, fmt.Errorf("invalid archived tool result sha256 %q", result.SHA256)
	} else {
		// Archive references are derived from the digest, never trusted from
		// conversation history. This also keeps previews deterministic.
		result.ArchiveRef = toolResultArchiveRef(result.SHA256)
	}

	if err := a.appendCallRecord(result); err != nil {
		return result, err
	}
	return result, nil
}

func writeImmutableToolResultObject(target string, object archivedToolResultObject) error {
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}
	_ = os.Chmod(dir, 0700)
	if _, err := os.Stat(target); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tool-result-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	_ = tmp.Chmod(0600)
	encErr := json.NewEncoder(tmp).Encode(object)
	if encErr == nil {
		encErr = tmp.Sync()
	}
	closeErr := tmp.Close()
	if encErr != nil {
		return encErr
	}
	if closeErr != nil {
		return closeErr
	}
	if err := os.Link(tmpPath, target); err != nil {
		if _, statErr := os.Stat(target); statErr == nil {
			return nil
		}
		return err
	}
	_ = os.Chmod(target, 0600)
	return nil
}

func (a *ToolResultArchive) appendCallRecord(result ToolResult) error {
	if result.CallID == "" {
		return nil
	}
	key := result.CallID + "\x00" + result.SHA256
	if a.seenCalls[key] {
		return nil
	}
	record := toolResultAuditRecord{
		Timestamp:          time.Now().UTC(),
		ThreadID:           a.threadID,
		CallID:             result.CallID,
		ToolName:           result.ToolName,
		OriginalChars:      result.OriginalChars,
		OriginalBytes:      result.OriginalBytes,
		OriginalImageBytes: result.OriginalImageBytes,
		SHA256:             result.SHA256,
		ArchiveRef:         result.ArchiveRef,
		IsError:            result.IsError,
	}
	f, err := os.OpenFile(a.callsPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_ = f.Chmod(0600)
	err = json.NewEncoder(f).Encode(record)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	a.seenCalls[key] = true
	a.callRefs[result.CallID] = record
	return nil
}

func (a *ToolResultArchive) Read(refOrHash string) (archivedToolResultObject, error) {
	if a == nil {
		return archivedToolResultObject{}, errors.New("tool result archive is unavailable")
	}
	hash := strings.TrimSpace(refOrHash)
	hash = strings.TrimPrefix(hash, "sha256:")
	if !isToolResultSHA256(hash) {
		base := filepath.Base(filepath.FromSlash(hash))
		hash = strings.TrimSuffix(base, filepath.Ext(base))
	}
	if !isToolResultSHA256(hash) {
		return archivedToolResultObject{}, fmt.Errorf("invalid archive reference %q", refOrHash)
	}

	f, err := os.Open(a.objectPath(hash))
	if err != nil {
		return archivedToolResultObject{}, err
	}
	defer f.Close()
	var object archivedToolResultObject
	if err := json.NewDecoder(io.LimitReader(f, maxArchivedToolResultObjectBytes)).Decode(&object); err != nil {
		return archivedToolResultObject{}, err
	}
	if object.Version != toolResultArchiveVersion || object.SHA256 != hash {
		return archivedToolResultObject{}, fmt.Errorf("archive object metadata mismatch for %s", hash)
	}
	if got := toolResultPayloadSHA256(object.Content, object.Image, object.IsError); got != hash {
		return archivedToolResultObject{}, fmt.Errorf("archive object checksum mismatch for %s", hash)
	}
	return object, nil
}

func (a *ToolResultArchive) RefForCall(callID string) (string, bool) {
	if a == nil {
		return "", false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	record, ok := a.callRefs[strings.TrimSpace(callID)]
	return record.ArchiveRef, ok
}

func deterministicToolResultExcerpt(content string, maxChars int) string {
	if maxChars <= 0 {
		return ""
	}
	if len(content) <= maxChars {
		return content
	}
	marker := "\n... [older tool-result content truncated by core] ...\n"
	if len(marker) >= maxChars {
		return validToolResultPrefix(content, maxChars)
	}
	available := maxChars - len(marker)
	head := available * 3 / 4
	tail := available - head
	return validToolResultPrefix(content, head) + marker + validToolResultSuffix(content, tail)
}

func validToolResultPrefix(content string, maxBytes int) string {
	if maxBytes >= len(content) {
		return content
	}
	if maxBytes <= 0 {
		return ""
	}
	end := maxBytes
	for end > 0 && !utf8.ValidString(content[:end]) {
		end--
	}
	return content[:end]
}

func validToolResultSuffix(content string, maxBytes int) string {
	if maxBytes >= len(content) {
		return content
	}
	if maxBytes <= 0 {
		return ""
	}
	start := len(content) - maxBytes
	for start < len(content) && !utf8.ValidString(content[start:]) {
		start++
	}
	return content[start:]
}

func projectArchivedToolResult(result ToolResult, maxChars int) ToolResult {
	projected := result
	projected.Image = nil
	projected.ContentIsPreview = true
	if maxChars <= 0 {
		projected.Content = ""
		return projected
	}
	if result.ContentIsPreview {
		projected.Content = deterministicToolResultExcerpt(result.Content, maxChars)
		return projected
	}
	header := strings.Join([]string{
		"[OLDER TOOL RESULT - truncated by core]",
		"tool=" + result.ToolName,
		"call_id=" + result.CallID,
		"original_chars=" + strconv.Itoa(result.OriginalChars),
		"original_bytes=" + strconv.Itoa(result.OriginalBytes),
		"preview:",
	}, "\n")
	if len(header) >= maxChars {
		projected.Content = validToolResultPrefix(header, maxChars)
		return projected
	}
	bodyBudget := maxChars - len(header) - 1
	projected.Content = header + "\n" + deterministicToolResultExcerpt(result.Content, bodyBudget)
	return projected
}

func metadataOnlyArchivedToolResult(result ToolResult) ToolResult {
	result.Content = ""
	result.Image = nil
	result.ContentIsPreview = true
	return result
}

// RenameThread keeps the old ledger path as an append-only audit alias while
// making the same records discoverable under the new thread ID after restart.
func (a *ToolResultArchive) RenameThread(newThreadID string) error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	newPath := safeSessionPath(filepath.Dir(a.callsPath), newThreadID)
	if newPath == a.callsPath {
		a.threadID = newThreadID
		return nil
	}
	if _, err := os.Stat(a.callsPath); err == nil {
		if _, targetErr := os.Stat(newPath); os.IsNotExist(targetErr) {
			// A hard link preserves the original append-only path and makes all
			// future appends visible through both audit names.
			if err := os.Link(a.callsPath, newPath); err != nil {
				return err
			}
		} else if targetErr != nil {
			return targetErr
		} else {
			data, readErr := os.ReadFile(a.callsPath)
			if readErr != nil {
				return readErr
			}
			f, openErr := os.OpenFile(newPath, os.O_APPEND|os.O_WRONLY, 0600)
			if openErr != nil {
				return openErr
			}
			_, writeErr := f.Write(data)
			if writeErr == nil {
				writeErr = f.Sync()
			}
			closeErr := f.Close()
			if writeErr != nil {
				return writeErr
			}
			if closeErr != nil {
				return closeErr
			}
		}
	} else if !os.IsNotExist(err) {
		return err
	}
	a.callsPath = newPath
	a.threadID = newThreadID
	return nil
}
