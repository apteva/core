package core

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
	"time"
)

// BlobStore is an in-process, in-memory store for binary payloads that
// flow between MCP tools without ever entering the LLM context.
//
// When an MCP tool returns a JSON envelope of the form
//
//	{"_binary": true, "base64": "...", "mimeType": "...", "size": N}
//
// the store keeps the decoded bytes and the tool-result text is rewritten
// to a compact handle:
//
//	{"_file": true, "ref": "blobref://<id>", "mimeType": "...", "size": N}
//
// The LLM reads and references the handle. When it calls another tool
// and passes either the scalar "blobref://<id>" or an object
// {"_file_ref": "..."} as an argument value, the store rehydrates the
// value back into the full _binary envelope before dispatch — so the
// downstream tool sees real bytes and the bytes never traverse the
// LLM boundary.
//
// State is in-memory only. Blobs age out after the configured TTL and
// are capped in aggregate size; the oldest blob is evicted first when
// the cap is reached on Put.
type BlobStore struct {
	closeOnce sync.Once
	mu        sync.Mutex
	blobs     map[string]*blobEntry
	total     int64
	maxTotal  int64
	ttl       time.Duration
	quit      chan struct{}
}

type blobEntry struct {
	mime    string
	data    []byte
	size    int
	created time.Time
}

const blobRefPrefix = "blobref://"

// Default caps — liberal enough for typical audio/small-video flows,
// strict enough that a runaway agent can't swallow the process.
const (
	DefaultBlobMaxTotal = int64(256 * 1024 * 1024) // 256 MB across all live blobs
	DefaultBlobTTL      = 30 * time.Minute         // age-based eviction
)

func NewBlobStore(maxTotal int64, ttl time.Duration) *BlobStore {
	if maxTotal <= 0 {
		maxTotal = DefaultBlobMaxTotal
	}
	if ttl <= 0 {
		ttl = DefaultBlobTTL
	}
	bs := &BlobStore{
		blobs:    make(map[string]*blobEntry),
		maxTotal: maxTotal,
		ttl:      ttl,
		quit:     make(chan struct{}),
	}
	go bs.janitor()
	return bs
}

func (bs *BlobStore) Close() { bs.closeOnce.Do(func() { close(bs.quit) }) }

func (bs *BlobStore) janitor() {
	t := time.NewTicker(30 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-t.C:
			bs.evictExpired()
		case <-bs.quit:
			return
		}
	}
}

func (bs *BlobStore) evictExpired() {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	cutoff := time.Now().Add(-bs.ttl)
	for id, b := range bs.blobs {
		if b.created.Before(cutoff) {
			bs.total -= int64(b.size)
			delete(bs.blobs, id)
		}
	}
}

// Put stores bytes and returns a ref of the form "blobref://<id>".
// Evicts the oldest blob(s) if the aggregate cap would be exceeded.
func (bs *BlobStore) Put(mime string, data []byte) string {
	if int64(len(data)) > bs.maxTotal {
		return ""
	}
	bs.mu.Lock()
	defer bs.mu.Unlock()

	for bs.total+int64(len(data)) > bs.maxTotal && len(bs.blobs) > 0 {
		var oldestID string
		var oldestTime time.Time
		first := true
		for id, b := range bs.blobs {
			if first || b.created.Before(oldestTime) {
				oldestID = id
				oldestTime = b.created
				first = false
			}
		}
		if b, ok := bs.blobs[oldestID]; ok {
			bs.total -= int64(b.size)
			delete(bs.blobs, oldestID)
		}
	}

	id := randomBlobID()
	bs.blobs[id] = &blobEntry{
		mime:    mime,
		data:    data,
		size:    len(data),
		created: time.Now(),
	}
	bs.total += int64(len(data))
	return blobRefPrefix + id
}

// Get retrieves bytes and the original mimeType. Accepts either the
// full "blobref://<id>" ref or a bare id.
func (bs *BlobStore) Get(ref string) ([]byte, string, bool) {
	id := strings.TrimPrefix(ref, blobRefPrefix)
	bs.mu.Lock()
	defer bs.mu.Unlock()
	b, ok := bs.blobs[id]
	if !ok {
		return nil, "", false
	}
	return b.data, b.mime, true
}

// Count returns the number of live blobs. Test/observability helper.
func (bs *BlobStore) Count() int {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return len(bs.blobs)
}

func randomBlobID() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// ---------- Tool-result rewriting (MCP → LLM) ----------

type binaryEnvelope struct {
	Binary   bool   `json:"_binary"`
	Base64   string `json:"base64"`
	MimeType string `json:"mimeType"`
	Size     int    `json:"size"`
}

// RewriteBinaryToHandle inspects a tool-result string. If it parses as
// a JSON envelope with "_binary":true, the bytes are stored in the
// BlobStore and a compact "_file" handle string is returned in place
// of the original envelope. Non-binary text is returned unchanged.
//
// The fast path (string does not contain "_binary") costs an O(n)
// substring search and no allocations, so this is cheap to call on
// every MCP tool result.
func (bs *BlobStore) RewriteBinaryToHandle(text string) string {
	if bs == nil || !strings.Contains(text, `"_binary"`) {
		return text
	}
	trimmed := strings.TrimSpace(text)
	var env binaryEnvelope
	if err := json.Unmarshal([]byte(trimmed), &env); err != nil || !env.Binary {
		return text
	}
	data, err := base64.StdEncoding.DecodeString(env.Base64)
	if err != nil {
		return text
	}
	ref := bs.Put(env.MimeType, data)
	handle := map[string]any{
		"_file":    true,
		"ref":      ref,
		"mimeType": env.MimeType,
		"size":     env.Size,
	}
	out, _ := json.Marshal(handle)
	return string(out)
}

// ---------- Tool-arg rehydration (LLM → MCP) ----------

// RehydrateFileRefs walks arg values and, for any value that references
// a known blob, replaces it with a full _binary envelope JSON string.
// Two reference forms are accepted:
//
//  1. The scalar "blobref://<id>" — the LLM passed the ref string
//     directly as an argument value.
//  2. A JSON object {"_file_ref": "blobref://<id>"} — the LLM wrapped
//     the ref to make intent explicit. The bare id form (without the
//     blobref:// prefix) is also accepted inside _file_ref.
//
// Unknown refs (expired / unknown ids) are left untouched so the
// downstream tool produces a clear error rather than a silent
// corruption. The original args map is not mutated; a new map is
// returned.
func (bs *BlobStore) RehydrateFileRefs(args map[string]string) map[string]string {
	if bs == nil || len(args) == 0 {
		return args
	}
	out := make(map[string]string, len(args))
	for k, v := range args {
		out[k] = bs.rehydrateValue(v)
	}
	return out
}

func (bs *BlobStore) rehydrateValue(v string) string {
	if strings.HasPrefix(v, blobRefPrefix) {
		if env := bs.makeEnvelope(v); env != "" {
			return env
		}
		return v
	}
	trimmed := strings.TrimSpace(v)
	if !strings.HasPrefix(trimmed, "{") || !strings.Contains(trimmed, `"_file_ref"`) {
		return v
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		return v
	}
	ref, ok := obj["_file_ref"].(string)
	if !ok || ref == "" {
		return v
	}
	if !strings.HasPrefix(ref, blobRefPrefix) {
		ref = blobRefPrefix + ref
	}
	if env := bs.makeEnvelope(ref); env != "" {
		return env
	}
	return v
}

func (bs *BlobStore) makeEnvelope(ref string) string {
	data, mime, ok := bs.Get(ref)
	if !ok {
		return ""
	}
	env := map[string]any{
		"_binary":  true,
		"base64":   base64.StdEncoding.EncodeToString(data),
		"mimeType": mime,
		"size":     len(data),
	}
	out, _ := json.Marshal(env)
	return string(out)
}

// blobPromptHint is appended to the system prompt whenever a BlobStore
// is attached, so the LLM knows how to recognize handles in tool
// results and how to reference them in subsequent tool arguments.
const blobPromptHint = `

[FILE HANDLES]
When a tool result is an object {"_file": true, "ref": "blobref://...", "mimeType": ..., "size": ...}, the raw bytes have been stashed server-side and this handle is a reference to them.
To pass the file to another tool, set that tool's argument to either the scalar "blobref://..." string or the object {"_file_ref": "blobref://..."}. The bytes are injected on your behalf — do NOT decode, base64, or inline the payload yourself. Handles stay valid within the same session.`
