package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// FileRef uses the same handle as blob-producing tools. Core retains metadata;
// a known local blob is resolved locally, otherwise the gateway resolves it.
type FileRef struct {
	File     bool   `json:"_file"`
	Ref      string `json:"ref"`
	Filename string `json:"filename,omitempty"`
	MimeType string `json:"mimeType"`
	Size     int64  `json:"size"`
}

// Require an explicit size (zero is a valid empty file), and reject byte
// payload fields rather than silently treating them as reference metadata.
func (ref *FileRef) UnmarshalJSON(data []byte) error {
	type metadata FileRef
	var decoded metadata
	wire := struct {
		*metadata
		Size       *int64          `json:"size"`
		LegacyMIME string          `json:"mime_type"`
		Binary     json.RawMessage `json:"_binary"`
		Base64     json.RawMessage `json:"base64"`
		Data       json.RawMessage `json:"data"`
	}{metadata: &decoded}
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return fmt.Errorf("file_ref must be one JSON object")
	}
	if wire.Size == nil {
		return fmt.Errorf("file_ref.size required")
	}
	if len(wire.Binary) > 0 || len(wire.Base64) > 0 || len(wire.Data) > 0 {
		return fmt.Errorf("file_ref must contain metadata, not bytes")
	}
	if decoded.MimeType == "" {
		decoded.MimeType = wire.LegacyMIME
	}
	decoded.File = true
	decoded.Size = *wire.Size
	*ref = FileRef(decoded)
	return nil
}

func validateFileRef(ref *FileRef) error {
	if ref == nil || strings.TrimSpace(ref.Ref) == "" {
		return fmt.Errorf("file_ref.ref required")
	}
	if strings.TrimSpace(ref.Ref) != ref.Ref || strings.ContainsAny(ref.Ref, "\r\n\x00") {
		return fmt.Errorf("file_ref.ref must be an opaque reference without surrounding whitespace or control separators")
	}
	if (strings.HasPrefix(ref.Ref, blobRefPrefix) && len(strings.TrimPrefix(ref.Ref, blobRefPrefix)) == 0) ||
		(strings.HasPrefix(ref.Ref, "apteva-file://") && len(strings.TrimPrefix(ref.Ref, "apteva-file://")) == 0) {
		return fmt.Errorf("file_ref.ref must include an opaque identifier")
	}
	if !strings.HasPrefix(ref.Ref, blobRefPrefix) && !strings.HasPrefix(ref.Ref, "apteva-file://") {
		return fmt.Errorf("file_ref.ref must be a blobref:// handle")
	}
	if strings.TrimSpace(ref.MimeType) == "" {
		return fmt.Errorf("file_ref.mimeType required")
	}
	if ref.Size < 0 {
		return fmt.Errorf("file_ref.size must be nonnegative")
	}
	return nil
}

func fileRefText(ref *FileRef) string {
	if ref == nil {
		return ""
	}
	var metadata bytes.Buffer
	encoder := json.NewEncoder(&metadata)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(ref)
	return "[FILE HANDLE] " + strings.TrimSuffix(metadata.String(), "\n") + "\nPass the ref unchanged, or the complete _file handle, to a compatible tool's file argument. Inputs and tool-produced files use the same blobref:// contract. This is metadata only; file contents are supplied by the runtime."
}

type blobCallerThreadKey struct{}

func withBlobCallerThread(ctx context.Context, threadID string) context.Context {
	return context.WithValue(ctx, blobCallerThreadKey{}, threadID)
}

func appendFileRefText(text string, parts []ContentPart) string {
	for _, part := range parts {
		if part.Type == "file_ref" && part.FileRef != nil {
			if text != "" {
				text += "\n"
			}
			text += fileRefText(part.FileRef)
		}
	}
	return text
}

// fileRefsForModel renders only metadata into provider-supported text parts.
// The structured originals remain in the inbox, session, and retry state.
func fileRefsForModel(parts []ContentPart) []ContentPart {
	var out []ContentPart
	for i, part := range parts {
		if part.Type == "file_ref" {
			if out == nil {
				out = append(make([]ContentPart, 0, len(parts)), parts[:i]...)
			}
			out = append(out, ContentPart{Type: "text", Text: fileRefText(part.FileRef)})
		} else if out != nil {
			out = append(out, part)
		}
	}
	if out == nil {
		return parts
	}
	return out
}

func fileRefMessagesForModel(messages []Message) []Message {
	var out []Message
	for i, msg := range messages {
		for _, part := range msg.Parts {
			if part.Type != "file_ref" {
				continue
			}
			if out == nil {
				out = append([]Message(nil), messages...)
			}
			out[i].Parts = fileRefsForModel(msg.Parts)
			break
		}
	}
	if out == nil {
		return messages
	}
	return out
}

// summaryWithFileRefs keeps handles independent of generated summary text.
// Summarizers may omit details; opaque references must never be reconstructed.
func summaryWithFileRefs(content string, history []Message) Message {
	msg := Message{Role: "user", Content: content}
	seen := map[FileRef]bool{}
	for _, old := range history {
		for _, part := range old.Parts {
			if part.Type != "file_ref" || part.FileRef == nil || seen[*part.FileRef] {
				continue
			}
			seen[*part.FileRef] = true
			if len(msg.Parts) == 0 {
				msg.Parts = []ContentPart{{Type: "text", Text: content}}
			}
			msg.Parts = append(msg.Parts, cloneContentParts([]ContentPart{part})...)
		}
	}
	return msg
}
