package core

import (
	"fmt"
	"net/url"
	"path"
	"strings"
)

const transientAttachmentHistoryNotice = "[Transient attachments were supplied for immediate analysis; their content and access URLs are not retained in conversation history.]"

// parseAttachmentURLs is the compatibility adapter for tool arguments that
// carry whitespace-separated attachment URLs. Everything after this boundary
// operates on generic ContentParts rather than on the originating field, tool,
// application, host, or storage provider.
func parseAttachmentURLs(raw string) ([]ContentPart, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	fields := strings.Fields(raw)
	parts := make([]ContentPart, 0, len(fields))
	for i, candidate := range fields {
		parsed, err := url.Parse(candidate)
		if err != nil || parsed.Scheme == "" {
			return nil, fmt.Errorf("attachment %d is not an absolute URL", i+1)
		}
		if encodedQuerySeparators(parsed) {
			return nil, fmt.Errorf("attachment %d URL appears to contain percent-encoded query separators; copy the opaque URL exactly from its source and retry", i+1)
		}

		ext := strings.ToLower(strings.TrimPrefix(path.Ext(parsed.Path), "."))
		switch ext {
		case "mp3", "wav", "aac", "ogg", "flac", "aiff", "m4a":
			parts = append(parts, ContentPart{Type: "audio_url", AudioURL: &AudioURL{URL: candidate}})
		default:
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: candidate}})
		}
	}
	return parts, nil
}

// encodedQuerySeparators catches the failure mode where an opaque URL was
// copied through a model and every top-level '&' became '%26'. We deliberately
// reject rather than rewrite: decoding a legitimate escaped ampersand inside a
// query value would silently change the resource or invalidate its signature.
func encodedQuerySeparators(parsed *url.URL) bool {
	if parsed == nil || parsed.RawQuery == "" || strings.Contains(parsed.RawQuery, "&") ||
		!strings.Contains(strings.ToLower(parsed.RawQuery), "%26") {
		return false
	}
	decoded, err := url.QueryUnescape(parsed.RawQuery)
	if err != nil {
		return false
	}
	segments := strings.Split(decoded, "&")
	if len(segments) < 3 {
		return false
	}
	for _, segment := range segments[1:] {
		key, _, ok := strings.Cut(segment, "=")
		if !ok || strings.TrimSpace(key) == "" {
			return false
		}
	}
	return true
}

func transientAttachmentCount(messages []Message) int {
	count := 0
	for _, msg := range messages {
		for _, part := range msg.Parts {
			if part.Type != "text" {
				count++
			}
		}
	}
	return count
}

func projectTransientAttachmentsFromMessage(msg Message) (Message, int) {
	count := 0
	var textParts []string
	for _, part := range msg.Parts {
		if part.Type == "text" {
			if strings.TrimSpace(part.Text) != "" {
				textParts = append(textParts, part.Text)
			}
			continue
		}
		count++
	}
	if count == 0 {
		return msg, 0
	}

	projected := msg
	projected.Parts = nil
	if strings.TrimSpace(projected.Content) == "" && len(textParts) > 0 {
		projected.Content = strings.Join(textParts, "\n")
	}
	if !strings.Contains(projected.Content, transientAttachmentHistoryNotice) {
		if strings.TrimSpace(projected.Content) != "" {
			projected.Content = strings.TrimRight(projected.Content, "\n") + "\n\n"
		}
		projected.Content += transientAttachmentHistoryNotice
	}
	return projected, count
}

func projectTransientAttachmentsFromMessages(messages []Message) ([]Message, int) {
	if transientAttachmentCount(messages) == 0 {
		return messages, 0
	}
	projected := cloneMessages(messages)
	total := 0
	for i := range projected {
		var count int
		projected[i], count = projectTransientAttachmentsFromMessage(projected[i])
		total += count
	}
	return projected, total
}

func consumeTransientAttachments(messages []Message) int {
	total := 0
	for i := range messages {
		var count int
		messages[i], count = projectTransientAttachmentsFromMessage(messages[i])
		total += count
	}
	return total
}

func projectTransientAttachmentsFromEntry(entry SessionEntry) (SessionEntry, int) {
	projected, count := projectTransientAttachmentsFromMessage(Message{
		Content: entry.Content,
		Parts:   entry.Parts,
	})
	if count == 0 {
		return entry, 0
	}
	entry.Content = projected.Content
	entry.Parts = nil
	return entry, count
}

func projectTransientAttachmentsFromEntries(entries []SessionEntry) ([]SessionEntry, int) {
	projected := make([]SessionEntry, len(entries))
	total := 0
	for i, entry := range entries {
		var count int
		projected[i], count = projectTransientAttachmentsFromEntry(entry)
		total += count
	}
	return projected, total
}

func isProviderAttachmentInputError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unable to download the file") ||
		strings.Contains(msg, "error while downloading file") ||
		strings.Contains(msg, "failed to download image") ||
		strings.Contains(msg, "invalid image url") ||
		strings.Contains(msg, "invalid_image_url")
}

func (t *Thinker) emitAttachmentLifecycle(eventType string, count int, provider, reason string) {
	if t == nil || t.telemetry == nil || count == 0 {
		return
	}
	t.telemetry.Emit(eventType, t.threadID, map[string]any{
		"count":     count,
		"provider":  provider,
		"reason":    reason,
		"iteration": t.iteration,
	})
}
