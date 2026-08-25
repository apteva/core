package core

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestParseAttachmentURLsPreservesOpaqueQueryExactly(t *testing.T) {
	raw := "https://objects.example.test/frame.jpg?algorithm=v1&credential=abc%2Fdef&date=20260825T080000Z&signature=123"
	parts, err := parseAttachmentURLs(raw)
	if err != nil {
		t.Fatalf("parseAttachmentURLs: %v", err)
	}
	if len(parts) != 1 || parts[0].ImageURL == nil {
		t.Fatalf("parts = %#v", parts)
	}
	if got := parts[0].ImageURL.URL; got != raw {
		t.Fatalf("URL changed across attachment parsing:\n got: %s\nwant: %s", got, raw)
	}
}

func TestParseAttachmentURLsRejectsEncodedTopLevelQuerySeparators(t *testing.T) {
	broken := "https://objects.example.test/frame.jpg?algorithm=v1%26credential=abc%26date=20260825T080000Z%26signature=123"
	if _, err := parseAttachmentURLs(broken); err == nil || !strings.Contains(err.Error(), "percent-encoded query separators") {
		t.Fatalf("error = %v, want encoded-query-separator rejection", err)
	}

	// A single escaped ampersand inside a value is not enough to classify the
	// whole query as a model-corrupted parameter list.
	legitimate := "https://example.test/frame.jpg?label=research%26development"
	if _, err := parseAttachmentURLs(legitimate); err != nil {
		t.Fatalf("legitimate escaped value rejected: %v", err)
	}
}

func TestSendRejectsCorruptedAttachmentAndSchedulesCorrection(t *testing.T) {
	thinker := retryTestThinker(nil)
	handler := mainToolHandler(thinker)
	_, _, results := handler(thinker, []toolCall{{
		Name:     "send",
		NativeID: "call-send",
		Args: map[string]string{
			"id":      "parent",
			"message": "Please inspect these frames.",
			"media":   "https://objects.example.test/frame.jpg?algorithm=v1%26credential=abc%26date=now%26signature=bad",
		},
	}}, nil)
	if len(results) != 1 || !results[0].IsError || !strings.Contains(results[0].Content, "copy the opaque URL exactly") {
		t.Fatalf("send results = %#v", results)
	}
	if !thinker.kickNextTurn {
		t.Fatal("rejected attachment did not schedule the sender's correction turn")
	}
}

func TestSessionPersistsAttachmentReceiptWithoutAccessURL(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "attachment-history")
	raw := "https://objects.example.test/frame.jpg?credential=secret&signature=also-secret"
	live := Message{
		Role:    "user",
		Content: "Inspect the attached frame.",
		Parts: []ContentPart{
			{Type: "text", Text: "Inspect the attached frame."},
			{Type: "image_url", ImageURL: &ImageURL{URL: raw}},
		},
	}
	if err := session.AppendMessage(live, 1, TokenUsage{}); err != nil {
		t.Fatalf("AppendMessage: %v", err)
	}
	if live.Parts[1].ImageURL.URL != raw {
		t.Fatal("durability projection mutated the live attachment")
	}

	durable, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatalf("read session: %v", err)
	}
	if strings.Contains(string(durable), "credential=") || strings.Contains(string(durable), "also-secret") {
		t.Fatalf("durable history retained an attachment access URL: %s", durable)
	}
	if !strings.Contains(string(durable), transientAttachmentHistoryNotice) {
		t.Fatalf("durable history omitted attachment receipt: %s", durable)
	}

	loaded, _ := session.LoadTail(10)
	if len(loaded) != 1 || len(loaded[0].Parts) != 0 {
		t.Fatalf("loaded messages = %#v, want one text-only receipt", loaded)
	}
	if !strings.Contains(loaded[0].Content, transientAttachmentHistoryNotice) {
		t.Fatalf("loaded content = %q", loaded[0].Content)
	}
}

func TestSessionLoadMigratesLegacyAttachmentWithoutReplayingIt(t *testing.T) {
	dir := t.TempDir()
	session := NewSession(dir, "legacy-attachment")
	raw := "https://objects.example.test/frame.jpg?algorithm=v1%26credential=abc%26date=now%26signature=secret"
	entry := SessionEntry{
		Role:    "user",
		Content: "Legacy attachment",
		Parts:   []ContentPart{{Type: "image_url", ImageURL: &ImageURL{URL: raw}}},
	}
	encoded, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	encoded = append(encoded, '\n')
	if err := os.WriteFile(session.path, encoded, 0600); err != nil {
		t.Fatal(err)
	}

	loaded, _ := session.LoadTail(10)
	if len(loaded) != 1 || len(loaded[0].Parts) != 0 {
		t.Fatalf("legacy attachment was replayed: %#v", loaded)
	}
	durable, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(durable), "objects.example.test") || strings.Contains(string(durable), "signature=secret") {
		t.Fatalf("legacy attachment URL survived migration: %s", durable)
	}
}

func TestConsumeTransientAttachmentsKeepsStableTextReceipt(t *testing.T) {
	messages := []Message{{
		Role:    "user",
		Content: "Analyze these inputs.",
		Parts: []ContentPart{
			{Type: "text", Text: "Analyze these inputs."},
			{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.test/a.jpg"}},
			{Type: "audio_url", AudioURL: &AudioURL{URL: "https://example.test/a.mp3"}},
		},
	}}
	if got := consumeTransientAttachments(messages); got != 2 {
		t.Fatalf("consumed = %d, want 2", got)
	}
	if len(messages[0].Parts) != 0 || !strings.Contains(messages[0].Content, transientAttachmentHistoryNotice) {
		t.Fatalf("projected message = %#v", messages[0])
	}
	if got := consumeTransientAttachments(messages); got != 0 {
		t.Fatalf("second consumption = %d, want idempotent zero", got)
	}
}
