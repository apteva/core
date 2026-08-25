package core

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestCodexAttachmentDownloadRecoverySmoke sends a deliberately unavailable
// generic image input to the real Responses provider. Core must quarantine the
// rejected attachment and complete the same prepared turn without retaining or
// resubmitting the URL.
func TestCodexAttachmentDownloadRecoverySmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_ATTACHMENT_RECOVERY_SMOKE") == "" {
		t.Skip("set RUN_CODEX_ATTACHMENT_RECOVERY_SMOKE=1 to run")
	}
	if testing.Short() {
		t.Skip("skipping Codex attachment recovery smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	thinker := retryTestThinker(NewOpenAICodexProvider(token))
	thinker.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	thinker.messages = []Message{
		{Role: "system", Content: "Follow the request. If an attachment is unavailable, continue from the text and reply exactly ATTACHMENT_RECOVERY_OK."},
		{
			Role:    "user",
			Content: "The attachment may be unavailable. Continue safely from this text.",
			Parts: []ContentPart{
				{Type: "text", Text: "The attachment may be unavailable. Continue safely from this text."},
				{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/definitely-not-an-image.png?algorithm=v1%26credential=bad%26date=now%26signature=bad"}},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
	defer cancel()
	resp, err := thinker.callLLMWithRetry(ctx)
	if err != nil {
		t.Fatalf("real Codex attachment recovery: %v", err)
	}
	if !strings.Contains(resp.Text, "ATTACHMENT_RECOVERY_OK") {
		t.Fatalf("Codex response = %q", resp.Text)
	}
	if transientAttachmentCount(thinker.messages) != 0 {
		t.Fatalf("attachment remained in live history: %#v", thinker.messages)
	}
	events, _ := thinker.telemetry.Events(0)
	quarantined := false
	for _, event := range events {
		if event.Type == "attachment.quarantined" {
			quarantined = true
		}
	}
	if !quarantined {
		t.Fatal("real provider did not exercise attachment quarantine")
	}
}
