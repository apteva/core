package core

import (
	"strings"
	"testing"
)

func TestConversationThreadPromptDoesNotRequireCompletionReportToParent(t *testing.T) {
	prompt := formatThreadBasePrompt(false, false, true, "chat-conv-1", "main coordinator")
	for _, forbidden := range []string{
		"You MUST report results to your parent",
		"A final send is mandatory",
		"process results, send report",
	} {
		if strings.Contains(prompt, forbidden) {
			t.Fatalf("conversation prompt contains generic worker rule %q:\n%s", forbidden, prompt)
		}
	}
	for _, required := range []string{
		"user-facing conversation, not a one-shot worker",
		"visible acknowledgement before a durable parent handoff",
		"deliver that acknowledgement first and wait for its receipt",
		"That send wakes the target immediately",
		"Never send an acknowledgement, confirmation, or completion report back to parent",
		"No parent completion report is required",
	} {
		if !strings.Contains(prompt, required) {
			t.Fatalf("conversation prompt missing %q:\n%s", required, prompt)
		}
	}
}
