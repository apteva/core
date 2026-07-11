package core

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const (
	contextPressureKeepRecent          = 20
	contextPressureTokenRatio          = 0.80
	semanticCompactionTokenRatio       = 0.65
	contextPressureCharFallback        = 512 * 1024
	semanticCompactionCharFallback     = 384 * 1024
	contextPressureEmptyStreak         = 2
	toolResultContextPreviewChars      = 2000
	compactionInputMessagePreviewChars = 6000
	semanticCompactionTimeout          = 90 * time.Second
)

func contextChars(messages []Message) int {
	n := 0
	for _, msg := range messages {
		n += len(msg.Role) + len(msg.Content) + len(msg.Reasoning)
		for _, part := range msg.Parts {
			n += len(part.Type) + len(part.Text)
			if part.ImageURL != nil {
				n += len(part.ImageURL.URL) + len(part.ImageURL.Detail)
			}
			if part.InputAudio != nil {
				n += len(part.InputAudio.Data) + len(part.InputAudio.Format)
			}
			if part.AudioURL != nil {
				n += len(part.AudioURL.URL) + len(part.AudioURL.MimeType)
			}
		}
		for _, call := range msg.ToolCalls {
			n += len(call.ID) + len(call.Name) + len(call.ThoughtSignature)
			for key, value := range call.Args {
				n += len(key) + len(value)
			}
		}
		for _, result := range msg.ToolResults {
			n += len(result.CallID) + len(result.Content) + len(result.Image)
		}
	}
	return n
}

func shouldCompactBeforeLLM(modelID string, messages []Message) bool {
	maxTokens := ModelEffectiveContextWindow(modelID)
	chars := contextChars(messages)
	underPressure := false
	if maxTokens > 0 && chars/4 >= int(float64(maxTokens)*semanticCompactionTokenRatio) {
		underPressure = true
	}
	if chars >= semanticCompactionCharFallback {
		underPressure = true
	}
	return underPressure && len(trimMessagesForContextPressure(messages, contextPressureKeepRecent)) < len(messages)
}

func shouldRecoverFromEmptyResponse(usage TokenUsage, modelID string, messages []Message, emptyStreak int) bool {
	maxTokens := ModelEffectiveContextWindow(modelID)
	if usage.PromptTokens > 0 && maxTokens > 0 && float64(usage.PromptTokens) >= float64(maxTokens)*contextPressureTokenRatio {
		return true
	}
	chars := contextChars(messages)
	if maxTokens > 0 && chars/4 >= int(float64(maxTokens)*contextPressureTokenRatio) {
		return true
	}
	if chars >= contextPressureCharFallback {
		return true
	}
	return emptyStreak >= contextPressureEmptyStreak
}

func trimMessagesForContextPressure(messages []Message, keepRecent int) []Message {
	if len(messages) <= 1 {
		return messages
	}
	if keepRecent < 1 {
		keepRecent = 1
	}
	if len(messages) <= keepRecent+1 {
		return sanitizeToolPairs(messages)
	}

	start := retainedStartForContextPressure(messages, keepRecent)

	next := make([]Message, 0, len(messages)-start+1)
	next = append(next, messages[0])
	next = append(next, messages[start:]...)
	return sanitizeToolPairs(next)
}

func retainedStartForContextPressure(messages []Message, keepRecent int) int {
	if keepRecent < 1 {
		keepRecent = 1
	}
	start := len(messages) - keepRecent
	if start < 1 {
		start = 1
	}

	neededCallIDs := make(map[string]bool)
	for _, msg := range messages[start:] {
		for _, result := range msg.ToolResults {
			if result.CallID != "" {
				neededCallIDs[result.CallID] = true
			}
		}
	}
	if len(neededCallIDs) > 0 {
		for i := start - 1; i >= 1; i-- {
			found := false
			for _, call := range messages[i].ToolCalls {
				if neededCallIDs[call.ID] {
					delete(neededCallIDs, call.ID)
					found = true
				}
			}
			if found {
				start = i
			}
			if len(neededCallIDs) == 0 {
				break
			}
		}
	}
	return start
}

type semanticCompactionResult struct {
	summary         string
	messages        []Message
	model           string
	usage           TokenUsage
	duration        time.Duration
	summarizedCount int
	retainedCount   int
}

func (t *Thinker) semanticCompactContext(reason string) (semanticCompactionResult, error) {
	var result semanticCompactionResult
	if t == nil || t.provider == nil {
		return result, fmt.Errorf("no provider available")
	}
	if len(t.messages) <= contextPressureKeepRecent+1 {
		return result, fmt.Errorf("not enough old context to compact")
	}

	start := retainedStartForContextPressure(t.messages, contextPressureKeepRecent)
	if start <= 1 {
		return result, fmt.Errorf("no old context to summarize")
	}
	old := append([]Message(nil), t.messages[1:start]...)
	retained := append([]Message(nil), t.messages[start:]...)
	if len(old) == 0 {
		return result, fmt.Errorf("no old context to summarize")
	}

	model := t.provider.Models()[ModelSmall]
	if model == "" {
		model = t.modelID()
	}
	if model == "" {
		return result, fmt.Errorf("no compaction model available")
	}

	prompt := buildSemanticCompactionPrompt(reason, old)
	ctx, cancel := context.WithTimeout(context.Background(), semanticCompactionTimeout)
	done := make(chan struct{})
	go func() {
		select {
		case <-t.quit:
			cancel()
		case <-done:
		}
	}()
	defer func() {
		close(done)
		cancel()
	}()

	started := time.Now()
	resp, err := t.provider.Chat(ctx, prompt, model, nil, nil, nil, nil)
	if err != nil {
		return result, err
	}
	summary := strings.TrimSpace(resp.Text)
	if summary == "" {
		return result, fmt.Errorf("compaction summary was empty")
	}

	summaryMessage := Message{
		Role:    "user",
		Content: "[COMPACTED CONTEXT]\n" + summary,
	}
	next := make([]Message, 0, len(retained)+2)
	next = append(next, t.messages[0], summaryMessage)
	next = append(next, retained...)
	next = sanitizeToolPairs(next)

	result.summary = summary
	result.messages = next
	result.model = model
	result.usage = resp.Usage
	result.duration = time.Since(started)
	result.summarizedCount = len(old)
	result.retainedCount = len(retained)
	return result, nil
}

func buildSemanticCompactionPrompt(reason string, old []Message) []Message {
	return []Message{
		{
			Role: "system",
			Content: strings.Join([]string{
				"You compact an autonomous agent conversation so the agent can continue without losing operational state.",
				"Summarize only the provided older context. Do not invent facts.",
				"Preserve exact identifiers, names, dates, tool names, call ids, decisions, constraints, user preferences, failures, and open tasks.",
				"Write concise markdown with these headings: Objective, Completed Work, Current State, Important Tool Results, Decisions And Constraints, Open Tasks, Risks And Failed Attempts.",
				"If a heading has nothing useful, write '- None known'.",
			}, "\n"),
		},
		{
			Role: "user",
			Content: fmt.Sprintf("Compaction reason: %s\n\nOlder context to summarize:\n\n%s",
				reason, renderMessagesForSemanticCompaction(old)),
		},
	}
}

func renderMessagesForSemanticCompaction(messages []Message) string {
	var b strings.Builder
	for i, msg := range messages {
		fmt.Fprintf(&b, "## Message %d (%s)\n", i+1, msg.Role)
		if strings.TrimSpace(msg.Content) != "" {
			b.WriteString(excerptForCompaction(msg.Content, compactionInputMessagePreviewChars))
			b.WriteString("\n")
		}
		if len(msg.ToolCalls) > 0 {
			b.WriteString("Tool calls:\n")
			for _, call := range msg.ToolCalls {
				fmt.Fprintf(&b, "- id=%s name=%s args=%v\n", call.ID, call.Name, call.Args)
			}
		}
		if len(msg.ToolResults) > 0 {
			b.WriteString("Tool results:\n")
			for _, result := range msg.ToolResults {
				fmt.Fprintf(&b, "- call_id=%s is_error=%v bytes=%d content:\n%s\n",
					result.CallID, result.IsError, len(result.Content), excerptForCompaction(result.Content, toolResultContextPreviewChars))
			}
		}
		b.WriteString("\n")
	}
	return b.String()
}

func excerptForCompaction(text string, maxChars int) string {
	text = strings.TrimSpace(text)
	if maxChars <= 0 || len(text) <= maxChars {
		return text
	}
	head := maxChars * 2 / 3
	tail := maxChars - head
	return strings.TrimSpace(text[:head]) +
		fmt.Sprintf("\n\n[... omitted %d bytes from compaction input ...]\n\n", len(text)-maxChars) +
		strings.TrimSpace(text[len(text)-tail:])
}
