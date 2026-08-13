package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
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

// estimatedContextTokens approximates provider-visible tokens without treating
// compressed media bytes as text. Image tokenization is based primarily on
// dimensions, not JPEG/PNG byte size; using len(image)/4 caused screenshots to
// look hundreds of thousands of tokens larger than the provider reported.
func estimatedContextTokens(messages []Message) int {
	textChars := 0
	imageTokens := 0
	for _, msg := range messages {
		textChars += len(msg.Role) + len(msg.Content) + len(msg.Reasoning)
		for _, part := range msg.Parts {
			textChars += len(part.Type) + len(part.Text)
			if part.ImageURL != nil {
				textChars += len(part.ImageURL.Detail)
				if data, ok := decodeImageDataURL(part.ImageURL.URL); ok {
					imageTokens += estimateImageTokens(data)
				} else {
					textChars += len(part.ImageURL.URL)
					imageTokens += defaultImageTokenEstimate
				}
			}
			// Audio estimation remains byte-based until providers expose a
			// duration-aware estimator. It is intentionally separate from images.
			if part.InputAudio != nil {
				textChars += len(part.InputAudio.Data) + len(part.InputAudio.Format)
			}
			if part.AudioURL != nil {
				textChars += len(part.AudioURL.URL) + len(part.AudioURL.MimeType)
			}
		}
		for _, call := range msg.ToolCalls {
			textChars += len(call.ID) + len(call.Name) + len(call.ThoughtSignature)
			for key, value := range call.Args {
				textChars += len(key) + len(value)
			}
		}
		for _, result := range msg.ToolResults {
			textChars += len(result.CallID) + len(result.Content)
			if len(result.Image) > 0 {
				imageTokens += estimateImageTokens(result.Image)
			}
		}
	}
	return (textChars+3)/4 + imageTokens
}

const defaultImageTokenEstimate = 2048

func estimateImageTokens(data []byte) int {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return defaultImageTokenEstimate
	}
	tilesWide := (cfg.Width + 511) / 512
	tilesHigh := (cfg.Height + 511) / 512
	// Twice the common 85 + 170-per-512px-tile formula is deliberately
	// conservative across providers while remaining independent of compression.
	estimate := 2 * (85 + 170*tilesWide*tilesHigh)
	if estimate < 512 {
		return 512
	}
	return estimate
}

func decodeImageDataURL(url string) ([]byte, bool) {
	if !strings.HasPrefix(url, "data:image/") {
		return nil, false
	}
	comma := strings.IndexByte(url, ',')
	if comma < 0 || !strings.Contains(url[:comma], ";base64") {
		return nil, false
	}
	data, err := base64.StdEncoding.DecodeString(url[comma+1:])
	return data, err == nil
}

const evictedScreenshotPlaceholder = "[previous screenshot replaced — see latest for current screen state]"
const computerScreenshotTailHeader = "[CURRENT COMPUTER SCREENSHOT — ephemeral uncached request tail]"

func toolNamesByCallID(messages []Message) map[string]string {
	names := make(map[string]string)
	for _, msg := range messages {
		for _, call := range msg.ToolCalls {
			if call.ID != "" {
				names[call.ID] = call.Name
			}
		}
	}
	return names
}

func isComputerToolName(name string) bool {
	return strings.HasPrefix(name, "computer_")
}

// evictStaleComputerScreenshots keeps only the newest computer frame. Unlike
// generated/user images, browser screenshots are transient observations: once
// a subsequent action returns a newer frame, replaying older frames adds cost
// and can confuse the model about the current UI state.
func evictStaleComputerScreenshots(messages []Message, keep int) int {
	if keep < 0 {
		keep = 0
	}
	names := toolNamesByCallID(messages)
	seen := 0
	removed := 0
	for i := len(messages) - 1; i >= 0; i-- {
		for j := len(messages[i].ToolResults) - 1; j >= 0; j-- {
			result := &messages[i].ToolResults[j]
			if len(result.Image) == 0 || !isComputerToolName(names[result.CallID]) {
				continue
			}
			seen++
			if seen > keep {
				result.Image = nil
				result.Content = evictedScreenshotPlaceholder
				removed++
			}
		}
	}
	return removed
}

// prepareComputerScreenshotTail returns a provider-only projection. Computer
// pixels are removed from their historical function outputs without changing
// the durable messages, then the newest frame is attached as a regular user
// image at the request tail. A changing frame can therefore invalidate only
// the volatile suffix instead of rewriting an earlier cached prefix.
func prepareComputerScreenshotTail(messages []Message) []Message {
	names := toolNamesByCallID(messages)
	latestCallID := ""
	var latestImage []byte
	for i := len(messages) - 1; i >= 0 && latestImage == nil; i-- {
		for j := len(messages[i].ToolResults) - 1; j >= 0; j-- {
			result := messages[i].ToolResults[j]
			if len(result.Image) == 0 || !isComputerToolName(names[result.CallID]) {
				continue
			}
			latestCallID = result.CallID
			latestImage = append([]byte(nil), result.Image...)
			break
		}
	}
	if len(latestImage) == 0 {
		return messages
	}

	projected := cloneMessages(messages)
	for i := range projected {
		for j := range projected[i].ToolResults {
			result := &projected[i].ToolResults[j]
			if len(result.Image) > 0 && isComputerToolName(names[result.CallID]) {
				result.Image = nil
			}
		}
	}

	mimeType := "image/png"
	if _, format, err := image.DecodeConfig(bytes.NewReader(latestImage)); err == nil {
		switch strings.ToLower(format) {
		case "jpeg", "jpg":
			mimeType = "image/jpeg"
		case "gif":
			mimeType = "image/gif"
		case "png":
			mimeType = "image/png"
		}
	}
	imageURL := "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(latestImage)
	projected = append(projected, Message{
		Role:           "user",
		RequestContext: true,
		Parts: []ContentPart{
			{Type: "text", Text: fmt.Sprintf("%s\nSource computer tool call: %s", computerScreenshotTailHeader, latestCallID)},
			{Type: "image_url", ImageURL: &ImageURL{URL: imageURL, Detail: "original"}},
		},
	})
	return projected
}

// checkpointHistoryWindow trims history in occasional blocks instead of
// deleting one oldest message on nearly every turn. maxHistory is the retained
// target: main/lead threads trigger above 200 and return to 100; workers trigger
// above 40 and return to 20. Large historical tool payloads have an independent
// aggregate bound, so allowing this wider message window does not restore
// unbounded results. The caller owns the one cache reset per checkpoint.
func checkpointHistoryWindow(messages []Message, maxHistory int, protectedToolCallIDs map[string]bool) ([]Message, int) {
	if len(messages) <= 1 || maxHistory <= 0 {
		return messages, 0
	}
	upper := maxHistory * 2
	target := maxHistory
	if len(messages)-1 <= upper {
		return messages, 0
	}

	start := len(messages) - target
	for start > 1 && len(messages[start].ToolResults) > 0 {
		start--
	}
	next := make([]Message, 0, 1+len(messages)-start)
	next = append(next, messages[0])
	next = append(next, messages[start:]...)
	// Checkpointing runs immediately after tool dispatch. Preserve native calls
	// whose asynchronous results are still in flight (or were dispatched this
	// turn and have a result waiting on the event bus); the normal pre-model
	// sanitizer will remove them later if they genuinely remain orphaned.
	next = append(next[:1], sanitizeToolPairs(next[1:], protectedToolCallIDs)...)
	dropped := len(messages) - len(next)
	if dropped <= 0 {
		return messages, 0
	}
	return next, dropped
}

// messageForSession removes transient computer pixels from the durable copy.
// The live message retains its image for the next model decision; after a
// restart, a fresh screenshot is safer than replaying an old browser frame.
func messageForSession(history []Message, msg Message) Message {
	names := toolNamesByCallID(history)
	if len(msg.ToolResults) == 0 {
		return msg
	}
	msg.ToolResults = append([]ToolResult(nil), msg.ToolResults...)
	for i := range msg.ToolResults {
		result := &msg.ToolResults[i]
		if len(result.Image) > 0 && isComputerToolName(names[result.CallID]) {
			result.Image = nil
		}
	}
	return msg
}

func shouldCompactBeforeLLM(modelID string, messages []Message) bool {
	maxTokens := ModelEffectiveContextWindow(modelID)
	trimmed := trimMessagesForContextPressure(messages, contextPressureKeepRecent)
	if len(trimmed) >= len(messages) {
		return false
	}

	if maxTokens > 0 {
		threshold := int(float64(maxTokens) * semanticCompactionTokenRatio)
		before := estimatedContextTokens(messages)
		if before < threshold {
			return false
		}
		after := estimatedContextTokens(trimmed)
		return after < threshold || before-after >= before/10
	}

	// The byte fallback exists only for models whose context window is unknown.
	// Applying it to a known 372K-token model caused 100KB JPEG screenshots to
	// trigger compaction while actual provider usage was only ~25K tokens.
	before := contextChars(messages)
	if before < semanticCompactionCharFallback {
		return false
	}
	after := contextChars(trimmed)
	return after < semanticCompactionCharFallback || before-after >= before/10
}

func shouldRecoverFromEmptyResponse(usage TokenUsage, modelID string, messages []Message, emptyStreak int) bool {
	maxTokens := ModelEffectiveContextWindow(modelID)
	if usage.PromptTokens > 0 && maxTokens > 0 && float64(usage.PromptTokens) >= float64(maxTokens)*contextPressureTokenRatio {
		return true
	}
	if maxTokens > 0 {
		if estimatedContextTokens(messages) >= int(float64(maxTokens)*contextPressureTokenRatio) {
			return true
		}
	} else if contextChars(messages) >= contextPressureCharFallback {
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
	ctx = withOpenAIPromptCacheScope(ctx, openAIPromptCacheScope{
		Identity: t.promptCacheIdentity() + "/context-compaction",
		Epoch:    t.promptCacheEpoch,
	})
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
