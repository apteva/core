package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
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
	toolResultContextPreviewChars      = 2000
	compactionInputMessagePreviewChars = 6000
	semanticCompactionTimeout          = 90 * time.Second
)

func contextChars(messages []Message) int {
	messages = fileRefMessagesForModel(messages)
	n := 0
	for _, msg := range messages {
		n += len(msg.Role) + len(msg.Content) + len(msg.Reasoning)
		if msg.ProviderState != nil {
			for _, raw := range msg.ProviderState.Items {
				n += len(raw)
			}
		}
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
			n += len(result.CallID) + len(result.Content) + toolImagesBytes(result.Images)
		}
	}
	return n
}

// estimatedContextTokens approximates provider-visible tokens without treating
// compressed media bytes as text. Image tokenization is based primarily on
// dimensions, not JPEG/PNG byte size; using len(image)/4 caused screenshots to
// look hundreds of thousands of tokens larger than the provider reported.
func estimatedContextTokens(messages []Message) int {
	messages = fileRefMessagesForModel(messages)
	textChars := 0
	imageTokens := 0
	for _, msg := range messages {
		// Opaque Responses items may contain much more than the visible text.
		// Count the larger representation without double-counting replayed calls.
		if msg.ProviderState != nil && len(msg.ProviderState.Items) > 0 {
			raw, _ := json.Marshal(msg.ProviderState.Items)
			copy := msg
			copy.ProviderState = nil
			visible := estimatedContextTokens([]Message{copy})
			rawTokens := (len(raw) + 3) / 4
			if rawTokens > visible {
				textChars += rawTokens * 4
			} else {
				textChars += visible * 4
			}
			continue
		}
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
			for _, img := range result.Images {
				imageTokens += estimateImageTokens(img.Data)
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

// toolResultName prefers the name recorded on the result, then its call.
func toolResultName(result ToolResult, names map[string]string) string {
	if name := names[result.CallID]; name != "" {
		return name
	}
	return result.ToolName
}

// latestToolImageCallIDs returns, for every group of "latest" images (one
// screen for computer_* tools, one per other tool), the call whose result
// holds the current image. That image is current evidence even after several
// model calls, so historical-result aging must not remove it.
func latestToolImageCallIDs(messages []Message) map[string]bool {
	names := toolNamesByCallID(messages)
	seenGroups := make(map[string]bool)
	current := make(map[string]bool)
	for i := len(messages) - 1; i >= 0; i-- {
		for j := len(messages[i].ToolResults) - 1; j >= 0; j-- {
			result := messages[i].ToolResults[j]
			name := toolResultName(result, names)
			if !resultHasLatestImages(result, name) {
				continue
			}
			group := latestToolImageGroup(name)
			if seenGroups[group] {
				continue
			}
			seenGroups[group] = true
			if result.CallID != "" {
				current[result.CallID] = true
			}
		}
	}
	return current
}

// evictReplacedToolImages keeps only the newest `keep` results of each group
// of "latest" images (screen captures and the like). Unlike kept images, they
// are observations of a state that has since changed: replaying older ones
// adds cost and can confuse the model about the current state.
func evictReplacedToolImages(messages []Message, keep int) int {
	if keep < 0 {
		keep = 0
	}
	names := toolNamesByCallID(messages)
	seen := make(map[string]int)
	removed := 0
	for i := len(messages) - 1; i >= 0; i-- {
		for j := len(messages[i].ToolResults) - 1; j >= 0; j-- {
			result := &messages[i].ToolResults[j]
			name := toolResultName(*result, names)
			if !resultHasLatestImages(*result, name) {
				continue
			}
			group := latestToolImageGroup(name)
			seen[group]++
			if seen[group] <= keep {
				continue
			}
			latest, kept := splitLatestToolImages(result.Images, name)
			result.Images = kept
			removed += len(latest)
			if isComputerToolName(name) {
				result.Content = evictedScreenshotPlaceholder
			} else {
				result.Content = appendToolResultNote(result.Content, fmt.Sprintf("[%d image(s) from %s replaced by a newer result]", len(latest), name))
			}
		}
	}
	return removed
}

// prepareLatestToolImageTail returns a provider-only projection. "Latest"
// images are removed from their historical function outputs without changing
// the durable messages, then the current one of each group is attached as a
// regular user image at the request tail. A changing frame can therefore
// invalidate only the volatile suffix instead of rewriting an earlier cached
// prefix. Kept images stay in their tool results.
func prepareLatestToolImageTail(messages []Message) []Message {
	current := latestToolImageCallIDs(messages)
	if len(current) == 0 {
		return messages
	}
	names := toolNamesByCallID(messages)
	type tailEntry struct {
		callID string
		name   string
		images []ToolImage
	}
	var tail []tailEntry
	projected := cloneMessages(messages)
	for i := range projected {
		for j := range projected[i].ToolResults {
			result := &projected[i].ToolResults[j]
			name := toolResultName(*result, names)
			if !resultHasLatestImages(*result, name) {
				continue
			}
			latest, kept := splitLatestToolImages(result.Images, name)
			result.Images = kept
			if current[result.CallID] {
				tail = append(tail, tailEntry{callID: result.CallID, name: name, images: latest})
			}
		}
	}
	for _, entry := range tail {
		var parts []ContentPart
		if isComputerToolName(entry.name) {
			parts = append(parts, ContentPart{Type: "text", Text: fmt.Sprintf("%s\nSource computer tool call: %s", computerScreenshotTailHeader, entry.callID)})
		} else {
			parts = append(parts, ContentPart{Type: "text", Text: fmt.Sprintf("[CURRENT IMAGE FROM %s — a newer result from this tool replaces it]\nSource tool call: %s", entry.name, entry.callID)})
		}
		for _, img := range entry.images {
			// A single screen capture keeps its long-standing two-part shape.
			if caption := toolImageCaption(img); caption != "" && (len(entry.images) > 1 || !isComputerToolName(entry.name)) {
				parts = append(parts, ContentPart{Type: "text", Text: caption})
			}
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: toolImageDataURL(img), Detail: "high"}})
		}
		projected = append(projected, Message{Role: "user", RequestContext: true, Parts: parts})
	}
	return projected
}

// limitKeptToolImages applies the shared image budget to a provider-only
// projection. Images from results the model has not seen yet are always
// sent and count first; older kept images go, oldest first, once the newest
// ones fill the budget. Each shortened result says exactly what was removed.
func limitKeptToolImages(messages []Message, unseen func(ToolResult) bool) ([]Message, int) {
	names := toolNamesByCallID(messages)
	count, tokens := 0, 0
	var projected []Message
	removedTotal := 0
	for i := len(messages) - 1; i >= 0; i-- {
		for j := len(messages[i].ToolResults) - 1; j >= 0; j-- {
			result := messages[i].ToolResults[j]
			if len(result.Images) == 0 {
				continue
			}
			name := toolResultName(result, names)
			fresh := unseen != nil && unseen(result)
			keep := make([]bool, len(result.Images))
			var removed []ToolImage
			for k := len(result.Images) - 1; k >= 0; k-- {
				img := result.Images[k]
				if toolImageIsLatest(img, name) {
					keep[k] = true // handled by the latest-image tail
					continue
				}
				cost := estimateImageTokens(img.Data)
				if fresh || (count < maxKeptToolImages && tokens+cost <= maxKeptToolImageTokens) {
					keep[k] = true
					count++
					tokens += cost
					continue
				}
				removed = append([]ToolImage{img}, removed...)
			}
			if len(removed) == 0 {
				continue
			}
			if projected == nil {
				projected = cloneMessages(messages)
			}
			target := &projected[i].ToolResults[j]
			var images []ToolImage
			for k, img := range target.Images {
				if keep[k] {
					images = append(images, img)
				}
			}
			target.Images = images
			target.Content = appendToolResultNote(target.Content, toolImagesRemovedNote(removed, name, "to save context"))
			removedTotal += len(removed)
		}
	}
	if projected == nil {
		return messages, 0
	}
	return projected, removedTotal
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

// messageForSession removes transient provider inputs from the durable copy.
// The live message retains them for the next model decision; after a restart,
// a stable receipt is safer than replaying old pixels or expired access URLs.
func messageForSession(history []Message, msg Message) Message {
	msg, _ = projectTransientAttachmentsFromMessage(msg)
	names := toolNamesByCallID(history)
	if len(msg.ToolResults) == 0 {
		return msg
	}
	msg.ToolResults = append([]ToolResult(nil), msg.ToolResults...)
	for i := range msg.ToolResults {
		result := &msg.ToolResults[i]
		name := toolResultName(*result, names)
		if resultHasLatestImages(*result, name) {
			_, kept := splitLatestToolImages(result.Images, name)
			result.Images = kept
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

// shouldRecoverFromEmptyResponse reports whether an empty answer came with a
// context near the model's limit, where compacting and asking again can help.
// On a smaller context compaction gains nothing, however many answers came
// back empty.
func shouldRecoverFromEmptyResponse(usage TokenUsage, modelID string, messages []Message) bool {
	maxTokens := ModelEffectiveContextWindow(modelID)
	if usage.PromptTokens > 0 && maxTokens > 0 && float64(usage.PromptTokens) >= float64(maxTokens)*contextPressureTokenRatio {
		return true
	}
	if maxTokens > 0 {
		return estimatedContextTokens(messages) >= int(float64(maxTokens)*contextPressureTokenRatio)
	}
	return contextChars(messages) >= contextPressureCharFallback
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
	ctx = t.providerSessionContext(ctx, "context-compaction")
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

	summaryMessage := summaryWithFileRefs("[COMPACTED CONTEXT]\n"+summary, old)
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
		text := appendFileRefText(msg.Content, msg.Parts)
		if strings.TrimSpace(text) != "" {
			b.WriteString(excerptForCompaction(text, compactionInputMessagePreviewChars))
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
