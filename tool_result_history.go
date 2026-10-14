package core

import "fmt"

const (
	largeToolResultThresholdBytes = 16 << 10
	toolResultFullRetentionCalls  = 3

	// Mature results are projected as one block instead of turning a rolling
	// browser workflow into one prompt-cache reset per action.
	toolResultProjectionBatchSize  = 12
	toolResultProjectionBatchChars = 512 << 10
	toolResultProjectionMaxAge     = 16
	// Do not wait for twelve results when a few procedure snapshots already
	// occupy most of an ordinary tool workflow's input.
	toolResultProjectionPressureBytes = 256 << 10
)

func toolResultKey(result ToolResult) string {
	if result.CallID != "" {
		return "call:" + result.CallID
	}
	if result.SHA256 != "" {
		return "sha256:" + result.SHA256
	}
	return ""
}

func shouldArchiveToolResult(result ToolResult) bool {
	return isToolResultSHA256(result.SHA256) || len(result.Image) > 0 || len(result.Content)+len(result.Image) > largeToolResultThresholdBytes
}

// archiveToolResultMessage assigns tool names and internally archives only
// large payloads and images. The live conversation keeps the original result;
// request projection ages it out later without introducing any agent tool.
func (t *Thinker) archiveToolResultMessage(message Message) Message {
	if len(message.ToolResults) == 0 {
		return message
	}
	names := toolNamesByCallID(t.messages)
	message.ToolResults = append([]ToolResult(nil), message.ToolResults...)
	for i := range message.ToolResults {
		if message.ToolResults[i].ToolName == "" {
			message.ToolResults[i].ToolName = names[message.ToolResults[i].CallID]
		}
	}
	if t.session != nil {
		archived, err := t.session.ArchiveToolResults(message)
		if err != nil {
			logMsg("SESSION", fmt.Sprintf("[%s] archive tool result: %v", t.threadID, err))
		} else {
			message = archived
		}
	}
	t.toolResultMu.Lock()
	if t.toolResultAge == nil {
		t.toolResultAge = make(map[string]int)
	}
	if t.toolResultHistorical == nil {
		t.toolResultHistorical = make(map[string]bool)
	}
	for _, result := range message.ToolResults {
		if key := toolResultKey(result); key != "" {
			delete(t.toolResultAge, key)
			delete(t.toolResultHistorical, key)
			if result.ContentIsPreview {
				t.toolResultHistorical[key] = true
			}
		}
	}
	t.toolResultMu.Unlock()
	return message
}

// markToolResultsConsumed advances an internal retention clock once per
// successful provider request. Crossing the maturity boundary only makes a
// result eligible for the next block projection; it does not rewrite an old
// request or rotate the prompt-cache epoch by itself.
func (t *Thinker) markToolResultsConsumed(messages []Message) {
	seen := make(map[string]bool)
	t.toolResultMu.Lock()
	if t.toolResultAge == nil {
		t.toolResultAge = make(map[string]int)
	}
	if t.toolResultHistorical == nil {
		t.toolResultHistorical = make(map[string]bool)
	}
	for _, message := range messages {
		for _, result := range message.ToolResults {
			key := toolResultKey(result)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			if result.ContentIsPreview {
				t.toolResultHistorical[key] = true
				continue
			}
			if !t.toolResultHistorical[key] {
				t.toolResultAge[key]++
			}
		}
	}
	t.toolResultMu.Unlock()
}

func (t *Thinker) markLoadedToolResultsHistorical(messages []Message) {
	t.toolResultMu.Lock()
	defer t.toolResultMu.Unlock()
	if t.toolResultAge == nil {
		t.toolResultAge = make(map[string]int)
	}
	if t.toolResultHistorical == nil {
		t.toolResultHistorical = make(map[string]bool)
	}
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if key := toolResultKey(result); key != "" {
				t.toolResultAge[key] = toolResultFullRetentionCalls
				t.toolResultHistorical[key] = true
			}
		}
	}
}

func (t *Thinker) toolResultIsHistorical(result ToolResult) bool {
	if result.ContentIsPreview {
		return true
	}
	key := toolResultKey(result)
	if key == "" {
		return false
	}
	t.toolResultMu.Lock()
	defer t.toolResultMu.Unlock()
	return t.toolResultHistorical[key]
}

type toolResultProjectionStats struct {
	count  int
	chars  int
	maxAge int
}

func (t *Thinker) matureToolResultStats(messages []Message) toolResultProjectionStats {
	var stats toolResultProjectionStats
	if !t.toolResultContextUnderPressure(messages) {
		return stats
	}
	protected := t.protectedHistoricalToolResults(messages)
	seen := make(map[string]bool)
	currentScreen := latestComputerScreenshotCallID(messages)
	t.toolResultMu.Lock()
	defer t.toolResultMu.Unlock()
	for _, message := range messages {
		for _, result := range message.ToolResults {
			key := toolResultKey(result)
			if key == "" || seen[key] || protected[result.CallID] || (currentScreen != "" && result.CallID == currentScreen) || t.toolResultHistorical[key] || !shouldArchiveToolResult(result) {
				continue
			}
			seen[key] = true
			age := t.toolResultAge[key]
			if age < toolResultFullRetentionCalls {
				continue
			}
			stats.count++
			stats.chars += len(result.Content)
			if age > stats.maxAge {
				stats.maxAge = age
			}
		}
	}
	return stats
}

func toolResultProjectionBatchReady(stats toolResultProjectionStats) bool {
	return stats.count >= toolResultProjectionBatchSize ||
		stats.chars >= toolResultProjectionBatchChars ||
		stats.maxAge >= toolResultProjectionMaxAge
}

// commitMatureToolResults switches every eligible retained result to the
// provider-only bounded projection in one operation. The durable archive and
// in-memory full payload remain untouched.
func (t *Thinker) commitMatureToolResults(messages []Message) int {
	if !t.toolResultContextUnderPressure(messages) {
		return 0
	}
	protected := t.protectedHistoricalToolResults(messages)
	committed := 0
	seen := make(map[string]bool)
	t.toolResultMu.Lock()
	defer t.toolResultMu.Unlock()
	if t.toolResultHistorical == nil {
		t.toolResultHistorical = make(map[string]bool)
	}
	for _, message := range messages {
		for _, result := range message.ToolResults {
			key := toolResultKey(result)
			if key == "" || seen[key] || protected[result.CallID] || t.toolResultHistorical[key] || !shouldArchiveToolResult(result) {
				continue
			}
			seen[key] = true
			if t.toolResultAge[key] < toolResultFullRetentionCalls {
				continue
			}
			t.toolResultHistorical[key] = true
			committed++
		}
	}
	return committed
}

// prepareHistoricalToolResults leaves recent results byte-for-byte unchanged.
// Once old, large results become deterministic bounded previews; aged small
// results remain full until the aggregate historical budget would be exceeded.
func (t *Thinker) prepareHistoricalToolResults(messages []Message) []Message {
	protected := t.protectedHistoricalToolResults(messages)
	protectCurrentScreen := func(result ToolResult) bool { return protected[result.CallID] }
	perResult, totalBudget := t.historicalToolResultBudgets()
	historicalCount := 0
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if !protectCurrentScreen(result) && t.toolResultIsHistorical(result) {
				historicalCount++
			}
		}
	}
	remaining := totalBudget
	remainingResults := historicalCount
	var projected []Message
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		message := messages[messageIndex]
		for resultIndex := len(message.ToolResults) - 1; resultIndex >= 0; resultIndex-- {
			result := message.ToolResults[resultIndex]
			if protectCurrentScreen(result) || !t.toolResultIsHistorical(result) {
				continue
			}
			maxForResult := remaining
			const minimumResultBudget = 64
			reserveForOlder := (remainingResults - 1) * minimumResultBudget
			if maxForResult > reserveForOlder {
				maxForResult -= reserveForOlder
			} else {
				maxForResult = 0
			}
			remainingResults--
			mustProject := len(result.Image) > 0 || len(result.Content) > min(perResult, maxForResult)
			if !mustProject {
				remaining -= len(result.Content)
				continue
			}
			if projected == nil {
				projected = cloneMessages(messages)
			}
			limit := perResult
			if maxForResult < limit {
				limit = maxForResult
			}
			bounded := projectArchivedToolResult(result, limit)
			projected[messageIndex].ToolResults[resultIndex] = bounded
			remaining -= len(bounded.Content)
			if remaining < 0 {
				remaining = 0
			}
		}
	}
	if projected != nil {
		return projected
	}
	return messages
}

func (t *Thinker) prepareToolResultRequest(messages []Message) []Message {
	retainedBytes := 0
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if !t.toolResultIsHistorical(result) {
				// Compressed pixels are budgeted as images, not text tokens.
				// The current computer frame is already handled by its tail.
				retainedBytes += len(result.Content)
			}
		}
	}
	if t.toolResultContextUnderPressure(messages) {
		if projected := t.commitMatureToolResults(messages); projected > 0 {
			t.advancePromptCacheEpoch("tool_result_context_pressure", false, map[string]any{
				"results_projected": projected, "retained_bytes_before": retainedBytes,
			})
		}
	}
	projected := prepareComputerScreenshotTail(t.prepareHistoricalToolResults(messages))
	if t.telemetry != nil {
		original, sent, shortened := 0, 0, 0
		for i, message := range messages {
			for j, result := range message.ToolResults {
				original += len(result.Content) + len(result.Image)
				next := projected[i].ToolResults[j]
				sent += len(next.Content) + len(next.Image)
				if next.Content != result.Content || len(next.Image) != len(result.Image) {
					shortened++
				}
			}
		}
		t.telemetry.Emit("tool.result.context", t.threadID, map[string]any{
			"iteration": t.iteration, "original_bytes": original,
			"projected_bytes": sent, "results_shortened": shortened,
		})
	}
	return projected
}

// OpenClaw-style whole-result caps. Go's limits use UTF-8 bytes, rather than
// UTF-16 characters; these are approximate text/token budgets, not tokenizer counts.
func (t *Thinker) historicalToolResultBudgets() (perResult, total int) {
	perResult, total = historicalToolResultPerResultChars, historicalToolResultTotalChars
	window := ModelEffectiveContextWindow(t.modelID())
	if window >= 200_000 {
		perResult = 64_000
	} else if window >= 100_000 {
		perResult = 32_000
	}
	if window > 0 {
		budget := estimatePreparedRequest("", t.modelID(), nil, nil)
		if t.provider != nil {
			budget = estimatePreparedRequest(t.provider.Name(), t.modelID(), nil, nil)
		}
		share := max(0, budget.InputBudget) * 4 * 3 / 10
		perResult, total = min(perResult, share), min(total, share)
	}
	return
}

func (t *Thinker) protectedHistoricalToolResults(messages []Message) map[string]bool {
	protected := t.toolCallIDsProtectedFromSanitization(nil)
	if screen := latestComputerScreenshotCallID(messages); screen != "" {
		protected[screen] = true
	}
	boundary, turns := len(messages), 0
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "assistant" {
			turns++
			boundary = i
			if turns == 3 {
				break
			}
		}
	}
	// With fewer than three turns, protect the entire recent exchange.
	if turns < 3 {
		boundary = 0
	}
	firstUser := -1
	for i, msg := range messages {
		if msg.Role == "user" && len(msg.ToolResults) == 0 {
			firstUser = i
			break
		}
	}
	for i, msg := range messages {
		if i < boundary && (firstUser < 0 || i >= firstUser) {
			continue
		}
		for _, call := range msg.ToolCalls {
			if call.ID != "" {
				protected[call.ID] = true
			}
		}
		for _, result := range msg.ToolResults {
			if result.CallID != "" {
				protected[result.CallID] = true
			}
		}
	}
	return protected
}

func (t *Thinker) toolResultContextUnderPressure(messages []Message) bool {
	messages = t.prepareHistoricalToolResults(messages)
	textBytes := 0
	for _, message := range messages {
		for _, result := range message.ToolResults {
			textBytes += len(result.Content)
		}
	}
	if textBytes >= toolResultProjectionPressureBytes {
		return true
	}
	window := ModelEffectiveContextWindow(t.modelID())
	if window <= 0 {
		return false
	}
	provider := ""
	if t.provider != nil {
		provider = t.provider.Name()
	}
	budget := estimatePreparedRequest(provider, t.modelID(), nil, nil)
	return estimatedContextTokens(messages) >= budget.InputBudget*3/10
}
