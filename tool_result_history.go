package core

import "fmt"

const (
	largeToolResultThresholdBytes = 16 << 10
	toolResultFullRetentionCalls  = 3
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
	for _, result := range message.ToolResults {
		if key := toolResultKey(result); key != "" {
			delete(t.toolResultAge, key)
		}
	}
	t.toolResultMu.Unlock()
	return message
}

// markToolResultsConsumed advances an internal retention clock once per
// successful provider request. Crossing the boundary is an intentional,
// infrequent prefix rewrite, so it also advances the cache epoch.
func (t *Thinker) markToolResultsConsumed(messages []Message) {
	seen := make(map[string]bool)
	expired := 0
	t.toolResultMu.Lock()
	if t.toolResultAge == nil {
		t.toolResultAge = make(map[string]int)
	}
	for _, message := range messages {
		for _, result := range message.ToolResults {
			key := toolResultKey(result)
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			if result.ContentIsPreview {
				t.toolResultAge[key] = toolResultFullRetentionCalls
				continue
			}
			age := t.toolResultAge[key]
			if age < toolResultFullRetentionCalls {
				age++
				t.toolResultAge[key] = age
				if age == toolResultFullRetentionCalls && shouldArchiveToolResult(result) {
					expired++
				}
			}
		}
	}
	t.toolResultMu.Unlock()
	if expired > 0 {
		t.advancePromptCacheEpoch("tool_result_retention_expired", false, map[string]any{
			"results_expired": expired,
			"retention_calls": toolResultFullRetentionCalls,
		})
	}
}

func (t *Thinker) markLoadedToolResultsHistorical(messages []Message) {
	t.toolResultMu.Lock()
	defer t.toolResultMu.Unlock()
	if t.toolResultAge == nil {
		t.toolResultAge = make(map[string]int)
	}
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if key := toolResultKey(result); key != "" {
				t.toolResultAge[key] = toolResultFullRetentionCalls
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
	return t.toolResultAge[key] >= toolResultFullRetentionCalls
}

// prepareHistoricalToolResults leaves recent results byte-for-byte unchanged.
// Once old, large results become deterministic bounded previews; aged small
// results remain full until the aggregate historical budget would be exceeded.
func (t *Thinker) prepareHistoricalToolResults(messages []Message) []Message {
	historicalCount := 0
	for _, message := range messages {
		for _, result := range message.ToolResults {
			if t.toolResultIsHistorical(result) {
				historicalCount++
			}
		}
	}
	remaining := historicalToolResultTotalChars
	remainingResults := historicalCount
	var projected []Message
	for messageIndex := len(messages) - 1; messageIndex >= 0; messageIndex-- {
		message := messages[messageIndex]
		for resultIndex := len(message.ToolResults) - 1; resultIndex >= 0; resultIndex-- {
			result := message.ToolResults[resultIndex]
			if !t.toolResultIsHistorical(result) {
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
			mustProject := shouldArchiveToolResult(result) || len(result.Content) > maxForResult
			if !mustProject {
				remaining -= len(result.Content)
				continue
			}
			if projected == nil {
				projected = cloneMessages(messages)
			}
			limit := historicalToolResultPerResultChars
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
	return prepareComputerScreenshotTail(t.prepareHistoricalToolResults(messages))
}
