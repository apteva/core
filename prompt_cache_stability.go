package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const promptCacheFingerprintBytes = 12

func promptCacheShortHash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:promptCacheFingerprintBytes])
}

func promptCacheIdentityFor(threadID string) string {
	agentID := strings.TrimSpace(os.Getenv("AGENT_ID"))
	if agentID == "" {
		agentID = strings.TrimSpace(os.Getenv("INSTANCE_ID"))
	}
	if agentID == "" {
		agentID = "local"
	}
	if strings.TrimSpace(threadID) == "" {
		threadID = "main"
	}
	return "agent:" + agentID + "/thread:" + threadID
}

func appendPromptCacheRecord(dst []byte, kind string, value any) []byte {
	b, err := json.Marshal(value)
	if err != nil {
		b = []byte(fmt.Sprintf("%q", fmt.Sprint(value)))
	}
	dst = append(dst, kind...)
	dst = append(dst, ':')
	dst = append(dst, b...)
	dst = append(dst, '\n')
	return dst
}

// promptCacheFingerprintInput deliberately uses newline-delimited records
// rather than one JSON array. Appending a message then appends bytes instead
// of changing a prior array-closing token, so commonPrefixBytes reflects the
// provider-facing append-only property we are trying to protect.
func promptCacheFingerprintInput(messages []Message, tools []NativeTool) (stable, request []byte) {
	for _, message := range messages {
		if message.Role == "system" {
			stable = appendPromptCacheRecord(stable, "system", message)
		}
	}
	stable = appendPromptCacheRecord(stable, "tools", tools)
	request = append(request, stable...)
	for _, message := range messages {
		if message.Role != "system" {
			request = appendPromptCacheRecord(request, "message", message)
		}
	}
	return stable, request
}

func commonPrefixBytes(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (t *Thinker) promptCacheIdentity() string {
	if t == nil {
		return promptCacheIdentityFor("main")
	}
	return promptCacheIdentityFor(t.threadID)
}

func (t *Thinker) advancePromptCacheEpoch(reason string, resetRequestContext bool, details map[string]any) {
	if t == nil {
		return
	}
	if strings.TrimSpace(reason) == "" {
		reason = "intentional_prefix_rewrite"
	}
	if resetRequestContext {
		t.requestContext.reset()
	}
	t.promptCacheEpoch++
	t.promptCacheResetReason = reason
	t.promptCachePreviousRequest = nil
	t.promptCacheStableHash = ""

	if t.telemetry != nil {
		data := map[string]any{
			"iteration":     t.iteration,
			"cache_epoch":   t.promptCacheEpoch,
			"reset_reason":  reason,
			"identity_hash": promptCacheShortHash([]byte(t.promptCacheIdentity())),
		}
		for key, value := range details {
			data[key] = value
		}
		t.telemetry.Emit("llm.prompt_cache_reset", t.threadID, data)
	}
}

func (t *Thinker) resetPromptCache(reason string) {
	t.advancePromptCacheEpoch(reason, true, nil)
}

func (t *Thinker) preparePromptCacheContext(ctx context.Context, messages []Message, tools []NativeTool) context.Context {
	stable, request := promptCacheFingerprintInput(messages, tools)
	stableHash := promptCacheShortHash(stable)
	if t.promptCacheStableHash != "" && t.promptCacheStableHash != stableHash {
		t.advancePromptCacheEpoch("stable_prefix_changed", false, map[string]any{
			"previous_stable_prefix_hash": t.promptCacheStableHash,
			"stable_prefix_hash":          stableHash,
		})
	}

	t.promptCacheStableHash = stableHash
	t.promptCacheRequestEpoch = t.promptCacheEpoch
	t.promptCacheRequestReason = t.promptCacheResetReason
	t.promptCacheRequestStableHash = stableHash
	t.promptCacheRequestHash = promptCacheShortHash(request)
	t.promptCacheCommonPrefixBytes = commonPrefixBytes(t.promptCachePreviousRequest, request)
	t.promptCachePreviousRequest = append(t.promptCachePreviousRequest[:0], request...)

	scope := openAIPromptCacheScope{
		Identity: t.promptCacheIdentity(),
		Epoch:    t.promptCacheEpoch,
	}
	return withOpenAIPromptCacheScope(ctx, scope)
}
