package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync"
)

const defaultOpenAIPromptCacheRetention = "24h"

type openAIPromptCacheHints struct {
	Key       string
	Retention string
}

type openAIPromptCacheScope struct {
	Identity string
	Epoch    uint64
}

type openAIPromptCacheScopeContextKey struct{}

func withOpenAIPromptCacheScope(ctx context.Context, scope openAIPromptCacheScope) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, openAIPromptCacheScopeContextKey{}, scope)
}

func openAIPromptCacheScopeFromContext(ctx context.Context) openAIPromptCacheScope {
	if ctx == nil {
		return openAIPromptCacheScope{}
	}
	scope, _ := ctx.Value(openAIPromptCacheScopeContextKey{}).(openAIPromptCacheScope)
	return scope
}

type openAIPromptCacheState struct {
	mu       sync.RWMutex
	disabled bool
}

func (s *openAIPromptCacheState) enabled() bool {
	if s == nil {
		return true
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return !s.disabled
}

func (s *openAIPromptCacheState) disable() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.disabled = true
	s.mu.Unlock()
}

func openAIPromptCacheHintsFor(providerName, model, stablePrefix string, tools any) openAIPromptCacheHints {
	return openAIPromptCacheHintsForScope(providerName, model, stablePrefix, tools, openAIPromptCacheScope{})
}

func openAIPromptCacheHintsForScope(providerName, model, stablePrefix string, tools any, scope openAIPromptCacheScope) openAIPromptCacheHints {
	if !openAIPromptCacheEnabled(providerName) {
		return openAIPromptCacheHints{}
	}
	// The ChatGPT Codex endpoint does not implement the public API's legacy
	// retention field. GPT-5.6+ deprecates it on the public API as well; both
	// paths still benefit from a stable prompt_cache_key and automatic caching.
	retention := ""
	if providerName != "openai-codex" && !openAIModelUsesModernPromptCache(model) {
		retention = strings.TrimSpace(os.Getenv("APTEVA_OPENAI_PROMPT_CACHE_RETENTION"))
		if retention == "" {
			retention = defaultOpenAIPromptCacheRetention
		}
	}

	h := sha256.New()
	h.Write([]byte("apteva-openai-cache-v2\n"))
	h.Write([]byte(providerName))
	h.Write([]byte{'\n'})
	h.Write([]byte(model))
	h.Write([]byte{'\n'})
	h.Write([]byte(scope.Identity))
	h.Write([]byte{'\n'})
	h.Write([]byte(strconv.FormatUint(scope.Epoch, 10)))
	h.Write([]byte{'\n'})
	h.Write([]byte(stablePrefix))
	h.Write([]byte{'\n'})
	if tools != nil {
		if b, err := json.Marshal(tools); err == nil {
			h.Write(b)
		}
	}
	sum := h.Sum(nil)
	return openAIPromptCacheHints{
		Key:       "apteva-v2-" + hex.EncodeToString(sum[:16]),
		Retention: retention,
	}
}

func openAIModelUsesModernPromptCache(model string) bool {
	value := strings.ToLower(strings.TrimSpace(model))
	idx := strings.Index(value, "gpt-")
	if idx < 0 {
		return false
	}
	version := value[idx+len("gpt-"):]
	majorText, rest, found := strings.Cut(version, ".")
	major, err := strconv.Atoi(leadingDigits(majorText))
	if err != nil {
		return false
	}
	if major > 5 {
		return true
	}
	if major < 5 || !found {
		return false
	}
	minor, err := strconv.Atoi(leadingDigits(rest))
	return err == nil && minor >= 6
}

func leadingDigits(value string) string {
	end := 0
	for end < len(value) && value[end] >= '0' && value[end] <= '9' {
		end++
	}
	return value[:end]
}

func openAIPromptCacheEnabled(providerName string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("APTEVA_OPENAI_PROMPT_CACHE"))) {
	case "0", "false", "off", "no", "disabled":
		return false
	}
	switch providerName {
	case "openai", "openai-codex":
		return true
	default:
		return false
	}
}

func openAICacheHintsUnsupported(status int, body string) bool {
	if status < 400 || status >= 500 {
		return false
	}
	lower := strings.ToLower(body)
	if strings.Contains(lower, "prompt_cache_key") || strings.Contains(lower, "prompt_cache_retention") {
		return true
	}
	if strings.Contains(lower, "unknown parameter") && strings.Contains(lower, "prompt_cache") {
		return true
	}
	if strings.Contains(lower, "unsupported parameter") && strings.Contains(lower, "prompt_cache") {
		return true
	}
	return false
}

func openAIPromptCacheStablePrefix(messages []any) string {
	if len(messages) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, raw := range messages {
		msg, ok := raw.(openaiMessage)
		if !ok {
			return sb.String()
		}
		switch msg.Role {
		case "system", "developer":
			sb.WriteString(msg.Role)
			sb.WriteByte(':')
			if s, ok := msg.Content.(string); ok {
				sb.WriteString(s)
			} else if b, err := json.Marshal(msg.Content); err == nil {
				sb.Write(b)
			}
			sb.WriteByte('\n')
		default:
			return sb.String()
		}
	}
	return sb.String()
}
