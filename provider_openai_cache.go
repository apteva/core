package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
)

const defaultOpenAIPromptCacheRetention = "24h"

type openAIPromptCacheHints struct {
	Key       string
	Retention string
}

func openAIPromptCacheHintsFor(providerName, model, stablePrefix string, tools any) openAIPromptCacheHints {
	if !openAIPromptCacheEnabled(providerName) {
		return openAIPromptCacheHints{}
	}
	retention := strings.TrimSpace(os.Getenv("APTEVA_OPENAI_PROMPT_CACHE_RETENTION"))
	if retention == "" {
		retention = defaultOpenAIPromptCacheRetention
	}

	h := sha256.New()
	h.Write([]byte("apteva-openai-cache-v1\n"))
	h.Write([]byte(providerName))
	h.Write([]byte{'\n'})
	h.Write([]byte(model))
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
		Key:       "apteva-v1-" + hex.EncodeToString(sum[:16]),
		Retention: retention,
	}
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
