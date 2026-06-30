package core

import "testing"

func TestOpenAIPromptCacheHintsFor(t *testing.T) {
	t.Setenv("APTEVA_OPENAI_PROMPT_CACHE", "")
	first := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-a"})
	second := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-a"})
	if first.Key == "" || first.Retention != "24h" {
		t.Fatalf("hints = %+v, want key + 24h retention", first)
	}
	if first != second {
		t.Fatalf("hints not stable: %+v vs %+v", first, second)
	}
	changed := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-b"})
	if changed.Key == first.Key {
		t.Fatal("tool surface change should produce a different cache key")
	}
	if got := openAIPromptCacheHintsFor("fireworks", "kimi", "system", nil); got.Key != "" || got.Retention != "" {
		t.Fatalf("non-OpenAI provider got hints: %+v", got)
	}

	t.Setenv("APTEVA_OPENAI_PROMPT_CACHE", "off")
	if got := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", nil); got.Key != "" || got.Retention != "" {
		t.Fatalf("disabled cache got hints: %+v", got)
	}
}

func TestOpenAICacheHintsUnsupported(t *testing.T) {
	if !openAICacheHintsUnsupported(400, `{"error":{"message":"Unknown parameter: prompt_cache_retention"}}`) {
		t.Fatal("expected prompt cache unsupported detection")
	}
	if openAICacheHintsUnsupported(401, `token expired`) {
		t.Fatal("auth errors should not be treated as prompt cache unsupported")
	}
	if openAICacheHintsUnsupported(500, `prompt_cache_retention`) {
		t.Fatal("server errors should not be retried as unsupported parameters")
	}
}
