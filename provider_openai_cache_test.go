package core

import "testing"

func TestOpenAIPromptCacheHintsFor(t *testing.T) {
	t.Setenv("APTEVA_OPENAI_PROMPT_CACHE", "")
	first := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-a"})
	second := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-a"})
	if first.Key == "" || first.Retention != "" {
		t.Fatalf("Codex hints = %+v, want key without public-API retention", first)
	}
	if first != second {
		t.Fatalf("hints not stable: %+v vs %+v", first, second)
	}
	changed := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", []any{"tool-b"})
	if changed.Key == first.Key {
		t.Fatal("tool surface change should produce a different cache key")
	}
	legacy := openAIPromptCacheHintsFor("openai", "gpt-5.5", "system", nil)
	if legacy.Key == "" || legacy.Retention != "24h" {
		t.Fatalf("legacy public hints = %+v, want key + 24h retention", legacy)
	}
	modern := openAIPromptCacheHintsFor("openai", "gpt-5.6", "system", nil)
	if modern.Key == "" || modern.Retention != "" {
		t.Fatalf("modern public hints = %+v, want key without deprecated retention", modern)
	}
	if got := openAIPromptCacheHintsFor("fireworks", "kimi", "system", nil); got.Key != "" || got.Retention != "" {
		t.Fatalf("non-OpenAI provider got hints: %+v", got)
	}

	t.Setenv("APTEVA_OPENAI_PROMPT_CACHE", "off")
	if got := openAIPromptCacheHintsFor("openai-codex", "gpt-5.5", "system", nil); got.Key != "" || got.Retention != "" {
		t.Fatalf("disabled cache got hints: %+v", got)
	}
}

func TestOpenAIModelUsesModernPromptCache(t *testing.T) {
	cases := map[string]bool{
		"gpt-5.5":       false,
		"gpt-5.6":       true,
		"gpt-5.6-terra": true,
		"GPT-6":         true,
		"o4-mini":       false,
	}
	for model, want := range cases {
		if got := openAIModelUsesModernPromptCache(model); got != want {
			t.Errorf("openAIModelUsesModernPromptCache(%q)=%v want %v", model, got, want)
		}
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
