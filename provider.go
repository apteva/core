package main

import (
	"fmt"
	"os"
)

// LLMProvider abstracts the LLM API call.
// All thinking, threading, tool handling stays in the Thinker.
// The provider only handles: send messages → get streaming response.
type LLMProvider interface {
	// Chat sends messages and streams the response.
	// onChunk is called for each token chunk as it arrives.
	// Returns the full response text, token usage, and any error.
	Chat(messages []Message, model string, onChunk func(string)) (string, TokenUsage, error)

	// Models returns model IDs for each tier.
	Models() map[ModelTier]string

	// Name returns the provider name for display/telemetry.
	Name() string

	// CostPer1M returns pricing per 1M tokens: (input, cached, output).
	CostPer1M() (float64, float64, float64)
}

// selectProvider picks the best available LLM provider based on env vars.
// Priority: explicit COGITO_PROVIDER env, then check for API keys.
func selectProvider() (LLMProvider, error) {
	explicit := os.Getenv("COGITO_PROVIDER")

	switch explicit {
	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY not set")
		}
		return NewOpenAIProvider(key), nil
	case "anthropic":
		key := os.Getenv("ANTHROPIC_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set")
		}
		return NewAnthropicProvider(key), nil
	case "google":
		key := os.Getenv("GOOGLE_API_KEY")
		if key == "" {
			return nil, fmt.Errorf("GOOGLE_API_KEY not set")
		}
		return NewGoogleProvider(key), nil
	case "ollama":
		host := os.Getenv("OLLAMA_HOST")
		if host == "" {
			host = "http://localhost:11434"
		}
		return NewOllamaProvider(host), nil
	}

	// Auto-detect: check env vars in priority order
	if key := os.Getenv("FIREWORKS_API_KEY"); key != "" {
		return NewFireworksProvider(key), nil
	}
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		return NewOpenAIProvider(key), nil
	}
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		return NewAnthropicProvider(key), nil
	}
	if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
		return NewGoogleProvider(key), nil
	}
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return NewOllamaProvider(host), nil
	}

	return nil, fmt.Errorf("no LLM provider configured — set FIREWORKS_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_HOST")
}

// availableProviders returns all providers that have credentials configured.
func availableProviders() []LLMProvider {
	var providers []LLMProvider
	if key := os.Getenv("FIREWORKS_API_KEY"); key != "" {
		providers = append(providers, NewFireworksProvider(key))
	}
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		providers = append(providers, NewOpenAIProvider(key))
	}
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		providers = append(providers, NewAnthropicProvider(key))
	}
	if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
		providers = append(providers, NewGoogleProvider(key))
	}
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		providers = append(providers, NewOllamaProvider(host))
	}
	return providers
}

// calculateCostForProvider computes cost using the provider's pricing.
func calculateCostForProvider(provider LLMProvider, usage TokenUsage) float64 {
	inputPer1M, cachedPer1M, outputPer1M := provider.CostPer1M()
	uncached := usage.PromptTokens - usage.CachedTokens
	if uncached < 0 {
		uncached = 0
	}
	return (float64(uncached)*inputPer1M +
		float64(usage.CachedTokens)*cachedPer1M +
		float64(usage.CompletionTokens)*outputPer1M) / 1_000_000
}
