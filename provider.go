package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"time"
)

// llmHTTPClient is a shared HTTP client for all LLM provider calls.
// It uses a response header timeout (30s to get first byte) but no overall
// timeout since streaming responses can legitimately take minutes.
var llmHTTPClient = &http.Client{
	Transport: &http.Transport{
		ResponseHeaderTimeout: 30 * time.Second,
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
	},
}

// NativeTool defines a tool sent to the provider API.
type NativeTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
}

// NativeToolCall is a structured tool call returned by the provider.
type NativeToolCall struct {
	ID   string            `json:"id"`   // provider-assigned ID for matching results
	Name string            `json:"name"`
	Args map[string]string `json:"args"`
}

// ToolResult is sent back to the provider after executing a tool.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content string `json:"content"`           // text result
	Image   []byte `json:"image,omitempty"`   // optional image (screenshot etc.)
	IsError bool   `json:"is_error,omitempty"`
}

// BuiltinTool defines a provider-side tool (executed by the LLM provider, not by us).
type BuiltinTool struct {
	Type string `json:"type"` // e.g. "code_execution_20250825", "code_interpreter"
	Name string `json:"name"` // e.g. "code_execution", "code_interpreter"
}

// ServerToolResult is the result of a built-in tool executed server-side.
type ServerToolResult struct {
	ToolName string `json:"tool_name"`
	Code     string `json:"code,omitempty"`   // code that was executed
	Output   string `json:"output,omitempty"` // stdout/result
	Error    string `json:"error,omitempty"`  // stderr if any
}

// ChatResponse is the structured return from Chat().
type ChatResponse struct {
	Text          string             // streamed text content
	ToolCalls     []NativeToolCall   // structured tool calls WE need to execute
	ServerResults []ServerToolResult // tools the PROVIDER already executed
	Usage         TokenUsage
}

// LLMProvider abstracts the LLM API call.
// All thinking, threading, tool handling stays in the Thinker.
// The provider only handles: send messages → get streaming response.
type LLMProvider interface {
	// Chat sends messages and streams the response.
	// tools: native tool definitions to include in the request (nil = no tools).
	// onChunk is called for each text token chunk as it arrives.
	// onToolChunk is called for each tool argument chunk as it streams (toolName, argChunk).
	// Returns ChatResponse with text, tool calls, and usage.
	Chat(messages []Message, model string, tools []NativeTool, onChunk func(string), onToolChunk func(toolName, chunk string)) (ChatResponse, error)

	// Models returns model IDs for each tier.
	Models() map[ModelTier]string

	// Name returns the provider name for display/telemetry.
	Name() string

	// CostPer1M returns pricing per 1M tokens: (input, cached, output).
	CostPer1M() (float64, float64, float64)

	// SupportsNativeTools returns true if this provider handles structured tool calling.
	SupportsNativeTools() bool

	// AvailableBuiltinTools returns built-in tools this provider supports.
	AvailableBuiltinTools() []BuiltinTool

	// SetBuiltinTools enables specific built-in tools.
	SetBuiltinTools(tools []string)

	// WithBuiltins returns a shallow clone of this provider with only the specified builtins enabled.
	// If builtins is nil, returns the provider unchanged (inherit all).
	WithBuiltins(builtins []string) LLMProvider
}

// createProviderByName creates a provider by name, returning nil if the required API key is missing.
func createProviderByName(name string) LLMProvider {
	switch name {
	case "fireworks":
		if key := os.Getenv("FIREWORKS_API_KEY"); key != "" {
			return NewFireworksProvider(key)
		}
	case "openai":
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return NewOpenAINativeProvider(key)
		}
	case "anthropic":
		if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
			return NewAnthropicProvider(key)
		}
	case "google":
		if key := os.Getenv("GOOGLE_API_KEY"); key != "" {
			return NewGoogleProvider(key)
		}
	case "ollama":
		host := os.Getenv("OLLAMA_HOST")
		if host == "" {
			host = "http://localhost:11434"
		}
		return NewOllamaProvider(host)
	}
	return nil
}

// applyModelOverrides sets model overrides on a provider from a config map.
func applyModelOverrides(provider LLMProvider, models map[string]string) {
	if models == nil {
		return
	}
	large := models["large"]
	medium := models["medium"]
	small := models["small"]

	switch p := provider.(type) {
	case *GoogleProvider:
		if large != "" {
			p.SetModel(large)
		}
		if medium != "" {
			p.models[ModelMedium] = medium
		}
		if small != "" {
			p.models[ModelSmall] = small
		}
	case *OpenAICompatProvider:
		if large != "" {
			p.models[ModelLarge] = large
		}
		if medium != "" {
			p.models[ModelMedium] = medium
		}
		if small != "" {
			p.models[ModelSmall] = small
		}
	case *AnthropicProvider:
		if large != "" {
			p.models[ModelLarge] = large
		}
		if medium != "" {
			p.models[ModelMedium] = medium
		}
		if small != "" {
			p.models[ModelSmall] = small
		}
	}
}

// ProviderPool holds multiple LLM providers keyed by name.
// Supports default selection and fallback on error.
type ProviderPool struct {
	providers map[string]LLMProvider // "fireworks" → instance
	order     []string              // provider names in config order (fallback order)
	default_  string                // default provider name
}

// Get returns a provider by name, or nil if not found.
func (pp *ProviderPool) Get(name string) LLMProvider {
	if pp == nil {
		return nil
	}
	return pp.providers[name]
}

// Default returns the default provider.
func (pp *ProviderPool) Default() LLMProvider {
	if pp == nil {
		return nil
	}
	if p, ok := pp.providers[pp.default_]; ok {
		return p
	}
	// Fallback: first available
	if len(pp.order) > 0 {
		return pp.providers[pp.order[0]]
	}
	return nil
}

// DefaultName returns the name of the default provider.
func (pp *ProviderPool) DefaultName() string {
	if pp == nil {
		return ""
	}
	return pp.default_
}

// Names returns all provider names in config order.
func (pp *ProviderPool) Names() []string {
	if pp == nil {
		return nil
	}
	return pp.order
}

// Fallback returns the next provider in the fallback chain after the excluded one.
func (pp *ProviderPool) Fallback(exclude string) LLMProvider {
	if pp == nil {
		return nil
	}
	for _, name := range pp.order {
		if name != exclude {
			if p, ok := pp.providers[name]; ok {
				return p
			}
		}
	}
	return nil
}

// Count returns the number of providers in the pool.
func (pp *ProviderPool) Count() int {
	if pp == nil {
		return 0
	}
	return len(pp.providers)
}

// ProviderSummary returns a description of a provider for system prompt injection.
func (pp *ProviderPool) ProviderSummary(name string) string {
	p, ok := pp.providers[name]
	if !ok {
		return ""
	}
	models := p.Models()
	summary := name
	if name == pp.default_ {
		summary += " (default)"
	}
	summary += " — models:"
	for _, tier := range []ModelTier{ModelLarge, ModelMedium, ModelSmall} {
		if m, ok := models[tier]; ok && m != "" {
			summary += " " + tier.String() + "=" + m
		}
	}
	builtins := p.AvailableBuiltinTools()
	if len(builtins) > 0 {
		summary += "\n    built-in:"
		for _, bt := range builtins {
			summary += " " + bt.Name
		}
	}
	return summary
}

// buildProviderPool creates a ProviderPool from config + env vars.
// Priority: CORE_PROVIDER env → config.json providers → auto-detect from API keys.
func buildProviderPool(cfg *Config) (*ProviderPool, error) {
	pool := &ProviderPool{providers: map[string]LLMProvider{}}

	// 1. Config providers array
	configs := cfg.GetProviders()
	for _, pc := range configs {
		p := createProviderByName(pc.Name)
		if p == nil {
			continue
		}
		applyModelOverrides(p, pc.Models)
		if len(pc.BuiltinTools) > 0 {
			p.SetBuiltinTools(pc.BuiltinTools)
		}
		pool.providers[pc.Name] = p
		pool.order = append(pool.order, pc.Name)
		if pc.Default {
			pool.default_ = pc.Name
		}
	}

	// 2. CORE_PROVIDER env override (force default)
	if explicit := os.Getenv("CORE_PROVIDER"); explicit != "" {
		if _, ok := pool.providers[explicit]; !ok {
			p := createProviderByName(explicit)
			if p == nil {
				return nil, fmt.Errorf("provider %q requested via CORE_PROVIDER but required API key not set", explicit)
			}
			pool.providers[explicit] = p
			pool.order = append([]string{explicit}, pool.order...)
		}
		pool.default_ = explicit
	}

	// 3. Auto-detect from API keys if nothing configured
	if len(pool.providers) == 0 {
		for _, name := range []string{"fireworks", "openai", "anthropic", "google", "ollama"} {
			if p := createProviderByName(name); p != nil {
				pool.providers[name] = p
				pool.order = append(pool.order, name)
			}
		}
	}

	if len(pool.providers) == 0 {
		return nil, fmt.Errorf("no LLM provider configured — set FIREWORKS_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_HOST")
	}

	// Default = first with Default flag, or first in order
	if pool.default_ == "" && len(pool.order) > 0 {
		pool.default_ = pool.order[0]
	}

	// 4. Env model overrides (highest priority for models, applied to default)
	envModels := map[string]string{}
	if v := os.Getenv("CORE_MODEL_LARGE"); v != "" {
		envModels["large"] = v
	}
	if v := os.Getenv("CORE_MODEL_MEDIUM"); v != "" {
		envModels["medium"] = v
	}
	if v := os.Getenv("CORE_MODEL_SMALL"); v != "" {
		envModels["small"] = v
	}
	if len(envModels) > 0 {
		if def := pool.Default(); def != nil {
			applyModelOverrides(def, envModels)
		}
	}

	return pool, nil
}

// selectProvider picks the default provider from a pool. Backward-compatible wrapper.
func selectProvider(cfg *Config) (LLMProvider, error) {
	pool, err := buildProviderPool(cfg)
	if err != nil {
		return nil, err
	}
	return pool.Default(), nil
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
