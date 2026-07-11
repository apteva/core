package core

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// llmHTTPClient is a shared HTTP client for all LLM provider calls.
// It uses a response header timeout to catch dead provider requests but no
// overall timeout since streaming responses can legitimately take minutes.
var llmHTTPClient = &http.Client{
	Transport: &http.Transport{
		ResponseHeaderTimeout: llmResponseHeaderTimeout(),
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
	},
}

func llmResponseHeaderTimeout() time.Duration {
	if v := os.Getenv("APTEVA_LLM_RESPONSE_HEADER_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 180 * time.Second
}

// streamIdleTimeout is how long we wait between bytes on a streaming
// provider response before declaring the stream stalled. Chosen above
// typical reasoning-model think pauses (which can hit 30-40s on deep
// chain-of-thought) but well below the point where a real user would
// give up. Override with APTEVA_STREAM_IDLE_TIMEOUT=<seconds>.
func streamIdleTimeout() time.Duration {
	if v := os.Getenv("APTEVA_STREAM_IDLE_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return 60 * time.Second
}

// extractProviderRequestIDs pulls known request-id headers from an
// HTTP response so we can tag telemetry with values the provider's own
// support can grep. Different vendors ship different headers:
//
//	Fireworks:  x-request-id
//	Anthropic:  request-id, x-request-id
//	OpenAI:     openai-request-id, x-request-id
//	Google:     x-request-id, x-goog-request-id
//
// Return as a name=value map preserving order of known headers we
// checked; keeps the logs readable.
func extractProviderRequestIDs(h http.Header) map[string]string {
	candidates := []string{
		"x-request-id",
		"x-fw-request-id",
		"fireworks-request-id",
		"openai-request-id",
		"request-id",
		"x-goog-request-id",
	}
	out := map[string]string{}
	for _, name := range candidates {
		if v := h.Get(name); v != "" {
			out[name] = v
		}
	}
	return out
}

// ErrStreamIdleTimeout is returned by a stream reader that went silent
// for longer than the idle window. Callers can branch on it to tag
// telemetry as a stall instead of a generic I/O error.
var ErrStreamIdleTimeout = errors.New("stream idle timeout (provider went silent mid-response)")

// idleReader wraps an io.ReadCloser with a per-read idle timer. Every
// Read resets the timer; if the timer fires (no bytes for idleTimeout),
// the underlying body is closed, the next Read returns
// ErrStreamIdleTimeout, and the connection goroutine unblocks cleanly.
//
// We rely on the HTTP server eventually flushing SOMETHING on a healthy
// stream (even an SSE comment `: keepalive\n\n` counts). A fully silent
// stall that lasts past idleTimeout is by definition unrecoverable
// from the client side — the right move is to abort and let the think
// loop retry on the next iteration.
type idleReader struct {
	body        io.ReadCloser
	timer       *time.Timer
	idleTimeout time.Duration
	onStall     func() // called when the timer fires; receiver can log
	mu          sync.Mutex
	closed      bool
	stalled     bool
}

func newIdleReader(body io.ReadCloser, idleTimeout time.Duration, onStall func()) *idleReader {
	ir := &idleReader{body: body, idleTimeout: idleTimeout, onStall: onStall}
	ir.timer = time.AfterFunc(idleTimeout, ir.fireStall)
	return ir
}

func (ir *idleReader) fireStall() {
	ir.mu.Lock()
	ir.stalled = true
	ir.mu.Unlock()
	if ir.onStall != nil {
		ir.onStall()
	}
	// Closing the body causes the in-flight Read to unblock with an
	// error immediately, which is what we want.
	_ = ir.body.Close()
}

func (ir *idleReader) Read(p []byte) (int, error) {
	n, err := ir.body.Read(p)
	if n > 0 {
		// Every byte resets the idle window — healthy activity keeps
		// the connection alive indefinitely.
		ir.timer.Reset(ir.idleTimeout)
	}
	if err != nil {
		ir.mu.Lock()
		stalled := ir.stalled
		ir.mu.Unlock()
		if stalled {
			return n, ErrStreamIdleTimeout
		}
	}
	return n, err
}

func (ir *idleReader) Close() error {
	ir.mu.Lock()
	if ir.closed {
		ir.mu.Unlock()
		return nil
	}
	ir.closed = true
	ir.mu.Unlock()
	ir.timer.Stop()
	return ir.body.Close()
}

// NativeTool defines a tool sent to the provider API.
type NativeTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
}

// NativeToolCall is a structured tool call returned by the provider.
type NativeToolCall struct {
	ID               string            `json:"id"` // provider-assigned ID for matching results
	Name             string            `json:"name"`
	Args             map[string]string `json:"args"`
	ThoughtSignature string            `json:"thought_signature,omitempty"` // Gemini: encrypted reasoning state
}

// ToolResult is sent back to the provider after executing a tool.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content string `json:"content"`         // text result
	Image   []byte `json:"image,omitempty"` // optional image (screenshot etc.)
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
	Reasoning     string             // accumulated chain-of-thought (Fireworks reasoning_content / OpenRouter reasoning); empty when the provider didn't emit any
	ToolCalls     []NativeToolCall   // structured tool calls WE need to execute
	ServerResults []ServerToolResult // tools the PROVIDER already executed
	Usage         TokenUsage
}

// LLMProvider abstracts the LLM API call.
// All thinking, threading, tool handling stays in the Thinker.
// The provider only handles: send messages → get streaming response.
type LLMProvider interface {
	// Chat sends messages and streams the response.
	// ctx is propagated to the underlying HTTP request so cancellation
	// (user abort, shutdown) unblocks an in-flight stream cleanly.
	// tools: native tool definitions to include in the request (nil = no tools).
	// onChunk is called for each text token chunk as it arrives.
	// onThinking is called for each reasoning/thinking token (separate from output).
	// onToolChunk is called for each tool argument chunk as it streams
	// (toolName, callID, argChunk). callID disambiguates two parallel
	// calls of the same tool in one response — providers pass their own
	// stable per-call id (tc.Index for OpenAI, content_block id for
	// Anthropic). Empty string is acceptable.
	// Returns ChatResponse with text, tool calls, and usage.
	Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(toolName, callID, chunk string)) (ChatResponse, error)

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

// ReasoningProvider is an optional LLMProvider extension for APIs that
// support request-level reasoning or thinking effort controls.
type ReasoningProvider interface {
	WithReasoning(settings ReasoningSettings) LLMProvider
}

// createRealtimeProviderByName creates a RealtimeProvider by name,
// returning nil if the required API key is missing or the name is
// unknown. Mirrors createProviderByName but for the realtime
// interface. Realtime providers are NEVER auto-detected — callers
// must opt in explicitly via config (cfg.Providers[] + RealtimeEnabled).
func createRealtimeProviderByName(name string) RealtimeProvider {
	switch name {
	case "openai-realtime":
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return NewOpenAIRealtimeProvider(key)
		}
	}
	return nil
}

// isRealtimeProviderName returns true if the given name belongs to a
// known realtime provider. Used to route registration in
// buildProviderPool without trying both factories blindly.
func isRealtimeProviderName(name string) bool {
	switch name {
	case "openai-realtime":
		return true
	}
	return false
}

// createProviderByName creates a provider by name, returning nil if the required API key is missing.
func createProviderByName(name string) LLMProvider {
	switch name {
	case "fireworks":
		if key := os.Getenv("FIREWORKS_API_KEY"); key != "" {
			return NewFireworksProvider(key)
		}
	case "opencode-go":
		if key := os.Getenv("OPENCODE_GO_API_KEY"); key != "" {
			return NewOpenCodeGoProvider(key)
		}
	case "openai":
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return NewOpenAINativeProvider(key)
		}
	case "openai-codex":
		if token := os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"); token != "" {
			return NewOpenAICodexProvider(token)
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
		// Require explicit OLLAMA_HOST. Previously we silently defaulted
		// to http://localhost:11434, which meant every install with no
		// LLM credentials at all got a "phantom" Ollama provider that
		// ALWAYS failed at request time (no Ollama running). When Ollama
		// was first in the auto-detect list, the result was an llama3.1
		// 404 every iteration before falling back to whichever real
		// provider was actually configured. Opt-in only.
		host := os.Getenv("OLLAMA_HOST")
		if host == "" {
			return nil
		}
		return NewOllamaProvider(host)
	case "nvidia":
		// NIM (NVIDIA Inference Microservices) — OpenAI-compatible REST
		// endpoint hosted at integrate.api.nvidia.com. See
		// provider_openai.go:NewNvidiaProvider for the default model
		// slugs.
		if key := os.Getenv("NVIDIA_API_KEY"); key != "" {
			return NewNvidiaProvider(key)
		}
	case "venice":
		// Venice AI — privacy-focused OpenAI-compatible gateway with a
		// large rotating model catalog (Llama, Qwen, GLM, Mistral,
		// plus Claude/Grok/Gemini reseller variants). See
		// provider_openai.go:NewVeniceProvider.
		if key := os.Getenv("VENICE_API_KEY"); key != "" {
			return NewVeniceProvider(key)
		}
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
	case *OpenAINativeProvider:
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
//
// Realtime providers (RealtimeProvider) live in a separate map so the
// existing LLM-provider plumbing (Default, Fallback, ProviderSummary,
// etc.) stays untouched. A name can in principle appear in both maps
// if a vendor offers both APIs, though in practice they're distinct
// (e.g. "openai" text vs "openai-realtime").
type ProviderPool struct {
	providers         map[string]LLMProvider      // "fireworks" → instance
	order             []string                    // provider names in config order (fallback order)
	default_          string                      // default provider name
	realtimeProviders map[string]RealtimeProvider // "openai-realtime" → instance
	realtimeOrder     []string                    // realtime names in registration order
	realtimeDefault   string                      // default realtime provider name
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

// HasRealtimeProvider reports whether any RealtimeProvider is
// registered. Used as one half of the realtime feature gate (the
// other being Config.RealtimeEnabled): if no provider is registered,
// the main thread is never told realtime exists and spawn rejects
// realtime=true.
func (pp *ProviderPool) HasRealtimeProvider() bool {
	return pp != nil && len(pp.realtimeProviders) > 0
}

// RealtimeDefault returns the default RealtimeProvider, or nil if
// none registered.
func (pp *ProviderPool) RealtimeDefault() RealtimeProvider {
	if pp == nil {
		return nil
	}
	if p, ok := pp.realtimeProviders[pp.realtimeDefault]; ok {
		return p
	}
	if len(pp.realtimeOrder) > 0 {
		return pp.realtimeProviders[pp.realtimeOrder[0]]
	}
	return nil
}

// RealtimeByName returns a RealtimeProvider by name, or nil.
func (pp *ProviderPool) RealtimeByName(name string) RealtimeProvider {
	if pp == nil {
		return nil
	}
	return pp.realtimeProviders[name]
}

// RealtimeNames returns all registered realtime provider names.
func (pp *ProviderPool) RealtimeNames() []string {
	if pp == nil {
		return nil
	}
	return pp.realtimeOrder
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
	if _, ok := p.(ReasoningProvider); ok {
		summary += "\n    reasoning: auto none minimal low medium high xhigh"
	}
	return summary
}

// buildProviderPool creates a ProviderPool from config + env vars.
// Priority: CORE_PROVIDER env → config.json providers → auto-detect from API keys.
func buildProviderPool(cfg *Config) (*ProviderPool, error) {
	pool := &ProviderPool{
		providers:         map[string]LLMProvider{},
		realtimeProviders: map[string]RealtimeProvider{},
	}
	resetRuntimeModelCapabilities()

	// 1. Config providers array
	configs := cfg.GetProviders()
	for _, pc := range configs {
		// Route realtime providers to their own map. Gated on
		// Config.RealtimeEnabled: when off, realtime entries are
		// silently skipped so HasRealtimeProvider() returns false and
		// the feature is completely invisible to main.
		if isRealtimeProviderName(pc.Name) {
			if !cfg.RealtimeEnabledFlag() {
				continue
			}
			rp := createRealtimeProviderByName(pc.Name)
			if rp == nil {
				continue
			}
			pool.realtimeProviders[pc.Name] = rp
			pool.realtimeOrder = append(pool.realtimeOrder, pc.Name)
			if pc.Default {
				pool.realtimeDefault = pc.Name
			}
			continue
		}
		p := createProviderByName(pc.Name)
		if p == nil {
			continue
		}
		applyModelOverrides(p, pc.Models)
		registerModelCapabilities(pc.ModelCapabilities)
		if native, ok := p.(*OpenAINativeProvider); ok {
			native.modelCapabilities = cloneModelCapabilitiesMap(pc.ModelCapabilities)
		}
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

	// 3. Auto-detect from API keys if nothing configured.
	//
	// Order matters: the first provider whose key is set becomes the
	// default. opencode-go is preferred over token-billed providers
	// because it's a flat-rate subscription — if the user configured
	// it, they want it used (otherwise the per-token providers below
	// will silently win and burn budget). Ollama is last because it
	// requires OLLAMA_HOST set explicitly and only returns non-nil
	// when the user has it running.
	if len(pool.providers) == 0 {
		for _, name := range []string{"opencode-go", "fireworks", "anthropic", "google", "openai", "venice", "nvidia", "ollama"} {
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
	if key := os.Getenv("OPENCODE_GO_API_KEY"); key != "" {
		providers = append(providers, NewOpenCodeGoProvider(key))
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
	if key := os.Getenv("NVIDIA_API_KEY"); key != "" {
		providers = append(providers, NewNvidiaProvider(key))
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

// calculateCostForRealtimeProvider mirrors calculateCostForProvider
// but uses the 4-arg pricing tuple (the audio rate is separate from
// text I/O on realtime APIs). Returned value is in dollars assuming
// the per-1M figures are dollar-denominated, same as the text path.
func calculateCostForRealtimeProvider(provider RealtimeProvider, usage TokenUsage) float64 {
	inputPer1M, cachedPer1M, outputPer1M, audioPer1M := provider.CostPer1M()
	uncached := usage.PromptTokens - usage.CachedTokens
	if uncached < 0 {
		uncached = 0
	}
	return (float64(uncached)*inputPer1M +
		float64(usage.CachedTokens)*cachedPer1M +
		float64(usage.CompletionTokens)*outputPer1M +
		float64(usage.AudioTokens)*audioPer1M) / 1_000_000
}
