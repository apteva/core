package core

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
)

// A core gets provider credentials only from its environment at start. When
// the provider chosen as the agent's default cannot be built from that
// environment, the agent keeps that choice and reports it instead of running
// on whichever other provider happens to be available. Requests fail with
// ErrProviderUnavailable, are not retried, and are never handed to another
// provider. Picking another provider remains an explicit choice.

// ErrProviderUnavailable marks a request for a provider this core cannot run.
var ErrProviderUnavailable = errors.New("provider unavailable")

// ProviderUnavailable names a chosen provider this core cannot run, and why.
type ProviderUnavailable struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

type providerUnavailableError struct{ info ProviderUnavailable }

func (e *providerUnavailableError) Error() string {
	return fmt.Sprintf("provider %q cannot run in this agent: %s. Connect it if needed, then restart the agent", e.info.Name, e.info.Reason)
}

func (e *providerUnavailableError) Unwrap() error { return ErrProviderUnavailable }

// providerRequiredEnv lists the variables createProviderByName needs for each
// provider. It only words the reason; createProviderByName stays the
// authority on whether a provider can be built.
var providerRequiredEnv = map[string][]string{
	"managed":             {"APTEVA_MANAGED_LLM_URL", "APTEVA_API_KEY"},
	"fireworks":           {"FIREWORKS_API_KEY"},
	"opencode-go":         {"OPENCODE_GO_API_KEY"},
	"openai":              {"OPENAI_API_KEY"},
	"openai-codex":        {"OPENAI_CODEX_ACCESS_TOKEN"},
	"grok-build":          {"GROK_BUILD_ACCESS_TOKEN"},
	"claude-subscription": {"CLAUDE_SUBSCRIPTION_CONNECTION_ID", "SERVER_URL", "APTEVA_API_KEY"},
	"anthropic":           {"ANTHROPIC_API_KEY"},
	"google":              {"GOOGLE_API_KEY"},
	"xai":                 {"XAI_API_KEY"},
	"ollama":              {"OLLAMA_HOST"},
	"nvidia":              {"NVIDIA_API_KEY"},
	"venice":              {"VENICE_API_KEY"},
}

// missingProviderEnv returns the required variables that are not set.
func missingProviderEnv(name string) []string {
	var missing []string
	for _, key := range providerRequiredEnv[name] {
		if strings.TrimSpace(os.Getenv(key)) == "" {
			missing = append(missing, key)
		}
	}
	return missing
}

func providerUnavailableReason(name string) string {
	if _, known := providerRequiredEnv[name]; !known {
		return "this core does not know a provider by that name"
	}
	missing := missingProviderEnv(name)
	switch len(missing) {
	case 0:
		return "its credentials were not usable when the agent started"
	case 1:
		return missing[0] + " was not set when the agent started"
	default:
		return strings.Join(missing, ", ") + " were not set when the agent started"
	}
}

// unavailableProvider stands in for a chosen provider that cannot run, so
// everything that reads the default provider sees the agent's real choice.
type unavailableProvider struct{ info ProviderUnavailable }

func (p *unavailableProvider) Chat(context.Context, []Message, string, []NativeTool, func(string), func(string), func(string, string, string)) (ChatResponse, error) {
	return ChatResponse{Provider: p.info.Name}, &providerUnavailableError{info: p.info}
}
func (p *unavailableProvider) Models() map[ModelTier]string           { return map[ModelTier]string{} }
func (p *unavailableProvider) Name() string                           { return p.info.Name }
func (p *unavailableProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (p *unavailableProvider) SupportsNativeTools() bool              { return true }
func (p *unavailableProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (p *unavailableProvider) SetBuiltinTools([]string)               {}
func (p *unavailableProvider) WithBuiltins([]string) LLMProvider      { return p }

// providerUnavailableInfo reports whether p is a chosen provider that cannot run.
func providerUnavailableInfo(p LLMProvider) *ProviderUnavailable {
	stub, ok := p.(*unavailableProvider)
	if !ok || stub == nil {
		return nil
	}
	info := stub.info
	return &info
}

// Unavailable returns the default provider this pool could not build, if any.
func (pp *ProviderPool) Unavailable() *ProviderUnavailable {
	if pp == nil || pp.unavailable == nil {
		return nil
	}
	return providerUnavailableInfo(pp.unavailable)
}

func (pp *ProviderPool) markDefaultUnavailable(name string) {
	pp.default_ = name
	pp.unavailable = &unavailableProvider{info: ProviderUnavailable{Name: name, Reason: providerUnavailableReason(name)}}
}
