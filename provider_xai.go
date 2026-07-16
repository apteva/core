package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"time"
)

const xAICacheReuseWindow = 24 * time.Hour

// XAIProvider is a thin vendor adapter over the shared OpenAI-compatible
// Chat Completions transport. The shared transport owns message conversion,
// native function calling, streaming, and usage parsing; this wrapper adds
// only xAI's reasoning parameter and cache-routing header.
type XAIProvider struct {
	compat    *OpenAICompatProvider
	reasoning ReasoningSettings
	now       func() time.Time
}

func NewXAIProvider(apiKey string) LLMProvider {
	return &XAIProvider{
		compat: &OpenAICompatProvider{
			name:       "xai",
			apiKey:     apiKey,
			url:        "https://api.x.ai/v1/chat/completions",
			authHeader: "Bearer",
			models: map[ModelTier]string{
				ModelLarge:  "grok-4.5",
				ModelMedium: "grok-4.3",
				ModelSmall:  "grok-4.3",
			},
			// Core no longer estimates request cost. The server enriches
			// telemetry from its live, model-specific pricing catalog.
			inputCost:  0,
			cachedCost: 0,
			outputCost: 0,
		},
		now: time.Now,
	}
}

func (p *XAIProvider) Name() string { return p.compat.Name() }

func (p *XAIProvider) Models() map[ModelTier]string { return p.compat.Models() }

func (p *XAIProvider) CostPer1M() (float64, float64, float64) {
	return p.compat.CostPer1M()
}

func (p *XAIProvider) SupportsNativeTools() bool { return true }

// Provider-managed xAI web/X/code tools are intentionally not exposed by the
// Chat Completions adapter. Apteva's normal MCP/function tools remain native.
func (p *XAIProvider) AvailableBuiltinTools() []BuiltinTool { return nil }

func (p *XAIProvider) SetBuiltinTools(_ []string) {}

func (p *XAIProvider) WithBuiltins(_ []string) LLMProvider { return p }

func (p *XAIProvider) WithReasoning(settings ReasoningSettings) LLMProvider {
	clone := *p
	clone.reasoning = settings
	return &clone
}

func (p *XAIProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	fields := map[string]any{
		// xAI only includes usage (including cached tokens and exact cost)
		// in the final SSE event when this streaming option is enabled.
		"stream_options": map[string]any{"include_usage": true},
	}
	if effort := xAIReasoningEffort(model, p.reasoning.Level); effort != "" {
		fields["reasoning_effort"] = effort
	}

	headers := map[string]string{}
	scope := openAIPromptCacheScopeFromContext(ctx)
	now := time.Now()
	if p.now != nil {
		now = p.now()
	}
	if conversationID := xAIConversationID(scope, model, now); conversationID != "" {
		headers["x-grok-conv-id"] = conversationID
	}

	ctx = withOpenAICompatRequestOptions(ctx, openAICompatRequestOptions{
		Fields:  fields,
		Headers: headers,
	})
	return p.compat.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
}

func xAIReasoningEffort(model string, level ReasoningLevel) string {
	switch normalizeReasoningLevel(level) {
	case ReasoningAuto:
		return ""
	case ReasoningNone:
		return xAIClampReasoningEffort(model, "none")
	case ReasoningMinimal:
		return xAIClampReasoningEffort(model, "low")
	case ReasoningLow:
		return xAIClampReasoningEffort(model, "low")
	case ReasoningMedium:
		return xAIClampReasoningEffort(model, "medium")
	case ReasoningHigh:
		return xAIClampReasoningEffort(model, "high")
	case ReasoningXHigh:
		return xAIClampReasoningEffort(model, "xhigh")
	default:
		return ""
	}
}

func xAIClampReasoningEffort(model, requested string) string {
	if caps, ok := capabilitiesForModel(model); ok {
		return reasoningEffortForCapabilities(caps, requested)
	}

	// Safe fallbacks for current xAI families when a core is started without
	// catalog metadata. Grok 4.5 cannot disable reasoning and supports up to
	// high; Grok 4.20 multi-agent additionally accepts xhigh.
	model = strings.ToLower(strings.TrimSpace(model))
	if strings.Contains(model, "grok-4.5") {
		switch requested {
		case "none":
			return "low"
		case "xhigh":
			return "high"
		}
	}
	if requested == "xhigh" && !strings.Contains(model, "multi-agent") {
		return "high"
	}
	return requested
}

// xAIConversationID uses the same agent/thread identity and intentional cache
// epoch as the rest of core. A UTC window bucket prevents Apteva from reusing
// a routing identity for more than 24 hours; xAI controls actual provider-side
// cache eviction and does not expose a Chat Completions retention setting.
func xAIConversationID(scope openAIPromptCacheScope, model string, now time.Time) string {
	if strings.TrimSpace(scope.Identity) == "" {
		return ""
	}
	bucket := now.UTC().Unix() / int64(xAICacheReuseWindow/time.Second)
	h := sha256.New()
	h.Write([]byte("apteva-xai-cache-v1\n"))
	h.Write([]byte(scope.Identity))
	h.Write([]byte{'\n'})
	h.Write([]byte(strconv.FormatUint(scope.Epoch, 10)))
	h.Write([]byte{'\n'})
	h.Write([]byte(strings.TrimSpace(model)))
	h.Write([]byte{'\n'})
	h.Write([]byte(strconv.FormatInt(bucket, 10)))
	sum := h.Sum(nil)
	return "apteva-xai-v1-" + hex.EncodeToString(sum[:16])
}

var _ LLMProvider = (*XAIProvider)(nil)
var _ ReasoningProvider = (*XAIProvider)(nil)
