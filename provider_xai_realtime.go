package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// XAIRealtimeProvider implements xAI's Voice Agent WebSocket API. The API is
// event-compatible with OpenAI Realtime, while retaining its own session
// configuration shape, transcription events, voices, reasoning, and billing.
type XAIRealtimeProvider struct {
	apiKey       string
	models       map[ModelTier]string
	endpoint     string
	defaultVoice string
}

func NewXAIRealtimeProvider(apiKey string) *XAIRealtimeProvider {
	return &XAIRealtimeProvider{
		apiKey:       apiKey,
		endpoint:     "wss://api.x.ai/v1/realtime",
		defaultVoice: "eve",
		models: map[ModelTier]string{
			ModelLarge:  "grok-voice-latest",
			ModelMedium: "grok-voice-latest",
			ModelSmall:  "grok-voice-latest",
		},
	}
}

func (p *XAIRealtimeProvider) Name() string { return "xai-realtime" }

func (p *XAIRealtimeProvider) Models() map[ModelTier]string {
	out := make(map[ModelTier]string, len(p.models))
	for tier, model := range p.models {
		out[tier] = model
	}
	return out
}

func (p *XAIRealtimeProvider) Pricing(string) RealtimePricing {
	return RealtimePricing{
		AudioInputPerMinute:  0.05,
		AudioOutputPerMinute: 0.05,
		TextInputPerMessage:  0.004,
	}
}

func (p *XAIRealtimeProvider) DefaultVoice() string {
	if p.defaultVoice == "" {
		return "eve"
	}
	return p.defaultVoice
}

func (p *XAIRealtimeProvider) DefaultTranscriptionModel() string { return "grok-transcribe" }

func (p *XAIRealtimeProvider) applyRealtimeConfig(config ProviderConfig) {
	applyRealtimeModelAndVoiceConfig(p.models, &p.defaultVoice, config)
}

func (p *XAIRealtimeProvider) Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	return openAICompatibleRealtimeSession(ctx, opts, openAICompatibleRealtimeConfig{
		providerName:       p.Name(),
		apiKey:             p.apiKey,
		endpoint:           p.endpoint,
		defaultVoice:       p.DefaultVoice(),
		buildSessionUpdate: buildXAISessionUpdate,
		buildConfigurationUpdate: func(instructions string, tools []NativeTool) map[string]any {
			return map[string]any{
				"type": "session.update",
				"session": map[string]any{
					"instructions": instructions,
					"tools":        sessionTools(tools),
				},
			}
		},
	})
}

func buildXAISessionUpdate(opts RealtimeSessionOpts, defaultVoice string) ([]byte, error) {
	normalizedTurnDetection, err := opts.TurnDetection.normalized()
	if err != nil {
		return nil, fmt.Errorf("xai-realtime turn detection: %w", err)
	}
	resolvedTurnDetection := normalizedTurnDetection.resolved()
	voice := strings.TrimSpace(opts.Voice)
	if voice == "" {
		voice = defaultVoice
	}
	inFmt := opts.AudioInFmt
	if inFmt == "" {
		inFmt = AudioPCM16
	}
	outFmt := opts.AudioOutFmt
	if outFmt == "" {
		outFmt = AudioPCM16
	}

	input := map[string]any{"format": openAIAudioFormat(inFmt, opts.AudioInRate)}
	if opts.TranscribeInput {
		model := strings.TrimSpace(opts.TranscriptionModel)
		if model == "" {
			model = "grok-transcribe"
		}
		input["transcription"] = map[string]any{"model": model}
	}

	turnDetection := map[string]any{"type": "server_vad"}
	if resolvedTurnDetection.PrefixPaddingMS > 0 {
		turnDetection["prefix_padding_ms"] = resolvedTurnDetection.PrefixPaddingMS
	}
	if resolvedTurnDetection.SilenceDurationMS > 0 {
		turnDetection["silence_duration_ms"] = resolvedTurnDetection.SilenceDurationMS
	}
	session := map[string]any{
		"instructions":   opts.Instructions,
		"voice":          voice,
		"turn_detection": turnDetection,
		"audio": map[string]any{
			"input":  input,
			"output": map[string]any{"format": openAIAudioFormat(outFmt, opts.AudioOutRate)},
		},
		"tools": sessionTools(opts.Tools),
	}
	if effort := strings.ToLower(strings.TrimSpace(opts.Reasoning)); effort != "" && effort != "auto" {
		if effort != "none" {
			effort = "high"
		}
		session["reasoning"] = map[string]any{"effort": effort}
	}
	return json.Marshal(map[string]any{"type": "session.update", "session": session})
}

var _ RealtimeProvider = (*XAIRealtimeProvider)(nil)
var _ configurableRealtimeProvider = (*XAIRealtimeProvider)(nil)
