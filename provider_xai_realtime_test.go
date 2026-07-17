package core

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"testing"
)

func TestXAIRealtimeSessionUpdateUsesVoiceAgentContract(t *testing.T) {
	payload, err := buildXAISessionUpdate(RealtimeSessionOpts{
		Model: "grok-voice-latest", Voice: "ara", Instructions: "full core prompt",
		Tools:      []NativeTool{{Name: "send", Description: "send", Parameters: map[string]any{"type": "object"}}},
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		Reasoning: "medium", TranscribeInput: true,
	}, "eve")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope["type"] != "session.update" {
		t.Fatalf("type = %v", envelope["type"])
	}
	session := envelope["session"].(map[string]any)
	if session["voice"] != "ara" || session["instructions"] != "full core prompt" {
		t.Fatalf("xAI session identity = %#v", session)
	}
	for _, unsupported := range []string{"type", "model", "output_modalities", "tool_choice", "truncation"} {
		if _, found := session[unsupported]; found {
			t.Fatalf("OpenAI-only %q leaked into xAI session: %#v", unsupported, session)
		}
	}
	if session["turn_detection"].(map[string]any)["type"] != "server_vad" {
		t.Fatalf("turn detection = %#v", session["turn_detection"])
	}
	audio := session["audio"].(map[string]any)
	input := audio["input"].(map[string]any)
	if input["transcription"].(map[string]any)["model"] != "grok-transcribe" {
		t.Fatalf("transcription = %#v", input)
	}
	if session["reasoning"].(map[string]any)["effort"] != "high" {
		t.Fatalf("reasoning = %#v", session["reasoning"])
	}
	if len(session["tools"].([]any)) != 1 {
		t.Fatalf("tools = %#v", session["tools"])
	}
}

func TestXAIRealtimeTranslatesCumulativeTranscriptionAndAccountsDuration(t *testing.T) {
	session := &openaiRealtimeSession{
		providerName:           "xai-realtime",
		events:                 make(chan RealtimeEvent, 8),
		outbox:                 make(chan realtimeOutboundFrame, 8),
		done:                   make(chan struct{}),
		audioInBytesPerSecond:  48000,
		audioOutBytesPerSecond: 48000,
	}
	if err := session.SendAudio(make([]byte, 48000)); err != nil {
		t.Fatal(err)
	}
	if err := session.appendText("user", "hello", false); err != nil {
		t.Fatal(err)
	}
	session.translate(&openaiRealtimeEvent{
		Type: "conversation.item.input_audio_transcription.updated", Transcript: "bonjour tout le monde",
	})
	partial := <-session.events
	if partial.Type != RealtimeEventTranscriptInput || partial.Transcript != "bonjour tout le monde" || partial.Final {
		t.Fatalf("partial transcript = %#v", partial)
	}
	session.translate(&openaiRealtimeEvent{Type: "response.text.delta", Delta: "bonjour"})
	textDelta := <-session.events
	if textDelta.Type != RealtimeEventTranscriptOutput || textDelta.Transcript != "bonjour" || textDelta.Final {
		t.Fatalf("text delta = %#v", textDelta)
	}
	session.translate(&openaiRealtimeEvent{
		Type:  "response.output_audio.delta",
		Delta: base64.StdEncoding.EncodeToString(make([]byte, 24000)),
	})
	<-session.events // audio delta
	session.translate(&openaiRealtimeEvent{Type: "response.done"})
	done := <-session.events
	if done.Usage.AudioInputSeconds != 1 || done.Usage.AudioOutputSeconds != 0.5 || done.Usage.TextInputMessages != 1 {
		t.Fatalf("duration usage = %#v", done.Usage)
	}
}

func TestXAIRealtimePricingUsesDurationAndTextMessages(t *testing.T) {
	provider := NewXAIRealtimeProvider("test")
	cost := calculateCostForRealtimeProvider(provider, "grok-voice-latest", RealtimeUsage{
		AudioInputSeconds: 60, AudioOutputSeconds: 30, TextInputMessages: 2,
	})
	if want := 0.083; math.Abs(cost-want) > 1e-12 {
		t.Fatalf("cost=%v want=%v", cost, want)
	}
}

func TestXAIRealtimeProviderRegistrationAndOverrides(t *testing.T) {
	t.Setenv("XAI_API_KEY", "xai-test")
	pool, err := buildProviderPool(&Config{
		RealtimeEnabled: true,
		Providers: []ProviderConfig{{
			Name: "xai-realtime", Default: true,
			Models:        map[string]string{"large": "grok-voice-pinned", "small": "grok-voice-fast-1.0"},
			RealtimeVoice: "rex",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := pool.RealtimeDefault().(*XAIRealtimeProvider)
	if !ok {
		t.Fatalf("provider = %T", pool.RealtimeDefault())
	}
	if provider.Models()[ModelLarge] != "grok-voice-pinned" || provider.Models()[ModelSmall] != "grok-voice-fast-1.0" || provider.DefaultVoice() != "rex" {
		t.Fatalf("models=%#v voice=%q", provider.Models(), provider.DefaultVoice())
	}
}

func TestRealtimeDefaultIsIndependentFromTextProviderMap(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "openai-test")
	t.Setenv("XAI_API_KEY", "xai-test")
	pool, err := buildProviderPool(&Config{
		RealtimeEnabled: true,
		Providers: []ProviderConfig{
			{Name: "openai"},
			{Name: "xai", Default: true},
			{Name: "openai-realtime"},
			{Name: "xai-realtime", Default: true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if pool.Default().Name() != "xai" || pool.RealtimeDefault().Name() != "xai-realtime" {
		t.Fatalf("text=%q realtime=%q", pool.Default().Name(), pool.RealtimeDefault().Name())
	}
}
