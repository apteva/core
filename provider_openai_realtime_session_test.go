package core

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/gobwas/ws"
)

func TestOpenAIRealtimeSessionUpdateUsesGAContract(t *testing.T) {
	payload, err := buildSessionUpdate(RealtimeSessionOpts{
		Model: "gpt-realtime-2.1", Voice: "marin", Instructions: "full core prompt",
		Tools:      []NativeTool{{Name: "send", Description: "send", Parameters: map[string]any{"type": "object"}}},
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		Reasoning: "high", TranscribeInput: true,
	}, "marin")
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
	if session["type"] != "realtime" || session["model"] != "gpt-realtime-2.1" {
		t.Fatalf("GA session identity missing: %#v", session)
	}
	if session["instructions"] != "full core prompt" {
		t.Fatalf("instructions = %v", session["instructions"])
	}
	if _, stale := session["modalities"]; stale {
		t.Fatal("preview modalities field must not be emitted")
	}
	if _, stale := session["input_audio_format"]; stale {
		t.Fatal("preview input_audio_format field must not be emitted")
	}
	audio := session["audio"].(map[string]any)
	input := audio["input"].(map[string]any)
	inputFormat := input["format"].(map[string]any)
	if inputFormat["type"] != "audio/pcm" || inputFormat["rate"] != float64(24000) {
		t.Fatalf("PCM input format missing rate: %#v", inputFormat)
	}
	output := audio["output"].(map[string]any)
	outputFormat := output["format"].(map[string]any)
	if outputFormat["type"] != "audio/pcm" || outputFormat["rate"] != float64(24000) {
		t.Fatalf("PCM output format missing rate: %#v", outputFormat)
	}
	if input["transcription"].(map[string]any)["model"] != "gpt-4o-mini-transcribe" {
		t.Fatalf("input transcription missing: %#v", input)
	}
	turnDetection := input["turn_detection"].(map[string]any)
	if turnDetection["interrupt_response"] != true || turnDetection["create_response"] != true {
		t.Fatalf("server VAD interruption missing: %#v", turnDetection)
	}
	if session["reasoning"].(map[string]any)["effort"] != "high" {
		t.Fatalf("reasoning not forwarded: %#v", session["reasoning"])
	}
	if strings.Contains(string(payload), "response.audio.delta") {
		t.Fatal("preview event name leaked into session payload")
	}
}

func newTranslationSession() *openaiRealtimeSession {
	return &openaiRealtimeSession{
		events: make(chan RealtimeEvent, 16),
		outbox: make(chan realtimeOutboundFrame, 16),
		done:   make(chan struct{}),
	}
}

func TestOpenAIRealtimeTranslatesGAEventsAndUsage(t *testing.T) {
	session := newTranslationSession()
	session.translate(&openaiRealtimeEvent{
		Type: "response.output_audio.delta", ItemID: "item-1",
		Delta: base64.StdEncoding.EncodeToString([]byte{1, 2, 3}),
	})
	audio := <-session.events
	if audio.Type != RealtimeEventAudioOut || audio.ItemID != "item-1" || string(audio.Audio) != string([]byte{1, 2, 3}) {
		t.Fatalf("audio event = %#v", audio)
	}

	var wire openaiRealtimeEvent
	if err := json.Unmarshal([]byte(`{
		"type":"response.done","response":{"id":"resp-1","usage":{
			"total_tokens":21,"input_tokens":13,"output_tokens":8,
			"input_token_details":{"text_tokens":5,"audio_tokens":8,"cached_tokens":4,
				"cached_tokens_details":{"text_tokens":1,"audio_tokens":3}},
			"output_token_details":{"text_tokens":2,"audio_tokens":6}
		}}}`), &wire); err != nil {
		t.Fatal(err)
	}
	session.translate(&wire)
	done := <-session.events
	if done.Type != RealtimeEventResponseDone || done.ResponseID != "resp-1" {
		t.Fatalf("done event = %#v", done)
	}
	if done.Usage.TextCachedTokens != 1 || done.Usage.AudioCachedTokens != 3 || done.Usage.AudioOutputTokens != 6 {
		t.Fatalf("usage = %#v", done.Usage)
	}

	session.translate(&openaiRealtimeEvent{Type: "input_audio_buffer.speech_started", ItemID: "input-1", AudioStartMS: 420})
	speech := <-session.events
	if speech.Type != RealtimeEventSpeechStarted || speech.AudioStartMS != 420 {
		t.Fatalf("speech event = %#v", speech)
	}
}

func TestOpenAIRealtimeCloseAndEnqueueDoNotRaceOnClosedChannel(t *testing.T) {
	session := newTranslationSession()
	var wait sync.WaitGroup
	for i := 0; i < 32; i++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for j := 0; j < 100; j++ {
				_ = session.enqueueFrame(realtimeOutboundFrame{op: ws.OpText, data: []byte("{}")})
			}
		}()
	}
	_ = session.Close()
	wait.Wait()
	if err := session.SendAudio([]byte{1}); err == nil {
		t.Fatal("send after close should fail")
	}
}

func TestRealtimeCostSeparatesCachedTextAndAudio(t *testing.T) {
	provider := NewOpenAIRealtimeProvider("test")
	cost := calculateCostForRealtimeProvider(provider, "gpt-realtime-2.1", RealtimeUsage{
		TextInputTokens: 100, TextCachedTokens: 25, TextOutputTokens: 10,
		AudioInputTokens: 80, AudioCachedTokens: 20, AudioOutputTokens: 5,
	})
	want := (75*4.0 + 25*0.4 + 10*24.0 + 60*32.0 + 20*0.4 + 5*64.0) / 1_000_000
	if cost != want {
		t.Fatalf("cost=%v want=%v", cost, want)
	}
}

func TestRealtimeProviderConfigAppliesModelsAndVoice(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	pool, err := buildProviderPool(&Config{
		RealtimeEnabled: true,
		Providers: []ProviderConfig{
			{Name: "openai", Default: true},
			{Name: "openai-realtime", Models: map[string]string{
				"large": "rt-custom-large", "medium": "rt-custom-medium", "small": "rt-custom-small",
			}, RealtimeVoice: "cedar"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := pool.RealtimeByName("openai-realtime").(*OpenAIRealtimeProvider)
	if !ok {
		t.Fatalf("provider = %T", pool.RealtimeByName("openai-realtime"))
	}
	if provider.Models()[ModelSmall] != "rt-custom-small" || provider.DefaultVoice() != "cedar" {
		t.Fatalf("models=%#v voice=%q", provider.Models(), provider.DefaultVoice())
	}
}
