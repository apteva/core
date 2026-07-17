package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

func newGoogleRealtimeTestSession() *googleRealtimeSession {
	return &googleRealtimeSession{
		events: make(chan RealtimeEvent, 32),
		outbox: make(chan realtimeOutboundFrame, 32),
		done:   make(chan struct{}), ready: make(chan error, 1),
		inputRate: 24000, callNames: map[string]string{},
	}
}

func TestGoogleLiveSetupUsesNativeProtocol(t *testing.T) {
	payload, err := buildGoogleLiveSetup(RealtimeSessionOpts{
		Model: "gemini-3.1-flash-live-preview", Voice: "Aoede", Instructions: "full core prompt",
		Tools: []NativeTool{{
			Name: "book", Description: "Book a callback.",
			Parameters: map[string]any{"type": "object", "properties": map[string]any{
				"slots": map[string]any{"type": "array"},
			}},
		}},
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
		Reasoning: "medium", TranscribeInput: true,
	}, "Kore")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	setup := envelope["setup"].(map[string]any)
	if setup["model"] != "models/gemini-3.1-flash-live-preview" {
		t.Fatalf("model = %v", setup["model"])
	}
	generation := setup["generationConfig"].(map[string]any)
	voice := generation["speechConfig"].(map[string]any)["voiceConfig"].(map[string]any)["prebuiltVoiceConfig"].(map[string]any)["voiceName"]
	if voice != "Aoede" || generation["thinkingConfig"].(map[string]any)["thinkingLevel"] != "medium" {
		t.Fatalf("generation config = %#v", generation)
	}
	if setup["inputAudioTranscription"] == nil || setup["outputAudioTranscription"] == nil {
		t.Fatalf("transcription config = %#v", setup)
	}
	if setup["historyConfig"].(map[string]any)["initialHistoryInClientContent"] != true {
		t.Fatalf("history config = %#v", setup["historyConfig"])
	}
	tools := setup["tools"].([]any)
	declarations := tools[0].(map[string]any)["functionDeclarations"].([]any)
	parameters := declarations[0].(map[string]any)["parameters"].(map[string]any)
	slots := parameters["properties"].(map[string]any)["slots"].(map[string]any)
	if slots["items"] == nil {
		t.Fatalf("Gemini array schema was not normalized: %#v", slots)
	}
}

func TestGoogleLiveSetupOmitsToolsWhenNoneAndValidatesAudio(t *testing.T) {
	payload, err := buildGoogleLiveSetup(RealtimeSessionOpts{
		Model: "gemini-3.1-flash-live-preview", AudioOutRate: 24000,
	}, "Kore")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	if _, found := envelope["setup"].(map[string]any)["tools"]; found {
		t.Fatalf("empty tools should be omitted: %s", payload)
	}
	if _, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "live", AudioInFmt: AudioG711ULaw}, "Kore"); err == nil {
		t.Fatal("unsupported input format was accepted")
	}
	if _, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "live", AudioOutRate: 16000}, "Kore"); err == nil {
		t.Fatal("unsupported output rate was accepted")
	}
}

func TestGoogleRealtimeTranslatesAudioTranscriptsInterruptionAndUsage(t *testing.T) {
	session := newGoogleRealtimeTestSession()
	pcm := []byte{1, 2, 3, 4}
	session.translate(&googleLiveServerMessage{ServerContent: &googleLiveServerContent{Interrupted: true}})
	if event := <-session.events; event.Type != RealtimeEventSpeechStarted {
		t.Fatalf("interruption = %#v", event)
	}

	turn := &googleLiveServerContent{TurnComplete: true,
		InputTranscription:  &googleLiveTranscription{Text: "bonjour"},
		OutputTranscription: &googleLiveTranscription{Text: "salut"},
	}
	turn.ModelTurn = &struct {
		Parts []googleLivePart `json:"parts"`
	}{Parts: []googleLivePart{{InlineData: &googleLiveInlineData{
		MimeType: "audio/pcm;rate=24000", Data: base64.StdEncoding.EncodeToString(pcm),
	}}}}
	session.translate(&googleLiveServerMessage{
		ServerContent: turn,
		UsageMetadata: &googleLiveUsage{
			PromptTokenCount: 13, ResponseTokenCount: 7, ThoughtsTokenCount: 1, TotalTokenCount: 21,
			PromptTokensDetails:   []googleLiveTokenDetail{{Modality: "TEXT", TokenCount: 3}, {Modality: "AUDIO", TokenCount: 10}},
			ResponseTokensDetails: []googleLiveTokenDetail{{Modality: "TEXT", TokenCount: 2}, {Modality: "AUDIO", TokenCount: 5}},
		},
	})
	events := make([]RealtimeEvent, 0, 6)
	for range 6 {
		events = append(events, <-session.events)
	}
	if events[0].Type != RealtimeEventAudioOut || string(events[0].Audio) != string(pcm) {
		t.Fatalf("audio = %#v", events[0])
	}
	if events[3].Type != RealtimeEventTranscriptInput || !events[3].Final || events[3].Transcript != "bonjour" {
		t.Fatalf("input final = %#v", events[3])
	}
	if events[4].Type != RealtimeEventTranscriptOutput || !events[4].Final || events[4].Transcript != "salut" {
		t.Fatalf("output final = %#v", events[4])
	}
	done := events[5]
	if done.Type != RealtimeEventResponseDone || done.Usage.TextInputTokens != 3 || done.Usage.AudioInputTokens != 10 || done.Usage.TextOutputTokens != 3 || done.Usage.AudioOutputTokens != 5 {
		t.Fatalf("done = %#v", done)
	}
}

func TestGoogleRealtimeBatchesToolResultsUntilContinuation(t *testing.T) {
	session := newGoogleRealtimeTestSession()
	session.translate(&googleLiveServerMessage{ToolCall: &struct {
		FunctionCalls []googleLiveFunctionCall `json:"functionCalls"`
	}{FunctionCalls: []googleLiveFunctionCall{
		{ID: "fc-1", Name: "first", Args: map[string]any{"n": 1}},
		{ID: "fc-2", Name: "second", Args: map[string]any{"n": 2}},
	}}})
	first, second, done := <-session.events, <-session.events, <-session.events
	if first.Type != RealtimeEventToolCall || second.Type != RealtimeEventToolCall || done.Type != RealtimeEventResponseDone || first.ResponseID != done.ResponseID || second.ResponseID != done.ResponseID {
		t.Fatalf("tool events = %#v %#v %#v", first, second, done)
	}
	if err := session.SendToolResult("fc-1", `{"ok":true}`, false); err != nil {
		t.Fatal(err)
	}
	if err := session.SendToolResult("fc-2", "failed", true); err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-session.outbox:
		t.Fatalf("tool response flushed early: %s", frame.data)
	default:
	}
	if err := session.RequestResponse(); err != nil {
		t.Fatal(err)
	}
	frame := <-session.outbox
	var payload map[string]any
	if err := json.Unmarshal(frame.data, &payload); err != nil {
		t.Fatal(err)
	}
	responses := payload["toolResponse"].(map[string]any)["functionResponses"].([]any)
	if len(responses) != 2 || responses[0].(map[string]any)["name"] != "first" || responses[1].(map[string]any)["name"] != "second" {
		t.Fatalf("batched response = %#v", responses)
	}
	firstResult := responses[0].(map[string]any)["response"].(map[string]any)["result"].(map[string]any)
	if firstResult["ok"] != true {
		t.Fatalf("structured result = %#v", firstResult)
	}
	if responses[1].(map[string]any)["response"].(map[string]any)["error"] != "failed" {
		t.Fatalf("error response = %#v", responses[1])
	}
}

func TestGoogleRealtimeRestoresHistoryAndQueuesRealtimeInput(t *testing.T) {
	session := newGoogleRealtimeTestSession()
	if err := session.RestoreConversation([]Message{
		{Role: "user", Content: "bonjour"},
		{Role: "assistant", Content: "salut"},
		{Role: "system", Content: "ignored"},
	}); err != nil {
		t.Fatal(err)
	}
	var history map[string]any
	if err := json.Unmarshal((<-session.outbox).data, &history); err != nil {
		t.Fatal(err)
	}
	content := history["clientContent"].(map[string]any)
	turns := content["turns"].([]any)
	if content["turnComplete"] != true || len(turns) != 2 || turns[1].(map[string]any)["role"] != "model" {
		t.Fatalf("history = %#v", content)
	}
	if err := session.SendAudio([]byte{1, 2}); err != nil {
		t.Fatal(err)
	}
	var audio map[string]any
	if err := json.Unmarshal((<-session.outbox).data, &audio); err != nil {
		t.Fatal(err)
	}
	blob := audio["realtimeInput"].(map[string]any)["audio"].(map[string]any)
	if blob["mimeType"] != "audio/pcm;rate=24000" {
		t.Fatalf("audio payload = %#v", blob)
	}
}

func TestGoogleRealtimeConfigurationChangeRequestsRenewalAfterTurn(t *testing.T) {
	session := newGoogleRealtimeTestSession()
	tools := []NativeTool{{Name: "one", Parameters: map[string]any{"type": "object"}}}
	session.configFingerprint = googleRealtimeConfigFingerprint("old", tools)
	if err := session.UpdateConfiguration("old", tools); err != nil {
		t.Fatal(err)
	}
	if session.restartAfterTurn {
		t.Fatal("identical configuration requested a restart")
	}
	if err := session.UpdateConfiguration("new", tools); err != nil {
		t.Fatal(err)
	}
	if !session.restartAfterTurn {
		t.Fatal("changed immutable Gemini setup did not request renewal")
	}
}

func TestGoogleRealtimeProviderRegistrationOverridesAndPricing(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "google-test")
	pool, err := buildProviderPool(&Config{
		RealtimeEnabled: true,
		Providers: []ProviderConfig{
			{Name: "google"},
			{Name: "google-realtime", Default: true,
				Models:        map[string]string{"large": "gemini-live-pinned", "small": "gemini-live-small"},
				RealtimeVoice: "Aoede"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := pool.RealtimeDefault().(*GoogleRealtimeProvider)
	if !ok {
		t.Fatalf("provider = %T", pool.RealtimeDefault())
	}
	if provider.Models()[ModelLarge] != "gemini-live-pinned" || provider.Models()[ModelSmall] != "gemini-live-small" || provider.DefaultVoice() != "Aoede" {
		t.Fatalf("models=%#v voice=%q", provider.Models(), provider.DefaultVoice())
	}
	cost := calculateCostForRealtimeProvider(provider, "gemini-live-pinned", RealtimeUsage{
		TextInputTokens: 1_000_000, TextOutputTokens: 1_000_000,
		AudioInputTokens: 1_000_000, AudioOutputTokens: 1_000_000,
	})
	if want := 20.25; math.Abs(cost-want) > 1e-12 {
		t.Fatalf("cost=%v want=%v", cost, want)
	}
}

func TestAppendGoogleTranscriptHandlesDeltasAndCumulativeUpdates(t *testing.T) {
	got := appendGoogleTranscript("Bon", "jour")
	got = appendGoogleTranscript(got, "Bonjour à")
	got = appendGoogleTranscript(got, " à bientôt")
	if got != "Bonjour à bientôt" {
		t.Fatalf("transcript = %q", got)
	}
}

func TestGoogleRealtimeOpenWaitsForSetupAndUnlocksHistory(t *testing.T) {
	serverErr := make(chan error, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("key") != "secret" {
			serverErr <- fmt.Errorf("key query = %q", r.URL.Query().Get("key"))
			return
		}
		conn, _, _, err := ws.UpgradeHTTP(r, w)
		if err != nil {
			serverErr <- err
			return
		}
		defer conn.Close()
		readObject := func() (map[string]any, error) {
			data, op, err := wsutil.ReadClientData(conn)
			if err != nil {
				return nil, err
			}
			if op != ws.OpText {
				return nil, fmt.Errorf("opcode = %v", op)
			}
			var object map[string]any
			if err := json.Unmarshal(data, &object); err != nil {
				return nil, err
			}
			return object, nil
		}
		setup, err := readObject()
		if err != nil || setup["setup"] == nil {
			serverErr <- fmt.Errorf("setup: %v %#v", err, setup)
			return
		}
		if err := wsutil.WriteServerMessage(conn, ws.OpBinary, []byte(`{"setupComplete":{}}`)); err != nil {
			serverErr <- err
			return
		}
		history, err := readObject()
		if err != nil || history["clientContent"].(map[string]any)["turnComplete"] != true {
			serverErr <- fmt.Errorf("history unlock: %v %#v", err, history)
			return
		}
		input, err := readObject()
		if err != nil || input["realtimeInput"].(map[string]any)["text"] != "bonjour" {
			serverErr <- fmt.Errorf("realtime input: %v %#v", err, input)
			return
		}
		response := `{"serverContent":{"outputTranscription":{"text":"salut"},"turnComplete":true},"usageMetadata":{"promptTokenCount":1,"responseTokenCount":1,"totalTokenCount":2,"promptTokensDetails":[{"modality":"TEXT","tokenCount":1}],"responseTokensDetails":[{"modality":"TEXT","tokenCount":1}]}}`
		if err := wsutil.WriteServerMessage(conn, ws.OpBinary, []byte(response)); err != nil {
			serverErr <- err
			return
		}
		serverErr <- nil
		<-release
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	provider := NewGoogleRealtimeProvider("secret")
	provider.endpoint = "ws" + strings.TrimPrefix(server.URL, "http")
	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model: "gemini-3.1-flash-live-preview", Voice: "Kore",
		AudioInFmt: AudioPCM16, AudioOutFmt: AudioPCM16,
		AudioInRate: 24000, AudioOutRate: 24000,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	defer close(release)
	if err := session.RestoreConversation(nil); err != nil {
		t.Fatal(err)
	}
	if err := session.SendText("user", "bonjour"); err != nil {
		t.Fatal(err)
	}
	for {
		select {
		case err := <-serverErr:
			if err != nil {
				t.Fatal(err)
			}
			serverErr = nil
		case event := <-session.Events():
			if event.Type == RealtimeEventError {
				t.Fatal(event.Err)
			}
			if event.Type == RealtimeEventTranscriptOutput && event.Final {
				if event.Transcript != "salut" {
					t.Fatalf("transcript = %q", event.Transcript)
				}
				return
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
}
