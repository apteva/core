package core

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// TestAudioInput tests sending audio to an LLM provider and getting a response about the content.
// Set RUN_AUDIO_TEST=1 and provide a provider API key + audio URL.
//
// Usage:
//
//	GOOGLE_API_KEY=... AUDIO_URL=https://example.com/audio.mp3 RUN_AUDIO_TEST=1 go test -v -run TestAudioInput -timeout 60s
//	ANTHROPIC_API_KEY=... AUDIO_URL=https://example.com/audio.mp3 RUN_AUDIO_TEST=1 go test -v -run TestAudioInput -timeout 60s
//	FIREWORKS_API_KEY=... AUDIO_URL=https://example.com/audio.mp3 RUN_AUDIO_TEST=1 go test -v -run TestAudioInput -timeout 60s
func TestAudioInput(t *testing.T) {
	if os.Getenv("RUN_AUDIO_TEST") == "" {
		t.Skip("set RUN_AUDIO_TEST=1 to run")
	}

	audioURL := os.Getenv("AUDIO_URL")
	if audioURL == "" {
		t.Fatal("AUDIO_URL required (public URL to an audio file)")
	}

	// Pick provider based on which key is set
	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider available: %v", err)
	}
	t.Logf("provider: %s", provider.Name())
	models := provider.Models()
	model := models[ModelSmall]
	t.Logf("model: %s", model)

	// Build messages with audio attachment
	audioPart := ContentPart{
		Type:     "audio_url",
		AudioURL: &AudioURL{URL: audioURL},
	}
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant. When given audio, describe what you hear in 1-2 sentences."},
		{
			Role: "user",
			Parts: []ContentPart{
				{Type: "text", Text: "What is in this audio? Describe it briefly."},
				audioPart,
			},
		},
	}

	t.Logf("sending audio: %s", audioURL)
	start := time.Now()

	var fullText strings.Builder
	resp, err := provider.Chat(context.Background(), messages, model, nil, func(chunk string) {
		fullText.WriteString(chunk)
	}, nil, nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	response := resp.Text
	if response == "" {
		response = fullText.String()
	}

	t.Logf("response (%dms, %d in/%d out tokens):", duration.Milliseconds(), resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	t.Logf("  %s", response)

	if len(response) < 10 {
		t.Error("response too short — model may not have processed the audio")
	}
}

func TestRealtimeAudioFailedUpgradeDoesNotConsumeToken(t *testing.T) {
	audioIn := make(chan []byte, 1)
	audioOut := make(chan []byte, 1)
	token := registerAudioBridge("voice-test", audioIn, audioOut, make(chan string, 1))
	defer unregisterAudioBridge("voice-test")

	req := httptest.NewRequest(http.MethodGet, "/realtime/audio?thread=voice-test&token="+token, nil)
	rec := httptest.NewRecorder()
	(&APIServer{}).realtimeAudioHandler(rec, req)
	if _, err := claimAudioBridge(token); err != nil {
		t.Fatalf("failed websocket upgrade consumed token: %v", err)
	}
}

func TestRealtimeAudioTokenRenewalInvalidatesPreviousCapability(t *testing.T) {
	audioIn := make(chan []byte, 1)
	audioOut := make(chan []byte, 1)
	control := make(chan string, 1)
	first := registerAudioBridge("voice-renew", audioIn, audioOut, control)
	defer unregisterAudioBridge("voice-renew")
	second := renewAudioBridge("voice-renew", audioIn, audioOut, control)
	if first == second {
		t.Fatal("renewal reused the previous audio capability")
	}
	if _, err := claimAudioBridge(first); err == nil {
		t.Fatal("previous audio capability remained valid after renewal")
	}
	reg, err := claimAudioBridge(second)
	if err != nil {
		t.Fatalf("renewed audio capability was rejected: %v", err)
	}
	if reg.threadID != "voice-renew" || reg.audioIn != audioIn || reg.audioOut != audioOut {
		t.Fatalf("renewed registration changed bridge identity: %#v", reg)
	}
}

// TestMediaInSend tests that parseMediaURLs correctly classifies audio and image URLs.
func TestMediaInSend(t *testing.T) {
	tests := []struct {
		urls     string
		wantType string
		wantLen  int
	}{
		{"https://example.com/audio.mp3", "audio_url", 1},
		{"https://example.com/song.wav", "audio_url", 1},
		{"https://example.com/photo.png", "image_url", 1},
		{"https://example.com/pic.jpg", "image_url", 1},
		{"https://example.com/a.mp3 https://example.com/b.png", "", 2},
		{"", "", 0},
	}

	for _, tt := range tests {
		parts := parseMediaURLs(tt.urls)
		if len(parts) != tt.wantLen {
			t.Errorf("parseMediaURLs(%q): got %d parts, want %d", tt.urls, len(parts), tt.wantLen)
			continue
		}
		if tt.wantLen == 1 && parts[0].Type != tt.wantType {
			t.Errorf("parseMediaURLs(%q): got type %s, want %s", tt.urls, parts[0].Type, tt.wantType)
		}
	}
}

// TestAudioViaParts tests that audio passed as ContentParts flows through
// the thinker and gets a response from the LLM about the audio content.
//
// Usage:
//
//	GOOGLE_API_KEY=... AUDIO_URL=https://example.com/audio.mp3 RUN_AUDIO_TEST=1 go test -v -run TestAudioViaParts -timeout 60s
func TestAudioViaParts(t *testing.T) {
	if os.Getenv("RUN_AUDIO_TEST") == "" {
		t.Skip("set RUN_AUDIO_TEST=1 to run")
	}

	audioURL := os.Getenv("AUDIO_URL")
	if audioURL == "" {
		t.Fatal("AUDIO_URL required")
	}

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	t.Logf("provider: %s", provider.Name())

	cfg := &Config{
		Directive: "You are an audio analyst. When you receive audio, describe what you hear in 1-2 sentences.",
		Mode:      ModeAutonomous,
	}

	thinker := NewThinker("", provider, cfg)
	thinker.session = nil                   // no session history — fresh start
	thinker.messages = thinker.messages[:1] // keep only system prompt
	thinker.registry = NewToolRegistry("")
	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(cfg.GetDirective(), ModeAutonomous, thinker.registry, toolDocs, thinker.mcpServers, nil, nil, nil)
	}

	// Parse the audio URL into media parts (same as send tool does)
	parts := parseMediaURLs(audioURL)
	if len(parts) == 0 {
		t.Fatal("parseMediaURLs returned no parts")
	}
	t.Logf("parsed %d media parts, type=%s", len(parts), parts[0].Type)

	go thinker.Run()
	time.Sleep(1 * time.Second)

	// Inject with media parts (same path as send with media)
	thinker.InjectWithParts("Describe this audio", parts)
	t.Logf("injected audio with parts")

	// Wait for a response
	time.Sleep(15 * time.Second)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)

	// Check assistant messages for audio description
	for _, m := range thinker.messages {
		if m.Role == "assistant" && len(m.Content) > 10 {
			lower := strings.ToLower(m.Content)
			if strings.Contains(lower, "audio") || strings.Contains(lower, "voice") ||
				strings.Contains(lower, "speak") || strings.Contains(lower, "pitch") ||
				strings.Contains(lower, "apteva") || strings.Contains(lower, "framework") {
				t.Logf("PASS: got audio description: %s", m.Content[:min(200, len(m.Content))])
				return
			}
		}
	}

	// Dump all messages for debugging
	t.Logf("messages: %d", len(thinker.messages))
	for i, m := range thinker.messages {
		content := m.Content
		if len(content) > 100 {
			content = content[:100] + "..."
		}
		t.Logf("  msg[%d] role=%s parts=%d content=%s", i, m.Role, len(m.Parts), content)
	}
	t.Error("FAIL: no audio analysis found in responses")
}

func min2(a, b int) int {
	if a < b {
		return a
	}
	return b
}
