package core

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"
)

func speechTestContent(audio []byte, text string, done bool) *googleLiveServerMessage {
	content := &googleLiveServerContent{TurnComplete: done}
	if audio != nil {
		content.ModelTurn = &struct {
			Parts []googleLivePart `json:"parts"`
		}{Parts: []googleLivePart{{InlineData: &googleLiveInlineData{MimeType: "audio/pcm;rate=24000", Data: base64.StdEncoding.EncodeToString(audio)}}}}
	}
	if text != "" {
		content.OutputTranscription = &googleLiveTranscription{Text: text}
	}
	return &googleLiveServerMessage{ServerContent: content}
}

func TestSpeechGuardRecordedNarrationBlocksAudioBeforeTranscript(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	s.translate(speechTestContent([]byte{1, 2, 3, 4}, "", false))
	if len(s.events) != 0 {
		t.Fatal("untranscribed audio escaped")
	}
	s.translate(speechTestContent(nil, "Private reason", false))
	s.translate(speechTestContent([]byte{5, 6}, "ing processed the tool call event. I should deliver this result to the caller.", true))
	blocked := 0
	for len(s.events) > 0 {
		e := <-s.events
		if e.Type == RealtimeEventAudioOut {
			t.Fatal("internal narration audio escaped")
		}
		if e.Type == RealtimeEventOutputBlocked {
			blocked++
		}
		if e.Type == RealtimeEventTranscriptOutput && e.Final {
			t.Fatal("rejected narration entered final transcript")
		}
	}
	if blocked != 1 {
		t.Fatalf("blocked events=%d", blocked)
	}
	// A fresh response is allowed and is not contaminated by the previous gate.
	s.translate(speechTestContent([]byte{7, 8}, "Your booking is confirmed.", true))
	heard := false
	for len(s.events) > 0 {
		e := <-s.events
		heard = heard || e.Type == RealtimeEventAudioOut
	}
	if !heard {
		t.Fatal("clean response remained muted")
	}
}

func TestSpeechGuardMissingTranscriptIsBoundedAndFailsClosed(t *testing.T) {
	for _, end := range []bool{false, true} {
		s := newGoogleRealtimeTestSession()
		size := googleSpeechPrefixMaxBytes + 2
		if end {
			size = 2
		}
		s.translate(speechTestContent(make([]byte, size), "", end))
		blocked := false
		for len(s.events) > 0 {
			e := <-s.events
			if e.Type == RealtimeEventAudioOut {
				t.Fatal("unverified audio escaped")
			}
			blocked = blocked || e.Type == RealtimeEventOutputBlocked
		}
		if !blocked || s.speech.bytes != 0 || len(s.speech.pending) != 0 {
			t.Fatal("missing transcript did not drop bounded buffer")
		}
	}
}

func TestSpeechGuardCleanSpeechStreamsAndLaterLeakStops(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	s.translate(speechTestContent([]byte{1, 2}, "I checked your appointment and it is confirmed", false))
	heard := false
	for len(s.events) > 0 {
		event := <-s.events
		heard = heard || event.Type == RealtimeEventAudioOut
	}
	if !heard {
		t.Fatal("clean response waited for turn completion")
	}
	s.translate(speechTestContent([]byte{3, 4}, ". I will now speak the marker for the user.", true))
	for len(s.events) > 0 {
		if (<-s.events).Type == RealtimeEventAudioOut {
			t.Fatal("later leaked segment escaped")
		}
	}
}

func TestSpeechGuardIgnoresFlaggedThoughtParts(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	message := speechTestContent([]byte{1, 2}, "", true)
	message.ServerContent.ModelTurn.Parts[0].Thought = true
	s.translate(message)
	for len(s.events) > 0 {
		e := <-s.events
		if e.Type == RealtimeEventAudioOut || e.Type == RealtimeEventTranscriptOutput || e.Type == RealtimeEventOutputBlocked {
			t.Fatalf("thought part escaped: %v", e.Type)
		}
	}
}

func TestSpeechGuardNormalConversationIsNotInternalNarration(t *testing.T) {
	for _, text := range []string{"I think Tuesday works.", "Let me check your booking.", "Your instructions say to call after five.", "Private reasoning means working something out before answering.", "La réservation est confirmée.", "La reserva está confirmada."} {
		if detectRealtimeInternalNarration(text) != "" {
			t.Errorf("normal speech blocked: %s", text)
		}
	}
}

func TestSpeechGuardCoreRecoveryIsBoundedAndPreservesToolResults(t *testing.T) {
	thinker := newTestThinker()
	defer thinker.Stop()
	rt := newRealtimeThinker(context.Background(), thinker, &fakeRealtimeProvider{}, "", nil, make(chan RealtimeAudioFrame, 4), make(chan string, 4))
	defer rt.cancel()
	session := newFakeRealtimeSession()
	rt.replaceSession(session)
	thinker.messages = append(thinker.messages, Message{Role: "user", Content: "Verified booking reference: BOOK-73"})
	for _, id := range []string{"one", "one", "two"} {
		rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventOutputBlocked, ResponseID: id, ItemID: id, OutputBlockReason: "internal_instruction_narration"})
	}
	if len(session.texts) != 1 {
		t.Fatalf("recovery prompts=%d, want one", len(session.texts))
	}
	if !strings.Contains(session.texts[0].Content, "Do not repeat completed actions") {
		t.Fatal("recovery invites action replay")
	}
	if thinker.messages[len(thinker.messages)-1].Content != "Verified booking reference: BOOK-73" {
		t.Fatal("recovery lost verified result")
	}
}

func TestRealtimePromptDoesNotIncludeWorkerThoughtAndTimerInstructions(t *testing.T) {
	for _, leader := range []bool{false, true} {
		prompt := formatThreadBasePrompt(leader, true, "voice", "main")
		for _, forbidden := range []string{"Keep each thought", "Thought 1:", "[WAKE STATE]", "continuous thinking engine", reasoningBaselineContract} {
			if strings.Contains(prompt, forbidden) {
				t.Errorf("voice prompt contains worker guidance %q", forbidden)
			}
		}
		if !strings.Contains(prompt, toolArgumentPresenceContract) {
			t.Fatal("voice prompt lost tool argument contract")
		}
	}
}
