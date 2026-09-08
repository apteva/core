package core

import "strings"

// Match self-directed instruction narration, not ordinary conversational phrases
// such as "I think", "let me check", or a caller discussing their instructions.
func detectRealtimeInternalNarration(text string) string {
	text = strings.ToLower(strings.Join(strings.Fields(text), " "))
	for _, phrase := range []string{
		"private reasoning processed", "private reasoning:", "internal reasoning:", "hidden reasoning:", "chain-of-thought:",
		"i should deliver this", "i should respond", "i need to respond", "i will now speak",
		"i must never call", "i don't need to use pace", "i do not need to use pace",
		"natural silence is allowed", "keep the live conversation open",
		"my system instructions", "my system prompt", "according to my instructions",
		"process the tool call event", "processed the tool call event",
	} {
		if strings.Contains(text, phrase) {
			return "internal_instruction_narration"
		}
	}
	return ""
}

// Buffer the start of a Google speech response until a short transcript prefix
// can be checked. This handles audio arriving before its transcript without
// buffering an entire clean response. Missing transcription fails closed at a
// bounded five seconds of PCM. Subsequent detected leaks stop future audio.
type googleSpeechGate struct {
	pending            []RealtimeEvent
	bytes              int
	approved, rejected bool
}

const googleSpeechPrefixMaxBytes = realtimePCMBytesPerSecond * 5

func (s *googleRealtimeSession) queueSpeechAudio(event RealtimeEvent) {
	if s.speech.rejected {
		return
	}
	if s.speech.approved {
		s.emitAudio(event)
		return
	}
	if s.speech.bytes+len(event.Audio) > googleSpeechPrefixMaxBytes {
		s.rejectSpeech(event.ResponseID, "missing_output_transcript")
		return
	}
	s.speech.pending = append(s.speech.pending, event)
	s.speech.bytes += len(event.Audio)
}

func (s *googleRealtimeSession) releaseSpeechAudio() {
	if s.speech.rejected {
		return
	}
	s.speech.approved = true
	for _, event := range s.speech.pending {
		s.emitAudio(event)
	}
	s.speech.pending = nil
	s.speech.bytes = 0
}

func (s *googleRealtimeSession) rejectSpeech(responseID, reason string) {
	if s.speech.rejected {
		return
	}
	s.speech = googleSpeechGate{rejected: true}
	s.emitControl(RealtimeEvent{Type: RealtimeEventOutputBlocked, ResponseID: responseID, ItemID: responseID, OutputBlockReason: reason})
}
