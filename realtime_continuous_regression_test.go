package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"testing"
	"time"
)

func TestContinuousMonitorStreamsFrenchWithoutCaptionThreshold(t *testing.T) {
	for _, first := range []bool{true, false} {
		t.Run(map[bool]string{true: "captions_first", false: "audio_first"}[first], func(t *testing.T) {
			g := realtimeContinuousGuard{}
			now := time.Now()
			text := RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now, Transcript: "Oui", TranscriptStartMS: 200, TranscriptEndMS: 350}
			if first {
				g.transcript(text, nil, nil)
			}
			pcm := []byte{255, 127, 2, 0, 0, 0}
			got := g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now, Audio: pcm}, nil)
			if len(got) != 1 || !bytes.Equal(got[0].Audio, pcm) {
				t.Fatalf("audio waited for captions: %+v", got)
			}
			if !first {
				g.transcript(text, nil, nil)
			}
			if extra := g.finish(); len(extra) != 0 {
				t.Fatalf("short French reply rejected: %+v", extra)
			}
		})
	}
}

func TestContinuousUncaptionedAudioAndSilenceStayOrdered(t *testing.T) {
	g := realtimeContinuousGuard{}
	now := time.Now()
	chunks := [][]byte{{255, 127}, make([]byte, 14400), {1, 2}}
	var got, want []byte
	for i, pcm := range chunks {
		want = append(want, pcm...)
		for _, e := range g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now.Add(time.Duration(i) * 20 * time.Millisecond), Audio: pcm}, nil) {
			if e.Type == RealtimeEventOutputBlocked {
				t.Fatal("silence/gap became unsafe speech")
			}
			got = append(got, e.Audio...)
		}
	}
	if !bytes.Equal(got, want) {
		t.Fatal("PCM reordered or silence removed")
	}
	events := g.finish()
	if len(events) != 1 || events[0].Type != RealtimeEventTranscriptIncomplete {
		t.Fatalf("missing captions misclassified: %+v", events)
	}
	if len(g.finish()) != 0 {
		t.Fatal("duplicate incomplete-caption incident")
	}
}

func TestContinuousMonitorRetainsSyntaxAcrossAudioGaps(t *testing.T) {
	g := realtimeContinuousGuard{}
	now := time.Now()
	g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now, Audio: []byte{255, 127}}, nil)
	g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now, Transcript: "call"}, nil, nil)
	g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now.Add(2 * time.Second), Audio: []byte{255, 127}}, nil)
	got := g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now.Add(3 * time.Second), Transcript: "to:secret{}"}, nil, nil)
	if len(got) != 1 || got[0].Type != RealtimeEventOutputBlocked {
		t.Fatalf("fragmented tool syntax escaped: %+v", got)
	}
}

func TestContinuousStrictCancellationIgnoresLateApproval(t *testing.T) {
	g := realtimeContinuousGuard{config: RealtimeSpeechGuardConfig{Mode: "transcript_prefix"}}
	now := time.Now()
	audio := RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: now, Audio: []byte{255, 127}}
	g.audio(audio, nil)
	cancelled := g.id()
	g.discard(cancelled)
	for _, e := range g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: now, Transcript: "Bonjour."}, nil, nil) {
		if e.Type == RealtimeEventAudioOut {
			t.Fatal("cancelled audio approved late")
		}
	}
	audio.ObservedAt = now.Add(2 * time.Second)
	g.audio(audio, nil)
	events := g.transcript(RealtimeEvent{Type: RealtimeEventTranscriptOutput, ObservedAt: audio.ObservedAt, Transcript: "Reprenons."}, nil, nil)
	if len(events) != 2 || events[1].ItemID == cancelled {
		t.Fatalf("new speech failed: %+v", events)
	}
}

func TestContinuousMissingCaptionDoesNotInterruptEarlierPlayback(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "playing", Audio: make([]byte, 4800)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventOutputBlocked, PlaybackID: "buffered-next", OutputBlockReason: "missing_output_transcript"})
	if len(rt.audioOut) != 1 {
		t.Fatal("missing caption cleared unrelated playback")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.interrupts != 0 || len(s.texts) != 0 {
		t.Fatal("missing caption requested leak recovery")
	}
}

func TestContinuousLocalInterruptionClearsPlaybackAndKeepsBackend(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, ProviderInterruption: true}
	control := make(chan string, 4)
	rt.audioControl = control
	backendCtx, cancel := context.WithCancel(rt.ctx)
	defer cancel()
	rt.backend.active = true
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "old", Audio: make([]byte, 4800)})
	inFlight := <-rt.audioOut
	rt.rendererSpeechStarted()
	if inFlight.cancelled == nil || !inFlight.cancelled() {
		t.Fatal("in-flight frame remained playable")
	}
	inFlight.written(false)
	var message map[string]any
	if err := json.Unmarshal([]byte(<-control), &message); err != nil || message["item_id"] != "old" {
		t.Fatalf("unscoped clear: %+v %v", message, err)
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "new", Audio: make([]byte, 480)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "old", Audio: make([]byte, 480)})
	if len(rt.audioOut) != 1 || (<-rt.audioOut).ItemID != "new" {
		t.Fatal("late cancelled frame escaped or new audio blocked")
	}
	select {
	case <-backendCtx.Done():
		t.Fatal("speech interruption cancelled backend")
	default:
	}
	if !rt.backend.active {
		t.Fatal("speech interruption reset backend")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.interrupts != 0 || len(s.truncations) != 0 {
		t.Fatal("provider yield confused with local playback clear")
	}
}

func TestContinuousSpeakingWaitsForDeliveryAndPlayback(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	rt.delivery.playbackReady()
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "speech", ObservedAt: time.Now().Add(-3 * time.Second), Audio: make([]byte, 4800)})
	rt.observeContinuousIdle()
	if rt.state != "speaking" {
		t.Fatal("old provider timestamp made freshly released audio idle")
	}
	rt.continuous.lastRelease = time.Now().Add(-time.Second)
	rt.observeContinuousIdle()
	if rt.state != "speaking" {
		t.Fatal("queued audio became idle")
	}
	frame := <-rt.audioOut
	frame.written(true)
	rt.setConversationState("working", RealtimeEvent{})
	rt.observeContinuousIdle()
	if rt.state != "speaking" {
		t.Fatal("bridge write treated as playback completion")
	}
	rt.acknowledgePlayback("speech", 100)
	rt.observeContinuousIdle()
	if rt.state != "listening" {
		t.Fatalf("drained playback stayed %q", rt.state)
	}
}

func TestContinuousCaptionsHaveIndependentTimelineAndGrouping(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	for _, e := range []RealtimeEvent{
		{Type: RealtimeEventTranscriptOutput, PlaybackID: "audio-1", Transcript: "Bon", TranscriptDelta: true, TranscriptStartMS: 100, TranscriptEndMS: 200},
		{Type: RealtimeEventTranscriptOutput, PlaybackID: "audio-2", Transcript: "jour.", TranscriptDelta: true, TranscriptStartMS: 200, TranscriptEndMS: 400},
	} {
		rt.handleSessionEvent(e)
	}
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 1 || entries[0].Content != "Bonjour." || entries[0].RealtimeTranscript.StartMS != 100 || entries[0].RealtimeTranscript.EndMS != 400 {
		t.Fatalf("audio boundary discarded transcript: %+v", entries)
	}
}

func TestBridgeNeverWritesCancelledPlayback(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()
	defer client.Close()
	socket := realtimeBridgeSocket{conn: server}
	written, err := socket.writePlaybackAudio([]byte(`{"type":"audio.frame"}`), []byte{1, 2}, func() bool { return true })
	if err != nil || written {
		t.Fatalf("cancelled frame reached socket: %v %v", written, err)
	}
}

func TestSpeechGuardDefaultsAreCapabilitySpecific(t *testing.T) {
	config := RealtimeSpeechGuardConfig{}
	if config.resolved().Mode != "transcript_prefix" || config.continuousPolicy().Mode != "monitor" {
		t.Fatal("changed legacy defaults or buffered continuous default")
	}
	explicit := RealtimeSpeechGuardConfig{Mode: "transcript_prefix"}
	if explicit.continuousPolicy().Mode != "transcript_prefix" {
		t.Fatal("strict policy ignored")
	}
}

func TestContinuousStrictMissingCaptionsAreNotALeak(t *testing.T) {
	g := realtimeContinuousGuard{config: RealtimeSpeechGuardConfig{Mode: "transcript_prefix", MaxBufferedAudioMS: 20}}
	got := g.audio(RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: time.Now(), Audio: make([]byte, 1920)}, nil)
	if len(got) != 1 || got[0].Type != RealtimeEventOutputBlocked || got[0].OutputBlockReason != "missing_output_transcript" {
		t.Fatalf("strict fallback: %+v", got)
	}
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	rt.handleSessionEvent(got[0])
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.interrupts != 0 || len(s.texts) != 0 {
		t.Fatal("missing captions requested provider recovery")
	}
}

func TestContinuousConfirmedSuppressionRecoversPerIncident(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true}
	for _, id := range []string{"one", "two"} {
		event := RealtimeEvent{Type: RealtimeEventOutputBlocked, PlaybackID: id, OutputBlockReason: "internal_instruction_narration"}
		rt.handleSessionEvent(event)
		rt.handleSessionEvent(event)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.texts) != 2 {
		t.Fatalf("recovery attempts=%d; want one per incident", len(s.texts))
	}
}

func TestContinuousSilencePreservesPCMWithoutKeepingSpeakingActive(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	rt.delivery.playbackReady()
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "speech", Audio: make([]byte, 4800)})
	frame := <-rt.audioOut
	frame.written(true)
	rt.acknowledgePlayback("speech", 100)
	rt.continuous.lastRelease = time.Now().Add(-time.Second)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "speech", AudioSilence: true, Audio: make([]byte, 4800)})
	rt.observeContinuousIdle()
	if rt.state != "listening" {
		t.Fatalf("silence kept state %q", rt.state)
	}
	if len(rt.audioOut) != 1 {
		t.Fatal("silence was removed from PCM timeline")
	}
	silence := <-rt.audioOut
	if !silence.Silent || silence.AudioEndMS != 200 {
		t.Fatalf("silence lost timing: %+v", silence)
	}
	silence.written(true)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "reply", Audio: make([]byte, 480)})
	if rt.state != "speaking" {
		t.Fatal("next reply did not resume speaking")
	}
}

func TestContinuousLateCheckCancelsOnlyItsOlderQueuedPlayback(t *testing.T) {
	rt, s := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, ProviderInterruption: true}
	control := make(chan string, 4)
	rt.audioControl = control
	rt.delivery.playbackReady()
	for _, id := range []string{"older", "newer"} {
		rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: id, Audio: make([]byte, 4800)})
	}
	if !rt.interruptPlayback("confirmed_content_suppression", "older", true, true) {
		t.Fatal("older queued playback was not cancelled")
	}
	// Kept playback is parked for the writer, behind a wake marker.
	if next := orderedAudioReader(rt.audioOut)(); next.ItemID != "newer" || len(rt.audioOut) != 0 {
		t.Fatal("late check removed newer playback")
	}
	if rt.outputItemID != "newer" {
		t.Fatal("late check changed current output identity")
	}
	var message map[string]any
	if json.Unmarshal([]byte(<-control), &message) != nil || message["item_id"] != "older" {
		t.Fatalf("wrong clear target: %+v", message)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.interrupts != 0 {
		t.Fatal("late check cancelled the provider's newer speech")
	}
}

func TestContinuousCallerInterruptionClearsExistingPlaybackSnapshot(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, ProviderInterruption: true}
	rt.audioControl = make(chan string, 4)
	rt.delivery.playbackReady()
	for _, id := range []string{"first", "second"} {
		rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: id, Audio: make([]byte, 4800)})
	}
	rt.rendererSpeechStarted()
	if len(rt.audioOut) != 0 {
		t.Fatal("older queued audio survived caller interruption")
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "next", Audio: make([]byte, 480)})
	if len(rt.audioOut) != 1 {
		t.Fatal("caller interruption blocked later speech")
	}
}

// Captured French session: a comma-ended recipe was followed by a caller
// interruption and an acknowledgment. The acknowledgment must be a new turn.
func TestContinuousInterruptedRecipeHasSeparateAcknowledgment(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	for _, e := range []RealtimeEvent{
		{Type: RealtimeEventTranscriptOutput, Transcript: "Tu mélanges de la farine, des œufs, du lait,", TranscriptDelta: true, TranscriptStartMS: 30600, TranscriptEndMS: 33200},
		{Type: RealtimeEventTranscriptInput, Transcript: "Non attends", TranscriptDelta: true, TranscriptStartMS: 33800, TranscriptEndMS: 34000},
		{Type: RealtimeEventTranscriptOutput, Transcript: "Oui,", TranscriptDelta: true, TranscriptStartMS: 34400, TranscriptEndMS: 34600},
		{Type: RealtimeEventTranscriptInput, Transcript: ", attends, arrête", TranscriptDelta: true, TranscriptStartMS: 34000, TranscriptEndMS: 34600},
		{Type: RealtimeEventTranscriptOutput, Transcript: " je t'écoute.", TranscriptDelta: true, TranscriptStartMS: 34600, TranscriptEndMS: 35000},
	} {
		rt.handleSessionEvent(e)
	}
	rt.flushPendingTranscripts("test")
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	if len(entries) != 3 || entries[0].Content != "Tu mélanges de la farine, des œufs, du lait," || entries[1].Content != "Non attends, attends, arrête" || entries[2].Content != "Oui, je t'écoute." {
		t.Fatalf("interruption mixed speakers: %+v", entries)
	}
	if len(entries[2].RealtimeTranscript.Fragments) != 2 {
		t.Fatal("fragment timeline lost")
	}
}

func TestContinuousDelayedCallerCaptionRetainsFragmentEvidence(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true, TranscriptFragments: true}
	for _, e := range []RealtimeEvent{
		{Type: RealtimeEventTranscriptOutput, Transcript: "Farine, lait,", TranscriptDelta: true, TranscriptStartMS: 30600, TranscriptEndMS: 33200},
		{Type: RealtimeEventTranscriptOutput, Transcript: " Oui, je t'écoute.", TranscriptDelta: true, TranscriptStartMS: 34400, TranscriptEndMS: 35000},
		{Type: RealtimeEventTranscriptInput, Transcript: "Non attends, arrête", TranscriptDelta: true, TranscriptStartMS: 33800, TranscriptEndMS: 34600},
	} {
		rt.handleSessionEvent(e)
	}
	rt.flushPendingTranscripts("test")
	entries := readSessionEntriesForTest(t, rt.Thinker.session.path)
	fragments := entries[0].RealtimeTranscript.Fragments
	if len(fragments) != 2 || fragments[0].EndMS != 33200 || fragments[1].StartMS != 34400 {
		t.Fatalf("cannot order delayed captions: %+v", fragments)
	}
}
