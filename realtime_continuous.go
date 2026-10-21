package core

import (
	"encoding/binary"
	"fmt"
	"strings"
	"time"
)

const realtimeObservedQuiet = 700 * time.Millisecond

// Continuous providers expose fragments rather than final turns. Grouping is
// an application observation, never a fabricated provider completion event.
type realtimeContinuousState struct {
	lastAudio time.Time
	segments  map[string]*realtimeContinuousTranscript
	ordinal   int
}

type realtimeContinuousTranscript struct {
	id                string
	start, end, bytes int
	last              time.Time
	closed            bool
}

func (rt *RealtimeThinker) normalizeContinuousEvent(e RealtimeEvent) RealtimeEvent {
	if !rt.capabilities.TranscriptFragments {
		return e
	}
	if e.ObservedAt.IsZero() {
		e.ObservedAt = time.Now()
	}
	if e.Type == RealtimeEventAudioOut {
		rt.continuous.lastAudio = e.ObservedAt
		if e.ItemID == "" {
			e.ItemID = fmt.Sprintf("stream-%d", rt.sessionGeneration)
		}
		return e
	}
	if e.Type != RealtimeEventTranscriptInput && e.Type != RealtimeEventTranscriptOutput {
		return e
	}
	role := string(e.Type)
	if e.Type == RealtimeEventTranscriptOutput && e.ItemID != "" {
		e.TranscriptGrouped = true
		e.Final = false
		return e
	}
	if rt.continuous.segments == nil {
		rt.continuous.segments = map[string]*realtimeContinuousTranscript{}
	}
	s := rt.continuous.segments[role]
	if s == nil || s.closed || s.bytes >= 2048 || e.TranscriptStartMS > s.end+1500 || e.ObservedAt.Sub(s.last) > 2*time.Second {
		rt.continuous.ordinal++
		s = &realtimeContinuousTranscript{id: fmt.Sprintf("fragment-%s-%d", role, rt.continuous.ordinal), start: e.TranscriptStartMS}
		rt.continuous.segments[role] = s
	}
	s.bytes += len(e.Transcript)
	s.end = max(s.end, e.TranscriptEndMS)
	s.last = e.ObservedAt
	// Punctuation is only a display boundary. Late text retains its own
	// timeline and the original fragments remain in the durable transcript.
	s.closed = strings.ContainsAny(strings.TrimSpace(e.Transcript), ".!?。！？")
	e.ItemID = s.id
	e.TranscriptGrouped = true
	e.Final = false
	return e
}

func (rt *RealtimeThinker) observeContinuousIdle() {
	if !rt.capabilities.ContinuousConversation {
		return
	}
	if !rt.continuous.lastAudio.IsZero() && time.Since(rt.continuous.lastAudio) >= realtimeObservedQuiet {
		rt.continuous.lastAudio = time.Time{}
		rt.responseMu.Lock()
		rt.responseActive = false
		rt.responsePending = false
		rt.responseMu.Unlock()
		if !rt.pendingToolWork() && !rt.backend.active {
			rt.setConversationState("listening", RealtimeEvent{})
		}
	}
}

// Reusable bounded prefix guard for continuous audio. Its segment boundaries
// are observed gaps/silence; they are never exposed as provider-final text.
type realtimeContinuousGuard struct {
	config             RealtimeSpeechGuardConfig
	pending            []RealtimeEvent
	bytes              int
	text               string
	approved, rejected bool
	serial             int
	lastAudio          time.Time
	lastTranscript     time.Time
	timing             *realtimeTiming
	silenceBytes       int
}

func (g *realtimeContinuousGuard) id() string { return fmt.Sprintf("continuous-audio-%d", g.serial) }
func (g *realtimeContinuousGuard) reset() {
	g.serial++
	g.pending = nil
	g.bytes = 0
	g.text = ""
	g.approved = false
	g.rejected = false
	g.silenceBytes = 0
}

func (g *realtimeContinuousGuard) audio(e RealtimeEvent, timing *realtimeTiming) []RealtimeEvent {
	g.timing = timing
	if !g.lastAudio.IsZero() && e.ObservedAt.Sub(g.lastAudio) > realtimeObservedQuiet {
		// Any unapproved buffered segment is explicitly blocked before reset.
		old := g.finish()
		g.reset()
		g.lastAudio = e.ObservedAt
		return append(old, g.audio(e, timing)...)
	}
	voiced := false
	for i := 0; i+1 < len(e.Audio); i += 2 {
		v := int(int16(binary.LittleEndian.Uint16(e.Audio[i:])))
		if v > 128 || v < -128 {
			voiced = true
			break
		}
	}
	if !voiced {
		g.silenceBytes += len(e.Audio)
		// Silence is not a speech segment and must not exhaust the guard.
		return nil
	}
	if g.silenceBytes >= realtimePCMBytesPerSecond*300/1000 {
		old := g.finish()
		g.reset()
		g.lastAudio = e.ObservedAt
		return append(old, g.audio(e, timing)...)
	}
	g.silenceBytes = 0
	g.lastAudio = e.ObservedAt
	e.ItemID = g.id()
	timing.audio("", e.ItemID, e.ObservedAt)
	if g.rejected {
		timing.dropped("", e.ItemID, len(e.Audio))
		return nil
	}
	if g.approved {
		timing.released("", e.ItemID, time.Now(), 0, "transcript_prefix")
		return []RealtimeEvent{e}
	}
	if g.bytes+len(e.Audio) > g.config.resolved().MaxBufferedAudioMS*realtimePCMBytesPerSecond/1000 {
		timing.dropped("", e.ItemID, len(e.Audio))
		return g.reject("missing_output_transcript")
	}
	g.pending = append(g.pending, e)
	g.bytes += len(e.Audio)
	timing.buffered("", e.ItemID, g.bytes)
	return nil
}

func (g *realtimeContinuousGuard) transcript(e RealtimeEvent, tools []NativeTool, timing *realtimeTiming) []RealtimeEvent {
	var previous []RealtimeEvent
	if !g.lastAudio.IsZero() && e.ObservedAt.Sub(g.lastAudio) > realtimeObservedQuiet && !g.lastTranscript.IsZero() && e.ObservedAt.Sub(g.lastTranscript) > realtimeObservedQuiet {
		previous = g.finish()
		g.reset()
		g.lastAudio = e.ObservedAt
	}
	g.lastTranscript = e.ObservedAt
	e.ItemID = g.id()
	g.text += e.Transcript
	_, _, leak := detectRealtimeToolMarkup(g.text, tools)
	if leak || detectRealtimeInternalNarration(g.text) != "" {
		return append(previous, g.reject("internal_instruction_narration")...)
	}
	result := []RealtimeEvent{e}
	if !g.rejected && !g.approved && (len(g.text) >= g.config.resolved().PrefixBytes || strings.ContainsAny(g.text, ".!?。！？")) {
		g.approved = true
		for _, pending := range g.pending {
			timing.released("", pending.ItemID, time.Now(), g.bytes, "transcript_prefix")
		}
		result = append(result, g.pending...)
		g.pending = nil
		g.bytes = 0
	}
	g.text = boundedRealtimeToolMarkupTail(g.text)
	return append(previous, result...)
}

func (g *realtimeContinuousGuard) reject(reason string) []RealtimeEvent {
	if g.rejected {
		return nil
	}
	for _, e := range g.pending {
		g.timing.dropped("", e.ItemID, len(e.Audio))
	}
	g.pending = nil
	g.bytes = 0
	g.rejected = true
	return []RealtimeEvent{{Type: RealtimeEventOutputBlocked, ItemID: g.id(), OutputBlockReason: reason}}
}

func (g *realtimeContinuousGuard) finish() []RealtimeEvent {
	if g.bytes > 0 && !g.rejected && strings.TrimSpace(g.text) != "" {
		g.approved = true
		result := g.pending
		for _, e := range result {
			g.timing.released("", e.ItemID, time.Now(), g.bytes, "transcript_prefix")
		}
		g.pending = nil
		g.bytes = 0
		return result
	}
	if g.bytes > 0 && !g.approved {
		return g.reject("missing_output_transcript")
	}
	return nil
}
