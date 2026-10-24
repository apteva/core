package core

import (
	"fmt"
	"strings"
	"time"
)

const realtimeObservedQuiet = 700 * time.Millisecond

// Continuous providers expose fragments rather than final turns. Grouping is
// an application observation, never a fabricated provider completion event.
type realtimeContinuousState struct {
	lastAudio   time.Time // provider observation; not renderer completion
	lastRelease time.Time // Core delivery time of accepted audio
	segments    map[string]*realtimeContinuousTranscript
	ordinal     int
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
		if !e.AudioSilence {
			rt.continuous.lastAudio = e.ObservedAt
		}
		if e.ItemID == "" {
			e.ItemID = fmt.Sprintf("stream-%d", rt.sessionGeneration)
		}
		return e
	}
	if e.Type == RealtimeEventOutputBlocked && e.OutputBlockReason != "missing_output_transcript" && e.ItemID == "" {
		if segment := rt.continuous.segments[string(RealtimeEventTranscriptOutput)]; segment != nil {
			e.ItemID = segment.id
		}
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
	otherRole := string(RealtimeEventTranscriptInput)
	if e.Type == RealtimeEventTranscriptInput {
		otherRole = string(RealtimeEventTranscriptOutput)
	}
	other := rt.continuous.segments[otherRole]
	// An opposite-speaker onset is a display boundary, even without punctuation.
	// Late captions retain their timed fragments for chronological presentation.
	crossedSpeaker := e.Type == RealtimeEventTranscriptOutput && s != nil && other != nil && other.start > s.start && e.TranscriptStartMS >= other.start
	if s == nil || s.closed || crossedSpeaker || s.bytes >= 2048 || e.TranscriptStartMS > s.end+1500 || e.ObservedAt.Sub(s.last) > 2*time.Second {
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
	if !rt.continuous.lastRelease.IsZero() && time.Since(rt.continuous.lastRelease) >= realtimeObservedQuiet {
		if rt.delivery.hasPendingPlayback() {
			return
		}
		rt.continuous.lastRelease = time.Time{}
		rt.currentTiming().finish("", "", "application_observed_quiet")
		_, _, backlog := rt.delivery.snapshot()
		rt.emit("realtime.playback_idle", backlog)
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
	format             AudioFormat
	config             RealtimeSpeechGuardConfig
	pending            []RealtimeEvent
	bytes              int
	text               string
	monitorText        string // independent of application audio segmentation
	namespace          string
	captionReported    bool
	approved, rejected bool
	serial             int
	lastAudio          time.Time
	lastTranscript     time.Time
	timing             *realtimeTiming
	silenceBytes       int
}

func (g *realtimeContinuousGuard) id() string {
	if g.namespace == "" {
		g.namespace = generateAudioToken()
	}
	return fmt.Sprintf("continuous-audio-%s-%d", g.namespace, g.serial)
}
func (g *realtimeContinuousGuard) reset() {
	g.serial++
	g.pending = nil
	g.bytes = 0
	g.text = ""
	g.approved = false
	g.rejected = false
	g.silenceBytes = 0
	g.captionReported = false
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
	voiced := mediaVoiced(e.Audio, g.format)
	// Silence may delimit an application playback segment, but it remains
	// ordered PCM. Never remove time from the renderer's audio timeline.
	if voiced && g.silenceBytes >= mediaBytesPerSecond(g.format)*300/1000 {
		old := g.finish()
		g.reset()
		g.lastAudio = e.ObservedAt
		return append(old, g.audio(e, timing)...)
	}
	if voiced {
		g.silenceBytes = 0
	} else {
		g.silenceBytes += len(e.Audio)
	}
	g.lastAudio = e.ObservedAt
	e.AudioSilence = !voiced
	e.ItemID = g.id()
	timing.audio("", e.ItemID, e.ObservedAt)
	if g.rejected {
		timing.dropped("", e.ItemID, len(e.Audio))
		return nil
	}
	if g.approved || g.config.continuousPolicy().Mode == "monitor" {
		timing.released("", e.ItemID, time.Now(), 0, g.config.continuousPolicy().Mode)
		return []RealtimeEvent{e}
	}
	if g.bytes+len(e.Audio) > g.config.resolved().MaxBufferedAudioMS*mediaBytesPerSecond(g.format)/1000 {
		timing.dropped("", e.ItemID, len(e.Audio))
		return g.missingTranscript()
	}
	g.pending = append(g.pending, e)
	g.bytes += len(e.Audio)
	timing.buffered("", e.ItemID, g.bytes)
	return nil
}

func (g *realtimeContinuousGuard) transcript(e RealtimeEvent, tools []NativeTool, timing *realtimeTiming) []RealtimeEvent {
	g.timing = timing
	g.lastTranscript = e.ObservedAt
	e.PlaybackID = g.id()
	// Transcript fragments have their own session timeline and display identity.
	// In particular an arrival gap must not erase a partial syntax match.
	g.monitorText += e.Transcript
	_, pattern, leak := detectRealtimeToolMarkup(g.monitorText, tools)
	if !leak {
		pattern = detectRealtimeInternalNarration(g.monitorText)
		leak = pattern != ""
	}
	if leak {
		g.monitorText = ""
		blocked := g.reject(pattern)
		for i := range blocked {
			blocked[i].TranscriptStartMS = e.TranscriptStartMS
			blocked[i].TranscriptEndMS = e.TranscriptEndMS
		}
		return blocked
	}
	g.monitorText = boundedRealtimeToolMarkupTail(g.monitorText)
	g.text += e.Transcript
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
	return result
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
	return []RealtimeEvent{{Type: RealtimeEventOutputBlocked, PlaybackID: g.id(), OutputBlockReason: reason}}
}

func (g *realtimeContinuousGuard) finish() []RealtimeEvent {
	if !g.lastAudio.IsZero() && g.text == "" && !g.rejected {
		return g.missingTranscript()
	}
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
		return g.missingTranscript()
	}
	return nil
}

func (g *realtimeContinuousGuard) missingTranscript() []RealtimeEvent {
	if g.captionReported {
		return nil
	}
	g.captionReported = true
	if g.config.continuousPolicy().Mode == "transcript_prefix" {
		return g.reject("missing_output_transcript")
	}
	return []RealtimeEvent{{Type: RealtimeEventTranscriptIncomplete, PlaybackID: g.id(), OutputBlockReason: "missing_output_transcript"}}
}

// Cancellation invalidates pending release, without cancelling delegated work.
func (g *realtimeContinuousGuard) discard(item string) {
	if item != "" && item != g.id() {
		return
	}
	for _, e := range g.pending {
		g.timing.dropped("", e.ItemID, len(e.Audio))
	}
	g.pending = nil
	g.bytes = 0
	g.rejected = true
}
