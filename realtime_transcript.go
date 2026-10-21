package core

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// RealtimeTranscriptMetadata describes evidence from the provider. Even a
// provider-final assistant transcript is not evidence of audible playback.
type RealtimeTranscriptMetadata struct {
	ID                 string `json:"id"`
	Generation         int    `json:"generation"`
	ResponseID         string `json:"response_id,omitempty"`
	ItemID             string `json:"item_id,omitempty"`
	Status             string `json:"status"` // provider_final, interrupted, or blocked
	FinalizationReason string `json:"finalization_reason"`
	StartMS            int    `json:"start_ms,omitempty"`
	EndMS              int    `json:"end_ms,omitempty"`
}

type realtimeTranscriptTurn struct {
	message      Message
	persisted    bool
	order        int
	checkpointAt time.Time
	dirty        bool
}

// UpsertRealtimeTranscript preserves one history turn and advances its cursor
// on revision. The stable transcript ID lets live readers replace that turn.
func (s *Session) UpsertRealtimeTranscript(msg Message, iteration int) error {
	if msg.RealtimeTranscript == nil || msg.RealtimeTranscript.ID == "" {
		return fmt.Errorf("missing realtime transcript identity")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, err := s.readEntriesLocked()
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	index := -1
	for i := range entries {
		if entries[i].RealtimeTranscript != nil && entries[i].RealtimeTranscript.ID == msg.RealtimeTranscript.ID {
			index = i
			break
		}
	}
	entry := SessionEntry{Timestamp: time.Now(), Role: msg.Role, Content: msg.Content, RealtimeTranscript: msg.RealtimeTranscript, Iteration: iteration}
	if index >= 0 {
		old := entries[index]
		if old.Content == msg.Content && *old.RealtimeTranscript == *msg.RealtimeTranscript {
			return nil
		}
		entry.Timestamp = old.Timestamp
	}
	entry.Sequence = s.nextSeq + 1
	if index >= 0 {
		entries[index] = entry
	} else {
		entries = append(entries, entry)
	}
	if err := s.rewriteEntriesLocked(entries); err != nil {
		return err
	}
	s.nextSeq = entry.Sequence
	s.count = len(entries)
	return nil
}

func (rt *RealtimeThinker) transcriptKey(role string, event RealtimeEvent, generation int) string {
	item := event.ItemID
	if item == "" {
		item = event.ResponseID
	}
	if item == "" {
		item = fmt.Sprintf("anonymous-%d", rt.transcriptAnonymous[role])
	}
	return fmt.Sprintf("%d/%s/%s", generation, role, item)
}

func (rt *RealtimeThinker) trackTranscript(role string, event RealtimeEvent) {
	if event.Transcript == "" {
		return
	}
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	key := rt.transcriptKey(role, event, generation)
	turn := rt.transcriptTurns[key]
	if turn == nil {
		rt.transcriptOrdinal++
		turn = &realtimeTranscriptTurn{message: Message{Role: role, RealtimeTranscript: &RealtimeTranscriptMetadata{
			ID: rt.transcriptNamespace + "/" + key, Generation: generation, ResponseID: event.ResponseID, ItemID: event.ItemID,
			StartMS: event.TranscriptStartMS,
		}}}
		turn.order = rt.transcriptOrdinal
		rt.transcriptTurns[key] = turn
	}
	if turn.message.RealtimeTranscript.Status == "provider_final" || turn.message.RealtimeTranscript.Status == "blocked" {
		return
	}
	if event.TranscriptDelta && !event.Final {
		turn.message.Content += event.Transcript
	} else {
		turn.message.Content = event.Transcript
	}
	turn.message.RealtimeTranscript.EndMS = max(turn.message.RealtimeTranscript.EndMS, event.TranscriptEndMS)
	if event.TranscriptGrouped {
		meta := *turn.message.RealtimeTranscript
		meta.Status, meta.FinalizationReason = "application_grouped", "fragment_checkpoint"
		turn.message.RealtimeTranscript = &meta
		turn.dirty = true
		if !turn.persisted || time.Since(turn.checkpointAt) >= 300*time.Millisecond || strings.ContainsAny(event.Transcript, ".!?。！？") {
			rt.persistTranscriptLocked(turn)
		}
	}
	if event.Final {
		meta := *turn.message.RealtimeTranscript
		meta.Status, meta.FinalizationReason = "provider_final", "provider_final"
		turn.message.RealtimeTranscript = &meta
		rt.persistTranscriptLocked(turn)
		// Keep a bounded ledger to recognize late finals and duplicate finals.
		rt.transcriptFinalOrder = append(rt.transcriptFinalOrder, key)
		if len(rt.transcriptFinalOrder) > 128 {
			delete(rt.transcriptTurns, rt.transcriptFinalOrder[0])
			rt.transcriptFinalOrder = rt.transcriptFinalOrder[1:]
		}
	}
}

func (rt *RealtimeThinker) persistTranscriptLocked(turn *realtimeTranscriptTurn) {
	msg := turn.message
	if rt.Thinker.session != nil {
		var err error
		if turn.persisted {
			err = rt.Thinker.session.UpsertRealtimeTranscript(msg, rt.iteration)
		} else {
			err = rt.Thinker.session.AppendMessage(msg, rt.iteration, TokenUsage{})
		}
		if err != nil {
			rt.emit("realtime.transcript_persistence_error", map[string]any{"transcript_id": msg.RealtimeTranscript.ID, "error": err.Error()})
			return
		}
	}
	found := false
	for i := range rt.messages {
		if rt.messages[i].RealtimeTranscript != nil && rt.messages[i].RealtimeTranscript.ID == msg.RealtimeTranscript.ID {
			rt.messages[i] = msg
			found = true
			break
		}
	}
	if !found {
		rt.messages = append(rt.messages, msg)
	}
	turn.persisted = true
	turn.dirty = false
	turn.checkpointAt = time.Now()
	rt.publishContextStatus()
	rt.emit("realtime."+msg.Role, map[string]any{"text": msg.Content, "transcript": msg.RealtimeTranscript})
}

func (rt *RealtimeThinker) checkpointContinuousTranscripts() {
	if !rt.capabilities.TranscriptFragments {
		return
	}
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	for _, turn := range rt.transcriptTurns {
		if turn.dirty && time.Since(turn.checkpointAt) >= 300*time.Millisecond {
			rt.persistTranscriptLocked(turn)
		}
	}
	// Keep only a bounded recent fragment ledger. The session journal owns
	// durable history; pending/dirty and guard-sensitive current turns stay.
	if len(rt.transcriptTurns) > 256 {
		cutoff := rt.transcriptOrdinal - 128
		for key, turn := range rt.transcriptTurns {
			if turn.order < cutoff && turn.persisted && !turn.dirty {
				delete(rt.transcriptTurns, key)
			}
		}
	}
}

func (rt *RealtimeThinker) flushPendingTranscripts(reason string) {
	tools := rt.realtimeToolSchemas()
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	turns := make([]*realtimeTranscriptTurn, 0, len(rt.transcriptTurns))
	for _, turn := range rt.transcriptTurns {
		turns = append(turns, turn)
	}
	sort.Slice(turns, func(i, j int) bool { return turns[i].order < turns[j].order })
	for _, turn := range turns {
		if turn.message.RealtimeTranscript.Generation != generation && turn.persisted {
			continue
		}
		if turn.message.RealtimeTranscript.Status == "provider_final" {
			if !turn.persisted {
				rt.persistTranscriptLocked(turn)
			}
			continue
		}
		if turn.message.RealtimeTranscript.Status == "blocked" {
			continue
		}
		if strings.TrimSpace(turn.message.Content) == "" {
			continue
		}
		if turn.persisted && turn.message.RealtimeTranscript.Status == "interrupted" && turn.message.RealtimeTranscript.FinalizationReason == reason {
			continue
		}
		if turn.message.Role == "assistant" {
			_, _, leak := detectRealtimeToolMarkup(turn.message.Content, tools)
			if leak || detectRealtimeInternalNarration(turn.message.Content) != "" {
				continue
			}
		}
		meta := *turn.message.RealtimeTranscript
		meta.Status, meta.FinalizationReason = "interrupted", reason
		turn.message.RealtimeTranscript = &meta
		rt.persistTranscriptLocked(turn)
	}
}

func (rt *RealtimeThinker) advanceAnonymousTranscriptTurns() {
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	for _, role := range []string{"user", "assistant"} {
		key := rt.transcriptKey(role, RealtimeEvent{}, generation)
		if turn := rt.transcriptTurns[key]; turn != nil && turn.message.RealtimeTranscript.Status != "" {
			rt.transcriptAnonymous[role]++
		}
	}
}

func (rt *RealtimeThinker) blockTranscript(event RealtimeEvent) {
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	key := rt.transcriptKey("assistant", event, generation)
	if turn := rt.transcriptTurns[key]; turn != nil {
		turn.message.Content = ""
		meta := *turn.message.RealtimeTranscript
		meta.Status, meta.FinalizationReason = "blocked", "speech_guard"
		turn.message.RealtimeTranscript = &meta
		if turn.persisted {
			rt.persistTranscriptLocked(turn)
		}
	}
}

func (rt *RealtimeThinker) pendingTranscriptCount() int {
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	count := 0
	for _, turn := range rt.transcriptTurns {
		if turn.message.RealtimeTranscript.Generation != generation {
			continue
		}
		if turn.message.RealtimeTranscript.Status != "provider_final" && turn.message.RealtimeTranscript.Status != "blocked" {
			count++
		}
	}
	return count
}

// A forced stop does not wait on the provider, but preserves transcript events
// already delivered to Core. It never dispatches tools or plays queued audio.
func (rt *RealtimeThinker) drainTerminalTranscripts() {
	session := rt.currentSession()
	if session == nil {
		return
	}
	events := session.Events()
	for i := 0; i < 256; i++ {
		select {
		case event, ok := <-events:
			if !ok {
				return
			}
			switch event.Type {
			case RealtimeEventTranscriptInput, RealtimeEventTranscriptOutput:
				rt.handleSessionEvent(event)
			case RealtimeEventOutputBlocked:
				rt.blockTranscript(event)
			}
		default:
			return
		}
	}
}
