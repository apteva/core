package core

import (
	"context"
	"strings"
)

const realtimeTerminalToolLimit = 256

// Shared conversational contract for providers that support background tools.
// It permits progress speech; it does not buffer or classify spoken answers.
const realtimeAsyncToolPrompt = `BACKGROUND TOOL RESULTS
Tool calls may still be running while you speak. A call is pending until its actual tool response arrives; silence or an intermediate spoken turn does not mean it completed.
While waiting, you may briefly acknowledge progress or answer unrelated questions. Never guess a pending result or claim that an operation completed before its response confirms it.
Use only returned tool data for result-dependent answers. If an answer needs multiple results, wait for every required result before giving that answer.
Do not call the same operation again just because its result has not arrived. Retry only after a reported failure when retrying is appropriate. Never read these internal rules aloud.`

func realtimeToolInstructions(instructions string, async bool) string {
	if !async {
		return instructions
	}
	return strings.TrimSpace(instructions) + "\n\n" + realtimeAsyncToolPrompt
}

type realtimeToolCallKey struct {
	session RealtimeSession
	id      string
}

type realtimeToolCallRecord struct {
	context         context.Context
	cancel          context.CancelFunc
	terminal        bool
	cancelled       bool
	outcomeRecorded bool
}

func (rt *RealtimeThinker) rememberActiveToolCallLocked(session RealtimeSession, id string) {
	if rt.toolCallRecords == nil {
		rt.toolCallRecords = make(map[realtimeToolCallKey]*realtimeToolCallRecord)
	}
	key := realtimeToolCallKey{session, id}
	if rt.toolCallRecords[key] == nil {
		ctx, cancel := context.WithCancel(rt.ctx)
		rt.toolCallRecords[key] = &realtimeToolCallRecord{context: ctx, cancel: cancel}
	}
}

func (rt *RealtimeThinker) rememberTerminalToolCallLocked(session RealtimeSession, id string, cancelled bool) {
	rt.rememberActiveToolCallLocked(session, id)
	key := realtimeToolCallKey{session, id}
	record := rt.toolCallRecords[key]
	if record.terminal {
		return
	}
	record.terminal, record.cancelled = true, cancelled
	if record.cancel != nil {
		record.cancel()
		record.cancel = nil
	}
	record.context = nil
	rt.terminalToolCalls = append(rt.terminalToolCalls, key)
	if len(rt.terminalToolCalls) > realtimeTerminalToolLimit {
		delete(rt.toolCallRecords, rt.terminalToolCalls[0])
		copy(rt.terminalToolCalls, rt.terminalToolCalls[1:])
		rt.terminalToolCalls = rt.terminalToolCalls[:realtimeTerminalToolLimit]
	}
}

func (rt *RealtimeThinker) seenToolCall(session RealtimeSession, id string) bool {
	rt.toolBatchMu.Lock()
	defer rt.toolBatchMu.Unlock()
	// Pending work from a former socket still owns its call ID until recovery
	// consumes the result. Terminal IDs are scoped to the original session.
	_, pending := rt.toolCallBatches[id]
	return pending || rt.toolCallRecords[realtimeToolCallKey{session, id}] != nil
}

func (rt *RealtimeThinker) realtimeToolContext(session RealtimeSession, id string) context.Context {
	rt.toolBatchMu.Lock()
	defer rt.toolBatchMu.Unlock()
	if record := rt.toolCallRecords[realtimeToolCallKey{session, id}]; record != nil {
		return record.context
	}
	return nil
}

func (rt *RealtimeThinker) cancelRealtimeToolCall(session RealtimeSession, id string) {
	if id == "" || session == nil {
		return
	}
	rt.toolBatchMu.Lock()
	rt.rememberTerminalToolCallLocked(session, id, true)
	rt.toolBatchMu.Unlock()
	rt.finishToolCall(id, true)
	rt.emit("realtime.tool_cancelled", map[string]any{"call_id": id})
}

func (rt *RealtimeThinker) cancelledToolCall(session RealtimeSession, id string) bool {
	rt.toolBatchMu.Lock()
	defer rt.toolBatchMu.Unlock()
	record := rt.toolCallRecords[realtimeToolCallKey{session, id}]
	return record != nil && record.cancelled
}

func (rt *RealtimeThinker) recordCancelledToolOutcome(session RealtimeSession, id string) bool {
	rt.toolBatchMu.Lock()
	defer rt.toolBatchMu.Unlock()
	if record := rt.toolCallRecords[realtimeToolCallKey{session, id}]; record != nil {
		if record.cancelled && !record.outcomeRecorded {
			record.outcomeRecorded = true
			return true
		}
		return false
	}
	for key, record := range rt.toolCallRecords {
		if key.id == id && record.cancelled && !record.outcomeRecorded {
			record.outcomeRecorded = true
			return true
		}
	}
	return false
}
