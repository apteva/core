package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// RealtimeThinker drives a thread whose conversation runs over a
// RealtimeSession (bidirectional WebSocket) instead of discrete
// request/response Chat() calls.
//
// It embeds *Thinker for shared state — registry, bus subscription,
// pause/quit channels, memory, config, telemetry. The standard Run()
// loop is replaced by an event-driven select that fans together:
//   - session events from the model (audio out, transcript, tool calls)
//   - inbound audio from the caller (mic / telephony)
//   - bus inbox events (send from other threads, lifecycle)
//   - pause / quit signals
//
// All Thinker mechanics that don't assume the iteration loop
// (executeTool, pendingTools tracking, persistence, telemetry) are
// reused unchanged.
type RealtimeThinker struct {
	*Thinker

	provider RealtimeProvider
	session  RealtimeSession
	voice    string

	audioIn  <-chan []byte
	audioOut chan<- []byte

	// transcript accumulates the final model utterances so the thread
	// has a readable log (visible via t.messages or session file). Not
	// resent to the model — the realtime API holds its own server-side
	// conversation state.
	transcriptMu sync.Mutex
}

// startRealtimeThinker builds a RealtimeThinker around an
// already-constructed Thinker and opens the session. Returns the
// RealtimeThinker ready to Run(); caller invokes Run() in a goroutine.
// On open failure, the embedded Thinker is left intact (caller is
// responsible for tearing it down) and the error is returned.
func startRealtimeThinker(
	ctx context.Context,
	thinker *Thinker,
	provider RealtimeProvider,
	directive string,
	voice string,
	audioIn <-chan []byte,
	audioOut chan<- []byte,
) (*RealtimeThinker, error) {
	if voice == "" {
		voice = provider.DefaultVoice()
	}

	// Native tool schemas — same registry the regular thinker uses.
	var nativeTools []NativeTool
	if thinker.registry != nil {
		nativeTools = thinker.registry.NativeTools(thinker.toolAllowlist, thinker.activeTools)
	}
	// interrupt is realtime-only and not in the shared registry. It
	// acts on the session itself (cancels the in-flight model
	// response) rather than running a goroutine, so the model needs
	// to see it in the tools list but it bypasses executeTool. The
	// special-case in dispatchToolCall handles invocation.
	nativeTools = append(nativeTools, NativeTool{
		Name:        "interrupt",
		Description: "Cancel your own in-flight speech immediately. Use when new context arrives mid-utterance and the rest of what you were about to say would be wrong or stale. Takes no arguments.",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	})

	model := ""
	if m := provider.Models(); m != nil {
		model = m[ModelLarge]
	}

	session, err := provider.Open(ctx, RealtimeSessionOpts{
		Model:        model,
		Voice:        voice,
		Instructions: directive,
		Tools:        nativeTools,
		AudioInFmt:   AudioPCM16,
		AudioOutFmt:  AudioPCM16,
	})
	if err != nil {
		return nil, fmt.Errorf("realtime open: %w", err)
	}

	return &RealtimeThinker{
		Thinker:  thinker,
		provider: provider,
		session:  session,
		voice:    voice,
		audioIn:  audioIn,
		audioOut: audioOut,
	}, nil
}

// Run is the event-driven counterpart to Thinker.Run. It blocks until
// the session ends, the thread is killed, or the bus quits. Tool
// calls fire as goroutines via the existing executeTool plumbing and
// their results are delivered back via session.SendToolResult.
func (rt *RealtimeThinker) Run() {
	defer func() {
		if rt.session != nil {
			_ = rt.session.Close()
		}
		if rt.onStop != nil {
			rt.onStop()
		}
	}()

	logMsg("REALTIME", fmt.Sprintf("[%s] session up, voice=%s", rt.threadID, rt.voice))

	// Wire executeTool's result publish path to this thread. The
	// existing tools.go executeTool publishes a ToolResult event on
	// the bus; we subscribe to our own thread's inbox to catch them
	// (same pattern as the regular Thinker iteration uses).

	for {
		var audioIn <-chan []byte = rt.audioIn // nil-safe — nil chan blocks forever
		select {

		case ev, ok := <-rt.session.Events():
			if !ok {
				logMsg("REALTIME", fmt.Sprintf("[%s] session closed", rt.threadID))
				return
			}
			rt.handleSessionEvent(ev)

		case audio, ok := <-audioIn:
			if !ok {
				rt.audioIn = nil // stop selecting on it
				continue
			}
			if err := rt.session.SendAudio(audio); err != nil {
				logMsg("REALTIME", fmt.Sprintf("[%s] send audio: %v", rt.threadID, err))
			}

		case ev := <-rt.sub.C:
			rt.handleBusEvent(ev)

		case paused := <-rt.pause:
			rt.paused = paused
			if paused {
				logMsg("REALTIME", fmt.Sprintf("[%s] paused (audio still received, model continues)", rt.threadID))
			} else {
				logMsg("REALTIME", fmt.Sprintf("[%s] resumed", rt.threadID))
			}

		case <-rt.quit:
			logMsg("REALTIME", fmt.Sprintf("[%s] quit", rt.threadID))
			return
		}
	}
}

// handleSessionEvent routes one RealtimeEvent from the model.
func (rt *RealtimeThinker) handleSessionEvent(ev RealtimeEvent) {
	switch ev.Type {
	case RealtimeEventAudioOut:
		if rt.audioOut != nil {
			select {
			case rt.audioOut <- ev.Audio:
			default:
				// Caller's consumer is slow — dropping a chunk is
				// preferable to stalling the session goroutine. In
				// practice the caller's sink should always drain
				// faster than the model produces.
			}
		}

	case RealtimeEventTranscriptOutput:
		if ev.Final && ev.Transcript != "" {
			rt.transcriptMu.Lock()
			rt.messages = append(rt.messages, Message{Role: "assistant", Content: ev.Transcript})
			rt.transcriptMu.Unlock()
			if rt.telemetry != nil {
				rt.telemetry.EmitLive("realtime.assistant", rt.threadID, map[string]any{
					"text": ev.Transcript,
				})
			}
		}

	case RealtimeEventTranscriptInput:
		if ev.Final && ev.Transcript != "" {
			rt.transcriptMu.Lock()
			rt.messages = append(rt.messages, Message{Role: "user", Content: ev.Transcript})
			rt.transcriptMu.Unlock()
			if rt.telemetry != nil {
				rt.telemetry.EmitLive("realtime.user", rt.threadID, map[string]any{
					"text": ev.Transcript,
				})
			}
		}

	case RealtimeEventToolCall:
		rt.dispatchToolCall(ev)

	case RealtimeEventResponseDone:
		// Lifecycle marker — nothing to do; transcript .done events
		// have already populated history.

	case RealtimeEventError:
		logMsg("REALTIME", fmt.Sprintf("[%s] session error: %v", rt.threadID, ev.Err))

	case RealtimeEventSessionEnded:
		// Read loop will close Events() channel; Run() will exit on
		// the next iteration via the !ok branch.
	}
}

// dispatchToolCall dispatches a tool call from the model. Reuses the
// standard executeTool path so realtime threads get the same panic
// recovery, telemetry, and result routing as text threads. The
// result is delivered to the session via handleBusEvent when
// executeTool publishes the result event on the bus.
func (rt *RealtimeThinker) dispatchToolCall(ev RealtimeEvent) {
	logMsg("REALTIME", fmt.Sprintf("[%s] tool call %s id=%s args=%dB",
		rt.threadID, ev.ToolName, ev.ToolCallID, len(ev.ToolArgs)))

	// Special-case the realtime-only interrupt tool — it acts on the
	// session directly, not through the registry.
	if ev.ToolName == "interrupt" {
		if err := rt.session.Interrupt(); err != nil {
			_ = rt.session.SendToolResult(ev.ToolCallID, "interrupt failed: "+err.Error(), true)
			return
		}
		_ = rt.session.SendToolResult(ev.ToolCallID, "interrupted", false)
		return
	}

	args := flattenJSONArgs(ev.ToolArgs)
	call := toolCall{
		Name:     ev.ToolName,
		Args:     args,
		Raw:      fmt.Sprintf("[[%s native_realtime]]", ev.ToolName),
		NativeID: ev.ToolCallID,
	}
	executeTool(rt.Thinker, call)
}

// flattenJSONArgs converts a JSON object string into map[string]string
// the same way the text providers do (string values pass through,
// non-strings are re-marshaled). Returns an empty map on parse error
// so the dispatcher can surface "unknown args" rather than panic.
func flattenJSONArgs(raw string) map[string]string {
	args := map[string]string{}
	if strings.TrimSpace(raw) == "" {
		return args
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return args
	}
	for k, v := range obj {
		switch tv := v.(type) {
		case string:
			args[k] = tv
		default:
			b, _ := json.Marshal(v)
			args[k] = string(b)
		}
	}
	return args
}

// handleBusEvent processes one inbound bus event. There are two
// kinds we care about:
//   - tool results from executeTool goroutines (EventInbox with a
//     non-nil ToolResult). These are forwarded to the session as
//     function_call_output so the model sees its tool result.
//   - text messages from other threads ("send" from main, or peer
//     workers). These are injected as system notes so the model can
//     react.
//
// The discriminator is ToolResult != nil — that's the same shape
// executeTool produces today (tools.go ~line 164). Lifecycle events
// (pause/quit) are handled separately in the main select.
func (rt *RealtimeThinker) handleBusEvent(ev Event) {
	if ev.Type != EventInbox {
		return
	}

	// Tool result delivery path. executeTool publishes EventInbox
	// with a populated ToolResult; forward it to the realtime session.
	if ev.ToolResult != nil {
		_ = rt.session.SendToolResult(
			ev.ToolResult.CallID,
			ev.ToolResult.Content,
			ev.ToolResult.IsError,
		)
		return
	}

	// Inter-thread send. Inject as a system note so the model can
	// react inside the conversation.
	var note string
	if ev.From != "" {
		note = fmt.Sprintf("[from:%s] %s", ev.From, ev.Text)
	} else {
		note = ev.Text
	}
	if strings.TrimSpace(note) == "" {
		return
	}
	if err := rt.session.SendText("system", note); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] inject text: %v", rt.threadID, err))
	}
}
