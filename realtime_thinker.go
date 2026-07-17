package core

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

const (
	realtimeReconnectMinDelay = time.Second
	realtimeReconnectMaxDelay = 30 * time.Second
	realtimeRestoreMessages   = 32
	realtimePCMBytesPerSecond = 24000 * 2
)

// RealtimeThinker is the event-driven counterpart to Thinker. It deliberately
// reuses Thinker's prompt, tool handler, execution gates, durable Session, bus,
// and telemetry instead of maintaining a second, weaker agent runtime.
type RealtimeThinker struct {
	*Thinker

	provider RealtimeProvider
	voice    string
	opts     RealtimeSessionOpts

	ctx    context.Context
	cancel context.CancelFunc

	sessionMu sync.RWMutex
	rtSession RealtimeSession

	audioIn      <-chan []byte
	audioOut     chan<- []byte
	audioControl chan<- string

	transcriptMu sync.Mutex
	outputMu     sync.Mutex
	outputItemID string
	outputBytes  int

	toolBatchMu     sync.Mutex
	toolBatches     map[string]*realtimeToolBatch
	toolCallBatches map[string]string
}

type realtimeToolBatch struct {
	session      RealtimeSession
	pending      int
	responseDone bool
}

func realtimeNativeTools(thinker *Thinker) []NativeTool {
	var tools []NativeTool
	if thinker.registry != nil {
		tools = thinker.registry.NativeTools(thinker.toolAllowlist, thinker.activeTools, thinker.systemThread)
	}
	return append(tools, NativeTool{
		Name:        "interrupt",
		Description: "Cancel your own in-flight speech immediately when the remaining utterance is stale. Takes no arguments.",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{},
		},
	})
}

func realtimeSafetyIdentifier(threadID string) string {
	sum := sha256.Sum256([]byte("apteva-realtime:" + threadID))
	return "apt_" + hex.EncodeToString(sum[:12])
}

func newRealtimeThinker(
	ctx context.Context,
	thinker *Thinker,
	provider RealtimeProvider,
	voice string,
	audioIn <-chan []byte,
	audioOut chan<- []byte,
	audioControl chan<- string,
) *RealtimeThinker {
	if voice == "" {
		voice = provider.DefaultVoice()
	}
	model := ""
	if models := provider.Models(); models != nil {
		model = models[thinker.agentModel]
		if model == "" {
			model = models[ModelLarge]
		}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithCancel(ctx)
	rt := &RealtimeThinker{
		Thinker: thinker, provider: provider, voice: voice,
		ctx: runCtx, cancel: cancel,
		audioIn: audioIn, audioOut: audioOut, audioControl: audioControl,
		toolBatches: map[string]*realtimeToolBatch{}, toolCallBatches: map[string]string{},
	}
	rt.opts = RealtimeSessionOpts{
		Model:              model,
		Voice:              voice,
		Instructions:       rt.currentInstructions(),
		Tools:              realtimeNativeTools(thinker),
		AudioInFmt:         AudioPCM16,
		AudioOutFmt:        AudioPCM16,
		AudioInRate:        24000,
		AudioOutRate:       24000,
		Reasoning:          thinker.agentReasoning.String(),
		SafetyIdentifier:   realtimeSafetyIdentifier(thinker.threadID),
		TranscribeInput:    true,
		TranscriptionModel: provider.DefaultTranscriptionModel(),
	}
	return rt
}

// startRealtimeThinker opens once before returning so a bad credential or
// contract fails the spawn synchronously. Run renews the session thereafter.
func startRealtimeThinker(
	ctx context.Context,
	thinker *Thinker,
	provider RealtimeProvider,
	voice string,
	audioIn <-chan []byte,
	audioOut chan<- []byte,
	audioControl chan<- string,
) (*RealtimeThinker, error) {
	rt := newRealtimeThinker(ctx, thinker, provider, voice, audioIn, audioOut, audioControl)
	if err := rt.openSession(false); err != nil {
		rt.cancel()
		return nil, fmt.Errorf("realtime open: %w", err)
	}
	return rt, nil
}

func (rt *RealtimeThinker) currentInstructions() string {
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	return rt.currentInstructionsLocked()
}

func (rt *RealtimeThinker) currentInstructionsLocked() string {
	if len(rt.messages) > 0 && rt.messages[0].Role == "system" {
		return rt.messages[0].Content
	}
	return rt.directive
}

func (rt *RealtimeThinker) configurationSnapshot() (string, []NativeTool) {
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	return rt.currentInstructionsLocked(), realtimeNativeTools(rt.Thinker)
}

func (rt *RealtimeThinker) currentSession() RealtimeSession {
	rt.sessionMu.RLock()
	defer rt.sessionMu.RUnlock()
	return rt.rtSession
}

func (rt *RealtimeThinker) replaceSession(next RealtimeSession) {
	rt.sessionMu.Lock()
	previous := rt.rtSession
	rt.rtSession = next
	rt.sessionMu.Unlock()
	if previous != nil && previous != next {
		rt.toolBatchMu.Lock()
		rt.toolBatches = map[string]*realtimeToolBatch{}
		rt.toolCallBatches = map[string]string{}
		rt.toolBatchMu.Unlock()
		_ = previous.Close()
	}
}

func (rt *RealtimeThinker) beginToolCall(event RealtimeEvent, session RealtimeSession) {
	batchID := event.ResponseID
	rt.toolBatchMu.Lock()
	batch := rt.toolBatches[batchID]
	if batch == nil {
		batch = &realtimeToolBatch{session: session}
		rt.toolBatches[batchID] = batch
	}
	batch.pending++
	rt.toolCallBatches[event.ToolCallID] = batchID
	rt.toolBatchMu.Unlock()
}

func (rt *RealtimeThinker) completeToolCall(callID string) {
	var continueSession RealtimeSession
	rt.toolBatchMu.Lock()
	batchID, exists := rt.toolCallBatches[callID]
	if exists {
		delete(rt.toolCallBatches, callID)
		if batch := rt.toolBatches[batchID]; batch != nil {
			if batch.pending > 0 {
				batch.pending--
			}
			if batch.responseDone && batch.pending == 0 {
				continueSession = batch.session
				delete(rt.toolBatches, batchID)
			}
		}
	}
	rt.toolBatchMu.Unlock()
	if continueSession != nil {
		_ = continueSession.RequestResponse()
	}
}

func (rt *RealtimeThinker) completeToolResponse(responseID string) {
	var continueSession RealtimeSession
	rt.toolBatchMu.Lock()
	batchID := responseID
	batch := rt.toolBatches[batchID]
	if batch == nil && responseID != "" {
		// Some compatible providers omit response_id on function-call events.
		// There can be only one active provider response on a session, so bind
		// an otherwise-unidentified batch when response.done arrives.
		batchID, batch = "", rt.toolBatches[""]
	}
	if batch != nil {
		batch.responseDone = true
		if batch.pending == 0 {
			continueSession = batch.session
			delete(rt.toolBatches, batchID)
		}
	}
	rt.toolBatchMu.Unlock()
	if continueSession != nil {
		_ = continueSession.RequestResponse()
	}
}

func (rt *RealtimeThinker) submitToolResult(session RealtimeSession, callID, result string, isError bool) {
	if err := session.SendToolResult(callID, result, isError); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] send tool result %s: %v", rt.threadID, callID, err))
		return
	}
	rt.completeToolCall(callID)
}

func (rt *RealtimeThinker) boundedTranscript() []Message {
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	start := 1
	if start > len(rt.messages) {
		start = len(rt.messages)
	}
	eligible := make([]Message, 0, len(rt.messages)-start)
	for _, msg := range rt.messages[start:] {
		if (msg.Role == "user" || msg.Role == "assistant") && strings.TrimSpace(msg.Content) != "" {
			eligible = append(eligible, Message{Role: msg.Role, Content: msg.Content})
		}
	}
	if len(eligible) > realtimeRestoreMessages {
		eligible = eligible[len(eligible)-realtimeRestoreMessages:]
	}
	return eligible
}

func (rt *RealtimeThinker) openSession(restore bool) error {
	instructions, tools := rt.configurationSnapshot()
	rt.transcriptMu.Lock()
	rt.opts.Instructions, rt.opts.Tools = instructions, tools
	opts := rt.opts
	rt.transcriptMu.Unlock()
	session, err := rt.provider.Open(rt.ctx, opts)
	if err != nil {
		return err
	}
	var history []Message
	if restore {
		history = rt.boundedTranscript()
	}
	// Called even for a new/empty session. Most providers treat this as a
	// no-op; Gemini Live uses it to finish its initial history-seeding phase
	// before accepting normal realtime input.
	if err := session.RestoreConversation(history); err != nil {
		_ = session.Close()
		return fmt.Errorf("restore conversation: %w", err)
	}
	rt.replaceSession(session)
	return nil
}

func (rt *RealtimeThinker) refreshConfiguration() {
	session := rt.currentSession()
	if session == nil {
		return
	}
	instructions, tools := rt.configurationSnapshot()
	if err := session.UpdateConfiguration(instructions, tools); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] update configuration: %v", rt.threadID, err))
	}
}

func (rt *RealtimeThinker) emit(eventType string, data map[string]any) {
	if rt.telemetry != nil {
		rt.telemetry.Emit(eventType, rt.threadID, data)
	}
}

// Run survives normal provider-enforced session endings. Only explicit thread
// stop/cancellation ends the worker and invokes the normal cleanup path.
func (rt *RealtimeThinker) Run() {
	defer func() {
		rt.cancel()
		rt.replaceSession(nil)
		if rt.onStop != nil {
			rt.onStop()
		}
	}()

	logMsg("REALTIME", fmt.Sprintf("[%s] session up, model=%s voice=%s", rt.threadID, rt.opts.Model, rt.voice))
	rt.emit("realtime.session_started", map[string]any{
		"model": rt.opts.Model, "voice": rt.voice, "provider": rt.provider.Name(),
	})
	reconnectDelay := realtimeReconnectMinDelay
	for {
		if rt.currentSession() == nil {
			if err := rt.openSession(true); err != nil {
				rt.emit("realtime.reconnect", map[string]any{"success": false, "error": err.Error(), "delay_ms": reconnectDelay.Milliseconds()})
				select {
				case <-rt.quit:
					return
				case <-rt.ctx.Done():
					return
				case <-time.After(reconnectDelay):
				}
				reconnectDelay = min(realtimeReconnectMaxDelay, reconnectDelay*2)
				continue
			}
			rt.emit("realtime.reconnect", map[string]any{"success": true})
			reconnectDelay = realtimeReconnectMinDelay
		}

		session := rt.currentSession()
		events := session.Events()
		select {
		case event, ok := <-events:
			if !ok {
				logMsg("REALTIME", fmt.Sprintf("[%s] provider session closed; renewing", rt.threadID))
				rt.replaceSession(nil)
				continue
			}
			rt.handleSessionEvent(event)

		case audio, ok := <-rt.audioIn:
			if !ok {
				rt.audioIn = nil
				continue
			}
			if rt.paused {
				continue
			}
			if err := session.SendAudio(audio); err != nil {
				logMsg("REALTIME", fmt.Sprintf("[%s] send audio: %v", rt.threadID, err))
			}

		case event := <-rt.sub.C:
			rt.handleBusEvent(event)

		case paused := <-rt.pause:
			rt.paused = paused
			if paused {
				_ = session.Interrupt()
			}
			rt.publishRuntimeStatus()

		case <-rt.quit:
			return
		case <-rt.ctx.Done():
			return
		}
	}
}

func (rt *RealtimeThinker) appendTranscript(role, transcript string) {
	message := Message{Role: role, Content: transcript}
	rt.transcriptMu.Lock()
	rt.messages = append(rt.messages, message)
	rt.publishContextStatus()
	rt.transcriptMu.Unlock()
	if rt.Thinker.session != nil {
		_ = rt.Thinker.session.AppendMessage(message, rt.iteration, TokenUsage{})
	}
	rt.emit("realtime."+role, map[string]any{"text": transcript})
}

func (rt *RealtimeThinker) handleSessionEvent(event RealtimeEvent) {
	switch event.Type {
	case RealtimeEventAudioOut:
		if rt.paused {
			return
		}
		if rt.audioOut != nil {
			select {
			case rt.audioOut <- event.Audio:
				rt.outputMu.Lock()
				if event.ItemID != "" && event.ItemID != rt.outputItemID {
					rt.outputItemID, rt.outputBytes = event.ItemID, 0
				}
				rt.outputBytes += len(event.Audio)
				rt.outputMu.Unlock()
			default:
				rt.emit("realtime.audio_drop", map[string]any{"direction": "output", "bytes": len(event.Audio), "reason": "consumer_backpressure"})
			}
		}

	case RealtimeEventSpeechStarted:
		rt.outputMu.Lock()
		itemID := rt.outputItemID
		playedMS := rt.outputBytes * 1000 / realtimePCMBytesPerSecond
		rt.outputItemID, rt.outputBytes = "", 0
		rt.outputMu.Unlock()
		if session := rt.currentSession(); session != nil && itemID != "" {
			_ = session.Truncate(itemID, playedMS)
		}
		if rt.audioControl != nil {
			select {
			case rt.audioControl <- "interrupt":
			default:
				rt.emit("realtime.control_drop", map[string]any{"control": "interrupt"})
			}
		}

	case RealtimeEventTranscriptOutput:
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.appendTranscript("assistant", event.Transcript)
		}

	case RealtimeEventTranscriptInput:
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.appendTranscript("user", event.Transcript)
		}

	case RealtimeEventToolCall:
		if rt.paused {
			if session := rt.currentSession(); session != nil {
				rt.beginToolCall(event, session)
				rt.submitToolResult(session, event.ToolCallID, "thread is paused", true)
			}
			return
		}
		rt.dispatchToolCall(event)

	case RealtimeEventResponseDone:
		rt.completeToolResponse(event.ResponseID)
		cost := calculateCostForRealtimeProvider(rt.provider, rt.opts.Model, event.Usage)
		rt.emit("realtime.usage", map[string]any{
			"model": rt.opts.Model, "cost": cost,
			"total_tokens":         event.Usage.TotalTokens,
			"text_input_tokens":    event.Usage.TextInputTokens,
			"text_cached_tokens":   event.Usage.TextCachedTokens,
			"text_output_tokens":   event.Usage.TextOutputTokens,
			"audio_input_tokens":   event.Usage.AudioInputTokens,
			"audio_cached_tokens":  event.Usage.AudioCachedTokens,
			"audio_output_tokens":  event.Usage.AudioOutputTokens,
			"audio_input_seconds":  event.Usage.AudioInputSeconds,
			"audio_output_seconds": event.Usage.AudioOutputSeconds,
			"text_input_messages":  event.Usage.TextInputMessages,
		})

	case RealtimeEventError:
		logMsg("REALTIME", fmt.Sprintf("[%s] session error: %v", rt.threadID, event.Err))
		rt.emit("realtime.error", map[string]any{"error": fmt.Sprint(event.Err)})

	case RealtimeEventSessionEnded:
		rt.emit("realtime.session_ended", map[string]any{"dropped_audio_events": event.DroppedAudio})
	}
}

// dispatchToolCall uses the exact same handler and execution-control gates as
// normal model turns. This is what makes core send/done/evolve/pace work and
// keeps external tools subject to the same authorization and telemetry path.
func (rt *RealtimeThinker) dispatchToolCall(event RealtimeEvent) {
	session := rt.currentSession()
	if session == nil {
		return
	}
	rt.beginToolCall(event, session)
	if event.ToolName == "interrupt" {
		if err := session.Interrupt(); err != nil {
			rt.submitToolResult(session, event.ToolCallID, "interrupt failed: "+err.Error(), true)
			return
		}
		rt.submitToolResult(session, event.ToolCallID, "interrupted", false)
		return
	}

	call := toolCall{
		Name: event.ToolName, Args: flattenJSONArgs(event.ToolArgs),
		Raw: event.ToolName, NativeID: event.ToolCallID,
	}
	if rt.toolAllowlist != nil && !rt.toolAllowlist[call.Name] {
		rt.submitToolResult(session, call.NativeID, "tool is not available to this thread", true)
		return
	}
	if !rt.executionGate(ExecutionPhaseToolBefore, ExecutionGate{
		Tool: call.Name, CallID: call.NativeID,
		Summary: toolSummary(call.Name, call.Args), Args: call.Args,
	}) {
		return
	}
	callMessage := Message{Role: "assistant", ToolCalls: []NativeToolCall{{
		ID: call.NativeID, Name: call.Name, Args: call.Args,
	}}}
	rt.transcriptMu.Lock()
	rt.messages = append(rt.messages, callMessage)
	rt.transcriptMu.Unlock()
	if rt.Thinker.session != nil {
		_ = rt.Thinker.session.AppendMessage(callMessage, rt.iteration, TokenUsage{})
	}
	if rt.handleTools == nil {
		rt.submitToolResult(session, call.NativeID, "tool handler unavailable", true)
		return
	}
	_, _, results := rt.handleTools(rt.Thinker, []toolCall{call}, nil)
	if len(results) == 0 {
		// External tools complete asynchronously and return through the bus.
		if rt.registry != nil {
			if def := rt.registry.Get(call.Name); def != nil && !def.Core {
				return
			}
		}
		rt.submitToolResult(session, call.NativeID, "tool is not available to this thread", true)
		return
	}

	message := rt.archiveToolResultMessage(Message{Role: "user", ToolResults: results})
	rt.transcriptMu.Lock()
	rt.messages = append(rt.messages, message)
	rt.publishContextStatus()
	rt.transcriptMu.Unlock()
	if rt.Thinker.session != nil {
		_ = rt.Thinker.session.AppendMessage(message, rt.iteration, TokenUsage{})
	}
	for _, result := range results {
		if !rt.executionGate(ExecutionPhaseToolAfter, ExecutionGate{
			Tool: call.Name, CallID: result.CallID,
			Summary: call.Name + " result ready", Result: result.Content,
		}) {
			return
		}
		isError := result.IsError || strings.HasPrefix(strings.ToLower(strings.TrimSpace(result.Content)), "error:")
		rt.submitToolResult(session, result.CallID, result.Content, isError)
	}
	rt.refreshConfiguration()
}

func flattenJSONArgs(raw string) map[string]string {
	args := map[string]string{}
	if strings.TrimSpace(raw) == "" {
		return args
	}
	var object map[string]any
	if err := json.Unmarshal([]byte(raw), &object); err != nil {
		return args
	}
	for key, value := range object {
		if text, ok := value.(string); ok {
			args[key] = text
			continue
		}
		encoded, _ := json.Marshal(value)
		args[key] = string(encoded)
	}
	return args
}

func (rt *RealtimeThinker) handleBusEvent(event Event) {
	if event.Type != EventInbox {
		return
	}
	session := rt.currentSession()
	if session == nil {
		return
	}
	if event.ToolResult != nil {
		message := rt.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{*event.ToolResult}})
		rt.transcriptMu.Lock()
		rt.messages = append(rt.messages, message)
		rt.publishContextStatus()
		rt.transcriptMu.Unlock()
		if rt.Thinker.session != nil {
			_ = rt.Thinker.session.AppendMessage(message, rt.iteration, TokenUsage{})
		}
		rt.submitToolResult(session, event.ToolResult.CallID, event.ToolResult.Content, event.ToolResult.IsError)
		return
	}
	note := event.Text
	if event.From != "" {
		note = fmt.Sprintf("[from:%s] %s", event.From, event.Text)
	}
	if strings.TrimSpace(note) == "" {
		return
	}
	if err := session.SendText("user", note); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] inject text: %v", rt.threadID, err))
	}
}

func (rt *RealtimeThinker) requestTextResponse(message string) error {
	session := rt.currentSession()
	if session == nil {
		return errors.New("realtime session unavailable")
	}
	if err := session.SendText("user", message); err != nil {
		return err
	}
	return session.RequestResponse()
}
