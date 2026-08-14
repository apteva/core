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
	realtimeReconnectMinDelay  = time.Second
	realtimeReconnectMaxDelay  = 30 * time.Second
	realtimeRestoreMessages    = 32
	realtimePCMBytesPerSecond  = 24000 * 2
	realtimeToolMarkupTailSize = 2048
)

const realtimeToolMarkupRecoveryPrompt = `[INTERNAL RECOVERY]
Your previous response exposed textual tool-call syntax. It was suppressed and no action was taken from that text. If the action is still needed, invoke the registered tool through a structured tool call. Otherwise, briefly explain that the action could not be completed. Never speak tool names, call syntax, JSON, identifiers, or arguments, and never claim success without a successful tool result.`

var ErrRealtimeConfigurationRestartRequired = errors.New("active realtime configuration change requires an explicit restart")

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
	audioOut     chan RealtimeAudioFrame
	audioControl chan<- string

	transcriptMu    sync.Mutex
	outputMu        sync.Mutex
	outputItemID    string
	outputBytes     int
	playedItemID    string
	playedMS        int
	playbackTracked bool
	interruptedItem string
	suppressOutput  bool
	responseMu      sync.Mutex
	responseActive  bool
	responsePending bool
	stateMu         sync.Mutex
	state           string
	statePhase      string

	lifecycleMu                 sync.Mutex
	sessionGeneration           int
	bridgeConnected             bool
	bridgeConnectedAt           time.Time
	initialMessage              string
	greetingRequestedGeneration int
	assistantAudioEmitted       bool
	firstAudioEmitted           bool
	terminalReason              string
	pendingReconnectReason      string
	pendingReconnectPlanned     bool

	toolBatchMu     sync.Mutex
	toolBatches     map[string]*realtimeToolBatch
	toolCallBatches map[string]string

	toolMarkupMu           sync.Mutex
	toolMarkupTails        map[string]string
	toolMarkupSuppressed   map[string]bool
	toolMarkupRecoveryUsed bool
}

type realtimeToolBatch struct {
	session      RealtimeSession
	pending      int
	responseDone bool
}

func realtimeNativeToolsFor(thinker *Thinker, allowlist map[string]bool, record bool) []NativeTool {
	var tools []NativeTool
	if thinker.registry != nil {
		tools = thinker.registry.NativeTools(thinker.authorizedToolAllowlist(allowlist), thinker.authorizedActiveTools(thinker.activeTools), thinker.systemThread)
	}
	tools = append(tools, NativeTool{
		Name:        "interrupt",
		Description: "Cancel your own in-flight speech immediately when the remaining utterance is stale. Takes no arguments.",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{},
		},
	})
	if record {
		thinker.recordPresentedTools(tools)
	}
	return tools
}

func realtimeNativeTools(thinker *Thinker) []NativeTool {
	return realtimeNativeToolsFor(thinker, thinker.toolAllowlist, true)
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
	audioOut chan RealtimeAudioFrame,
	audioControl chan<- string,
	turnDetection ...RealtimeTurnDetectionConfig,
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
	var turnDetectionConfig RealtimeTurnDetectionConfig
	if len(turnDetection) > 0 {
		turnDetectionConfig = turnDetection[0]
	}
	runCtx, cancel := context.WithCancel(ctx)
	rt := &RealtimeThinker{
		Thinker: thinker, provider: provider, voice: voice,
		ctx: runCtx, cancel: cancel,
		audioIn: audioIn, audioOut: audioOut, audioControl: audioControl,
		toolBatches: map[string]*realtimeToolBatch{}, toolCallBatches: map[string]string{},
		toolMarkupTails: map[string]string{}, toolMarkupSuppressed: map[string]bool{},
		terminalReason: "server_shutdown",
	}
	reasoning := thinker.agentReasoning.String()
	if reasoning == "" || reasoning == "auto" {
		if defaults, ok := provider.(realtimeReasoningDefaultProvider); ok {
			if preferred := strings.TrimSpace(defaults.DefaultRealtimeReasoning()); preferred != "" {
				reasoning = preferred
			}
		}
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
		Reasoning:          reasoning,
		SafetyIdentifier:   realtimeSafetyIdentifier(thinker.threadID),
		TranscribeInput:    true,
		TranscriptionModel: provider.DefaultTranscriptionModel(),
		TurnDetection:      turnDetectionConfig,
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
	audioOut chan RealtimeAudioFrame,
	audioControl chan<- string,
	turnDetection ...RealtimeTurnDetectionConfig,
) (*RealtimeThinker, error) {
	rt := newRealtimeThinker(ctx, thinker, provider, voice, audioIn, audioOut, audioControl, turnDetection...)
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
		rt.responseMu.Lock()
		rt.responseActive = false
		rt.responsePending = false
		rt.responseMu.Unlock()
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
		rt.setConversationState("thinking", RealtimeEvent{})
		_ = rt.requestProviderResponse(continueSession)
	}
}

func (rt *RealtimeThinker) completeToolResponse(responseID string) bool {
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
	hadToolBatch := batch != nil
	if hadToolBatch {
		batch.responseDone = true
		if batch.pending == 0 {
			continueSession = batch.session
			delete(rt.toolBatches, batchID)
		}
	}
	rt.toolBatchMu.Unlock()
	if continueSession != nil {
		rt.setConversationState("thinking", RealtimeEvent{ResponseID: responseID})
		_ = rt.requestProviderResponse(continueSession)
	}
	return hadToolBatch
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
	rt.lifecycleMu.Lock()
	rt.sessionGeneration++
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.emit("realtime.session_opened", map[string]any{
		"generation": generation, "restored": restore,
	})
	rt.requestInitialMessageIfNeeded()
	return nil
}

func (rt *RealtimeThinker) refreshConfiguration() {
	session := rt.currentSession()
	if session == nil {
		return
	}
	instructions, tools := rt.configurationSnapshot()
	disposition := rt.previewConfigurationUpdate(session, instructions, tools)
	if disposition == RealtimeConfigurationUnchanged {
		return
	}
	if disposition == RealtimeConfigurationRestartRequired {
		rt.lifecycleMu.Lock()
		rt.pendingReconnectReason = "self_configuration_update"
		rt.pendingReconnectPlanned = true
		rt.lifecycleMu.Unlock()
		rt.emit("realtime.reconnect_planned", map[string]any{
			"reason": "self_configuration_update", "planned": true,
		})
	}
	if err := session.UpdateConfiguration(instructions, tools); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] update configuration: %v", rt.threadID, err))
	}
}

func (rt *RealtimeThinker) previewConfigurationUpdate(session RealtimeSession, instructions string, tools []NativeTool) RealtimeConfigurationDisposition {
	if previewer, ok := session.(RealtimeConfigurationPreviewer); ok {
		return previewer.PreviewConfigurationUpdate(instructions, tools)
	}
	return RealtimeConfigurationAppliedLive
}

func (rt *RealtimeThinker) configurationDisposition(instructions string, tools []NativeTool) RealtimeConfigurationDisposition {
	session := rt.currentSession()
	if session == nil {
		return RealtimeConfigurationAppliedLive
	}
	return rt.previewConfigurationUpdate(session, instructions, tools)
}

// applyExternalConfigurationChange is used for parent/API updates after their
// state transaction commits. Immutable providers are restarted only when the
// caller explicitly permitted it (or before an audio bridge is connected).
func (rt *RealtimeThinker) applyExternalConfigurationChange(allowRestart bool, reason string) (bool, error) {
	session := rt.currentSession()
	if session == nil {
		return false, nil
	}
	instructions, tools := rt.configurationSnapshot()
	disposition := rt.previewConfigurationUpdate(session, instructions, tools)
	switch disposition {
	case RealtimeConfigurationUnchanged:
		return false, nil
	case RealtimeConfigurationRestartRequired:
		if !allowRestart {
			return false, ErrRealtimeConfigurationRestartRequired
		}
		rt.emit("realtime.reconnect_planned", map[string]any{
			"reason": reason, "planned": true,
		})
		rt.lifecycleMu.Lock()
		rt.pendingReconnectReason = reason
		rt.pendingReconnectPlanned = true
		rt.lifecycleMu.Unlock()
		rt.replaceSession(nil)
		return true, nil
	default:
		if err := session.UpdateConfiguration(instructions, tools); err != nil {
			return false, err
		}
		return false, nil
	}
}

func (rt *RealtimeThinker) setInitialMessage(message string) {
	rt.lifecycleMu.Lock()
	rt.initialMessage = strings.TrimSpace(message)
	rt.lifecycleMu.Unlock()
}

func (rt *RealtimeThinker) audioBridgeConnected() {
	rt.lifecycleMu.Lock()
	if !rt.bridgeConnected {
		rt.bridgeConnected = true
		rt.bridgeConnectedAt = time.Now()
	}
	rt.lifecycleMu.Unlock()
	rt.requestInitialMessageIfNeeded()
}

func (rt *RealtimeThinker) audioBridgeDisconnected() {
	rt.lifecycleMu.Lock()
	rt.bridgeConnected = false
	rt.lifecycleMu.Unlock()
}

func (rt *RealtimeThinker) requestInitialMessageIfNeeded() {
	session := rt.currentSession()
	if session == nil {
		return
	}
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	if generation == 0 {
		generation = 1
		rt.sessionGeneration = generation
	}
	message := rt.initialMessage
	if !rt.bridgeConnected || message == "" || rt.assistantAudioEmitted ||
		rt.greetingRequestedGeneration == generation {
		rt.lifecycleMu.Unlock()
		return
	}
	rt.greetingRequestedGeneration = generation
	rt.lifecycleMu.Unlock()
	if err := rt.requestTextResponse(message); err != nil {
		rt.lifecycleMu.Lock()
		if rt.greetingRequestedGeneration == generation {
			rt.greetingRequestedGeneration = 0
		}
		rt.lifecycleMu.Unlock()
		logMsg("REALTIME", fmt.Sprintf("[%s] initial response: %v", rt.threadID, err))
	}
}

func (rt *RealtimeThinker) markAssistantAudioEmitted() {
	rt.lifecycleMu.Lock()
	rt.assistantAudioEmitted = true
	if rt.firstAudioEmitted {
		rt.lifecycleMu.Unlock()
		return
	}
	rt.firstAudioEmitted = true
	generation := rt.sessionGeneration
	connectedAt := rt.bridgeConnectedAt
	replayed := generation > 1 && rt.greetingRequestedGeneration == generation
	rt.lifecycleMu.Unlock()
	data := map[string]any{"generation": generation, "replayed_greeting": replayed}
	if !connectedAt.IsZero() {
		data["bridge_to_first_audio_ms"] = time.Since(connectedAt).Milliseconds()
	}
	rt.emit("realtime.first_audio", data)
}

func (rt *RealtimeThinker) setTerminalReason(reason string) {
	if strings.TrimSpace(reason) == "" {
		return
	}
	rt.lifecycleMu.Lock()
	rt.terminalReason = reason
	rt.lifecycleMu.Unlock()
}

func (rt *RealtimeThinker) emit(eventType string, data map[string]any) {
	if rt.telemetry != nil {
		rt.telemetry.Emit(eventType, rt.threadID, data)
	}
}

func (rt *RealtimeThinker) setConversationState(state string, event RealtimeEvent) {
	state = strings.TrimSpace(state)
	phase := strings.TrimSpace(event.Phase)
	if state == "" {
		return
	}
	rt.stateMu.Lock()
	if rt.state == state && rt.statePhase == phase {
		rt.stateMu.Unlock()
		return
	}
	previous := rt.state
	rt.state, rt.statePhase = state, phase
	rt.stateMu.Unlock()
	data := map[string]any{"state": state}
	if previous != "" {
		data["previous_state"] = previous
	}
	if event.ResponseID != "" {
		data["response_id"] = event.ResponseID
	}
	if event.ItemID != "" {
		data["item_id"] = event.ItemID
	}
	if phase != "" {
		data["phase"] = phase
	}
	rt.emit("realtime.state", data)
}

func (rt *RealtimeThinker) requestProviderResponse(session RealtimeSession) error {
	if session == nil {
		return errors.New("realtime session unavailable")
	}
	rt.responseMu.Lock()
	if rt.responseActive {
		rt.responsePending = true
		rt.responseMu.Unlock()
		return nil
	}
	rt.responseActive = true
	rt.responseMu.Unlock()
	if err := session.RequestResponse(); err != nil {
		rt.responseMu.Lock()
		rt.responseActive = false
		rt.responseMu.Unlock()
		return err
	}
	return nil
}

func (rt *RealtimeThinker) responseStarted() {
	rt.responseMu.Lock()
	rt.responseActive = true
	rt.responseMu.Unlock()
}

func (rt *RealtimeThinker) responseFinished(session RealtimeSession) {
	rt.responseMu.Lock()
	pending := rt.responsePending
	rt.responsePending = false
	rt.responseActive = false
	rt.responseMu.Unlock()
	if pending {
		_ = rt.requestProviderResponse(session)
	}
}

func realtimeToolMarkupResponseKey(event RealtimeEvent) string {
	if strings.TrimSpace(event.ResponseID) != "" {
		return strings.TrimSpace(event.ResponseID)
	}
	if strings.TrimSpace(event.ItemID) != "" {
		return strings.TrimSpace(event.ItemID)
	}
	return "current"
}

func boundedRealtimeToolMarkupTail(text string) string {
	if len(text) <= realtimeToolMarkupTailSize {
		return text
	}
	return strings.ToValidUTF8(text[len(text)-realtimeToolMarkupTailSize:], "")
}

func detectRealtimeToolMarkup(text string, tools []NativeTool) (toolName, pattern string, found bool) {
	lower := strings.ToLower(text)
	if marker := strings.Index(lower, "callto:"); marker >= 0 {
		after := lower[marker+len("callto:"):]
		for _, tool := range tools {
			name := strings.ToLower(strings.TrimSpace(tool.Name))
			if name != "" && strings.Contains(after, name) {
				return tool.Name, "callto", true
			}
		}
		return "", "callto", true
	}
	for _, tool := range tools {
		name := strings.ToLower(strings.TrimSpace(tool.Name))
		if name == "" {
			continue
		}
		for remaining := lower; ; {
			index := strings.Index(remaining, name)
			if index < 0 {
				break
			}
			after := strings.TrimSpace(remaining[index+len(name):])
			if strings.HasPrefix(after, "{") || strings.HasPrefix(after, "(") || strings.HasPrefix(after, "[") {
				return tool.Name, "registered_tool_with_arguments", true
			}
			remaining = remaining[index+len(name):]
		}
	}
	return "", "", false
}

func (rt *RealtimeThinker) realtimeToolSchemas() []NativeTool {
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	return append([]NativeTool(nil), rt.opts.Tools...)
}

func (rt *RealtimeThinker) resetToolMarkupTurn() {
	rt.toolMarkupMu.Lock()
	rt.toolMarkupTails = map[string]string{}
	rt.toolMarkupSuppressed = map[string]bool{}
	rt.toolMarkupRecoveryUsed = false
	rt.toolMarkupMu.Unlock()
}

func (rt *RealtimeThinker) finishToolMarkupResponse(responseID string) {
	key := strings.TrimSpace(responseID)
	if key == "" {
		key = "current"
	}
	rt.toolMarkupMu.Lock()
	delete(rt.toolMarkupTails, key)
	delete(rt.toolMarkupSuppressed, key)
	rt.toolMarkupMu.Unlock()
}

func (rt *RealtimeThinker) toolMarkupResponseSuppressed(event RealtimeEvent) bool {
	key := realtimeToolMarkupResponseKey(event)
	rt.toolMarkupMu.Lock()
	suppressed := rt.toolMarkupSuppressed[key]
	rt.toolMarkupMu.Unlock()
	return suppressed
}

func (rt *RealtimeThinker) queueToolMarkupRecovery(session RealtimeSession) error {
	if session == nil {
		return errors.New("realtime session unavailable")
	}
	if err := session.SendText("system", realtimeToolMarkupRecoveryPrompt); err != nil {
		return err
	}
	// The current provider response must finish before the corrective response
	// starts. Reusing responsePending preserves the normal single-response
	// ordering and prevents a recovery request from racing in-flight audio.
	rt.responseMu.Lock()
	rt.responsePending = true
	rt.responseMu.Unlock()
	return nil
}

func (rt *RealtimeThinker) suppressLeakedToolMarkup(event RealtimeEvent) bool {
	if strings.TrimSpace(event.Transcript) == "" {
		return false
	}
	key := realtimeToolMarkupResponseKey(event)
	tools := rt.realtimeToolSchemas()

	rt.toolMarkupMu.Lock()
	if rt.toolMarkupSuppressed[key] {
		rt.toolMarkupMu.Unlock()
		return true
	}
	combined := rt.toolMarkupTails[key] + event.Transcript
	rt.toolMarkupTails[key] = boundedRealtimeToolMarkupTail(combined)
	toolName, pattern, leaked := detectRealtimeToolMarkup(combined, tools)
	if !leaked {
		if event.Final {
			delete(rt.toolMarkupTails, key)
			rt.toolMarkupRecoveryUsed = false
		}
		rt.toolMarkupMu.Unlock()
		return false
	}
	rt.toolMarkupSuppressed[key] = true
	delete(rt.toolMarkupTails, key)
	retry := !rt.toolMarkupRecoveryUsed
	if retry {
		rt.toolMarkupRecoveryUsed = true
	}
	rt.toolMarkupMu.Unlock()

	interrupted := rt.interruptPlayback("provider_tool_markup_leaked", "", true, true)
	recoveryRequested := false
	if retry {
		if err := rt.queueToolMarkupRecovery(rt.currentSession()); err != nil {
			rt.emit("realtime.tool_markup_recovery_error", map[string]any{
				"provider": rt.provider.Name(), "response_id": event.ResponseID, "error": err.Error(),
			})
		} else {
			recoveryRequested = true
		}
	}
	sum := sha256.Sum256([]byte(combined))
	rt.emit("realtime.tool_markup_leaked", map[string]any{
		"provider": rt.provider.Name(), "response_id": event.ResponseID, "item_id": event.ItemID,
		"tool": toolName, "pattern": pattern, "transcript_bytes": len(combined),
		"transcript_sha256": hex.EncodeToString(sum[:]), "audio_interrupted": interrupted,
		"recovery_requested": recoveryRequested,
	})
	return true
}

func (rt *RealtimeThinker) acknowledgePlayback(itemID string, audioEndMS int) {
	if itemID == "" || audioEndMS < 0 {
		return
	}
	rt.outputMu.Lock()
	defer rt.outputMu.Unlock()
	if itemID != rt.outputItemID {
		return
	}
	generatedMS := rt.outputBytes * 1000 / realtimePCMBytesPerSecond
	if audioEndMS > generatedMS {
		audioEndMS = generatedMS
	}
	if itemID != rt.playedItemID {
		rt.playedItemID, rt.playedMS = itemID, 0
	}
	if audioEndMS > rt.playedMS {
		rt.playedMS = audioEndMS
	}
	rt.playbackTracked = true
}

func (rt *RealtimeThinker) discardQueuedOutput() (frames, bytes int) {
	if rt.audioOut == nil {
		return 0, 0
	}
	for {
		select {
		case frame, ok := <-rt.audioOut:
			if !ok {
				return frames, bytes
			}
			frames++
			bytes += len(frame.Audio)
		default:
			return frames, bytes
		}
	}
}

func (rt *RealtimeThinker) interruptPlayback(reason, expectedItemID string, cancelProvider, requirePlaybackAck bool) bool {
	rt.outputMu.Lock()
	itemID := rt.outputItemID
	if expectedItemID != "" && itemID != expectedItemID {
		rt.outputMu.Unlock()
		return false
	}
	generatedMS := rt.outputBytes * 1000 / realtimePCMBytesPerSecond
	playedMS := generatedMS
	if rt.playbackTracked && rt.playedItemID == itemID {
		playedMS = rt.playedMS
	} else if requirePlaybackAck {
		playedMS = 0
	}
	rt.outputItemID, rt.outputBytes = "", 0
	rt.playedItemID, rt.playedMS = "", 0
	rt.playbackTracked = false
	rt.interruptedItem = itemID
	rt.suppressOutput = itemID != ""
	rt.outputMu.Unlock()

	drainedFrames, drainedBytes := rt.discardQueuedOutput()
	session := rt.currentSession()
	if cancelProvider && session != nil {
		if err := session.Interrupt(); err != nil {
			rt.emit("realtime.interrupt_error", map[string]any{"reason": reason, "error": err.Error()})
		}
	}
	if session != nil && itemID != "" {
		if err := session.Truncate(itemID, playedMS); err != nil {
			rt.emit("realtime.truncate_error", map[string]any{"reason": reason, "item_id": itemID, "error": err.Error()})
		}
	}
	if itemID != "" && rt.audioControl != nil {
		select {
		case rt.audioControl <- "interrupt":
		case <-rt.ctx.Done():
		case <-time.After(100 * time.Millisecond):
			rt.emit("realtime.control_overflow", map[string]any{"control": "interrupt", "reason": reason})
		}
	}
	rt.emit("realtime.playback_interrupted", map[string]any{
		"reason": reason, "item_id": itemID, "generated_ms": generatedMS, "played_ms": playedMS,
		"drained_frames": drainedFrames, "drained_bytes": drainedBytes, "provider_cancelled": cancelProvider,
	})
	return itemID != "" || drainedFrames > 0
}

func (rt *RealtimeThinker) rendererSpeechStarted() {
	rt.setConversationState("listening", RealtimeEvent{})
	rt.interruptPlayback("renderer_speech_started", "", true, true)
}

func (rt *RealtimeThinker) rendererPlaybackOverflow(itemID string) {
	if rt.interruptPlayback("renderer_overflow", itemID, true, true) {
		rt.setConversationState("listening", RealtimeEvent{})
	}
}

// Run survives normal provider-enforced session endings. Only explicit thread
// stop/cancellation ends the worker and invokes the normal cleanup path.
func (rt *RealtimeThinker) Run() {
	defer func() {
		rt.lifecycleMu.Lock()
		terminalReason := rt.terminalReason
		generation := rt.sessionGeneration
		rt.lifecycleMu.Unlock()
		rt.emit("realtime.thread_ended", map[string]any{
			"reason": terminalReason, "generation": generation,
		})
		rt.cancel()
		rt.replaceSession(nil)
		if rt.onStop != nil {
			rt.onStop()
		}
	}()

	logMsg("REALTIME", fmt.Sprintf("[%s] session up, model=%s voice=%s", rt.threadID, rt.opts.Model, rt.voice))
	rt.emit("realtime.session_started", map[string]any{
		"model": rt.opts.Model, "voice": rt.voice, "provider": rt.provider.Name(),
		"turn_detection": rt.opts.TurnDetection.telemetryData(),
	})
	rt.setConversationState("listening", RealtimeEvent{})
	reconnectDelay := realtimeReconnectMinDelay
	for {
		if rt.currentSession() == nil {
			if err := rt.openSession(true); err != nil {
				rt.lifecycleMu.Lock()
				nextGeneration := rt.sessionGeneration + 1
				reconnectReason := rt.pendingReconnectReason
				reconnectPlanned := rt.pendingReconnectPlanned
				rt.lifecycleMu.Unlock()
				if reconnectReason == "" {
					reconnectReason = "provider_session_closed"
				}
				rt.emit("realtime.reconnect", map[string]any{
					"success": false, "error": err.Error(), "delay_ms": reconnectDelay.Milliseconds(),
					"reason": reconnectReason, "planned": reconnectPlanned, "generation": nextGeneration,
				})
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
			rt.lifecycleMu.Lock()
			generation := rt.sessionGeneration
			reconnectReason := rt.pendingReconnectReason
			reconnectPlanned := rt.pendingReconnectPlanned
			rt.pendingReconnectReason = ""
			rt.pendingReconnectPlanned = false
			rt.lifecycleMu.Unlock()
			if reconnectReason == "" {
				reconnectReason = "provider_session_closed"
			}
			rt.emit("realtime.reconnect", map[string]any{
				"success": true, "turn_detection": rt.opts.TurnDetection.telemetryData(),
				"reason": reconnectReason, "planned": reconnectPlanned, "generation": generation,
			})
			reconnectDelay = realtimeReconnectMinDelay
		}

		session := rt.currentSession()
		events := session.Events()
		select {
		case event, ok := <-events:
			if !ok {
				logMsg("REALTIME", fmt.Sprintf("[%s] provider session closed; renewing", rt.threadID))
				rt.lifecycleMu.Lock()
				generation := rt.sessionGeneration
				if rt.pendingReconnectReason == "" {
					rt.pendingReconnectReason = "provider_session_closed"
					rt.pendingReconnectPlanned = false
				}
				reconnectReason := rt.pendingReconnectReason
				rt.lifecycleMu.Unlock()
				rt.emit("realtime.session_closed", map[string]any{
					"reason": reconnectReason, "generation": generation,
				})
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
		if rt.toolMarkupResponseSuppressed(event) {
			rt.emit("realtime.audio_drop", map[string]any{
				"direction": "output", "bytes": len(event.Audio), "reason": "provider_tool_markup_leaked",
				"response_id": event.ResponseID, "item_id": event.ItemID,
			})
			return
		}
		if rt.audioOut != nil {
			rt.outputMu.Lock()
			if rt.suppressOutput && (event.ItemID == "" || event.ItemID == rt.interruptedItem) {
				rt.outputMu.Unlock()
				rt.emit("realtime.audio_drop", map[string]any{"direction": "output", "bytes": len(event.Audio), "reason": "interrupted_item"})
				return
			}
			if rt.suppressOutput && event.ItemID != "" && event.ItemID != rt.interruptedItem {
				rt.suppressOutput = false
				rt.interruptedItem = ""
			}
			if event.ItemID != "" && event.ItemID != rt.outputItemID {
				rt.outputItemID, rt.outputBytes = event.ItemID, 0
				rt.playedItemID, rt.playedMS = event.ItemID, 0
				rt.playbackTracked = false
			}
			endMS := (rt.outputBytes + len(event.Audio)) * 1000 / realtimePCMBytesPerSecond
			frame := RealtimeAudioFrame{Audio: event.Audio, ResponseID: event.ResponseID, ItemID: event.ItemID, AudioEndMS: endMS}
			rt.outputBytes += len(event.Audio)
			rt.outputMu.Unlock()
			select {
			case rt.audioOut <- frame:
				rt.markAssistantAudioEmitted()
			default:
				rt.emit("realtime.audio_overflow", map[string]any{"direction": "output", "bytes": len(event.Audio), "reason": "consumer_backpressure"})
				rt.interruptPlayback("core_output_overflow", event.ItemID, true, true)
				return
			}
		}
		rt.setConversationState("speaking", event)

	case RealtimeEventSpeechStarted:
		rt.setConversationState("listening", event)
		rt.interruptPlayback("provider_speech_started", "", false, false)

	case RealtimeEventTranscriptOutput:
		if rt.suppressLeakedToolMarkup(event) {
			return
		}
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.appendTranscript("assistant", event.Transcript)
		}

	case RealtimeEventTranscriptInput:
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.resetToolMarkupTurn()
			rt.appendTranscript("user", event.Transcript)
			rt.setConversationState("thinking", event)
		}

	case RealtimeEventResponseStarted:
		rt.outputMu.Lock()
		rt.suppressOutput = false
		rt.interruptedItem = ""
		rt.outputMu.Unlock()
		rt.responseStarted()
		rt.setConversationState("thinking", event)

	case RealtimeEventToolCall:
		if rt.paused {
			if session := rt.currentSession(); session != nil {
				rt.beginToolCall(event, session)
				rt.submitToolResult(session, event.ToolCallID, "thread is paused", true)
			}
			return
		}
		rt.setConversationState("working", event)
		rt.dispatchToolCall(event)

	case RealtimeEventResponseDone:
		hadToolBatch := rt.completeToolResponse(event.ResponseID)
		rt.responseFinished(rt.currentSession())
		rt.finishToolMarkupResponse(event.ResponseID)
		if !hadToolBatch {
			rt.setConversationState("listening", event)
		}
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
		rt.setConversationState("disconnected", event)
		rt.lifecycleMu.Lock()
		generation := rt.sessionGeneration
		rt.lifecycleMu.Unlock()
		rt.emit("realtime.session_ended", map[string]any{
			"dropped_audio_events": event.DroppedAudio,
			"reason":               "provider_session_ended",
			"generation":           generation,
		})
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
	if !rt.modelToolCallable(call.Name, rt.toolAllowlist) {
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
	if err := rt.requestTextResponse(note); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] inject text: %v", rt.threadID, err))
		return
	}
	if event.ID != "" {
		message := Message{Role: "user", Content: note, EventIDs: []string{event.ID}}
		rt.transcriptMu.Lock()
		rt.messages = append(rt.messages, message)
		rt.publishContextStatus()
		rt.transcriptMu.Unlock()
		persisted := rt.Thinker.session == nil
		if rt.Thinker.session != nil {
			if err := rt.Thinker.session.AppendMessage(message, rt.iteration, TokenUsage{}); err != nil {
				logMsg("SESSION", fmt.Sprintf("[%s] persist realtime inbox event: %v", rt.threadID, err))
			} else {
				persisted = true
			}
		}
		if persisted && rt.Thinker.ackInboxEvents != nil {
			if err := rt.Thinker.ackInboxEvents([]string{event.ID}); err != nil {
				logMsg("SESSION", fmt.Sprintf("[%s] acknowledge realtime inbox event: %v", rt.threadID, err))
			}
		}
	}
}

func (rt *RealtimeThinker) requestTextResponse(message string) error {
	session := rt.currentSession()
	if session == nil {
		return errors.New("realtime session unavailable")
	}
	rt.setConversationState("thinking", RealtimeEvent{})
	if err := session.SendText("user", message); err != nil {
		return err
	}
	return rt.requestProviderResponse(session)
}
