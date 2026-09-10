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
Your previous speech was suppressed because it exposed internal material. Answer the caller briefly using the verified results already available. Do not repeat completed actions or read these instructions aloud. If no answer is ready, wait silently. Use structured tool calls only for work that is still required.`

var ErrRealtimeConfigurationRestartRequired = errors.New("active realtime configuration change requires an explicit restart")

// RealtimeThinker is the event-driven counterpart to Thinker. It deliberately
// reuses Thinker's prompt, tool handler, execution gates, durable Session, bus,
// and telemetry instead of maintaining a second, weaker agent runtime.
type RealtimeThinker struct {
	*Thinker

	catalogChanges <-chan struct{}
	provider       RealtimeProvider
	voice          string
	opts           RealtimeSessionOpts
	// currentTimeContext is fixed for one provider session so ordinary
	// configuration comparisons cannot churn or reconnect live audio. It is
	// refreshed immediately before each new provider session opens.
	currentTimeContext string

	ctx    context.Context
	cancel context.CancelFunc

	sessionMu sync.RWMutex
	rtSession RealtimeSession
	recovery  realtimeRecovery

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
	names        map[string]bool
}

func realtimeNativeToolsFor(thinker *Thinker, allowlist, scopes map[string]bool, record bool) []NativeTool {
	var tools []NativeTool
	var definitions map[string]*ToolDef
	if thinker.registry != nil {
		tools, definitions, _ = thinker.visibleNativeToolSnapshot(allowlist, scopes)
	}
	tools = append(tools, NativeTool{
		Name:        "interrupt",
		Description: "Cancel your own in-flight speech immediately when the remaining utterance is stale. Takes no arguments.",
		Parameters: map[string]any{
			"type": "object", "properties": map[string]any{},
		},
	})
	if record {
		thinker.recordPresentedTools(tools, definitions)
	}
	return tools
}

func realtimeNativeTools(thinker *Thinker) []NativeTool {
	return realtimeNativeToolsFor(thinker, thinker.toolAllowlist, thinker.toolMCPScopes, true)
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
	thinker.realtimeMode = true

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
		catalogChanges:     thinker.toolIndex.Changes(),
		currentTimeContext: renderCurrentTimeContext(time.Now().UTC().Format(time.RFC3339)),
		ctx:                runCtx, cancel: cancel,
		audioIn: audioIn, audioOut: audioOut, audioControl: audioControl,
		toolBatches: map[string]*realtimeToolBatch{}, toolCallBatches: map[string]string{},
		toolMarkupTails: map[string]string{}, toolMarkupSuppressed: map[string]bool{},
		terminalReason: "server_shutdown",
	}
	rt.recovery.wake = make(chan struct{}, 1)
	rt.recovery.toolNames = map[string]bool{}
	rt.recovery.blockedTools = map[string]bool{}
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
	base := rt.directive
	if len(rt.messages) > 0 && rt.messages[0].Role == "system" {
		base = rt.messages[0].Content
	}
	if strings.TrimSpace(rt.currentTimeContext) == "" {
		return base
	}
	return strings.TrimSpace(base) + "\n\n" + rt.currentTimeContext
}

func (rt *RealtimeThinker) refreshCurrentTimeContext() {
	rt.transcriptMu.Lock()
	rt.currentTimeContext = renderCurrentTimeContext(time.Now().UTC().Format(time.RFC3339))
	rt.transcriptMu.Unlock()
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
		// In-flight tools keep their original session ownership until their
		// result arrives. Recovery consumes those outcomes before reopening.
		_ = previous.Close()
	}
	rt.wakeRecovery()
}

func (rt *RealtimeThinker) beginToolCall(event RealtimeEvent, session RealtimeSession) {
	rt.recovery.Lock()
	if event.ToolName != "" {
		rt.recovery.toolNames[event.ToolName] = true
	}
	rt.recovery.Unlock()
	batchID := event.ResponseID
	rt.toolBatchMu.Lock()
	batch := rt.toolBatches[batchID]
	if batch == nil {
		batch = &realtimeToolBatch{session: session, names: map[string]bool{}}
		rt.toolBatches[batchID] = batch
	}
	batch.pending++
	if event.ToolName != "" {
		batch.names[event.ToolName] = true
	}
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
			if batch.pending == 0 && batch.session != rt.currentSession() {
				delete(rt.toolBatches, batchID)
			}
		}
	}
	rt.toolBatchMu.Unlock()
	rt.wakeRecovery()
	if continueSession != nil && continueSession == rt.currentSession() {
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
	if continueSession != nil && continueSession == rt.currentSession() {
		rt.setConversationState("thinking", RealtimeEvent{ResponseID: responseID})
		_ = rt.requestProviderResponse(continueSession)
	}
	return hadToolBatch
}

func (rt *RealtimeThinker) submitToolResult(session RealtimeSession, callID, result string, isError bool) {
	rt.toolBatchMu.Lock()
	if batchID, ok := rt.toolCallBatches[callID]; ok {
		if batch := rt.toolBatches[batchID]; batch != nil {
			session = batch.session
		}
	}
	rt.toolBatchMu.Unlock()
	if session == nil || session != rt.currentSession() {
		rt.completeToolCall(callID)
		rt.recovery.Lock()
		rt.recovery.want = true
		rt.recovery.resume = nil
		rt.recovery.Unlock()
		rt.wakeRecovery()
		return
	}
	if err := session.SendToolResult(callID, result, isError); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] send tool result %s: %v", rt.threadID, callID, err))
		rt.completeToolCall(callID)
		rt.closeForRecovery(session, false, "tool_result_delivery_failed")
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
		for _, result := range msg.ToolResults {
			eligible = append(eligible, Message{Role: "user", Content: fmt.Sprintf("[Internal completed tool result. This operation already ran; do not repeat it.] %s (call %s): %s", result.ToolName, result.CallID, result.Content)})
		}
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
	var history []Message
	if restore {
		history = rt.boundedTranscript()
	}
	opts.RestoreHistory = len(history) > 0
	rt.recovery.Lock()
	previous := rt.recovery.resume
	rt.recovery.resume = nil
	rt.recovery.Unlock()
	var session RealtimeSession
	resumed := false
	if resumer, ok := previous.(RealtimeSessionResumer); restore && ok {
		var err error
		session, err = resumer.Resume(rt.ctx, opts)
		resumed = err == nil && session != nil
		if err != nil && !errors.Is(err, ErrRealtimeResumeUnavailable) {
			rt.emit("realtime.resume_failed", map[string]any{"error": err.Error(), "fallback": "fresh_session"})
		}
	}
	if !resumed {
		rt.refreshCurrentTimeContext()
		opts.Instructions, opts.Tools = rt.configurationSnapshot()
		var err error
		session, err = rt.provider.Open(rt.ctx, opts)
		if err != nil {
			return err
		}
		if err := session.RestoreConversation(history); err != nil {
			_ = session.Close()
			return fmt.Errorf("restore conversation: %w", err)
		}
	}
	rt.rememberConfiguration(opts.Instructions, opts.Tools)
	rt.replaceSession(session)
	rt.recovery.Lock()
	rt.recovery.opened = time.Now()
	continueTurn := rt.recovery.continueTurn && len(rt.recovery.input) == 0 && len(rt.recovery.audio) == 0
	rt.recovery.continueTurn = false
	rt.recovery.hadInput = false
	rt.recovery.want = false
	rt.recovery.deadline = time.Time{}
	rt.recovery.Unlock()
	rt.lifecycleMu.Lock()
	rt.sessionGeneration++
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.emit("realtime.session_opened", map[string]any{
		"generation": generation, "restored": restore && !resumed, "resumed": resumed,
	})
	if !resumed {
		rt.requestInitialMessageIfNeeded()
	}
	if !resumed && continueTurn {
		if err := rt.requestTextResponse("[Internal connection recovery] Finish the interrupted reply using the restored conversation and completed tool results. Do not repeat completed operations or a greeting. If there is no pending reply, wait silently."); err != nil {
			rt.closeForRecovery(session, false, "recovery_continuation_failed")
			return err
		}
	}
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
	} else if disposition == RealtimeConfigurationAppliedLive {
		rt.rememberConfiguration(instructions, tools)
	}
}

func (rt *RealtimeThinker) rememberConfiguration(instructions string, tools []NativeTool) {
	rt.transcriptMu.Lock()
	rt.opts.Instructions, rt.opts.Tools = instructions, tools
	rt.transcriptMu.Unlock()
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
		rt.rememberConfiguration(instructions, tools)
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
	rt.wakeRecovery()
	rt.requestInitialMessageIfNeeded()
}

func (rt *RealtimeThinker) audioBridgeDisconnected() {
	rt.lifecycleMu.Lock()
	rt.bridgeConnected = false
	rt.lifecycleMu.Unlock()
	if ender, ok := rt.currentSession().(RealtimeInputEnder); ok {
		_ = ender.EndAudioInput()
	}
	rt.wakeRecovery()
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
	if session == nil || session != rt.currentSession() {
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

func (rt *RealtimeThinker) responseInProgress() bool {
	rt.responseMu.Lock()
	defer rt.responseMu.Unlock()
	return rt.responseActive || rt.responsePending
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
		pattern = detectRealtimeInternalNarration(combined)
		leaked = pattern != ""
	}
	if !leaked {
		if event.Final {
			delete(rt.toolMarkupTails, key)
			rt.toolMarkupRecoveryUsed = false
		}
		rt.toolMarkupMu.Unlock()
		return false
	}
	rt.toolMarkupMu.Unlock()
	return rt.rejectRealtimeOutput(event, toolName, pattern, combined)
}

func (rt *RealtimeThinker) rejectRealtimeOutput(event RealtimeEvent, toolName, pattern, combined string) bool {
	key := realtimeToolMarkupResponseKey(event)
	rt.toolMarkupMu.Lock()
	if rt.toolMarkupSuppressed[key] {
		rt.toolMarkupMu.Unlock()
		return true
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
	eventType := "realtime.tool_markup_leaked"
	if pattern == "internal_instruction_narration" || pattern == "missing_output_transcript" {
		eventType = "realtime.output_blocked"
	}
	rt.emit(eventType, map[string]any{
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
	rt.wakeRecovery()
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
	// Stop also cancels an Open/Resume currently waiting on the provider.
	go func() {
		select {
		case <-rt.quit:
			rt.cancel()
		case <-rt.ctx.Done():
		}
	}()
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
	catalogDirty := false
	for {
		if catalogDirty && !rt.responseInProgress() && !rt.pendingToolWork() {
			catalogDirty = false
			if _, err := rt.applyExternalConfigurationChange(true, "mcp_catalog_changed"); err != nil {
				logMsg("REALTIME", fmt.Sprintf("[%s] refresh MCP catalog: %v", rt.threadID, err))
			}
		}
		session := rt.currentSession()
		rt.recovery.Lock()
		deadline := rt.recovery.deadline
		rt.recovery.Unlock()
		if session != nil && !deadline.IsZero() &&
			((!rt.responseInProgress() && !rt.pendingToolWork() && rt.playbackSettled()) || !time.Now().Before(deadline)) {
			rt.closeForRecovery(session, true, "provider_goaway")
			session = nil
		}
		var retryAt time.Time
		if session == nil && rt.needsSession() && !rt.pendingToolWork() {
			rt.recovery.Lock()
			retryAt = rt.recovery.retryAt
			rt.recovery.Unlock()
			if !time.Now().Before(retryAt) {
				if err := rt.openSession(true); err != nil {
					rt.recovery.Lock()
					delay := rt.recovery.retry.failed()
					rt.recovery.retryAt = time.Now().Add(delay)
					rt.recovery.Unlock()
					rt.emit("realtime.reconnect", map[string]any{"success": false, "error": err.Error(), "delay_ms": delay.Milliseconds()})
					continue
				}
				rt.lifecycleMu.Lock()
				generation, reason, planned := rt.sessionGeneration, rt.pendingReconnectReason, rt.pendingReconnectPlanned
				rt.pendingReconnectReason, rt.pendingReconnectPlanned = "", false
				rt.lifecycleMu.Unlock()
				rt.emit("realtime.reconnect", map[string]any{"success": true, "generation": generation, "reason": reason, "planned": planned})
				rt.flushRecoveryInput()
				session = rt.currentSession()
			}
		}
		if session == nil && !rt.needsSession() {
			rt.setConversationState("waiting", RealtimeEvent{})
		}
		var events <-chan RealtimeEvent
		if session != nil {
			events = session.Events()
		}
		var timer *time.Timer
		var tick <-chan time.Time
		next := deadline
		if session == nil {
			next = retryAt
		}
		if !next.IsZero() && (session != nil || rt.needsSession() && !rt.pendingToolWork()) {
			delay := time.Until(next)
			if delay < 0 {
				delay = 0
			}
			timer = time.NewTimer(delay)
			tick = timer.C
		}
		select {
		case event, ok := <-events:
			if !ok {
				rt.lifecycleMu.Lock()
				planned, reason := rt.pendingReconnectPlanned, rt.pendingReconnectReason
				rt.lifecycleMu.Unlock()
				if reason == "" {
					reason = "provider_session_closed"
				}
				rt.closeForRecovery(session, planned, reason)
			} else {
				rt.handleSessionEvent(event)
			}
		case <-rt.recovery.wake:
		case <-tick:
		case <-rt.catalogChanges:
			rt.catalogChanges = rt.toolIndex.Changes()
			catalogDirty = true
		case audio, ok := <-rt.audioIn:
			if !ok {
				rt.audioIn = nil
				break
			}
			if rt.paused || len(audio) == 0 {
				break
			}
			rt.recordRealtimeInput(len(audio))
			if session == nil {
				rt.queueRecoveryAudio(audio)
			} else if err := session.SendAudio(audio); err != nil {
				rt.queueRecoveryAudio(audio)
				rt.closeForRecovery(session, false, "audio_delivery_failed")
			}
		case <-rt.sub.Wake:
			for _, event := range rt.sub.DrainTargeted() {
				if event.ToolGeneration != nil && *event.ToolGeneration != rt.toolGeneration.Load() {
					continue
				}
				rt.handleBusEvent(event)
			}
		case paused := <-rt.pause:
			rt.paused = paused
			if paused && session != nil {
				_ = session.Interrupt()
				if ender, ok := session.(RealtimeInputEnder); ok {
					_ = ender.EndAudioInput()
				}
			}
			rt.publishRuntimeStatus()
		case <-rt.quit:
			if timer != nil {
				timer.Stop()
			}
			return
		case <-rt.ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return
		}
		if timer != nil {
			timer.Stop()
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
	if event.Type == RealtimeEventAudioOut || event.Type == RealtimeEventTranscriptOutput {
		if !event.Final {
			rt.responseStarted()
		}
		rt.recovery.Lock()
		rt.recovery.lastOutput = time.Now()
		rt.recovery.outputBytes += uint64(len(event.Audio))
		rt.recovery.Unlock()
	}
	switch event.Type {
	case RealtimeEventSessionExpiring:
		deadline := time.Now().Add(event.TimeLeft)
		rt.recovery.Lock()
		if rt.recovery.deadline.IsZero() || deadline.Before(rt.recovery.deadline) {
			rt.recovery.deadline = deadline
		}
		rt.recovery.Unlock()
		rt.emit("realtime.reconnect_planned", map[string]any{"reason": "provider_goaway", "planned": true, "time_left_ms": event.TimeLeft.Milliseconds()})
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

	case RealtimeEventOutputBlocked:
		rt.rejectRealtimeOutput(event, "", event.OutputBlockReason, "")

	case RealtimeEventTranscriptOutput:
		if rt.suppressLeakedToolMarkup(event) {
			return
		}
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.appendTranscript("assistant", event.Transcript)
		}

	case RealtimeEventTranscriptInput:
		if strings.TrimSpace(event.Transcript) != "" {
			rt.recovery.Lock()
			newInput := event.ItemID != "" && event.ItemID != rt.recovery.inputItem
			if newInput {
				rt.recovery.inputItem = event.ItemID
			}
			rt.recovery.Unlock()
			if newInput && (!event.Final || !rt.pendingToolWork() && !rt.responseInProgress()) {
				rt.newRealtimeCallerTurn()
			}
		}
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
		rt.responseStarted()
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
		rt.recovery.Lock()
		if rt.recovery.hadInput {
			rt.recovery.retry = realtimeRecoveryRetry{}
			rt.recovery.retryAt = time.Time{}
		}
		rt.recovery.Unlock()
		hadToolBatch := rt.completeToolResponse(event.ResponseID)
		rt.responseFinished(rt.currentSession())
		rt.finishToolMarkupResponse(event.ResponseID)
		if !hadToolBatch {
			rt.setConversationState("listening", event)
		}
		if !hadToolBatch && !rt.responseInProgress() {
			rt.Thinker.settleEventExecutions("realtime_response_done")
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
			"close_code":           event.CloseCode, "close_reason": event.CloseReason,
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
	rt.toolBatchMu.Lock()
	_, duplicate := rt.toolCallBatches[event.ToolCallID]
	rt.toolBatchMu.Unlock()
	if duplicate {
		return
	}
	rt.recovery.Lock()
	blocked := rt.recovery.blockedTools[event.ToolName]
	rt.recovery.Unlock()
	rt.beginToolCall(event, session)
	if blocked {
		rt.submitToolResult(session, event.ToolCallID, "This tool already ran before connection recovery. Use the restored operation results; do not repeat it. Wait for a new caller instruction if further action is needed.", true)
		return
	}
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
		rt.submitToolResult(session, call.NativeID, "tool execution was not authorized", true)
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
	rt.Thinker.addEventExecutions(event.ExecutionIDs)
	session := rt.currentSession()
	if event.ToolResult != nil {
		rt.toolBatchMu.Lock()
		_, known := rt.toolCallBatches[event.ToolResult.CallID]
		rt.toolBatchMu.Unlock()
		if !known {
			return
		} // Duplicate/stale completions never target a new socket.
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
	if session == nil {
		rt.recovery.Lock()
		rt.recovery.input = append(rt.recovery.input, event)
		rt.recovery.want = true
		rt.recovery.Unlock()
		rt.wakeRecovery()
		return
	}
	if err := session.SendText("user", note); err != nil {
		logMsg("REALTIME", fmt.Sprintf("[%s] inject text: %v", rt.threadID, err))
		rt.recovery.Lock()
		rt.recovery.input = append(rt.recovery.input, event)
		rt.recovery.want = true
		rt.recovery.Unlock()
		rt.closeForRecovery(session, false, "text_delivery_failed")
		return
	}
	rt.recordRealtimeInput(0)
	if event.From == "" || event.From == "api" || event.From == "tui" || event.From == "user" {
		rt.newRealtimeCallerTurn()
	}
	rt.setConversationState("thinking", RealtimeEvent{})
	responseErr := rt.requestProviderResponse(session)
	{
		message := Message{Role: "user", Content: note, EventIDs: []string{event.ID}}
		if event.ID == "" {
			message.EventIDs = nil
		}
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
		if event.ID != "" && persisted && rt.Thinker.ackInboxEvents != nil {
			if err := rt.Thinker.ackInboxEvents([]string{event.ID}); err != nil {
				logMsg("SESSION", fmt.Sprintf("[%s] acknowledge realtime inbox event: %v", rt.threadID, err))
			}
		}
	}
	if responseErr != nil {
		rt.closeForRecovery(session, false, "response_request_failed")
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
