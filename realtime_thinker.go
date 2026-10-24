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
	"sync/atomic"
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
	capabilities   RealtimeCapabilities
	continuous     realtimeContinuousState
	backend        realtimeBackend
	voice          string
	opts           RealtimeSessionOpts
	// currentTimeContext is fixed for one provider session so ordinary
	// configuration comparisons cannot churn or reconnect live audio. It is
	// refreshed immediately before each new provider session opens.
	currentTimeContext string
	sessionDirective   string // voice role at startup; backend catalog changes do not restart it

	ctx    context.Context
	cancel context.CancelFunc

	sessionMu           sync.RWMutex
	rtSession           RealtimeSession
	recovery            realtimeRecovery
	input               realtimeInputQueue
	audioLane           realtimeAudioLane
	outputPending       []RealtimeAudioFrame
	outputPendingBytes  int
	inputPaused         atomic.Bool
	callerSpeechPending atomic.Bool
	outputPaused        atomic.Bool
	rendererScoped      atomic.Bool
	opening             *realtimeOpenAttempt
	openResults         chan realtimeOpenResult
	sessionCancel       context.CancelFunc
	providerStateMu     sync.Mutex
	providerStatus      string
	providerStateWake   chan struct{}

	audioIn      <-chan []byte
	audioOut     chan RealtimeAudioFrame
	audioControl chan<- string

	transcriptNamespace    string
	transcriptTurns        map[string]*realtimeTranscriptTurn
	transcriptAnonymous    map[string]int
	transcriptFinalOrder   []string
	transcriptOrdinal      int
	completionReview       realtimeCompletionReview
	lastCallerVoice        atomic.Int64
	completion             realtimeCompletion
	delivery               realtimeDelivery
	shutdownOnce           sync.Once
	transcriptMu           sync.Mutex
	outputMu               sync.Mutex
	outputItemID           string
	outputBytes            int
	playedItemID           string
	playedMS               int
	playbackTracked        bool
	interruptedItem        string
	suppressOutput         bool
	cancelledPlaybacks     map[string]bool
	cancelledPlaybackOrder []string
	playbackClearRequested map[string]time.Time
	toolMarkupRecoveryKeys map[string]bool
	responseMu             sync.Mutex
	responseActive         bool
	responsePending        bool
	stateMu                sync.Mutex
	state                  string
	statePhase             string

	lifecycleMu                 sync.Mutex
	timing                      *realtimeTiming // protected by lifecycleMu
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

	toolBatchMu       sync.Mutex
	toolBatches       map[string]*realtimeToolBatch
	toolCallBatches   map[string]string
	toolCallRecords   map[realtimeToolCallKey]*realtimeToolCallRecord
	terminalToolCalls []realtimeToolCallKey

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
		terminalReason:      "server_shutdown",
		transcriptNamespace: generateAudioToken(),
		transcriptTurns:     map[string]*realtimeTranscriptTurn{},
		transcriptAnonymous: map[string]int{},
	}
	rt.initAudioLane()
	rt.openResults = make(chan realtimeOpenResult)
	rt.providerStateWake = make(chan struct{}, 1)
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
	rt.capabilities = capabilitiesForRealtime(provider, model)
	rt.configureInputQueue()
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
		rt.cancelOpening()
		rt.stopExecutionThread()
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
		if rt.capabilities.CoreExecution {
			rt.backend.delegation = ""
			rt.backend.seen = nil
			rt.backend.order = nil
			rt.backend.offsetMS = 0
			rt.backend.inputEndMS = 0
		}
		rt.responseMu.Lock()
		rt.responseActive = false
		rt.responsePending = false
		rt.responseMu.Unlock()
		// In-flight tools keep their original session ownership until their
		// result arrives. Recovery consumes those outcomes before reopening.
		if rt.sessionCancel != nil {
			rt.sessionCancel()
			rt.sessionCancel = nil
		}
		go func() { _ = previous.Close() }()
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
	rt.rememberActiveToolCallLocked(session, event.ToolCallID)
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
	rt.finishToolCall(callID, false)
}

func (rt *RealtimeThinker) finishToolCall(callID string, cancelled bool) {
	var continueSession RealtimeSession
	rt.toolBatchMu.Lock()
	batchID, exists := rt.toolCallBatches[callID]
	if exists {
		delete(rt.toolCallBatches, callID)
		if batch := rt.toolBatches[batchID]; batch != nil {
			rt.rememberTerminalToolCallLocked(batch.session, callID, cancelled)
			if batch.pending > 0 {
				batch.pending--
			}
			if (realtimeToolsAreAsync(batch.session) || rt.capabilities.CoreExecution) && batch.pending == 0 {
				delete(rt.toolBatches, batchID)
			} else if (batch.responseDone || cancelled) && batch.pending == 0 {
				if !cancelled {
					continueSession = batch.session
				}
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
			if !realtimeToolsAreAsync(batch.session) {
				continueSession = batch.session
			}
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
	if rt.capabilities.CoreExecution {
		// All outcomes, including admission failures, must pair with the
		// backend's persisted call. No native result reaches the voice model.
		found := false
		rt.transcriptMu.Lock()
		for _, m := range rt.messages {
			for _, r := range m.ToolResults {
				if r.CallID == callID {
					found = true
				}
			}
		}
		rt.transcriptMu.Unlock()
		if !found {
			rt.persistExecutionMessage(Message{Role: "user", ToolResults: []ToolResult{{CallID: callID, Content: result, IsError: isError}}}, TokenUsage{})
		}
		rt.completeToolCall(callID)
		rt.backend.want = true
		rt.wakeRecovery()
		return
	}
	if rt.cancelledToolCall(session, callID) {
		return
	}
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
		if errors.Is(err, ErrRealtimeToolCallCancelled) {
			rt.cancelRealtimeToolCall(session, callID)
			// The result was already archived by the caller of this method.
			rt.recordCancelledToolOutcome(session, callID)
			return
		}
		logMsg("REALTIME", fmt.Sprintf("[%s] send tool result %s: %v", rt.threadID, callID, err))
		rt.completeToolCall(callID)
		rt.closeForRecovery(session, false, "tool_result_delivery_failed", err)
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
		text := appendFileRefText(msg.Content, msg.Parts)
		if (msg.Role == "user" || msg.Role == "assistant") && strings.TrimSpace(text) != "" {
			eligible = append(eligible, Message{Role: msg.Role, Content: text, RealtimeBackend: msg.RealtimeBackend})
		}
	}
	if len(eligible) > realtimeRestoreMessages {
		eligible = eligible[len(eligible)-realtimeRestoreMessages:]
	}
	return eligible
}

func (rt *RealtimeThinker) currentTiming() *realtimeTiming {
	rt.lifecycleMu.Lock()
	defer rt.lifecycleMu.Unlock()
	return rt.timing
}

func startRealtimeThinkerWithOutput(ctx context.Context, thinker *Thinker, provider RealtimeProvider, voice string, audioIn <-chan []byte, audioOut chan RealtimeAudioFrame, audioControl chan<- string, turn RealtimeTurnDetectionConfig, output RealtimeOutputConfig, initialMessage ...string) (*RealtimeThinker, error) {
	rt := newRealtimeThinker(ctx, thinker, provider, voice, audioIn, audioOut, audioControl, turn)
	rt.opts.OutputConfig = output
	rt.configureMedia()
	if len(initialMessage) > 0 {
		rt.setInitialMessage(initialMessage[0])
	}
	if err := rt.openSession(false); err != nil {
		rt.cancel()
		return nil, fmt.Errorf("realtime open: %w", err)
	}
	return rt, nil
}

func (rt *RealtimeThinker) prepareSession(restore bool) (*realtimeOpenAttempt, error) {
	if rt.opts.OutputConfig.Media != (RealtimeMediaConfig{}) {
		provider, ok := rt.provider.(realtimeMediaProvider)
		if !ok {
			return nil, fmt.Errorf("provider does not support explicit realtime media selection")
		}
		if err := provider.ValidateRealtimeMedia(rt.opts.Model, rt.opts.OutputConfig.Media); err != nil {
			return nil, err
		}
	}

	connectedAt := time.Now()
	rt.lifecycleMu.Lock()
	nextGeneration := rt.sessionGeneration + 1
	rt.lifecycleMu.Unlock()
	timing := newRealtimeTiming(uint64(nextGeneration), rt.emit)
	timing.connectedAt = connectedAt
	timing.bytesPerSecond = rt.outputRate()
	rt.refreshCurrentTimeContext()
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
	opts.History = history
	rt.lifecycleMu.Lock()
	if !rt.assistantAudioEmitted {
		opts.InitialMessage = rt.initialMessage
	}
	rt.lifecycleMu.Unlock()
	opts.ConversationInstructions = "You are the live voice of this agent. Speak only words intended for the caller. Follow the requested language and speaking style. Your assigned role and requirements:\n" + rt.directive
	if err := rt.configureExecution(); err != nil {
		return nil, err
	}
	if rt.capabilities.CoreExecution {
		worker, err := rt.ensureExecutionThread()
		if err != nil {
			return nil, err
		}
		opts.ExecutionMode = rt.backend.mode
		opts.ExecutionModel = modelIDForProvider(rt.backend.provider, rt.backend.model)
		opts.ExecutionInstructions, opts.ExecutionTools = worker.Thinker.work.setup()
	}
	opts.timing = timing
	rt.recovery.Lock()
	previous := rt.recovery.resume
	rt.recovery.resume = nil
	rt.recovery.Unlock()
	ctx, cancel := context.WithCancel(rt.ctx)
	return &realtimeOpenAttempt{ctx: ctx, cancel: cancel, opts: opts, previous: previous, restore: restore, timing: timing, connectedAt: connectedAt, generation: nextGeneration - 1, directive: rt.directive}, nil
}

func connectRealtimeSession(provider RealtimeProvider, a *realtimeOpenAttempt) realtimeOpenResult {
	ctx, opts, previous, restore := a.ctx, a.opts, a.previous, a.restore
	history := opts.History
	var resumeErr error
	var session RealtimeSession
	resumed := false
	if resumer, ok := previous.(RealtimeSessionResumer); restore && ok {
		var err error
		session, err = resumer.Resume(ctx, opts)
		resumed = err == nil && session != nil
		if err != nil && !errors.Is(err, ErrRealtimeResumeUnavailable) {
			resumeErr = err
		}
	}
	if !resumed {
		var err error
		session, err = provider.Open(ctx, opts)
		if err != nil {
			return realtimeOpenResult{attempt: a, err: err, resumeErr: resumeErr}
		}
		if session == nil {
			return realtimeOpenResult{attempt: a, err: errors.New("provider returned nil realtime session")}
		}
		if err := session.RestoreConversation(history); err != nil {
			go func() { _ = session.Close() }()
			return realtimeOpenResult{attempt: a, err: fmt.Errorf("restore conversation: %w", err), resumeErr: resumeErr}
		}
	}
	if session == nil {
		return realtimeOpenResult{attempt: a, err: errors.New("provider returned nil realtime session")}
	}
	return realtimeOpenResult{attempt: a, session: session, resumed: resumed, resumeErr: resumeErr}
}

func (rt *RealtimeThinker) installSession(result realtimeOpenResult) error {
	a := result.attempt
	opts, timing, connectedAt, restore := a.opts, a.timing, a.connectedAt, a.restore
	session, resumed := result.session, result.resumed
	if result.resumeErr != nil {
		rt.emit("realtime.resume_failed", map[string]any{"error": result.resumeErr.Error(), "fallback": "fresh_session"})
	}
	rt.rememberConfiguration(opts.Instructions, opts.Tools)
	rt.sessionDirective = rt.directive
	rt.recovery.Lock()
	rt.recovery.opened = time.Now()
	continueTurn := rt.recovery.continueTurn && len(rt.recovery.input) == 0 && !rt.input.pending()
	rt.recovery.continueTurn = false
	rt.recovery.hadInput = false
	rt.recovery.want = false
	rt.recovery.deadline = time.Time{}
	rt.recovery.Unlock()
	rt.lifecycleMu.Lock()
	previousTiming := rt.timing
	rt.timing = timing
	rt.sessionGeneration++
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	rt.outputMu.Lock()
	rt.delivery.abandonPlayback("")
	rt.cancelledPlaybacks = map[string]bool{}
	rt.cancelledPlaybackOrder = nil
	rt.suppressOutput = false
	rt.interruptedItem = ""
	rt.outputItemID = ""
	rt.outputBytes = 0
	rt.playedItemID = ""
	rt.playedMS = 0
	rt.playbackTracked = false
	rt.outputMu.Unlock()
	rt.replaceSession(session)
	rt.sessionCancel = a.cancel
	previousTiming.finish("", "", "session_replaced")
	data := map[string]any{"generation": generation, "restored": restore && !resumed, "resumed": resumed,
		"connection_to_open_ms": float64(time.Since(connectedAt)) / float64(time.Millisecond), "provider": rt.provider.Name(),
		"requested": realtimeRequestedSettings(opts), "applied": nil}
	data["capabilities"] = rt.capabilities
	if reporter, ok := session.(realtimeSettingsSession); ok {
		data["applied"] = reporter.RealtimeSettings()
		if reporter.RealtimeSettings()["source"] == "accepted_setup" {
			readyAt := time.Now()
			if ready, ok := session.(realtimeReadySession); ok {
				readyAt = ready.RealtimeReadyAt()
			}
			timing.ready(readyAt, reporter.RealtimeSettings())
		}
	}
	rt.emit("realtime.session_opened", data)
	rt.providerState("ready", nil)
	if !resumed {
		rt.requestInitialMessageIfNeeded()
	}
	if !resumed && continueTurn {
		if err := rt.requestTextResponse("[Internal connection recovery] Finish the interrupted reply using the restored conversation and completed tool results. Do not repeat completed operations or a greeting. If there is no pending reply, wait silently."); err != nil {
			rt.closeForRecovery(session, false, "recovery_continuation_failed", err)
			return err
		}
	}
	return nil
}

func (rt *RealtimeThinker) openSession(restore bool) error {
	a, err := rt.prepareSession(restore)
	if err != nil {
		return err
	}
	r := connectRealtimeSession(rt.provider, a)
	if r.err != nil {
		a.cancel()
		return r.err
	}
	return rt.installSession(r)
}

func (rt *RealtimeThinker) refreshConfiguration() {
	if rt.completionRequested() {
		return
	}
	session := rt.currentSession()
	if session == nil {
		return
	}
	instructions, tools := rt.configurationSnapshot()
	if rt.capabilities.CoreExecution && rt.sessionDirective == rt.directive {
		rt.rememberConfiguration(instructions, tools)
		return
	}
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
	rt.audioLane.tools.Store(append([]NativeTool(nil), tools...))
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
		rt.flushPendingTranscripts(reason)
		rt.recordSessionClosure(reason, true, nil)
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
	rt.flushPendingTranscripts("bridge_disconnected")
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
	data := map[string]any{"generation": generation, "replayed_greeting": replayed, "measurement_stage": "output_enqueue"}
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
	if rt.audioLane.active.Load() {
		select {
		case rt.audioLane.telemetry <- realtimeLaneTelemetry{eventType, data}:
		default:
			rt.audioLane.telemetryDropped.Add(1)
		}
		return
	}
	if rt.telemetry != nil {
		rt.telemetry.Emit(eventType, rt.threadID, data)
	}
}

func (rt *RealtimeThinker) setConversationState(state string, event RealtimeEvent) {
	state = strings.TrimSpace(state)
	phase := strings.TrimSpace(event.Phase)
	if rt.capabilities.ContinuousConversation && (state == "listening" || state == "thinking" || state == "working") && rt.delivery.hasPendingPlayback() {
		state = "speaking"
	}
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
	if rt.capabilities.ContinuousConversation {
		return nil
	}
	if rt.completionRequested() {
		return nil
	}
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
	if event.PlaybackID != "" {
		return event.PlaybackID
	}
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
	if rt.completionRequested() {
		return errors.New("realtime completion requested")
	}
	if session == nil {
		return errors.New("realtime session unavailable")
	}
	if rt.capabilities.ContinuousConversation {
		return rt.appendRealtimeContext(RealtimeContextUpdate{Kind: "instructions", Content: realtimeToolMarkupRecoveryPrompt})
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
	rt.blockTranscript(event)
	key := realtimeToolMarkupResponseKey(event)
	rt.toolMarkupMu.Lock()
	if rt.toolMarkupSuppressed[key] {
		rt.toolMarkupMu.Unlock()
		return true
	}
	rt.toolMarkupSuppressed[key] = true
	delete(rt.toolMarkupTails, key)
	retry := !rt.toolMarkupRecoveryUsed
	if rt.capabilities.ContinuousConversation {
		if rt.toolMarkupRecoveryKeys == nil {
			rt.toolMarkupRecoveryKeys = map[string]bool{}
		}
		retry = !rt.toolMarkupRecoveryKeys[key]
		if len(rt.toolMarkupRecoveryKeys) >= 128 {
			rt.toolMarkupRecoveryKeys = map[string]bool{}
		}
		rt.toolMarkupRecoveryKeys[key] = true
	}
	if retry {
		rt.toolMarkupRecoveryUsed = true
	}
	rt.toolMarkupMu.Unlock()

	rt.currentTiming().observed(event.ResponseID, event.ItemID)
	rt.currentTiming().finish(event.ResponseID, event.ItemID, "blocked")
	reason := "provider_tool_markup_leaked"
	if pattern == "internal_instruction_narration" {
		reason = pattern
	}
	target := event.PlaybackID
	if target == "" {
		target = event.ItemID
	}
	interrupted := rt.interruptPlayback(reason, target, true, true)
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

func (rt *RealtimeThinker) acknowledgePlayback(itemID string, audioEndMS int, generations ...int64) {
	rt.lifecycleMu.Lock()
	generation := rt.sessionGeneration
	rt.lifecycleMu.Unlock()
	currentGeneration := generation
	if len(generations) > 0 && generations[0] > 0 {
		generation = int(generations[0])
	}
	if itemID != "" && audioEndMS >= 0 {
		rt.delivery.acknowledge(fmt.Sprintf("%d/%s", generation, itemID), audioEndMS)
	}
	if generation != currentGeneration || itemID == "" || audioEndMS < 0 {
		return
	}
	rt.outputMu.Lock()
	if itemID != rt.outputItemID {
		rt.outputMu.Unlock()
		return
	}
	generatedMS := rt.outputBytes * 1000 / rt.outputRate()
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
	rt.outputMu.Unlock()
	rt.currentTiming().playback(itemID, audioEndMS, time.Now())
	rt.wakeRecovery()
}

func (rt *RealtimeThinker) discardQueuedOutput() (frames, bytes int) {
	return rt.discardQueuedPlayback("")
}

// Caller serializes producers with outputMu. Keep later, unrelated playback.
func (rt *RealtimeThinker) discardQueuedPlayback(item string) (frames, bytes int) {
	remaining := rt.outputPending[:0]
	for _, frame := range rt.outputPending {
		if item != "" && frame.ItemID != item {
			remaining = append(remaining, frame)
			continue
		}
		frame.timing.written(frame.ResponseID, frame.ItemID, time.Time{}, len(frame.Audio), false)
		frame.written(false)
		frames++
		bytes += len(frame.Audio)
		rt.outputPendingBytes -= len(frame.Audio)
	}
	for i := len(remaining); i < len(rt.outputPending); i++ {
		rt.outputPending[i] = RealtimeAudioFrame{}
	}
	rt.outputPending = remaining
	if rt.audioOut == nil {
		return frames, bytes
	}
	var keep []RealtimeAudioFrame
	defer func() {
		for _, frame := range keep {
			rt.audioOut <- frame
		}
	}()
	for i, n := 0, len(rt.audioOut); i < n; i++ {
		select {
		case frame, ok := <-rt.audioOut:
			if !ok {
				return frames, bytes
			}
			if item != "" && frame.ItemID != item {
				keep = append(keep, frame)
				continue
			}
			frame.timing.written(frame.ResponseID, frame.ItemID, time.Time{}, len(frame.Audio), false)
			frame.written(false)
			frames++
			bytes += len(frame.Audio)
		default:
			return frames, bytes
		}
	}
	return frames, bytes
}

func (rt *RealtimeThinker) interruptPlayback(reason, expectedItemID string, cancelProvider, requirePlaybackAck bool) bool {
	startedAt := time.Now()
	rt.outputMu.Lock()
	itemID := rt.outputItemID
	current := expectedItemID == "" || itemID == expectedItemID
	generatedMS := rt.outputBytes * 1000 / rt.outputRate()
	playedMS := generatedMS
	if !current {
		var active bool
		generatedMS, playedMS, active = rt.delivery.playbackPosition(expectedItemID)
		if !active {
			rt.outputMu.Unlock()
			return false
		}
		itemID = expectedItemID
		// A late check can invalidate older queued playback, but must not send a
		// provider-wide cancellation against the newer speech now being generated.
		cancelProvider = false
	} else {
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
	}
	if itemID != "" {
		if rt.cancelledPlaybacks == nil {
			rt.cancelledPlaybacks = map[string]bool{}
		}
		rt.cancelledPlaybacks[itemID] = true
		if rt.playbackClearRequested == nil {
			rt.playbackClearRequested = map[string]time.Time{}
		}
		rt.playbackClearRequested[itemID] = startedAt
		rt.cancelledPlaybackOrder = append(rt.cancelledPlaybackOrder, itemID)
		if len(rt.cancelledPlaybackOrder) > 256 {
			delete(rt.cancelledPlaybacks, rt.cancelledPlaybackOrder[0])
			delete(rt.playbackClearRequested, rt.cancelledPlaybackOrder[0])
			rt.cancelledPlaybackOrder = rt.cancelledPlaybackOrder[1:]
		}
		rt.delivery.abandonPlayback(itemID)
	}
	// Record cancellation before dropping frames so delivery failures cannot
	// misclassify an intentionally interrupted playback.
	if itemID != "" {
		rt.currentTiming().finish("", itemID, "interrupted")
	}
	drainedFrames, drainedBytes := rt.discardQueuedPlayback(itemID)
	// Queue the clear before another playback can be enqueued. Scoped metadata
	// lets updated renderers ignore stale controls; legacy renderers keep type.
	if itemID != "" && rt.audioControl != nil {
		control := "interrupt"
		if rt.capabilities.ContinuousConversation || rt.rendererScoped.Load() {
			payload, _ := json.Marshal(map[string]any{"type": "interrupt", "item_id": itemID, "session_generation": rt.delivery.generationFor(itemID, generationForAudioFrame(rt)), "reason": reason, "requested_at_ms": time.Now().UnixMilli()})
			control = string(payload)
		}
		select {
		case rt.audioControl <- control:
		case <-rt.ctx.Done():
		case <-time.After(100 * time.Millisecond):
			rt.emit("realtime.control_overflow", map[string]any{"control": "interrupt", "reason": reason})
		}
	}
	rt.outputMu.Unlock()

	session := rt.currentSession()
	if local, ok := session.(interface{ DiscardPlayback(string) }); ok && itemID != "" {
		local.DiscardPlayback(itemID)
	}
	command := "none"
	commandQueued := false
	if cancelProvider && session != nil {
		command = "interrupt"
		if reporter, ok := session.(interface{ RealtimeInterruptCommand() string }); ok {
			command = reporter.RealtimeInterruptCommand()
		}
		if err := session.Interrupt(); err != nil {
			rt.emit("realtime.interrupt_error", map[string]any{"reason": reason, "error": err.Error()})
		} else {
			commandQueued = true
		}
	}
	if session != nil && itemID != "" && rt.capabilities.AudioTruncation {
		if err := session.Truncate(itemID, playedMS); err != nil {
			rt.emit("realtime.truncate_error", map[string]any{"reason": reason, "item_id": itemID, "error": err.Error()})
		}
	}
	rt.emit("realtime.playback_interrupted", map[string]any{
		"reason": reason, "item_id": itemID, "generated_ms": generatedMS, "played_ms": playedMS,
		"local_clear_ms": float64(time.Since(startedAt)) / float64(time.Millisecond),
		"drained_frames": drainedFrames, "drained_bytes": drainedBytes, "provider_cancel_requested": cancelProvider, "provider_command": command, "provider_command_queued": commandQueued, "provider_cancellation_confirmed": false,
	})
	return itemID != "" || drainedFrames > 0
}

func (rt *RealtimeThinker) rendererSpeechStarted() {
	rt.callerSpeechPending.Store(true)
	rt.clearCallerInterruptedPlayback("renderer_speech_started", !rt.capabilities.ProviderInterruption, true)
	rt.setConversationState("listening", RealtimeEvent{})
}

func (rt *RealtimeThinker) rendererPlaybackOverflow(itemID string) {
	if rt.interruptPlayback("renderer_overflow", itemID, !rt.capabilities.ProviderInterruption, true) {
		rt.setConversationState("listening", RealtimeEvent{})
	}
}

// Run survives normal provider-enforced session endings. Only explicit thread
// stop/cancellation ends the worker and invokes the normal cleanup path.
func (rt *RealtimeThinker) Run() {
	defer func() {
		if rt.completionReview.cancel != nil {
			rt.completionReview.cancel()
		}
	}()
	rt.beginRuntime()
	rt.configureInputQueue()
	rt.inputPaused.Store(rt.paused)
	rt.outputPaused.Store(rt.paused)
	intakeCtx, intakeCancel := context.WithCancel(rt.ctx)
	intakeDone := make(chan struct{})
	rt.startAudioLane(intakeCtx)
	go func() { defer close(intakeDone); rt.ingestRealtimeAudio(intakeCtx) }()
	defer rt.endRuntime()
	rt.workerMilestone("ready", "")
	// Stop also cancels an Open/Resume currently waiting on the provider.
	go func() {
		select {
		case <-rt.quit:
			rt.cancel()
		case <-rt.ctx.Done():
		}
	}()
	defer func() {
		rt.cancelOpening()
		rt.inputPaused.Store(true)
		intakeCancel()
		<-intakeDone
		rt.stopAudioLane()
		rt.input.discard("thread_stopped")
		rt.reportInputDrops()
		rt.stopExecutionThread()
		rt.drainTerminalTranscripts()
		rt.finishShutdown()
		rt.Thinker.Stop()
		rt.currentTiming().finish("", "", "thread_ended")
		rt.cancel()
		rt.replaceSession(nil)
		rt.flushAudioTelemetry()
		rt.audioLane.active.Store(false)
		if rt.onStop != nil {
			rt.onStop()
		}
	}()

	logMsg("REALTIME", fmt.Sprintf("[%s] session up, model=%s voice=%s", rt.threadID, rt.opts.Model, rt.voice))
	rt.emit("realtime.session_started", map[string]any{
		"model": rt.opts.Model, "voice": rt.voice, "provider": rt.provider.Name(),
		"turn_detection": rt.effectiveTurnDetectionTelemetry(),
	})
	rt.setConversationState("listening", RealtimeEvent{})
	catalogDirty := false
	for {
		rt.flushAudioTelemetry()
		if changed := rt.applyRuntimeMutations(); changed {
			if rt.opening != nil && rt.opening.directive != rt.directive {
				rt.cancelOpening()
			}
			if rt.capabilities.CoreExecution {
				rt.backend.revision++
				if rt.backend.active || rt.backend.want {
					rt.backend.want = true
				}
			}
		}
		rt.observeContinuousIdle()
		rt.checkpointContinuousTranscripts()
		rt.maybeStartExecution()
		rt.pollCompletionReview(time.Now())
		if rt.completionRequested() {
			rt.cancelOpening()
			rt.inputPaused.Store(true)
			rt.finishRealtimeCompletion(rt.currentSession())
			return
		}
		if catalogDirty && !rt.responseInProgress() && !rt.pendingToolWork() {
			catalogDirty = false
			if rt.capabilities.CoreExecution {
				instructions, tools := rt.configurationSnapshot()
				rt.rememberConfiguration(instructions, tools)
			} else {
				if _, err := rt.applyExternalConfigurationChange(true, "mcp_catalog_changed"); err != nil {
					logMsg("REALTIME", fmt.Sprintf("[%s] refresh MCP catalog: %v", rt.threadID, err))
				}
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
		rt.reportInputDrops()
		var retryAt time.Time
		if rt.opening != nil && time.Since(rt.opening.connectedAt) >= 30*time.Second {
			rt.cancelOpening()
			rt.openFailed(fmt.Errorf("realtime connection attempt timed out: %w", context.DeadlineExceeded))
		}
		if session == nil && rt.opening == nil && rt.needsSession() && !rt.pendingToolWork() {
			rt.recovery.Lock()
			retryAt = rt.recovery.retryAt
			rt.recovery.Unlock()
			if !time.Now().Before(retryAt) {
				rt.startOpening()
				rt.recovery.Lock()
				retryAt = rt.recovery.retryAt
				rt.recovery.Unlock()
			}
		}
		session = rt.currentSession()
		if session == nil && !rt.needsSession() {
			rt.setConversationState("waiting", RealtimeEvent{})
		}
		var events <-chan realtimeLaneEvent = rt.audioLane.events
		var timer *time.Timer
		var tick <-chan time.Time
		next := deadline
		if session == nil {
			next = retryAt
			if rt.opening != nil {
				next = rt.opening.connectedAt.Add(30 * time.Second)
			}
		}
		if rt.input.pending() {
			inputNext := time.Now().Add(100 * time.Millisecond)

			if next.IsZero() || inputNext.Before(next) {
				next = inputNext
			}
		}
		if rt.capabilities.ContinuousConversation && session != nil && (next.IsZero() || time.Until(next) > 100*time.Millisecond) {
			next = time.Now().Add(100 * time.Millisecond)
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
		case <-rt.mutationWake:

		case delivered := <-events:
			if delivered.session != rt.currentSession() {
				break
			}
			if delivered.closed {
				rt.lifecycleMu.Lock()
				planned, reason := rt.pendingReconnectPlanned, rt.pendingReconnectReason
				rt.lifecycleMu.Unlock()
				if reason == "" {
					reason = "provider_session_closed"
				}
				rt.closeForRecovery(session, planned, reason)
			} else {
				rt.observeAudioLane()
				rt.handleSessionEvent(delivered.event)
			}
		case <-rt.audioLane.wake:
			rt.observeAudioLane()
		case failure := <-rt.audioLane.failures:
			if failure.session == rt.currentSession() {
				rt.recordInputFailure(failure.err, failure.reason)
				if strings.HasPrefix(failure.reason, "output_") {
					rt.clearCallerInterruptedPlayback(failure.reason, false, true)
				}
				rt.closeForRecovery(failure.session, false, failure.reason, failure.err)
			}
		case <-rt.recovery.wake:
		case <-tick:
		case <-rt.catalogChanges:
			rt.cancelOpening()
			rt.catalogChanges = rt.toolIndex.Changes()
			catalogDirty = true
		case result := <-rt.openResults:
			rt.finishOpening(result)
		case <-rt.sub.Wake:
			for _, event := range rt.sub.DrainTargeted() {
				if event.ToolGeneration != nil && *event.ToolGeneration != rt.toolGeneration.Load() {
					continue
				}
				rt.handleBusEvent(event)
			}
		case paused := <-rt.pause:
			rt.paused = paused
			rt.inputPaused.Store(paused)
			rt.outputPaused.Store(paused)
			if paused {
				rt.input.discard("paused")
			}
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
	event = rt.normalizeContinuousEvent(event)
	rt.noteCompletionEvent(event)
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
	case RealtimeEventBackendProgress:
		rt.emit("realtime.backend_event", map[string]any{"phase": event.Phase, "response_id": event.ResponseID})
	case RealtimeEventDelegation:
		rt.acceptDelegation(event)
	case RealtimeEventSessionUsage:
		rt.emit("realtime.usage", map[string]any{"model": rt.opts.Model, "session_seconds": event.Usage.SessionSeconds, "cost": calculateCostForRealtimeProvider(rt.provider, rt.opts.Model, event.Usage), "accounting": "delta"})
	case RealtimeEventContextAcknowledged:
		rt.emit("realtime.context_acknowledged", map[string]any{"client_event_id": event.ClientEventID, "context_start_ms": event.TranscriptStartMS, "context_end_ms": event.TranscriptEndMS, "playback_confirmed": false})
	case RealtimeEventSessionReady:
		rt.currentTiming().ready(event.ObservedAt, event.Settings)
	case RealtimeEventSessionExpiring:
		deadline := time.Now().Add(event.TimeLeft)
		rt.recovery.Lock()
		if rt.recovery.deadline.IsZero() || deadline.Before(rt.recovery.deadline) {
			rt.recovery.deadline = deadline
		}
		rt.recovery.Unlock()
		rt.emit("realtime.reconnect_planned", map[string]any{"reason": "provider_goaway", "planned": true, "time_left_ms": event.TimeLeft.Milliseconds()})
	case RealtimeEventAudioOut:
		if !event.audioForwarded {
			if !rt.forwardRealtimeAudio(event, generationForAudioFrame(rt)) {
				return
			}
		}
		if !event.AudioSilence && !rt.outputPaused.Load() {
			rt.markAssistantAudioEmitted()
			if rt.capabilities.ContinuousConversation {
				rt.continuous.lastRelease = event.ObservedAt
				if rt.continuous.lastRelease.IsZero() {
					rt.continuous.lastRelease = time.Now()
				}
			}
			rt.setConversationState("speaking", event)
		}

	case RealtimeEventSpeechStarted:
		rt.backend.speechPending = true
		if event.ResponseID != "" || event.ItemID != "" {
			rt.currentTiming().finish(event.ResponseID, event.ItemID, "interrupted")
		}
		if !event.audioForwarded {
			rt.clearCallerInterruptedPlayback("provider_speech_started", false, false)
		}
		rt.setConversationState("listening", event)

	case RealtimeEventSpeechStopped:
		// The adapter timestamps its received VAD event; no inferred Gemini speech end.
	case RealtimeEventTranscriptIncomplete:
		rt.emit("realtime.transcript_incomplete", map[string]any{"item_id": event.PlaybackID, "reason": event.OutputBlockReason})
	case RealtimeEventOutputBlocked:
		if event.OutputBlockReason == "missing_output_transcript" {
			target := event.PlaybackID
			if target == "" {
				target = event.ItemID
			}
			if target == "" {
				target = event.ResponseID
			}
			rt.currentTiming().finish(event.ResponseID, target, "caption_missing")
			// With no association, preserve unrelated playback rather than guessing.
			if target != "" {
				rt.interruptPlayback("missing_output_transcript", target, false, true)
			}
			rt.emit("realtime.output_blocked", map[string]any{"reason": "missing_output_transcript", "item_id": event.PlaybackID, "recovery_requested": false, "strict_buffering": true})
			return
		}
		rt.rejectRealtimeOutput(event, "", event.OutputBlockReason, "")

	case RealtimeEventTranscriptOutput:
		rt.noteCompletionTranscript(event)
		if strings.TrimSpace(event.Transcript) != "" {
			rt.currentTiming().observed(event.ResponseID, event.ItemID)
		}
		if rt.suppressLeakedToolMarkup(event) {
			return
		}
		rt.trackTranscript("assistant", event)

	case RealtimeEventTranscriptInput:
		rt.noteCompletionTranscript(event)
		rt.noteBackendInput(event)
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
		rt.trackTranscript("user", event)
		if event.Final && strings.TrimSpace(event.Transcript) != "" {
			rt.resetToolMarkupTurn()
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
		if rt.completionRequested() {
			return
		}
		if rt.seenToolCall(rt.currentSession(), event.ToolCallID) {
			return
		}
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

	case RealtimeEventToolCallCancelled:
		rt.cancelRealtimeToolCall(rt.currentSession(), event.ToolCallID)
		if rt.pendingToolWork() {
			rt.setConversationState("working", event)
		} else if !rt.responseInProgress() {
			rt.setConversationState("listening", event)
		}

	case RealtimeEventUtteranceDone:
		rt.advanceAnonymousTranscriptTurns()
		rt.currentTiming().finish(event.ResponseID, event.ItemID, "completed")
		// Progress speech may finish while background reasoning/tools continue.
		// Never release response ownership, settle work, or return to listening.
		rt.finishToolMarkupResponse(event.ResponseID)
		if rt.pendingToolWork() {
			rt.setConversationState("working", event)
		} else {
			rt.setConversationState("thinking", event)
		}

	case RealtimeEventResponseDone:
		rt.advanceAnonymousTranscriptTurns()
		rt.currentTiming().finish(event.ResponseID, "", "completed")
		rt.recovery.Lock()
		if rt.recovery.hadInput {
			rt.recovery.retry = realtimeRecoveryRetry{}
			rt.recovery.retryAt = time.Time{}
		}
		rt.recovery.Unlock()
		hadToolBatch := rt.completeToolResponse(event.ResponseID)
		rt.responseFinished(rt.currentSession())
		rt.finishToolMarkupResponse(event.ResponseID)
		pendingTools := rt.pendingToolWork()
		if pendingTools {
			// Standard async models can finish a spoken turn while a tool from
			// an earlier response still runs. That does not finish Core's work.
			rt.setConversationState("working", event)
		} else if !hadToolBatch {
			rt.setConversationState("listening", event)
		}
		if !hadToolBatch && !pendingTools && !rt.responseInProgress() {
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
		rt.recordInputFailure(event.Err, "provider_event")

	case RealtimeEventSessionEnded:
		if event.Terminal {
			rt.lifecycleMu.Lock()
			rt.terminalReason = event.CloseReason
			rt.lifecycleMu.Unlock()
			rt.cancel()
		}
		rt.currentTiming().finish("", "", "session_ended")
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
	if rt.seenToolCall(session, event.ToolCallID) {
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
		context: rt.realtimeToolContext(session, event.ToolCallID),
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
	if !rt.capabilities.CoreExecution {
		rt.transcriptMu.Lock()
		rt.messages = append(rt.messages, callMessage)
		rt.transcriptMu.Unlock()
		if rt.Thinker.session != nil {
			_ = rt.Thinker.session.AppendMessage(callMessage, rt.iteration, TokenUsage{})
		}
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
	if event.Work != nil && rt.capabilities.CoreExecution {
		rt.consumeThreadWork(event)
		return
	}
	rt.Thinker.addEventExecutions(event.ExecutionIDs)
	session := rt.currentSession()
	if event.ToolResult != nil {
		rt.toolBatchMu.Lock()
		_, known := rt.toolCallBatches[event.ToolResult.CallID]
		rt.toolBatchMu.Unlock()
		if !known {
			if rt.recordCancelledToolOutcome(session, event.ToolResult.CallID) {
				// Cancellation cannot undo completed side effects. Preserve the
				// actual late outcome for audit/recovery, without waking the model
				// or sending a response for a cancelled provider call.
				message := rt.archiveToolResultMessage(Message{Role: "user", ToolResults: []ToolResult{*event.ToolResult}})
				rt.transcriptMu.Lock()
				rt.messages = append(rt.messages, message)
				rt.transcriptMu.Unlock()
				if rt.Thinker.session != nil {
					_ = rt.Thinker.session.AppendMessage(message, rt.iteration, TokenUsage{})
				}
			}
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
	messageText := note
	note = appendFileRefText(note, event.Parts)
	if rt.capabilities.CoreExecution && strings.TrimSpace(note) != "" {
		rt.noteBackendInput(RealtimeEvent{Transcript: note})
		rt.acceptDelegation(RealtimeEvent{DelegationID: "input_" + generateAudioToken(), ApplicationOwned: true})
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
		message := Message{Role: "user", Content: messageText, EventIDs: []string{event.ID}}
		if rt.capabilities.CoreExecution && len(event.Parts) > 0 {
			message.Parts = append([]ContentPart{{Type: "text", Text: messageText}}, cloneContentParts(event.Parts)...)
		}
		for _, part := range event.Parts {
			if part.Type == "file_ref" && !rt.capabilities.CoreExecution {
				if len(message.Parts) == 0 {
					message.Parts = []ContentPart{{Type: "text", Text: messageText}}
				}
				message.Parts = append(message.Parts, cloneContentParts([]ContentPart{part})...)
			}
		}
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
	if rt.capabilities.ContinuousConversation {
		kind := "commentary"
		if strings.HasPrefix(message, "[Internal") {
			kind = "instructions"
		}
		return rt.appendRealtimeContext(RealtimeContextUpdate{Kind: kind, Content: message})
	}
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

func (rt *RealtimeThinker) acknowledgePlaybackClear(item string) {
	rt.outputMu.Lock()
	requested := rt.playbackClearRequested[item]
	delete(rt.playbackClearRequested, item)
	rt.outputMu.Unlock()
	if requested.IsZero() {
		return
	}
	rt.emit("realtime.playback_clear_acknowledged", map[string]any{"item_id": item, "interrupt_to_clear_ack_ms": float64(time.Since(requested)) / float64(time.Millisecond), "source": "client_reported_clear_received"})
}

// Caller interruption cancels the snapshot of existing playback, including an
// older segment still in the player. Later speech gets a new identity and is
// unaffected. Content checks continue to cancel only their specific item.
func (rt *RealtimeThinker) clearCallerInterruptedPlayback(reason string, cancelProvider, requireAck bool) {
	rt.outputMu.Lock()
	current := rt.outputItemID
	ids := rt.delivery.pendingPlaybackIDs()
	rt.outputMu.Unlock()
	for _, item := range ids {
		if item != current {
			rt.interruptPlayback(reason, item, false, requireAck)
		}
	}
	rt.interruptPlayback(reason, current, cancelProvider, requireAck)
}
