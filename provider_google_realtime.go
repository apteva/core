package core

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

const googleRealtimeEndpoint = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

// GoogleRealtimeProvider implements the Gemini Live bidirectional WebSocket
// API. Gemini uses its own BidiGenerateContent protocol, so only the generic
// RealtimeProvider boundary is shared with OpenAI/xAI sessions.
type GoogleRealtimeProvider struct {
	apiKey       string
	models       map[ModelTier]string
	endpoint     string
	defaultVoice string
}

func NewGoogleRealtimeProvider(apiKey string) *GoogleRealtimeProvider {
	return &GoogleRealtimeProvider{
		apiKey:       apiKey,
		endpoint:     googleRealtimeEndpoint,
		defaultVoice: "Kore",
		models: map[ModelTier]string{
			ModelLarge:  "gemini-3.1-flash-live-preview",
			ModelMedium: "gemini-3.1-flash-live-preview",
			ModelSmall:  "gemini-3.1-flash-live-preview",
		},
	}
}

func (p *GoogleRealtimeProvider) Name() string { return "google-realtime" }

func (p *GoogleRealtimeProvider) Models() map[ModelTier]string {
	out := make(map[ModelTier]string, len(p.models))
	for tier, model := range p.models {
		out[tier] = model
	}
	return out
}

func (p *GoogleRealtimeProvider) Pricing(string) RealtimePricing {
	return RealtimePricing{
		TextInput: 0.75, TextOutput: 4.50,
		AudioInput: 3.00, AudioOutput: 12.00,
	}
}

func (p *GoogleRealtimeProvider) DefaultVoice() string {
	if strings.TrimSpace(p.defaultVoice) == "" {
		return "Kore"
	}
	return p.defaultVoice
}

// Gemini Live provides transcription as part of the session setup rather
// than through a separately named transcription model.
func (p *GoogleRealtimeProvider) DefaultTranscriptionModel() string { return "" }

func (p *GoogleRealtimeProvider) applyRealtimeConfig(config ProviderConfig) {
	applyRealtimeModelAndVoiceConfig(p.models, &p.defaultVoice, config)
}

func (p *GoogleRealtimeProvider) Open(ctx context.Context, opts RealtimeSessionOpts) (RealtimeSession, error) {
	return openGoogleRealtimeSession(ctx, p, opts)
}

func googleLiveThinkingLevel(reasoning string) string {
	switch strings.ToLower(strings.TrimSpace(reasoning)) {
	case "minimal", "none":
		return "minimal"
	case "low":
		return "low"
	case "medium":
		return "medium"
	case "high", "xhigh":
		return "high"
	default:
		return ""
	}
}

func googleLiveTools(tools []NativeTool) []map[string]any {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		declaration := map[string]any{
			"name":       tool.Name,
			"parameters": geminiToolParameters(tool.Parameters),
		}
		if strings.TrimSpace(tool.Description) != "" {
			declaration["description"] = tool.Description
		}
		declarations = append(declarations, declaration)
	}
	return []map[string]any{{"functionDeclarations": declarations}}
}

func buildGoogleLiveSetup(opts RealtimeSessionOpts, defaultVoice string) ([]byte, error) {
	model := strings.TrimSpace(opts.Model)
	if model == "" {
		return nil, fmt.Errorf("google-realtime: realtime model is required")
	}
	if opts.AudioInFmt != "" && opts.AudioInFmt != AudioPCM16 {
		return nil, fmt.Errorf("google-realtime: input format %q is unsupported; use pcm16", opts.AudioInFmt)
	}
	if opts.AudioOutFmt != "" && opts.AudioOutFmt != AudioPCM16 {
		return nil, fmt.Errorf("google-realtime: output format %q is unsupported; use pcm16", opts.AudioOutFmt)
	}
	if opts.AudioOutRate != 0 && opts.AudioOutRate != 24000 {
		return nil, fmt.Errorf("google-realtime: output rate must be 24000 Hz")
	}
	voice := strings.TrimSpace(opts.Voice)
	if voice == "" {
		voice = defaultVoice
	}
	generation := map[string]any{
		"responseModalities": []string{"AUDIO"},
		"speechConfig": map[string]any{
			"voiceConfig": map[string]any{
				"prebuiltVoiceConfig": map[string]any{"voiceName": voice},
			},
		},
	}
	if level := googleLiveThinkingLevel(opts.Reasoning); level != "" {
		generation["thinkingConfig"] = map[string]any{"thinkingLevel": level}
	}
	normalizedTurnDetection, err := opts.TurnDetection.normalized()
	if err != nil {
		return nil, fmt.Errorf("google-realtime turn detection: %w", err)
	}
	turnDetection := normalizedTurnDetection.resolved()
	automaticActivityDetection := map[string]any{"disabled": false}
	switch turnDetection.StartSensitivity {
	case RealtimeSensitivityHigh:
		automaticActivityDetection["startOfSpeechSensitivity"] = "START_SENSITIVITY_HIGH"
	case RealtimeSensitivityLow:
		automaticActivityDetection["startOfSpeechSensitivity"] = "START_SENSITIVITY_LOW"
	}
	if turnDetection.PrefixPaddingMS > 0 {
		automaticActivityDetection["prefixPaddingMs"] = turnDetection.PrefixPaddingMS
	}
	switch turnDetection.EndSensitivity {
	case RealtimeSensitivityHigh:
		automaticActivityDetection["endOfSpeechSensitivity"] = "END_SENSITIVITY_HIGH"
	case RealtimeSensitivityLow:
		automaticActivityDetection["endOfSpeechSensitivity"] = "END_SENSITIVITY_LOW"
	}
	if turnDetection.SilenceDurationMS > 0 {
		automaticActivityDetection["silenceDurationMs"] = turnDetection.SilenceDurationMS
	}
	activityHandling := "START_OF_ACTIVITY_INTERRUPTS"
	if turnDetection.Interruption == RealtimeInterruptionDisable {
		activityHandling = "NO_INTERRUPTION"
	}
	setup := map[string]any{
		"model":             "models/" + strings.TrimPrefix(model, "models/"),
		"generationConfig":  generation,
		"systemInstruction": map[string]any{"parts": []map[string]any{{"text": opts.Instructions}}},
		"realtimeInputConfig": map[string]any{
			"automaticActivityDetection": automaticActivityDetection,
			"activityHandling":           activityHandling,
			"turnCoverage":               "TURN_INCLUDES_ONLY_ACTIVITY",
		},
		"contextWindowCompression": map[string]any{"slidingWindow": map[string]any{}},
		// Every session begins in history-seeding mode. RealtimeThinker calls
		// RestoreConversation even for an empty initial history to unlock live
		// input after setupComplete.
		"historyConfig": map[string]any{"initialHistoryInClientContent": true},
	}
	if tools := googleLiveTools(opts.Tools); len(tools) > 0 {
		setup["tools"] = tools
	}
	if opts.TranscribeInput {
		setup["inputAudioTranscription"] = map[string]any{}
	}
	setup["outputAudioTranscription"] = map[string]any{}
	return json.Marshal(map[string]any{"setup": setup})
}

type googleLiveInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type googleLivePart struct {
	Text       string                `json:"text,omitempty"`
	Thought    bool                  `json:"thought,omitempty"`
	InlineData *googleLiveInlineData `json:"inlineData,omitempty"`
}

type googleLiveTranscription struct {
	Text string `json:"text"`
}

type googleLiveServerContent struct {
	ModelTurn *struct {
		Parts []googleLivePart `json:"parts"`
	} `json:"modelTurn,omitempty"`
	GenerationComplete  bool                     `json:"generationComplete,omitempty"`
	TurnComplete        bool                     `json:"turnComplete,omitempty"`
	Interrupted         bool                     `json:"interrupted,omitempty"`
	InputTranscription  *googleLiveTranscription `json:"inputTranscription,omitempty"`
	OutputTranscription *googleLiveTranscription `json:"outputTranscription,omitempty"`
}

type googleLiveFunctionCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type googleLiveTokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

type googleLiveUsage struct {
	PromptTokenCount        int                     `json:"promptTokenCount"`
	CachedContentTokenCount int                     `json:"cachedContentTokenCount"`
	ResponseTokenCount      int                     `json:"responseTokenCount"`
	ThoughtsTokenCount      int                     `json:"thoughtsTokenCount"`
	TotalTokenCount         int                     `json:"totalTokenCount"`
	PromptTokensDetails     []googleLiveTokenDetail `json:"promptTokensDetails"`
	CacheTokensDetails      []googleLiveTokenDetail `json:"cacheTokensDetails"`
	ResponseTokensDetails   []googleLiveTokenDetail `json:"responseTokensDetails"`
}

type googleLiveServerMessage struct {
	SetupComplete json.RawMessage          `json:"setupComplete,omitempty"`
	ServerContent *googleLiveServerContent `json:"serverContent,omitempty"`
	ToolCall      *struct {
		FunctionCalls []googleLiveFunctionCall `json:"functionCalls"`
	} `json:"toolCall,omitempty"`
	ToolCallCancellation *struct {
		IDs []string `json:"ids"`
	} `json:"toolCallCancellation,omitempty"`
	GoAway *struct {
		TimeLeft string `json:"timeLeft"`
	} `json:"goAway,omitempty"`
	SessionResumptionUpdate *struct {
		NewHandle string `json:"newHandle"`
		Resumable bool   `json:"resumable"`
	} `json:"sessionResumptionUpdate,omitempty"`
	UsageMetadata *googleLiveUsage `json:"usageMetadata,omitempty"`
	Error         *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

type googleLiveFunctionResponse struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type googleRealtimeSession struct {
	conn   net.Conn
	events chan RealtimeEvent
	outbox chan realtimeOutboundFrame
	done   chan struct{}
	ready  chan error

	closeOnce sync.Once
	readyOnce sync.Once
	lifecycle realtimeSessionLifecycle
	seq       atomic.Uint64
	dropped   atomic.Uint64

	inputRate int

	mu                  sync.Mutex
	currentResponseID   string
	inputTranscript     string
	outputTranscript    string
	lastUsage           RealtimeUsage
	callNames           map[string]string
	pendingResponses    []googleLiveFunctionResponse
	configFingerprint   string
	restartAfterTurn    bool
	turnOutputSinceTool bool
}

func openGoogleRealtimeSession(ctx context.Context, provider *GoogleRealtimeProvider, opts RealtimeSessionOpts) (*googleRealtimeSession, error) {
	setup, err := buildGoogleLiveSetup(opts, provider.DefaultVoice())
	if err != nil {
		return nil, err
	}
	endpoint, err := url.Parse(provider.endpoint)
	if err != nil {
		return nil, fmt.Errorf("google-realtime endpoint: %w", err)
	}
	query := endpoint.Query()
	query.Set("key", provider.apiKey)
	endpoint.RawQuery = query.Encode()

	dialer := ws.Dialer{Timeout: 10 * time.Second}
	conn, _, _, err := dialer.Dial(ctx, endpoint.String())
	if err != nil {
		return nil, fmt.Errorf("google-realtime dial: %w", err)
	}
	inputRate := opts.AudioInRate
	if inputRate == 0 {
		inputRate = 16000
	}
	s := &googleRealtimeSession{
		conn: conn, events: make(chan RealtimeEvent, realtimeEventBuffer),
		outbox: make(chan realtimeOutboundFrame, realtimeOutboxBuffer), done: make(chan struct{}), ready: make(chan error, 1),
		inputRate: inputRate, callNames: map[string]string{},
		configFingerprint: googleRealtimeConfigFingerprint(opts.Instructions, opts.Tools),
	}
	s.outbox <- realtimeOutboundFrame{op: ws.OpText, data: setup}
	s.lifecycle.start(s.events, s.readLoop, s.writeLoop, s.pingLoop)
	go func() {
		select {
		case <-ctx.Done():
			_ = s.Close()
		case <-s.done:
		}
	}()

	select {
	case err := <-s.ready:
		if err != nil {
			_ = s.Close()
			return nil, err
		}
		return s, nil
	case <-ctx.Done():
		_ = s.Close()
		return nil, fmt.Errorf("google-realtime setup: %w", ctx.Err())
	case <-time.After(10 * time.Second):
		_ = s.Close()
		return nil, fmt.Errorf("google-realtime setup timed out")
	}
}

func googleRealtimeConfigFingerprint(instructions string, tools []NativeTool) string {
	encoded, _ := json.Marshal(struct {
		Instructions string       `json:"instructions"`
		Tools        []NativeTool `json:"tools"`
	}{instructions, tools})
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func (s *googleRealtimeSession) signalReady(err error) {
	s.readyOnce.Do(func() { s.ready <- err })
}

func (s *googleRealtimeSession) nextResponseID() string {
	return fmt.Sprintf("gemini_live_%d", s.seq.Add(1))
}

func (s *googleRealtimeSession) currentOrNextResponseID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.currentResponseID == "" {
		s.currentResponseID = s.nextResponseID()
	}
	return s.currentResponseID
}

func (s *googleRealtimeSession) finishResponseID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	id := s.currentResponseID
	if id == "" {
		id = s.nextResponseID()
	}
	s.currentResponseID = ""
	return id
}

func googleTokenCount(details []googleLiveTokenDetail, modality string) int {
	total := 0
	for _, detail := range details {
		if strings.EqualFold(detail.Modality, modality) {
			total += detail.TokenCount
		}
	}
	return total
}

func googleRealtimeUsage(usage *googleLiveUsage) RealtimeUsage {
	if usage == nil {
		return RealtimeUsage{}
	}
	result := RealtimeUsage{
		TotalTokens:       usage.TotalTokenCount,
		InputTokens:       usage.PromptTokenCount,
		OutputTokens:      usage.ResponseTokenCount,
		TextInputTokens:   googleTokenCount(usage.PromptTokensDetails, "TEXT"),
		TextCachedTokens:  googleTokenCount(usage.CacheTokensDetails, "TEXT"),
		TextOutputTokens:  googleTokenCount(usage.ResponseTokensDetails, "TEXT") + usage.ThoughtsTokenCount,
		AudioInputTokens:  googleTokenCount(usage.PromptTokensDetails, "AUDIO"),
		AudioCachedTokens: googleTokenCount(usage.CacheTokensDetails, "AUDIO"),
		AudioOutputTokens: googleTokenCount(usage.ResponseTokensDetails, "AUDIO"),
	}
	if result.TextInputTokens+result.AudioInputTokens == 0 {
		result.TextInputTokens = usage.PromptTokenCount
	}
	if result.TextOutputTokens+result.AudioOutputTokens == usage.ThoughtsTokenCount {
		result.TextOutputTokens += usage.ResponseTokenCount
	}
	if result.TextCachedTokens+result.AudioCachedTokens == 0 && usage.CachedContentTokenCount > 0 {
		result.TextCachedTokens = usage.CachedContentTokenCount
	}
	return result
}

// appendGoogleTranscript accepts both cumulative and incremental provider
// updates while avoiding duplicated text when the provider revises a partial.
func appendGoogleTranscript(current, update string) string {
	if update == "" {
		return current
	}
	if current == "" || strings.HasPrefix(update, current) {
		return update
	}
	if strings.HasSuffix(current, update) {
		return current
	}
	maxOverlap := min(len(current), len(update))
	for overlap := maxOverlap; overlap > 0; overlap-- {
		if current[len(current)-overlap:] == update[:overlap] {
			return current + update[overlap:]
		}
	}
	return current + update
}

func (s *googleRealtimeSession) readLoop() {
	for {
		data, op, err := wsutil.ReadServerData(s.conn)
		if err != nil {
			select {
			case <-s.done:
				return
			default:
			}
			signalErr := fmt.Errorf("google-realtime read: %w", err)
			s.signalReady(signalErr)
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: signalErr})
			s.emitControl(RealtimeEvent{Type: RealtimeEventSessionEnded, DroppedAudio: s.dropped.Load()})
			_ = s.Close()
			return
		}
		// Gemini currently returns its JSON protocol messages in binary
		// WebSocket frames, while local/test servers may use text frames.
		if op != ws.OpText && op != ws.OpBinary {
			continue
		}
		var message googleLiveServerMessage
		if err := json.Unmarshal(data, &message); err != nil {
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("google-realtime decode: %w", err)})
			continue
		}
		s.translate(&message)
	}
}

func (s *googleRealtimeSession) translate(message *googleLiveServerMessage) {
	if message.Error != nil {
		err := fmt.Errorf("google-realtime: %s/%d: %s", message.Error.Status, message.Error.Code, message.Error.Message)
		s.signalReady(err)
		s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: err})
		return
	}
	if message.SetupComplete != nil {
		s.signalReady(nil)
	}
	if message.UsageMetadata != nil {
		s.mu.Lock()
		s.lastUsage = googleRealtimeUsage(message.UsageMetadata)
		s.mu.Unlock()
	}
	if message.ToolCallCancellation != nil {
		s.mu.Lock()
		for _, id := range message.ToolCallCancellation.IDs {
			delete(s.callNames, id)
		}
		s.mu.Unlock()
	}
	if message.ServerContent != nil {
		content := message.ServerContent
		if content.Interrupted {
			s.emitControl(RealtimeEvent{Type: RealtimeEventSpeechStarted})
		}
		responseID := s.currentOrNextResponseID()
		if content.ModelTurn != nil {
			for _, part := range content.ModelTurn.Parts {
				if part.Thought {
					continue
				}
				if part.InlineData != nil && strings.HasPrefix(strings.ToLower(part.InlineData.MimeType), "audio/") {
					pcm, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
					if err != nil {
						s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("google-realtime audio: %w", err)})
						continue
					}
					s.mu.Lock()
					s.turnOutputSinceTool = true
					s.mu.Unlock()
					s.emitAudio(RealtimeEvent{Type: RealtimeEventAudioOut, Audio: pcm, ItemID: responseID})
				}
			}
		}
		if content.InputTranscription != nil && content.InputTranscription.Text != "" {
			s.mu.Lock()
			s.inputTranscript = appendGoogleTranscript(s.inputTranscript, content.InputTranscription.Text)
			partial := s.inputTranscript
			s.mu.Unlock()
			s.emitAudio(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: partial})
		}
		if content.OutputTranscription != nil && content.OutputTranscription.Text != "" {
			s.mu.Lock()
			s.outputTranscript = appendGoogleTranscript(s.outputTranscript, content.OutputTranscription.Text)
			partial := s.outputTranscript
			s.turnOutputSinceTool = true
			s.mu.Unlock()
			s.emitAudio(RealtimeEvent{Type: RealtimeEventTranscriptOutput, Transcript: partial})
		}
		if content.TurnComplete {
			s.finishTurn()
		}
	}
	if message.ToolCall != nil && len(message.ToolCall.FunctionCalls) > 0 {
		responseID := s.currentOrNextResponseID()
		s.mu.Lock()
		for _, call := range message.ToolCall.FunctionCalls {
			s.callNames[call.ID] = call.Name
		}
		s.mu.Unlock()
		for _, call := range message.ToolCall.FunctionCalls {
			args, _ := json.Marshal(call.Args)
			s.emitControl(RealtimeEvent{
				Type: RealtimeEventToolCall, ResponseID: responseID,
				ToolCallID: call.ID, ToolName: call.Name, ToolArgs: string(args),
			})
		}
		// Gemini's dedicated toolCall message ends the function-selection
		// response. The queued tool responses themselves trigger continuation;
		// RequestResponse only flushes them as one atomic batch.
		s.finishTurn()
	}
}

func (s *googleRealtimeSession) finishTurn() {
	s.mu.Lock()
	input, output := strings.TrimSpace(s.inputTranscript), strings.TrimSpace(s.outputTranscript)
	s.inputTranscript, s.outputTranscript = "", ""
	usage := s.lastUsage
	s.lastUsage = RealtimeUsage{}
	restart := s.restartAfterTurn
	s.restartAfterTurn = false
	s.turnOutputSinceTool = false
	s.mu.Unlock()
	if input != "" {
		s.emitControl(RealtimeEvent{Type: RealtimeEventTranscriptInput, Transcript: input, Final: true})
	}
	if output != "" {
		s.emitControl(RealtimeEvent{Type: RealtimeEventTranscriptOutput, Transcript: output, Final: true})
	}
	s.emitControl(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: s.finishResponseID(), Usage: usage})
	if restart {
		_ = s.Close()
	}
}

func (s *googleRealtimeSession) writeLoop() {
	for {
		select {
		case <-s.done:
			return
		case frame := <-s.outbox:
			if err := wsutil.WriteClientMessage(s.conn, frame.op, frame.data); err != nil {
				s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("google-realtime write: %w", err)})
				_ = s.Close()
				return
			}
		}
	}
}

func (s *googleRealtimeSession) pingLoop() {
	ticker := time.NewTicker(realtimePingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			_ = s.enqueueFrame(realtimeOutboundFrame{op: ws.OpPing})
		}
	}
}

func (s *googleRealtimeSession) enqueueFrame(frame realtimeOutboundFrame) error {
	select {
	case <-s.done:
		return fmt.Errorf("google-realtime session closed")
	default:
	}
	select {
	case s.outbox <- frame:
		return nil
	case <-s.done:
		return fmt.Errorf("google-realtime session closed")
	default:
		return fmt.Errorf("google-realtime outbox full")
	}
}

func (s *googleRealtimeSession) enqueue(payload map[string]any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.enqueueFrame(realtimeOutboundFrame{op: ws.OpText, data: data})
}

func (s *googleRealtimeSession) emitAudio(event RealtimeEvent) {
	select {
	case s.events <- event:
	default:
		s.dropped.Add(1)
	}
}

func (s *googleRealtimeSession) emitControl(event RealtimeEvent) {
	select {
	case s.events <- event:
	case <-s.done:
	}
}

func (s *googleRealtimeSession) SendAudio(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	return s.enqueue(map[string]any{
		"realtimeInput": map[string]any{
			"audio": map[string]any{
				"mimeType": fmt.Sprintf("audio/pcm;rate=%d", s.inputRate),
				"data":     base64.StdEncoding.EncodeToString(pcm),
			},
		},
	})
}

func (s *googleRealtimeSession) SendText(role, text string) error {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if role != "" && role != "user" {
		text = "[" + role + "] " + text
	}
	return s.enqueue(map[string]any{"realtimeInput": map[string]any{"text": text}})
}

func googleLiveResultValue(result string) any {
	var parsed any
	if json.Unmarshal([]byte(result), &parsed) == nil {
		return parsed
	}
	return result
}

func (s *googleRealtimeSession) SendToolResult(callID, result string, isError bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	name := s.callNames[callID]
	if name == "" {
		return fmt.Errorf("google-realtime: unknown tool call %q", callID)
	}
	response := map[string]any{"result": googleLiveResultValue(result)}
	if isError {
		response = map[string]any{"error": result}
	}
	s.pendingResponses = append(s.pendingResponses, googleLiveFunctionResponse{
		ID: callID, Name: name, Response: response,
	})
	return nil
}

func (s *googleRealtimeSession) RequestResponse() error {
	s.mu.Lock()
	responses := append([]googleLiveFunctionResponse(nil), s.pendingResponses...)
	s.pendingResponses = nil
	for _, response := range responses {
		delete(s.callNames, response.ID)
	}
	s.mu.Unlock()
	if len(responses) == 0 {
		return nil
	}
	return s.enqueue(map[string]any{
		"toolResponse": map[string]any{"functionResponses": responses},
	})
}

func (s *googleRealtimeSession) UpdateConfiguration(instructions string, tools []NativeTool) error {
	next := googleRealtimeConfigFingerprint(instructions, tools)
	s.mu.Lock()
	defer s.mu.Unlock()
	if next == s.configFingerprint {
		return nil
	}
	// Gemini setup is immutable. Finish the current tool continuation, then
	// close cleanly; RealtimeThinker reopens with its latest prompt and tools
	// and restores bounded conversation history.
	s.configFingerprint = next
	s.restartAfterTurn = true
	return nil
}

func (s *googleRealtimeSession) RestoreConversation(messages []Message) error {
	turns := make([]map[string]any, 0, len(messages))
	for _, message := range messages {
		text := strings.TrimSpace(message.Content)
		if text == "" {
			continue
		}
		role := "user"
		if message.Role == "assistant" {
			role = "model"
		} else if message.Role != "user" {
			continue
		}
		turns = append(turns, map[string]any{
			"role": role, "parts": []map[string]any{{"text": text}},
		})
	}
	content := map[string]any{"turnComplete": true}
	if len(turns) > 0 {
		content["turns"] = turns
	}
	return s.enqueue(map[string]any{"clientContent": content})
}

// Gemini interrupts output automatically when new realtime activity begins.
// There is no separate cancel or conversation-truncation client event.
func (s *googleRealtimeSession) Interrupt() error             { return nil }
func (s *googleRealtimeSession) Truncate(string, int) error   { return nil }
func (s *googleRealtimeSession) Events() <-chan RealtimeEvent { return s.events }
func (s *googleRealtimeSession) Close() error {
	var err error
	s.closeOnce.Do(func() {
		close(s.done)
		if s.conn != nil {
			err = s.conn.Close()
		}
	})
	return err
}

var _ RealtimeProvider = (*GoogleRealtimeProvider)(nil)
var _ configurableRealtimeProvider = (*GoogleRealtimeProvider)(nil)
var _ RealtimeSession = (*googleRealtimeSession)(nil)
