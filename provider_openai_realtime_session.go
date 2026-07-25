package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

const (
	realtimeEventBuffer  = 256
	realtimeOutboxBuffer = 256
	realtimePingInterval = 30 * time.Second
)

type realtimeOutboundFrame struct {
	op   ws.OpCode
	data []byte
}

// openaiRealtimeSession is the shared transport for providers implementing the
// OpenAI-compatible realtime event protocol. Provider-specific session shapes
// remain injected callbacks; the thread runtime only sees RealtimeSession.
// The outbox is never closed: Close signals done and closes the connection,
// eliminating the enqueue-vs-close send-on-closed-channel race.
type openaiRealtimeSession struct {
	conn   net.Conn
	events chan RealtimeEvent
	outbox chan realtimeOutboundFrame
	done   chan struct{}

	providerName             string
	buildConfigurationUpdate func(string, []NativeTool) map[string]any
	audioInBytes             atomic.Uint64
	audioOutBytes            atomic.Uint64
	textInputMessages        atomic.Uint64
	audioInBytesPerSecond    float64
	audioOutBytesPerSecond   float64
	closeOnce                sync.Once
	lifecycle                realtimeSessionLifecycle
	droppedAudio             atomic.Uint64
	itemPhases               map[string]string
}

type openAICompatibleRealtimeConfig struct {
	providerName             string
	apiKey                   string
	endpoint                 string
	defaultVoice             string
	safetyIdentifierHeader   string
	buildSessionUpdate       func(RealtimeSessionOpts, string) ([]byte, error)
	buildConfigurationUpdate func(string, []NativeTool) map[string]any
}

type openaiRealtimeUsage struct {
	TotalTokens  int `json:"total_tokens"`
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	InputDetails struct {
		TextTokens    int `json:"text_tokens"`
		AudioTokens   int `json:"audio_tokens"`
		CachedTokens  int `json:"cached_tokens"`
		CachedDetails struct {
			TextTokens  int `json:"text_tokens"`
			AudioTokens int `json:"audio_tokens"`
		} `json:"cached_tokens_details"`
	} `json:"input_token_details"`
	OutputDetails struct {
		TextTokens  int `json:"text_tokens"`
		AudioTokens int `json:"audio_tokens"`
	} `json:"output_token_details"`
}

type openaiRealtimeOutputItem struct {
	ID    string `json:"id,omitempty"`
	Type  string `json:"type,omitempty"`
	Phase string `json:"phase,omitempty"`
}

type openaiRealtimeEvent struct {
	Type         string                   `json:"type"`
	EventID      string                   `json:"event_id,omitempty"`
	ResponseID   string                   `json:"response_id,omitempty"`
	ItemID       string                   `json:"item_id,omitempty"`
	Delta        string                   `json:"delta,omitempty"`
	Transcript   string                   `json:"transcript,omitempty"`
	Text         string                   `json:"text,omitempty"`
	CallID       string                   `json:"call_id,omitempty"`
	Name         string                   `json:"name,omitempty"`
	Arguments    string                   `json:"arguments,omitempty"`
	Phase        string                   `json:"phase,omitempty"`
	AudioStartMS int                      `json:"audio_start_ms,omitempty"`
	Item         openaiRealtimeOutputItem `json:"item,omitempty"`

	Response struct {
		ID     string                     `json:"id"`
		Status string                     `json:"status"`
		Usage  openaiRealtimeUsage        `json:"usage"`
		Output []openaiRealtimeOutputItem `json:"output,omitempty"`
	} `json:"response,omitempty"`

	Error struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (p *OpenAIRealtimeProvider) openSession(ctx context.Context, opts RealtimeSessionOpts) (*openaiRealtimeSession, error) {
	return openAICompatibleRealtimeSession(ctx, opts, openAICompatibleRealtimeConfig{
		providerName: "openai-realtime", apiKey: p.apiKey, endpoint: p.endpoint,
		defaultVoice: p.DefaultVoice(), safetyIdentifierHeader: "OpenAI-Safety-Identifier",
		buildSessionUpdate: buildSessionUpdate,
		buildConfigurationUpdate: func(instructions string, tools []NativeTool) map[string]any {
			return map[string]any{
				"type": "session.update",
				"session": map[string]any{
					"type": "realtime", "instructions": instructions, "tools": sessionTools(tools),
				},
			}
		},
	})
}

func openAICompatibleRealtimeSession(ctx context.Context, opts RealtimeSessionOpts, config openAICompatibleRealtimeConfig) (*openaiRealtimeSession, error) {
	model := opts.Model
	if model == "" {
		return nil, fmt.Errorf("%s: realtime model is required", config.providerName)
	}
	endpoint, err := url.Parse(config.endpoint)
	if err != nil {
		return nil, fmt.Errorf("realtime endpoint: %w", err)
	}
	query := endpoint.Query()
	query.Set("model", model)
	endpoint.RawQuery = query.Encode()

	header := http.Header{"Authorization": []string{"Bearer " + config.apiKey}}
	if opts.SafetyIdentifier != "" && config.safetyIdentifierHeader != "" {
		header.Set(config.safetyIdentifierHeader, opts.SafetyIdentifier)
	}
	dialer := ws.Dialer{
		Timeout: 10 * time.Second,
		Header:  ws.HandshakeHeaderHTTP(header),
	}
	conn, _, _, err := dialer.Dial(ctx, endpoint.String())
	if err != nil {
		return nil, fmt.Errorf("realtime dial: %w", err)
	}

	s := &openaiRealtimeSession{
		conn: conn, events: make(chan RealtimeEvent, realtimeEventBuffer),
		outbox: make(chan realtimeOutboundFrame, realtimeOutboxBuffer), done: make(chan struct{}),
		providerName: config.providerName, buildConfigurationUpdate: config.buildConfigurationUpdate,
		itemPhases:             make(map[string]string),
		audioInBytesPerSecond:  audioBytesPerSecond(opts.AudioInFmt, opts.AudioInRate),
		audioOutBytesPerSecond: audioBytesPerSecond(opts.AudioOutFmt, opts.AudioOutRate),
	}

	if config.buildSessionUpdate == nil {
		_ = conn.Close()
		return nil, fmt.Errorf("%s: session update builder is required", config.providerName)
	}
	sessUpdate, err := config.buildSessionUpdate(opts, config.defaultVoice)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	s.outbox <- realtimeOutboundFrame{op: ws.OpText, data: sessUpdate}

	s.lifecycle.start(s.events, s.readLoop, s.writeLoop, s.pingLoop)
	go func() {
		select {
		case <-ctx.Done():
			_ = s.Close()
		case <-s.done:
		}
	}()
	return s, nil
}

func sessionTools(tools []NativeTool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		out = append(out, map[string]any{
			"type": "function", "name": tool.Name,
			"description": tool.Description, "parameters": tool.Parameters,
		})
	}
	return out
}

func openAIAudioFormat(format AudioFormat, rate int) map[string]any {
	if rate <= 0 {
		rate = 24000
	}
	var config map[string]any
	switch format {
	case AudioG711ULaw:
		config = map[string]any{"type": "audio/pcmu"}
	case AudioG711ALaw:
		config = map[string]any{"type": "audio/pcma"}
	default:
		config = map[string]any{"type": "audio/pcm", "rate": rate}
	}
	return config
}

func audioBytesPerSecond(format AudioFormat, rate int) float64 {
	switch format {
	case AudioG711ULaw, AudioG711ALaw:
		if rate <= 0 {
			rate = 8000
		}
		return float64(rate)
	default:
		if rate <= 0 {
			rate = 24000
		}
		return float64(rate * 2) // signed 16-bit mono PCM
	}
}

func buildSessionUpdate(opts RealtimeSessionOpts, defaultVoice string) ([]byte, error) {
	voice := opts.Voice
	if voice == "" {
		voice = defaultVoice
	}
	inFmt := opts.AudioInFmt
	if inFmt == "" {
		inFmt = AudioPCM16
	}
	outFmt := opts.AudioOutFmt
	if outFmt == "" {
		outFmt = AudioPCM16
	}

	normalizedTurnDetection, err := opts.TurnDetection.normalized()
	if err != nil {
		return nil, fmt.Errorf("openai-realtime turn detection: %w", err)
	}
	resolvedTurnDetection := normalizedTurnDetection.resolved()
	turnDetection := map[string]any{
		"type": "server_vad", "create_response": true, "interrupt_response": true,
	}
	if resolvedTurnDetection.PrefixPaddingMS > 0 {
		turnDetection["prefix_padding_ms"] = resolvedTurnDetection.PrefixPaddingMS
	}
	if resolvedTurnDetection.SilenceDurationMS > 0 {
		turnDetection["silence_duration_ms"] = resolvedTurnDetection.SilenceDurationMS
	}
	if resolvedTurnDetection.Interruption == RealtimeInterruptionDisable {
		turnDetection["interrupt_response"] = false
	}
	input := map[string]any{
		"format":         openAIAudioFormat(inFmt, opts.AudioInRate),
		"turn_detection": turnDetection,
	}
	if opts.TranscribeInput {
		model := opts.TranscriptionModel
		if model == "" {
			model = "gpt-4o-mini-transcribe"
		}
		input["transcription"] = map[string]any{"model": model}
	}

	session := map[string]any{
		"type":              "realtime",
		"model":             opts.Model,
		"instructions":      opts.Instructions,
		"output_modalities": []string{"audio"},
		"audio": map[string]any{
			"input": input,
			"output": map[string]any{
				"format": openAIAudioFormat(outFmt, opts.AudioOutRate),
				"voice":  voice,
			},
		},
		"tools":       sessionTools(opts.Tools),
		"tool_choice": "auto",
		"truncation":  map[string]any{"type": "retention_ratio", "retention_ratio": 0.8},
	}
	if effort := strings.TrimSpace(opts.Reasoning); effort != "" && effort != "auto" && effort != "none" {
		session["reasoning"] = map[string]any{"effort": effort}
	}
	return json.Marshal(map[string]any{"type": "session.update", "session": session})
}

func (s *openaiRealtimeSession) readLoop() {
	for {
		data, op, err := wsutil.ReadServerData(s.conn)
		if err != nil {
			select {
			case <-s.done:
				return
			default:
			}
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("ws read: %w", err)})
			s.emitControl(RealtimeEvent{Type: RealtimeEventSessionEnded, DroppedAudio: s.droppedAudio.Load()})
			_ = s.Close()
			return
		}
		if op != ws.OpText {
			continue
		}
		var event openaiRealtimeEvent
		if err := json.Unmarshal(data, &event); err != nil {
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("decode event: %w", err)})
			continue
		}
		s.translate(&event)
	}
}

func (s *openaiRealtimeSession) translate(event *openaiRealtimeEvent) {
	base := RealtimeEvent{ResponseID: event.ResponseID, ItemID: event.ItemID, Phase: event.Phase}
	if base.ItemID != "" && base.Phase == "" {
		base.Phase = s.itemPhases[base.ItemID]
	}
	switch event.Type {
	case "response.created":
		base.Type = RealtimeEventResponseStarted
		if event.Response.ID != "" {
			base.ResponseID = event.Response.ID
		}
		s.emitControl(base)
	case "response.output_item.added":
		itemID := event.Item.ID
		if itemID == "" {
			itemID = event.ItemID
		}
		phase := strings.TrimSpace(event.Item.Phase)
		if phase == "" {
			phase = strings.TrimSpace(event.Phase)
		}
		if itemID != "" && phase != "" {
			if s.itemPhases == nil {
				s.itemPhases = make(map[string]string)
			}
			s.itemPhases[itemID] = phase
		}
	case "response.output_item.done":
		itemID := event.Item.ID
		if itemID == "" {
			itemID = event.ItemID
		}
		if itemID != "" {
			delete(s.itemPhases, itemID)
		}
	case "response.output_audio.delta", "response.audio.delta":
		pcm, err := base64.StdEncoding.DecodeString(event.Delta)
		if err != nil {
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("decode output audio: %w", err)})
			return
		}
		s.audioOutBytes.Add(uint64(len(pcm)))
		base.Type, base.Audio = RealtimeEventAudioOut, pcm
		s.emitAudio(base)
	case "response.output_audio_transcript.delta", "response.output_text.delta", "response.text.delta":
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptOutput, event.Delta, false
		s.emitAudio(base)
	case "response.output_audio_transcript.done":
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptOutput, event.Transcript, true
		s.emitControl(base)
	case "response.output_text.done":
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptOutput, event.Text, true
		s.emitControl(base)
	case "conversation.item.input_audio_transcription.delta":
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptInput, event.Delta, false
		s.emitAudio(base)
	case "conversation.item.input_audio_transcription.updated":
		transcript := event.Transcript
		if transcript == "" {
			transcript = event.Delta
		}
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptInput, transcript, false
		s.emitAudio(base)
	case "conversation.item.input_audio_transcription.completed":
		base.Type, base.Transcript, base.Final = RealtimeEventTranscriptInput, event.Transcript, true
		s.emitControl(base)
	case "input_audio_buffer.speech_started":
		base.Type, base.AudioStartMS = RealtimeEventSpeechStarted, event.AudioStartMS
		s.emitControl(base)
	case "response.function_call_arguments.done":
		base.Type = RealtimeEventToolCall
		base.ToolCallID, base.ToolName, base.ToolArgs = event.CallID, event.Name, event.Arguments
		s.emitControl(base)
	case "response.done":
		usage := event.Response.Usage
		base.Type, base.ResponseID = RealtimeEventResponseDone, event.Response.ID
		for _, item := range event.Response.Output {
			if item.ID != "" {
				delete(s.itemPhases, item.ID)
			}
		}
		base.Usage = RealtimeUsage{
			TotalTokens: usage.TotalTokens, InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens,
			TextInputTokens:   usage.InputDetails.TextTokens,
			TextCachedTokens:  usage.InputDetails.CachedDetails.TextTokens,
			TextOutputTokens:  usage.OutputDetails.TextTokens,
			AudioInputTokens:  usage.InputDetails.AudioTokens,
			AudioCachedTokens: usage.InputDetails.CachedDetails.AudioTokens,
			AudioOutputTokens: usage.OutputDetails.AudioTokens,
		}
		if s.audioInBytesPerSecond > 0 {
			base.Usage.AudioInputSeconds = float64(s.audioInBytes.Swap(0)) / s.audioInBytesPerSecond
		}
		if s.audioOutBytesPerSecond > 0 {
			base.Usage.AudioOutputSeconds = float64(s.audioOutBytes.Swap(0)) / s.audioOutBytesPerSecond
		}
		base.Usage.TextInputMessages = int(s.textInputMessages.Swap(0))
		s.emitControl(base)
	case "rate_limits.updated":
		base.Type = RealtimeEventRateLimits
		s.emitControl(base)
	case "error":
		providerName := s.providerName
		if providerName == "" {
			providerName = "realtime"
		}
		s.emitControl(RealtimeEvent{Type: RealtimeEventError,
			Err: fmt.Errorf("%s: %s/%s: %s", providerName, event.Error.Type, event.Error.Code, event.Error.Message)})
	}
}

// Audio and partial transcript deltas may be dropped under sustained
// backpressure. Lifecycle, error, tool, final transcript, and usage events are
// never silently discarded.
func (s *openaiRealtimeSession) emitAudio(event RealtimeEvent) {
	select {
	case s.events <- event:
	default:
		s.droppedAudio.Add(1)
	}
}

func (s *openaiRealtimeSession) emitControl(event RealtimeEvent) {
	select {
	case s.events <- event:
	case <-s.done:
	}
}

func (s *openaiRealtimeSession) writeLoop() {
	for {
		select {
		case <-s.done:
			return
		case frame := <-s.outbox:
			if err := wsutil.WriteClientMessage(s.conn, frame.op, frame.data); err != nil {
				s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("ws write: %w", err)})
				_ = s.Close()
				return
			}
		}
	}
}

func (s *openaiRealtimeSession) pingLoop() {
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

func (s *openaiRealtimeSession) enqueueFrame(frame realtimeOutboundFrame) error {
	select {
	case <-s.done:
		return fmt.Errorf("realtime session closed")
	default:
	}
	select {
	case s.outbox <- frame:
		return nil
	case <-s.done:
		return fmt.Errorf("realtime session closed")
	default:
		return fmt.Errorf("realtime outbox full")
	}
}

func (s *openaiRealtimeSession) enqueue(payload map[string]any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.enqueueFrame(realtimeOutboundFrame{op: ws.OpText, data: data})
}

func (s *openaiRealtimeSession) SendAudio(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	if err := s.enqueue(map[string]any{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(pcm)}); err != nil {
		return err
	}
	s.audioInBytes.Add(uint64(len(pcm)))
	return nil
}

func (s *openaiRealtimeSession) appendText(role, text string, respond bool) error {
	contentType := "input_text"
	if role == "assistant" {
		contentType = "output_text"
	}
	if err := s.enqueue(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type": "message", "role": role,
			"content": []map[string]any{{"type": contentType, "text": text}},
		},
	}); err != nil {
		return err
	}
	s.textInputMessages.Add(1)
	if respond {
		return s.enqueue(map[string]any{"type": "response.create"})
	}
	return nil
}

func (s *openaiRealtimeSession) SendText(role, text string) error {
	return s.appendText(role, text, false)
}

func (s *openaiRealtimeSession) SendToolResult(callID, result string, isError bool) error {
	if isError {
		result = "ERROR: " + result
	}
	return s.enqueue(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{"type": "function_call_output", "call_id": callID, "output": result},
	})
}

func (s *openaiRealtimeSession) RequestResponse() error {
	return s.enqueue(map[string]any{"type": "response.create"})
}

func (s *openaiRealtimeSession) UpdateConfiguration(instructions string, tools []NativeTool) error {
	if s.buildConfigurationUpdate == nil {
		return fmt.Errorf("%s: configuration updates are not supported", s.providerName)
	}
	return s.enqueue(s.buildConfigurationUpdate(instructions, tools))
}

func (s *openaiRealtimeSession) RestoreConversation(messages []Message) error {
	for _, message := range messages {
		if strings.TrimSpace(message.Content) == "" {
			continue
		}
		if message.Role != "user" && message.Role != "assistant" && message.Role != "system" {
			continue
		}
		if err := s.appendText(message.Role, message.Content, false); err != nil {
			return err
		}
	}
	return nil
}

func (s *openaiRealtimeSession) Interrupt() error {
	return s.enqueue(map[string]any{"type": "response.cancel"})
}

func (s *openaiRealtimeSession) Truncate(itemID string, audioEndMS int) error {
	if itemID == "" {
		return fmt.Errorf("truncate requires item id")
	}
	if audioEndMS < 0 {
		audioEndMS = 0
	}
	return s.enqueue(map[string]any{
		"type": "conversation.item.truncate", "item_id": itemID,
		"content_index": 0, "audio_end_ms": audioEndMS,
	})
}

func (s *openaiRealtimeSession) Events() <-chan RealtimeEvent { return s.events }

func (s *openaiRealtimeSession) Close() error {
	var err error
	s.closeOnce.Do(func() {
		close(s.done)
		if s.conn != nil {
			err = s.conn.Close()
		}
	})
	return err
}
