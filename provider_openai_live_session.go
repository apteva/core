package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

// Live is a second wire protocol of the existing OpenAI API integration.
// It reuses its credentials and transport machinery, never a hosted agent.
type openaiLiveSession struct {
	media       RealtimeMediaConfig
	rtcMessages <-chan liveMediaMessage
	execution   openaiLiveExecution
	*openaiRealtimeSession
	ready          chan error
	closeRequested sync.Once
	closeError     error
	settingsMu     sync.RWMutex
	instructions   string
	usageSeconds   float64
	tools          []NativeTool
	guard          realtimeContinuousGuard
	guardMu        sync.Mutex
}

type openaiLiveEvent struct {
	Event         json.RawMessage `json:"event"`
	Type          string          `json:"type"`
	ClientEventID string          `json:"client_event_id"`
	Delta         string          `json:"delta"`
	StartMS       int             `json:"start_ms"`
	EndMS         int             `json:"end_ms"`
	OffsetMS      int             `json:"offset_ms"`
	Reason        string          `json:"reason"`
	Session       map[string]any  `json:"session"`
	Usage         struct {
		Seconds float64 `json:"seconds"`
	} `json:"usage"`
	Delegation struct{ ID, Target string } `json:"delegation"`
	Error      struct {
		Code          string `json:"code"`
		Message       string `json:"message"`
		ClientEventID string `json:"client_event_id"`
	} `json:"error"`
}

func buildOpenAILiveStart(opts RealtimeSessionOpts, defaultVoice string) ([]byte, error) {
	if opts.Model == "" {
		return nil, fmt.Errorf("OpenAI Live model is required")
	}
	m, err := opts.OutputConfig.Media.normalized()
	if err != nil {
		return nil, err
	}
	if opts.AudioInFmt != "" && opts.AudioInFmt != m.format() || opts.AudioOutFmt != "" && opts.AudioOutFmt != m.format() || opts.AudioInRate != 0 && opts.AudioInRate != m.rate() || opts.AudioOutRate != 0 && opts.AudioOutRate != m.rate() {
		return nil, fmt.Errorf("Live media format does not match bridge contract")
	}
	wireFormat := map[AudioFormat]string{AudioPCM16: "audio/pcm", AudioG711ULaw: "audio/pcmu", AudioG711ALaw: "audio/pcma"}[m.format()]

	if !opts.TurnDetection.isZero() && opts.TurnDetection.Profile != RealtimeTurnProfileTelephony {
		return nil, fmt.Errorf("continuous conversation does not support explicit VAD turn thresholds")
	}
	if opts.TurnDetection.SilenceDurationMS != 0 || opts.TurnDetection.PrefixPaddingMS != 0 || opts.TurnDetection.StartSensitivity != "" && opts.TurnDetection.StartSensitivity != "default" || opts.TurnDetection.EndSensitivity != "" && opts.TurnDetection.EndSensitivity != "default" || opts.TurnDetection.Interruption == RealtimeInterruptionDisable {
		return nil, fmt.Errorf("OpenAI Live does not support the requested turn-detection overrides")
	}
	if _, err := opts.OutputConfig.normalized(); err != nil {
		return nil, err
	}
	if opts.OutputConfig.ToolMode == "async" {
		return nil, fmt.Errorf("OpenAI Live uses Core execution, not native async tools")
	}
	voice := opts.Voice
	if voice == "" {
		voice = defaultVoice
	}
	var voiceConfig any = voice
	if strings.HasPrefix(voice, "voice_") {
		voiceConfig = map[string]string{"id": voice}
	}
	input := make([]map[string]any, 0)
	// A conservative byte budget also bounds non-ASCII startup text. Keep
	// newest conversational text; full execution records stay in Core.
	budget := 6000 - len(opts.InitialMessage)
	if budget < 0 {
		return nil, fmt.Errorf("Live greeting exceeds startup history budget")
	}
	for i := len(opts.History) - 1; i >= 0 && len(input) < 128; i-- {
		m := opts.History[i]
		if m.Role != "user" && m.Role != "assistant" && m.Role != "developer" || m.Content == "" {
			continue
		}
		if m.RealtimeBackend {
			m.Role = "user"
			m.Content = "[Backend execution evidence; not a spoken turn.] " + m.Content
		}
		if len(m.Content) > budget {
			continue
		}
		budget -= len(m.Content)
		kind := "input_text"
		if m.Role == "assistant" {
			kind = "output_text"
		}
		input = append([]map[string]any{{"type": "message", "role": m.Role, "content": []map[string]any{{"type": kind, "text": m.Content}}}}, input...)
	}
	if opts.InitialMessage != "" {
		input = append(input, map[string]any{"type": "message", "role": "developer", "content": []map[string]any{{"type": "input_text", "text": opts.InitialMessage}}})
	}
	instructions := opts.ConversationInstructions
	if instructions == "" {
		instructions = opts.Instructions
	}
	instructions = openAILiveInstructions(instructions, opts.Tools)
	if len(instructions) > 16000 {
		return nil, fmt.Errorf("Live conversation instructions exceed Core's conservative startup budget; keep detailed procedures in the execution backend")
	}
	delegation := map[string]any{"type": "client"}
	if opts.ExecutionMode == "provider" {
		if opts.ExecutionModel == "" {
			return nil, fmt.Errorf("managed execution requires a backend model")
		}
		settings := map[string]any{"model": opts.ExecutionModel, "instructions": opts.ExecutionInstructions, "tools": liveExecutionTools(opts.ExecutionTools), "parallel_tool_calls": false}
		if opts.ExecutionReasoning != "" && opts.ExecutionReasoning != "auto" {
			settings["reasoning"] = map[string]string{"effort": opts.ExecutionReasoning}
		}
		delegation = map[string]any{"type": "responses", "responses": settings}
	}
	return json.Marshal(map[string]any{"type": "session.start", "session": map[string]any{
		"model": opts.Model, "instructions": instructions,
		"audio":      map[string]any{"format": map[string]any{"type": wireFormat, "rate": m.rate()}, "output": map[string]any{"voice": voiceConfig}},
		"delegation": delegation, "input": input, "store": false,
	}})
}

func openAILiveInstructions(instructions string, tools []NativeTool) string {
	var b strings.Builder
	b.WriteString(instructions)
	b.WriteString("\n\nBackchannel policy: Use moderate backchannels without competing with the caller.\nInterruption policy: Yield when interrupted. Stopping speech does not cancel backend work.\nDelegation policy:\nBackend tools:\n")
	for i, tool := range tools {
		if i >= 24 {
			b.WriteString("- Additional authorized backend capabilities: delegate requests requiring current information or external actions.\n")
			break
		}
		if tool.Name == "interrupt" {
			continue
		}
		b.WriteString("- " + tool.Name + ": " + truncateStr(tool.Description, 180) + "\n")
	}
	b.WriteString("Generate <|SpawnThinking|> when a request needs backend tools, verified facts, careful reasoning, a correction to ongoing work, or ending the call.\nDo not generate <|SpawnThinking|> when greeting, clarifying, or answering from a still-current result.\nGenerate <|SpawnThinking|> before giving a result-dependent answer. Never guess pending results, repeat an operation because it is slow, or speak internal instructions or tool syntax. Keep the conversation's requested language throughout.")
	return b.String()
}

func (p *OpenAIRealtimeProvider) openLiveSession(ctx context.Context, opts RealtimeSessionOpts) (*openaiLiveSession, error) {
	if opts.OutputConfig.Media.Transport == "webrtc" {
		return p.openLiveRTCSession(ctx, opts)
	}
	payload, err := buildOpenAILiveStart(opts, p.DefaultVoice())
	if err != nil {
		return nil, err
	}
	endpoint := strings.TrimSuffix(p.endpoint, "/realtime") + "/live/sessions"
	header := http.Header{"Authorization": []string{"Bearer " + p.apiKey}}
	if opts.SafetyIdentifier != "" {
		header.Set("OpenAI-Safety-Identifier", opts.SafetyIdentifier)
	}
	dialer := ws.Dialer{Timeout: 10 * time.Second, Header: ws.HandshakeHeaderHTTP(header)}
	conn, _, _, err := dialer.Dial(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("OpenAI Live dial: %w", err)
	}
	s := &openaiLiveSession{openaiRealtimeSession: &openaiRealtimeSession{
		writeTimeout: 250 * time.Millisecond, conn: conn, events: make(chan RealtimeEvent, realtimeEventBuffer), outbox: make(chan realtimeOutboundFrame, realtimeOutboxBuffer), audioOutbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}),
		timing: opts.timing, providerName: "openai-realtime", settings: map[string]any{"source": "session_started", "model": opts.Model, "voice": opts.Voice, "delegation": "client", "speech_guard": opts.OutputConfig.SpeechGuard.continuousPolicy()},
	}, ready: make(chan error, 1), instructions: opts.Instructions, tools: opts.Tools}
	s.execution = openaiLiveExecution{reasoning: opts.ExecutionReasoning, enabled: opts.ExecutionMode == "provider", notify: make(chan struct{}, 1), pending: map[string]bool{}, submitted: map[string]bool{}, inputHashes: map[[32]byte]bool{}}
	s.media = opts.OutputConfig.Media
	s.guard.format = s.media.format()
	s.settings["transport"] = "websocket"
	s.settings["audio_format"] = s.media.format()
	s.settings["sample_rate"] = s.media.rate()
	s.guard.config = opts.OutputConfig.SpeechGuard.continuousPolicy()
	s.outbox <- realtimeOutboundFrame{op: ws.OpText, data: payload}
	s.lifecycle.start(s.events, s.readLiveLoop, s.writeLoop, s.pingLoop, s.flushLiveSpeech)
	go func() {
		select {
		case <-ctx.Done():
			_ = s.Close()
		case <-s.done:
		}
	}()
	timer := time.NewTimer(15 * time.Second)
	defer timer.Stop()
	select {
	case err := <-s.ready:
		if err != nil {
			_ = s.Close()
			return nil, err
		}
		return s, nil
	case <-ctx.Done():
		_ = s.Close()
		return nil, ctx.Err()
	case <-s.done:
		return nil, fmt.Errorf("OpenAI Live closed before session.started")
	case <-timer.C:
		_ = s.Close()
		return nil, fmt.Errorf("OpenAI Live startup timed out")
	}
}

func (s *openaiLiveSession) readLiveLoop() {
	defer s.Close()
	for {
		var data []byte
		var op ws.OpCode
		var err error
		if s.rtcMessages != nil {
			select {
			case <-s.done:
				return
			case m := <-s.rtcMessages:
				s.readStartedAt, s.readReturnedAt = time.Time{}, time.Time{}
				if m.audio != nil {
					s.acceptLiveAudio(m.audio, time.Now())
					continue
				}
				data, op, err = m.data, ws.OpText, m.err
			}
		} else {
			readStarted := time.Now()
			data, op, err = wsutil.ReadServerData(s.conn)
			s.readStartedAt, s.readReturnedAt = readStarted, time.Now()
		}
		if err != nil {
			select {
			case <-s.done:
				return
			default:
			}
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: realtimeAdapterError(RealtimeErrorTransport, fmt.Errorf("Live read: %w", err))})
			return
		}
		if op != ws.OpText {
			continue
		}
		var e openaiLiveEvent
		if err := json.Unmarshal(data, &e); err != nil {
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: realtimeAdapterError(RealtimeErrorProtocol, fmt.Errorf("invalid Live event: %w", err))})
			continue
		}
		s.translateLive(e)
		if e.Type == "session.closed" {
			return
		}
	}
}

func (s *openaiLiveSession) translateLive(e openaiLiveEvent) {
	now := time.Now()
	switch e.Type {
	case "session.started":
		s.emitControl(RealtimeEvent{Type: RealtimeEventSessionReady, ObservedAt: now, Settings: s.RealtimeSettings()})
		select {
		case s.ready <- nil:
		default:
		}
	case "session.input_transcript.delta", "session.output_transcript.delta":
		kind := RealtimeEventTranscriptInput
		if e.Type == "session.output_transcript.delta" {
			kind = RealtimeEventTranscriptOutput
		}
		event := RealtimeEvent{Type: kind, ObservedAt: now, Transcript: e.Delta, TranscriptDelta: true, TranscriptStartMS: e.StartMS, TranscriptEndMS: e.EndMS}
		if kind == RealtimeEventTranscriptOutput {
			s.guardMu.Lock()
			events := s.guard.transcript(event, s.tools, s.timing)
			s.guardMu.Unlock()
			for _, ev := range events {
				s.emitControl(ev)
			}
		} else {
			s.emitControl(event)
		}
	case "session.output_audio.delta":
		audio, err := base64.StdEncoding.DecodeString(e.Delta)
		if err != nil || s.media.format() == AudioPCM16 && len(audio)%2 != 0 {
			s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("invalid Live PCM audio")})
			return
		}
		s.acceptLiveAudio(audio, now)
	case "response.event":
		s.translateExecution(e.Event)
	case "session.delegation.created":
		if (e.Delegation.Target == "client" || e.Delegation.Target == "responses") && e.Delegation.ID != "" {
			s.emitControl(RealtimeEvent{Type: RealtimeEventDelegation, DelegationID: e.Delegation.ID, DelegationOffsetMS: e.OffsetMS, ObservedAt: now})
		}
	case "session.instructions.appended", "session.thinking.appended", "session.commentary.appended":
		s.emitControl(RealtimeEvent{Type: RealtimeEventContextAcknowledged, ClientEventID: e.ClientEventID, TranscriptStartMS: e.StartMS, TranscriptEndMS: e.EndMS, ObservedAt: now})
	case "session.usage.updated", "session.closed":
		if e.Usage.Seconds >= s.usageSeconds {
			delta := e.Usage.Seconds - s.usageSeconds
			s.usageSeconds = e.Usage.Seconds
			s.emitControl(RealtimeEvent{Type: RealtimeEventSessionUsage, Usage: RealtimeUsage{SessionSeconds: delta}, ObservedAt: now})
		}
		if e.Type == "session.closed" {
			s.guardMu.Lock()
			events := s.guard.finish()
			s.guardMu.Unlock()
			for _, ev := range events {
				s.emitControl(ev)
			}
			s.emitControl(RealtimeEvent{Type: RealtimeEventSessionEnded, CloseReason: e.Reason, Terminal: e.Reason == "content" || e.Reason == "remote_hangup", ObservedAt: now})
		}
	case "error":
		err := fmt.Errorf("OpenAI Live %s: %s", e.Error.Code, e.Error.Message)
		select {
		case s.ready <- err:
		default:
		}
		if s.execution.enabled {
			s.execution.mu.Lock()
			s.execution.queue = append(s.execution.queue, realtimeExecutionResult{err: err})
			s.execution.mu.Unlock()
			s.executionWake()
		}
		s.emitControl(RealtimeEvent{Type: RealtimeEventError, Err: err, ObservedAt: now})
	}
}

func (s *openaiLiveSession) flushLiveSpeech() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case <-ticker.C:
			s.guardMu.Lock()
			var events []RealtimeEvent
			if !s.guard.lastAudio.IsZero() && time.Since(s.guard.lastAudio) >= realtimeObservedQuiet {
				events = s.guard.finish()
			}
			s.guardMu.Unlock()
			for _, e := range events {
				s.emitControl(e)
			}
		}
	}
}

func (s *openaiLiveSession) SendAudio(pcm []byte) error {
	if s.media.format() == AudioPCM16 && len(pcm)%2 != 0 {
		return realtimeAdapterError(RealtimeErrorProtocol, fmt.Errorf("PCM16 requires complete samples"))
	}
	if len(pcm) == 0 {
		return nil
	}
	return s.enqueueAudio(map[string]any{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(pcm)})
}

func (s *openaiLiveSession) AppendContext(u RealtimeContextUpdate) error {
	kind := map[string]string{"instructions": "instructions", "context": "thinking", "commentary": "commentary"}[u.Kind]
	if kind == "" {
		return fmt.Errorf("unsupported realtime context kind %q", u.Kind)
	}
	// At most one token per UTF-8 byte is a conservative bound for these
	// text appends. Split at rune boundaries; never silently truncate results.
	for i, part := range splitRealtimeContext(u.Content, 480) {
		eventID := "apt_" + generateAudioToken()
		if i == 0 && u.ClientEventID != "" {
			eventID = u.ClientEventID
		}
		var id any
		if u.DelegationID != "" {
			id = u.DelegationID
		}
		if err := s.enqueue(map[string]any{"type": "session." + kind + ".append", "event_id": eventID, "delegation_id": id, "content": part}); err != nil {
			return err
		}
	}
	return nil
}

func splitRealtimeContext(text string, limit int) []string {
	var parts []string
	for len(text) > limit {
		n := limit
		for n > 0 && text[n]&0xc0 == 0x80 {
			n--
		}
		parts = append(parts, text[:n])
		text = text[n:]
	}
	if text != "" {
		parts = append(parts, text)
	}
	return parts
}

func (s *openaiLiveSession) SendText(role, text string) error {
	kind := "context"
	if role == "system" {
		kind = "instructions"
	}
	return s.AppendContext(RealtimeContextUpdate{Kind: kind, Content: text})
}

func (s *openaiLiveSession) RequestResponse() error { return nil } // autonomous conversation, no voice turn trigger
func (s *openaiLiveSession) SendToolResult(string, string, bool) error {
	return fmt.Errorf("Live client delegation does not accept native tool results")
}
func (s *openaiLiveSession) RestoreConversation([]Message) error { return nil } // history was seeded in session.start
func (s *openaiLiveSession) Interrupt() error {
	return s.AppendContext(RealtimeContextUpdate{Kind: "instructions", Content: "Stop the current speech and listen. Backend work remains pending unless explicitly cancelled."})
}
func (s *openaiLiveSession) Truncate(string, int) error {
	return fmt.Errorf("Live cannot truncate conversation audio; discard local playback instead")
}
func (s *openaiLiveSession) BeginClose() (err error) {
	s.closeRequested.Do(func() { s.closeError = s.enqueue(map[string]any{"type": "session.close"}) })
	return s.closeError
}
func (s *openaiLiveSession) RealtimeSettings() map[string]any {
	s.settingsMu.RLock()
	defer s.settingsMu.RUnlock()
	result := map[string]any{}
	for k, v := range s.settings {
		result[k] = v
	}
	return result
}
func (s *openaiLiveSession) PreviewConfigurationUpdate(instructions string, tools []NativeTool) RealtimeConfigurationDisposition {
	s.settingsMu.RLock()
	defer s.settingsMu.RUnlock()
	if instructions == s.instructions {
		return RealtimeConfigurationUnchanged
	}
	return RealtimeConfigurationRestartRequired
}
func (s *openaiLiveSession) UpdateConfiguration(instructions string, tools []NativeTool) error {
	if s.PreviewConfigurationUpdate(instructions, tools) == RealtimeConfigurationUnchanged {
		return nil
	}
	return ErrRealtimeConfigurationRestartRequired
}

var _ RealtimeSession = (*openaiLiveSession)(nil)

// DiscardPlayback only invalidates local audio/check state. Live's own yield
// policy is separate, and backend work is deliberately unaffected.
func (s *openaiLiveSession) DiscardPlayback(item string) {
	s.guardMu.Lock()
	s.guard.discard(item)
	s.guardMu.Unlock()
}
func (s *openaiLiveSession) RealtimeInterruptCommand() string { return "session.instructions.append" }

func (s *openaiLiveSession) SendAudioFrame(f RealtimeInputAudio) error {
	if s.media.format() == AudioPCM16 && len(f.Audio)%2 != 0 {
		return realtimeAdapterError(RealtimeErrorProtocol, fmt.Errorf("PCM16 requires complete samples"))
	}
	return s.sendAudioFrame(f, map[string]any{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(f.Audio)})
}

func (s *openaiLiveSession) acceptLiveAudio(audio []byte, at time.Time) {
	s.guardMu.Lock()
	event := RealtimeEvent{Type: RealtimeEventAudioOut, ObservedAt: at, Audio: audio}
	if !s.readStartedAt.IsZero() {
		event.providerReadStartedAt, event.providerReadReturnedAt = s.readStartedAt, s.readReturnedAt
		event.providerDeltaDurationMS = float64(len(audio)) * 1000 / audioBytesPerSecond(s.media.format(), s.media.rate())
	}
	events := s.guard.audio(event, s.timing)
	s.guardMu.Unlock()
	for _, event := range events {
		s.emitControl(event)
	}
}
