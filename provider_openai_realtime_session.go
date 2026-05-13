package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

// openaiRealtimeSession is a live WebSocket session against OpenAI's
// Realtime API. One goroutine reads server frames and translates them
// into RealtimeEvent on the events channel; one goroutine serializes
// outbound messages from the outbox channel onto the wire. Public
// methods enqueue into the outbox without blocking on I/O so callers
// never stall the audio loop.
type openaiRealtimeSession struct {
	conn    net.Conn
	events  chan RealtimeEvent
	outbox  chan []byte
	closeMu sync.Mutex
	closed  bool
}

// openaiRealtimeEvent matches the wire envelope. Only fields we
// inspect are tagged; everything else is preserved via raw passthrough
// when we need it.
type openaiRealtimeEvent struct {
	Type    string `json:"type"`
	EventID string `json:"event_id,omitempty"`

	// response.audio.delta
	Delta string `json:"delta,omitempty"`

	// response.audio_transcript.delta / .done, response.text.delta / .done
	Transcript string `json:"transcript,omitempty"`
	Text       string `json:"text,omitempty"`

	// response.function_call_arguments.* and conversation.item.*
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// error
	Error struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Open dials OpenAI's realtime endpoint and returns a live session.
// ctx governs the dial + handshake; the session inherits it for
// in-flight reads. Caller is responsible for calling Close().
func (p *OpenAIRealtimeProvider) openSession(ctx context.Context, opts RealtimeSessionOpts) (*openaiRealtimeSession, error) {
	model := opts.Model
	if model == "" {
		model = p.models[ModelLarge]
	}
	endpoint := &url.URL{
		Scheme:   "wss",
		Host:     "api.openai.com",
		Path:     "/v1/realtime",
		RawQuery: "model=" + url.QueryEscape(model),
	}

	dialer := ws.Dialer{
		Timeout: 10 * time.Second,
		Header: ws.HandshakeHeaderHTTP(http.Header{
			"Authorization": []string{"Bearer " + p.apiKey},
			"OpenAI-Beta":   []string{"realtime=v1"},
		}),
	}
	conn, _, _, err := dialer.Dial(ctx, endpoint.String())
	if err != nil {
		return nil, fmt.Errorf("realtime dial: %w", err)
	}

	s := &openaiRealtimeSession{
		conn:   conn,
		events: make(chan RealtimeEvent, 64),
		outbox: make(chan []byte, 64),
	}

	// Initial session.update — directive, voice, audio formats, tools.
	sessUpdate, err := buildSessionUpdate(opts, p.DefaultVoice())
	if err != nil {
		conn.Close()
		return nil, err
	}
	// Non-blocking: outbox is buffered.
	s.outbox <- sessUpdate

	go s.readLoop()
	go s.writeLoop()

	return s, nil
}

// buildSessionUpdate marshals the initial session.update payload from
// our generic opts.
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

	type sessionTool struct {
		Type        string         `json:"type"`
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Parameters  map[string]any `json:"parameters"`
	}
	tools := make([]sessionTool, 0, len(opts.Tools))
	for _, t := range opts.Tools {
		tools = append(tools, sessionTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}

	payload := map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"instructions":       opts.Instructions,
			"voice":              voice,
			"input_audio_format": string(inFmt),
			"output_audio_format": string(outFmt),
			"modalities":         []string{"audio", "text"},
			"tools":              tools,
		},
	}
	if opts.Temperature > 0 {
		payload["session"].(map[string]any)["temperature"] = opts.Temperature
	}
	return json.Marshal(payload)
}

// readLoop pulls server frames, decodes them, and emits
// RealtimeEvent. On a fatal error or close, it closes the events
// channel and drops out. Only text frames are expected from OpenAI
// realtime — audio comes back base64-encoded inside JSON events.
func (s *openaiRealtimeSession) readLoop() {
	defer close(s.events)
	for {
		data, op, err := wsutil.ReadServerData(s.conn)
		if err != nil {
			s.emit(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("ws read: %w", err)})
			s.emit(RealtimeEvent{Type: RealtimeEventSessionEnded})
			return
		}
		if op != ws.OpText {
			// Realtime API doesn't use binary frames; ignore.
			continue
		}
		var ev openaiRealtimeEvent
		if err := json.Unmarshal(data, &ev); err != nil {
			s.emit(RealtimeEvent{Type: RealtimeEventError, Err: fmt.Errorf("decode event: %w", err)})
			continue
		}
		s.translate(&ev)
	}
}

// translate converts the OpenAI wire event into our provider-neutral
// RealtimeEvent and pushes it onto the events channel. Unknown event
// types are silently dropped — the API emits many lifecycle events
// (session.created, response.content_part.added, etc.) that consumers
// don't need.
func (s *openaiRealtimeSession) translate(ev *openaiRealtimeEvent) {
	switch ev.Type {
	case "response.audio.delta":
		if ev.Delta != "" {
			pcm, err := base64.StdEncoding.DecodeString(ev.Delta)
			if err == nil {
				s.emit(RealtimeEvent{Type: RealtimeEventAudioOut, Audio: pcm})
			}
		}
	case "response.audio_transcript.delta":
		s.emit(RealtimeEvent{
			Type:       RealtimeEventTranscriptOutput,
			Transcript: ev.Delta,
			Final:      false,
		})
	case "response.audio_transcript.done":
		s.emit(RealtimeEvent{
			Type:       RealtimeEventTranscriptOutput,
			Transcript: ev.Transcript,
			Final:      true,
		})
	case "conversation.item.input_audio_transcription.completed":
		s.emit(RealtimeEvent{
			Type:       RealtimeEventTranscriptInput,
			Transcript: ev.Transcript,
			Final:      true,
		})
	case "response.function_call_arguments.done":
		s.emit(RealtimeEvent{
			Type:       RealtimeEventToolCall,
			ToolCallID: ev.CallID,
			ToolName:   ev.Name,
			ToolArgs:   ev.Arguments,
		})
	case "response.done":
		s.emit(RealtimeEvent{Type: RealtimeEventResponseDone})
	case "error":
		s.emit(RealtimeEvent{
			Type: RealtimeEventError,
			Err:  fmt.Errorf("openai realtime: %s/%s: %s", ev.Error.Type, ev.Error.Code, ev.Error.Message),
		})
	}
}

func (s *openaiRealtimeSession) emit(ev RealtimeEvent) {
	select {
	case s.events <- ev:
	default:
		// Drop on saturation rather than block the read loop. The
		// alternative — blocking — would starve the WebSocket buffer
		// and ultimately cause the server to close the connection.
		// Consumers should size their loops to drain promptly.
	}
}

// writeLoop drains the outbox and writes each message as a text
// frame. Exits when outbox is closed (via Close()).
func (s *openaiRealtimeSession) writeLoop() {
	for msg := range s.outbox {
		if err := wsutil.WriteClientText(s.conn, msg); err != nil {
			// Once the connection is dead, the read loop will surface
			// the error too; we just stop writing and let Close clean up.
			return
		}
	}
}

// enqueue serializes a message to JSON and pushes it onto the outbox.
// Drops on saturation to keep the caller (audio loop, tool handler)
// non-blocking. If the session has been closed, drops silently.
func (s *openaiRealtimeSession) enqueue(payload map[string]any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return nil
	}
	s.closeMu.Unlock()
	select {
	case s.outbox <- data:
		return nil
	default:
		return fmt.Errorf("realtime outbox full")
	}
}

// SendAudio base64-encodes PCM and pushes an input_audio_buffer.append.
func (s *openaiRealtimeSession) SendAudio(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	return s.enqueue(map[string]any{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(pcm),
	})
}

// SendText injects a conversation.item.create with the given role.
// Triggers a response.create afterwards so the model actually speaks
// (without it, the message sits in history until the next user audio
// turn completes).
func (s *openaiRealtimeSession) SendText(role, text string) error {
	if err := s.enqueue(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type": "message",
			"role": role,
			"content": []map[string]any{
				{"type": "input_text", "text": text},
			},
		},
	}); err != nil {
		return err
	}
	return s.enqueue(map[string]any{"type": "response.create"})
}

// SendToolResult delivers a function_call_output item back to the
// model.
func (s *openaiRealtimeSession) SendToolResult(callID, result string, isError bool) error {
	output := result
	if isError {
		output = "ERROR: " + result
	}
	if err := s.enqueue(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	}); err != nil {
		return err
	}
	return s.enqueue(map[string]any{"type": "response.create"})
}

// UpdateInstructions issues session.update with the new directive.
// Other session fields (voice, audio format, tools) are left untouched
// — OpenAI allows partial updates.
func (s *openaiRealtimeSession) UpdateInstructions(directive string) error {
	return s.enqueue(map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"instructions": directive,
		},
	})
}

// Interrupt cancels the model's current response. OpenAI's
// response.cancel handles the case where audio is mid-flight cleanly.
func (s *openaiRealtimeSession) Interrupt() error {
	return s.enqueue(map[string]any{"type": "response.cancel"})
}

func (s *openaiRealtimeSession) Events() <-chan RealtimeEvent { return s.events }

// Close terminates the session. Safe to call multiple times. Closes
// the WS connection (which unblocks the read loop) and the outbox
// (which unblocks the write loop). The events channel closes when
// the read loop exits.
func (s *openaiRealtimeSession) Close() error {
	s.closeMu.Lock()
	if s.closed {
		s.closeMu.Unlock()
		return nil
	}
	s.closed = true
	close(s.outbox)
	s.closeMu.Unlock()
	return s.conn.Close()
}
