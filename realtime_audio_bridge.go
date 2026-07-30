package core

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

const (
	maxRealtimeAudioFrameBytes       = 1 << 20
	realtimeBridgeWriteTimeout       = 15 * time.Second
	realtimeBridgeCloseHandshakeWait = 250 * time.Millisecond
)

// realtimeAudioBridge holds the in-memory mapping between a
// single-use audio token (handed out at realtime-spawn time) and
// the audio channels of the corresponding realtime thread.
//
// External callers (the telephony sidecar, a browser WebRTC bridge,
// future signal sources) connect to GET /realtime/audio with
// ?thread=X&token=Y, the handler resolves the channels, and
// shuffles binary frames in both directions.
//
// Tokens are valid until first use; the handler unregisters on
// connect so a leaked token can't reattach later. Expiry is bounded
// by a coarse 5-minute timeout for unused tokens — if no bridge
// connects within that window, the realtime thread spins without
// audio. Realtime threads survive without an audio bridge (text-only
// fallback) — the WebSocket connection is optional, not required.

type audioBridgeRegistration struct {
	threadID string
	audioIn  chan []byte             // bridge writes here (caller → model)
	audioOut chan RealtimeAudioFrame // bridge reads here (model → caller)
	control  chan string             // low-volume JSON/text playback controls
	created  time.Time
}

var (
	audioBridgeMu    sync.Mutex
	audioBridgeByTok = map[string]*audioBridgeRegistration{}
	audioBridgeLive  = map[string]*audioBridgeConnection{}
	audioBridgeTTL   = 5 * time.Minute
)

type audioBridgeConnection struct {
	cancel context.CancelCauseFunc
}

// realtimeBridgeDisconnectData is deliberately provider-neutral. The audio
// bridge is shared by browser, telephony, and future realtime transports, so
// its terminal telemetry describes the WebSocket boundary rather than a
// provider-specific session.
type realtimeBridgeDisconnectData struct {
	Initiator   string `json:"initiator"`
	Operation   string `json:"operation"`
	CloseCode   int    `json:"close_code"`
	CloseReason string `json:"close_reason"`
	ErrorClass  string `json:"error_class"`
	ConnectedMS int64  `json:"connected_ms"`
	Normal      bool   `json:"normal"`
}

type realtimeBridgeEnd struct {
	data realtimeBridgeDisconnectData
}

func (e *realtimeBridgeEnd) Error() string {
	if e == nil {
		return "realtime audio bridge ended"
	}
	if e.data.CloseReason != "" {
		return e.data.CloseReason
	}
	return "realtime audio bridge ended"
}

type realtimeBridgeControlError struct {
	op  ws.OpCode
	err error
}

func (e *realtimeBridgeControlError) Error() string { return e.err.Error() }
func (e *realtimeBridgeControlError) Unwrap() error { return e.err }

// realtimeBridgeSocket serializes every complete WebSocket write. net.Conn
// permits concurrent Write calls, but a WebSocket frame is written as a
// separate header and payload. Without this lock a reader-generated pong can
// land between an outbound audio frame's header and payload.
type realtimeBridgeSocket struct {
	conn net.Conn

	writeMu   sync.Mutex
	closeSent bool
}

func (s *realtimeBridgeSocket) writeAudio(metadata, audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.closeSent {
		return net.ErrClosed
	}
	if err := s.conn.SetWriteDeadline(time.Now().Add(realtimeBridgeWriteTimeout)); err != nil {
		return err
	}
	// Keep metadata and its audio frame adjacent. Control frames may legally
	// occur between data frames, but the bridge's application protocol treats
	// this pair as one unit.
	if err := wsutil.WriteServerMessage(s.conn, ws.OpText, metadata); err != nil {
		return err
	}
	return wsutil.WriteServerBinary(s.conn, audio)
}

func (s *realtimeBridgeSocket) writeText(payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.closeSent {
		return net.ErrClosed
	}
	if err := s.conn.SetWriteDeadline(time.Now().Add(realtimeBridgeWriteTimeout)); err != nil {
		return err
	}
	return wsutil.WriteServerMessage(s.conn, ws.OpText, payload)
}

func (s *realtimeBridgeSocket) writeClose(code ws.StatusCode, reason string) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if s.closeSent {
		return nil
	}
	if err := s.conn.SetWriteDeadline(time.Now().Add(realtimeBridgeWriteTimeout)); err != nil {
		return err
	}
	if err := wsutil.WriteServerMessage(s.conn, ws.OpClose, ws.NewCloseFrameBody(code, reason)); err != nil {
		return err
	}
	s.closeSent = true
	return nil
}

func (s *realtimeBridgeSocket) hasSentClose() bool {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.closeSent
}

// handleControl holds the same complete-frame lock as audio and metadata
// writes. ControlFrameHandler writes pong and close responses directly to its
// destination, so passing the raw connection without this boundary is unsafe.
func (s *realtimeBridgeSocket) handleControl(header ws.Header, reader io.Reader) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	if header.OpCode == ws.OpClose && s.closeSent {
		// We initiated the close handshake. Consume and validate the peer's
		// response without sending a second close frame.
		return &realtimeBridgeControlError{op: header.OpCode, err: readRealtimeBridgePeerClose(header, reader)}
	}
	if s.closeSent {
		// No frames other than Close are meaningful after our close frame.
		_, err := io.CopyN(io.Discard, reader, header.Length)
		if err != nil {
			return &realtimeBridgeControlError{op: header.OpCode, err: err}
		}
		return nil
	}
	if err := s.conn.SetWriteDeadline(time.Now().Add(realtimeBridgeWriteTimeout)); err != nil {
		return &realtimeBridgeControlError{op: header.OpCode, err: err}
	}
	err := wsutil.ControlFrameHandler(s.conn, ws.StateServerSide)(header, reader)
	if header.OpCode == ws.OpClose {
		// A valid peer Close is answered inside ControlFrameHandler. Protocol
		// failures also attempt a protocol-error close, so no later writer
		// should emit application data on this connection.
		s.closeSent = true
	}
	if err != nil {
		return &realtimeBridgeControlError{op: header.OpCode, err: err}
	}
	return nil
}

func readRealtimeBridgePeerClose(header ws.Header, reader io.Reader) error {
	if header.Length == 0 {
		return wsutil.ClosedError{Code: ws.StatusNoStatusRcvd}
	}
	payload := make([]byte, header.Length)
	if _, err := io.ReadFull(reader, payload); err != nil {
		return err
	}
	code, reason := ws.ParseCloseFrameData(payload)
	if err := ws.CheckCloseFrameData(code, reason); err != nil {
		return err
	}
	return wsutil.ClosedError{Code: code, Reason: reason}
}

// generateAudioToken returns a 32-hex-char random token. Cryptographic
// randomness — single-use, but still worth not being guessable.
func generateAudioToken() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// registerAudioBridge stashes the channels under a freshly-generated
// token and returns the token. Sweeps any tokens older than TTL so
// the map doesn't grow unbounded if callers never connect.
func registerAudioBridge(threadID string, audioIn chan []byte, audioOut chan RealtimeAudioFrame, control chan string) string {
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	now := time.Now()
	for tok, reg := range audioBridgeByTok {
		if now.Sub(reg.created) > audioBridgeTTL {
			delete(audioBridgeByTok, tok)
		}
	}
	tok := generateAudioToken()
	audioBridgeByTok[tok] = &audioBridgeRegistration{
		threadID: threadID,
		audioIn:  audioIn,
		audioOut: audioOut,
		control:  control,
		created:  now,
	}
	return tok
}

// renewAudioBridge invalidates every unused token and any live bridge for the
// thread, then mints a fresh single-use token over the same long-lived audio
// channels. The realtime provider session and transcript stay intact while a
// browser changes network or wakes from sleep.
func renewAudioBridge(threadID string, audioIn chan []byte, audioOut chan RealtimeAudioFrame, control chan string) string {
	unregisterAudioBridge(threadID)
	return registerAudioBridge(threadID, audioIn, audioOut, control)
}

// unregisterAudioBridge removes any registrations bound to threadID.
// Used when a realtime spawn fails post-token-issuance, or when the
// thread is killed before a bridge connected.
func unregisterAudioBridge(threadID string) {
	audioBridgeMu.Lock()
	for tok, reg := range audioBridgeByTok {
		if reg.threadID == threadID {
			delete(audioBridgeByTok, tok)
		}
	}
	live := audioBridgeLive[threadID]
	delete(audioBridgeLive, threadID)
	audioBridgeMu.Unlock()
	if live != nil {
		live.cancel(newRealtimeBridgeEnd(
			"core", "close", ws.StatusNormalClosure,
			"audio bridge unregistered", "", true,
		))
	}
}

func renameAudioBridgeThread(oldID, newID string) {
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	for _, reg := range audioBridgeByTok {
		if reg.threadID == oldID {
			reg.threadID = newID
		}
	}
	if live := audioBridgeLive[oldID]; live != nil {
		delete(audioBridgeLive, oldID)
		audioBridgeLive[newID] = live
	}
}

// claimAudioBridge atomically pops a registration by token. Returns
// the channels (and thread id) or an error. Single-use: the second
// caller for the same token gets ErrAudioTokenUnknown.
func claimAudioBridge(token string) (*audioBridgeRegistration, error) {
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	reg, ok := audioBridgeByTok[token]
	if !ok {
		return nil, errAudioTokenUnknown
	}
	delete(audioBridgeByTok, token)
	if time.Since(reg.created) > audioBridgeTTL {
		return nil, errAudioTokenExpired
	}
	return reg, nil
}

func peekAudioBridge(token string) (*audioBridgeRegistration, error) {
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	reg, ok := audioBridgeByTok[token]
	if !ok {
		return nil, errAudioTokenUnknown
	}
	if time.Since(reg.created) > audioBridgeTTL {
		delete(audioBridgeByTok, token)
		return nil, errAudioTokenExpired
	}
	return reg, nil
}

var (
	errAudioTokenUnknown = errors.New("audio token unknown or already used")
	errAudioTokenExpired = errors.New("audio token expired")
)

func newRealtimeBridgeEnd(
	initiator, operation string,
	closeCode ws.StatusCode,
	reason, errorClass string,
	normal bool,
) *realtimeBridgeEnd {
	return &realtimeBridgeEnd{data: realtimeBridgeDisconnectData{
		Initiator:   initiator,
		Operation:   operation,
		CloseCode:   int(closeCode),
		CloseReason: reason,
		ErrorClass:  errorClass,
		Normal:      normal,
	}}
}

func classifyRealtimeBridgeError(initiator, operation string, err error) *realtimeBridgeEnd {
	if err == nil {
		return newRealtimeBridgeEnd(
			initiator, operation, ws.StatusNormalClosure,
			"audio bridge closed", "", true,
		)
	}
	var terminal *realtimeBridgeEnd
	if errors.As(err, &terminal) {
		return terminal
	}
	var controlErr *realtimeBridgeControlError
	if errors.As(err, &controlErr) {
		switch controlErr.op {
		case ws.OpPing:
			operation = "pong"
		case ws.OpPong:
			operation = "ping"
		case ws.OpClose:
			operation = "close"
		}
	}
	var closed wsutil.ClosedError
	if errors.As(err, &closed) {
		normal := closed.Code == ws.StatusNormalClosure || closed.Code == ws.StatusGoingAway ||
			closed.Code == ws.StatusNoStatusRcvd
		return newRealtimeBridgeEnd(
			"client", "close", closed.Code, closed.Reason, "", normal,
		)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return newRealtimeBridgeEnd(
			"context", operation, ws.StatusGoingAway,
			err.Error(), "context_cancelled", false,
		)
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return newRealtimeBridgeEnd(
			initiator, operation, ws.StatusAbnormalClosure,
			err.Error(), "timeout", false,
		)
	}
	if errors.Is(err, io.EOF) || errors.Is(err, net.ErrClosed) {
		return newRealtimeBridgeEnd(
			initiator, operation, ws.StatusAbnormalClosure,
			err.Error(), "eof", false,
		)
	}
	return newRealtimeBridgeEnd(
		initiator, operation, ws.StatusProtocolError,
		err.Error(), "protocol", false,
	)
}

func realtimeBridgeCloseCode(end *realtimeBridgeEnd) ws.StatusCode {
	if end == nil {
		return ws.StatusNormalClosure
	}
	code := ws.StatusCode(end.data.CloseCode)
	switch code {
	case ws.StatusNormalClosure,
		ws.StatusGoingAway,
		ws.StatusProtocolError,
		ws.StatusUnsupportedData,
		ws.StatusInvalidFramePayloadData,
		ws.StatusPolicyViolation,
		ws.StatusMessageTooBig,
		ws.StatusInternalServerError:
		return code
	default:
		return ws.StatusInternalServerError
	}
}

func realtimeBridgeShouldSendClose(end *realtimeBridgeEnd) bool {
	if end == nil {
		return true
	}
	if end.data.Operation == "write" || end.data.ErrorClass == "eof" {
		return false
	}
	return true
}

// realtimeAudioHandler is the HTTP handler for GET /realtime/audio.
// Performs the WebSocket upgrade, resolves the token, then runs coordinated
// read and write paths that shuttle binary frames between the WS and the
// thread's audio channels. PCM16 over the wire — no transcoding happens here;
// that's the bridge's (caller's) job.
//
// Closing the WS or killing the thread unwinds the pair: the read
// side returns an error, the write side stops on the audioOut channel
// closing, and the handler returns.
func (a *APIServer) realtimeAudioHandler(w http.ResponseWriter, r *http.Request) {
	thread := r.URL.Query().Get("thread")
	token := r.URL.Query().Get("token")
	if thread == "" || token == "" {
		http.Error(w, "thread and token query params required", http.StatusBadRequest)
		return
	}
	reg, err := peekAudioBridge(token)
	if err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}
	if reg.threadID != thread {
		http.Error(w, "token does not match thread", http.StatusForbidden)
		return
	}

	conn, _, _, err := ws.UpgradeHTTP(r, w)
	if err != nil {
		logMsg("REALTIME-AUDIO", fmt.Sprintf("upgrade failed for thread=%s: %v", thread, err))
		return
	}
	reg, err = claimAudioBridge(token)
	if err != nil {
		_ = conn.Close()
		return
	}
	connectedAt := time.Now()
	socket := &realtimeBridgeSocket{conn: conn}
	logMsg("REALTIME-AUDIO", fmt.Sprintf("bridge connected for thread=%s", thread))
	if a.thinker != nil && a.thinker.threads != nil {
		a.thinker.threads.realtimeBridgeConnected(thread)
	}
	if a.thinker != nil && a.thinker.telemetry != nil {
		a.thinker.telemetry.Emit("realtime.bridge_connected", thread, map[string]any{
			"encoding": "pcm16", "sample_rate": 24000, "channels": 1,
		})
	}

	// The handler owns application writes while the reader handles inbound
	// frames. Both paths share realtimeBridgeSocket, which serializes entire
	// WebSocket frames (including reader-generated pong and close responses).
	ctx, cancel := context.WithCancelCause(r.Context())
	live := &audioBridgeConnection{cancel: cancel}
	audioBridgeMu.Lock()
	audioBridgeLive[thread] = live
	audioBridgeMu.Unlock()

	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		if readErr := runRealtimeAudioBridgeReader(ctx, socket, reg, a, thread); readErr != nil {
			end := classifyRealtimeBridgeError("client", "read", readErr)
			logMsg("REALTIME-AUDIO", fmt.Sprintf(
				"read end for thread=%s operation=%s class=%s: %v",
				thread, end.data.Operation, end.data.ErrorClass, readErr,
			))
			cancel(end)
		}
	}()

	var termination *realtimeBridgeEnd
bridgeLoop:
	for {
		select {
		case <-ctx.Done():
			termination = classifyRealtimeBridgeError("context", "close", context.Cause(ctx))
			break bridgeLoop
		case frame, ok := <-reg.audioOut:
			if !ok {
				termination = newRealtimeBridgeEnd(
					"core", "close", ws.StatusNormalClosure,
					"audio output closed", "", true,
				)
				cancel(termination)
				break bridgeLoop
			}
			metadata, _ := json.Marshal(map[string]any{
				"type": "audio.frame", "response_id": frame.ResponseID,
				"item_id": frame.ItemID, "audio_end_ms": frame.AudioEndMS,
			})
			if err := socket.writeAudio(metadata, frame.Audio); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("write end for thread=%s: %v", thread, err))
				termination = classifyRealtimeBridgeError("core", "write", err)
				cancel(termination)
				break bridgeLoop
			}
		case control, ok := <-reg.control:
			if !ok {
				reg.control = nil
				continue
			}
			payload := []byte(fmt.Sprintf(`{"type":%q}`, control))
			if err := socket.writeText(payload); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("control write end for thread=%s: %v", thread, err))
				termination = classifyRealtimeBridgeError("core", "write", err)
				cancel(termination)
				break bridgeLoop
			}
		}
	}

	if termination == nil {
		termination = classifyRealtimeBridgeError("context", "close", context.Cause(ctx))
	}
	if realtimeBridgeShouldSendClose(termination) && !socket.hasSentClose() {
		code := realtimeBridgeCloseCode(termination)
		reason := termination.data.CloseReason
		if err := socket.writeClose(code, reason); err != nil {
			logMsg("REALTIME-AUDIO", fmt.Sprintf("close write end for thread=%s: %v", thread, err))
			if termination.data.Normal {
				termination = classifyRealtimeBridgeError("core", "close", err)
			}
		} else {
			// Give a cooperative peer a short opportunity to return its Close
			// frame. This delay occurs only during shutdown, never in audio flow.
			_ = conn.SetReadDeadline(time.Now().Add(realtimeBridgeCloseHandshakeWait))
			timer := time.NewTimer(realtimeBridgeCloseHandshakeWait)
			select {
			case <-readerDone:
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
			case <-timer.C:
			}
		}
	}

	cancel(termination)
	_ = conn.Close()
	select {
	case <-readerDone:
	case <-time.After(time.Second):
		logMsg("REALTIME-AUDIO", fmt.Sprintf("reader did not stop promptly for thread=%s", thread))
	}

	audioBridgeMu.Lock()
	for id, connected := range audioBridgeLive {
		if connected == live {
			delete(audioBridgeLive, id)
		}
	}
	audioBridgeMu.Unlock()

	termination.data.ConnectedMS = time.Since(connectedAt).Milliseconds()
	if a.thinker != nil && a.thinker.telemetry != nil {
		a.thinker.telemetry.Emit("realtime.bridge_disconnected", thread, termination.data)
	}
	if a.thinker != nil && a.thinker.threads != nil {
		a.thinker.threads.realtimeBridgeDisconnected(thread)
	}
}

func runRealtimeAudioBridgeReader(
	ctx context.Context,
	socket *realtimeBridgeSocket,
	reg *audioBridgeRegistration,
	a *APIServer,
	thread string,
) error {
	reader := wsutil.Reader{
		Source:         socket.conn,
		State:          ws.StateServerSide,
		CheckUTF8:      true,
		MaxFrameSize:   maxRealtimeAudioFrameBytes,
		OnIntermediate: socket.handleControl,
	}
	for {
		if err := socket.conn.SetReadDeadline(time.Now().Add(2 * time.Minute)); err != nil {
			return err
		}
		header, err := reader.NextFrame()
		if err != nil {
			return err
		}
		if header.OpCode.IsControl() {
			if err := socket.handleControl(header, &reader); err != nil {
				return err
			}
			continue
		}
		data, err := io.ReadAll(io.LimitReader(&reader, maxRealtimeAudioFrameBytes+1))
		if err != nil {
			return err
		}
		if len(data) > maxRealtimeAudioFrameBytes {
			return newRealtimeBridgeEnd(
				"client", "read", ws.StatusMessageTooBig,
				fmt.Sprintf("audio bridge frame exceeds %d bytes", maxRealtimeAudioFrameBytes),
				"protocol", false,
			)
		}
		if header.OpCode == ws.OpText {
			var control struct {
				Type       string `json:"type"`
				ItemID     string `json:"item_id"`
				AudioEndMS int    `json:"audio_end_ms"`
			}
			if json.Unmarshal(data, &control) == nil && a.thinker != nil && a.thinker.threads != nil {
				switch control.Type {
				case "playback.progress":
					if control.ItemID != "" && control.AudioEndMS >= 0 {
						a.thinker.threads.realtimePlaybackProgress(thread, control.ItemID, control.AudioEndMS)
					}
				case "playback.overflow":
					if control.ItemID != "" {
						a.thinker.threads.realtimePlaybackOverflow(thread, control.ItemID)
					}
				case "input.speech_started":
					a.thinker.threads.realtimeInputSpeechStarted(thread)
				}
			}
			continue
		}
		if header.OpCode != ws.OpBinary || len(data) == 0 {
			continue
		}
		timer := time.NewTimer(100 * time.Millisecond)
		select {
		case reg.audioIn <- data:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		case <-ctx.Done():
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			return context.Cause(ctx)
		case <-timer.C:
			if a.thinker != nil && a.thinker.telemetry != nil {
				a.thinker.telemetry.Emit("realtime.audio_overflow", thread, map[string]any{
					"direction": "input", "bytes": len(data), "reason": "thread_backpressure",
				})
			}
			return newRealtimeBridgeEnd(
				"core", "read", ws.StatusInternalServerError,
				"audio input delivery timed out", "timeout", false,
			)
		}
	}
}
