package core

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

type realtimeBridgeTestAddr string

func (a realtimeBridgeTestAddr) Network() string { return string(a) }
func (a realtimeBridgeTestAddr) String() string  { return string(a) }

// blockedRealtimeBridgeConn stops the second raw Write. ws.WriteFrame writes
// the frame header first and payload second, so this deterministically parks
// an audio metadata frame at the exact point where the old reader-side pong
// could corrupt it.
type blockedRealtimeBridgeConn struct {
	mu sync.Mutex

	wire         bytes.Buffer
	writeCalls   int
	activeWrites int
	maxActive    int
	secondWrite  chan struct{}
	release      chan struct{}
}

func newBlockedRealtimeBridgeConn() *blockedRealtimeBridgeConn {
	return &blockedRealtimeBridgeConn{
		secondWrite: make(chan struct{}),
		release:     make(chan struct{}),
	}
}

func (c *blockedRealtimeBridgeConn) Read([]byte) (int, error) { return 0, io.EOF }

func (c *blockedRealtimeBridgeConn) Write(payload []byte) (int, error) {
	c.mu.Lock()
	c.writeCalls++
	call := c.writeCalls
	c.activeWrites++
	if c.activeWrites > c.maxActive {
		c.maxActive = c.activeWrites
	}
	c.mu.Unlock()

	if call == 2 {
		close(c.secondWrite)
		<-c.release
	}

	c.mu.Lock()
	n, err := c.wire.Write(payload)
	c.activeWrites--
	c.mu.Unlock()
	return n, err
}

func (c *blockedRealtimeBridgeConn) Close() error                     { return nil }
func (c *blockedRealtimeBridgeConn) LocalAddr() net.Addr              { return realtimeBridgeTestAddr("local") }
func (c *blockedRealtimeBridgeConn) RemoteAddr() net.Addr             { return realtimeBridgeTestAddr("remote") }
func (c *blockedRealtimeBridgeConn) SetDeadline(time.Time) error      { return nil }
func (c *blockedRealtimeBridgeConn) SetReadDeadline(time.Time) error  { return nil }
func (c *blockedRealtimeBridgeConn) SetWriteDeadline(time.Time) error { return nil }
func (c *blockedRealtimeBridgeConn) snapshot() ([]byte, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]byte(nil), c.wire.Bytes()...), c.maxActive
}

func TestRealtimeAudioBridgeSerializesPongWithAudioFrame(t *testing.T) {
	conn := newBlockedRealtimeBridgeConn()
	socket := &realtimeBridgeSocket{conn: conn}
	audioDone := make(chan error, 1)
	go func() {
		audioDone <- socket.writeAudio([]byte(`{"type":"audio.frame"}`), []byte{1, 2, 3, 4})
	}()

	select {
	case <-conn.secondWrite:
	case <-time.After(time.Second):
		t.Fatal("audio write did not reach its payload")
	}

	pongStarted := make(chan struct{})
	pongDone := make(chan error, 1)
	go func() {
		close(pongStarted)
		pongDone <- socket.handleControl(
			ws.Header{Fin: true, OpCode: ws.OpPing, Length: 0},
			bytes.NewReader(nil),
		)
	}()
	<-pongStarted
	// Give the pong goroutine an opportunity to contend for the writer. It
	// must remain behind the complete metadata+audio pair.
	time.Sleep(20 * time.Millisecond)
	close(conn.release)

	if err := <-audioDone; err != nil {
		t.Fatalf("write audio: %v", err)
	}
	if err := <-pongDone; err != nil {
		t.Fatalf("handle ping: %v", err)
	}

	wire, maxActive := conn.snapshot()
	if maxActive != 1 {
		t.Fatalf("raw connection had %d concurrent writers, want 1", maxActive)
	}
	reader := bytes.NewReader(wire)
	var operations []ws.OpCode
	for reader.Len() > 0 {
		header, err := ws.ReadHeader(reader)
		if err != nil {
			t.Fatalf("corrupt websocket stream after concurrent ping/audio: %v", err)
		}
		payload := make([]byte, header.Length)
		if _, err := io.ReadFull(reader, payload); err != nil {
			t.Fatalf("truncated %v frame after concurrent ping/audio: %v", header.OpCode, err)
		}
		operations = append(operations, header.OpCode)
	}
	want := []ws.OpCode{ws.OpText, ws.OpBinary, ws.OpPong}
	if len(operations) != len(want) {
		t.Fatalf("frame operations = %v, want %v", operations, want)
	}
	for i := range want {
		if operations[i] != want[i] {
			t.Fatalf("frame operations = %v, want %v", operations, want)
		}
	}
}

func TestRealtimeAudioBridgeCloseHandshakeAndTelemetry(t *testing.T) {
	t.Run("core initiated", func(t *testing.T) {
		fixture := newRealtimeAudioBridgeFixture(t, "bridge-core-close")
		close(fixture.audioOut)

		header, payload := readRealtimeBridgeTestFrame(t, fixture.conn)
		if header.OpCode != ws.OpClose {
			t.Fatalf("server frame = %v, want close", header.OpCode)
		}
		code, reason := ws.ParseCloseFrameData(payload)
		if code != ws.StatusNormalClosure || reason != "audio output closed" {
			t.Fatalf("server close = %d %q, want 1000 audio output closed", code, reason)
		}
		writeRealtimeBridgeClientClose(t, fixture.conn, code, "ack")

		disconnect := waitForRealtimeBridgeDisconnect(t, fixture.telemetry, fixture.thread)
		assertRealtimeBridgeDisconnectFields(t, disconnect)
		if disconnect.Initiator != "core" || disconnect.Operation != "close" ||
			disconnect.CloseCode != int(ws.StatusNormalClosure) || !disconnect.Normal {
			t.Fatalf("disconnect = %#v, want normal core close", disconnect)
		}
	})

	t.Run("client initiated", func(t *testing.T) {
		fixture := newRealtimeAudioBridgeFixture(t, "bridge-client-close")
		writeRealtimeBridgeClientClose(t, fixture.conn, ws.StatusNormalClosure, "client done")

		header, payload := readRealtimeBridgeTestFrame(t, fixture.conn)
		if header.OpCode != ws.OpClose {
			t.Fatalf("server frame = %v, want close response", header.OpCode)
		}
		code, _ := ws.ParseCloseFrameData(payload)
		if code != ws.StatusNormalClosure {
			t.Fatalf("server close response code = %d, want 1000", code)
		}

		disconnect := waitForRealtimeBridgeDisconnect(t, fixture.telemetry, fixture.thread)
		assertRealtimeBridgeDisconnectFields(t, disconnect)
		if disconnect.Initiator != "client" || disconnect.Operation != "close" ||
			disconnect.CloseCode != int(ws.StatusNormalClosure) ||
			disconnect.CloseReason != "client done" || !disconnect.Normal {
			t.Fatalf("disconnect = %#v, want normal client close", disconnect)
		}
	})
}

type realtimeAudioBridgeFixture struct {
	thread    string
	audioOut  chan RealtimeAudioFrame
	conn      net.Conn
	telemetry *Telemetry
}

func newRealtimeAudioBridgeFixture(t *testing.T, thread string) *realtimeAudioBridgeFixture {
	t.Helper()
	t.Setenv("TELEMETRY_URL", "")
	t.Setenv("TELEMETRY_LIVE_URL", "")
	t.Setenv("SERVER_URL", "")

	telemetry := NewTelemetry()
	t.Cleanup(telemetry.Stop)
	thinker := &Thinker{telemetry: telemetry}
	api := &APIServer{thinker: thinker}
	server := httptest.NewServer(http.HandlerFunc(api.realtimeAudioHandler))
	t.Cleanup(server.Close)

	audioIn := make(chan []byte, 4)
	audioOut := make(chan RealtimeAudioFrame, 4)
	control := make(chan string, 4)
	token := registerAudioBridge(thread, audioIn, audioOut, control)
	t.Cleanup(func() { unregisterAudioBridge(thread) })

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") +
		"/realtime/audio?thread=" + thread + "&token=" + token
	conn, buffered, _, err := ws.Dial(context.Background(), wsURL)
	if err != nil {
		t.Fatalf("dial audio bridge: %v", err)
	}
	if buffered != nil {
		ws.PutReader(buffered)
		t.Fatal("unexpected buffered websocket data during bridge handshake")
	}
	t.Cleanup(func() { _ = conn.Close() })
	waitForRealtimeBridgeEvent(t, telemetry, thread, "realtime.bridge_connected")
	return &realtimeAudioBridgeFixture{
		thread: thread, audioOut: audioOut, conn: conn, telemetry: telemetry,
	}
}

func readRealtimeBridgeTestFrame(t *testing.T, conn net.Conn) (ws.Header, []byte) {
	t.Helper()
	if err := conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatal(err)
	}
	header, err := ws.ReadHeader(conn)
	if err != nil {
		t.Fatalf("read websocket header: %v", err)
	}
	payload := make([]byte, header.Length)
	if _, err := io.ReadFull(conn, payload); err != nil {
		t.Fatalf("read websocket payload: %v", err)
	}
	return header, payload
}

func writeRealtimeBridgeClientClose(t *testing.T, conn net.Conn, code ws.StatusCode, reason string) {
	t.Helper()
	frame := ws.MaskFrame(ws.NewCloseFrame(ws.NewCloseFrameBody(code, reason)))
	if err := ws.WriteFrame(conn, frame); err != nil {
		t.Fatalf("write client close: %v", err)
	}
}

func waitForRealtimeBridgeEvent(t *testing.T, telemetry *Telemetry, thread, eventType string) TelemetryEvent {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		events, _ := telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID == thread && event.Type == eventType {
				return event
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s on %s", eventType, thread)
	return TelemetryEvent{}
}

func waitForRealtimeBridgeDisconnect(
	t *testing.T,
	telemetry *Telemetry,
	thread string,
) realtimeBridgeDisconnectData {
	t.Helper()
	event := waitForRealtimeBridgeEvent(t, telemetry, thread, "realtime.bridge_disconnected")
	var fields map[string]any
	if err := json.Unmarshal(event.Data, &fields); err != nil {
		t.Fatalf("decode disconnect telemetry fields: %v", err)
	}
	for _, field := range []string{
		"initiator", "operation", "close_code", "close_reason",
		"error_class", "connected_ms", "normal",
	} {
		if _, ok := fields[field]; !ok {
			t.Errorf("disconnect telemetry missing %q: %s", field, event.Data)
		}
	}
	var disconnect realtimeBridgeDisconnectData
	if err := json.Unmarshal(event.Data, &disconnect); err != nil {
		t.Fatalf("decode disconnect telemetry: %v", err)
	}
	return disconnect
}

func assertRealtimeBridgeDisconnectFields(t *testing.T, disconnect realtimeBridgeDisconnectData) {
	t.Helper()
	if disconnect.ConnectedMS < 0 {
		t.Errorf("connected_ms = %d, want non-negative", disconnect.ConnectedMS)
	}
}

type realtimeBridgeTimeoutError struct{}

func (realtimeBridgeTimeoutError) Error() string   { return "bridge write timeout" }
func (realtimeBridgeTimeoutError) Timeout() bool   { return true }
func (realtimeBridgeTimeoutError) Temporary() bool { return true }

func TestRealtimeAudioBridgeErrorClassification(t *testing.T) {
	closed := classifyRealtimeBridgeError(
		"client", "read",
		wsutil.ClosedError{Code: ws.StatusNormalClosure, Reason: "done"},
	)
	if !closed.data.Normal || closed.data.Initiator != "client" ||
		closed.data.Operation != "close" || closed.data.CloseReason != "done" {
		t.Fatalf("normal close classification = %#v", closed.data)
	}

	eof := classifyRealtimeBridgeError("client", "read", io.EOF)
	if eof.data.Normal || eof.data.ErrorClass != "eof" ||
		eof.data.CloseCode != int(ws.StatusAbnormalClosure) {
		t.Fatalf("EOF classification = %#v", eof.data)
	}

	cancelled := classifyRealtimeBridgeError("context", "close", context.Canceled)
	if cancelled.data.Normal || cancelled.data.ErrorClass != "context_cancelled" ||
		cancelled.data.Initiator != "context" {
		t.Fatalf("context classification = %#v", cancelled.data)
	}

	pongTimeout := classifyRealtimeBridgeError(
		"client", "read",
		&realtimeBridgeControlError{op: ws.OpPing, err: realtimeBridgeTimeoutError{}},
	)
	if pongTimeout.data.Normal || pongTimeout.data.Operation != "pong" ||
		pongTimeout.data.ErrorClass != "timeout" {
		t.Fatalf("pong timeout classification = %#v", pongTimeout.data)
	}
}
