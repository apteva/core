package core

import (
	"net"
	"sync"
	"testing"
	"time"

	"github.com/gobwas/ws"
)

type delayedRealtimeWriteFailureConn struct {
	writeStarted chan struct{}
	releaseWrite chan struct{}
	closed       chan struct{}
	writeOnce    sync.Once
	closeOnce    sync.Once
}

func newDelayedRealtimeWriteFailureConn() *delayedRealtimeWriteFailureConn {
	return &delayedRealtimeWriteFailureConn{
		writeStarted: make(chan struct{}),
		releaseWrite: make(chan struct{}),
		closed:       make(chan struct{}),
	}
}

func (c *delayedRealtimeWriteFailureConn) Read([]byte) (int, error) {
	<-c.closed
	return 0, net.ErrClosed
}

func (c *delayedRealtimeWriteFailureConn) Write([]byte) (int, error) {
	c.writeOnce.Do(func() { close(c.writeStarted) })
	<-c.releaseWrite
	return 0, net.ErrClosed
}

func (c *delayedRealtimeWriteFailureConn) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return nil
}

func (*delayedRealtimeWriteFailureConn) LocalAddr() net.Addr              { return &net.TCPAddr{} }
func (*delayedRealtimeWriteFailureConn) RemoteAddr() net.Addr             { return &net.TCPAddr{} }
func (*delayedRealtimeWriteFailureConn) SetDeadline(time.Time) error      { return nil }
func (*delayedRealtimeWriteFailureConn) SetReadDeadline(time.Time) error  { return nil }
func (*delayedRealtimeWriteFailureConn) SetWriteDeadline(time.Time) error { return nil }

func assertRealtimeConcurrentClose(
	t *testing.T,
	events <-chan RealtimeEvent,
	stopped <-chan struct{},
	writeStarted <-chan struct{},
	releaseWrite chan struct{},
	closeSession func() error,
) {
	t.Helper()

	select {
	case <-writeStarted:
	case <-time.After(time.Second):
		t.Fatal("outbound write did not start")
	}

	started := time.Now()
	if err := closeSession(); err != nil {
		t.Fatalf("close session: %v", err)
	}
	if elapsed := time.Since(started); elapsed > 250*time.Millisecond {
		t.Fatalf("Close blocked for %s while an outbound write was active", elapsed)
	}

	select {
	case <-stopped:
		t.Fatal("session stopped before the active writer exited")
	default:
	}

	close(releaseWrite)
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("session goroutines did not terminate")
	}

	timeout := time.NewTimer(time.Second)
	defer timeout.Stop()
	for {
		select {
		case _, ok := <-events:
			if !ok {
				if err := closeSession(); err != nil {
					t.Fatalf("repeated close: %v", err)
				}
				return
			}
		case <-timeout.C:
			t.Fatal("event stream did not close")
		}
	}
}

func TestGoogleRealtimeConcurrentClose(t *testing.T) {
	conn := newDelayedRealtimeWriteFailureConn()
	session := &googleRealtimeSession{
		conn: conn, events: make(chan RealtimeEvent, 8),
		outbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{}),
		ready: make(chan error, 1), callNames: map[string]string{},
	}
	session.outbox <- realtimeOutboundFrame{op: ws.OpText, data: []byte(`{"test":true}`)}
	session.lifecycle.start(session.events, session.readLoop, session.writeLoop, session.pingLoop)

	assertRealtimeConcurrentClose(
		t, session.events, session.lifecycle.stopped,
		conn.writeStarted, conn.releaseWrite, session.Close,
	)
}

func TestOpenAIRealtimeConcurrentClose(t *testing.T) {
	conn := newDelayedRealtimeWriteFailureConn()
	session := &openaiRealtimeSession{
		conn: conn, events: make(chan RealtimeEvent, 8),
		outbox: make(chan realtimeOutboundFrame, 1), done: make(chan struct{}),
		providerName: "openai-realtime", itemPhases: map[string]string{},
	}
	session.outbox <- realtimeOutboundFrame{op: ws.OpText, data: []byte(`{"type":"test"}`)}
	session.lifecycle.start(session.events, session.readLoop, session.writeLoop, session.pingLoop)

	assertRealtimeConcurrentClose(
		t, session.events, session.lifecycle.stopped,
		conn.writeStarted, conn.releaseWrite, session.Close,
	)
}
