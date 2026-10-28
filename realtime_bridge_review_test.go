package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

// runBridgeReaderMessages feeds client text controls through the real bridge
// reader for a managed realtime thread and waits for the reader to stop.
func runBridgeReaderMessages(t *testing.T, rt *RealtimeThinker, socket *realtimeBridgeSocket, payloads ...string) {
	t.Helper()
	parent := newTestThinker()
	defer parent.Stop()
	parent.threads.threads["voice"] = &Thread{ID: "voice", Realtime: rt, Thinker: rt.Thinker, IsRealtime: true}
	api := &APIServer{thinker: parent}
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	socket.conn = conn
	done := make(chan error, 1)
	go func() {
		done <- runRealtimeAudioBridgeReader(context.Background(), socket, &audioBridgeRegistration{}, api, "voice")
	}()
	for _, payload := range payloads {
		if err := wsutil.WriteClientMessage(peer, ws.OpText, []byte(payload)); err != nil {
			t.Fatal(err)
		}
	}
	peer.Close()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("reader did not stop")
	}
}

func TestRendererOverflowReturnsPlaybackCredit(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true}
	rt.sessionGeneration = 3
	rt.outputItemID = "reply"
	rt.outputBytes = 960
	rt.delivery.enqueue("3/reply", "reply", 960)(true)
	socket := &realtimeBridgeSocket{}
	socket.window.configure(20)
	if !socket.window.reserve(3, "reply", 960, 960) {
		t.Fatal("fixture reserve failed")
	}
	runBridgeReaderMessages(t, rt, socket, `{"type":"playback.overflow","item_id":"reply","session_generation":3}`)
	if !socket.window.reserve(3, "next", 960, 960) {
		t.Fatal("overflowed item kept its playback credit")
	}
}

func TestPlaybackWindowReleasesCreditForCancelledChunk(t *testing.T) {
	s := &realtimeBridgeSocket{writeTimeout: time.Second}
	s.window.configure(20)
	checks := 0
	// The pre-reserve check passes; cancellation lands before the write.
	cancelled := func() bool { checks++; return checks > 1 }
	f := RealtimeAudioFrame{SessionGeneration: 1, ItemID: "old", Audio: make([]byte, 960), AudioEndMS: 20, cancelled: cancelled}
	written, err := s.writeFlowControlledAudio(context.Background(), f, func() error { return nil })
	if written != 0 || err != nil {
		t.Fatal(written, err)
	}
	if !s.window.reserve(1, "next", 960, 960) {
		t.Fatal("credit reserved for an unsent chunk was never returned")
	}
}

func TestPlaybackWindowReadyResetsStaleCredits(t *testing.T) {
	var w realtimePlaybackWindow
	w.configure(20)
	if !w.reserve(1, "old", 960, 960) {
		t.Fatal("fixture reserve failed")
	}
	w.configure(20) // renderer re-sent playback.ready with a fresh player
	if !w.reserve(1, "next", 960, 960) {
		t.Fatal("stale credits survived playback.ready")
	}
}

func TestPlaybackWindowUsesByteExactFrameOffsets(t *testing.T) {
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	s := &realtimeBridgeSocket{conn: conn, writeTimeout: time.Second}
	s.window.configure(200)
	// 4810 bytes is 100.2 ms at 48000 B/s; AudioEndMS truncates to 100.
	f := RealtimeAudioFrame{SessionGeneration: 2, ItemID: "reply", Audio: make([]byte, 1930), AudioEndMS: 100, audioEndBytes: 4810}
	done := make(chan error, 1)
	go func() {
		_, err := s.writeFlowControlledAudio(context.Background(), f, func() error { return nil })
		done <- err
	}()
	var ends []any
	for i := 0; i < 3; i++ {
		raw, _, err := wsutil.ReadServerData(peer)
		if err != nil {
			t.Fatal(err)
		}
		var meta map[string]any
		_ = json.Unmarshal(raw, &meta)
		ends = append(ends, meta["audio_end_ms"])
		if _, _, err := wsutil.ReadServerData(peer); err != nil {
			t.Fatal(err)
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if ends[0] != float64(80) || ends[1] != float64(100) || ends[2] != float64(100) {
		t.Fatal("frame offsets drifted from item bytes", ends)
	}
	s.window.mu.Lock()
	sent := s.window.items[playbackCreditKey(2, "reply")].sent
	s.window.mu.Unlock()
	if sent != 4810 {
		t.Fatal("reserved", sent)
	}
	s.window.progress(2, "reply", 100)
	s.window.mu.Lock()
	credit := s.window.items[playbackCreditKey(2, "reply")]
	s.window.mu.Unlock()
	if credit.played != credit.sent {
		t.Fatal("whole-ms receipt left sub-ms credit outstanding", credit)
	}
}

func TestDiscardKeepsUnrelatedPlaybackInOrderAheadOfBacklog(t *testing.T) {
	rt := &RealtimeThinker{audioOut: make(chan RealtimeAudioFrame, 4)}
	for i, item := range []string{"b", "a", "b", "a"} {
		rt.audioOut <- RealtimeAudioFrame{ItemID: item, AudioEndMS: i, Audio: []byte{0, 0}}
	}
	now := time.Now()
	rt.outputPending = []RealtimeAudioFrame{{ItemID: "b", AudioEndMS: 4, Audio: []byte{0, 0}, queuedAt: now}, {ItemID: "a", AudioEndMS: 5, Audio: []byte{0, 0}, queuedAt: now}}
	rt.outputPendingBytes = 4
	rt.outputMu.Lock()
	frames, _ := rt.discardQueuedPlayback("a")
	rt.outputMu.Unlock()
	if frames != 3 {
		t.Fatal("discarded", frames)
	}
	rt.pumpRealtimeOutput()
	var got []int
	next := orderedAudioReader(rt.audioOut)
	for len(got) < 3 {
		f := next()
		if f.ItemID != "b" {
			t.Fatal("cancelled item survived", f.AudioEndMS)
		}
		got = append(got, f.AudioEndMS)
	}
	if len(got) != 3 || got[0] != 0 || got[1] != 2 || got[2] != 4 || len(rt.outputPending) != 0 || rt.outputPendingBytes != 0 {
		t.Fatal("kept playback reordered", got, len(rt.outputPending), rt.outputPendingBytes)
	}
}

func TestRendererInterruptNeverWaitsOnBridgeAndClearPrecedesLaterAudio(t *testing.T) {
	audioOut := make(chan RealtimeAudioFrame, 8)
	control := make(chan string, 1)
	control <- `{"type":"busy"}` // a stalled bridge has not drained the control lane
	rt := newRealtimeThinker(context.Background(), newTestThinker(), &fakeRealtimeProvider{}, "", nil, audioOut, control)
	rt.rendererScoped.Store(true)
	rt.replaceSession(newFakeRealtimeSession())
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "one", Audio: make([]byte, 480)})
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "two", Audio: make([]byte, 480)})
	started := time.Now()
	rt.clearCallerInterruptedPlayback("test", false, true)
	rt.outputMu.Lock()
	queued := len(rt.pendingControls)
	coalesced := queued == 2
	if coalesced {
		// A repeated clear for the same generation/item replaces, never duplicates.
		rt.queueRendererControlLocked(rt.pendingControls[0].key, rt.pendingControls[0].payload)
		coalesced = len(rt.pendingControls) == queued
	}
	rt.outputMu.Unlock()
	if !coalesced {
		t.Fatal("clears were not held in order for the stalled bridge", queued)
	}
	if elapsed := time.Since(started); elapsed > 50*time.Millisecond {
		t.Fatal("interrupt waited on the bridge control lane", elapsed)
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventAudioOut, ItemID: "three", Audio: make([]byte, 480)})
	if len(audioOut) != 0 {
		t.Fatal("later audio overtook an undelivered clear")
	}
	var clears []string
	for len(clears) < 2 {
		var message map[string]any
		_ = json.Unmarshal([]byte(<-control), &message)
		if message["type"] == "interrupt" {
			clears = append(clears, message["item_id"].(string)+"/"+message["reason"].(string))
		}
		if len(clears) < 2 && len(audioOut) != 0 {
			t.Fatal("audio released before every clear was delivered")
		}
		rt.pumpRealtimeOutput()
	}
	if clears[0] != "one/test" || clears[1] != "two/test" {
		t.Fatal("clears lost, reordered or not coalesced", clears)
	}
	if len(control) != 0 {
		t.Fatal("duplicate clear", <-control)
	}
	if len(audioOut) != 1 || (<-audioOut).ItemID != "three" {
		t.Fatal("later audio not released after the clears")
	}
}

type replacementBridgeFixture struct {
	thinker  *Thinker
	server   *httptest.Server
	thread   string
	audioIn  chan []byte
	audioOut chan RealtimeAudioFrame
	control  chan string
}

func newReplacementBridgeFixture(t *testing.T, thread string, rt *RealtimeThinker) *replacementBridgeFixture {
	t.Helper()
	t.Setenv("TELEMETRY_URL", "")
	t.Setenv("TELEMETRY_LIVE_URL", "")
	t.Setenv("SERVER_URL", "")
	thinker := newTestThinker()
	thinker.telemetry = NewTelemetry()
	t.Cleanup(thinker.telemetry.Stop)
	t.Cleanup(thinker.Stop)
	thinker.threads.threads[thread] = &Thread{ID: thread, IsRealtime: true, BridgeDisconnectTTL: time.Hour, Realtime: rt}
	server := httptest.NewServer(http.HandlerFunc((&APIServer{thinker: thinker}).realtimeAudioHandler))
	t.Cleanup(server.Close)
	f := &replacementBridgeFixture{thinker: thinker, server: server, thread: thread, audioIn: make(chan []byte, 4), audioOut: make(chan RealtimeAudioFrame, 8), control: make(chan string, 8)}
	t.Cleanup(func() { unregisterAudioBridge(thread) })
	return f
}

func (f *replacementBridgeFixture) dial(t *testing.T) net.Conn {
	t.Helper()
	token := registerAudioBridge(f.thread, f.audioIn, f.audioOut, f.control)
	url := "ws" + strings.TrimPrefix(f.server.URL, "http") + "/realtime/audio?thread=" + f.thread + "&token=" + token
	conn, buffered, _, err := ws.Dial(context.Background(), url)
	if err != nil {
		t.Fatal("dial", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if buffered != nil {
		// Carried audio may follow the handshake in the same read.
		return &bufferedTestConn{Conn: conn, r: buffered}
	}
	return conn
}

type bufferedTestConn struct {
	net.Conn
	r *bufio.Reader
}

func (c *bufferedTestConn) Read(p []byte) (int, error) { return c.r.Read(p) }

func TestRealtimeBridgeReplacementOwnsOutputAndKeepsCallAlive(t *testing.T) {
	f := newReplacementBridgeFixture(t, "voice-replaced", nil)
	_ = f.dial(t)
	waitForRealtimeBridgeEvent(t, f.thinker.telemetry, f.thread, "realtime.bridge_connected")
	second := f.dial(t)
	// The replaced bridge ends promptly instead of competing for audio.
	disconnect := waitForRealtimeBridgeDisconnect(t, f.thinker.telemetry, f.thread)
	if disconnect.ErrorClass != "replaced" {
		t.Fatalf("replaced bridge end = %#v", disconnect)
	}
	for i := 1; i <= 4; i++ {
		f.audioOut <- RealtimeAudioFrame{ItemID: "reply", Audio: []byte{byte(i), 0}, AudioEndMS: i}
	}
	for i := 1; i <= 4; i++ {
		if header, _ := readRealtimeBridgeTestFrame(t, second); header.OpCode != ws.OpText {
			t.Fatal("expected metadata", header.OpCode)
		}
		header, data := readRealtimeBridgeTestFrame(t, second)
		if header.OpCode != ws.OpBinary || !bytes.Equal(data, []byte{byte(i), 0}) {
			t.Fatalf("frame %d went elsewhere or out of order: %v", i, data)
		}
	}
	owner, thread := f.thinker.threads.findManagedThread(f.thread)
	owner.mu.Lock()
	connected, timer := thread.bridgeConnected, thread.bridgeCleanupTimer
	owner.mu.Unlock()
	if !connected || timer != nil {
		t.Fatal("replaced bridge reported the live call as disconnected")
	}
}

func TestRealtimeBridgeConnectDiscardsStaleRendererControls(t *testing.T) {
	rt, _ := completionFixture(t)
	f := newReplacementBridgeFixture(t, "voice-stale-controls", rt)
	rt.audioControl = f.control
	f.control <- `{"type":"interrupt","item_id":"old-player","requested_at_ms":1}`
	rt.outputMu.Lock()
	rt.queueRendererControlLocked("1/older", `{"type":"interrupt","item_id":"older"}`)
	rt.outputMu.Unlock()
	conn := f.dial(t)
	waitForRealtimeBridgeEvent(t, f.thinker.telemetry, f.thread, "realtime.bridge_connected")
	f.audioOut <- RealtimeAudioFrame{ItemID: "fresh", Audio: []byte{7, 0}, AudioEndMS: 1}
	header, data := readRealtimeBridgeTestFrame(t, conn)
	var message map[string]any
	if header.OpCode != ws.OpText || json.Unmarshal(data, &message) != nil || message["type"] != "audio.frame" || message["item_id"] != "fresh" {
		t.Fatalf("stale control replayed to the new renderer: %s", data)
	}
	rt.outputMu.Lock()
	pending := len(rt.pendingControls)
	rt.outputMu.Unlock()
	if pending != 0 {
		t.Fatal("pending controls survived reconnect", pending)
	}
}

func TestRealtimeFrameTraceCarriesProviderReadTiming(t *testing.T) {
	s := &openaiRealtimeSession{events: make(chan RealtimeEvent, 1), itemPhases: map[string]string{}, audioOutFormat: AudioPCM16, audioOutRate: 24000}
	started := time.Now().Add(-30 * time.Millisecond)
	s.readStartedAt, s.readReturnedAt = started, started.Add(12*time.Millisecond)
	s.translate(&openaiRealtimeEvent{Type: "response.output_audio.delta", ItemID: "reply", Delta: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}) // 24 bytes = 0.5 ms
	event := <-s.events
	frame := RealtimeAudioFrame{providerReadStartedAt: event.providerReadStartedAt, providerReadReturnedAt: event.providerReadReturnedAt, providerDeltaDurationMS: event.providerDeltaDurationMS}
	timing := realtimeFrameTiming(frame, time.Now(), 20)
	if timing["provider_read_started_at"] != started.UTC().Format(time.RFC3339Nano) || timing["provider_read_wait_ms"] != float64(12) || timing["provider_delta_duration_ms"] != 0.5 {
		t.Fatal(timing)
	}
	// WebRTC media has no provider WebSocket read; the fields are omitted.
	timing = realtimeFrameTiming(RealtimeAudioFrame{}, time.Now(), 20)
	for _, key := range []string{"provider_read_started_at", "provider_read_wait_ms", "provider_delta_duration_ms"} {
		if _, ok := timing[key]; ok {
			t.Fatal("fabricated", key)
		}
	}
}

func TestRealtimeBridgeReplacementAbortsStalledWriteAndCarriesFrame(t *testing.T) {
	f := newReplacementBridgeFixture(t, "voice-stalled", nil)
	_ = f.dial(t) // never reads: the old writer stalls inside its write
	waitForRealtimeBridgeEvent(t, f.thinker.telemetry, f.thread, "realtime.bridge_connected")
	audio := make([]byte, 8<<20)
	audio[len(audio)-1] = 9
	f.audioOut <- RealtimeAudioFrame{ItemID: "reply", Audio: audio, AudioEndMS: 1}
	deadline := time.Now().Add(2 * time.Second)
	for len(f.audioOut) > 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond)
	started := time.Now()
	second := f.dial(t)
	if header, _ := readRealtimeBridgeTestFrame(t, second); header.OpCode != ws.OpText {
		t.Fatal("expected metadata", header.OpCode)
	}
	header, data := readRealtimeBridgeTestFrame(t, second)
	if header.OpCode != ws.OpBinary || len(data) != len(audio) || data[len(data)-1] != 9 {
		t.Fatal("in-flight frame lost on replacement", header.OpCode, len(data))
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatal("replacement waited on the stalled writer", elapsed)
	}
}

// A writer that receives while a late interrupt drains the channel must never
// overtake frames the drain is about to re-queue.
func TestDiscardConcurrentWriterKeepsPlaybackOrder(t *testing.T) {
	for round := 0; round < 200; round++ {
		out := make(chan RealtimeAudioFrame, 64)
		rt := &RealtimeThinker{audioOut: out}
		now := time.Now()
		var written []uint64
		done := make(chan struct{})
		stop := make(chan struct{})
		go func() {
			defer close(done)
			for {
				select {
				case f := <-out:
					for _, w := range takeOrderedAudio(out, f) {
						written = append(written, w.seq)
					}
				case <-stop:
					return
				}
			}
		}()
		for i := 0; i < 48; i++ {
			item := "keep"
			if i%3 == 1 {
				item = "cut"
			}
			rt.outputMu.Lock()
			rt.queueRealtimeOutputLocked(RealtimeAudioFrame{ItemID: item, Audio: []byte{0, 0}, queuedAt: now})
			if i%8 == 7 {
				rt.discardQueuedPlayback("cut")
			}
			rt.pumpRealtimeOutputLocked()
			rt.outputMu.Unlock()
		}
		for deadline := time.Now().Add(time.Second); time.Now().Before(deadline); {
			rt.outputMu.Lock()
			rt.pumpRealtimeOutputLocked()
			empty := len(rt.outputPending) == 0 && len(out) == 0
			rt.outputMu.Unlock()
			if empty {
				break
			}
			time.Sleep(time.Millisecond)
		}
		time.Sleep(2 * time.Millisecond)
		close(stop)
		<-done
		for _, w := range takeOrderedAudio(out, RealtimeAudioFrame{wake: true}) {
			written = append(written, w.seq)
		}
		for i := 1; i < len(written); i++ {
			if written[i] <= written[i-1] {
				t.Fatalf("round %d: playback reordered: %v", round, written)
			}
		}
	}
}

// orderedAudioReader receives renderer audio the way the bridge writer does:
// parked frames and the received one in seq order, wake markers skipped.
func orderedAudioReader(out chan RealtimeAudioFrame) func() RealtimeAudioFrame {
	var buffered []RealtimeAudioFrame
	return func() RealtimeAudioFrame {
		for len(buffered) == 0 {
			buffered = takeOrderedAudio(out, <-out)
		}
		f := buffered[0]
		buffered = buffered[1:]
		return f
	}
}
