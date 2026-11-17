package core

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
	"net"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRendererInputBoundIsAudioDuration(t *testing.T) {
	input := make(chan []byte, 64)
	pcm := make([]byte, 5760)
	for i := range pcm {
		pcm[i] = byte(i)
	}
	if err := enqueueLiveRendererInput(context.Background(), input, pcm); err != nil {
		t.Fatal(err)
	}
	if len(input) != 6 {
		t.Fatal("not normalized", len(input))
	}
	if err := enqueueLiveRendererInput(context.Background(), input, make([]byte, 960)); err != nil {
		t.Fatal("fresh input rejected after only 120ms", err)
	}
	pcm = append(pcm, make([]byte, 960)...)
	var got []byte
	for len(input) > 0 {
		got = append(got, (<-input)...)
	}
	if !bytes.Equal(got, pcm) {
		t.Fatal("microphone PCM changed")
	}
	if err := enqueueLiveRendererInput(context.Background(), input, []byte{1}); err == nil {
		t.Fatal("unaligned PCM admitted")
	}
}

func TestRendererCapabilitiesSeparateProviderVADFromInterruption(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = capabilitiesForRealtime(NewOpenAIRealtimeProvider("fixture"), "gpt-live-1")
	parent := newTestThinker()
	defer parent.Stop()
	parent.threads.threads["voice"] = &Thread{ID: "voice", Realtime: rt, Thinker: rt.Thinker, IsRealtime: true}
	api := &APIServer{thinker: parent}
	request := httptest.NewRequest("GET", "/realtime/audio?audio_protocol=1", nil)
	raw := api.realtimeRendererContract("voice", request)
	var caps map[string]any
	if json.Unmarshal([]byte(raw), &caps) != nil {
		t.Fatal("missing contract")
	}
	if caps["continuous_conversation"] != true || caps["input_speech_events"] != false || caps["local_interruption_required"] != false || caps["provider_interruption"] != true {
		t.Fatal(caps)
	}
	request = httptest.NewRequest("GET", "/realtime/audio", nil)
	if api.realtimeRendererContract("voice", request) != "" {
		t.Fatal("legacy changed")
	}
}

func TestRendererReceiptsDoNotInventAudibleConfirmation(t *testing.T) {
	d := &realtimeDelivery{}
	d.setEvidence("carrier_mark")
	d.playbackReady()
	_, _, data := d.snapshot()
	if data["audible_playback_confirmed"] == true {
		t.Fatal("ready is not playback")
	}
	sent := d.enqueue("7/reply", "reply", 960)
	sent(true)
	for _, e := range []string{"estimated", "rtp_transmitted"} {
		d.setEvidence(e)
		d.acknowledge("7/reply", 20)
		_, _, data = d.snapshot()
		if data["audible_playback_confirmed"] == true {
			t.Fatal(e, "invented playback")
		}
	}
	d.setEvidence("carrier_mark")
	_, _, data = d.snapshot()
	if data["audible_playback_confirmed"] != true {
		t.Fatal("mark not retained")
	}
	d.abandonPlayback("reply")
	if d.generationFor("reply", 8) != 7 {
		t.Fatal("cancel lost original generation")
	}
	d.acknowledge("7/reply", 999)
	_, _, data = d.snapshot()
	if data["acknowledged_audio_ms"] != float64(20) {
		t.Fatal("late cleared mark accepted")
	}
}

func TestRendererProviderControlLanePrecedesAudio(t *testing.T) {
	conn, peer := net.Pipe()
	defer peer.Close()
	session := &openaiRealtimeSession{conn: conn, events: make(chan RealtimeEvent, 8), outbox: make(chan realtimeOutboundFrame, 8), audioOutbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}), writeTimeout: 50 * time.Millisecond}
	defer session.Close()
	session.audioOutbox <- realtimeOutboundFrame{op: ws.OpText, data: []byte(`{"type":"audio"}`)}
	session.outbox <- realtimeOutboundFrame{op: ws.OpText, data: []byte(`{"type":"control"}`)}
	go session.writeLoop()
	_ = peer.SetReadDeadline(time.Now().Add(time.Second))
	for _, want := range []string{`{"type":"control"}`, `{"type":"audio"}`} {
		raw, _, err := wsutil.ReadClientData(peer)
		if err != nil || string(raw) != want {
			t.Fatal(string(raw), err)
		}
	}
}

func TestRendererProviderBlockedWriteEndsWithinBudget(t *testing.T) {
	conn, peer := net.Pipe()
	defer peer.Close()
	session := &openaiRealtimeSession{conn: conn, events: make(chan RealtimeEvent, 8), outbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}), writeTimeout: 50 * time.Millisecond}
	defer session.Close()
	start := time.Now()
	go session.writeLoop()
	session.outbox <- realtimeOutboundFrame{op: ws.OpText, data: []byte(`{"type":"control"}`)}
	select {
	case <-session.done:
		if time.Since(start) > 500*time.Millisecond {
			t.Fatal("unbounded write")
		}
	case <-time.After(time.Second):
		t.Fatal("blocked indefinitely")
	}
}

func TestRendererStaleOverflowCannotCancelNextGeneration(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{ContinuousConversation: true}
	rt.sessionGeneration = 8
	rt.outputItemID = "same-id"
	rt.outputBytes = 960
	rt.delivery.enqueue("7/same-id", "same-id", 960)(true)
	rt.delivery.enqueue("8/same-id", "same-id", 960)(true)
	rt.rendererPlaybackOverflowForGeneration("same-id", 7)
	if !rt.delivery.isCancelled("7/same-id") || rt.delivery.isCancelled("8/same-id") || rt.cancelledPlaybacks["same-id"] {
		t.Fatal("old generation cancelled new reply")
	}
}

func BenchmarkRendererInputAdmission(b *testing.B) {
	input := make(chan []byte, 6)
	pcm := make([]byte, 960)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := enqueueLiveRendererInput(context.Background(), input, pcm); err != nil {
			b.Fatal(err)
		}
		<-input
	}
}

func TestRendererProgressWireRetainsEvidenceAndRejectsStaleGeneration(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.sessionGeneration = 3
	rt.outputItemID = "newer-output"
	receipts := make(chan map[string]any, 4)
	rt.timing = newRealtimeTiming(3, func(k string, d map[string]any) {
		if k == "realtime.playback_timing" {
			receipts <- d
		}
	})
	at := time.Now()
	rt.timing.audio("r", "old-output", at)
	rt.timing.enqueued("r", "old-output", at, 1, 960)
	rt.timing.written("r", "old-output", at, 960, true)
	rt.timing.finish("r", "old-output", "completed")
	parent := newTestThinker()
	defer parent.Stop()
	parent.threads.threads["voice"] = &Thread{ID: "voice", Realtime: rt, Thinker: rt.Thinker, IsRealtime: true}
	api := &APIServer{thinker: parent}
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	done := make(chan error, 1)
	go func() {
		done <- runRealtimeAudioBridgeReader(context.Background(), &realtimeBridgeSocket{conn: conn}, &audioBridgeRegistration{}, api, "voice")
	}()
	for _, payload := range []string{
		`{"type":"playback.progress","item_id":"old-output","audio_end_ms":20,"session_generation":2,"evidence":"carrier_mark"}`,
		`{"type":"playback.progress","item_id":"old-output","audio_end_ms":20,"session_generation":3,"evidence":"invented"}`,
		`{"type":"playback.progress","item_id":"old-output","audio_end_ms":20,"session_generation":3,"evidence":"rtp_transmitted"}`,
	} {
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
	if len(receipts) != 1 {
		t.Fatalf("receipts=%d", len(receipts))
	}
	receipt := <-receipts
	if receipt["evidence"] != "rtp_transmitted" || receipt["audible_playback_confirmed"] != false {
		t.Fatalf("lost receipt meaning: %v", receipt)
	}
}
