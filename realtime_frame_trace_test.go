package core

import (
	"context"
	"encoding/json"
	"github.com/gobwas/ws/wsutil"
	"net"
	"testing"
	"time"
)

func TestRealtimeFrameTraceOptInAndClockLocal(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		conn, peer := net.Pipe()
		s := &realtimeBridgeSocket{conn: conn, writeTimeout: time.Second}
		s.window.configure(200)
		s.window.configureTrace(enabled)
		base := time.Now().Add(-50 * time.Millisecond)
		f := RealtimeAudioFrame{SessionGeneration: 4, ItemID: "reply", Audio: make([]byte, 1920), AudioEndMS: 40, providerObservedAt: base, queuedAt: base.Add(10 * time.Millisecond)}
		done := make(chan error, 1)
		go func() {
			_, err := s.writeFlowControlledAudio(context.Background(), f, func() error { return nil })
			done <- err
		}()
		for i := 1; i <= 2; i++ {
			raw, _, err := wsutil.ReadServerData(peer)
			if err != nil {
				t.Fatal(err)
			}
			var meta map[string]any
			if err = json.Unmarshal(raw, &meta); err != nil {
				t.Fatal(err)
			}
			timing, ok := meta["timing"].(map[string]any)
			if ok != enabled {
				t.Fatal(meta)
			}
			if enabled {
				if timing["duration_ms"] != float64(20) || timing["provider_to_write_ms"].(float64) < 50 || timing["core_queue_ms"].(float64) < 40 {
					t.Fatal(timing)
				}
			}
			if meta["audio_end_ms"] != float64(i*20) {
				t.Fatal(meta)
			}
			pcm, _, err := wsutil.ReadServerData(peer)
			if err != nil || len(pcm) != 960 {
				t.Fatal(err, len(pcm))
			}
		}
		if err := <-done; err != nil {
			t.Fatal(err)
		}
		conn.Close()
		peer.Close()
	}
	future := time.Now().Add(time.Hour)
	timing := realtimeFrameTiming(RealtimeAudioFrame{providerObservedAt: future, queuedAt: future}, time.Now(), 20)
	if _, ok := timing["provider_to_write_ms"]; ok {
		t.Fatal("fabricated clock duration")
	}
}
