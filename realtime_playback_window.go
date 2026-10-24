package core

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

type realtimePlaybackCredit struct{ sent, played int }

// Socket-local credits require exact generation/item receipts. Late receipts
// cannot open the window of a replacement session or another playback.
type realtimePlaybackWindow struct {
	bytesPerSecond int
	mu             sync.Mutex
	capacity       int
	items          map[string]realtimePlaybackCredit
}

func playbackCreditKey(g int64, item string) string { return fmt.Sprintf("%d/%s", g, item) }
func (w *realtimePlaybackWindow) configure(ms int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if ms < 20 {
		ms = 20
	}
	if ms > 15000 {
		ms = 15000
	}
	w.capacity = ms * defaultMediaRate(w.bytesPerSecond) / 1000
	if w.items == nil {
		w.items = map[string]realtimePlaybackCredit{}
	}
}
func (w *realtimePlaybackWindow) progress(g int64, item string, ms int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	key := playbackCreditKey(g, item)
	p, ok := w.items[key]
	if !ok {
		return
	}
	end := min(p.sent, ms*defaultMediaRate(w.bytesPerSecond)/1000)
	if end > p.played {
		p.played = end
	}
	w.items[key] = p
}
func (w *realtimePlaybackWindow) clear(g int64, item string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	delete(w.items, playbackCreditKey(g, item))
}
func (w *realtimePlaybackWindow) enabled() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.capacity > 0
}
func (w *realtimePlaybackWindow) reserve(g int64, item string, n, end int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.capacity == 0 {
		return true
	}
	pending := 0
	for _, p := range w.items {
		pending += p.sent - p.played
	}
	if pending+n > w.capacity {
		return false
	}
	key := playbackCreditKey(g, item)
	p := w.items[key]
	p.sent = end
	// Prune fully played receipts, while never dropping outstanding credits.
	if len(w.items) >= 256 {
		for k, v := range w.items {
			if v.played >= v.sent {
				delete(w.items, k)
			}
		}
	}
	w.items[key] = p
	return true
}

// The writer continues flushing interruption/state controls while waiting for
// playback progress. The reader and provider audio lane remain independent.
func (s *realtimeBridgeSocket) writeFlowControlledAudio(ctx context.Context, f RealtimeAudioFrame, controls func() error) (int, error) {
	started := time.Now()
	frameBytes := len(f.Audio)
	if s.window.enabled() {
		frameBytes = defaultMediaRate(s.window.bytesPerSecond) / 50
	}
	if frameBytes == 0 {
		return 0, nil
	}
	startBytes := f.AudioEndMS*defaultMediaRate(s.window.bytesPerSecond)/1000 - len(f.Audio)
	offset := 0
	waitingSince := time.Now()
	for offset < len(f.Audio) {
		if err := controls(); err != nil {
			return offset, err
		}
		if f.cancelled != nil && f.cancelled() {
			return offset, nil
		}
		end := min(offset+frameBytes, len(f.Audio))
		if !s.window.reserve(f.SessionGeneration, f.ItemID, end-offset, startBytes+end) {
			if time.Since(started) > realtimeOutputBufferDuration {
				return offset, fmt.Errorf("renderer playback progress timed out")
			}
			timer := time.NewTimer(5 * time.Millisecond)
			select {
			case <-ctx.Done():
				timer.Stop()
				return offset, context.Cause(ctx)
			case <-timer.C:
			}
			continue
		}
		metadata, _ := json.Marshal(map[string]any{"type": "audio.frame", "response_id": f.ResponseID, "item_id": f.ItemID, "audio_end_ms": (startBytes + end) * 1000 / defaultMediaRate(s.window.bytesPerSecond), "silence": f.Silent, "session_generation": f.SessionGeneration})
		writeStarted := time.Now()
		written, err := s.writePlaybackAudio(metadata, f.Audio[offset:end], f.cancelled)
		f.timing.rendererFlow(f.ResponseID, f.ItemID, writeStarted.Sub(waitingSince), time.Since(writeStarted))
		if !written || err != nil {
			return offset, err
		}
		offset = end
		waitingSince = time.Now()
	}
	return offset, nil
}
