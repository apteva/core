package core

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// Paid opt-in comparison: identical recorded phrase/history, continuously paced
// input and acoustic onset. No phone numbers, tools or application mutations.
// REALTIME_MEDIA_REPLAY_PCM contains mono 24kHz PCM16 of the caller phrase.
func TestRealtimeMediaTransportLiveComparison(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_REALTIME_MEDIA_COMPARE") != "1" {
		t.Skip("explicit paid comparison only")
	}
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Fatal("OPENAI_API_KEY required")
	}
	phrase, e := os.ReadFile(os.Getenv("REALTIME_MEDIA_REPLAY_PCM"))
	if e != nil {
		t.Fatal(e)
	}
	formats := []AudioFormat{AudioPCM16, AudioG711ULaw}
	for _, transport := range []string{"websocket", "webrtc"} {
		for _, format := range formats {
			t.Run(transport+"/"+string(format), func(t *testing.T) {
				m := RealtimeMediaConfig{Transport: transport, Format: format}
				rate := mediaBytesPerSecond(format)
				pcm := make([]byte, 2*48000+len(phrase)+5*48000)
				copy(pcm[2*48000:], phrase)
				input := pcm
				if format == AudioG711ULaw {
					input = make([]byte, len(pcm)/6)
					for i := range input { // same 8k speech source, decimated only for this controlled fixture
						sample := int16(binary.LittleEndian.Uint16(pcm[6*i:]))
						input[i] = replayULaw(sample)
					}
				}
				p := NewOpenAIRealtimeProvider(key)
				ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
				defer cancel()
				start := time.Now()
				s, e := p.Open(ctx, RealtimeSessionOpts{Model: "gpt-live-1", Voice: "brise", ConversationInstructions: "Vous êtes un assistant vocal francophone. Répondez directement aux questions simples avec une phrase courte. Ne saluez pas de nouveau. Attendez la question suivante.", History: []Message{{Role: "user", Content: "Quelle est la capitale de la France ?"}, {Role: "assistant", Content: "Paris."}}, OutputConfig: RealtimeOutputConfig{Media: m}})
				if e != nil {
					t.Fatal(e)
				}
				defer s.Close()
				setup := time.Since(start)
				var mu sync.Mutex
				var lastInput, firstOutput time.Time
				var output []byte
				var transcript strings.Builder
				var events []map[string]any
				done := make(chan struct{})
				go func() {
					defer close(done)
					for ev := range s.Events() {
						mu.Lock()
						if ev.Type == RealtimeEventError {
							events = append(events, map[string]any{"error": ev.Err.Error()})
						}
						if len(ev.Audio) > 0 {
							output = append(output, ev.Audio...)
							if replayRMS(ev.Audio, format) > 150 && !lastInput.IsZero() && firstOutput.IsZero() {
								firstOutput = time.Now()
							}
						}
						if ev.Transcript != "" {
							transcript.WriteString(ev.Transcript)
						}
						mu.Unlock()
					}
				}()
				origin := time.Now()
				for off := 0; off < len(input); off += rate / 50 {
					delay := time.Until(origin.Add(time.Duration(off) * time.Second / time.Duration(rate)))
					if delay > 0 {
						select {
						case <-ctx.Done():
							t.Fatal(ctx.Err())
						case <-time.After(delay):
						}
					}
					frame := input[off:min(off+rate/50, len(input))]
					mu.Lock()
					if replayRMS(frame, format) > 150 {
						lastInput = time.Now()
					}
					mu.Unlock()
					if e = s.SendAudio(frame); e != nil {
						t.Fatal(e)
					}
				}
				_ = s.(RealtimeGracefulSession).BeginClose()
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					_ = s.Close()
					<-done
				}
				mu.Lock()
				defer mu.Unlock()
				report := map[string]any{"transport": transport, "format": format, "setup_ms": float64(setup) / float64(time.Millisecond), "reply_ms": float64(firstOutput.Sub(lastInput)) / float64(time.Millisecond), "transcript": transcript.String(), "events": events, "output_bytes": len(output)}
				data, _ := json.MarshalIndent(report, "", "  ")
				t.Log(string(data))
				if dir := os.Getenv("REALTIME_MEDIA_REPORT_DIR"); dir != "" {
					_ = os.MkdirAll(dir, 0700)
					base := filepath.Join(dir, transport+"-"+string(format))
					_ = os.WriteFile(base+".json", data, 0600)
					_ = os.WriteFile(base+".audio", output, 0600)
				}
				if firstOutput.IsZero() {
					t.Error("no audible reply")
				}
			})
		}
	}
}
func replayRMS(b []byte, f AudioFormat) float64 {
	var energy float64
	n := len(b)
	if f == AudioPCM16 {
		n /= 2
	}
	if n == 0 {
		return 0
	}
	for i := 0; i < n; i++ {
		var v int
		if f == AudioPCM16 {
			v = int(int16(binary.LittleEndian.Uint16(b[2*i:])))
		} else {
			v = mediaSample(b[i], f)
		}
		energy += float64(v) * float64(v)
	}
	return math.Sqrt(energy / float64(n))
}
func replayULaw(v int16) byte {
	n := int(v)
	sign := byte(0)
	if n < 0 {
		sign = 0x80
		n = -n
	}
	if n > 32635 {
		n = 32635
	}
	n += 132
	exponent := byte(7)
	for mask := 0x4000; exponent > 0 && n&mask == 0; mask >>= 1 {
		exponent--
	}
	return ^(sign | exponent<<4 | byte(n>>uint(exponent+3))&15)
}
