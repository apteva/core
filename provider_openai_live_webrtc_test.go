package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// Real DTLS/SRTP/SCTP on loopback; only the provider's model is mocked.
func TestOpenAILiveWebRTCG711MediaAndControl(t *testing.T) {
	for _, format := range []AudioFormat{AudioPCM16, AudioG711ULaw, AudioG711ALaw} {
		t.Run(string(format), func(t *testing.T) {
			var mu sync.Mutex
			var peers []*webrtc.PeerConnection
			defer func() {
				mu.Lock()
				defer mu.Unlock()
				for _, p := range peers {
					_ = p.Close()
				}
			}()
			incoming := make(chan []byte, 20)
			controls := make(chan string, 10)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Session   map[string]any             `json:"session"`
					Transport struct{ Type, SDP string } `json:"transport"`
				}
				if r.Method != "POST" || r.URL.Path != "/v1/live/sessions" || json.NewDecoder(r.Body).Decode(&req) != nil {
					w.WriteHeader(400)
					return
				}
				if _, ok := req.Session["audio"].(map[string]any)["format"]; ok {
					t.Error("WebRTC must negotiate format with SDP")
				}
				pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
				if err != nil {
					t.Error(err)
					w.WriteHeader(500)
					return
				}
				mu.Lock()
				peers = append(peers, pc)
				mu.Unlock()
				mime := webrtc.MimeTypePCMU
				clock := uint32(8000)
				channels := uint16(1)
				if format == AudioPCM16 {
					mime = webrtc.MimeTypeOpus
					clock = 48000
					channels = 2
				}
				if format == AudioG711ALaw {
					mime = webrtc.MimeTypePCMA
				}
				track, _ := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: mime, ClockRate: clock, Channels: channels}, "out", "model")
				sender, err := pc.AddTrack(track)
				if err != nil {
					t.Error(err)
					return
				}
				go func() {
					b := make([]byte, 1500)
					for {
						if _, _, e := sender.Read(b); e != nil {
							return
						}
					}
				}()
				pc.OnDataChannel(func(dc *webrtc.DataChannel) {
					if dc.Label() != "oai-events" {
						t.Error("wrong control channel")
					}
					dc.OnMessage(func(m webrtc.DataChannelMessage) {
						var v map[string]any
						_ = json.Unmarshal(m.Data, &v)
						controls <- v["type"].(string)
					})
					dc.OnOpen(func() { _ = dc.SendText(`{"type":"session.started","session":{"id":"test"}}`) })
				})
				pc.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
					go func() {
						for {
							p, _, e := remote.ReadRTP()
							if e != nil {
								return
							}
							incoming <- append([]byte(nil), p.Payload...)
							for i := 0; i < 80; i++ {
								_ = track.WriteSample(media.Sample{Data: p.Payload, Duration: 20 * time.Millisecond})
							}
						}
					}()
				})
				if e := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: req.Transport.SDP}); e != nil {
					t.Error(e)
					return
				}
				answer, _ := pc.CreateAnswer(nil)
				g := webrtc.GatheringCompletePromise(pc)
				_ = pc.SetLocalDescription(answer)
				<-g
				w.WriteHeader(201)
				_ = json.NewEncoder(w).Encode(map[string]any{"session": map[string]string{"id": "test"}, "transport": map[string]string{"sdp": pc.LocalDescription().SDP}})
			}))
			defer server.Close()
			p := NewOpenAIRealtimeProvider("fixture-key")
			p.endpoint = strings.Replace(server.URL, "http", "ws", 1) + "/v1/realtime"
			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
			defer cancel()
			s, e := p.Open(ctx, RealtimeSessionOpts{Model: "gpt-live-1", OutputConfig: RealtimeOutputConfig{Media: RealtimeMediaConfig{Transport: "webrtc", Format: format}}})
			if e != nil {
				t.Fatal(e)
			}
			defer s.Close()
			payload := bytes.Repeat([]byte{0x19}, 160)
			if format == AudioPCM16 {
				payload = make([]byte, 960)
			}
			if e = s.SendAudio(payload); e != nil {
				t.Fatal(e)
			}
			select {
			case got := <-incoming:
				if format != AudioPCM16 && !bytes.Equal(got, payload) {
					t.Fatal("native input changed")
				}
			case <-ctx.Done():
				t.Fatal("no SRTP input")
			}
			receivedBytes := 0
			for {
				select {
				case ev := <-s.Events():
					if ev.Type == RealtimeEventError {
						t.Fatal(ev)
					}
					if len(ev.Audio) > 0 {
						if format != AudioPCM16 && !bytes.Equal(ev.Audio, payload) {
							t.Fatal("native output changed")
						}
						receivedBytes += len(ev.Audio)
						if receivedBytes == 80*len(payload) {
							goto received
						}
					}
				case <-ctx.Done():
					t.Fatal("no decoded output")
				}
			}
		received:
			if e = s.(RealtimeGracefulSession).BeginClose(); e != nil {
				t.Fatal(e)
			}
			select {
			case kind := <-controls:
				if kind != "session.close" {
					t.Fatalf("unexpected control %s (audio/session.start must not use SCTP)", kind)
				}
			case <-ctx.Done():
				t.Fatal("no close control")
			}
		})
	}
}

func TestRealtimeMediaContractAndDurations(t *testing.T) {
	for _, f := range []AudioFormat{AudioPCM16, AudioG711ULaw, AudioG711ALaw} {
		m := RealtimeMediaConfig{Format: f}
		rate := mediaBytesPerSecond(f)
		rt, _ := completionFixture(t)
		rt.opts.OutputConfig.Media = m
		rt.configureMedia()
		if rt.input.duration(rate/50) != 20*time.Millisecond || rt.outputRate() != rate {
			t.Fatal("duration changed", f)
		}
		input := make([]byte, rate/50)
		if f == AudioG711ULaw {
			for i := range input {
				input[i] = 255
			}
		}
		if f == AudioG711ALaw {
			for i := range input {
				input[i] = 0xd5
			}
		}
		if mediaVoiced(input, f) {
			t.Fatal("silence flagged as speech", f)
		}
		rt.input.push(input, time.Now())
		if rt.input.backlog() != 20*time.Millisecond {
			t.Fatal("queue accounting", f)
		}
		if _, e := buildOpenAILiveStart(RealtimeSessionOpts{Model: "gpt-live-1", AudioInFmt: f, AudioOutFmt: f, AudioInRate: m.rate(), AudioOutRate: m.rate(), OutputConfig: RealtimeOutputConfig{Media: m}}, "brise"); e != nil {
			t.Fatal(e)
		}
	}
	if _, e := (RealtimeMediaConfig{Transport: "invalid"}).normalized(); e == nil {
		t.Fatal("unknown transport accepted")
	}
	if e := NewOpenAIRealtimeProvider("fixture").ValidateRealtimeMedia("gpt-realtime", RealtimeMediaConfig{Transport: "webrtc"}); e == nil {
		t.Fatal("unsupported explicit option accepted")
	}
}

func TestRealtimeNativeOpusRoundTrip(t *testing.T) {
	c, e := newRealtimeOpus()
	if e != nil {
		t.Skip(e)
	}
	defer c.Close()
	p, e := c.Encode(make([]byte, 960))
	if e != nil {
		t.Fatal(e)
	}
	decoded, e := c.Decode(p)
	if e != nil || len(decoded) != 960 {
		t.Fatalf("roundtrip samples %d: %v", len(decoded), e)
	}
	c.Close()
	if _, e = c.Encode(make([]byte, 960)); e == nil {
		t.Fatal("closed codec accepted input")
	}
}
