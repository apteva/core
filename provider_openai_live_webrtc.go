package core

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gobwas/ws"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"io"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

type liveMediaMessage struct {
	data, audio []byte
	err         error
}

// One session owns media, event delivery and cleanup. No provider keys or SDP
// are exposed to the renderer. Explicit WebRTC never silently falls back.
func (p *OpenAIRealtimeProvider) openLiveRTCSession(ctx context.Context, opts RealtimeSessionOpts) (*openaiLiveSession, error) {
	raw, err := buildOpenAILiveStart(opts, p.DefaultVoice())
	if err != nil {
		return nil, err
	}
	var start map[string]any
	if err = json.Unmarshal(raw, &start); err != nil {
		return nil, err
	}
	config := start["session"].(map[string]any)
	delete(config["audio"].(map[string]any), "format")
	m := opts.OutputConfig.Media
	codec := webrtc.RTPCodecParameters{RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2, SDPFmtpLine: "minptime=10;useinbandfec=1;stereo=0;sprop-stereo=0"}, PayloadType: 111}
	var opus *realtimeOpus
	switch m.format() {
	case AudioPCM16:
		opus, err = newRealtimeOpus()
		if err != nil {
			return nil, err
		}
	case AudioG711ULaw:
		codec = webrtc.RTPCodecParameters{RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: 1}, PayloadType: 0}
	case AudioG711ALaw:
		codec = webrtc.RTPCodecParameters{RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMA, ClockRate: 8000, Channels: 1}, PayloadType: 8}
	}
	engine := &webrtc.MediaEngine{}
	if err = engine.RegisterCodec(codec, webrtc.RTPCodecTypeAudio); err != nil {
		if opus != nil {
			opus.Close()
		}
		return nil, err
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(engine)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		if opus != nil {
			opus.Close()
		}
		return nil, err
	}
	msgs := make(chan liveMediaMessage, 256)
	s := &openaiLiveSession{media: m, rtcMessages: msgs, ready: make(chan error, 1), instructions: opts.Instructions, tools: opts.Tools, openaiRealtimeSession: &openaiRealtimeSession{
		events: make(chan RealtimeEvent, realtimeEventBuffer), outbox: make(chan realtimeOutboundFrame, realtimeOutboxBuffer), audioOutbox: make(chan realtimeOutboundFrame, 8), done: make(chan struct{}), writeTimeout: 250 * time.Millisecond, providerName: "openai-realtime", timing: opts.timing,
		settings: map[string]any{"source": "session_started", "model": opts.Model, "voice": opts.Voice, "transport": "webrtc", "audio_format": m.format(), "sample_rate": m.rate(), "media_codec": codec.MimeType, "speech_guard": opts.OutputConfig.SpeechGuard.continuousPolicy()},
	}}
	s.closeTransport = func() error {
		e := pc.Close()
		if opus != nil {
			opus.Close()
		}
		return e
	}
	success := false
	defer func() {
		if !success {
			_ = s.Close()
		}
	}()
	s.guard.config = opts.OutputConfig.SpeechGuard.continuousPolicy()
	s.guard.format = m.format()
	s.execution = openaiLiveExecution{enabled: opts.ExecutionMode == "provider", notify: make(chan struct{}, 1), pending: map[string]bool{}, submitted: map[string]bool{}, inputHashes: map[[32]byte]bool{}}
	deliver := func(msg liveMediaMessage) bool {
		select {
		case <-s.done:
			return false
		case msgs <- msg:
			return true
		default:
			select {
			case s.ready <- fmt.Errorf("WebRTC event/media backlog exceeded"):
			default:
			}
			go s.Close()
			return false
		}
	}
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		if state == webrtc.PeerConnectionStateFailed {
			deliver(liveMediaMessage{err: fmt.Errorf("WebRTC transport failed")})
		}
	})
	track, err := webrtc.NewTrackLocalStaticSample(codec.RTPCodecCapability, "audio", "apteva")
	if err != nil {
		return nil, err
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		return nil, err
	}
	go func() {
		b := make([]byte, 1500)
		for {
			if _, _, err := sender.Read(b); err != nil {
				return
			}
		}
	}()
	pc.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if !strings.EqualFold(remote.Codec().MimeType, codec.MimeType) {
			deliver(liveMediaMessage{err: fmt.Errorf("unexpected negotiated WebRTC audio codec")})
			return
		}
		go func() {
			var order realtimeRTPOrder
			for {
				// Read/decode in one consumer: a valid provider burst must not
				// overflow a second tiny handoff queue before it is scheduled.
				deadline := time.Time{}
				if !order.waiting.IsZero() {
					deadline = order.waiting.Add(realtimeRTPReorderWait)
				}
				if e := remote.SetReadDeadline(deadline); e != nil {
					deliver(liveMediaMessage{err: e})
					return
				}
				packet, _, e := remote.ReadRTP()
				var ready []*rtp.Packet
				if e != nil {
					select {
					case <-s.done:
						return
					default:
					}
					if errors.Is(e, os.ErrDeadlineExceeded) {
						ready = order.pop(time.Now(), false)
					} else {
						deliver(liveMediaMessage{err: fmt.Errorf("WebRTC media read ended: %w", e)})
						return
					}
				} else {
					ready = order.push(packet, time.Now())
				}

				for _, packet := range ready {
					audio := packet.Payload
					if opus != nil {
						var e error
						audio, e = opus.Decode(packet.Payload)
						if e != nil {
							deliver(liveMediaMessage{err: e})
							return
						}
					} else {
						audio = append([]byte(nil), audio...)
					}
					if !deliver(liveMediaMessage{audio: audio}) {
						return
					}
				}
			}
		}()
	})
	dc, err := pc.CreateDataChannel("oai-events", nil)
	if err != nil {
		return nil, err
	}
	dc.OnMessage(func(msg webrtc.DataChannelMessage) {
		if len(msg.Data) > 1<<20 {
			deliver(liveMediaMessage{err: fmt.Errorf("WebRTC event too large")})
			return
		}
		deliver(liveMediaMessage{data: append([]byte(nil), msg.Data...)})
	})
	dc.OnClose(func() {
		select {
		case <-s.done:
		default:
			deliver(liveMediaMessage{err: fmt.Errorf("WebRTC event channel closed")})
		}
	})
	var mediaPending []byte
	s.writeTransport = func(frame realtimeOutboundFrame, budget time.Duration) (result error) {
		if frame.op == ws.OpPing {
			return nil
		}
		var msg struct {
			Type  string `json:"type"`
			Audio string `json:"audio"`
		}
		if err := json.Unmarshal(frame.data, &msg); err != nil {
			return err
		}
		// Closing the peer interrupts a blocked SCTP/media write. No goroutine is
		// abandoned on a timeout and subsequent media cannot slip into a new session.
		var completed atomic.Bool
		timer := time.AfterFunc(budget, func() {
			if completed.CompareAndSwap(false, true) {
				_ = pc.Close()
			}
		})
		defer func() {
			if !completed.CompareAndSwap(false, true) {
				result = context.DeadlineExceeded
			}
			timer.Stop()
		}()
		if msg.Type == "session.input_audio.append" {
			b, err := base64.StdEncoding.DecodeString(msg.Audio)
			if err != nil {
				return err
			}
			mediaPending = append(mediaPending, b...)
			size := mediaBytesPerSecond(m.format()) / 50
			for len(mediaPending) >= size {
				frame := mediaPending[:size]
				if opus != nil {
					frame, err = opus.Encode(frame)
					if err != nil {
						return err
					}
				}
				if err = track.WriteSample(media.Sample{Data: frame, Duration: 20 * time.Millisecond}); err != nil {
					return err
				}
				mediaPending = mediaPending[size:]
			}
			return nil
		}
		if dc.ReadyState() != webrtc.DataChannelStateOpen || dc.BufferedAmount() > 65536 {
			return fmt.Errorf("WebRTC control channel unavailable or congested")
		}
		return dc.SendText(string(frame.data))
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return nil, err
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err = pc.SetLocalDescription(offer); err != nil {
		return nil, err
	}
	setup, stop := context.WithTimeout(ctx, 20*time.Second)
	defer stop()
	select {
	case <-gathered:
	case <-setup.Done():
		return nil, setup.Err()
	}
	body, _ := json.Marshal(map[string]any{"session": config, "transport": map[string]any{"type": "webrtc", "sdp": pc.LocalDescription().SDP}})
	endpoint := strings.TrimSuffix(strings.TrimSuffix(p.endpoint, "/realtime"), "/") + "/live/sessions"
	endpoint = strings.Replace(endpoint, "wss://", "https://", 1)
	endpoint = strings.Replace(endpoint, "ws://", "http://", 1)
	req, err := http.NewRequestWithContext(setup, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")
	if opts.SafetyIdentifier != "" {
		req.Header.Set("OpenAI-Safety-Identifier", opts.SafetyIdentifier)
	}
	// Session creation is never automatically retried after an ambiguous failure.
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Live WebRTC session creation failed: %w", err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusCreated && response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Live WebRTC creation HTTP %d", response.StatusCode)
	}
	var answer struct {
		Session struct {
			ID string `json:"id"`
		} `json:"session"`
		Transport struct {
			SDP string `json:"sdp"`
		} `json:"transport"`
	}
	if err = json.Unmarshal(data, &answer); err != nil || answer.Transport.SDP == "" {
		return nil, fmt.Errorf("invalid Live WebRTC session answer")
	}
	if err = pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer.Transport.SDP}); err != nil {
		return nil, err
	}
	s.lifecycle.start(s.events, s.readLiveLoop, s.writeLoop, s.flushLiveSpeech)
	go func() {
		select {
		case <-ctx.Done():
			_ = s.Close()
		case <-s.done:
		}
	}()
	select {
	case err = <-s.ready:
		if err != nil {
			return nil, err
		}
	case <-setup.Done():
		return nil, setup.Err()
	case <-s.done:
		return nil, fmt.Errorf("WebRTC session closed during initialization")
	}
	success = true
	return s, nil
}
