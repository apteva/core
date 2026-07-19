package core

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

const maxRealtimeAudioFrameBytes = 1 << 20

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
	cancel context.CancelFunc
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
		live.cancel()
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

// realtimeAudioHandler is the HTTP handler for GET /realtime/audio.
// Performs the WebSocket upgrade, resolves the token, then runs two
// goroutines that shuttle binary frames between the WS and the
// thread's audio channels. PCM16 over the wire — no transcoding
// happens here; that's the bridge's (caller's) job.
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
	defer conn.Close()
	logMsg("REALTIME-AUDIO", fmt.Sprintf("bridge connected for thread=%s", thread))
	if a.thinker != nil && a.thinker.threads != nil {
		a.thinker.threads.realtimeBridgeConnected(thread)
	}
	if a.thinker != nil && a.thinker.telemetry != nil {
		a.thinker.telemetry.Emit("realtime.bridge_connected", thread, map[string]any{
			"encoding": "pcm16", "sample_rate": 24000, "channels": 1,
		})
	}

	// Run reader (WS → audioIn) and writer (audioOut → WS) in
	// goroutines. Either side dying closes the WS, which unblocks the
	// other.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	live := &audioBridgeConnection{cancel: cancel}
	audioBridgeMu.Lock()
	audioBridgeLive[thread] = live
	audioBridgeMu.Unlock()
	defer func() {
		audioBridgeMu.Lock()
		for id, connected := range audioBridgeLive {
			if connected == live {
				delete(audioBridgeLive, id)
			}
		}
		audioBridgeMu.Unlock()
		if a.thinker != nil && a.thinker.telemetry != nil {
			a.thinker.telemetry.Emit("realtime.bridge_disconnected", thread, map[string]any{})
		}
		if a.thinker != nil && a.thinker.threads != nil {
			a.thinker.threads.realtimeBridgeDisconnected(thread)
		}
	}()

	go func() {
		defer cancel()
		reader := wsutil.Reader{
			Source:         conn,
			State:          ws.StateServerSide,
			CheckUTF8:      true,
			MaxFrameSize:   maxRealtimeAudioFrameBytes,
			OnIntermediate: wsutil.ControlFrameHandler(conn, ws.StateServerSide),
		}
		for {
			_ = conn.SetReadDeadline(time.Now().Add(2 * time.Minute))
			header, err := reader.NextFrame()
			if err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("read end for thread=%s: %v", thread, err))
				return
			}
			if header.OpCode.IsControl() {
				if err := wsutil.ControlFrameHandler(conn, ws.StateServerSide)(header, &reader); err != nil {
					return
				}
				continue
			}
			data, err := io.ReadAll(io.LimitReader(&reader, maxRealtimeAudioFrameBytes+1))
			if err != nil || len(data) > maxRealtimeAudioFrameBytes {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("oversized/invalid frame for thread=%s: bytes=%d err=%v", thread, len(data), err))
				return
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
				return
			case <-timer.C:
				if a.thinker != nil && a.thinker.telemetry != nil {
					a.thinker.telemetry.Emit("realtime.audio_overflow", thread, map[string]any{
						"direction": "input", "bytes": len(data), "reason": "thread_backpressure",
					})
				}
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case frame, ok := <-reg.audioOut:
			if !ok {
				return
			}
			_ = conn.SetWriteDeadline(time.Now().Add(15 * time.Second))
			metadata, _ := json.Marshal(map[string]any{
				"type": "audio.frame", "response_id": frame.ResponseID,
				"item_id": frame.ItemID, "audio_end_ms": frame.AudioEndMS,
			})
			if err := wsutil.WriteServerMessage(conn, ws.OpText, metadata); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("metadata write end for thread=%s: %v", thread, err))
				return
			}
			if err := wsutil.WriteServerBinary(conn, frame.Audio); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("write end for thread=%s: %v", thread, err))
				return
			}
		case control, ok := <-reg.control:
			if !ok {
				reg.control = nil
				continue
			}
			_ = conn.SetWriteDeadline(time.Now().Add(15 * time.Second))
			payload := []byte(fmt.Sprintf(`{"type":%q}`, control))
			if err := wsutil.WriteServerMessage(conn, ws.OpText, payload); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("control write end for thread=%s: %v", thread, err))
				return
			}
		}
	}
}
