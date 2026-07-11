package core

import (
	"context"
	"crypto/rand"
	"encoding/hex"
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
	audioIn  chan []byte // bridge writes here (caller → model)
	audioOut chan []byte // bridge reads here (model → caller)
	created  time.Time
}

var (
	audioBridgeMu    sync.Mutex
	audioBridgeByTok = map[string]*audioBridgeRegistration{}
	audioBridgeTTL   = 5 * time.Minute
)

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
func registerAudioBridge(threadID string, audioIn, audioOut chan []byte) string {
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
		created:  now,
	}
	return tok
}

// unregisterAudioBridge removes any registrations bound to threadID.
// Used when a realtime spawn fails post-token-issuance, or when the
// thread is killed before a bridge connected.
func unregisterAudioBridge(threadID string) {
	audioBridgeMu.Lock()
	defer audioBridgeMu.Unlock()
	for tok, reg := range audioBridgeByTok {
		if reg.threadID == threadID {
			delete(audioBridgeByTok, tok)
		}
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

	// Run reader (WS → audioIn) and writer (audioOut → WS) in
	// goroutines. Either side dying closes the WS, which unblocks the
	// other.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

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
			if header.OpCode != ws.OpBinary || len(data) == 0 {
				continue
			}
			select {
			case reg.audioIn <- data:
			case <-ctx.Done():
				return
			default:
				// Thread's audio loop is slow — drop the frame
				// rather than backpressure the network. Matches the
				// realtime thread's own drop-on-saturation policy.
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case pcm, ok := <-reg.audioOut:
			if !ok {
				return
			}
			_ = conn.SetWriteDeadline(time.Now().Add(15 * time.Second))
			if err := wsutil.WriteServerBinary(conn, pcm); err != nil {
				logMsg("REALTIME-AUDIO", fmt.Sprintf("write end for thread=%s: %v", thread, err))
				return
			}
		}
	}
}
