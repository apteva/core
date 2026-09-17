package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

func googleRecoveryMessage(t *testing.T, s *googleRealtimeSession, raw string) {
	t.Helper()
	var m googleLiveServerMessage
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatal(err)
	}
	s.translate(&m)
}

func TestGoogleRealtimeGoAwayAndAudioStreamEnd(t *testing.T) {
	s := newGoogleRealtimeTestSession()
	googleRecoveryMessage(t, s, `{"goAway":{"timeLeft":"12.5s"}}`)
	e := <-s.Events()
	if e.Type != RealtimeEventSessionExpiring || e.TimeLeft != 12500*time.Millisecond {
		t.Fatalf("GoAway = %#v", e)
	}
	googleRecoveryMessage(t, s, `{"goAway":{"timeLeft":"invalid"}}`)
	if e := <-s.Events(); e.Type != RealtimeEventError {
		t.Fatal("invalid deadline accepted")
	}
	if err := s.EndAudioInput(); err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	_ = json.Unmarshal((<-s.outbox).data, &out)
	if out["realtimeInput"].(map[string]any)["audioStreamEnd"] != true {
		t.Fatal("stream end missing")
	}
	if err := s.SendAudio([]byte{0, 0}); err != nil {
		t.Fatal("audio did not restart", err)
	}
}

func TestGoogleRealtimeCloseMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, _, _, err := ws.UpgradeHTTP(r, w)
		if err != nil {
			return
		}
		defer conn.Close()
		_, _, _ = wsutil.ReadClientData(conn)
		_ = wsutil.WriteServerMessage(conn, ws.OpText, []byte(`{"setupComplete":{}}`))
		_ = wsutil.WriteServerMessage(conn, ws.OpClose, ws.NewCloseFrameBody(ws.StatusPolicyViolation, "The operation was aborted."))
	}))
	defer server.Close()
	p := NewGoogleRealtimeProvider("test-key")
	p.endpoint = "ws" + strings.TrimPrefix(server.URL, "http")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	s, err := p.Open(ctx, RealtimeSessionOpts{Model: "test-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for e := range s.Events() {
		if e.Type == RealtimeEventSessionEnded {
			if e.CloseCode != 1008 || e.CloseReason != "The operation was aborted." {
				t.Fatalf("close metadata=%#v", e)
			}
			return
		}
	}
	t.Fatal("missing close metadata")
}

func TestGoogleRealtimeHistoryIsOptInAndResponseIDsAreSessionScoped(t *testing.T) {
	data, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "test-model", RestoreHistory: true}, "Kore")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(data, &m)
	if m["setup"].(map[string]any)["historyConfig"] == nil {
		t.Fatal("restoration setup missing")
	}
	s := newGoogleRealtimeTestSession()
	if err := s.RestoreConversation(nil); err != nil || len(s.outbox) != 0 {
		t.Fatal("empty history triggered input")
	}
	if err := s.RestoreConversation([]Message{{Role: "user", Content: "hello"}}); err == nil {
		t.Fatal("unconfigured history could trigger generation")
	}
	a, b := newGoogleRealtimeTestSession(), newGoogleRealtimeTestSession()
	a.sessionID = 1
	b.sessionID = 2
	if a.nextResponseID() == b.nextResponseID() {
		t.Fatal("response IDs collide across sockets")
	}
}
