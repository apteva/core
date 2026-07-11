package core

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func clearManagedCoreEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"AGENT_ID", "INSTANCE_ID", "SERVER_URL", "TELEMETRY_URL"} {
		t.Setenv(key, "")
	}
}

func TestManagedCoreRequiresAPIKey(t *testing.T) {
	t.Setenv("AGENT_ID", "42")
	t.Setenv("APTEVA_API_KEY", "")
	if _, err := newCoreHTTPServer(&Thinker{}); err == nil {
		t.Fatal("managed core started without APTEVA_API_KEY")
	}
}

func TestCoreHTTPServerLimitsBodiesHeadersAndIdleConnections(t *testing.T) {
	clearManagedCoreEnv(t)
	t.Setenv("APTEVA_API_KEY", "secret")
	server, err := newCoreHTTPServer(&Thinker{})
	if err != nil {
		t.Fatal(err)
	}
	if server.ReadHeaderTimeout != 5*time.Second || server.IdleTimeout != 2*time.Minute || server.MaxHeaderBytes != 64<<10 {
		t.Fatalf("unexpected server limits: header=%s idle=%s max_header=%d", server.ReadHeaderTimeout, server.IdleTimeout, server.MaxHeaderBytes)
	}
	body := strings.NewReader(strings.Repeat("x", maxAPIRequestBytes+1))
	req := httptest.NewRequest(http.MethodPut, "/config", body)
	req.Header.Set("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	server.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusRequestEntityTooLarge)
	}
}

func TestCoreHTTPServerAuthenticatesStaticFallback(t *testing.T) {
	clearManagedCoreEnv(t)
	t.Setenv("APTEVA_API_KEY", "secret")
	server, err := newCoreHTTPServer(&Thinker{})
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	server.Handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/anything", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestStartAPIReturnsListenerError(t *testing.T) {
	clearManagedCoreEnv(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	if err := startAPI(&Thinker{}, listener.Addr().String()); err == nil {
		t.Fatal("startAPI hid listener bind failure")
	}
}
