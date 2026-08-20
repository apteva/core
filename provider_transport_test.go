package core

import (
	"crypto/tls"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// trustTestServer points a transport at a httptest server's CA without
// disturbing any other field, so the behaviour under test stays the
// production configuration.
func trustTestServer(tr *http.Transport, srv *httptest.Server) {
	srvTLS := srv.Client().Transport.(*http.Transport).TLSClientConfig
	if tr.TLSClientConfig == nil {
		tr.TLSClientConfig = &tls.Config{}
	}
	tr.TLSClientConfig.RootCAs = srvTLS.RootCAs
}

// net/http disables automatic HTTP/2 whenever a custom DialContext is set,
// which this transport does for the dial timeout. Without ForceAttemptHTTP2
// every provider call is HTTP/1.1, where concurrent requests cannot share a
// connection at all — the pool limits below cannot compensate for that.
func TestLLMTransportNegotiatesHTTP2(t *testing.T) {
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	srv.EnableHTTP2 = true
	srv.StartTLS()
	defer srv.Close()

	tr := newLLMTransport()
	trustTestServer(tr, srv)
	defer tr.CloseIdleConnections()

	resp, err := (&http.Client{Transport: tr}).Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	if resp.Proto != "HTTP/2.0" {
		t.Errorf("provider transport negotiated %s, want HTTP/2.0 — check ForceAttemptHTTP2, "+
			"which a custom DialContext otherwise suppresses", resp.Proto)
	}
}

// countingTLSServer returns an h2-or-h1 TLS server that counts accepted TCP
// connections, plus a client transport that trusts it.
func countingTLSServer(t *testing.T, h2 bool, hold time.Duration) (*httptest.Server, *http.Transport, *int64) {
	t.Helper()
	var conns int64
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(hold) // keep requests overlapping
		w.Write([]byte("ok"))
	}))
	srv.EnableHTTP2 = h2
	srv.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			atomic.AddInt64(&conns, 1)
		}
	}
	srv.StartTLS()
	t.Cleanup(srv.Close)

	tr := newLLMTransport()
	trustTestServer(tr, srv)
	t.Cleanup(tr.CloseIdleConnections)
	return srv, tr, &conns
}

// concurrentGets fires n overlapping requests and waits for all of them.
func concurrentGets(client *http.Client, url string, n int) {
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := client.Get(url)
			if err != nil {
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}()
	}
	wg.Wait()
}

// The payoff of HTTP/2: once a connection exists, concurrent requests share it
// as streams. Over HTTP/1.1 a connection carries one request at a time, so N
// overlapping requests force N sockets no matter how large the idle pool is.
// Both halves run against the same transport and workload, so the only
// variable is the negotiated protocol.
//
// A cold burst is deliberately not measured: with an empty pool every request
// races to dial before the first connection is ready, so both protocols open
// ~N sockets. The connection has to be warm for the difference to appear.
func TestLLMTransportMultiplexesOverWarmConnection(t *testing.T) {
	const concurrent = 50

	measure := func(h2 bool) int64 {
		srv, tr, conns := countingTLSServer(t, h2, 20*time.Millisecond)
		client := &http.Client{Transport: tr}

		// Warm exactly one connection.
		resp, err := client.Get(srv.URL)
		if err != nil {
			t.Fatalf("warmup (h2=%v): %v", h2, err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if want := "HTTP/2.0"; h2 && resp.Proto != want {
			t.Fatalf("expected %s, got %s", want, resp.Proto)
		}
		warm := atomic.LoadInt64(conns)

		concurrentGets(client, srv.URL, concurrent)
		return atomic.LoadInt64(conns) - warm
	}

	overH2 := measure(true)
	overH1 := measure(false)

	t.Logf("%d concurrent requests on a warm connection: h2 opened %d new socket(s), h1.1 opened %d",
		concurrent, overH2, overH1)

	if overH2 > 2 {
		t.Errorf("HTTP/2 opened %d new connections for %d concurrent requests; expected them to multiplex",
			overH2, concurrent)
	}
	if overH1 <= overH2 {
		t.Errorf("expected HTTP/1.1 to need far more sockets than HTTP/2 (h1=%d, h2=%d); "+
			"if these are equal the h2 upgrade is not taking effect", overH1, overH2)
	}
}

// Pool reuse over HTTP/1.1, which is what a provider that does not offer h2
// falls back to. With Go's default MaxIdleConnsPerHost of 2, a second burst
// reopens nearly every connection.
func TestLLMTransportReusesIdleConnectionsAcrossBursts(t *testing.T) {
	var conns int64
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	srv.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		if state == http.StateNew {
			atomic.AddInt64(&conns, 1)
		}
	}
	srv.Start() // plain HTTP — HTTP/1.1
	defer srv.Close()

	tr := newLLMTransport()
	defer tr.CloseIdleConnections()
	client := &http.Client{Transport: tr}

	const burst = 20
	fire := func() {
		var wg sync.WaitGroup
		for i := 0; i < burst; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				resp, err := client.Get(srv.URL)
				if err != nil {
					return
				}
				// Body must be drained AND closed for the connection to
				// return to the idle pool.
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
			}()
		}
		wg.Wait()
	}

	fire()
	afterFirst := atomic.LoadInt64(&conns)
	fire()
	afterSecond := atomic.LoadInt64(&conns)

	newInSecond := afterSecond - afterFirst
	// burst (20) is well under defaultLLMMaxIdleConnsPerHost (64), so every
	// connection from the first burst should still be pooled.
	if newInSecond > 2 {
		t.Errorf("second burst opened %d new connections (total %d); idle pool of %d should have covered it",
			newInSecond, afterSecond, llmMaxIdleConnsPerHost())
	}
	t.Logf("burst1=%d conns, burst2 opened %d new", afterFirst, newInSecond)
}

// MaxConnsPerHost blocks rather than erroring, which is what makes it a
// backpressure knob and an fd-exhaustion backstop.
func TestLLMTransportMaxConnsPerHostCapsConcurrency(t *testing.T) {
	t.Setenv("APTEVA_LLM_MAX_CONNS_PER_HOST", "4")

	var inFlight, peak int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cur := atomic.AddInt64(&inFlight, 1)
		for {
			old := atomic.LoadInt64(&peak)
			if cur <= old || atomic.CompareAndSwapInt64(&peak, old, cur) {
				break
			}
		}
		time.Sleep(30 * time.Millisecond)
		atomic.AddInt64(&inFlight, -1)
		w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tr := newLLMTransport()
	defer tr.CloseIdleConnections()
	if tr.MaxConnsPerHost != 4 {
		t.Fatalf("env override not applied: MaxConnsPerHost=%d", tr.MaxConnsPerHost)
	}
	client := &http.Client{Transport: tr}

	var wg sync.WaitGroup
	for i := 0; i < 25; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := client.Get(srv.URL)
			if err != nil {
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}()
	}
	wg.Wait()

	if got := atomic.LoadInt64(&peak); got > 4 {
		t.Errorf("peak concurrent server-side requests %d exceeds MaxConnsPerHost=4", got)
	}
}

func TestLLMTransportPoolDefaults(t *testing.T) {
	tr := newLLMTransport()
	if tr.MaxIdleConnsPerHost != defaultLLMMaxIdleConnsPerHost {
		t.Errorf("MaxIdleConnsPerHost=%d, want %d (unset means http.DefaultMaxIdleConnsPerHost=%d)",
			tr.MaxIdleConnsPerHost, defaultLLMMaxIdleConnsPerHost, http.DefaultMaxIdleConnsPerHost)
	}
	if tr.MaxIdleConns != defaultLLMMaxIdleConns {
		t.Errorf("MaxIdleConns=%d, want %d", tr.MaxIdleConns, defaultLLMMaxIdleConns)
	}
	// A per-host idle limit above the global total is silently capped by the
	// total, which would make the per-host setting a lie.
	if tr.MaxIdleConns < tr.MaxIdleConnsPerHost {
		t.Errorf("MaxIdleConns (%d) must be >= MaxIdleConnsPerHost (%d) or the global total wins",
			tr.MaxIdleConns, tr.MaxIdleConnsPerHost)
	}
	if tr.MaxConnsPerHost != defaultLLMMaxConnsPerHost {
		t.Errorf("MaxConnsPerHost=%d, want %d", tr.MaxConnsPerHost, defaultLLMMaxConnsPerHost)
	}
	// Zero means "retain idle connections forever" on a hand-built Transport.
	if tr.IdleConnTimeout == 0 {
		t.Error("IdleConnTimeout must be set; zero retains idle connections indefinitely")
	}
	if !tr.ForceAttemptHTTP2 {
		t.Error("ForceAttemptHTTP2 must be set or the custom DialContext suppresses h2")
	}
}

func TestLLMTransportEnvOverrides(t *testing.T) {
	t.Setenv("APTEVA_LLM_MAX_IDLE_CONNS_PER_HOST", "8")
	t.Setenv("APTEVA_LLM_MAX_IDLE_CONNS", "16")
	t.Setenv("APTEVA_LLM_MAX_CONNS_PER_HOST", "32")

	tr := newLLMTransport()
	if tr.MaxIdleConnsPerHost != 8 || tr.MaxIdleConns != 16 || tr.MaxConnsPerHost != 32 {
		t.Errorf("env overrides not applied: idlePerHost=%d idle=%d connsPerHost=%d",
			tr.MaxIdleConnsPerHost, tr.MaxIdleConns, tr.MaxConnsPerHost)
	}
}

// 0 is meaningful for MaxConnsPerHost (unlimited) but not for the idle limits,
// where it would mean "use Go's default of 2" — the bug being fixed.
func TestLLMTransportZeroAndGarbageEnvHandling(t *testing.T) {
	t.Setenv("APTEVA_LLM_MAX_CONNS_PER_HOST", "0")
	if got := llmMaxConnsPerHost(); got != 0 {
		t.Errorf("explicit 0 must restore unlimited conns per host, got %d", got)
	}

	for _, bad := range []string{"", "abc", "-1", "0"} {
		t.Setenv("APTEVA_LLM_MAX_IDLE_CONNS_PER_HOST", bad)
		if got := llmMaxIdleConnsPerHost(); got != defaultLLMMaxIdleConnsPerHost {
			t.Errorf("idle-per-host %q should fall back to %d, got %d",
				bad, defaultLLMMaxIdleConnsPerHost, got)
		}
	}

	t.Setenv("APTEVA_LLM_MAX_CONNS_PER_HOST", "garbage")
	if got := llmMaxConnsPerHost(); got != defaultLLMMaxConnsPerHost {
		t.Errorf("unparseable conns-per-host should fall back to %d, got %d",
			defaultLLMMaxConnsPerHost, got)
	}
}

// The package-level client every provider shares must carry the tuned
// transport, not a bare or default one.
func TestSharedLLMClientUsesTunedTransport(t *testing.T) {
	tr, ok := llmHTTPClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("llmHTTPClient.Transport is %T, want *http.Transport", llmHTTPClient.Transport)
	}
	if !tr.ForceAttemptHTTP2 || tr.MaxIdleConnsPerHost <= http.DefaultMaxIdleConnsPerHost {
		t.Errorf("shared client is untuned: forceH2=%v idlePerHost=%d",
			tr.ForceAttemptHTTP2, tr.MaxIdleConnsPerHost)
	}
	// No overall client timeout: streaming responses legitimately run for
	// minutes, and ResponseHeaderTimeout is what catches a dead provider.
	if llmHTTPClient.Timeout != 0 {
		t.Errorf("shared client must not set an overall timeout, got %s", llmHTTPClient.Timeout)
	}
}
