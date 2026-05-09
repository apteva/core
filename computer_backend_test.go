package core

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	aptcomputer "github.com/apteva/computer"
	"github.com/apteva/core/pkg/computer"
)

// Live-Computer registry for emergency cleanup. Every cloud-backend
// session left open costs real money (Browserbase + Steel charge per
// session-minute, with non-trivial creation-time minimums). When a
// test goroutine panics in a sibling goroutine OR the user pkills
// the test process, normal `defer comp.Close()` doesn't fire and
// the cloud session leaks until its idle timeout — that can be
// 30+ minutes of billed time per leak.
//
// TestMain registers a SIGINT/SIGTERM handler that walks this
// registry and Close()s every live Computer before exiting. Tests
// register on creation, deregister on their own deferred Close.
// Best-effort: a SIGKILL still leaks (we can't catch that), and
// a panic-storm crash still leaks if it bypasses the runtime's
// signal forwarding. But Ctrl-C and `kill <pid>` (the most common
// test-kill paths during dev) now release sessions cleanly.
var (
	liveComputersMu sync.Mutex
	liveComputers   = map[computer.Computer]struct{}{}
)

func registerLiveComputer(c computer.Computer) {
	liveComputersMu.Lock()
	liveComputers[c] = struct{}{}
	liveComputersMu.Unlock()
}

func deregisterLiveComputer(c computer.Computer) {
	liveComputersMu.Lock()
	delete(liveComputers, c)
	liveComputersMu.Unlock()
}

// closeAllLiveComputers iterates the registry and Closes everything
// in parallel. Called from TestMain's signal handler. Returns once
// every Close() has either returned OR a 5s grace per-Computer has
// elapsed — we don't want a hung release blocking process exit
// indefinitely while the user waits for their Ctrl-C to take.
func closeAllLiveComputers() {
	liveComputersMu.Lock()
	all := make([]computer.Computer, 0, len(liveComputers))
	for c := range liveComputers {
		all = append(all, c)
	}
	liveComputersMu.Unlock()
	if len(all) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "[CLEANUP] closing %d live computer session(s) before exit...\n", len(all))
	var wg sync.WaitGroup
	for _, c := range all {
		c := c
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = c.Close()
		}()
	}
	wg.Wait()
}

// buildComputerFromEnv picks a Computer backend from TEST_BROWSER:
//
//	local | browserbase | steel | browser-engine   (default: local)
//
// Cloud backends skip the test when their credentials are missing so
// CI without those secrets stays green.
//
// Two opt-in flags extend the default config uniformly across all
// cloud backends (ignored on "local"):
//
//	TEST_BROWSER_PROXY=1       → managed residential proxy
//	TEST_BROWSER_SOLVE_CAPTCHA=1 (default on) → managed CAPTCHA solver
//	TEST_BROWSER_PROXY_COUNTRY → country hint (browser-engine only)
//
// Each cloud backend maps these onto its vendor-specific field:
// Browserbase.Proxies=true, Steel.UseProxy=true,
// BrowserEngine.ProxyEnabled=true. Without a unified flag any test
// that needs proxy must know the three field names — flipping them
// here keeps caller sites simple.
func buildComputerFromEnv(t *testing.T) computer.Computer {
	t.Helper()
	backend := strings.ToLower(os.Getenv("TEST_BROWSER"))
	if backend == "" {
		backend = "local"
	}

	// 1280×800 is the test default — ~30% fewer screenshot bytes
	// than 1600×900 (so cheaper + faster per LLM round-trip with
	// vision models that bill by image bytes), badges still readable
	// at 22×16 fixed pixel size, modern web UIs render fine.
	// Override via TEST_BROWSER_WIDTH / TEST_BROWSER_HEIGHT for a
	// specific test that needs a wider viewport.
	w, h := 1280, 800
	if v := os.Getenv("TEST_BROWSER_WIDTH"); v != "" {
		fmt.Sscanf(v, "%d", &w)
	}
	if v := os.Getenv("TEST_BROWSER_HEIGHT"); v != "" {
		fmt.Sscanf(v, "%d", &h)
	}
	useProxy := os.Getenv("TEST_BROWSER_PROXY") == "1"
	// CAPTCHA solver defaults on — disable with TEST_BROWSER_SOLVE_CAPTCHA=0.
	solveCaptcha := os.Getenv("TEST_BROWSER_SOLVE_CAPTCHA") != "0"
	proxyCountry := os.Getenv("TEST_BROWSER_PROXY_COUNTRY")
	// Session lifetime in seconds, applied at creation across every
	// cloud backend. Browserbase + Steel cannot extend post-create
	// via API (verified against their SDKs), so multi-step flows
	// must request a generous lease here.
	//
	// Default 300s (5 min) caps the worst-case cost when a test is
	// hard-killed before its `defer comp.Close()` runs — without an
	// explicit timeout, providers default to 30+ min minimum lease,
	// so a SIGKILL during dev burns 30 minutes of billed cloud
	// time per leaked session. Tests that need longer (Patreon 2FA
	// = 1200, video-draft = 1500) explicitly override via
	// TEST_BROWSER_SESSION_TIMEOUT in the test setup.
	sessionTimeout := 300
	if v := os.Getenv("TEST_BROWSER_SESSION_TIMEOUT"); v != "" {
		fmt.Sscanf(v, "%d", &sessionTimeout)
	}

	// Helper closure used by every branch: register the Computer in
	// the live-cleanup map, and queue a deregister via t.Cleanup so a
	// normal test exit + the test's own defer comp.Close() don't end
	// up double-registered for the next test in the same binary.
	track := func(c computer.Computer) computer.Computer {
		registerLiveComputer(c)
		t.Cleanup(func() { deregisterLiveComputer(c) })
		return c
	}

	switch backend {
	case "local":
		c, err := aptcomputer.New(aptcomputer.Config{Type: "local", Width: w, Height: h})
		if err != nil {
			t.Fatalf("create local: %v", err)
		}
		return track(c)
	case "browserbase":
		k := os.Getenv("BROWSERBASE_API_KEY")
		p := os.Getenv("BROWSERBASE_PROJECT_ID")
		if k == "" || p == "" {
			t.Skip("TEST_BROWSER=browserbase requires BROWSERBASE_API_KEY + BROWSERBASE_PROJECT_ID")
		}
		cfg := aptcomputer.Config{
			Type:          "browserbase",
			APIKey:        k,
			ProjectID:     p,
			Width:         w,
			Height:        h,
			SolveCaptchas: solveCaptcha,
			Timeout:       sessionTimeout,
		}
		if useProxy {
			cfg.Proxies = true // Browserbase managed residential proxy
		}
		c, err := aptcomputer.New(cfg)
		if err != nil {
			t.Fatalf("create browserbase: %v", err)
		}
		if dbg, ok := c.(interface{ DebugURL() string }); ok && dbg.DebugURL() != "" {
			t.Logf("[BROWSERBASE] live view: %s", dbg.DebugURL())
		}
		return track(c)
	case "steel":
		k := os.Getenv("STEEL_API_KEY")
		if k == "" {
			t.Skip("TEST_BROWSER=steel requires STEEL_API_KEY")
		}
		c, err := aptcomputer.New(aptcomputer.Config{
			Type:         "steel",
			APIKey:       k,
			Width:        w,
			Height:       h,
			SolveCaptcha: solveCaptcha,
			UseProxy:     useProxy,
			Timeout:      sessionTimeout, // factory converts seconds → ms for Steel
		})
		if err != nil {
			t.Fatalf("create steel: %v", err)
		}
		if dbg, ok := c.(interface{ DebugURL() string }); ok && dbg.DebugURL() != "" {
			t.Logf("[STEEL] viewer: %s", dbg.DebugURL())
		}
		return track(c)
	case "browser-engine":
		k := os.Getenv("BROWSER_API_KEY")
		if k == "" {
			k = os.Getenv("NEXT_PUBLIC_BROWSER_API_KEY")
		}
		if k == "" {
			t.Skip("TEST_BROWSER=browser-engine requires BROWSER_API_KEY (or NEXT_PUBLIC_BROWSER_API_KEY)")
		}
		baseURL := os.Getenv("BROWSER_API_URL")
		if baseURL == "" {
			baseURL = os.Getenv("NEXT_PUBLIC_BROWSER_API_URL")
		}
		// Legacy env var BROWSER_PROXY_ENABLED still supported for
		// backwards compatibility with earlier test scripts.
		proxyEnabled := useProxy || os.Getenv("BROWSER_PROXY_ENABLED") == "1"
		country := proxyCountry
		if country == "" {
			country = os.Getenv("BROWSER_PROXY_COUNTRY")
		}
		c, err := aptcomputer.New(aptcomputer.Config{
			Type:         "browser-engine",
			APIKey:       k,
			URL:          baseURL,
			Timeout:      sessionTimeout,
			Width:        w,
			Height:       h,
			ProxyEnabled: proxyEnabled,
			ProxyCountry: country,
		})
		if err != nil {
			t.Fatalf("create browser-engine: %v", err)
		}
		if dbg, ok := c.(interface{ DebugURL() string }); ok && dbg.DebugURL() != "" {
			t.Logf("[BROWSER_ENGINE] debug: %s", dbg.DebugURL())
		}
		if sv, ok := c.(interface{ StreamURL() string }); ok && sv.StreamURL() != "" {
			t.Logf("[BROWSER_ENGINE] stream: %s", sv.StreamURL())
		}
		return track(c)
	}
	t.Fatalf("unknown TEST_BROWSER=%q (want local|browserbase|steel|browser-engine)", backend)
	return nil
}

func backendName(t *testing.T) string {
	t.Helper()
	b := os.Getenv("TEST_BROWSER")
	if b == "" {
		return "local"
	}
	return b
}
