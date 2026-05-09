package core

import (
	"os"
	"os/signal"
	"syscall"
	"testing"
)

// TestMain sets APTEVA_SOM=1 as the package-wide default before any
/// test runs. Rationale: every non-Anthropic model we drive in tests
// (Kimi via Fireworks, Google, etc.) cannot reliably produce pixel
// coordinates from a raw screenshot — empirically verified on
// example.com where Kimi stalls at (225, ~190) trying to hit a link
// whose true center is 60+px away. Set-of-Mark (label-badged)
// screenshots are the path we ship, so tests should default to the
// same path unless they explicitly probe raw-pixel behavior (e.g.
// the matrix test, which opts back out).
//
// Individual tests can still override via t.Setenv("APTEVA_SOM", "")
// to restore raw-pixel mode.
func TestMain(m *testing.M) {
	if os.Getenv("APTEVA_SOM") == "" {
		os.Setenv("APTEVA_SOM", "1")
	}
	// Open the log file so THINK / RUN / COMPUTER lines (file-only by
	// default — see logger.go) get captured during test runs. Without
	// this, debugging an agent that stalls mid-loop is impossible
	// because provider.Chat enter/exit traces are silently dropped.
	initLogger()

	// Emergency-cleanup signal handler. If a developer Ctrl-C's the
	// test binary mid-run (or sends SIGTERM via `kill <pid>`), Go's
	// default behaviour is to exit immediately without running any
	// `defer comp.Close()` in the test functions. On cloud backends
	// (Browserbase, Steel, BrowserEngine) that means the session
	// stays alive — and billed — until the provider's idle timeout
	// (typically 30+ minutes per leaked session).
	//
	// Solution: catch SIGINT/SIGTERM here, walk the live-Computer
	// registry that buildComputerFromEnv populates, and Close every
	// session in parallel before letting the process exit. SIGKILL
	// (`kill -9`) still leaks — that's unavoidable — but Ctrl-C and
	// `kill <pid>`, the common dev-loop kills, now release cleanly.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		closeAllLiveComputers()
		// Re-raise the default behaviour so Go test reports the
		// signal in its exit code rather than silently returning 0.
		signal.Reset(syscall.SIGINT, syscall.SIGTERM)
		// Exit code 130 = 128 + SIGINT, the conventional code for
		// "killed by SIGINT" — matches what bash uses.
		os.Exit(130)
	}()

	os.Exit(m.Run())
}
