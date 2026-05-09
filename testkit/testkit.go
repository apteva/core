// Package testkit provides a thin HTTP harness for writing phased,
// real-LLM tests against a live apteva-server. A Session wraps one
// server + one agent instance and exposes the operations you'd do by
// hand through the dashboard: set a directive, inject console events,
// reset the context window, query telemetry, tear down.
//
// Typical shape:
//
//	func TestMyAgent(t *testing.T) {
//	    s := testkit.New(t)           // auto-starts a local server if
//	                                  // APTEVA_SERVER_URL isn't live
//	    s.SetDirective(`Respond "pong" to "ping" messages.`)
//	    s.Run("ping replies", func(p *testkit.Phase) {
//	        p.Inject("[console] ping")
//	        p.WaitUntil(30*time.Second, "an iteration completed", func() bool {
//	            return s.Status().Iteration >= 1
//	        })
//	    })
//	    report := s.Report()
//	    t.Logf("tokens=%d cost=$%.4f", report.TokensIn+report.TokensOut, report.Cost)
//	}
//
// No credentials or client data appear in committed code — tests that
// target production systems live in core/private_scenarios (gitignored)
// and read URLs/keys from the environment.
package testkit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"
)

// Session is the test harness for one apteva-server + one agent
// instance. Create with New(t). Safe for use only from a single test
// goroutine.
type Session struct {
	t          *testing.T
	serverURL  string
	apiKey     string
	projectID  string
	instanceID int64
	client     *http.Client
	startTime  time.Time

	// When testkit spawned the server itself, autostart holds the
	// subprocess + temp dir so Cleanup can reclaim them. Nil when the
	// test is running against an externally-managed server.
	autostart *serverProc
}

// Config lets callers override the automatic environment detection.
// Most tests should use zero-value New(t) and drive through env vars
// (APTEVA_SERVER_URL, APTEVA_API_KEY, APTEVA_TEST_INSTANCE_ID).
type Config struct {
	ServerURL    string        // default: env APTEVA_SERVER_URL, else http://localhost:5280
	APIKey       string        // default: env APTEVA_API_KEY
	InstanceID   int64         // default: env APTEVA_TEST_INSTANCE_ID (if 0 and autostart, one is created)
	ProjectID    string        // default: env APTEVA_TEST_PROJECT_ID
	// DisableAutoStart, when true, forbids spawning a server — tests
	// then fail fast if APTEVA_SERVER_URL isn't already live. Useful
	// for CI where you want the infrastructure pre-set and failures
	// to be loud rather than silent. Default is zero-value (autostart
	// enabled) because the zero-value of a bool is what Go hands a
	// user who writes `testkit.New(t, testkit.Config{...})` without
	// thinking about AutoStart — opt-out is the only way to keep the
	// "just works" shape.
	DisableAutoStart bool
	StartTimeout     time.Duration // default: 15s — how long to wait for a spawned server to respond to /health

	// CreateInstance forces creation of a fresh throwaway instance
	// instead of reusing InstanceID / APTEVA_TEST_INSTANCE_ID. The
	// instance is deleted via t.Cleanup so each test run is fully
	// isolated from the last. Use this for "proper" tests — when you
	// want guaranteed no leftover state, no stale threads, no
	// cross-contamination between test files.
	CreateInstance bool

	// InstanceName applies only with CreateInstance. Defaults to a
	// randomized "testkit-<hex>" so parallel tests don't collide.
	InstanceName string

	// Directive is the initial directive for a newly-created instance.
	// Set here OR call s.SetDirective() after New — they're equivalent
	// except setting it here saves one round-trip.
	Directive string

	// AttachMCPs lists MCP server names to attach as CATALOG entries
	// on the instance — visible to main as a "spawn workers with
	// mcp=X" hint but NOT exposed as native tools on main's registry.
	// This forces the agent to exercise the spawn path: main decides
	// to create a worker, passes the MCP name on spawn, the worker
	// connects and uses the tool. That matches the hub-and-spoke
	// shape real instances run under and gives the tests visibility
	// into thread lifecycle.
	//
	// Each name must match an already-registered MCP server in the
	// project (set up via Dashboard → Settings → MCP Servers or via
	// Composio). Credentials and URLs are looked up from the server's
	// DB — tests never see them.
	AttachMCPs []string

	// AttachMCPsMainAccess is the escape hatch for tests that
	// specifically want to validate main-thread direct tool use
	// (e.g. testing a single lightweight tool call without spawn
	// overhead). Entries listed here are attached with main_access=true;
	// entries in AttachMCPs are catalog-only. Most tests should leave
	// this empty — exercising spawn is the point.
	AttachMCPsMainAccess []string

	// IncludeAptevaServer / IncludeChannels match the same flags on
	// instance creation. Both default to false for test instances —
	// most tests don't want the management gateway or the chat bridge
	// cluttering their tool list.
	IncludeAptevaServer bool
	IncludeChannels     bool

	// Computer enables the browser/computer-use tool surface
	// (browser_session + computer_use) on the instance. When non-nil
	// testkit forwards this block to PUT /api/instances/:id/config in
	// the same call that lands the real directive + MCPs, so core
	// spins up its computer backend before the agent's first real
	// iteration. nil = no computer mode (default).
	//
	// Type=="browserbase"/"steel" relies on a saved provider in the
	// project for credentials — the server enriches the block before
	// forwarding to core. Type=="local" runs against a local Chrome
	// (or an existing CDP endpoint if the matching `browser` provider
	// has CDP_URL set). Type=="service" points at a custom HTTP
	// browser service (URL required).
	Computer *ComputerConfig
}

// ComputerConfig mirrors core.ComputerConfig (we restate the shape
// here so testkit users don't have to import core just to flip on
// browser tools). Only Type is required; Width/Height/URL/APIKey/
// ProjectID are optional and the server fills in credentials from
// saved providers when Type is "browserbase" or "steel".
type ComputerConfig struct {
	Type      string `json:"type"`                 // "local" | "browserbase" | "steel" | "browser-engine" | "service"
	URL       string `json:"url,omitempty"`        // for "service", or local CDP endpoint
	APIKey    string `json:"api_key,omitempty"`    // override saved provider API key (rare)
	ProjectID string `json:"project_id,omitempty"` // for "browserbase"
	Width     int    `json:"width,omitempty"`      // 0 = core default per LLM
	Height    int    `json:"height,omitempty"`
}

// New wires up a Session against the URL in APTEVA_SERVER_URL (or
// http://localhost:5280), using APTEVA_API_KEY + APTEVA_TEST_INSTANCE_ID
// from the environment. If the URL isn't reachable AND AutoStart is
// enabled (default), New spawns apteva-server with an ephemeral data
// directory, bootstraps the setup-token flow to mint an API key, and
// creates a fresh test instance. Everything it spawned is cleaned up
// via t.Cleanup.
func New(t *testing.T, cfg ...Config) *Session {
	t.Helper()
	var c Config
	if len(cfg) > 0 {
		c = cfg[0]
	}
	if c.StartTimeout == 0 {
		c.StartTimeout = 15 * time.Second
	}
	if c.ServerURL == "" {
		c.ServerURL = getenvDefault("APTEVA_SERVER_URL", "http://localhost:5280")
	}
	if c.APIKey == "" {
		c.APIKey = os.Getenv("APTEVA_API_KEY")
	}
	if c.InstanceID == 0 {
		if s := os.Getenv("APTEVA_TEST_INSTANCE_ID"); s != "" {
			fmt.Sscanf(s, "%d", &c.InstanceID)
		}
	}
	if c.ProjectID == "" {
		c.ProjectID = os.Getenv("APTEVA_TEST_PROJECT_ID")
	}

	sess := &Session{
		t:          t,
		serverURL:  strings.TrimRight(c.ServerURL, "/"),
		apiKey:     c.APIKey,
		projectID:  c.ProjectID,
		instanceID: c.InstanceID,
		client:     &http.Client{Timeout: 30 * time.Second},
	}

	// Pre-fill credentials from the user's CLI config if the caller
	// didn't provide them. Covers the case where a live server is
	// already running on the probe URL (so autostart won't fire and
	// fill these itself) but the test runner hasn't exported
	// APTEVA_API_KEY. Reading the same file the `apteva` CLI writes
	// matches the "just works" expectation — if you can `apteva`
	// normally, you can run these tests.
	if sess.apiKey == "" || sess.projectID == "" {
		if cliCfg := lookupCLIConfig(); cliCfg != nil {
			if sess.apiKey == "" {
				sess.apiKey = cliCfg.APIKey
			}
			if sess.projectID == "" {
				sess.projectID = cliCfg.ProjectID
			}
			// Only adopt the CLI's instance id when the caller hasn't
			// asked us to create a fresh one — CreateInstance means
			// "ignore stored id, make a new one for isolation".
			if sess.instanceID == 0 && !c.CreateInstance {
				sess.instanceID = cliCfg.InstanceID
			}
		}
	}

	// Reachability probe. On success we use the externally-managed
	// server. On failure we optionally start one ourselves (default
	// behavior; pass DisableAutoStart=true to refuse and fail fast).
	if !sess.serverAlive() {
		if c.DisableAutoStart {
			t.Fatalf("testkit: server %s not reachable and DisableAutoStart=true", sess.serverURL)
		}
		// Prefer reusing the user's real DB (MCPs, integrations, API
		// keys, and instance IDs they've set up via the dashboard) so
		// tests exercise the same environment they develop against.
		// Falls through to an ephemeral bootstrap if the real DB
		// isn't present (fresh dev machines, CI, etc.).
		proc := startServerReusingLocalDB(t, c.StartTimeout)
		if proc == nil {
			proc = startServer(t, c.StartTimeout)
		}
		sess.serverURL = proc.URL
		if sess.apiKey == "" {
			sess.apiKey = proc.APIKey
		}
		if sess.projectID == "" {
			sess.projectID = proc.ProjectID
		}
		if sess.instanceID == 0 {
			sess.instanceID = proc.InstanceID
		}
		sess.autostart = proc
		t.Cleanup(proc.Stop)
	}

	if sess.apiKey == "" {
		t.Fatalf("testkit: no API key — set APTEVA_API_KEY or let testkit auto-start a server")
	}

	// CreateInstance: ignore any inherited instance id, create a fresh
	// throwaway one, attach requested MCPs, and register a cleanup
	// that deletes it. This is the "proper test" path — fully
	// isolated, repeatable, no cross-run leakage.
	if c.CreateInstance {
		id, err := sess.createFreshInstance(c)
		if err != nil {
			t.Fatalf("testkit: create instance: %v", err)
		}
		sess.instanceID = id
		t.Cleanup(func() { sess.deleteInstance() })
		// SIGINT/SIGTERM catch — Ctrl+C on `go test` bypasses
		// t.Cleanup, so without this the instance row survives the
		// crash and shows up in the user's dashboard. We delete the
		// instance ourselves, stop the autostarted server, then
		// re-raise the signal so Go's exit path still runs.
		sess.installCleanupOnSignal()
	}

	if sess.instanceID == 0 {
		// Neither env nor autostart produced an instance id. Last
		// resort: use the user's first instance. This matches the
		// "point a test at whatever's already running" UX.
		id, err := sess.firstInstanceID()
		if err != nil {
			t.Fatalf("testkit: no APTEVA_TEST_INSTANCE_ID and couldn't discover one: %v", err)
		}
		sess.instanceID = id
	}

	sess.startTime = time.Now()
	// Ensure the target instance is actually running before we push
	// events. When autostart reused a local DB, the stored
	// instance_id may point at a stopped instance — calling /start
	// boots it. Already-running instances yield a 4xx we swallow.
	if err := sess.ensureInstanceRunning(); err != nil {
		t.Fatalf("testkit: ensureInstanceRunning: %v", err)
	}
	// Begin every test from a clean context so iteration counts, token
	// totals, and thread state are reproducible across runs.
	if err := sess.resetContext(); err != nil {
		t.Logf("testkit: warning — context reset failed: %v", err)
	}
	// Live event watcher — prints new telemetry events as they arrive
	// and heartbeats during long silent tool calls (e.g. deepgram waits
	// for audio transcription). Disable with TESTKIT_NO_STREAM=1.
	sess.startStreamer()
	return sess
}

// ensureInstanceRunning boots the target instance if not already live.
// POST /instances/:id/start is idempotent on the server side — an
// already-running instance responds with a 4xx which we swallow. We
// poll /status until it answers cleanly or a 20s ceiling elapses, to
// cover instance boot cost dominated by core binary spawn + provider
// init.
func (s *Session) ensureInstanceRunning() error {
	_ = s.post("/instances/"+itoa(s.instanceID)+"/start", map[string]any{}, nil)
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		var status StatusInfo
		if err := s.get("/instances/"+itoa(s.instanceID)+"/status", &status); err == nil {
			return nil
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("instance %d did not come online within 20s", s.instanceID)
}

// --- Directive / events / reset ------------------------------------------

// installCleanupOnSignal wires SIGINT/SIGTERM so an interrupted
// `go test` still deletes the instance it created and stops the
// autostarted server. Registering more than once per session is
// harmless — the delete handler is idempotent and proc.Stop has its
// own guard.
func (s *Session) installCleanupOnSignal() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		fmt.Fprintf(os.Stderr, "\ntestkit: received %s — cleaning up instance %d\n", sig, s.instanceID)
		s.deleteInstance()
		if s.autostart != nil {
			s.autostart.Stop()
		}
		// Stop catching; let the next signal (or the default one we
		// just forwarded) terminate the process the normal way.
		signal.Stop(sigCh)
		syscall.Kill(syscall.Getpid(), sig.(syscall.Signal))
	}()
}

// SetProviderModels forces an LLM provider to use specific model IDs
// for large/medium/small tiers on this instance. Useful to pin tests to
// a specific model regardless of what the user configured in their DB.
func (s *Session) SetProviderModels(provider string, large, medium, small string) {
	s.t.Helper()
	body := map[string]any{
		"providers": []map[string]any{{
			"name":    provider,
			"default": true,
			"models": map[string]string{
				"large":  large,
				"medium": medium,
				"small":  small,
			},
		}},
	}
	if err := s.putConfig(body); err != nil {
		s.t.Fatalf("testkit: SetProviderModels: %v", err)
	}
}

// SetDirective rewrites the instance's directive. Use this to iterate:
// edit your test's directive string, re-run, compare reports.
func (s *Session) SetDirective(directive string) {
	s.t.Helper()
	if err := s.putConfig(map[string]any{"directive": directive}); err != nil {
		s.t.Fatalf("testkit: SetDirective: %v", err)
	}
}

// Inject sends a console event to main. The text is what the agent
// sees in its [console] block, e.g. "[console] [chat] Hi".
func (s *Session) Inject(text string) {
	s.t.Helper()
	body := map[string]any{"message": text, "thread_id": "main"}
	if err := s.post("/instances/"+itoa(s.instanceID)+"/event", body, nil); err != nil {
		s.t.Fatalf("testkit: Inject %q: %v", text, err)
	}
}

// ResetContext wipes the main thread's message history and kills every
// sub-thread. Called automatically by New; call again between phases
// if you want isolation.
func (s *Session) ResetContext() {
	s.t.Helper()
	if err := s.resetContext(); err != nil {
		s.t.Fatalf("testkit: ResetContext: %v", err)
	}
	s.startTime = time.Now()
}

func (s *Session) resetContext() error {
	return s.putConfig(map[string]any{
		"reset": map[string]any{"history": true, "threads": true},
	})
}

// --- Status + telemetry --------------------------------------------------

// StatusInfo is the minimum subset of the /status payload tests care
// about. Exposed so tests can write conditions like
// `s.Status().Iteration >= 3` without depending on core types.
type StatusInfo struct {
	Iteration int    `json:"iteration"`
	Threads   int    `json:"threads"`
	Paused    bool   `json:"paused"`
	Mode      string `json:"mode"`
	Rate      string `json:"rate"`
	Model     string `json:"model"`
}

// Status returns the current agent status. Fails the test on HTTP error.
func (s *Session) Status() StatusInfo {
	s.t.Helper()
	var out StatusInfo
	if err := s.get("/instances/"+itoa(s.instanceID)+"/status", &out); err != nil {
		s.t.Fatalf("testkit: Status: %v", err)
	}
	return out
}

// Report summarises what the session spent since it started. Computed
// from the server's telemetry-stats endpoint filtered to the session's
// start time (whichever is smaller: sinceStart vs 1h).
type Report struct {
	Iterations  int     `json:"iterations"`
	TokensIn    int     `json:"tokens_in"`
	TokensOut   int     `json:"tokens_out"`
	Cost        float64 `json:"cost"`
	ToolCalls   int     `json:"tool_calls"`
	Errors      int     `json:"errors"`
	DurationSec float64 `json:"duration_sec"`
}

// Report fetches cumulative stats since New() or the last ResetContext().
// Uses the server's 1h stats window and lets the caller subtract the
// session start — close enough for iteration-scale tests.
//
// Brief settle window before the fetch: core posts telemetry events
// to the server asynchronously (batched through forwardLoop), so the
// most recent iteration's llm.done may not have landed in the DB yet
// when a fast test wraps up. Waiting ~1s covers the typical forward
// latency without noticeably slowing tests.
func (s *Session) Report() Report {
	s.t.Helper()
	time.Sleep(1 * time.Second)
	var stats struct {
		LLMCalls       int     `json:"llm_calls"`
		TotalTokensIn  int     `json:"total_tokens_in"`
		TotalTokensOut int     `json:"total_tokens_out"`
		TotalCost      float64 `json:"total_cost"`
		ToolCalls      int     `json:"tool_calls"`
		Errors         int     `json:"errors"`
	}
	if err := s.get("/telemetry/stats?instance_id="+itoa(s.instanceID)+"&period=1h", &stats); err != nil {
		s.t.Logf("testkit: Report stats: %v", err)
	}
	return Report{
		Iterations:  stats.LLMCalls,
		TokensIn:    stats.TotalTokensIn,
		TokensOut:   stats.TotalTokensOut,
		Cost:        stats.TotalCost,
		ToolCalls:   stats.ToolCalls,
		Errors:      stats.Errors,
		DurationSec: time.Since(s.startTime).Seconds(),
	}
}

// --- Phased runner -------------------------------------------------------

// Phase is the argument passed to Session.Run's callback. Inside the
// callback you inject events, poll for conditions, and assert. If any
// method on Phase fails, the phase (and test) aborts with a clear
// message identifying which phase failed.
type Phase struct {
	s       *Session
	name    string
	t       *testing.T
	entered time.Time
}

// Run executes a named phase. The callback receives a Phase bound to
// the session. Each phase's runtime + token/cost stats are logged;
// nothing about Phase auto-resets context — call s.ResetContext()
// yourself between phases if needed.
func (s *Session) Run(name string, fn func(p *Phase)) {
	s.t.Helper()
	s.t.Logf("=== phase: %s ===", name)
	p := &Phase{s: s, name: name, t: s.t, entered: time.Now()}
	fn(p)
	stats := s.StatsSince(p.entered)
	s.t.Logf("    phase %q ok (%.1fs) — iters=%d tool_calls=%d tokens_in=%d tokens_out=%d cost=$%.4f",
		name, time.Since(p.entered).Seconds(),
		stats.Iterations, stats.ToolCalls, stats.TokensIn, stats.TokensOut, stats.Cost)
	// Per-thread context usage for this phase — surfaces context-window
	// bloat before it crashes a later phase. Skip when nothing ran.
	if rows := s.ContextUsage(p.entered); len(rows) > 0 {
		logContextUsage(s.t, fmt.Sprintf("phase %q", name), rows)
	}
}

// LogContextUsage prints the full-session per-thread context usage
// table. Call from the test at end-of-run for a final bloat check.
func (s *Session) LogContextUsage() {
	rows := s.ContextUsage(s.startTime)
	logContextUsage(s.t, "session total", rows)
}

// PhaseStats is the per-window rollup logPhaseStats / StatsSince
// returns. Time bounds are caller-supplied so the same helper works for
// phase-scoped, inject-to-now, or arbitrary-slice queries.
type PhaseStats struct {
	Iterations int     // llm.done events
	ToolCalls  int     // tool.call events
	TokensIn   int     // sum of llm.done.tokens_in
	TokensOut  int     // sum of llm.done.tokens_out
	Cost       float64 // sum of server-enriched llm.done.cost_usd
	Errors     int     // llm.error events
}

// StatsSince walks telemetry for this session's instance and sums
// iteration / tool-call / token / cost totals for events at or after
// `since`. Use it to measure a specific phase or inject-to-now window;
// for whole-session cumulative totals use Report().
func (s *Session) StatsSince(since time.Time) PhaseStats {
	s.t.Helper()
	cutoff := since.UTC().Format(time.RFC3339Nano)
	var out PhaseStats

	for _, e := range s.Events("llm.done", 1000) {
		if e.Time < cutoff {
			continue
		}
		out.Iterations++
		out.TokensIn += asTelemetryInt(e.Data["tokens_in"])
		out.TokensOut += asTelemetryInt(e.Data["tokens_out"])
		if v, ok := e.Data["cost_usd"].(float64); ok {
			out.Cost += v
		}
	}
	for _, e := range s.Events("tool.call", 1000) {
		if e.Time < cutoff {
			continue
		}
		out.ToolCalls++
	}
	for _, e := range s.Events("llm.error", 500) {
		if e.Time < cutoff {
			continue
		}
		out.Errors++
	}
	return out
}

// Stats returns totals for the events emitted inside this phase so
// far. Called automatically at end of Run; available mid-phase too for
// fine-grained assertions (e.g. "this phase should cost under $X").
func (p *Phase) Stats() PhaseStats { return p.s.StatsSince(p.entered) }

func asTelemetryInt(v interface{}) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	}
	return 0
}

// Inject is the phase-scoped equivalent of Session.Inject. Here as a
// convenience so phase callbacks don't close over the session.
func (p *Phase) Inject(text string) { p.s.Inject(text) }

// WaitUntil is the phase-scoped shim around Session.WaitUntil — it
// exists so Phase callbacks can call p.WaitUntil without closing over
// the session. New tests should prefer Session.WaitUntil directly.
func (p *Phase) WaitUntil(timeout time.Duration, desc string, cond func() bool) {
	p.t.Helper()
	p.s.WaitUntil(timeout, desc, cond)
}

// Verify runs assertions. It's just a named wrapper so phase traces
// include which assertion block ran. Use standard t.Errorf/Fatalf
// inside.
func (p *Phase) Verify(desc string, fn func()) {
	p.t.Helper()
	p.t.Logf("    verify: %s", desc)
	fn()
}

// WaitUntil polls cond every 500ms and returns when cond returns true
// or when timeout elapses. On timeout it fatals the test. desc is used
// in the timeout message and in the 15-second progress nudge so silent
// waits don't look like a frozen test runner.
func (s *Session) WaitUntil(timeout time.Duration, desc string, cond func() bool) {
	s.t.Helper()
	deadline := time.Now().Add(timeout)
	poll := 500 * time.Millisecond
	nudge := 15 * time.Second
	nextNudge := time.Now().Add(nudge)
	started := time.Now()
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		if time.Now().After(nextNudge) {
			s.t.Logf("[wait] still waiting (%.0fs elapsed, %.0fs left): %s",
				time.Since(started).Seconds(), time.Until(deadline).Seconds(), desc)
			nextNudge = time.Now().Add(nudge)
		}
		time.Sleep(poll)
	}
	s.t.Fatalf("testkit: timeout after %v waiting for: %s", timeout, desc)
}

// Verify is the session-level counterpart of Phase.Verify — named
// assertion block so test logs show which checks ran. Intended for
// tests that don't use s.Run phases.
func (s *Session) Verify(desc string, fn func()) {
	s.t.Helper()
	s.t.Logf("verify: %s", desc)
	fn()
}

// --- HTTP helpers --------------------------------------------------------

func (s *Session) serverAlive() bool {
	resp, err := s.client.Get(s.serverURL + "/api/health")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (s *Session) firstInstanceID() (int64, error) {
	var list []struct {
		ID int64 `json:"id"`
	}
	if err := s.get("/instances", &list); err != nil {
		return 0, err
	}
	if len(list) == 0 {
		return 0, fmt.Errorf("no instances found")
	}
	return list[0].ID, nil
}

func (s *Session) putConfig(body map[string]any) error {
	return s.put("/instances/"+itoa(s.instanceID)+"/config", body, nil)
}

func (s *Session) get(path string, out any) error {
	return s.do("GET", path, nil, out)
}

func (s *Session) post(path string, body, out any) error {
	return s.do("POST", path, body, out)
}

func (s *Session) put(path string, body, out any) error {
	return s.do("PUT", path, body, out)
}

func (s *Session) do(method, path string, body, out any) error {
	var buf io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		buf = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, s.serverURL+"/api"+path, buf)
	if err != nil {
		return err
	}
	if s.apiKey != "" {
		// Server's authMiddleware accepts the API key as a Bearer
		// token (auth.go: strings.TrimPrefix(auth, "Bearer ")). The
		// dashboard's own fetch wrapper uses the same header, so this
		// matches the production path — tests are auth'd the way a
		// real UI session is.
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		msg, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s %s -> %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

// --- Telemetry query ------------------------------------------------------

// TelemetryEvent is the subset of a stored telemetry row tests look
// at: type (e.g. "tool.call"), thread id, time, and the free-form
// data map provided by the emitter. Tests typically grep Data for
// {"name": "pushover_send_notification", ...} or similar.
type TelemetryEvent struct {
	ID       string                 `json:"id"`
	ThreadID string                 `json:"thread_id"`
	Type     string                 `json:"type"`
	Time     string                 `json:"time"`
	Data     map[string]interface{} `json:"data"`
}

// Events fetches recent telemetry events for the session's instance,
// optionally filtered by type. limit caps the result (default 200 if
// zero). Ordered newest-first by the server.
//
// Typical use: after injecting an event, poll Events("tool.call",...)
// until the expected tool name shows up, then assert on the args.
func (s *Session) Events(typeFilter string, limit int) []TelemetryEvent {
	s.t.Helper()
	if limit <= 0 {
		limit = 200
	}
	path := fmt.Sprintf("/telemetry?instance_id=%d&limit=%d", s.instanceID, limit)
	if typeFilter != "" {
		path += "&type=" + typeFilter
	}
	var out []TelemetryEvent
	if err := s.get(path, &out); err != nil {
		s.t.Fatalf("testkit: Events: %v", err)
	}
	return out
}

// HasToolCall returns true if a tool.call event for the named tool has
// been recorded since the session started. The check is exact on the
// tool name — pass "pushover_send_notification" not "pushover".
func (s *Session) HasToolCall(toolName string) bool {
	for _, e := range s.Events("tool.call", 500) {
		name, _ := e.Data["name"].(string)
		if name == toolName {
			return true
		}
	}
	return false
}

// HasToolCallWithPrefix returns true when any tool.call event's name
// starts with prefix. Useful for MCP-backed tools where the agent
// picks the specific endpoint (pushover_send_notification vs.
// pushover_send_priority_alert vs. ...) — the test cares that "some
// pushover tool fired", not which one.
// WaitIdle blocks until we're confident the scenario has settled.
// Returns when ANY of these three conditions is met (whichever comes
// first), bounded by `timeout`:
//
//   1. `quiet` consecutive time with no new telemetry events. Simple
//      plain-silence rule for tests where the agent just stops emitting.
//
//   2. main has paced to a long sleep (≥1m) and a short grace period
//      (`min(quiet, 5s)`) has passed since it did so. Main is the
//      orchestrator — if it's asleep for an hour, nothing meaningful
//      is going to happen regardless of what workers do. This
//      specifically avoids the trap where a worker's 5m pace fires a
//      heartbeat event that resets the quiet timer long after the
//      real work is done.
//
//   3. All live threads have paced with a long sleep. Same spirit as
//      (2) but for tests that finish before main emits a long pace
//      (e.g. when the scenario uses a different top-level strategy).
func (s *Session) WaitIdle(timeout, quiet time.Duration) {
	s.t.Helper()
	deadline := time.Now().Add(timeout)
	lastSeen := latestEventTimeSession(s)
	lastChange := time.Now()
	started := time.Now()
	nextNudge := time.Now().Add(15 * time.Second)
	mainPacedGrace := quiet
	if mainPacedGrace > 5*time.Second {
		mainPacedGrace = 5 * time.Second
	}

	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		cur := latestEventTimeSession(s)
		if cur != lastSeen {
			lastSeen = cur
			lastChange = time.Now()
		}
		if time.Since(lastChange) >= quiet {
			return
		}
		if mainLongPaced(s) && time.Since(lastChange) >= mainPacedGrace {
			s.t.Logf("[idle] main has paced to a long sleep — short-circuiting")
			return
		}
		if allThreadsPaced(s) {
			s.t.Logf("[idle] all threads paced — short-circuiting quiet wait")
			return
		}
		if time.Now().After(nextNudge) {
			s.t.Logf("[idle] waiting for quiescence (%.0fs elapsed, last event %.0fs ago, need %.0fs quiet)",
				time.Since(started).Seconds(), time.Since(lastChange).Seconds(), quiet.Seconds())
			nextNudge = time.Now().Add(15 * time.Second)
		}
	}
	s.t.Logf("[idle] timeout after %v without %v of quiet", timeout, quiet)
}

// mainLongPaced reports true when main's most recent tool.call is a
// pace with sleep ≥ 1m. Used as the primary "test is done" signal
// because main is the orchestrator: a long pace on main means the
// scenario has committed to going idle, even if sub-threads haven't
// cleanly paced themselves.
func mainLongPaced(s *Session) bool {
	for _, e := range s.eventsRaw("tool.call", 100) {
		if e.ThreadID != "main" {
			continue
		}
		name, _ := e.Data["name"].(string)
		if name != "pace" {
			return false
		}
		args, _ := e.Data["args"].(map[string]interface{})
		sleep, _ := args["sleep"].(string)
		return isLongSleep(sleep)
	}
	return false
}

func latestEventTimeSession(s *Session) string {
	evs := s.eventsRaw("", 1)
	if len(evs) == 0 {
		return ""
	}
	return evs[0].Time
}

// allThreadsPaced reports true when every thread that has emitted
// activity has followed up with a pace(sleep≥1m) after its last
// non-pace tool call. This is the intended terminal state for most
// tests — the agent has paced to sleep on its own.
func allThreadsPaced(s *Session) bool {
	// Group recent tool.calls by thread, newest first. A thread is
	// "settled" if its newest tool.call is a pace with sleep ≥ 1m.
	type last struct {
		name  string
		sleep string
	}
	lastPerThread := map[string]last{}
	for _, e := range s.eventsRaw("tool.call", 500) {
		if _, ok := lastPerThread[e.ThreadID]; ok {
			continue
		}
		name, _ := e.Data["name"].(string)
		args, _ := e.Data["args"].(map[string]interface{})
		sleep, _ := args["sleep"].(string)
		lastPerThread[e.ThreadID] = last{name: name, sleep: sleep}
	}
	if len(lastPerThread) == 0 {
		return false
	}
	for _, l := range lastPerThread {
		if l.name != "pace" {
			return false
		}
		if !isLongSleep(l.sleep) {
			return false
		}
	}
	return true
}

// isLongSleep reports whether a pace sleep string represents at least
// a one-minute sleep. Accepts formats like "1m", "5m", "1h".
func isLongSleep(sleep string) bool {
	if sleep == "" {
		return false
	}
	d, err := time.ParseDuration(sleep)
	if err != nil {
		return false
	}
	return d >= 60*time.Second
}

// ThreadContextUsage summarises context-window usage for one thread
// across a telemetry slice. Peaks come from llm.done events.
type ThreadContextUsage struct {
	ThreadID     string
	PeakIn       int // max tokens_in seen
	LastIn       int // most recent tokens_in
	ContextMax   int // model's max_context_tokens (0 if unknown)
	PeakMsgs     int // max context_msgs
	PeakChars    int // max context_chars
	Iters        int // # llm.done events
}

// ContextUsage walks llm.done events since `since` and returns a
// peak-tokens snapshot per thread. Used by the end-of-phase and
// end-of-run reports to surface context bloat before it bites.
func (s *Session) ContextUsage(since time.Time) []ThreadContextUsage {
	s.t.Helper()
	cutoff := since.UTC().Format(time.RFC3339Nano)
	byThread := map[string]*ThreadContextUsage{}
	// Events are newest-first; iterate all and track per-thread peaks.
	for _, e := range s.eventsRaw("llm.done", 2000) {
		if e.Time < cutoff {
			continue
		}
		u, ok := byThread[e.ThreadID]
		if !ok {
			u = &ThreadContextUsage{ThreadID: e.ThreadID}
			byThread[e.ThreadID] = u
		}
		u.Iters++
		in := asTelemetryInt(e.Data["tokens_in"])
		if in > u.PeakIn {
			u.PeakIn = in
		}
		if u.LastIn == 0 {
			u.LastIn = in // first we hit is the newest (newest-first)
		}
		if m := asTelemetryInt(e.Data["max_context_tokens"]); m > u.ContextMax {
			u.ContextMax = m
		}
		if m := asTelemetryInt(e.Data["context_msgs"]); m > u.PeakMsgs {
			u.PeakMsgs = m
		}
		if c := asTelemetryInt(e.Data["context_chars"]); c > u.PeakChars {
			u.PeakChars = c
		}
	}
	out := make([]ThreadContextUsage, 0, len(byThread))
	for _, u := range byThread {
		out = append(out, *u)
	}
	// Stable-sort: main first, then by peak tokens desc so noisy
	// threads surface at the top.
	sortContextUsage(out)
	return out
}

// logContextUsage formats a ContextUsage slice as a human-readable
// table and logs it on `t`. Flags any thread above 50% of its context
// window with a ⚠ marker so bloat is impossible to miss in CI logs.
func logContextUsage(t logger, label string, rows []ThreadContextUsage) {
	if len(rows) == 0 {
		return
	}
	t.Logf("context usage (%s):", label)
	for _, u := range rows {
		pct := ""
		flag := ""
		if u.ContextMax > 0 {
			p := float64(u.PeakIn) * 100.0 / float64(u.ContextMax)
			pct = fmt.Sprintf(" (%.1f%%)", p)
			if p >= 50 {
				flag = "  ⚠ over 50% of context window"
			}
		}
		maxStr := "?"
		if u.ContextMax > 0 {
			maxStr = fmt.Sprintf("%d", u.ContextMax)
		}
		t.Logf("  %-16s  peak=%d/%s%s  msgs=%d  chars=%d  iters=%d%s",
			u.ThreadID, u.PeakIn, maxStr, pct, u.PeakMsgs, u.PeakChars, u.Iters, flag)
	}
}

// logger is a narrow interface — *testing.T satisfies it — so the
// context report can be driven from anywhere without pulling testing
// into unrelated callers.
type logger interface {
	Logf(format string, args ...any)
}

// sortContextUsage puts "main" at the top, then ranks the rest by peak
// input tokens descending so noisy threads are easy to spot.
func sortContextUsage(rows []ThreadContextUsage) {
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].ThreadID == "main" {
			return true
		}
		if rows[j].ThreadID == "main" {
			return false
		}
		return rows[i].PeakIn > rows[j].PeakIn
	})
}

func (s *Session) HasToolCallWithPrefix(prefix string) bool {
	for _, e := range s.Events("tool.call", 500) {
		name, _ := e.Data["name"].(string)
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// SpawnedThreads returns the IDs of sub-threads that were spawned
// during the session. Driven by thread.spawn telemetry events so it
// works after the thread has already finished — no polling race with
// active-thread enumeration.
func (s *Session) SpawnedThreads() []string {
	seen := map[string]bool{}
	var ids []string
	for _, e := range s.Events("thread.spawn", 500) {
		// thread.spawn events are emitted ON the new thread, so
		// e.ThreadID is the spawned thread's id itself.
		if e.ThreadID == "" || seen[e.ThreadID] {
			continue
		}
		seen[e.ThreadID] = true
		ids = append(ids, e.ThreadID)
	}
	return ids
}

// ToolCallsByThread returns all recorded tool.call events grouped by
// the thread that made them. Lets tests assert specifically that
// "some sub-thread (not main) called X" — which is the hub-and-spoke
// shape we want most tests to validate rather than short-circuiting
// tool calls straight from main.
func (s *Session) ToolCallsByThread() map[string][]string {
	out := map[string][]string{}
	for _, e := range s.Events("tool.call", 500) {
		name, _ := e.Data["name"].(string)
		out[e.ThreadID] = append(out[e.ThreadID], name)
	}
	return out
}

// --- Fresh-instance lifecycle --------------------------------------------

// createFreshInstance creates a brand-new instance and (optionally)
// attaches MCPs. Returns the new instance id so the Session can adopt
// it and later tear it down on t.Cleanup.
//
// Ordering matters here. PUT /config forwards mcp_servers to core's
// reconcileMCP only when the core is live — if we attach before
// starting, the config update hits the DB but never reaches core, so
// the MCP tools never get registered on the agent. We therefore:
//
//   1. POST /api/instances with a placeholder directive, start=true.
//      We deliberately boot with a no-op directive so the agent
//      doesn't fire any actions before its tools are wired.
//   2. Wait for /status to respond (core is live).
//   3. For each AttachMCPs name: look up the project's registered
//      MCP server rows, build an MCPServerConfig, PUT /config with
//      mcp_servers + the real directive in one call. Core runs
//      reconcileMCP synchronously; the agent's next iteration has
//      the tools + the real directive together.
func (s *Session) createFreshInstance(c Config) (int64, error) {
	name := c.InstanceName
	if name == "" {
		name = "testkit-" + randHexLocal(4)
	}
	realDirective := c.Directive
	if realDirective == "" {
		realDirective = "Idle. Waiting for test directives."
	}
	// Placeholder directive so the agent stays quiet while we wire up
	// tools. Replaced below in the same PUT that attaches MCPs.
	const placeholder = "Idle. Initializing test environment — wait for updated directive."

	body := map[string]any{
		"name":                  name,
		"directive":             placeholder,
		"mode":                  "autonomous",
		"project_id":            s.projectID,
		"include_apteva_server": c.IncludeAptevaServer,
		"include_channels":      c.IncludeChannels,
	}
	var resp struct {
		ID int64 `json:"id"`
	}
	if err := s.post("/instances", body, &resp); err != nil {
		return 0, fmt.Errorf("POST /instances: %w", err)
	}
	s.t.Logf("testkit: created fresh instance %d (name=%q)", resp.ID, name)
	s.instanceID = resp.ID

	// Wait for core to be live before any PUT that needs to reach it.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		var st StatusInfo
		if err := s.get("/instances/"+itoa(resp.ID)+"/status", &st); err == nil {
			break
		}
		time.Sleep(300 * time.Millisecond)
	}

	// Build the update payload: real directive + MCP list (if any) +
	// computer config (if any). The server's PUT /config handler
	// forwards mcp_servers / computer / providers / threads /
	// unconscious to core; everything else stays at the server layer.
	update := map[string]any{"directive": realDirective}

	if c.Computer != nil {
		update["computer"] = c.Computer
		s.t.Logf("testkit: enabling computer mode (type=%q)", c.Computer.Type)
	}

	if len(c.AttachMCPs) > 0 || len(c.AttachMCPsMainAccess) > 0 {
		mcps, err := s.fetchMCPServers()
		if err != nil {
			return resp.ID, fmt.Errorf("fetch mcp-servers: %w", err)
		}
		// mainAccess[name]=true → main can call the tool directly.
		// Anything present in AttachMCPs but not in AttachMCPsMainAccess
		// is catalog-only, which forces the agent through the spawn
		// path to use it.
		wanted := map[string]bool{}
		mainAccess := map[string]bool{}
		for _, n := range c.AttachMCPs {
			wanted[n] = true
		}
		for _, n := range c.AttachMCPsMainAccess {
			wanted[n] = true
			mainAccess[n] = true
		}
		var entries []map[string]any
		found := map[string]bool{}
		for _, m := range mcps {
			if m.ProxyConfig == nil {
				continue
			}
			name := m.Name
			if !wanted[name] && !wanted[m.ProxyConfig.Name] {
				continue
			}
			found[m.Name] = true
			found[m.ProxyConfig.Name] = true
			entry := map[string]any{
				"name":        m.ProxyConfig.Name,
				"transport":   m.ProxyConfig.Transport,
				"main_access": mainAccess[m.Name] || mainAccess[m.ProxyConfig.Name],
			}
			if m.ProxyConfig.URL != "" {
				entry["url"] = m.ProxyConfig.URL
			}
			if m.ProxyConfig.Command != "" {
				entry["command"] = m.ProxyConfig.Command
			}
			if len(m.ProxyConfig.Args) > 0 {
				entry["args"] = m.ProxyConfig.Args
			}
			entries = append(entries, entry)
		}
		for n := range wanted {
			if !found[n] {
				return resp.ID, fmt.Errorf("MCP %q not found in project %q — register it via Dashboard → Settings → MCP Servers first", n, s.projectID)
			}
		}
		update["mcp_servers"] = entries
		s.t.Logf("testkit: attached %d MCPs on live core (catalog: %v, main_access: %v)",
			len(entries), c.AttachMCPs, c.AttachMCPsMainAccess)
	}

	if err := s.putConfig(update); err != nil {
		return resp.ID, fmt.Errorf("PUT /config with mcp_servers + directive: %w", err)
	}

	// Give core one beat to run reconcileMCP + pick up the new
	// directive before the test starts injecting events. Without
	// this, fast tests can inject faster than the next iteration
	// which would see the update.
	time.Sleep(500 * time.Millisecond)
	return resp.ID, nil
}

// deleteInstance removes the session's instance via the same endpoint
// the dashboard's "Delete" button uses. Called from t.Cleanup when
// CreateInstance=true. Best-effort: we log but don't fail the test
// if teardown hits an error (the test result is what matters).
func (s *Session) deleteInstance() {
	if s.instanceID == 0 {
		return
	}
	err := s.do("DELETE", "/instances/"+itoa(s.instanceID), nil, nil)
	if err != nil {
		s.t.Logf("testkit: delete instance %d: %v", s.instanceID, err)
		return
	}
	s.t.Logf("testkit: deleted instance %d", s.instanceID)
}

// mcpServerRow is the subset of /api/mcp-servers we care about for
// attach. The real endpoint returns more fields (status, tool_count,
// etc.) — we ignore them here.
type mcpServerRow struct {
	ID          int64             `json:"id"`
	Name        string            `json:"name"`
	ProxyConfig *mcpProxyConfig   `json:"proxy_config,omitempty"`
}

type mcpProxyConfig struct {
	Name      string   `json:"name"`
	Transport string   `json:"transport"`
	URL       string   `json:"url,omitempty"`
	Command   string   `json:"command,omitempty"`
	Args      []string `json:"args,omitempty"`
}

func (s *Session) fetchMCPServers() ([]mcpServerRow, error) {
	path := "/mcp-servers"
	if s.projectID != "" {
		path += "?project_id=" + s.projectID
	}
	var rows []mcpServerRow
	if err := s.get(path, &rows); err != nil {
		return nil, err
	}
	return rows, nil
}

// --- Small utilities -----------------------------------------------------

func getenvDefault(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func itoa(n int64) string { return fmt.Sprintf("%d", n) }

// randHexLocal is a small wrapper so testkit.go doesn't need to
// duplicate autostart.go's crypto/rand import. Kept here because the
// helper is used by fresh-instance name generation, which lives in
// this file rather than autostart.go.
func randHexLocal(n int) string { return randHex(n) }
