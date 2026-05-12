package testkit

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/cookiejar"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// serverProc is the bookkeeping for an apteva-server subprocess testkit
// spawned. Nothing outside this file needs to look at it; callers get a
// fully-configured Session from New() and the cleanup wiring is
// registered via t.Cleanup.
type serverProc struct {
	cmd        *exec.Cmd
	dataDir    string
	URL        string
	APIKey     string
	ProjectID  string
	InstanceID int64
	stderrBuf  *syncBuffer
	stopped    bool
	stopMu     sync.Mutex
}

// startServerReusingLocalDB tries to spawn apteva-server pointed at
// the user's existing ~/.apteva data directory. Authenticates with
// the API key stored in ~/.apteva/apteva.json (populated by the
// normal `apteva` CLI setup). Returns nil if the real DB/config
// isn't present — the caller then falls back to the ephemeral
// bootstrap path.
//
// Why this exists: most of the value of testkit is running against
// the MCPs, integrations, and providers you configured via the
// dashboard. Re-seeding all of that on every test is both painful
// and impossible for OAuth-gated Composio connections. The real DB
// has it, so we use it.
//
// Cost: the server runs against persistent state. Tests that mutate
// that state (instance config, saved threads) will be visible on
// the next `apteva` run. Use a dedicated test instance, not your
// daily driver.
func startServerReusingLocalDB(t *testing.T, timeout time.Duration) *serverProc {
	t.Helper()

	// Find the user's data dir. APTEVA_DATA_DIR overrides; otherwise
	// the normal CLI installs at ~/.apteva.
	dataDir := os.Getenv("APTEVA_DATA_DIR")
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		if home == "" {
			return nil
		}
		dataDir = filepath.Join(home, ".apteva")
	}
	if _, err := os.Stat(filepath.Join(dataDir, "apteva.db")); err != nil {
		return nil
	}

	// Read apteva.json for the API key + default instance. Absence is
	// not fatal — the caller may still pass APTEVA_API_KEY etc. via
	// env — but we log a warning so the mode difference is obvious.
	apiKey := os.Getenv("APTEVA_API_KEY")
	var instanceID int64
	projectID := os.Getenv("APTEVA_TEST_PROJECT_ID")
	if cfg := readAptevaCLIConfig(filepath.Join(dataDir, "apteva.json")); cfg != nil {
		if apiKey == "" {
			apiKey = cfg.APIKey
		}
		if instanceID == 0 &&
			os.Getenv("APTEVA_TEST_AGENT_ID") == "" &&
			os.Getenv("APTEVA_TEST_INSTANCE_ID") == "" {
			instanceID = cfg.InstanceID
		}
		if projectID == "" {
			projectID = cfg.ProjectID
		}
	}
	if apiKey == "" {
		t.Logf("testkit: ~/.apteva present but no API key found (set APTEVA_API_KEY); falling back to ephemeral")
		return nil
	}

	// Purge orphaned testkit-* instance rows before apteva-server boots
	// and tries to resume them. These are instances from prior test
	// runs that were interrupted before t.Cleanup could run — typical
	// case is a user Ctrl+C on go test. The row hangs around, and on
	// the next boot apteva-server auto-spawns a core for every
	// "running" instance, leaking processes and ports. A test-owned
	// name prefix + explicit sweep is the cleanest fix.
	if n := purgeStaleTestInstances(filepath.Join(dataDir, "apteva.db")); n > 0 {
		t.Logf("testkit: purged %d stale testkit-* instance(s) from %s", n, filepath.Base(dataDir))
	}

	serverBin := findServerBinary(t)
	coreBin := findCoreBinary(t)
	port, err := freePort()
	if err != nil {
		t.Fatalf("testkit: allocate port: %v", err)
	}

	cmd := exec.Command(serverBin)
	cmd.Env = append(os.Environ(),
		"PORT="+itoa(int64(port)),
		"DATA_DIR="+dataDir,
		"DB_PATH="+filepath.Join(dataDir, "apteva.db"),
		"CORE_CMD="+coreBin,
		// Registration already happened in this DB — "locked" mode
		// refuses new registrations, which is what we want.
		"APTEVA_REGISTRATION=locked",
	)
	stderr := newSyncBuffer()
	cmd.Stderr = io.MultiWriter(stderr, os.Stderr)
	cmd.Stdout = os.Stdout
	if err := cmd.Start(); err != nil {
		t.Logf("testkit: failed to spawn server against local DB: %v — falling back", err)
		return nil
	}
	proc := &serverProc{
		cmd:        cmd,
		dataDir:    "", // DO NOT delete — it's the user's real data dir
		URL:        fmt.Sprintf("http://127.0.0.1:%d", port),
		APIKey:     apiKey,
		ProjectID:  projectID,
		InstanceID: instanceID,
		stderrBuf:  stderr,
	}
	if err := waitHealth(proc.URL, timeout); err != nil {
		proc.Stop()
		t.Fatalf("testkit: server pointed at %s did not become healthy: %v\n\n--- stderr ---\n%s",
			dataDir, err, stderr.String())
	}
	t.Logf("testkit: reusing local DB at %s (instance=%d, project=%s)", dataDir, proc.InstanceID, proc.ProjectID)
	return proc
}

// aptevaCLIConfig mirrors the subset of ~/.apteva/apteva.json that
// testkit cares about — the API key the CLI minted for itself plus
// the instance/project IDs it's currently pointing at.
type aptevaCLIConfig struct {
	APIKey     string `json:"api_key"`
	InstanceID int64  `json:"instance_id"`
	ProjectID  string `json:"project_id"`
}

// lookupCLIConfig finds and parses the apteva CLI config at the
// canonical path. APTEVA_DATA_DIR overrides; default is ~/.apteva.
// Returns nil if the file doesn't exist or can't be parsed — callers
// use that as "no CLI setup, fall back to env vars".
func lookupCLIConfig() *aptevaCLIConfig {
	dataDir := os.Getenv("APTEVA_DATA_DIR")
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		if home == "" {
			return nil
		}
		dataDir = filepath.Join(home, ".apteva")
	}
	return readAptevaCLIConfig(filepath.Join(dataDir, "apteva.json"))
}

func readAptevaCLIConfig(path string) *aptevaCLIConfig {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var cfg aptevaCLIConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil
	}
	return &cfg
}

// startServer picks a free port, finds the apteva-server binary,
// launches it with a throwaway data directory, waits for /health, then
// bootstraps: register-with-setup-token → mint an API key → create an
// agent instance. The returned proc can be stopped via Stop(); the
// caller registers that as a t.Cleanup.
func startServer(t *testing.T, timeout time.Duration) *serverProc {
	t.Helper()
	serverBin := findServerBinary(t)
	coreBin := findCoreBinary(t)

	port, err := freePort()
	if err != nil {
		t.Fatalf("testkit: allocate port: %v", err)
	}
	dataDir, err := os.MkdirTemp("", "apteva-testkit-*")
	if err != nil {
		t.Fatalf("testkit: mktemp: %v", err)
	}

	// We don't force APTEVA_REGISTRATION. Leaving it empty tells the
	// server to auto-detect: fresh DB → has no users → mode "setup" +
	// generate a one-time token printed to stderr. We scrape the
	// token from stderr once /health responds, which is exactly the
	// same flow a human operator follows from the dashboard's setup
	// screen.
	cmd := exec.Command(serverBin)
	cmd.Env = append(os.Environ(),
		"PORT="+itoa(int64(port)),
		"DATA_DIR="+dataDir,
		"DB_PATH="+filepath.Join(dataDir, "apteva.db"),
		"CORE_CMD="+coreBin,
	)
	stderr := newSyncBuffer()
	cmd.Stderr = io.MultiWriter(stderr, os.Stderr)
	cmd.Stdout = os.Stdout

	if err := cmd.Start(); err != nil {
		os.RemoveAll(dataDir)
		t.Fatalf("testkit: start server: %v", err)
	}
	proc := &serverProc{
		cmd:       cmd,
		dataDir:   dataDir,
		URL:       fmt.Sprintf("http://127.0.0.1:%d", port),
		stderrBuf: stderr,
	}

	if err := waitHealth(proc.URL, timeout); err != nil {
		proc.Stop()
		t.Fatalf("testkit: server at %s did not become healthy in %v: %v\n\n--- server stderr ---\n%s",
			proc.URL, timeout, err, stderr.String())
	}

	// Scrape the setup token from the server's startup banner. If
	// /health came up before the setup-token line was flushed we'll
	// briefly retry the scrape — the flush is usually a millisecond
	// after the HTTP listener is live but we don't want to race.
	setupToken := waitForSetupToken(stderr, 3*time.Second)
	if setupToken == "" {
		proc.Stop()
		t.Fatalf("testkit: could not locate setup token in server stderr\n\n--- server stderr ---\n%s", stderr.String())
	}

	if err := bootstrap(proc, setupToken); err != nil {
		proc.Stop()
		t.Fatalf("testkit: bootstrap: %v", err)
	}
	return proc
}

// waitForSetupToken polls the stderr buffer up to `within` for the
// "Setup token: apt_..." banner. The server prints it shortly after
// listener setup; the sequence is always "listen → print banner", but
// there's a small flush delay we accommodate here.
func waitForSetupToken(buf *syncBuffer, within time.Duration) string {
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if tok := scrapeSetupToken(buf.String()); tok != "" {
			return tok
		}
		time.Sleep(50 * time.Millisecond)
	}
	return scrapeSetupToken(buf.String())
}

// purgeStaleTestInstances deletes every instance row whose name starts
// with "testkit-". We only ever own rows we created — the prefix is
// reserved for testkit — so wiping them all is safe. Uses the sqlite3
// binary to avoid adding a DB driver dependency to testkit.
func purgeStaleTestInstances(dbPath string) int {
	if _, err := os.Stat(dbPath); err != nil {
		return 0
	}
	out, err := exec.Command("sqlite3", dbPath,
		"DELETE FROM instances WHERE name LIKE 'testkit-%'; SELECT changes();",
	).Output()
	if err != nil {
		return 0
	}
	var n int
	fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &n)
	return n
}

// Stop terminates the subprocess and removes the ephemeral data dir.
// Safe to call multiple times — the mutex + stopped guard means extra
// calls are no-ops.
//
// Two-phase shutdown: SIGTERM first so apteva-server can run its own
// cleanup (StopAll on child apteva-core processes) and close its
// stderr pipe. If it doesn't exit within a grace window, SIGKILL.
//
// Why it matters: Go's exec sets up an internal pipe for any non-file
// Stderr (we use io.MultiWriter). Orphaned apteva-core children
// inherit that pipe's write-end; a bare SIGKILL on the server leaves
// the pipe open through the children, and cmd.Wait() blocks until the
// `go test` timeout fires. A clean SIGTERM → server.StopAll avoids
// that by closing the whole family tree.
func (p *serverProc) Stop() {
	p.stopMu.Lock()
	defer p.stopMu.Unlock()
	if p.stopped {
		return
	}
	p.stopped = true
	if p.cmd != nil && p.cmd.Process != nil {
		done := make(chan error, 1)
		go func() { _, err := p.cmd.Process.Wait(); done <- err }()

		_ = p.cmd.Process.Signal(syscall.SIGTERM)
		select {
		case <-done:
			// graceful exit
		case <-time.After(6 * time.Second):
			_ = p.cmd.Process.Kill()
			<-done
		}
	}
	if p.dataDir != "" {
		os.RemoveAll(p.dataDir)
	}
}

// --- Binary discovery ----------------------------------------------------

func findServerBinary(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("APTEVA_SERVER_BIN"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	// Try common locations: sibling dir from core (repo layout) +
	// installed npm path.
	candidates := []string{
		"../server/apteva-server",
		"../../server/apteva-server",
	}
	// Absolute fallbacks: versioned install dirs under ~/.apteva/bin.
	home, _ := os.UserHomeDir()
	if home != "" {
		matches, _ := filepath.Glob(filepath.Join(home, ".apteva/bin/*/apteva-server"))
		candidates = append(candidates, matches...)
	}
	for _, c := range candidates {
		if info, err := os.Stat(c); err == nil && info.Mode().IsRegular() {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	// Last resort: PATH.
	if p, err := exec.LookPath("apteva-server"); err == nil {
		return p
	}
	t.Fatalf("testkit: apteva-server binary not found — set APTEVA_SERVER_BIN or run build-local.sh")
	return ""
}

func findCoreBinary(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("APTEVA_CORE_BIN"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	candidates := []string{
		"./apteva-core",
		"../core/apteva-core",
	}
	home, _ := os.UserHomeDir()
	if home != "" {
		matches, _ := filepath.Glob(filepath.Join(home, ".apteva/bin/*/apteva-core"))
		candidates = append(candidates, matches...)
	}
	for _, c := range candidates {
		if info, err := os.Stat(c); err == nil && info.Mode().IsRegular() {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	if p, err := exec.LookPath("apteva-core"); err == nil {
		return p
	}
	t.Fatalf("testkit: apteva-core binary not found — set APTEVA_CORE_BIN or run build-local.sh")
	return ""
}

// --- Health probe --------------------------------------------------------

func waitHealth(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get(url + "/api/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("timeout")
}

// --- Bootstrap: register → login → key → instance -----------------------

// bootstrap creates the first user, mints an API key via the session
// cookie, and creates a test instance. The sequence mirrors what a new
// operator does the first time they open the dashboard, so the paths
// this hits are the same ones a real user hits — keeps the test
// harness honest.
func bootstrap(p *serverProc, setupToken string) error {
	jar, _ := cookiejar.New(nil)
	client := &http.Client{Timeout: 10 * time.Second, Jar: jar}

	email := "testkit+" + randHex(4) + "@local"
	password := "testkit-" + randHex(8)

	// 1. Register with setup token.
	regBody, _ := json.Marshal(map[string]string{"email": email, "password": password})
	req, _ := http.NewRequest("POST", p.URL+"/api/auth/register", bytes.NewReader(regBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Setup-Token", setupToken)
	if err := doExpect2xx(client, req, "register"); err != nil {
		return err
	}

	// 2. Login to get a session cookie (stored in the jar).
	loginBody, _ := json.Marshal(map[string]string{"email": email, "password": password})
	req, _ = http.NewRequest("POST", p.URL+"/api/auth/login", bytes.NewReader(loginBody))
	req.Header.Set("Content-Type", "application/json")
	if err := doExpect2xx(client, req, "login"); err != nil {
		return err
	}

	// 3. Mint an API key.
	keyBody, _ := json.Marshal(map[string]string{"name": "testkit"})
	req, _ = http.NewRequest("POST", p.URL+"/api/auth/keys", bytes.NewReader(keyBody))
	req.Header.Set("Content-Type", "application/json")
	var keyResp struct {
		Key string `json:"key"`
	}
	if err := doExpectJSON(client, req, &keyResp, "create api key"); err != nil {
		return err
	}
	if keyResp.Key == "" {
		return fmt.Errorf("empty api key in response")
	}
	p.APIKey = keyResp.Key

	// 4a. Seed at least one LLM provider from the caller's env so the
	// fresh server has someone to talk to. Testkit doesn't care which
	// provider wins — it just needs one to exist. Checked in order of
	// "most likely to be set on a dev machine running apteva". If the
	// caller wants a specific one, they can still create more via the
	// dashboard API after New() returns.
	providerSeeds := []struct {
		envKey     string
		typ        string
		name       string
		typeID     int64
	}{
		{"FIREWORKS_API_KEY", "fireworks", "Fireworks", 1},
		{"ANTHROPIC_API_KEY", "anthropic", "Anthropic", 0},
		{"OPENAI_API_KEY", "openai", "OpenAI", 0},
		{"GOOGLE_API_KEY", "google", "Google", 0},
	}
	for _, ps := range providerSeeds {
		key := os.Getenv(ps.envKey)
		if key == "" {
			continue
		}
		provBody, _ := json.Marshal(map[string]any{
			"type": ps.typ,
			"name": ps.name,
			"data": map[string]string{ps.envKey: key},
			"provider_type_id": ps.typeID,
		})
		req, _ := http.NewRequest("POST", p.URL+"/api/providers", bytes.NewReader(provBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
		// Best-effort: log a warning but don't fail — the test may
		// fall through to a different provider that was wired up by
		// a prior run, or fail loudly at the first /status call. The
		// latter is fine; it's the caller's responsibility to have at
		// least one LLM credential available to the test runner.
		_ = doExpect2xx(client, req, "seed provider "+ps.typ)
	}

	// 4. Find the default project the register handler auto-created.
	req, _ = http.NewRequest("GET", p.URL+"/api/projects", nil)
	req.Header.Set("Authorization", "Bearer "+p.APIKey)
	var projects []struct {
		ID string `json:"id"`
	}
	if err := doExpectJSON(client, req, &projects, "list projects"); err != nil {
		return err
	}
	if len(projects) > 0 {
		p.ProjectID = projects[0].ID
	}

	// 5. Create a test instance. Start=false so tests that want to
	// boot with a specific directive can do so via SetDirective +
	// start-instance. We'll just start it now with a minimal
	// directive so the agent is live when the first test runs.
	instBody, _ := json.Marshal(map[string]any{
		"name":      "testkit-" + randHex(4),
		"directive": "Idle. Waiting for test directives.",
		"mode":      "autonomous",
		"project_id": p.ProjectID,
		// Skip the two system MCPs; tests that need them will enable
		// explicitly via the System MCPs UI equivalent API.
		"include_apteva_server": false,
		"include_channels":      false,
	})
	req, _ = http.NewRequest("POST", p.URL+"/api/instances", bytes.NewReader(instBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)
	var instResp struct {
		ID int64 `json:"id"`
	}
	if err := doExpectJSON(client, req, &instResp, "create instance"); err != nil {
		return err
	}
	p.InstanceID = instResp.ID
	return nil
}

func doExpect2xx(c *http.Client, req *http.Request, label string) error {
	resp, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: %d: %s", label, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func doExpectJSON(c *http.Client, req *http.Request, out any, label string) error {
	resp, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: %d: %s", label, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

// --- Stderr scraping for the setup token --------------------------------

var setupTokenRe = regexp.MustCompile(`Setup token:\s*(apt_[a-zA-Z0-9]+)`)

func scrapeSetupToken(stderr string) string {
	if m := setupTokenRe.FindStringSubmatch(stderr); len(m) >= 2 {
		return m[1]
	}
	// Best-effort line scan in case the format drifts.
	sc := bufio.NewScanner(strings.NewReader(stderr))
	for sc.Scan() {
		line := sc.Text()
		if strings.Contains(line, "apt_") {
			for _, tok := range strings.Fields(line) {
				if strings.HasPrefix(tok, "apt_") {
					return tok
				}
			}
		}
	}
	return ""
}

// --- Small utilities -----------------------------------------------------

func freePort() (int, error) {
	// Bind to :0 on the loopback, grab the allocated port, close the
	// listener immediately. There's a small race window but it's fine
	// for tests — nothing else on the box is racing for loopback
	// ports in the same millisecond.
	addr := "127.0.0.1:0"
	if runtime.GOOS == "windows" {
		addr = "localhost:0"
	}
	l, err := net.Listen("tcp", addr)
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func randHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// syncBuffer is an io.Writer that's safe to read concurrently with
// writes. Plain bytes.Buffer can race when cmd.Stderr is written from
// the child's goroutine while the test goroutine calls String().
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func newSyncBuffer() *syncBuffer { return &syncBuffer{} }

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
