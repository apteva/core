package core

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// TestComputerAppMCP_CodexSoMLabelSmoke is the Tier 3 app smoke: a real
// OpenAI Codex provider drives the Computer app through its HTTP MCP
// endpoint. It proves the app path returns screenshots as image tool
// results and that the model uses label=N, not guessed coordinates, to
// click Example Domain's Learn More link.
//
//	RUN_COMPUTER_APP_CODEX_SMOKE=1 OPENAI_CODEX_ACCESS_TOKEN=... APTEVA_HEADLESS_BROWSER=1 go test -run TestComputerAppMCP_CodexSoMLabelSmoke -timeout 5m .
func TestComputerAppMCP_CodexSoMLabelSmoke(t *testing.T) {
	if os.Getenv("RUN_COMPUTER_APP_CODEX_SMOKE") == "" {
		t.Skip("set RUN_COMPUTER_APP_CODEX_SMOKE=1 to run the Computer app Codex smoke")
	}
	token := os.Getenv("OPENAI_CODEX_ACCESS_TOKEN")
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set")
	}
	if testing.Short() {
		t.Skip("skipping Computer app Codex smoke in short mode")
	}
	resetPersistentAgentState(t)

	appURL, stop := startComputerAppSidecar(t)
	defer stop()

	provider := NewOpenAICodexProvider(token)
	apiKey := firstNonEmptyEnv("FIREWORKS_API_KEY", "OPENAI_API_KEY")
	cfg := &Config{
		Directive: "You have the Computer app MCP server attached. Use it to operate the browser. Do not spawn workers.",
		Mode:      ModeAutonomous,
		MCPServers: []MCPServerConfig{{
			Name:      "computer",
			Transport: "http",
			URL:       appURL + "/mcp",
		}},
	}
	thinker := NewThinker(apiKey, provider, cfg)
	defer thinker.Stop()

	// This test targets the real Computer app tool path, not search
	// quality. Preload the exact app tools so Codex sees their schemas
	// from turn 1 while still dispatching through MCP.
	for _, name := range []string{"computer_browser_session", "computer_computer_use", "computer_browser_close"} {
		thinker.activeTools[name] = true
	}

	obs := thinker.bus.SubscribeAll("test-computer-app-codex", 1000)
	var chunks strings.Builder
	done := make(chan struct{})
	var sawOpen, sawScreenshot, sawLabelClick, sawCoordinateClick, sawIANA, sawResult bool
	clickWithLabel := regexp.MustCompile(`computer_computer_use\([^)]*action=click[^)]*label=|computer_computer_use\([^)]*label=[^)]*action=click`)
	clickWithCoordinate := regexp.MustCompile(`computer_computer_use\([^)]*action=click[^)]*coordinate=|computer_computer_use\([^)]*coordinate=[^)]*action=click`)

	go func() {
		for {
			select {
			case ev := <-obs.C:
				if ev.Type != EventChunk {
					continue
				}
				chunks.WriteString(ev.Text)
				s := chunks.String()
				sawOpen = sawOpen || strings.Contains(s, "computer_browser_session")
				sawScreenshot = sawScreenshot || strings.Contains(s, "computer_computer_use") && strings.Contains(s, "action=screenshot")
				sawLabelClick = sawLabelClick || clickWithLabel.MatchString(s)
				sawCoordinateClick = sawCoordinateClick || clickWithCoordinate.MatchString(s)
				sawIANA = sawIANA || strings.Contains(strings.ToLower(s), "iana.org")
				sawResult = sawResult || strings.Contains(s, "RESULT:")
				if sawOpen && sawScreenshot && sawLabelClick && sawIANA && sawResult {
					close(done)
					return
				}
			case <-time.After(4 * time.Minute):
				return
			}
		}
	}()

	go thinker.Run()
	thinker.InjectConsole(strings.Join([]string{
		`Open https://example.com with computer_browser_session(action="open", url="https://example.com").`,
		`Then call computer_computer_use(action="screenshot") and inspect the attached image.`,
		`The screenshot has numbered Set-of-Mark badges on clickable elements.`,
		`Click the Learn More link using computer_computer_use(action="click", label=N) with the visible badge number. Do not use coordinate for this click.`,
		`After the click, if the current URL contains iana.org, reply exactly: RESULT: label click worked`,
		`Do not call pace. Do not spawn.`,
	}, "\n"))

	select {
	case <-done:
	case <-time.After(180 * time.Second):
		t.Log("timeout waiting for Computer app Codex smoke; evaluating captured chunks")
	}

	full := chunks.String()
	t.Logf("=== chunks ===\n%s", full)
	if !sawOpen {
		t.Fatal("model never called computer_browser_session")
	}
	if !sawScreenshot {
		t.Fatal("model never called computer_computer_use action=screenshot")
	}
	if !sawLabelClick {
		t.Fatal("model never clicked with label=N")
	}
	if sawCoordinateClick {
		t.Fatal("model clicked with coordinate= despite label-badged screenshot")
	}
	if !sawIANA {
		t.Fatal("tool results never showed navigation to iana.org")
	}
	if !sawResult {
		t.Fatal("model did not reply with RESULT")
	}
}

func startComputerAppSidecar(t *testing.T) (string, func()) {
	t.Helper()
	appDir := computerAppDir(t)
	port := freeTCPPort(t)
	url := fmt.Sprintf("http://127.0.0.1:%d", port)

	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "run", ".")
	cmd.Dir = appDir
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("APTEVA_APP_PORT=%d", port),
		"APTEVA_APP_TOKEN=test-computer-app-codex",
		"APTEVA_PROJECT_ID=test-computer-app-codex",
		"APTEVA_HEADLESS_BROWSER=1",
		"DB_PATH="+filepath.Join(t.TempDir(), "computer.db"),
	)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start computer app sidecar: %v", err)
	}
	stop := func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		if t.Failed() {
			t.Logf("computer app stdout:\n%s", stdout.String())
			t.Logf("computer app stderr:\n%s", stderr.String())
		}
	}

	deadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url + "/health")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return url, stop
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	stop()
	t.Fatalf("computer app sidecar did not become healthy at %s", url)
	return "", func() {}
}

func computerAppDir(t *testing.T) string {
	t.Helper()
	if dir := strings.TrimSpace(os.Getenv("COMPUTER_APP_DIR")); dir != "" {
		if st, err := os.Stat(dir); err == nil && st.IsDir() {
			return dir
		}
		t.Fatalf("COMPUTER_APP_DIR %q is not a directory", dir)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for _, rel := range []string{
		"../apps/mcp/computer",
		"../apteva-apps-computer-release/mcp/computer",
	} {
		dir := filepath.Clean(filepath.Join(wd, rel))
		if st, err := os.Stat(dir); err == nil && st.IsDir() {
			return dir
		}
	}
	t.Fatal("computer app directory not found; set COMPUTER_APP_DIR")
	return ""
}

func freeTCPPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("pick free port: %v", err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}
