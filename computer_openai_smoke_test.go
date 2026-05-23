package core

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// TestOpenAIComputerUseSmoke exercises the OpenAI/Codex computer paths against
// a real browser and real model. It intentionally supports two modes:
//
//   - APTEVA_OPENAI_COMPUTER_MODE=native (default): Responses API computer tool.
//   - APTEVA_OPENAI_COMPUTER_MODE=custom: our computer_use function tool returns
//     screenshots as image tool outputs, proving the vision fallback path.
//
// Runs:
//
//	RUN_OPENAI_COMPUTER_SMOKE=1 OPENAI_COMPUTER_PROVIDER=codex \
//	OPENAI_CODEX_ACCESS_TOKEN=... APTEVA_HEADLESS_BROWSER=1 \
//	go test -v -run TestOpenAIComputerUseSmoke -timeout 4m ./
//
// Or with the public OpenAI API:
//
//	RUN_OPENAI_COMPUTER_SMOKE=1 OPENAI_COMPUTER_PROVIDER=openai \
//	OPENAI_API_KEY=... APTEVA_OPENAI_COMPUTER_MODE=custom \
//	go test -v -run TestOpenAIComputerUseSmoke -timeout 4m ./
func TestOpenAIComputerUseSmoke(t *testing.T) {
	if os.Getenv("RUN_OPENAI_COMPUTER_SMOKE") == "" {
		t.Skip("set RUN_OPENAI_COMPUTER_SMOKE=1 to run the OpenAI/Codex computer smoke")
	}
	if testing.Short() {
		t.Skip("skipping OpenAI/Codex computer smoke in short mode")
	}
	loadIntegrationEnv()
	resetPersistentAgentState(t)

	apiKey, provider, source := openAIComputerSmokeProvider(t)
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("APTEVA_OPENAI_COMPUTER_MODE")))
	if mode == "" {
		mode = "native"
	}
	t.Logf("provider=%s mode=%s model_large=%s", source, mode, provider.Models()[ModelLarge])

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			_ = comp.Close()
		}
	}()
	t.Logf("browser connected: backend=%s display=%dx%d", backendName(t), comp.DisplaySize().Width, comp.DisplaySize().Height)

	cfg := &Config{
		Directive: "You have a browser. Follow console instructions exactly. Do not spawn workers.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	thinker.InjectConsole(strings.Join([]string{
		`Open https://example.com using browser_session(action=open, url="https://example.com").`,
		`Then inspect the page with computer_use(action=screenshot).`,
		`Use the screenshot image as ground truth. If you can read the page title/heading, reply in normal text with exactly: RESULT: Example Domain`,
		`Do not call pace. Do not spawn. Do not stop after opening the session; you must inspect the screenshot and reply with RESULT.`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-openai-computer", 1000)
	logFile, err := os.Create("computer_openai_smoke_chunks.log")
	if err != nil {
		t.Fatalf("create log: %v", err)
	}
	defer logFile.Close()

	var sawBrowserSession, sawComputerUse, sawScreenshot, sawResult, sawExample bool
	done := make(chan struct{})
	closed := false
	var buf strings.Builder

	go func() {
		for {
			select {
			case ev := <-obs.C:
				if ev.Type == EventThinkDone {
					fmt.Fprintf(logFile, "\n=== THOUGHT #%d DONE (tok=%d/%d cached=%d) ===\n",
						ev.Iteration, ev.Usage.PromptTokens, ev.Usage.CompletionTokens, ev.Usage.CachedTokens)
				}
				if ev.Type == EventChunk {
					fmt.Fprint(logFile, ev.Text)
					buf.WriteString(ev.Text)
					s := buf.String()
					if strings.Contains(s, "→ browser_session") || strings.Contains(s, "← browser_session") {
						sawBrowserSession = true
					}
					if strings.Contains(s, "→ computer_use") {
						sawComputerUse = true
					}
					if strings.Contains(s, "← computer_use: screenshot") {
						sawScreenshot = true
					}
					if strings.Contains(s, "RESULT:") {
						sawResult = true
					}
					if strings.Contains(strings.ToLower(s), "example domain") {
						sawExample = true
					}
				}
				if sawBrowserSession && sawComputerUse && sawScreenshot && sawResult && sawExample && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(4 * time.Minute):
				return
			}
		}
	}()

	go thinker.Run()

	select {
	case <-done:
		t.Log("OpenAI/Codex computer smoke completed via stream")
	case <-time.After(180 * time.Second):
		t.Log("timeout waiting for OpenAI/Codex computer smoke — evaluating captured stream")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(300 * time.Millisecond)
	_ = logFile.Sync()
	logContent, _ := os.ReadFile("computer_openai_smoke_chunks.log")
	fullText := string(logContent)
	t.Logf("=== Chunks log ===\n%s", fullText)
	t.Logf("=== Final URL: %s", finalURL)

	if !sawBrowserSession {
		t.Fatal("FAIL: model never opened a browser_session")
	}
	if !strings.Contains(finalURL, "example.com") {
		t.Fatalf("FAIL: final URL %q, expected example.com", finalURL)
	}
	if !sawComputerUse {
		t.Fatal("FAIL: model never requested computer_use/native computer action")
	}
	if !sawScreenshot {
		t.Fatal("FAIL: screenshot result never returned to the model")
	}
	if !(sawResult && sawExample) {
		t.Fatalf("FAIL: model did not prove image understanding with RESULT: Example Domain")
	}

	if err := comp.Close(); err != nil {
		t.Errorf("comp.Close() returned error: %v", err)
	}
	comp = nil
}

func openAIComputerSmokeProvider(t *testing.T) (string, LLMProvider, string) {
	t.Helper()
	switch strings.ToLower(strings.TrimSpace(os.Getenv("OPENAI_COMPUTER_PROVIDER"))) {
	case "openai":
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			t.Skip("OPENAI_API_KEY not set")
		}
		return key, NewOpenAINativeProvider(key), "openai"
	case "codex", "openai-codex", "":
		if token := os.Getenv("OPENAI_CODEX_ACCESS_TOKEN"); token != "" {
			return firstNonEmptyEnv("FIREWORKS_API_KEY", "OPENAI_API_KEY"), NewOpenAICodexProvider(token), "openai-codex"
		}
		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return key, NewOpenAINativeProvider(key), "openai"
		}
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN or OPENAI_API_KEY not set")
	default:
		t.Fatalf("OPENAI_COMPUTER_PROVIDER must be codex or openai")
	}
	return "", nil, ""
}
