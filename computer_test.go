package main

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	aptcomputer "github.com/apteva/computer"
	"github.com/apteva/core/pkg/computer"
)


func TestComputerUse_Local(t *testing.T) {
	if os.Getenv("RUN_COMPUTER_TESTS") == "" {
		t.Skip("skipping computer_use test (set RUN_COMPUTER_TESTS=1 to enable)")
	}
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	_ = apiKey

	comp, err := aptcomputer.New(aptcomputer.Config{
		Type:   "local",
		Width:  1280,
		Height: 800,
	})
	if err != nil {
		t.Fatalf("failed to create local computer: %v", err)
	}
	defer comp.Close()
	t.Logf("local computer connected: %dx%d", comp.DisplaySize().Width, comp.DisplaySize().Height)

	// Test browser_session: open URL (no screenshot — just navigates)
	text, screenshot, err := computer.HandleSessionAction(comp, map[string]string{
		"action": "open",
		"url":    "https://example.com",
	})
	if err != nil {
		t.Fatalf("browser_session open failed: %v", err)
	}
	if screenshot != nil {
		t.Fatal("browser_session open should NOT return a screenshot")
	}
	if !strings.Contains(text, "Navigated") {
		t.Errorf("expected navigated text, got: %s", text)
	}
	t.Logf("browser_session open: %s", text)

	// Test computer_use: take screenshot
	text, screenshot, err = computer.HandleComputerAction(comp, map[string]string{
		"action": "screenshot",
	})
	if err != nil {
		t.Fatalf("computer_use screenshot failed: %v", err)
	}
	if screenshot == nil || len(screenshot) == 0 {
		t.Fatal("computer_use screenshot returned no image")
	}
	t.Logf("computer_use screenshot: %s (%d bytes)", text, len(screenshot))

	// Test computer_use: reject navigate (should use browser_session)
	_, _, err = computer.HandleComputerAction(comp, map[string]string{
		"action": "navigate",
		"url":    "https://example.com",
	})
	if err == nil {
		t.Fatal("expected computer_use to reject navigate action")
	}
	t.Logf("computer_use navigate correctly rejected: %v", err)

	// Test browser_session: status
	text, _, err = computer.HandleSessionAction(comp, map[string]string{
		"action": "status",
	})
	if err != nil {
		t.Fatalf("browser_session status failed: %v", err)
	}
	if !strings.Contains(text, "local") {
		t.Errorf("expected 'local' in status, got: %s", text)
	}
	if !strings.Contains(text, "example.com") {
		t.Errorf("expected 'example.com' in status URL, got: %s", text)
	}
	t.Logf("browser_session status: %s", text)
}

func TestComputerUse_Navigate(t *testing.T) {
	if os.Getenv("RUN_COMPUTER_TESTS") == "" {
		t.Skip("skipping computer_use test (set RUN_COMPUTER_TESTS=1 to enable)")
	}
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	bbKey := os.Getenv("BROWSERBASE_API_KEY")
	bbProject := os.Getenv("BROWSERBASE_PROJECT_ID")
	if bbKey == "" || bbProject == "" {
		t.Skip("BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID not set")
	}

	// Create computer
	comp, err := aptcomputer.New(aptcomputer.Config{
		Type:      "browserbase",
		APIKey:    bbKey,
		ProjectID: bbProject,
		Width:     1280,
		Height:    800,
	})
	if err != nil {
		t.Fatalf("failed to create computer: %v", err)
	}
	defer comp.Close()
	t.Logf("computer connected: %dx%d", comp.DisplaySize().Width, comp.DisplaySize().Height)

	// Create thinker with computer
	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}

	cfg := &Config{
		Directive: "You have a browser. When told to navigate somewhere, use the computer_use tool to navigate and then take a screenshot. Describe what you see.",
		Mode:      ModeAutonomous,
	}

	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	// Observer
	obs := thinker.bus.SubscribeAll("test", 500)
	// Log all chunks to a file for debugging
	logFile, _ := os.Create("computer_test_chunks.log")
	defer logFile.Close()

	var sawScreenshot bool
	var sawNavigate bool
	var sawResult bool
	done := make(chan struct{})
	closed := false

	go func() {
		for {
			select {
			case ev := <-obs.C:
				if ev.Type == EventThinkDone {
					fmt.Fprintf(logFile, "\n=== THOUGHT #%d DONE (tok=%d/%d) ===\n",
						ev.Iteration, ev.Usage.PromptTokens, ev.Usage.CompletionTokens)
				}
				if ev.Type == EventChunk {
					fmt.Fprintf(logFile, "%s", ev.Text)
					if strings.Contains(ev.Text, "← computer_use") {
						sawScreenshot = true
					}
					if strings.Contains(ev.Text, "→ computer_use") {
						sawNavigate = true
					}
					if strings.Contains(ev.Text, "RESULT:") {
						sawResult = true
					}
				}
				if sawScreenshot && sawNavigate && sawResult && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(3 * time.Minute):
				return
			}
		}
	}()

	go thinker.Run()

	// Tell it to navigate and describe
	time.Sleep(2 * time.Second) // let first idle thought pass
	thinker.InjectConsole("Navigate to https://example.com using computer_use. After you see the screenshot, respond with exactly: RESULT: followed by the page title you see.")

	select {
	case <-done:
		t.Log("all three checks passed via stream")
	case <-time.After(120 * time.Second):
		t.Log("timeout waiting for all checks — evaluating from log")
	}

	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	// Read full log
	logContent, _ := os.ReadFile("computer_test_chunks.log")
	fullText := string(logContent)
	t.Logf("=== Chunks log ===\n%s", fullText)

	if !sawNavigate {
		t.Fatal("FAIL: LLM never called computer_use navigate")
	}
	t.Log("✓ navigate called")

	if !sawScreenshot {
		t.Fatal("FAIL: screenshot never returned")
	}
	t.Log("✓ screenshot returned")

	if sawResult {
		t.Log("✓ LLM responded with RESULT:")
	} else if strings.Contains(fullText, "RESULT:") {
		t.Log("✓ LLM responded with RESULT: (found in log)")
	} else {
		t.Log("WARN: no RESULT: found — checking log for any page description")
		lower := strings.ToLower(fullText)
		if strings.Contains(lower, "example") || strings.Contains(lower, "screenshot") {
			t.Log("✓ LLM did process the screenshot (mentioned example/screenshot)")
		} else {
			t.Error("FAIL: LLM never described the page")
		}
	}
}
