package core

import (
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

)

// TestComputerUse_PatreonLogin drives a real Patreon login end-to-end.
// Patreon emails a 6-digit verification code after the email-submit
// step; the test pauses when the agent announces "AWAITING_CODE" and
// resumes once you provide the code via HTTP, file, or env var.
//
// Backends (env TEST_BROWSER, default "local"):
//
//	local        → local Chrome (often blocked by Patreon's bot
//	               detection; useful for harness debugging only)
//	browserbase  → cloud Chrome via Browserbase (recommended;
//	               requires BROWSERBASE_API_KEY + BROWSERBASE_PROJECT_ID)
//	steel        → cloud Chrome via Steel (requires STEEL_API_KEY)
//
// Provide the login code mid-test via any of:
//
//	curl -d 123456 $PATREON_CODE_URL      # printed on test stderr
//	echo 123456 > /tmp/patreon_code.txt   # file fallback
//	PATREON_CODE=123456 go test ...       # baked in at start
//
//	RUN_PATREON_TEST=1 PATREON_EMAIL=you@example.com \
//	FIREWORKS_API_KEY=fw_... APTEVA_HEADLESS_BROWSER=1 \
//	TEST_BROWSER=browserbase BROWSERBASE_API_KEY=... BROWSERBASE_PROJECT_ID=... \
//	go test -v -run TestComputerUse_PatreonLogin -timeout 15m ./
func TestComputerUse_PatreonLogin(t *testing.T) {
	if os.Getenv("RUN_PATREON_TEST") == "" {
		t.Skip("set RUN_PATREON_TEST=1 (real Patreon login — interactive, burns tokens)")
	}
	tp := getTestProvider(t)
	apiKey := tp.APIKey
	email := os.Getenv("PATREON_EMAIL")
	if email == "" {
		t.Skip("PATREON_EMAIL not set")
	}
	password := os.Getenv("PATREON_PASSWORD")
	if password == "" {
		t.Skip("PATREON_PASSWORD not set")
	}

	// Proxy is the agent's decision now — it sees the capability in
	// the browser_session tool description and gets a Patreon-specific
	// hint in the directive below. The harness no longer pre-flips
	// TEST_BROWSER_PROXY; this run is a proper test of agent-owned
	// policy. Operators can still force it on with PATREON_USE_PROXY=1.
	if os.Getenv("PATREON_USE_PROXY") == "1" {
		t.Setenv("TEST_BROWSER_PROXY", "1")
	}
	// Patreon's email-code flow can take 4–10 min; the cloud
	// backends' default lease (300s on Browser Engine, plan-default
	// on others) often expires mid-flow. Browserbase + Steel can't
	// extend post-create, so we set a generous lease at creation
	// for every cloud backend. Override with PATREON_SESSION_TIMEOUT.
	if os.Getenv("TEST_BROWSER_SESSION_TIMEOUT") == "" {
		secs := "1200"
		if v := os.Getenv("PATREON_SESSION_TIMEOUT"); v != "" {
			secs = v
		}
		t.Setenv("TEST_BROWSER_SESSION_TIMEOUT", secs)
	}

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			comp.Close()
		}
	}()
	t.Logf("computer connected: %dx%d backend=%s",
		comp.DisplaySize().Width, comp.DisplaySize().Height, backendName(t))

	// Code-intake server: random loopback port. The test prints the URL
	// so the operator can POST the emailed code mid-run. Also watches a
	// fallback file and accepts a pre-baked PATREON_CODE env var.
	codeCh := make(chan string, 1)
	if pre := os.Getenv("PATREON_CODE"); pre != "" {
		codeCh <- strings.TrimSpace(pre)
	}
	srvURL, stopSrv := startCodeServer(t, codeCh)
	defer stopSrv()
	codeFile := os.Getenv("PATREON_CODE_FILE")
	if codeFile == "" {
		codeFile = "/tmp/patreon_code.txt"
	}
	stopWatch := watchCodeFile(t, codeFile, codeCh)
	defer stopWatch()

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	cfg := &Config{
		Directive: "You have a browser with Set-of-Mark grounding — interactive elements have colored numeric badges. Prefer label=N for clicks.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	// Prompt structure: two phases. PHASE A is mandatory and must run
	// in order; pace() is forbidden during it. PHASE B only runs if
	// PHASE A's final screenshot shows a 6-digit code-entry input —
	// otherwise the agent reports RESULT and stops without ever
	// invoking pace.
	//
	// Why the explicit two-phase / OUTCOMES split: a previous prompt
	// described the 2FA path as a conditional final step ("if you see
	// a code-entry screen, output AWAITING_CODE then pace(1h)"). Some
	// models (notably Kimi K2.6 via OpenCode-Go) read that as the
	// FIRST instruction and called pace immediately, skipping all the
	// actual login work. Naming OUTCOMES at the post-submit decision
	// point + a top-level "do NOT pace during PHASE A" rule kills
	// that failure mode.
	thinker.InjectConsole(strings.Join([]string{
		`Log in to Patreon. Do not invent a verification code.`,
		``,
		fmt.Sprintf(`Email:    %s`, email),
		fmt.Sprintf(`Password: %s`, password),
		``,
		`This task has two phases. PHASE A is mandatory and must be`,
		`completed in order before doing anything else. PHASE B only`,
		`runs if Patreon prompts for an emailed verification code AFTER`,
		`you submit the login form. Do NOT call pace() during PHASE A.`,
		``,
		`═══ PHASE A — login (always run, in order) ═══`,
		``,
		`A1) browser_session(action=open, url=https://www.patreon.com/login, timeout=1200, proxy=true)`,
		`    timeout=1200 keeps the session alive long enough for PHASE B`,
		`    if it triggers; proxy=true is critical because Patreon`,
		`    fingerprints datacenter IPs and serves Cloudflare to them.`,
		``,
		`A2) computer_use(action=screenshot). If a cookie/consent banner`,
		`    covers the page, click Accept by label first, then`,
		`    screenshot again.`,
		``,
		`A3) Find the email input (orange badge). Click it by label,`,
		`    then computer_use(action=type, text="<the email above>").`,
		``,
		`A4) Click the Continue/Log-in button (green badge) by label.`,
		`    The page should advance to a password field — or directly`,
		`    to a logged-in dashboard if Patreon trusts this session.`,
		``,
		`A5) If a password field is now visible: click it by label,`,
		`    type the password, click the Log-in/Continue button.`,
		`    If no password field appeared, skip A5.`,
		``,
		`A6) Screenshot ONCE. Look at the page and pick exactly one`,
		`    OUTCOME:`,
		``,
		`    OUTCOME 1 — landed on the logged-in dashboard`,
		`                (home feed / your avatar / /home / /c/...).`,
		`                → Reply RESULT: logged_in on its own line.`,
		`                STOP. Do not call pace, do not navigate.`,
		``,
		`    OUTCOME 2 — page shows a 6-digit code-entry input`,
		`                (Patreon mailed a verification code).`,
		`                → Output the literal token AWAITING_CODE on`,
		`                its own line, then proceed to PHASE B.`,
		``,
		`    OUTCOME 3 — anything else (error, captcha, gated page).`,
		`                → Reply RESULT: failed: <one-sentence reason>`,
		`                on its own line. STOP.`,
		``,
		`═══ PHASE B — only after AWAITING_CODE ═══`,
		``,
		`B1) Call pace(1h). The harness will inject a console message`,
		`    "CODE: 123456" when the operator forwards it.`,
		``,
		`B2) On receiving CODE, click the code-input field by label,`,
		`    type the 6 digits, click submit.`,
		``,
		`B3) Screenshot. If logged in, reply RESULT: logged_in on its`,
		`    own line. Otherwise reply RESULT: failed: code rejected.`,
		``,
		`═══ Rules ═══`,
		``,
		`• Use label= for every click, never coordinate.`,
		`• Do NOT call pace() unless you have just emitted AWAITING_CODE.`,
		`• Do NOT invent or guess a verification code; wait for the`,
		`  injected one.`,
		`• Exactly one final RESULT: line at the end.`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-patreon", 2000)
	logFile, _ := os.Create("computer_test_patreon_chunks.log")
	defer logFile.Close()

	// Agent progress tracking. awaitingAt pins the offset where
	// "AWAITING_CODE" first appears in the post-prompt stream, so we
	// don't trigger on the prompt echoing it back to us.
	var sawAwaiting, codeInjected, loggedIn, cheated bool
	promptLen := 0 // snapshot len after initial prompt; set below
	done := make(chan struct{})
	closed := false
	var buf strings.Builder
	var bufMu sync.Mutex

	// One-shot code injector: drains codeCh, injects "CODE: X" into the
	// console, logs the action. Run in a goroutine so the observer
	// can fire it without blocking the event loop.
	injectCode := func() {
		select {
		case code := <-codeCh:
			code = strings.TrimSpace(code)
			if len(code) == 0 {
				return
			}
			t.Logf("[PATREON_CODE] received code (%d chars) — injecting", len(code))
			thinker.InjectConsole(fmt.Sprintf("CODE: %s\n\nThe verification code from your email is: %s. Click the code input field and type these digits, then submit.", code, code))
			codeInjected = true
		case <-time.After(10 * time.Minute):
			t.Logf("[PATREON_CODE] timed out waiting for code after 10 min")
		}
	}

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
					bufMu.Lock()
					buf.WriteString(ev.Text)
					s := buf.String()
					bufMu.Unlock()

					// AWAITING_CODE: only count occurrences past the end
					// of our own prompt — the prompt itself mentions the
					// token.
					if !sawAwaiting && strings.Contains(s[promptLen:], "AWAITING_CODE") {
						sawAwaiting = true
						t.Logf("[PATREON_CODE] agent emitted AWAITING_CODE — blocking for code")
						t.Logf("[PATREON_CODE] provide code via:")
						t.Logf("[PATREON_CODE]   curl -d 123456 %s/code", srvURL)
						t.Logf("[PATREON_CODE]   echo 123456 > %s", codeFile)
						go injectCode()
					}
					if !cheated && sawAwaiting && codeInjected &&
						strings.Contains(s[promptLen:], "AWAITING_CODE") {
						// Second AWAITING_CODE after we fed one in =
						// wrong code. Surface it.
						post := s[promptLen:]
						if strings.Count(post, "AWAITING_CODE") >= 2 {
							cheated = true // repurpose: "rejected"
						}
					}
					if strings.Contains(s[promptLen:], "RESULT: logged_in") {
						loggedIn = true
					}
				}
				if loggedIn && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(15 * time.Minute):
				return
			}
		}
	}()

	// Snapshot prompt length once the initial thought starts streaming
	// so AWAITING_CODE detection skips the prompt echo.
	bufMu.Lock()
	promptLen = buf.Len()
	bufMu.Unlock()

	go thinker.Run()

	select {
	case <-done:
		t.Log("login completed — stopping agent")
	case <-time.After(13 * time.Minute):
		t.Log("timeout — evaluating final state")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	logContent, _ := os.ReadFile("computer_test_patreon_chunks.log")
	t.Logf("=== Chunks log ===\n%s", string(logContent))
	t.Logf("=== Final URL: %s", finalURL)

	// Two valid happy paths:
	//   1. Direct login — Patreon accepted email+password without prompting
	//      for an emailed code (residential/trusted IP, prior good session).
	//      sawAwaiting=false, loggedIn=true.
	//   2. 2FA path — Patreon mailed a code, the agent paused, the code
	//      was injected, login completed. sawAwaiting=true, codeInjected=true,
	//      cheated=false, loggedIn=true.
	// Anything else is a failure.
	switch {
	case loggedIn && !sawAwaiting:
		t.Log("✓ logged in directly (no email-code challenge from Patreon for this IP/session)")
		t.Log("✓ RESULT: logged_in observed — Patreon login flow passed end-to-end (no 2FA)")
		return
	case !loggedIn && !sawAwaiting:
		t.Fatalf("FAIL: agent neither reached the code-entry step nor logged in — form likely errored before submit (final URL %q)", finalURL)
	}
	// 2FA path from here on.
	t.Log("✓ agent reached the code-entry step")
	if !codeInjected {
		t.Fatal("FAIL: no verification code was provided within the window — nothing to verify")
	}
	t.Log("✓ verification code injected")
	if cheated {
		t.Fatal("FAIL: agent emitted AWAITING_CODE again after the code was provided — Patreon rejected the code (wrong/expired) or agent mis-typed")
	}
	if !loggedIn {
		t.Fatalf("FAIL: agent never emitted RESULT: logged_in (final URL %q)", finalURL)
	}
	t.Log("✓ RESULT: logged_in observed — Patreon login flow passed end-to-end (with 2FA)")
}

// TestComputerUse_PatreonPublishTextPost extends the login flow with
// a full compose-and-publish pass: log in, reach the text-post
// composer, type a clearly-labelled "automated test" title and body,
// click publish, and verify the page lands on a published-post URL.
//
// Side effect: writes ONE post to the configured Patreon account.
// The post body identifies itself outright as an automated agent
// test so anyone who sees it knows. The post is published with
// whatever default visibility Patreon offers (typically Public);
// delete it manually after the run if you don't want it lingering.
//
//	RUN_PATREON_POST_TEST=1 PATREON_EMAIL=you@example.com \
//	PATREON_PASSWORD=… OPENCODE_GO_API_KEY=sk-… \
//	go test -v -run TestComputerUse_PatreonPublishTextPost -timeout 20m ./
func TestComputerUse_PatreonPublishTextPost(t *testing.T) {
	if os.Getenv("RUN_PATREON_POST_TEST") == "" {
		t.Skip("set RUN_PATREON_POST_TEST=1 (real Patreon login + publishes a post — interactive, burns tokens, leaves a real post on the account)")
	}
	tp := getTestProvider(t)
	apiKey := tp.APIKey
	email := os.Getenv("PATREON_EMAIL")
	if email == "" {
		t.Skip("PATREON_EMAIL not set")
	}
	password := os.Getenv("PATREON_PASSWORD")
	if password == "" {
		t.Skip("PATREON_PASSWORD not set")
	}
	if os.Getenv("PATREON_USE_PROXY") == "1" {
		t.Setenv("TEST_BROWSER_PROXY", "1")
	}
	if os.Getenv("TEST_BROWSER_SESSION_TIMEOUT") == "" {
		secs := "1500" // 25 min — login + 2FA window + post compose
		if v := os.Getenv("PATREON_SESSION_TIMEOUT"); v != "" {
			secs = v
		}
		t.Setenv("TEST_BROWSER_SESSION_TIMEOUT", secs)
	}

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			comp.Close()
		}
	}()
	t.Logf("computer connected: %dx%d backend=%s",
		comp.DisplaySize().Width, comp.DisplaySize().Height, backendName(t))

	// Same code-intake plumbing as the login test — the post test
	// hits the login flow first, so 2FA may still fire for it.
	codeCh := make(chan string, 1)
	if pre := os.Getenv("PATREON_CODE"); pre != "" {
		codeCh <- strings.TrimSpace(pre)
	}
	srvURL, stopSrv := startCodeServer(t, codeCh)
	defer stopSrv()
	codeFile := os.Getenv("PATREON_CODE_FILE")
	if codeFile == "" {
		codeFile = "/tmp/patreon_code.txt"
	}
	stopWatch := watchCodeFile(t, codeFile, codeCh)
	defer stopWatch()

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	cfg := &Config{
		Directive: "You have a browser with Set-of-Mark grounding — interactive elements have colored numeric badges. Prefer label=N for clicks.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	// Today's date stamps the title and body so a human (or future
	// cleanup script) can grep the account for automated test posts
	// and prune them.
	today := time.Now().Format("2006-01-02")
	postTitle := fmt.Sprintf("[automated test] apteva agent — %s", today)
	postBody := fmt.Sprintf(
		"This post was created by an automated apteva agent end-to-end "+
			"test on %s. It contains no real content and is safe to "+
			"delete.", today)

	// Reuse the proven PHASE A/B login prompt, plus PHASE C (reach
	// composer), D (compose + publish), E (verify + report).
	//
	// Key prompt guards (history of failures we're avoiding):
	//   - "MANDATORY FIRST ACTION" at the very top, naming the exact
	//     first tool call. Without this, Kimi K2.6 has been observed
	//     reading the long multi-phase prompt and immediately doing
	//     the first imperative it sees (e.g. "Click title by label"
	//     in D1 — on about:blank, with no login attempt at all).
	//   - "Do NOT call pace() during PHASES A/C/D/E" — Kimi pace'd
	//     immediately on the original prompt seeing the "pace(1h)"
	//     mention in the 2FA branch.
	//   - PHASE D/E phrased as "WHEN you reach this phase, do X"
	//     rather than imperative "Click X" to avoid the same
	//     skip-ahead failure mode.
	thinker.InjectConsole(strings.Join([]string{
		`═══ MANDATORY FIRST ACTION ═══`,
		``,
		`Your very first tool call MUST be:`,
		``,
		`  browser_session(action=open, url=https://www.patreon.com/login, timeout=1500, proxy=true)`,
		``,
		`The browser is currently on about:blank with NO labels and`,
		`NO Patreon page loaded. Do NOT try to click, screenshot, or`,
		`type before opening Patreon. Do NOT pace before opening.`,
		`Open Patreon FIRST, then proceed to PHASE A2.`,
		``,
		`═══ Goal ═══`,
		``,
		`Log into Patreon, open the text-post composer, type a clearly-`,
		`labelled automated-test title and body, then publish the post.`,
		`Do not invent a verification code.`,
		``,
		fmt.Sprintf(`Email:    %s`, email),
		fmt.Sprintf(`Password: %s`, password),
		``,
		`This task has FIVE phases that MUST be executed in order:`,
		`A → B (only if needed) → C → D → E.`,
		``,
		`Do NOT call pace() during PHASES A, C, D, or E. pace() is only`,
		`valid in PHASE B after AWAITING_CODE has been emitted.`,
		``,
		`═══ Background — working with SaaS web composers ═══`,
		``,
		`Three traps that consistently break agent flows in editors`,
		`(Patreon, Twitter, Substack, ...). Internalise these BEFORE`,
		`PHASE C/D so you don't repeat the same mistakes:`,
		``,
		`/new allocates state — visit it only once. URLs like`,
		`/posts/new, /compose, /create create a draft on the server`,
		`before the editor renders, then redirect to /posts/<id>/edit.`,
		`Each visit spawns a duplicate draft. If the editor seems`,
		`broken, do NOT navigate to /new again as a reset — recover`,
		`in-place by pressing Escape and re-clicking the field.`,
		``,
		`Inline pickers steal focus from the body. Most rich-text`,
		`editors expose a "+" or "Add ..." button INSIDE the empty`,
		`body area. Clicking it opens a content-type picker menu`,
		`instead of focusing the text editor — telltale signs are an`,
		`icon-only or "Click to add" label, and a popover appearing`,
		`instead of a cursor. Press Escape to dismiss, then click a`,
		`blank area inside the body region to focus the actual text`,
		`editor before typing.`,
		``,
		`Publish is usually two clicks. The first Publish click opens`,
		`a confirmation modal (visibility, tiers, schedule). The post`,
		`is NOT published until you click Publish/Confirm/Post inside`,
		`that modal. Tell you only did click 1: URL didn't change.`,
		`After publish, dismiss any share/success modal before reading`,
		`the URL — until then, screenshots and clicks target the`,
		`modal, not the post.`,
		``,
		`Before reporting "done": URL is off /edit, any post-publish`,
		`modal is dismissed, the public post URL is visible.`,
		``,
		`═══ PHASE A — login (always run, in order) ═══`,
		``,
		`A1) browser_session(action=open, url=https://www.patreon.com/login, timeout=1500, proxy=true)`,
		``,
		`A2) computer_use(action=screenshot). If a cookie/consent banner`,
		`    covers the page, click Accept by label first, then`,
		`    screenshot again.`,
		``,
		`A3) Find the email input (orange badge). Click it by label,`,
		`    then computer_use(action=type, text="<the email above>").`,
		``,
		`A4) Click Continue/Log-in (green badge) by label. Page advances`,
		`    to a password field — or directly to a logged-in dashboard`,
		`    if Patreon trusts this session.`,
		``,
		`A5) If a password field appears: click it by label, type the`,
		`    password, click Log-in/Continue. If no password field, skip A5.`,
		``,
		`A6) Screenshot ONCE. Pick exactly one OUTCOME:`,
		``,
		`    OUTCOME 1 — landed on logged-in dashboard. Proceed to PHASE C.`,
		`    OUTCOME 2 — 6-digit code-entry input visible. Output the`,
		`                literal token AWAITING_CODE on its own line,`,
		`                then enter PHASE B.`,
		`    OUTCOME 3 — anything else. Reply RESULT: failed: <reason>`,
		`                on its own line. STOP.`,
		``,
		`═══ PHASE B — only after AWAITING_CODE ═══`,
		``,
		`B1) Call pace(1h). The harness will inject "CODE: 123456" when`,
		`    the operator forwards it.`,
		``,
		`B2) On receiving CODE, click the code-input field by label,`,
		`    type the 6 digits, click submit.`,
		``,
		`B3) Screenshot. If logged in, proceed to PHASE C. Otherwise`,
		`    reply RESULT: failed: code rejected and STOP.`,
		``,
		`═══ PHASE C — open the post composer ═══`,
		``,
		`C1) Navigate to the composer. Either click "Create" / "+" /`,
		`    "New post" from the dashboard and pick "Text" if a menu`,
		`    opens, OR call browser_session(action=open, url=https://www.patreon.com/posts/new).`,
		`    Direct nav is faster and equivalent.`,
		``,
		`C2) Screenshot. Confirm you're on the composer: URL contains`,
		`    "/posts/" and the page shows an empty title field plus a`,
		`    large editable content area. If after TWO attempts you`,
		`    still aren't on the composer, reply RESULT: failed:`,
		`    could not reach composer and STOP.`,
		``,
		`═══ PHASE D — compose and publish (only after PHASE C succeeds) ═══`,
		``,
		`Once you have confirmed the composer is loaded (PHASE C done):`,
		``,
		`D1) WHEN the composer is visible, locate the title input (large`,
		`    empty text field at the top, often with placeholder "Add a`,
		`    title"). Click it by label, then call computer_use with`,
		fmt.Sprintf(`    action=type and text=%q.`, postTitle),
		``,
		`D2) NEXT, locate the body / content area (large editable region`,
		`    below the title). Click it by label, then call computer_use`,
		fmt.Sprintf(`    with action=type and text=%q.`, postBody),
		``,
		`D3) AFTER both title and body are typed, locate the "Publish"`,
		`    button (often top-right; may be labelled "Publish",`,
		`    "Publish now", or have an arrow next to "Publish ▾").`,
		`    Click it by label.`,
		``,
		`D4) Patreon MAY then show a confirmation modal asking who can`,
		`    see the post (Public / Patrons / specific tiers). If it`,
		`    appears: prefer "Public" or "Everyone"; otherwise pick the`,
		`    pre-selected default. Then click the final "Publish" /`,
		`    "Confirm" / "Post" button. If no modal appears and the`,
		`    page navigates, skip D4.`,
		``,
		`D5) AFTER the publish click, call computer_use(action=wait) so`,
		`    the post can finish publishing. The URL should change to a`,
		`    published-post URL — typically "/posts/<numeric-id>" without`,
		`    "/edit", or remain at "/posts/<id>/edit" with a "Published"`,
		`    indicator visible.`,
		``,
		`═══ PHASE E — verify and report (only after PHASE D succeeds) ═══`,
		``,
		`E1) WHEN PHASE D's publish step has completed, take a final`,
		`    screenshot. If the URL contains "/posts/" AND the page`,
		`    shows clear evidence the post was published (a "Published"`,
		`    badge, the post visible inline, or a redirect to the`,
		`    public post URL), reply on its own line:`,
		`      RESULT: post_url=<the current full URL>`,
		``,
		`E2) Otherwise reply on its own line:`,
		`      RESULT: failed: <one-sentence reason describing what you saw>`,
		``,
		`═══ Rules ═══`,
		``,
		`• Use label= for every click, never coordinate.`,
		`• Do NOT call pace() unless you have just emitted AWAITING_CODE.`,
		`• Do NOT invent or guess a verification code.`,
		`• Publish exactly ONE post — do not create or publish anything else.`,
		`• Exactly one final RESULT: line at the end.`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-patreon-publish", 2000)
	logFile, _ := os.Create("computer_test_patreon_publish_chunks.log")
	defer logFile.Close()

	var sawAwaiting, published, failed bool
	var postURL, failReason string
	promptLen := 0
	done := make(chan struct{})
	closed := false
	var buf strings.Builder
	var bufMu sync.Mutex

	injectCode := func() {
		select {
		case code := <-codeCh:
			code = strings.TrimSpace(code)
			if len(code) == 0 {
				return
			}
			t.Logf("[PATREON_CODE] received code (%d chars) — injecting", len(code))
			thinker.InjectConsole(fmt.Sprintf("CODE: %s\n\nThe verification code from your email is: %s. Click the code input field and type these digits, then submit.", code, code))
		case <-time.After(10 * time.Minute):
			t.Logf("[PATREON_CODE] timed out waiting for code after 10 min")
		}
	}

	// Result-line parser: looks for "RESULT: post_url=..." (success)
	// or "RESULT: failed:..." (clean fail). Only matches text past
	// the prompt-end offset to avoid prompt-echo false positives.
	//
	// IMPORTANT: capture only when a terminating newline has already
	// been streamed past the marker. Streaming chunks can split
	// mid-URL (we hit this once: chunk #1 ended at "...=https", chunk
	// #2 brought the rest), and committing on the first match would
	// freeze the captured value at the chunk boundary. The newline
	// guarantees the entire RESULT line is in the buffer before we
	// commit and gate further parsing.
	parseResult := func(s string) {
		if published || failed {
			return
		}
		const okMarker = "RESULT: post_url="
		if i := strings.Index(s, okMarker); i >= 0 {
			rest := s[i+len(okMarker):]
			nl := strings.IndexAny(rest, "\r\n")
			if nl < 0 {
				return // wait for the line to finish streaming
			}
			postURL = strings.TrimSpace(rest[:nl])
			published = true
			return
		}
		const failMarker = "RESULT: failed:"
		if i := strings.Index(s, failMarker); i >= 0 {
			rest := s[i+len(failMarker):]
			nl := strings.IndexAny(rest, "\r\n")
			if nl < 0 {
				return
			}
			failReason = strings.TrimSpace(rest[:nl])
			failed = true
		}
	}

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
					bufMu.Lock()
					buf.WriteString(ev.Text)
					s := buf.String()
					bufMu.Unlock()

					if !sawAwaiting && strings.Contains(s[promptLen:], "AWAITING_CODE") {
						sawAwaiting = true
						t.Logf("[PATREON_CODE] agent emitted AWAITING_CODE — blocking for code")
						t.Logf("[PATREON_CODE] provide code via:")
						t.Logf("[PATREON_CODE]   curl -d 123456 %s/code", srvURL)
						t.Logf("[PATREON_CODE]   echo 123456 > %s", codeFile)
						go injectCode()
					}
					parseResult(s[promptLen:])
				}
				if (published || failed) && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(20 * time.Minute):
				return
			}
		}
	}()

	bufMu.Lock()
	promptLen = buf.Len()
	bufMu.Unlock()

	go thinker.Run()

	select {
	case <-done:
		t.Log("agent reported RESULT — stopping")
	case <-time.After(18 * time.Minute):
		t.Log("timeout — evaluating final state")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	logContent, _ := os.ReadFile("computer_test_patreon_publish_chunks.log")
	t.Logf("=== Chunks log ===\n%s", string(logContent))
	t.Logf("=== Final URL: %s", finalURL)

	if failed {
		t.Fatalf("FAIL: agent reported failure: %s (final URL %q)", failReason, finalURL)
	}

	// publishedURLLike — matches "/posts/<digits>" NOT followed by
	// /edit (which is the composer URL). The "?pr=true" Patreon
	// appends after a successful publish satisfies this; an in-flight
	// "?isLoading=true" or a draft "/posts/<id>/edit" doesn't.
	publishedURLLike := func(u string) bool {
		re := regexp.MustCompile(`/posts/\d+([?#/]|$)`)
		if !re.MatchString(u) {
			return false
		}
		return !strings.Contains(u, "/edit")
	}

	// Two valid success paths:
	//   1. Agent emitted "RESULT: post_url=<url>" — clean signal.
	//   2. Agent forgot to emit RESULT in plain text (saw this happen
	//      when the model tried to "report completion" via a tool call
	//      like done() or send(parent), neither of which is in our
	//      tool set), BUT the final live URL of the browser already
	//      points at a published-post URL. In that case the task
	//      succeeded; the agent just confused the reporting protocol.
	switch {
	case published && publishedURLLike(postURL):
		t.Logf("✓ post_url=%s (agent emitted RESULT)", postURL)
	case published && publishedURLLike(finalURL):
		t.Logf("✓ post_url=%s (agent emitted RESULT but reported draft URL; live URL %s confirms published)", postURL, finalURL)
		postURL = finalURL
	case !published && publishedURLLike(finalURL):
		t.Logf("✓ inferred success from final URL %s — agent never emitted RESULT: but the page is at the published post (likely reporting confusion, not real failure)", finalURL)
		postURL = finalURL
	default:
		t.Fatalf("FAIL: no success signal — agent emitted RESULT? %v (postURL=%q), final URL %q does not look like a published post", published, postURL, finalURL)
	}
	t.Log("✓ post URL is a published-post URL — text post published end-to-end")
	t.Logf("REMINDER: a real post was published on the Patreon account — delete it manually if you don't want it lingering. URL: %s", postURL)
}

// TestComputerUse_PatreonContextResume demonstrates the local
// backend's persistent-context feature with a real LLM driving the
// flow. Prompts the agent to:
//
//	1. Open Patreon WITH a context_id pointing at a known dir
//	2. Inspect the page and decide:
//	     - Already on /home (context resumed)   → emit RESULT: resumed=...
//	     - Stuck on /login (no session)         → log in fresh, then emit
//	                                              RESULT: logged_in_fresh=...
//	     - Anything else                        → RESULT: failed: <reason>
//
// First invocation will hit the login path; the context_id dir gets
// populated with cookies + localStorage on graceful Close. Second
// invocation (same context_id) takes the resumed path and finishes
// in seconds.
//
// To actually see both paths:
//
//	# Run 1 — fresh login (~3-5min)
//	RUN_PATREON_TEST=1 PATREON_EMAIL=… PATREON_PASSWORD=… \
//	  PATREON_CONTEXT_ID=alexa-resume \
//	  go test -v -run TestComputerUse_PatreonContextResume -timeout 8m ./
//
//	# Run 2 — resume (much faster, no login form)
//	(same command)
//
// Override PATREON_CONTEXT_ID to control where state lives. Default
// is "patreon-resume-test". The context dir is NEVER wiped by this
// test (the persistence is the whole point) — operator manages
// cleanup manually under ~/.apteva/local-contexts/.
func TestComputerUse_PatreonContextResume(t *testing.T) {
	if os.Getenv("RUN_PATREON_TEST") == "" {
		t.Skip("set RUN_PATREON_TEST=1 (real Patreon login — interactive)")
	}
	tp := getTestProvider(t)
	apiKey := tp.APIKey
	email := os.Getenv("PATREON_EMAIL")
	password := os.Getenv("PATREON_PASSWORD")
	if email == "" || password == "" {
		t.Skip("PATREON_EMAIL + PATREON_PASSWORD required")
	}
	ctxID := os.Getenv("PATREON_CONTEXT_ID")
	if ctxID == "" {
		ctxID = "patreon-resume-test"
	}
	t.Logf("using context_id=%q (state lives at ~/.apteva/local-contexts/%s/ unless APTEVA_LOCAL_CONTEXT_DIR overrides)", ctxID, ctxID)

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			comp.Close()
		}
	}()
	t.Logf("computer connected: %dx%d backend=%s", comp.DisplaySize().Width, comp.DisplaySize().Height, backendName(t))

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	cfg := &Config{
		Directive: "You have a browser with Set-of-Mark grounding — interactive elements have colored numeric badges. Prefer label=N for clicks.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	// Adaptive prompt: handles BOTH first-run-login and
	// already-resumed paths in one flow. The MANDATORY FIRST ACTION
	// uses context_id and lands on /home (logged-in surface) — if
	// the context already has a session, Patreon serves the page
	// directly; otherwise it redirects to /login and we fall through
	// to the form-fill phase.
	thinker.InjectConsole(strings.Join([]string{
		`═══ MANDATORY FIRST ACTION ═══`,
		``,
		`Your very first tool call MUST be:`,
		``,
		fmt.Sprintf(`  browser_session(action=open, context_id=%q, url=https://www.patreon.com/home, timeout=300)`, ctxID),
		``,
		`Do NOT click, screenshot, type, or pace before opening Patreon.`,
		`Open Patreon FIRST with the context_id above, then proceed.`,
		``,
		`═══ Goal ═══`,
		``,
		`Detect whether the local persistent context already holds a`,
		`logged-in Patreon session. If yes, report it and stop. If no,`,
		`log in once so the next run can resume.`,
		``,
		fmt.Sprintf(`Email:    %s`, email),
		fmt.Sprintf(`Password: %s`, password),
		``,
		`═══ PHASE A — detect session state ═══`,
		``,
		`A1) (already done above)`,
		``,
		`A2) computer_use(action=screenshot). Look at the URL via`,
		`    browser_session(action=status) if needed.`,
		``,
		`    OUTCOME 1 — URL contains "/home" or any logged-in surface`,
		`                (your avatar visible, feed, /c/<creator>, etc.)`,
		`                → context resumed successfully. Reply on its`,
		`                own line:`,
		`                  RESULT: resumed=<the current full URL>`,
		`                STOP.`,
		``,
		`    OUTCOME 2 — URL contains "/login" / "/signup" or shows the`,
		`                Patreon login form → no saved session. Proceed`,
		`                to PHASE B to log in fresh.`,
		``,
		`    OUTCOME 3 — anything else (error, captcha, gate) → reply`,
		`                  RESULT: failed: <one-sentence reason>`,
		`                STOP.`,
		``,
		`═══ PHASE B — fresh login (only if PHASE A picked OUTCOME 2) ═══`,
		``,
		`B1) If a cookie/consent banner covers the page, click Accept`,
		`    by label first.`,
		``,
		`B2) Find the email input (orange badge). Click by label, then`,
		`    computer_use(action=type, text="<the email above>").`,
		``,
		`B3) Click Continue / Log-in (green badge) by label.`,
		``,
		`B4) If a password field appears: click by label, type the`,
		`    password, click Log-in/Continue. If no password field,`,
		`    skip B4.`,
		``,
		`B5) Wait briefly, then take a screenshot. If you now see the`,
		`    logged-in dashboard, reply on its own line:`,
		`      RESULT: logged_in_fresh=<the current full URL>`,
		`    STOP.`,
		``,
		`    Otherwise reply RESULT: failed: <reason> and STOP.`,
		``,
		`═══ Rules ═══`,
		``,
		`• Use label= for every click, never coordinate.`,
		`• Do NOT call pace() at any point.`,
		`• Do NOT navigate to /posts/new or any composer.`,
		`• Exactly one final RESULT: line at the end.`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-patreon-resume", 1000)
	logFile, _ := os.Create("computer_test_patreon_resume_chunks.log")
	defer logFile.Close()

	var resumed, freshLogin, failed bool
	var resultURL, failReason string
	promptLen := 0
	done := make(chan struct{})
	closed := false
	var buf strings.Builder
	var bufMu sync.Mutex

	parseResult := func(s string) {
		if resumed || freshLogin || failed {
			return
		}
		// Newline-required parsing — same fix as the publish test
		// for the chunk-split-mid-URL case.
		check := func(marker string) (string, bool) {
			i := strings.Index(s, marker)
			if i < 0 {
				return "", false
			}
			rest := s[i+len(marker):]
			nl := strings.IndexAny(rest, "\r\n")
			if nl < 0 {
				return "", false
			}
			return strings.TrimSpace(rest[:nl]), true
		}
		if v, ok := check("RESULT: resumed="); ok {
			resultURL, resumed = v, true
			return
		}
		if v, ok := check("RESULT: logged_in_fresh="); ok {
			resultURL, freshLogin = v, true
			return
		}
		if v, ok := check("RESULT: failed:"); ok {
			failReason, failed = v, true
		}
	}

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
					bufMu.Lock()
					buf.WriteString(ev.Text)
					s := buf.String()
					bufMu.Unlock()
					parseResult(s[promptLen:])
				}
				if (resumed || freshLogin || failed) && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(10 * time.Minute):
				return
			}
		}
	}()

	bufMu.Lock()
	promptLen = buf.Len()
	bufMu.Unlock()

	go thinker.Run()

	select {
	case <-done:
		t.Log("agent reported RESULT — stopping")
	case <-time.After(8 * time.Minute):
		t.Log("timeout — evaluating final state")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	logContent, _ := os.ReadFile("computer_test_patreon_resume_chunks.log")
	t.Logf("=== Chunks log ===\n%s", string(logContent))
	t.Logf("=== Final URL: %s", finalURL)

	switch {
	case failed:
		t.Fatalf("FAIL: agent reported failure: %s (final URL %q)", failReason, finalURL)
	case resumed:
		t.Logf("✓ PATH=resumed — context %q held a logged-in session, no login form needed (URL %s)", ctxID, resultURL)
		t.Log("✓ Run this test again with the same PATREON_CONTEXT_ID to confirm — it should take the resumed path again, much faster.")
	case freshLogin:
		t.Logf("✓ PATH=logged_in_fresh — no saved session, agent logged in fresh (URL %s)", resultURL)
		t.Logf("✓ Context %q now has cookies persisted to disk. Re-run this test with the same PATREON_CONTEXT_ID to see the resumed path skip the login entirely.", ctxID)
	default:
		t.Fatalf("FAIL: agent never emitted a RESULT line (final URL %q)", finalURL)
	}
}

// TestComputerUse_PatreonPublishFromContext is the publish test in
// "logged in already" mode. Requires PATREON_CONTEXT_ID to point at
// a context dir that already has a Patreon session — populate it
// once via TestComputerUse_PatreonContextResume (which logs in and
// flushes cookies to disk).
//
// What this proves end-to-end:
//   1. local backend's persistent context restores a real session
//   2. the agent's tool surface is the same as browserbase
//      (browser_session with context_id)
//   3. the composer skill prevents the empty-draft mistake
//   4. publishing works from a resumed session without re-login
//
// Should be ~3x faster than the cold-start publish test because
// PHASE A (login) is gone — the agent jumps straight to the
// composer, types content, publishes, reports.
//
// Side effect: writes ONE post on the configured Patreon account.
// Body is identified as automated; delete after if undesired.
//
//	# 1. Populate the context (run once, ~2-5min):
//	RUN_PATREON_TEST=1 PATREON_EMAIL=… PATREON_PASSWORD=… \
//	  PATREON_CONTEXT_ID=alexa-resume \
//	  go test -v -run TestComputerUse_PatreonContextResume -timeout 8m ./
//
//	# 2. Publish from that context (this test, ~2-3min):
//	RUN_PATREON_POST_TEST=1 PATREON_CONTEXT_ID=alexa-resume \
//	  go test -v -run TestComputerUse_PatreonPublishFromContext -timeout 10m ./
func TestComputerUse_PatreonPublishFromContext(t *testing.T) {
	if os.Getenv("RUN_PATREON_POST_TEST") == "" {
		t.Skip("set RUN_PATREON_POST_TEST=1 (real Patreon post — interactive, leaves a real post on the account)")
	}
	ctxID := os.Getenv("PATREON_CONTEXT_ID")
	if ctxID == "" {
		t.Skip("PATREON_CONTEXT_ID not set — run TestComputerUse_PatreonContextResume first to populate one")
	}
	tp := getTestProvider(t)
	apiKey := tp.APIKey

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			comp.Close()
		}
	}()
	t.Logf("computer connected: %dx%d backend=%s", comp.DisplaySize().Width, comp.DisplaySize().Height, backendName(t))
	t.Logf("using context_id=%q (must already have a logged-in Patreon session)", ctxID)

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	cfg := &Config{
		Directive: "You have a browser with Set-of-Mark grounding — interactive elements have colored numeric badges. Prefer label=N for clicks.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	today := time.Now().Format("2006-01-02")
	postTitle := fmt.Sprintf("[automated test] apteva agent — %s (from-context)", today)
	postBody := fmt.Sprintf(
		"This post was created by an automated apteva agent end-to-end "+
			"test on %s, using a resumed local context (no login needed). "+
			"Safe to delete.", today)

	thinker.InjectConsole(strings.Join([]string{
		`═══ MANDATORY FIRST ACTION ═══`,
		``,
		`Your very first tool call MUST be:`,
		``,
		fmt.Sprintf(`  browser_session(action=open, context_id=%q, url=https://www.patreon.com/posts/new, timeout=300)`, ctxID),
		``,
		`The browser is currently on about:blank. Open Patreon's post`,
		`composer FIRST — the context_id holds a logged-in session,`,
		`so Patreon will redirect you straight into the composer.`,
		`Do NOT click, screenshot, type, or pace before opening.`,
		``,
		`═══ Goal ═══`,
		``,
		`Publish ONE clearly-labelled automated test post on Patreon.`,
		`The context already holds a logged-in session — no login`,
		`needed. Three phases: A=verify composer loaded, B=compose+`,
		`publish, C=report. Do NOT call pace() at any point.`,
		``,
		`═══ Background — working with SaaS web composers ═══`,
		``,
		`Three traps that consistently break agent flows in editors`,
		`(Patreon, Twitter, Substack, ...). Internalise these BEFORE`,
		`PHASE B so you don't repeat the same mistakes:`,
		``,
		`/new allocates state — visit it only once. URLs like`,
		`/posts/new, /compose, /create create a draft on the server`,
		`before the editor renders, then redirect to /posts/<id>/edit.`,
		`Each visit spawns a duplicate draft. If the editor seems`,
		`broken, do NOT navigate to /new again as a reset — recover`,
		`in-place by pressing Escape and re-clicking the field.`,
		``,
		`Inline pickers steal focus from the body. Most rich-text`,
		`editors expose a "+" or "Add ..." button INSIDE the empty`,
		`body area. Clicking it opens a content-type picker menu`,
		`instead of focusing the text editor. Press Escape to`,
		`dismiss, then click a blank area inside the body region to`,
		`focus the actual text editor before typing.`,
		``,
		`Recognise the body field. The body editor is the LARGE`,
		`empty area below the title. Its SoM badge is ORANGE`,
		`(input/textbox tier) and its label text shows the editor's`,
		`placeholder — e.g. "Start writing…", "Type here",`,
		`"Tell your story…". Buttons reading "Click to add", "Add`,
		`image/video/poll", "+" are content-type pickers, NOT the`,
		`body field — they have GREEN badges (button tier) and`,
		`open menus instead of accepting text. Always click the`,
		`textbox-tier label first.`,
		``,
		`Publish is usually two clicks. The first Publish click opens`,
		`a confirmation modal (visibility, tiers, schedule). The post`,
		`is NOT published until you click Publish/Confirm/Post inside`,
		`that modal. After publish, dismiss any share/success modal`,
		`before reading the URL — until then, screenshots and clicks`,
		`target the modal, not the post.`,
		``,
		`Before reporting "done": URL is off /edit, any post-publish`,
		`modal is dismissed, the public post URL is visible.`,
		``,
		`═══ PHASE A — verify composer loaded (after MANDATORY FIRST ACTION) ═══`,
		``,
		`A1) computer_use(action=screenshot). Check the URL — it should`,
		`    contain "/posts/" (Patreon redirected /posts/new to`,
		`    /posts/<id>/edit). The page should show an empty title`,
		`    field and a body editor.`,
		``,
		`    If you see a /login URL or any auth surface, the context`,
		`    has expired. Reply on its own line:`,
		`      RESULT: failed: context not logged in`,
		`    and STOP.`,
		``,
		`═══ PHASE B — compose and publish ═══`,
		``,
		`B1) WHEN the composer is visible, locate the title input`,
		`    (placeholder "Add a title" or similar). Click by label,`,
		fmt.Sprintf(`    then computer_use(action=type, text=%q).`, postTitle),
		``,
		`B2) NEXT, locate the body / content area. Click by label`,
		`    (avoid the "+ Add ..." picker buttons — see Background),`,
		fmt.Sprintf(`    then computer_use(action=type, text=%q).`, postBody),
		``,
		`B3) AFTER both fields are typed, click the "Publish" button`,
		`    (top-right, often labelled "Publish" or "Publish ▾").`,
		``,
		`B4) Patreon MAY show a confirmation modal (visibility / tiers).`,
		`    If it appears, prefer "Public" / "Everyone"; otherwise pick`,
		`    the pre-selected default. Click the final Publish button.`,
		`    If no modal, skip B4.`,
		``,
		`B5) Wait briefly. The URL should change to a published-post URL`,
		`    (typically /posts/<numeric-id> without /edit, or stays at`,
		`    /posts/<id>/edit with a "Published" indicator).`,
		``,
		`═══ PHASE C — verify and report ═══`,
		``,
		`C1) Take a final screenshot. Dismiss any share/success modal`,
		`    that may be covering the page (look for a Close / X /`,
		`    Done button by label).`,
		``,
		`C2) If the URL contains "/posts/" AND the page shows clear`,
		`    evidence the post was published, reply on its own line:`,
		`      RESULT: post_url=<the current full URL>`,
		``,
		`    Otherwise reply on its own line:`,
		`      RESULT: failed: <one-sentence reason>`,
		``,
		`═══ Rules ═══`,
		``,
		`• Use label= for every click, never coordinate.`,
		`• Do NOT call pace() at any point.`,
		`• Publish exactly ONE post — do not create or publish anything else.`,
		`• Exactly one final RESULT: line at the end (plain text, not a tool call argument).`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-patreon-publish-from-context", 1500)
	logFile, _ := os.Create("computer_test_patreon_publish_from_context_chunks.log")
	defer logFile.Close()

	var published, failed bool
	var postURL, failReason string
	promptLen := 0
	done := make(chan struct{})
	closed := false
	var buf strings.Builder
	var bufMu sync.Mutex

	parseResult := func(s string) {
		if published || failed {
			return
		}
		// Newline-required parsing (chunks can split mid-URL).
		const okMarker = "RESULT: post_url="
		if i := strings.Index(s, okMarker); i >= 0 {
			rest := s[i+len(okMarker):]
			if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
				postURL = strings.TrimSpace(rest[:nl])
				published = true
				return
			}
		}
		const failMarker = "RESULT: failed:"
		if i := strings.Index(s, failMarker); i >= 0 {
			rest := s[i+len(failMarker):]
			if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
				failReason = strings.TrimSpace(rest[:nl])
				failed = true
			}
		}
	}

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
					bufMu.Lock()
					buf.WriteString(ev.Text)
					s := buf.String()
					bufMu.Unlock()
					parseResult(s[promptLen:])
				}
				if (published || failed) && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(15 * time.Minute):
				return
			}
		}
	}()

	bufMu.Lock()
	promptLen = buf.Len()
	bufMu.Unlock()

	go thinker.Run()

	select {
	case <-done:
		t.Log("agent reported RESULT — stopping")
	case <-time.After(8 * time.Minute):
		t.Log("timeout — evaluating final state")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	logContent, _ := os.ReadFile("computer_test_patreon_publish_from_context_chunks.log")
	t.Logf("=== Chunks log ===\n%s", string(logContent))
	t.Logf("=== Final URL: %s", finalURL)

	if failed {
		t.Fatalf("FAIL: agent reported failure: %s (final URL %q)", failReason, finalURL)
	}

	// Same loose URL match as TestComputerUse_PatreonPublishTextPost:
	// /posts/<digits> not followed by /edit means published.
	publishedURLLike := func(u string) bool {
		re := regexp.MustCompile(`/posts/\d+([?#/]|$)`)
		if !re.MatchString(u) {
			return false
		}
		return !strings.Contains(u, "/edit")
	}

	switch {
	case published && publishedURLLike(postURL):
		t.Logf("✓ post_url=%s (agent emitted RESULT)", postURL)
	case published && publishedURLLike(finalURL):
		t.Logf("✓ post_url=%s (agent emitted RESULT but reported draft URL; live URL %s confirms published)", postURL, finalURL)
		postURL = finalURL
	case !published && publishedURLLike(finalURL):
		t.Logf("✓ inferred success from final URL %s — agent never emitted RESULT but the page is at the published post", finalURL)
		postURL = finalURL
	default:
		t.Fatalf("FAIL: no success signal — agent emitted RESULT? %v (postURL=%q), final URL %q does not look like a published post", published, postURL, finalURL)
	}
	t.Log("✓ post URL is a published-post URL — text post published end-to-end FROM RESUMED CONTEXT")
	t.Logf("REMINDER: a real post was published — delete it manually if you don't want it lingering. URL: %s", postURL)
}

// TestComputerUse_PatreonVideoDraftFromContext exercises the
// composer's video-embed flow and stops at draft (no publish).
// Requires PATREON_CONTEXT_ID with a logged-in session — populate
// once via TestComputerUse_PatreonContextResume.
//
// Goal: open the composer, give it a title, embed the supplied
// video URL via Patreon's "Video" content picker, leave the post
// as a draft. Patreon auto-saves drafts on edit; we verify by
// asserting the final URL still contains /posts/<id>/edit (a
// published post would have redirected away from /edit).
//
// Side effect: ONE draft post on the account. Body labels itself
// as automated; delete after if undesired.
//
//	RUN_PATREON_DRAFT_TEST=1 PATREON_CONTEXT_ID=alexa-resume \
//	  go test -v -run TestComputerUse_PatreonVideoDraftFromContext -timeout 8m ./
func TestComputerUse_PatreonVideoDraftFromContext(t *testing.T) {
	if os.Getenv("RUN_PATREON_DRAFT_TEST") == "" {
		t.Skip("set RUN_PATREON_DRAFT_TEST=1 (real Patreon draft creation — interactive, leaves a draft on the account)")
	}
	ctxID := os.Getenv("PATREON_CONTEXT_ID")
	if ctxID == "" {
		t.Skip("PATREON_CONTEXT_ID not set — run TestComputerUse_PatreonContextResume first to populate one")
	}
	tp := getTestProvider(t)
	apiKey := tp.APIKey

	comp := buildComputerFromEnv(t)
	defer func() {
		if comp != nil {
			comp.Close()
		}
	}()
	t.Logf("computer connected: %dx%d backend=%s", comp.DisplaySize().Width, comp.DisplaySize().Height, backendName(t))
	t.Logf("using context_id=%q", ctxID)

	provider, err := selectProvider(&Config{})
	if err != nil {
		t.Fatalf("no provider: %v", err)
	}
	cfg := &Config{
		Directive: "You have a browser with Set-of-Mark grounding — interactive elements have colored numeric badges. Prefer label=N for clicks.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker(apiKey, provider, cfg)
	thinker.SetComputer(comp)

	today := time.Now().Format("2006-01-02")
	postTitle := fmt.Sprintf("[automated test] video draft — %s", today)
	const videoEmbedURL = "https://player.mediadelivery.net/play/638087/b910eb7b-982d-4862-9389-4b819b08ac41"

	thinker.InjectConsole(strings.Join([]string{
		`═══ MANDATORY FIRST ACTION ═══`,
		``,
		`Your very first tool call MUST be:`,
		``,
		fmt.Sprintf(`  browser_session(action=open, context_id=%q, url=https://www.patreon.com/posts/new, timeout=300)`, ctxID),
		``,
		`The browser is currently on about:blank. Open Patreon's post`,
		`composer FIRST — the context_id holds a logged-in session,`,
		`so Patreon will redirect straight into the composer at`,
		`/posts/<id>/edit. Do NOT click, screenshot, or pace before opening.`,
		``,
		`═══ Goal ═══`,
		``,
		`Create a DRAFT post on Patreon with:`,
		`  - title: as given below`,
		`  - body: empty is fine`,
		`  - one embedded video using the URL given below`,
		``,
		`Save it as a DRAFT. Do NOT publish. Patreon auto-saves`,
		`drafts on edit, so just composing the post is enough — there`,
		`is usually a "Saved" indicator near the top of the page.`,
		``,
		fmt.Sprintf(`Title:     %s`, postTitle),
		fmt.Sprintf(`Video URL: %s`, videoEmbedURL),
		``,
		`═══ Background — recognising the body and the video picker ═══`,
		``,
		`/new allocates state — visit it only once. URLs like`,
		`/posts/new create a draft on the server before the editor`,
		`renders, then redirect to /posts/<id>/edit. Each visit`,
		`spawns a duplicate. If the editor seems broken, recover in-`,
		`place — do NOT navigate to /new again.`,
		``,
		`Body editor: ORANGE badge (input/textbox tier), large`,
		`empty rectangle below the title. Its placeholder reads`,
		`"Start writing..." or similar.`,
		``,
		`Content pickers (Video / Audio / Image / Link / Attachment /`,
		`More): GREEN badges (button tier), arranged in a horizontal`,
		`toolbar above the body editor. The "Video" button opens a`,
		`dialog where you paste an embed URL.`,
		``,
		`Modal flows: when a dialog is open, ONLY click labels that`,
		`are spatially INSIDE the dialog's box. Labels on the sidebar,`,
		`top nav, or anywhere outside the dialog rectangle are`,
		`covered by the modal overlay even when their badges show —`,
		`clicking them either dismisses the modal (outside-click) or`,
		`does nothing. Identify the dialog by its Cancel/X close`,
		`button at one corner; everything else inside that bounded`,
		`area is the modal's surface.`,
		``,
		`Inline pickers (Patreon, Notion, Linear, Twitter editors).`,
		`Many composers don't show floating modals — instead, the`,
		`Video / Audio / Image / Link / Embed picker REPLACES the`,
		`body-editor area with an inline panel right where you used`,
		`to type. The panel has a Cancel/X button at one corner and`,
		`its own URL input (orange badge, "Type or paste URL").`,
		`After clicking the picker button:`,
		`  1. Look at the area where the body editor used to be —`,
		`     that's where the picker is now.`,
		`  2. Click the URL input INSIDE that panel (orange badge).`,
		`  3. Type the URL.`,
		`  4. Press Enter (key=Enter) to commit. Most embed pickers`,
		`     auto-commit on Enter — there is often NO visible`,
		`     "Insert" or "Embed" button. If Enter doesn't commit,`,
		`     try Tab to move focus, then look for a small arrow → or`,
		`     check icon near the URL field.`,
		`Do NOT click random other tabs (Upload / Connect Vimeo)`,
		`unless the URL approach failed twice — those are alternative`,
		`flows for files / OAuth and need different handling.`,
		``,
		`═══ PHASE A — verify composer loaded ═══`,
		``,
		`A1) computer_use(action=screenshot). Confirm URL contains`,
		`    "/posts/" (Patreon redirected /posts/new to`,
		`    /posts/<id>/edit). If not, reply RESULT: failed:`,
		`    composer not loaded and STOP.`,
		``,
		`═══ PHASE B — set the title ═══`,
		``,
		`B1) Find the title input (textarea, ORANGE badge, near the`,
		`    top of the composer area). Click by label.`,
		``,
		fmt.Sprintf(`B2) computer_use(action=type, text=%q).`, postTitle),
		``,
		`═══ PHASE C — embed the video ═══`,
		``,
		`C1) Find the "Video" button in the content-picker toolbar`,
		`    (GREEN badge, label text "Video"). Click it.`,
		``,
		`C2) A dialog should appear with a URL input field for the`,
		`    embed. Click the input field by label.`,
		``,
		fmt.Sprintf(`C3) computer_use(action=type, text=%q).`, videoEmbedURL),
		``,
		`C4) Look for a confirm/embed/add/insert button (typically`,
		`    "Embed", "Add", "Insert", or just an arrow → icon).`,
		`    Click it by label to commit the embed.`,
		``,
		`C5) If after TWO attempts the embed dialog isn't behaving as`,
		`    expected, fall back to Patreon's "Link" picker instead`,
		`    of "Video" — a player.mediadelivery.net URL also works`,
		`    via the link embedder. Otherwise reply RESULT: failed:`,
		`    could not embed video and STOP.`,
		``,
		`═══ PHASE D — confirm the draft saved ═══`,
		``,
		`D1) Wait briefly (action=wait, duration=2000) for Patreon`,
		`    to auto-save.`,
		``,
		`D2) Take a final screenshot. The URL should still be on`,
		`    "/posts/<id>/edit" — that's evidence the post is still`,
		`    a DRAFT (a published post redirects away from /edit).`,
		`    The page often shows a "Saved" indicator near the top.`,
		``,
		`D3) If the URL contains "/posts/" AND "/edit", reply on its`,
		`    own line:`,
		`      RESULT: draft_url=<the current full URL>`,
		``,
		`    Otherwise reply on its own line:`,
		`      RESULT: failed: <one-sentence reason>`,
		``,
		`═══ Rules ═══`,
		``,
		`• Use label= for every click; coordinates only as a last resort.`,
		`• Do NOT click any button labelled "Publish", "Post", "Schedule",`,
		`  or "Submit". This task ends with a DRAFT, not a published post.`,
		`• Do NOT call pace() at any point.`,
		`• Exactly one final RESULT: line at the end (plain text).`,
	}, "\n"))

	obs := thinker.bus.SubscribeAll("test-patreon-video-draft", 1500)
	logFile, _ := os.Create("computer_test_patreon_video_draft_chunks.log")
	defer logFile.Close()

	var saved, failed bool
	var draftURL, failReason string
	promptLen := 0
	done := make(chan struct{})
	closed := false
	var buf strings.Builder
	var bufMu sync.Mutex

	parseResult := func(s string) {
		if saved || failed {
			return
		}
		const okMarker = "RESULT: draft_url="
		if i := strings.Index(s, okMarker); i >= 0 {
			rest := s[i+len(okMarker):]
			if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
				draftURL = strings.TrimSpace(rest[:nl])
				saved = true
				return
			}
		}
		const failMarker = "RESULT: failed:"
		if i := strings.Index(s, failMarker); i >= 0 {
			rest := s[i+len(failMarker):]
			if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
				failReason = strings.TrimSpace(rest[:nl])
				failed = true
			}
		}
	}

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
					bufMu.Lock()
					buf.WriteString(ev.Text)
					s := buf.String()
					bufMu.Unlock()
					parseResult(s[promptLen:])
				}
				if (saved || failed) && !closed {
					closed = true
					close(done)
					return
				}
			case <-time.After(15 * time.Minute):
				return
			}
		}
	}()

	bufMu.Lock()
	promptLen = buf.Len()
	bufMu.Unlock()

	go thinker.Run()

	select {
	case <-done:
		t.Log("agent reported RESULT — stopping")
	case <-time.After(8 * time.Minute):
		t.Log("timeout — evaluating final state")
	}

	finalURL := currentURL(comp)
	thinker.Stop()
	time.Sleep(500 * time.Millisecond)
	logFile.Sync()

	logContent, _ := os.ReadFile("computer_test_patreon_video_draft_chunks.log")
	t.Logf("=== Chunks log ===\n%s", string(logContent))
	t.Logf("=== Final URL: %s", finalURL)

	if failed {
		t.Fatalf("FAIL: agent reported failure: %s (final URL %q)", failReason, finalURL)
	}

	// Draft URL shape: /posts/<digits>/edit means the post is still
	// a draft. The agent might have reported the URL with or without
	// query params; the live currentURL is the authoritative check.
	draftURLLike := func(u string) bool {
		re := regexp.MustCompile(`/posts/\d+/edit`)
		return re.MatchString(u)
	}

	switch {
	case saved && draftURLLike(draftURL):
		t.Logf("✓ draft_url=%s (agent emitted RESULT and URL is on /edit)", draftURL)
	case saved && draftURLLike(finalURL):
		t.Logf("✓ draft saved at %s (agent's reported URL %q wasn't /edit-shaped, but live URL is)", finalURL, draftURL)
		draftURL = finalURL
	case !saved && draftURLLike(finalURL):
		t.Logf("✓ inferred draft from final URL %s — agent never emitted RESULT but the page is on /edit", finalURL)
		draftURL = finalURL
	default:
		t.Fatalf("FAIL: no draft signal — agent emitted RESULT? %v (draftURL=%q), final URL %q is not on /posts/<id>/edit", saved, draftURL, finalURL)
	}
	t.Log("✓ URL stays on /posts/<id>/edit — post saved as DRAFT, not published")
	t.Logf("REMINDER: a real draft was created on the Patreon account. URL: %s", draftURL)
}

// buildComputerFromEnv + backendName are defined in computer_backend_test.go.

// startCodeServer listens on a random loopback port and pushes POSTed
// bodies into codeCh. Returns the base URL and a close function.
// Accepts both `curl -d 123456 URL/code` (raw body) and form encoding.
func startCodeServer(t *testing.T, codeCh chan<- string) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/code", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		code := strings.TrimSpace(string(body))
		// Accept form encoding too: code=123456
		if eq := strings.SplitN(code, "=", 2); len(eq) == 2 && eq[0] == "code" {
			code = eq[1]
		}
		if code == "" {
			http.Error(w, "empty body; POST the code as the raw body", http.StatusBadRequest)
			return
		}
		select {
		case codeCh <- code:
			fmt.Fprintln(w, "ok")
		default:
			http.Error(w, "code channel full — already received one", http.StatusConflict)
		}
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	base := "http://" + ln.Addr().String()
	return base, func() {
		srv.Close()
	}
}

// watchCodeFile polls the file for a code, pushes it into codeCh when
// seen, and deletes the file so a second iteration doesn't re-trigger.
// No inotify — 500ms polling is fine for an interactive test.
func watchCodeFile(t *testing.T, path string, codeCh chan<- string) func() {
	t.Helper()
	stop := make(chan struct{})
	go func() {
		tk := time.NewTicker(500 * time.Millisecond)
		defer tk.Stop()
		for {
			select {
			case <-stop:
				return
			case <-tk.C:
				b, err := os.ReadFile(path)
				if err != nil {
					continue
				}
				code := strings.TrimSpace(string(bytes.TrimSpace(b)))
				if code == "" {
					continue
				}
				os.Remove(path)
				select {
				case codeCh <- code:
				default:
				}
			}
		}
	}()
	return func() { close(stop) }
}
