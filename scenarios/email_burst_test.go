package scenarios

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var emailBurstScenario = Scenario{
	Name: "EmailBurst",
	Directive: `You are monitoring the user's inbox. Use the email tools to check
for new messages and handle each real one.

For every real message that needs a response:
  1. Read its body with email_read(id).
  2. Send a thoughtful reply with email_reply(id, body).
  3. Archive it with email_archive(id).
Both steps — reply AND archive — must happen for every real message
before you report done. An email with a reply sent but still in the
inbox counts as NOT handled.

Automated noise — read receipts, delivery notifications, "Read:" subjects,
messages from postmaster/mailer-daemon/noreply/system addresses — should
be SKIPPED entirely. Do NOT reply to noise. Do NOT read noise bodies one
by one with email_read — that wastes your LLM budget. Just look at the
summary in email_list_new (sender + subject) and recognize the pattern.
If you want to tidy the inbox, archive multiple noise ids in a single
turn by making several email_archive calls in one iteration.

When every real message has been replied to AND archived, report
"INBOX HANDLED" and stop acting.`,
	MCPServers: []MCPServerConfig{
		{
			Name:       "email",
			Command:    "", // filled in by test
			Env:        map[string]string{"FAKE_EMAIL_DATA_DIR": "{{dataDir}}"},
			
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed 3 real messages the agent must handle.
		inbox := []map[string]string{
			{
				"id":      "msg-001",
				"from":    "alice@acme.com",
				"subject": "Question about pricing",
				"body":    "Hi, I'm interested in your Enterprise plan. Can you send me the pricing breakdown for 50 seats? Thanks!",
				"kind":    "real",
			},
			{
				"id":      "msg-002",
				"from":    "bob@globex.io",
				"subject": "Meeting request — Tuesday 2pm?",
				"body":    "Hey, can we meet Tuesday at 2pm to discuss the integration? Should take about 30 minutes.",
				"kind":    "real",
			},
			{
				"id":      "msg-003",
				"from":    "carol@initech.com",
				"subject": "Feature request: SSO support",
				"body":    "Our security team is asking whether you support SAML SSO. Is this on your roadmap?",
				"kind":    "real",
			},
		}
		WriteJSONFile(t, dir, "inbox.json", inbox)
	},
	Phases: []Phase{
		{
			// Phase 1: let the agent do its first list_new call, then
			// immediately flood the inbox with 50 noise emails. The agent
			// on its next poll sees 53 entries mixed together and has to
			// pick out the 3 real ones by pattern. This is the "burst"
			// event — noise arriving via the real email path, not via a
			// side channel, which matches what a buggy email provider
			// would actually do.
			Name:    "Burst of 50 noise emails lands in the inbox",
			Timeout: 90 * time.Second,
			Wait: (func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						// Has the agent called list_new yet? We want the
						// burst to land AFTER the agent has started
						// looking at the inbox, so it has to notice the
						// shape change on its next poll rather than seeing
						// a pre-flooded inbox from its very first call.
						entries := ReadAuditEntries(dir)
						if CountTool(entries, "list_new") == 0 {
							return false
						}
						t.Log("  ... agent has polled list_new — flooding inbox with 50 noise emails")

						// Read the current inbox.json, append 50 noise
						// entries, write it back. flock isn't necessary
						// here since the MCP subprocess does its own
						// locking; this write happens between agent
						// iterations so the window is safe enough.
						path := filepath.Join(dir, "inbox.json")
						data, _ := os.ReadFile(path)
						var inbox []map[string]string
						json.Unmarshal(data, &inbox)
						for i := 1; i <= 50; i++ {
							inbox = append(inbox, map[string]string{
								"id":      fmt.Sprintf("noise-%03d", i),
								"from":    fmt.Sprintf("postmaster-%d@system.noreply", i),
								"subject": "Read: your earlier email",
								"body":    "Automated read receipt — no content. Your message was opened at <timestamp>.",
								"kind":    "noise",
							})
						}
						out, _ := json.MarshalIndent(inbox, "", "  ")
						os.WriteFile(path, out, 0644)
						injected = true
					}
					// We're done with this phase as soon as the noise is
					// in. Phase 2 handles the "wait for everything to be
					// handled" part.
					return true
				}
			})(),
		},
		{
			// Phase 2: the hard requirement — every real email replied to
			// AND archived, while the 50 noise entries sat alongside them
			// in the inbox. Gate on BOTH counts so we don't declare
			// victory on a half-done run.
			Name:    "All 3 real emails replied to AND archived",
			Timeout: 4 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Count replies by id.
				replies := map[string]bool{}
				if data, err := os.ReadFile(filepath.Join(dir, "sent.jsonl")); err == nil {
					for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
						if line == "" {
							continue
						}
						var r struct {
							ID string `json:"id"`
						}
						json.Unmarshal([]byte(line), &r)
						if r.ID != "" {
							replies[r.ID] = true
						}
					}
				}
				// Count archives by id.
				archived := map[string]bool{}
				if data, err := os.ReadFile(filepath.Join(dir, "archive.jsonl")); err == nil {
					for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
						if line == "" {
							continue
						}
						var r struct {
							ID string `json:"id"`
						}
						json.Unmarshal([]byte(line), &r)
						if r.ID != "" {
							archived[r.ID] = true
						}
					}
				}
				realIDs := []string{"msg-001", "msg-002", "msg-003"}
				allReplied := true
				allArchived := true
				for _, id := range realIDs {
					if !replies[id] {
						allReplied = false
					}
					if !archived[id] {
						allArchived = false
					}
				}
				t.Logf("  ... real: replied=%d/3 archived=%d/3", len(replies), len(archived))
				return allReplied && allArchived
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Strict checks: only the 3 real messages should have
				// received replies, and none of the 50 noise entries.
				data, _ := os.ReadFile(filepath.Join(dir, "sent.jsonl"))
				type reply struct {
					ID   string `json:"id"`
					Body string `json:"body"`
				}
				var replies []reply
				for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
					if line == "" {
						continue
					}
					var r reply
					json.Unmarshal([]byte(line), &r)
					replies = append(replies, r)
				}

				realIDs := map[string]bool{"msg-001": true, "msg-002": true, "msg-003": true}
				seen := map[string]bool{}
				for _, r := range replies {
					if strings.HasPrefix(r.ID, "noise-") {
						t.Errorf("agent replied to noise message %s: %q", r.ID, truncForLog(r.Body, 80))
					}
					if !realIDs[r.ID] && !strings.HasPrefix(r.ID, "noise-") {
						t.Errorf("agent replied to unknown id %q", r.ID)
					}
					seen[r.ID] = true
				}
				for id := range realIDs {
					if !seen[id] {
						t.Errorf("real email %s never received a reply", id)
					}
				}

				// Archive check — every real email should have ended up
				// in archive.jsonl.
				archData, _ := os.ReadFile(filepath.Join(dir, "archive.jsonl"))
				archivedReal := 0
				for _, line := range strings.Split(strings.TrimSpace(string(archData)), "\n") {
					if line == "" {
						continue
					}
					var e struct {
						ID string `json:"id"`
					}
					json.Unmarshal([]byte(line), &e)
					if realIDs[e.ID] {
						archivedReal++
					}
				}
				if archivedReal < 3 {
					t.Errorf("only %d/3 real emails archived — agent left work unfinished", archivedReal)
				}
				t.Logf("archive: %d real emails archived (total lines: %d)",
					archivedReal,
					len(strings.Split(strings.TrimSpace(string(archData)), "\n")),
				)
			},
		},
	},
	// Hard timeout + iteration ceiling enforced via the scenario runner's
	// token accounting. Without a ceiling, a naive agent that processed
	// every notification would still eventually finish all 3 real replies
	// and pass the correctness checks — we use the scenario's Timeout to
	// catch runaway burn.
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_EmailBurst(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	emailBin := BuildMCPBinary(t, "mcps/fake_email")
	t.Logf("built fake_email=%s", emailBin)

	s := emailBurstScenario
	s.MCPServers[0].Command = emailBin
	RunScenario(t, s)
}
