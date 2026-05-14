package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var selfOnboardingScenario = Scenario{
	Name: "SelfOnboarding",
	Directive: `You are a helpful assistant for a social-media power user. You have
access to an integrations gateway (mcp server name "gateway") and you can
spawn workers with mcp="<name>" for any connected integration.

The user will tell you what they want to do. Your job:

1. Call gateway_list_integrations to see what's available.
2. Pick the integration that matches the user's request.
3. Call gateway_get_integration(slug=...) to discover which credentials it
   needs AND which tools it offers. Look at the full "tools" list in the
   response — it shows every operation the integration supports.
4. Decide which tools from that integration the user actually needs. Read
   the user's request carefully: if they say "only posting" or "don't
   allow deletes", you MUST narrow the scope to the minimum set. This is
   least-privilege — never enable a tool the user didn't ask for.
5. Ask the user for ONLY the credentials that are required. Use
   channels_respond(channel="cli", text="...") to message them. Be
   specific about which field you need (e.g. "Please provide your
   FakePost api_key"). Do NOT invent credentials.
6. Wait for the user to reply with the credentials.
7. Once you have the key, call gateway_create_connection(slug=...,
   credentials=..., allowed_tools="tool1,tool2") — credentials as a JSON
   string, allowed_tools as a comma-separated list of the tool names you
   picked in step 4. The tool returns a connect_now hint, an mcp_name
   to spawn against, and enabled_tools showing what's actually reachable.
8. Spawn a worker with mcp=<mcp_name from the response>, give it the post
   content, and have it call the appropriate tool to publish.
9. When the worker reports success, tell the user "DONE" via
   channels_respond.

Stay patient — ask for credentials, then wait for the user reply before
moving on. Do not guess or fabricate keys. When the user says "only
posting, no deletes", you MUST set allowed_tools=post (or post and
schedule_post if they mention both) — not every tool the integration
offers.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "gateway",
			Command: "", // filled in by test
			Env:     map[string]string{"FAKE_GATEWAY_DATA_DIR": "{{dataDir}}"},
		},
		{
			// Non-main-access: cataloged, so spawned workers can connect
			// to it via mcp="fake_post" but main itself can't call its
			// tools directly. This mirrors the real flow where the
			// agent delegates tool work to workers.
			Name:    "fake_post",
			Command: "", // filled in by test
			Env: map[string]string{
				"FAKE_POST_DATA_DIR":    "{{dataDir}}",
				"FAKE_GATEWAY_DATA_DIR": "{{dataDir}}",
			},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Nothing to seed — the gateway writes connections.json itself
		// when the agent calls create_connection, and fake_post writes
		// posts.jsonl when a successful post lands.
	},
	Phases: []Phase{
		{
			// Phase 1: kick off the conversation. We inject the request
			// via InjectConsole (simulated CLI user message) on the very
			// first poll so the agent has something to act on.
			Name:    "User request dispatched + agent discovers integrations",
			Timeout: 90 * time.Second,
			Wait: (func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						// Deliberately phrased to force the scoping
						// decision: the user wants posting only, and
						// explicitly mentions delete/list should be
						// off-limits. The agent MUST pick a subset.
						th.InjectConsole("I need to post the status update 'Hello from apteva' to my FakePost social account. IMPORTANT: I only want posting enabled — do NOT enable delete_post or list_posts or anything else. Scope the connection tightly to posting only. Please set up whatever you need and publish it.")
						injected = true
					}
					// Wait until the agent actually called list_integrations +
					// get_integration against the gateway. Once both appear
					// in the audit we're past the discovery step.
					entries := ReadAuditEntries(dir)
					listed := CountTool(entries, "list_integrations") > 0
					gotten := CountTool(entries, "get_integration") > 0
					return listed && gotten
				}
			})(),
		},
		{
			// Phase 2: the agent should now be blocked waiting for
			// credentials. Inject the api_key as the user's reply.
			// The agent's next iteration drains this as an external event
			// and continues to create_connection.
			Name:    "Agent asked for credentials; user provides api_key",
			Timeout: 90 * time.Second,
			Wait: (func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						// Small delay so the agent has a chance to
						// actually send its question before our reply
						// lands. Not strictly required (out-of-order
						// messages are fine) but makes logs cleaner.
						time.Sleep(2 * time.Second)
						th.InjectConsole("Sure — my FakePost api_key is fp_TESTKEY_abc123. Use it.")
						injected = true
					}
					// Wait for create_connection to actually fire against
					// the gateway. That's proof the agent took our
					// credential reply and used it.
					entries := ReadAuditEntries(dir)
					return CountTool(entries, "create_connection") > 0
				}
			})(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// The gateway should now have a connections.json with one
				// entry for fake_post carrying the injected key. If the
				// agent made up a different key, we catch it here.
				data, err := os.ReadFile(filepath.Join(dir, "connections.json"))
				if err != nil {
					t.Fatalf("connections.json missing: %v", err)
				}
				var conns []struct {
					Slug         string            `json:"slug"`
					Credentials  map[string]string `json:"credentials"`
					AllowedTools []string          `json:"allowed_tools"`
				}
				json.Unmarshal(data, &conns)
				if len(conns) == 0 {
					t.Fatal("no connections written to connections.json")
				}
				found := false
				for _, c := range conns {
					if c.Slug != "fake_post" {
						continue
					}
					key := c.Credentials["api_key"]
					if key == "" {
						t.Errorf("fake_post connection has no api_key")
					}
					// Allow some flexibility — the agent might have
					// trimmed/quoted the key — but it must contain the
					// distinctive substring from our injected message.
					if !strings.Contains(key, "fp_TESTKEY") {
						t.Errorf("fake_post api_key = %q, expected to contain %q — agent may have invented a different key",
							key, "fp_TESTKEY")
					}
					// Scope check — the user explicitly asked for
					// posting-only, so allowed_tools must be populated
					// and must NOT contain delete_post or list_posts.
					// "post" alone or "post,schedule_post" are both
					// reasonable interpretations; we allow both but
					// reject anything with deletes.
					if len(c.AllowedTools) == 0 {
						t.Error("allowed_tools is empty — agent ignored the least-privilege instruction and left every tool enabled")
					}
					forbidden := map[string]bool{
						"delete_post": true,
						"list_posts":  true,
					}
					for _, name := range c.AllowedTools {
						if forbidden[name] {
							t.Errorf("allowed_tools contains %q which the user explicitly excluded", name)
						}
					}
					// And "post" itself MUST be in the set — otherwise
					// the agent can't publish.
					hasPost := false
					for _, name := range c.AllowedTools {
						if name == "post" {
							hasPost = true
							break
						}
					}
					if !hasPost {
						t.Errorf("allowed_tools = %v does not include 'post' — agent can't publish", c.AllowedTools)
					}
					t.Logf("scoped connection allowed_tools = %v", c.AllowedTools)
					found = true
				}
				if !found {
					t.Error("no fake_post connection found — agent created the wrong slug")
				}
			},
		},
		{
			// Phase 3: the worker (spawned via mcp="fake_post") should
			// have called `post`, which only succeeds if the connection
			// row exists. Verify the posts.jsonl file landed with our
			// content substring.
			Name:    "Agent spawned a worker and successfully posted",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "posts.jsonl"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), "Hello from apteva")
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Should have at least one post with our content.
				data, _ := os.ReadFile(filepath.Join(dir, "posts.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				if len(lines) == 0 || lines[0] == "" {
					t.Fatal("posts.jsonl is empty")
				}
				if !strings.Contains(string(data), "Hello from apteva") {
					t.Errorf("post content doesn't match — got: %s", data)
				}
				// Verify the post tool was called via a thread, not main,
				// by checking that a spawn actually happened. Threads
				// count at the ThreadManager level proves the agent
				// spawned a worker rather than trying to call fake_post
				// directly from main (which is impossible anyway because
				// fake_post isn't main_access).
				all := AllThreadInfos(th.Threads())
				t.Logf("threads at verify: %d", len(all))

				// Scope enforcement check. fake_post's audit.jsonl records
				// every tool call the agent's worker made against it,
				// including rejections. We require:
				//   - at least one successful `post` call
				//   - zero `delete_post` calls (forbidden by scope)
				//   - zero `list_posts` calls (forbidden by scope)
				// A scoped MCP that filters tools/list should prevent
				// the agent from even seeing delete_post, so it should
				// never show up in the audit at all.
				entries := ReadAuditEntries(dir)
				var fakePostEntries []ScenarioAuditEntry
				for _, e := range entries {
					// fake_post writes its own audit.jsonl alongside
					// fake_gateway's — they share the dir in this
					// scenario. Entries are distinguishable by tool
					// name since the tool sets are disjoint.
					switch e.Tool {
					case "post", "schedule_post", "delete_post", "list_posts":
						fakePostEntries = append(fakePostEntries, e)
					}
				}
				posts := CountTool(fakePostEntries, "post")
				deletes := CountTool(fakePostEntries, "delete_post")
				lists := CountTool(fakePostEntries, "list_posts")
				if posts == 0 {
					t.Error("no post calls in fake_post audit — agent never actually published")
				}
				if deletes > 0 {
					t.Errorf("agent called delete_post %d times — scope was not enforced", deletes)
				}
				if lists > 0 {
					t.Errorf("agent called list_posts %d times — scope was not enforced", lists)
				}
				t.Logf("fake_post audit summary: post=%d schedule_post=%d delete_post=%d list_posts=%d",
					posts, CountTool(fakePostEntries, "schedule_post"), deletes, lists)
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_SelfOnboarding(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	gatewayBin := BuildMCPBinary(t, "mcps/fake_gateway")
	postBin := BuildMCPBinary(t, "mcps/fake_post")
	t.Logf("built fake_gateway=%s fake_post=%s", gatewayBin, postBin)

	s := selfOnboardingScenario
	s.MCPServers[0].Command = gatewayBin
	s.MCPServers[1].Command = postBin
	RunScenario(t, s)
}
