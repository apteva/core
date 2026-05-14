package scenarios

import (
	"testing"
	"time"

	. "github.com/apteva/core"
)

// vagueDiscoveryScenario deliberately exercises the search_tools path.
//
// Every other scenario names its tools in the directive — so the
// directive-fed BM25 preload surfaces them and the agent rarely needs
// to search. This one withholds the tool names on purpose: the
// directive says "tools are connected but not loaded — discover them
// with search_tools." The agent MUST search to get anything done.
//
// What it proves end-to-end in a real scenario (not just a unit test):
//   - the agent calls search_tools when it has no tool for a job
//   - the schemas it gets back are actually callable next turn
//   - it can chain discovery → call → discovery → call across a
//     multi-step task (see queue → look up answer → reply → close)
//
// The Phase assertions are outcome-based (the helpdesk audit log).
// That search_tools specifically fired is confirmed by reading the
// run log — the harness can't observe inline meta-tools from a Phase.
var vagueDiscoveryScenario = Scenario{
	Name: "VagueDiscovery",
	Directive: `You are an operations assistant for a small business.

Several tool servers are connected to you, but their tools are NOT loaded
into your context up front — you do not know their names. Whenever you
need a capability, call search_tools with a short description of what you
want, read what it returns, then call the tool it surfaced.

Your job: keep the customer request queue clear.
- On startup, search for a way to see pending customer requests, and check
  the queue.
- For each pending request: search for a way to look up the answer, then a
  way to send a response back to the customer, then a way to mark the
  request resolved.
Work the queue until it is empty, then idle.`,
	MCPServers: []MCPServerConfig{{
		Name:    "helpdesk",
		Command: "", // filled in test
		Env:     map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"},
	}},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "kb.json", map[string]string{
			"hours":    "We are open Monday to Friday, 9am to 5pm.",
			"delivery": "We deliver within 10 miles for free.",
			"returns":  "You can return items within 30 days with a receipt.",
		})
		// Two requests waiting from the start — the agent has to
		// discover the queue tool before it can even see them.
		WriteJSONFile(t, dir, "tickets.json", []map[string]string{
			{"id": "r1", "question": "What are your opening hours?"},
			{"id": "r2", "question": "What is your return policy?"},
		})
	},
	Phases: []Phase{
		{
			Name:    "Discovery — agent searches for and checks the queue",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Outcome: the agent found and called the queue tool.
				// It can only have the tool's name by searching for it
				// — the directive never names it.
				entries := ReadAuditEntries(dir)
				lists := CountTool(entries, "list_tickets")
				t.Logf("  ... list_tickets=%d", lists)
				return lists > 0
			},
		},
		{
			Name:    "Resolution — both requests answered and closed",
			Timeout: 240 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := ReadAuditEntries(dir)
				replies := CountTool(entries, "reply_ticket")
				closes := CountTool(entries, "close_ticket")
				t.Logf("  ... lookup=%d replies=%d closes=%d",
					CountTool(entries, "lookup_kb"), replies, closes)
				return replies >= 2 && closes >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				if CountTool(entries, "lookup_kb") == 0 {
					t.Logf("NOTE: lookup_kb never called — agent may have answered from the question text alone")
				}
			},
		},
	},
	Timeout: 8 * time.Minute,
}

func TestScenario_VagueDiscovery(t *testing.T) {
	bin := BuildMCPBinary(t, "mcps/helpdesk")
	t.Logf("built mcp-helpdesk: %s", bin)

	s := vagueDiscoveryScenario
	s.MCPServers[0].Command = bin
	RunScenario(t, s)
}
