package scenarios

import (
	"testing"
	"time"

	. "github.com/apteva/core"
)

var chatScenario = Scenario{
	Name: "Chat",
	Directive: `You are a helpful assistant. Messages arrive as console events.
When a message arrives, spawn a thread to handle it. The thread should reply using send_reply and your answer.
Be concise, accurate, and helpful. Answer questions directly.`,
	MCPServers: []MCPServerConfig{{
		Name:    "chat",
		Command: "", // filled in test
		Env:     map[string]string{"CHAT_DATA_DIR": "{{dataDir}}"},
	}},
	DataSetup: func(t *testing.T, dir string) {},
	Phases: []Phase{
		{
			Name:    "Factual question — What is the capital of France?",
			Timeout: 60 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("What is the capital of France?")
					}
					replies := ReadChatReplies(dir)
					t.Logf("  ... replies=%d threads=%v", len(replies), ThreadIDs(th))
					return len(replies) >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := ReadChatReplies(dir)
				last := replies[len(replies)-1]
				t.Logf("Reply to alice: %q", last.Message)
				if !ChatContainsAny(replies, "Paris") {
					t.Errorf("expected reply to mention Paris, got: %q", last.Message)
				}
			},
		},
		{
			Name:    "Follow-up question — What is its population?",
			Timeout: 60 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("What is its population?")
					}
					replies := ReadChatReplies(dir)
					t.Logf("  ... replies=%d", len(replies))
					return len(replies) >= 2
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := ReadChatReplies(dir)
				last := replies[len(replies)-1]
				t.Logf("Reply to alice: %q", last.Message)
				if !ChatContainsAny(replies[len(replies)-1:], "million", "2", "11", "12") {
					t.Logf("NOTE: reply may not contain population figure: %q", last.Message)
				}
			},
		},
		{
			Name:    "Multi-user — bob asks 2+2",
			Timeout: 60 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("What is 2 + 2?")
					}
					replies := ReadChatReplies(dir)
					hasBob := false
					for _, r := range replies {
						if r.User == "bob" {
							hasBob = true
						}
					}
					t.Logf("  ... replies=%d bob=%v threads=%v", len(replies), hasBob, ThreadIDs(th))
					return hasBob
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := ReadChatReplies(dir)
				var bobReply string
				for _, r := range replies {
					if r.User == "bob" {
						bobReply = r.Message
					}
				}
				t.Logf("Bob's reply: %q", bobReply)
				if !ChatContainsAny([]ChatReply{{Message: bobReply}}, "4") {
					t.Errorf("expected reply to contain '4', got: %q", bobReply)
				}
			},
		},
	},
	Timeout: 3 * time.Minute,
}

func TestScenario_Chat(t *testing.T) {
	bin := BuildMCPBinary(t, "mcps/chat")
	t.Logf("built mcp-chat: %s", bin)

	s := chatScenario
	s.MCPServers[0].Command = bin
	RunScenario(t, s)
}
