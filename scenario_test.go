package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- Scenario framework ---

// Scenario defines a complete agent behavior test.
type Scenario struct {
	Name       string
	Directive  string
	MCPServers []MCPServerConfig // {{dataDir}} in Env values is replaced at runtime
	DataSetup  func(t *testing.T, dir string)
	Phases     []Phase
	Timeout    time.Duration // hard cap for entire scenario
	MaxThreads int           // peak thread count limit (0 = no limit)
}

// Phase is a step in a scenario.
type Phase struct {
	Name    string
	Setup   func(t *testing.T, dir string)                    // optional: inject data before this phase
	Wait    func(t *testing.T, dir string, th *Thinker) bool  // poll condition (return true when done)
	Verify  func(t *testing.T, dir string, th *Thinker)       // optional: assertions after Wait succeeds
	Timeout time.Duration
}

// runScenario executes a scenario end-to-end.
func runScenario(t *testing.T, s Scenario) {
	t.Helper()
	apiKey := getAPIKey(t)

	if s.Timeout == 0 {
		s.Timeout = 3 * time.Minute
	}

	// Hard deadline
	deadline := time.AfterFunc(s.Timeout, func() {
		t.Errorf("HARD TIMEOUT after %v — stopping to prevent token burn", s.Timeout)
	})
	defer deadline.Stop()

	// Data directory
	dataDir := t.TempDir()
	if s.DataSetup != nil {
		s.DataSetup(t, dataDir)
	}

	// Replace {{dataDir}} in MCP server env
	mcpServers := make([]MCPServerConfig, len(s.MCPServers))
	for i, cfg := range s.MCPServers {
		mcpServers[i] = cfg
		if mcpServers[i].Env != nil {
			env := make(map[string]string)
			for k, v := range cfg.Env {
				env[k] = strings.ReplaceAll(v, "{{dataDir}}", dataDir)
			}
			mcpServers[i].Env = env
		}
	}

	// Create thinker
	thinker := newScenarioThinker(t, apiKey, s.Directive, mcpServers)

	// Track peak thread count
	var peakThreads atomic.Int32
	var stopped atomic.Bool

	// Observer: log events
	obs := thinker.bus.SubscribeAll("test-observer", 500)
	go func() {
		for !stopped.Load() {
			select {
			case ev := <-obs.C:
				switch ev.Type {
				case EventThinkDone:
					t.Logf("[%s iter %d] threads=%d rate=%s tools=%v events=%d",
						ev.From, ev.Iteration, ev.ThreadCount, ev.Rate, ev.ToolCalls, len(ev.ConsumedEvents))
				case EventThreadStart, EventThreadDone:
					t.Logf("[%s] %s %s", ev.From, ev.Type, ev.Text)
				case EventInbox:
					text := ev.Text
					if len(text) > 80 {
						text = text[:80] + "..."
					}
					t.Logf("[bus] %s→%s %s", ev.From, ev.To, text)
				}
			case <-thinker.quit:
				return
			}
		}
	}()

	// Track peak threads
	go func() {
		for !stopped.Load() {
			count := int32(thinker.threads.Count())
			for {
				old := peakThreads.Load()
				if count <= old || peakThreads.CompareAndSwap(old, count) {
					break
				}
			}
			time.Sleep(500 * time.Millisecond)
		}
	}()

	go thinker.Run()
	defer func() {
		stopped.Store(true)
		thinker.Stop()
	}()

	// Run phases
	for i, phase := range s.Phases {
		t.Logf("=== Phase %d: %s ===", i+1, phase.Name)

		if phase.Setup != nil {
			phase.Setup(t, dataDir)
		}

		timeout := phase.Timeout
		if timeout == 0 {
			timeout = 60 * time.Second
		}

		if phase.Wait != nil {
			waitFor(t, timeout, 3*time.Second, phase.Name, func() bool {
				return phase.Wait(t, dataDir, thinker)
			})
		}

		if phase.Verify != nil {
			phase.Verify(t, dataDir, thinker)
		}

		t.Logf("Phase %d PASSED", i+1)
	}

	// Check peak threads
	peak := int(peakThreads.Load())
	t.Logf("Peak threads: %d, final threads: %d", peak, thinker.threads.Count())
	if s.MaxThreads > 0 && peak > s.MaxThreads {
		t.Errorf("peak thread count %d exceeded limit of %d", peak, s.MaxThreads)
	}

	t.Logf("=== Scenario %q PASSED ===", s.Name)
}

// --- Helpers ---

func buildMCPBinary(t *testing.T, dir string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), filepath.Base(dir))
	cmd := exec.Command("go", "build", "-o", bin, ".")
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v\n%s", dir, err, out)
	}
	return bin
}

func newScenarioThinker(t *testing.T, apiKey, directive string, mcpServers []MCPServerConfig) *Thinker {
	t.Helper()

	tmpDir := t.TempDir()

	cfg := &Config{
		path:       filepath.Join(tmpDir, "config.json"),
		Directive:  directive,
		MCPServers: mcpServers,
	}
	cfg.Save()

	memStore := &MemoryStore{
		apiKey: apiKey,
		path:   filepath.Join(tmpDir, "memory.jsonl"),
	}

	bus := NewEventBus()
	thinker := &Thinker{
		apiKey: apiKey,
		messages: []Message{
			{Role: "system", Content: ""},
		},
		config:    cfg,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateReactive,
		agentRate: RateSlow,
		memory:    memStore,
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
	}
	thinker.threads = NewThreadManager(thinker)
	thinker.registry = NewToolRegistry(apiKey)

	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, thinker.registry, "")}

	go thinker.registry.EmbedAll(memStore)

	thinker.filterEvents = func(events []string) []string {
		var kept []string
		for _, ev := range events {
			if !thinker.threads.Route(ev) {
				kept = append(kept, ev)
			}
		}
		return kept
	}
	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(cfg.GetDirective(), thinker.registry, toolDocs)
	}

	if len(mcpServers) > 0 {
		servers := connectAndRegisterMCP(mcpServers, thinker.registry, memStore)
		t.Cleanup(func() {
			for _, s := range servers {
				s.Close()
			}
		})
	}

	return thinker
}

func waitFor(t *testing.T, timeout, interval time.Duration, desc string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(interval)
	}
	t.Fatalf("timeout after %v waiting for: %s", timeout, desc)
}

type scenarioAuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

func readAuditEntries(dir string) []scenarioAuditEntry {
	data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
	if err != nil {
		return nil
	}
	var entries []scenarioAuditEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if line == "" {
			continue
		}
		var e scenarioAuditEntry
		json.Unmarshal([]byte(line), &e)
		entries = append(entries, e)
	}
	return entries
}

func countTool(entries []scenarioAuditEntry, tool string) int {
	n := 0
	for _, e := range entries {
		if e.Tool == tool {
			n++
		}
	}
	return n
}

func writeJSONFile(t *testing.T, dir, name string, v any) {
	t.Helper()
	data, _ := json.MarshalIndent(v, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, name), data, 0644); err != nil {
		t.Fatal(err)
	}
}

func threadIDs(th *Thinker) []string {
	var ids []string
	for _, info := range th.threads.List() {
		ids = append(ids, info.ID)
	}
	return ids
}

// --- Scenarios ---

var helpdeskScenario = Scenario{
	Name: "Helpdesk",
	Directive: `You run the support desk for my small business.
We have a helpdesk ticketing system — check it every 10-20 seconds for new tickets.
When tickets come in, look up the answer in our knowledge base, reply to the customer, and close the ticket.
Don't let more than 3 tickets be handled at the same time.`,
	MCPServers: []MCPServerConfig{{
		Name:    "helpdesk",
		Command: "", // filled in test
		Env:     map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"},
	}},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "kb.json", map[string]string{
			"hours":    "We are open Monday to Friday, 9am to 5pm.",
			"delivery": "We deliver within 10 miles for free.",
			"returns":  "You can return items within 30 days with a receipt.",
		})
		writeJSONFile(t, dir, "tickets.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — thread spawned and list_tickets called",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.threads.Count() == 0 {
					return false
				}
				entries := readAuditEntries(dir)
				lists := countTool(entries, "list_tickets")
				t.Logf("  ... list_tickets=%d threads=%v", lists, threadIDs(th))
				return lists > 0
			},
		},
		{
			Name:    "Process 2 tickets",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "tickets.json", []map[string]string{
					{"id": "t1", "question": "What are your hours?"},
					{"id": "t2", "question": "Do you deliver?"},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				replies := countTool(entries, "reply_ticket")
				closes := countTool(entries, "close_ticket")
				t.Logf("  ... lookup=%d replies=%d closes=%d threads=%v",
					countTool(entries, "lookup_kb"), replies, closes, threadIDs(th))
				return replies >= 2 && closes >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				t.Logf("Audit log (%d entries):", len(entries))
				for _, e := range entries {
					t.Logf("  %s %v", e.Tool, e.Args)
				}
				lookups := countTool(entries, "lookup_kb")
				if lookups < 2 {
					t.Logf("NOTE: lookup_kb called %d times (LLM may have answered without KB)", lookups)
				}
			},
		},
		{
			Name:    "Quiescence — workers done",
			Timeout: 45 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.threads.Count()
				t.Logf("  ... threads=%d %v", count, threadIDs(th))
				return count <= 1
			},
		},
	},
	Timeout:    3 * time.Minute,
	MaxThreads: 5,
}

// chatReply holds a parsed reply from the audit log for evaluation.
type chatReply struct {
	User    string
	Message string
}

func readChatReplies(dir string) []chatReply {
	entries := readAuditEntries(dir)
	var replies []chatReply
	for _, e := range entries {
		if e.Tool == "send_reply" {
			replies = append(replies, chatReply{User: e.Args["user"], Message: e.Args["message"]})
		}
	}
	return replies
}

// chatContainsAny checks if any reply contains at least one of the keywords (case-insensitive).
func chatContainsAny(replies []chatReply, keywords ...string) bool {
	for _, r := range replies {
		lower := strings.ToLower(r.Message)
		for _, kw := range keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				return true
			}
		}
	}
	return false
}

var chatScenario = Scenario{
	Name: "Chat",
	Directive: `You are a helpful assistant. Users send messages via [user:name] events.
When a user writes to you, spawn a thread to handle them. The thread should reply using send_reply with the user's name and your answer.
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
			Setup: func(t *testing.T, dir string) {
				// Inject via thinker — will be called with th in Wait
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Inject on first poll
				if countTool(readAuditEntries(dir), "send_reply") == 0 && th.iteration <= 2 {
					th.InjectUserMessage("alice", "What is the capital of France?")
				}
				replies := readChatReplies(dir)
				t.Logf("  ... replies=%d threads=%v", len(replies), threadIDs(th))
				return len(replies) >= 1
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := readChatReplies(dir)
				last := replies[len(replies)-1]
				t.Logf("Reply to alice: %q", last.Message)
				if !chatContainsAny(replies, "Paris") {
					t.Errorf("expected reply to mention Paris, got: %q", last.Message)
				}
			},
		},
		{
			Name:    "Follow-up question — What is its population?",
			Timeout: 60 * time.Second,
			Setup: func(t *testing.T, dir string) {},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Inject once
				replies := readChatReplies(dir)
				if len(replies) == 1 {
					th.InjectUserMessage("alice", "What is its population?")
				}
				t.Logf("  ... replies=%d", len(replies))
				return len(replies) >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := readChatReplies(dir)
				last := replies[len(replies)-1]
				t.Logf("Reply to alice: %q", last.Message)
				if !chatContainsAny(replies[len(replies)-1:], "million", "2", "11", "12") {
					t.Logf("NOTE: reply may not contain population figure: %q", last.Message)
				}
			},
		},
		{
			Name:    "Multi-user — bob asks 2+2",
			Timeout: 60 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Not using Setup because we need th
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				replies := readChatReplies(dir)
				hasBob := false
				for _, r := range replies {
					if r.User == "bob" {
						hasBob = true
					}
				}
				if !hasBob && len(replies) >= 2 {
					th.InjectUserMessage("bob", "What is 2 + 2?")
				}
				t.Logf("  ... replies=%d bob=%v threads=%v", len(replies), hasBob, threadIDs(th))
				return hasBob
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				replies := readChatReplies(dir)
				var bobReply string
				for _, r := range replies {
					if r.User == "bob" {
						bobReply = r.Message
					}
				}
				t.Logf("Bob's reply: %q", bobReply)
				if !chatContainsAny([]chatReply{{Message: bobReply}}, "4") {
					t.Errorf("expected reply to contain '4', got: %q", bobReply)
				}
			},
		},
	},
	Timeout:    3 * time.Minute,
	MaxThreads: 5,
}

var bakeryScenario = Scenario{
	Name: "Bakery",
	Directive: `You manage a small bakery with two team members:
1. Spawn an "order-clerk" thread that monitors new orders. It can only use the orders system. When it finds a pending order, it sends the order details to main and waits for instructions.
2. Spawn a "stock-keeper" thread that manages inventory. It can only use the inventory system. When main asks it to check or use stock, it does so and reports back.

When order-clerk reports a new order, ask stock-keeper to check if we have enough. If yes, tell stock-keeper to deduct the stock, then tell order-clerk to mark it preparing then ready. If not enough stock, tell order-clerk to cancel it.

Both threads must stay at normal pace and never sleep — they are permanent workers.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "orders",
			Command: "", // filled in test
			Env:     map[string]string{"ORDERS_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "inventory",
			Command: "", // filled in test
			Env:     map[string]string{"INVENTORY_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "stock.json", map[string]int{
			"croissant": 10,
			"baguette":  5,
			"muffin":    0,
		})
		writeJSONFile(t, dir, "orders.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — both workers spawned",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.threads.Count()
				t.Logf("  ... threads=%d %v", count, threadIDs(th))
				return count >= 2
			},
		},
		{
			Name:    "Simple order — croissant x2",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "orders.json", []map[string]any{
					{"id": "o1", "item": "croissant", "qty": 2, "status": "pending"},
				})
			},
			// Note: we don't wake the thread — it should check on its own cycle
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				checks := countTool(entries, "check_stock")
				uses := countTool(entries, "use_stock")
				updates := countTool(entries, "update_order")
				t.Logf("  ... check=%d use=%d update=%d threads=%v",
					checks, uses, updates, threadIDs(th))
				// Need at least: check_stock + use_stock + update to preparing + update to ready
				return uses >= 1 && updates >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				t.Logf("Phase 2 audit (%d entries):", len(entries))
				for _, e := range entries {
					t.Logf("  %s %v", e.Tool, e.Args)
				}
				// Verify stock was deducted
				hasUse := false
				for _, e := range entries {
					if e.Tool == "use_stock" && e.Args["item"] == "croissant" {
						hasUse = true
					}
				}
				if !hasUse {
					t.Logf("NOTE: use_stock for croissant not found — agent may have used a different approach")
				}
			},
		},
		{
			Name:    "Out of stock — muffin x3",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "orders.json", []map[string]any{
					{"id": "o2", "item": "muffin", "qty": 3, "status": "pending"},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				// Look for o2 being cancelled
				for _, e := range entries {
					if e.Tool == "update_order" && e.Args["id"] == "o2" && e.Args["status"] == "cancelled" {
						return true
					}
				}
				updates := countTool(entries, "update_order")
				t.Logf("  ... updates=%d threads=%v", updates, threadIDs(th))
				return false
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				// Verify no use_stock for muffin (should not deduct when out of stock)
				for _, e := range entries {
					if e.Tool == "use_stock" && e.Args["item"] == "muffin" {
						t.Logf("NOTE: use_stock was called for muffin — should have been skipped (0 stock)")
					}
				}
			},
		},
		{
			Name:    "Batch — 3 orders, one should fail",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "orders.json", []map[string]any{
					{"id": "o3", "item": "baguette", "qty": 2, "status": "pending"},
					{"id": "o4", "item": "croissant", "qty": 3, "status": "pending"},
					{"id": "o5", "item": "baguette", "qty": 5, "status": "pending"},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				// Count update_order calls for o3, o4, o5
				processed := map[string]bool{}
				for _, e := range entries {
					if e.Tool == "update_order" {
						id := e.Args["id"]
						if id == "o3" || id == "o4" || id == "o5" {
							s := e.Args["status"]
							if s == "ready" || s == "cancelled" {
								processed[id] = true
							}
						}
					}
				}
				t.Logf("  ... processed=%v threads=%v", processed, threadIDs(th))
				return len(processed) >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				t.Logf("Phase 4 audit (%d total entries):", len(entries))
				// Show only batch-related
				for _, e := range entries {
					if e.Args["id"] == "o3" || e.Args["id"] == "o4" || e.Args["id"] == "o5" ||
						e.Args["item"] == "baguette" || e.Args["item"] == "croissant" {
						t.Logf("  %s %v", e.Tool, e.Args)
					}
				}
			},
		},
		{
			Name:    "Quiescence — workers still alive, no pending orders",
			Timeout: 30 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Both permanent workers should still be running
				// Just verify no pending orders remain in the system
				entries := readAuditEntries(dir)
				// Count orders that reached a final state (ready or cancelled)
				final := 0
				for _, e := range entries {
					if e.Tool == "update_order" && (e.Args["status"] == "ready" || e.Args["status"] == "cancelled") {
						final++
					}
				}
				t.Logf("  ... threads=%d final_orders=%d %v", th.threads.Count(), final, threadIDs(th))
				// o1 + o2 + o3 + o4 + o5 = 5 orders all resolved
				return final >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Both permanent workers should still be alive
				count := th.threads.Count()
				if count < 2 {
					t.Errorf("expected 2 permanent workers still alive, got %d: %v", count, threadIDs(th))
				}
			},
		},
	},
	Timeout:    5 * time.Minute,
	MaxThreads: 5,
}

var socialTeamScenario = Scenario{
	Name: "SocialTeam",
	Directive: `You manage social media for a small coffee shop called "Bean & Brew".
Spawn three permanent team members:
1. A planner who checks the content schedule for slots that need content
2. A creative who generates post text and images when asked
3. A social manager who posts content to channels when given ready content

When planner finds a planned slot, coordinate the team: ask creative to generate
a post and image for the topic and channel, then give the content to social manager
to post it, then have planner update the schedule slot to posted.
All team members should stay at normal pace.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "schedule",
			Command: "", // filled in test
			Env:     map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "creative",
			Command: "", // filled in test
			Env:     map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "social",
			Command: "", // filled in test
			Env:     map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "schedule.json", []map[string]string{
			{"id": "s1", "channel": "twitter", "topic": "Monday morning coffee special", "time": "09:00", "status": "planned"},
			{"id": "s2", "channel": "instagram", "topic": "New seasonal latte art", "time": "12:00", "status": "planned"},
		})
		writeJSONFile(t, dir, "posts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 team members spawned",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.threads.Count()
				t.Logf("  ... threads=%d %v", count, threadIDs(th))
				return count >= 3
			},
		},
		{
			Name:    "Content pipeline — 2 posts created and published",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				posts := countTool(entries, "post")
				generates := countTool(entries, "generate_post")
				images := countTool(entries, "generate_image")
				updates := countTool(entries, "update_slot")
				t.Logf("  ... generate_post=%d generate_image=%d post=%d update_slot=%d threads=%v",
					generates, images, posts, updates, threadIDs(th))
				return posts >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				t.Logf("Pipeline audit (%d entries):", len(entries))
				for _, e := range entries {
					if e.Tool != "get_schedule" {
						t.Logf("  %s %v", e.Tool, e.Args)
					}
				}
				generates := countTool(entries, "generate_post")
				if generates < 2 {
					t.Logf("NOTE: generate_post called %d times (expected 2)", generates)
				}
				// Check posts were actually published
				for _, e := range entries {
					if e.Tool == "post" {
						if e.Args["channel"] == "" || e.Args["content"] == "" {
							t.Errorf("post missing channel or content: %v", e.Args)
						}
					}
				}
			},
		},
		{
			Name:    "New slot — linkedin hiring post",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Read current schedule and append new slot
				data, _ := os.ReadFile(filepath.Join(dir, "schedule.json"))
				var slots []map[string]string
				json.Unmarshal(data, &slots)
				slots = append(slots, map[string]string{
					"id": "s3", "channel": "linkedin", "topic": "Hiring baristas for summer", "time": "15:00", "status": "planned",
				})
				writeJSONFile(t, dir, "schedule.json", slots)
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				posts := countTool(entries, "post")
				t.Logf("  ... posts=%d threads=%v", posts, threadIDs(th))
				return posts >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				// Check linkedin post exists
				hasLinkedin := false
				for _, e := range entries {
					if e.Tool == "post" && e.Args["channel"] == "linkedin" {
						hasLinkedin = true
						t.Logf("LinkedIn post: %s", e.Args["content"])
					}
				}
				if !hasLinkedin {
					t.Logf("NOTE: no linkedin post found in audit")
				}
			},
		},
		{
			Name:    "Quiescence — 3 workers alive, all slots processed",
			Timeout: 30 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := readAuditEntries(dir)
				posts := countTool(entries, "post")
				t.Logf("  ... threads=%d posts=%d %v", th.threads.Count(), posts, threadIDs(th))
				return posts >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				count := th.threads.Count()
				if count < 3 {
					t.Errorf("expected 3 permanent workers alive, got %d: %v", count, threadIDs(th))
				}
			},
		},
	},
	Timeout:    5 * time.Minute,
	MaxThreads: 5,
}

// --- Test functions ---

func TestScenario_Helpdesk(t *testing.T) {
	bin := buildMCPBinary(t, "mcps/helpdesk")
	t.Logf("built mcp-helpdesk: %s", bin)

	s := helpdeskScenario
	s.MCPServers[0].Command = bin
	runScenario(t, s)
}

func TestScenario_Chat(t *testing.T) {
	bin := buildMCPBinary(t, "mcps/chat")
	t.Logf("built mcp-chat: %s", bin)

	s := chatScenario
	s.MCPServers[0].Command = bin
	runScenario(t, s)
}

func TestScenario_SocialTeam(t *testing.T) {
	scheduleBin := buildMCPBinary(t, "mcps/schedule")
	creativeBin := buildMCPBinary(t, "mcps/creative")
	socialBin := buildMCPBinary(t, "mcps/social")
	t.Logf("built schedule: %s, creative: %s, social: %s", scheduleBin, creativeBin, socialBin)

	s := socialTeamScenario
	s.MCPServers[0].Command = scheduleBin
	s.MCPServers[1].Command = creativeBin
	s.MCPServers[2].Command = socialBin
	runScenario(t, s)
}

func TestScenario_Bakery(t *testing.T) {
	ordersBin := buildMCPBinary(t, "mcps/orders")
	inventoryBin := buildMCPBinary(t, "mcps/inventory")
	t.Logf("built mcp-orders: %s", ordersBin)
	t.Logf("built mcp-inventory: %s", inventoryBin)

	s := bakeryScenario
	s.MCPServers[0].Command = ordersBin
	s.MCPServers[1].Command = inventoryBin
	runScenario(t, s)
}
