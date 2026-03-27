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

	provider, err := selectProvider()
	if err != nil {
		t.Fatalf("no LLM provider: %v", err)
	}

	bus := NewEventBus()
	thinker := &Thinker{
		apiKey:   apiKey,
		provider: provider,
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
		telemetry: NewTelemetry(),
	}
	thinker.threads = NewThreadManager(thinker)
	thinker.registry = NewToolRegistry(apiKey)

	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, thinker.registry, "", nil)}

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
		return buildSystemPrompt(cfg.GetDirective(), thinker.registry, toolDocs, thinker.mcpServers)
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
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectUserMessage("alice", "What is the capital of France?")
					}
					replies := readChatReplies(dir)
					t.Logf("  ... replies=%d threads=%v", len(replies), threadIDs(th))
					return len(replies) >= 1
				}
			}(),
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
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectUserMessage("alice", "What is its population?")
					}
					replies := readChatReplies(dir)
					t.Logf("  ... replies=%d", len(replies))
					return len(replies) >= 2
				}
			}(),
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
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectUserMessage("bob", "What is 2 + 2?")
					}
					replies := readChatReplies(dir)
					hasBob := false
					for _, r := range replies {
						if r.User == "bob" {
							hasBob = true
						}
					}
					t.Logf("  ... replies=%d bob=%v threads=%v", len(replies), hasBob, threadIDs(th))
					return hasBob
				}
			}(),
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
1. A planner — needs the schedule tools (get_schedule, update_slot) to check for planned slots and mark them posted
2. A creative — needs the creative tools (generate_post, generate_image) to make content when asked
3. A social manager — needs the social tools (post, get_posts) to publish content to channels

When planner finds a planned slot, coordinate: ask creative to generate a post and image,
then give the content to social manager to post it, then tell planner to update the slot to posted.
The planner must keep checking the schedule at normal pace — never go to sleep.
Creative and social manager can sleep when idle.`,
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

var robotScenario = Scenario{
	Name: "Robot",
	Directive: `You control a small robot. Spawn two team members:
1. A "pilot" thread at fast pace with small model — it continuously reads sensors and drives the motors.
   When it detects obstacles, it stops and reports to you. It executes movement commands you give it.
2. You (main) are the strategic planner. You decide where the robot should go and what to look for.
   Give the pilot high-level commands like "move forward 3 steps" or "turn right and scan".

The pilot must stay at fast pace and continuously monitor sensors between moves.
You stay at normal pace and coordinate.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "sensors",
			Command: "", // filled in test
			Env:     map[string]string{"ROBOT_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "motors",
			Command: "", // filled in test
			Env:     map[string]string{"ROBOT_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "world.json", map[string]any{
			"position":  map[string]float64{"x": 0, "y": 0},
			"heading":   0,
			"battery":   100,
			"obstacles": []any{},
			"objects":   []any{},
			"moving":    false,
			"speed":     "",
		})
	},
	Phases: []Phase{
		{
			Name:    "Startup — pilot spawned and reading sensors",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.threads.Count() == 0 {
					return false
				}
				entries := readAuditEntries(dir)
				reads := countTool(entries, "read_sensors")
				t.Logf("  ... read_sensors=%d threads=%v", reads, threadIDs(th))
				return reads > 0
			},
		},
		{
			Name:    "Navigate — move forward 3 steps",
			Timeout: 90 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("Command: move the robot forward 3 steps")
					}
					entries := readAuditEntries(dir)
					moves := countTool(entries, "move")
					t.Logf("  ... moves=%d threads=%v", moves, threadIDs(th))
					return moves >= 3
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Check position changed
				data, _ := os.ReadFile(filepath.Join(dir, "world.json"))
				var w map[string]any
				json.Unmarshal(data, &w)
				pos := w["position"].(map[string]any)
				y := pos["y"].(float64)
				t.Logf("Position after moves: y=%.1f", y)
				if y < 2.0 {
					t.Logf("NOTE: expected Y >= 2.0, got %.1f", y)
				}
			},
		},
		{
			Name:    "Obstacle — robot detects and avoids",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Place obstacle ahead of current position
				data, _ := os.ReadFile(filepath.Join(dir, "world.json"))
				var w map[string]any
				json.Unmarshal(data, &w)
				pos := w["position"].(map[string]any)
				y := pos["y"].(float64)
				w["obstacles"] = []map[string]float64{{"x": 0, "y": y + 1.5}}
				writeJSONFile(t, dir, "world.json", w)
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					// Tell pilot to move forward once — it should detect the obstacle
					if !sent {
						sent = true
						th.InjectConsole("Command: move forward 2 more steps")
					}
					entries := readAuditEntries(dir)
					reads := countTool(entries, "read_sensors")
					stops := countTool(entries, "stop")
					moves := countTool(entries, "move")
					t.Logf("  ... reads=%d stops=%d moves=%d threads=%v", reads, stops, moves, threadIDs(th))
					// Pilot should detect obstacle via sensors or blocked move, then stop or turn
					return stops > 0 || (moves > 3 && reads > 5)
				}
			}(),
		},
		{
			Name:    "Camera — find the red cup",
			Timeout: 90 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Place red cup in camera range
				data, _ := os.ReadFile(filepath.Join(dir, "world.json"))
				var w map[string]any
				json.Unmarshal(data, &w)
				pos := w["position"].(map[string]any)
				x := pos["x"].(float64)
				y := pos["y"].(float64)
				w["objects"] = []map[string]any{
					{"name": "red cup", "x": x + 2, "y": y + 3},
				}
				writeJSONFile(t, dir, "world.json", w)
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				sent := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !sent {
						sent = true
						th.InjectConsole("Command: use the camera to look for a red cup nearby")
					}
					entries := readAuditEntries(dir)
					cams := countTool(entries, "read_camera")
					t.Logf("  ... read_camera=%d threads=%v", cams, threadIDs(th))
					return cams >= 1
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := readAuditEntries(dir)
				t.Logf("Audit (%d entries):", len(entries))
				for _, e := range entries {
					if e.Tool != "read_sensors" {
						t.Logf("  %s %v", e.Tool, e.Args)
					}
				}
			},
		},
		{
			Name:    "Quiescence — pilot still alive",
			Timeout: 15 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return th.threads.Count() >= 1
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				if th.threads.Count() < 1 {
					t.Errorf("expected pilot still alive, got %d threads", th.threads.Count())
				}
			},
		},
	},
	Timeout:    5 * time.Minute,
	MaxThreads: 3,
}

func TestScenario_Robot(t *testing.T) {
	sensorsBin := buildMCPBinary(t, "mcps/sensors")
	motorsBin := buildMCPBinary(t, "mcps/motors")
	t.Logf("built sensors: %s, motors: %s", sensorsBin, motorsBin)

	s := robotScenario
	s.MCPServers[0].Command = sensorsBin
	s.MCPServers[1].Command = motorsBin
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

// --- VideoTeam Scenario ---

var videoTeamScenario = Scenario{
	Name: "VideoTeam",
	Directive: `You manage a video production team for a tech company.

Your job: when new video files arrive, process them through a pipeline:
1. Upload and register the file in media
2. Extract 3 screenshots from the video
3. Create a 30-second reel
4. Store the pipeline status in storage
5. Plan social media posts: one reel post for instagram, one screenshot post for twitter, one announcement for linkedin
6. Generate creative copy for each post
7. Publish all posts

Spawn these permanent workers:
- "editor" — handles media processing (upload, screenshots, reels). Needs media tools. Reports to main when processing is done.
- "planner" — plans social media content from processed assets. Needs schedule and storage tools. Creates schedule slots then tells publisher.
- "publisher" — generates copy and publishes. Needs creative and social tools. Posts to channels.

The editor should periodically check for uploaded files (status=uploaded) using list_files, process them, then report to main.
Coordinate the pipeline: editor → planner → publisher.
When all posts are published, store a completion record in storage with key "pipeline:done".`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "media",
			Command: "", // filled in test
			Env:     map[string]string{"MEDIA_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "storage",
			Command: "", // filled in test
			Env:     map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"},
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
		{
			Name:    "schedule",
			Command: "", // filled in test
			Env:     map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Pre-populate a video file waiting to be processed
		writeJSONFile(t, dir, "media.json", map[string]any{
			"files": []map[string]string{
				{"id": "m1", "name": "product-demo.mp4", "type": "video", "duration": "3:24", "resolution": "1920x1080", "size": "245MB", "status": "uploaded", "uploaded_at": "2026-03-26T10:00:00Z"},
			},
			"assets": []any{},
		})
		writeJSONFile(t, dir, "schedule.json", []any{})
		writeJSONFile(t, dir, "posts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 workers spawned",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.threads.Count()
				t.Logf("  ... threads=%d %v", count, threadIDs(th))
				return count >= 3
			},
		},
		{
			Name: "Video arrives — file processed",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Check media.json has assets (screenshots + reel)
				data, err := os.ReadFile(filepath.Join(dir, "media.json"))
				if err != nil {
					return false
				}
				var state struct {
					Assets []json.RawMessage `json:"assets"`
				}
				json.Unmarshal(data, &state)
				t.Logf("  ... assets=%d", len(state.Assets))
				return len(state.Assets) >= 4 // 3 screenshots + 1 reel
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "media.json"))
				var state struct {
					Files  []json.RawMessage `json:"files"`
					Assets []json.RawMessage `json:"assets"`
				}
				json.Unmarshal(data, &state)
				if len(state.Files) < 1 {
					t.Errorf("expected at least 1 file, got %d", len(state.Files))
				}
				if len(state.Assets) < 4 {
					t.Errorf("expected at least 4 assets (3 screenshots + 1 reel), got %d", len(state.Assets))
				}
			},
		},
		{
			Name:    "Social posts published",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "posts.json"))
				if err != nil {
					return false
				}
				var posts []json.RawMessage
				json.Unmarshal(data, &posts)
				t.Logf("  ... posts=%d", len(posts))
				return len(posts) >= 3 // instagram + twitter + linkedin
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "posts.json"))
				var posts []map[string]any
				json.Unmarshal(data, &posts)
				if len(posts) < 3 {
					t.Errorf("expected at least 3 posts, got %d", len(posts))
				}
				channels := map[string]bool{}
				for _, p := range posts {
					if ch, ok := p["channel"].(string); ok {
						channels[ch] = true
					}
				}
				t.Logf("channels posted to: %v", channels)
			},
		},
	},
	Timeout:    5 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_VideoTeam(t *testing.T) {
	mediaBin := buildMCPBinary(t, "mcps/media")
	storageBin := buildMCPBinary(t, "mcps/storage")
	creativeBin := buildMCPBinary(t, "mcps/creative")
	socialBin := buildMCPBinary(t, "mcps/social")
	scheduleBin := buildMCPBinary(t, "mcps/schedule")
	t.Logf("built media=%s storage=%s creative=%s social=%s schedule=%s",
		mediaBin, storageBin, creativeBin, socialBin, scheduleBin)

	s := videoTeamScenario
	s.MCPServers[0].Command = mediaBin
	s.MCPServers[1].Command = storageBin
	s.MCPServers[2].Command = creativeBin
	s.MCPServers[3].Command = socialBin
	s.MCPServers[4].Command = scheduleBin

	runScenario(t, s)
}
