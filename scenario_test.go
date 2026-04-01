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
	scenarioStart := time.Now()
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

	// Token/cost tracking
	var totalPrompt, totalCached, totalCompletion atomic.Int64
	var iterCount atomic.Int64

	// Observer: log events
	obs := thinker.bus.SubscribeAll("test-observer", 500)
	go func() {
		for !stopped.Load() {
			select {
			case ev := <-obs.C:
				switch ev.Type {
				case EventThinkDone:
					totalPrompt.Add(int64(ev.Usage.PromptTokens))
					totalCached.Add(int64(ev.Usage.CachedTokens))
					totalCompletion.Add(int64(ev.Usage.CompletionTokens))
					iterCount.Add(1)
					cost := calculateCostForProvider(thinker.provider, ev.Usage)
					t.Logf("[%s iter %d] threads=%d rate=%s tok=%d/%d/$%.4f tools=%v events=%d",
						ev.From, ev.Iteration, ev.ThreadCount, ev.Rate,
						ev.Usage.PromptTokens, ev.Usage.CompletionTokens, cost,
						ev.ToolCalls, len(ev.ConsumedEvents))
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

	// Token/cost summary
	prompt := totalPrompt.Load()
	cached := totalCached.Load()
	completion := totalCompletion.Load()
	iters := iterCount.Load()
	totalTok := prompt + completion
	totalCost := calculateCostForProvider(thinker.provider, TokenUsage{
		PromptTokens: int(prompt), CachedTokens: int(cached), CompletionTokens: int(completion),
	})
	elapsed := time.Since(scenarioStart)

	t.Logf("────────────────────────────────────────")
	t.Logf("Scenario: %s", s.Name)
	t.Logf("Duration: %s | Iterations: %d | Peak threads: %d", elapsed.Round(time.Second), iters, peak)
	t.Logf("Tokens:   %d total (in:%d cached:%d out:%d)", totalTok, prompt, cached, completion)
	t.Logf("Cost:     $%.4f | Provider: %s", totalCost, thinker.provider.Name())
	t.Logf("────────────────────────────────────────")

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

	// Use a clean config (no persisted provider) so env vars control provider selection
	cleanCfg := &Config{path: filepath.Join(tmpDir, "provider.json")}
	provider, err := selectProvider(cleanCfg)
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

	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, thinker.registry, "", nil, nil)}

	go thinker.registry.EmbedAll(memStore)

	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(cfg.GetDirective(), thinker.registry, toolDocs, thinker.mcpServers, nil)
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
						th.InjectConsole("What is its population?")
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
						th.InjectConsole("What is 2 + 2?")
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

// --- Lead Team Scenario ---

var leadTeamScenario = Scenario{
	Name: "LeadTeam",
	Directive: `You manage a lead processing pipeline for a business running Facebook ads.

Spawn and maintain 3 threads:
1. "file-intake" — receives file URLs, fetches them, checks for duplicates, marks as pending.
   Tools: files_fetch_file, files_list_files, files_file_status, send, done
2. "file-processor" — reads CSV files, extracts leads, records ad spend.
   Tools: files_read_csv, files_file_status, ads_record_spend, storage_store, send, done
3. "ad-monitor" — checks ad performance periodically, pauses over-budget ads, sends alerts.
   Tools: ads_get_performance, ads_get_budgets, ads_pause_ad, ads_get_alerts, send, done

When you receive a console event with a file URL, forward it to file-intake.
When file-intake finishes, tell file-processor to process it.
When file-processor finishes, tell ad-monitor to check performance.`,
	MCPServers: []MCPServerConfig{
		{Name: "files", Command: "", Env: map[string]string{"FILES_DATA_DIR": "{{dataDir}}"}},
		{Name: "ads", Command: "", Env: map[string]string{"ADS_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed ad budgets
		writeJSONFile(t, dir, "budgets.json", map[string]*struct {
			AdID        string  `json:"ad_id"`
			DailyBudget float64 `json:"daily_budget"`
			MaxCPL      float64 `json:"max_cpl"`
			Status      string  `json:"status"`
			UpdatedAt   string  `json:"updated_at"`
		}{
			"fb-summer-2026":  {AdID: "fb-summer-2026", DailyBudget: 100, MaxCPL: 10.0, Status: "active", UpdatedAt: "2026-03-01T00:00:00Z"},
			"fb-winter-promo": {AdID: "fb-winter-promo", DailyBudget: 50, MaxCPL: 15.0, Status: "active", UpdatedAt: "2026-03-01T00:00:00Z"},
		})

		// Create CSV batch 1 — normal leads, within budget
		csv1 := "name,email,phone,ad_id,cost\n" +
			"Alice Smith,alice@example.com,555-0101,fb-summer-2026,8.50\n" +
			"Bob Jones,bob@example.com,555-0102,fb-summer-2026,9.20\n" +
			"Carol White,carol@example.com,555-0103,fb-winter-promo,12.00\n"
		os.WriteFile(filepath.Join(dir, "leads-batch-1.csv"), []byte(csv1), 0644)

		// Create CSV batch 2 — expensive leads that push fb-summer-2026 over CPL limit
		csv2 := "name,email,phone,ad_id,cost\n" +
			"Dave Brown,dave@example.com,555-0201,fb-summer-2026,25.00\n" +
			"Eve Black,eve@example.com,555-0202,fb-summer-2026,30.00\n" +
			"Frank Green,frank@example.com,555-0203,fb-summer-2026,28.00\n"
		os.WriteFile(filepath.Join(dir, "leads-batch-2.csv"), []byte(csv2), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				ids := threadIDs(th)
				return len(ids) >= 3
			},
		},
		{
			Name:    "File ingestion — batch 1 processed",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-1.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
					}
					// Check if file was processed
					data, err := os.ReadFile(filepath.Join(dir, "files.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), `"processed"`)
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify file record exists
				data, _ := os.ReadFile(filepath.Join(dir, "files.json"))
				if !strings.Contains(string(data), "leads-batch-1.csv") {
					t.Error("expected leads-batch-1.csv in files.json")
				}
			},
		},
		{
			Name:    "Duplicate rejection — same file rejected",
			Timeout: 90 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				startTime := time.Now()
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-1.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
						startTime = time.Now()
					}
					// Wait a bit for the system to process and verify no new file added
					return time.Since(startTime) > 15*time.Second
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "files.json"))
				// Should still only have 1 file entry (the original)
				var files map[string]any
				json.Unmarshal(data, &files)
				if len(files) != 1 {
					t.Errorf("expected 1 file record (duplicate rejected), got %d", len(files))
				}
			},
		},
		{
			Name:    "Ad monitoring — expensive batch triggers pause",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-2.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
					}
					// Check if fb-summer-2026 was paused
					data, err := os.ReadFile(filepath.Join(dir, "budgets.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), `"paused"`)
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify the right ad was paused
				data, _ := os.ReadFile(filepath.Join(dir, "budgets.json"))
				var budgets map[string]json.RawMessage
				json.Unmarshal(data, &budgets)
				if b, ok := budgets["fb-summer-2026"]; ok {
					if !strings.Contains(string(b), `"paused"`) {
						t.Error("expected fb-summer-2026 to be paused")
					}
				} else {
					t.Error("fb-summer-2026 not found in budgets")
				}
				// Verify winter promo still active
				if b, ok := budgets["fb-winter-promo"]; ok {
					if !strings.Contains(string(b), `"active"`) {
						t.Error("expected fb-winter-promo to still be active")
					}
				}
				// Verify alert was created
				alertData, _ := os.ReadFile(filepath.Join(dir, "alerts.json"))
				if !strings.Contains(string(alertData), "fb-summer-2026") {
					t.Error("expected alert for fb-summer-2026")
				}
			},
		},
	},
	Timeout:    5 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_LeadTeam(t *testing.T) {
	filesBin := buildMCPBinary(t, "mcps/files")
	adsBin := buildMCPBinary(t, "mcps/ads")
	storageBin := buildMCPBinary(t, "mcps/storage")
	t.Logf("built files=%s ads=%s storage=%s", filesBin, adsBin, storageBin)

	s := leadTeamScenario
	s.MCPServers[0].Command = filesBin
	s.MCPServers[1].Command = adsBin
	s.MCPServers[2].Command = storageBin
	runScenario(t, s)
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

// --- DevTeam Scenario ---

// seedTodoApp writes a minimal Go todo app into the given directory.
func seedTodoApp(t *testing.T, dir string) {
	t.Helper()
	appDir := filepath.Join(dir, "app")
	os.MkdirAll(appDir, 0755)

	// go.mod
	os.WriteFile(filepath.Join(appDir, "go.mod"), []byte("module todo\n\ngo 1.21\n"), 0644)

	// todo.go — basic CRUD, no priority field
	os.WriteFile(filepath.Join(appDir, "todo.go"), []byte(`package todo

type Todo struct {
	ID        int    `+"`"+`json:"id"`+"`"+`
	Title     string `+"`"+`json:"title"`+"`"+`
	Completed bool   `+"`"+`json:"completed"`+"`"+`
}

var todos []Todo
var nextID = 1

func Create(title string) Todo {
	t := Todo{ID: nextID, Title: title}
	nextID++
	todos = append(todos, t)
	return t
}

func List() []Todo {
	return todos
}

func Complete(id int) bool {
	for i := range todos {
		if todos[i].ID == id {
			todos[i].Completed = true
			return true
		}
	}
	return false
}

func Delete(id int) bool {
	for i := range todos {
		if todos[i].ID == id {
			todos = append(todos[:i], todos[i+1:]...)
			return true
		}
	}
	return false
}

func Reset() {
	todos = nil
	nextID = 1
}
`), 0644)

	// todo_test.go — basic tests
	os.WriteFile(filepath.Join(appDir, "todo_test.go"), []byte(`package todo

import "testing"

func TestCreate(t *testing.T) {
	Reset()
	td := Create("Buy milk")
	if td.Title != "Buy milk" {
		t.Errorf("expected 'Buy milk', got %q", td.Title)
	}
	if td.ID != 1 {
		t.Errorf("expected ID 1, got %d", td.ID)
	}
}

func TestList(t *testing.T) {
	Reset()
	Create("Task 1")
	Create("Task 2")
	if len(List()) != 2 {
		t.Errorf("expected 2 todos, got %d", len(List()))
	}
}

func TestComplete(t *testing.T) {
	Reset()
	td := Create("Do laundry")
	if !Complete(td.ID) {
		t.Error("expected Complete to return true")
	}
	if !List()[0].Completed {
		t.Error("expected todo to be completed")
	}
}

func TestDelete(t *testing.T) {
	Reset()
	td := Create("Temp")
	if !Delete(td.ID) {
		t.Error("expected Delete to return true")
	}
	if len(List()) != 0 {
		t.Error("expected empty list after delete")
	}
}
`), 0644)

	// test.sh at root level (codebase dir) since run_tests runs from there
	os.WriteFile(filepath.Join(dir, "test.sh"), []byte("#!/bin/bash\ncd app && go test ./... 2>&1\n"), 0644)
}

var devTeamScenario = Scenario{
	Name: "DevTeam",
	Directive: `You manage a small development team maintaining a Todo SaaS app.
The codebase is in the "app/" directory. It is a Go package with todo.go and todo_test.go.

Spawn and maintain 3 threads:
1. "support" — monitors helpdesk tickets, triages them (bug vs feature), reports to main with recommendations.
   Tools: helpdesk_list_tickets, helpdesk_reply_ticket, helpdesk_close_ticket, send, done
2. "dev" — reads/writes code, implements features and fixes. Always reads existing code before modifying.
   Tools: codebase_read_file, codebase_write_file, codebase_list_files, codebase_search, send, done
3. "qa" — runs the test suite and reports results. Triggered by main after dev finishes.
   Tools: codebase_run_tests, codebase_read_file, send, done

Workflow:
- Support finds a ticket and tells you what it is
- You decide what to do and tell dev to implement it
- After dev is done, tell qa to run tests
- If tests fail, send dev back to fix. If pass, tell support to close the ticket.`,
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		seedTodoApp(t, dir)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
		},
		{
			Name:    "Feature request — add priority field",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "tickets.json", []map[string]string{
					{"id": "T-101", "question": "Feature request: Please add a Priority field to todos. It should be a string with values low, medium, or high. Default to low. The Create function should accept an optional priority parameter."},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				code, err := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if err != nil {
					return false
				}
				if !strings.Contains(string(code), "Priority") {
					return false
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				code, _ := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if !strings.Contains(string(code), "Priority") {
					t.Error("expected Priority field in todo.go")
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("tests should pass after feature: %s", string(out))
				}
			},
		},
		{
			Name:    "Bug fix — empty title validation",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "tickets.json", []map[string]string{
					{"id": "T-102", "question": "Bug report: Creating a todo with an empty title succeeds but it should not. The Create function should return an error when the title is empty."},
				})
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				code, err := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if err != nil {
					return false
				}
				if !strings.Contains(string(code), "error") && !strings.Contains(string(code), "Error") {
					return false
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				code, _ := os.ReadFile(filepath.Join(dir, "app", "todo.go"))
				if !strings.Contains(string(code), "error") && !strings.Contains(string(code), "Error") {
					t.Error("expected error handling for empty title in todo.go")
				}
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Errorf("tests should pass after bug fix: %s", string(out))
				}
			},
		},
	},
	Timeout:    8 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_DevTeam(t *testing.T) {
	helpdeskBin := buildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := buildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := devTeamScenario
	s.MCPServers[0].Command = helpdeskBin
	s.MCPServers[1].Command = codebaseBin
	runScenario(t, s)
}

// --- Ecommerce Scenario ---

var ecommerceScenario = Scenario{
	Name: "Ecommerce",
	Directive: `You manage order fulfillment for an online bakery.

Spawn and maintain 3 threads:
1. "warehouse" — checks inventory for pending orders, reserves stock, marks orders as ready.
   Tools: inventory_check_stock, inventory_use_stock, inventory_list_stock, orders_get_orders, orders_get_order, orders_update_order, send, done
2. "shipping" — picks up ready orders, marks them as shipped, stores tracking info.
   Tools: orders_get_orders, orders_update_order, storage_store, send, done
3. "comms" — sends customer notifications when orders ship.
   Tools: pushover_send_notification, storage_get, send, done

Workflow:
- When you receive a console event about new orders, tell warehouse to process them.
- Warehouse checks stock, reserves ingredients, marks order as "ready".
- If out of stock, warehouse reports to you and you notify comms.
- When warehouse finishes, tell shipping to dispatch.
- When shipping finishes, tell comms to notify the customer.`,
	MCPServers: []MCPServerConfig{
		{Name: "orders", Command: "", Env: map[string]string{"ORDERS_DATA_DIR": "{{dataDir}}"}},
		{Name: "inventory", Command: "", Env: map[string]string{"INVENTORY_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "stock.json", map[string]int{
			"chocolate cake": 10, "croissant": 50, "baguette": 30, "muffin": 25, "chocolate truffle": 8,
		})
		writeJSONFile(t, dir, "orders.json", []map[string]any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
		},
		{
			Name:    "Order fulfillment — process and ship",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "orders.json", []map[string]string{
					{"id": "ORD-001", "item": "chocolate cake", "qty": "2", "status": "pending"},
					{"id": "ORD-002", "item": "croissant", "qty": "12", "status": "pending"},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New orders received: ORD-001 (chocolate cake x2), ORD-002 (croissant x12). Please process them.")
						injected = true
					}
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					// Check if at least one order was updated beyond pending
					return strings.Contains(string(data), "ready") || strings.Contains(string(data), "shipped")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				s := string(data)
				if !strings.Contains(s, "ready") && !strings.Contains(s, "shipped") {
					t.Error("expected at least one order to be ready or shipped")
				}
			},
		},
		{
			Name:    "Out of stock — chocolate depleted",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				writeJSONFile(t, dir, "stock.json", map[string]int{
					"chocolate cake": 10, "croissant": 50, "baguette": 30, "muffin": 25, "chocolate truffle": 0,
				})
				writeJSONFile(t, dir, "orders.json", []map[string]string{
					{"id": "ORD-003", "item": "chocolate truffle", "qty": "5", "status": "pending"},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New order: ORD-003 (chocolate truffle x5). Please process. If out of stock, mark the order as cancelled.")
						injected = true
					}
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "cancelled")
				}
			}(),
		},
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_Ecommerce(t *testing.T) {
	ordersBin := buildMCPBinary(t, "mcps/orders")
	inventoryBin := buildMCPBinary(t, "mcps/inventory")
	storageBin := buildMCPBinary(t, "mcps/storage")
	pushoverBin := buildMCPBinary(t, "mcps/pushover")
	t.Logf("built orders=%s inventory=%s storage=%s pushover=%s", ordersBin, inventoryBin, storageBin, pushoverBin)

	s := ecommerceScenario
	s.MCPServers[0].Command = ordersBin
	s.MCPServers[1].Command = inventoryBin
	s.MCPServers[2].Command = storageBin
	s.MCPServers[3].Command = pushoverBin
	runScenario(t, s)
}

// --- Incident Scenario ---

var incidentScenario = Scenario{
	Name: "Incident",
	Directive: `You are the on-call SRE coordinator for a web platform with services: api, web, worker.

Spawn and maintain 3 threads:
1. "monitor" — continuously reads metrics for all services, watches for threshold violations.
   Tools: metrics_get_metrics, metrics_get_history, metrics_set_threshold, metrics_get_alerts, send, done
2. "responder" — investigates alerts, reads config/logs, applies fixes.
   Tools: codebase_read_file, codebase_write_file, codebase_search, metrics_get_history, metrics_acknowledge_alert, send, done
3. "comms" — sends status updates to stakeholders via pushover.
   Tools: pushover_send_notification, send, done

On startup, have monitor set thresholds:
- cpu max 80 for all services
- error_rate max 5 for all services
- latency_ms max 200 for api

Workflow:
- Monitor checks metrics and reports alerts to you.
- You dispatch responder to investigate and fix.
- You tell comms to send status updates.
- After fix is applied, have monitor verify recovery.`,
	MCPServers: []MCPServerConfig{
		{Name: "metrics", Command: "", Env: map[string]string{"METRICS_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Create a config file the responder can read/edit
		os.MkdirAll(filepath.Join(dir, "config"), 0755)
		os.WriteFile(filepath.Join(dir, "config", "api.yaml"), []byte("max_connections: 100\ntimeout_ms: 5000\ncache_enabled: true\n"), 0644)
		os.WriteFile(filepath.Join(dir, "config", "worker.yaml"), []byte("concurrency: 10\nretry_limit: 3\n"), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads and thresholds set",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if len(threadIDs(th)) < 3 {
					return false
				}
				// Check if thresholds were set
				data, err := os.ReadFile(filepath.Join(dir, "thresholds.json"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), "cpu") && strings.Contains(string(data), "error_rate")
			},
		},
		{
			Name:    "Incident — CPU spike detected and investigated",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Seed a CPU spike in metrics history so get_metrics returns high values
				writeJSONFile(t, dir, "metrics.json", []map[string]any{
					{"service": "api", "metric": "cpu", "value": 95.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
					{"service": "api", "metric": "error_rate", "value": 12.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
					{"service": "api", "metric": "latency_ms", "value": 350.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("ALERT: api service showing high CPU and errors. Please investigate immediately.")
						injected = true
					}
					// Check if alerts were generated
					data, err := os.ReadFile(filepath.Join(dir, "alerts.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "api")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Alerts should exist (acknowledged or not — the key is that they were detected)
				data, _ := os.ReadFile(filepath.Join(dir, "alerts.json"))
				if !strings.Contains(string(data), "api") {
					t.Error("expected alerts for api service")
				}
			},
		},
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_Incident(t *testing.T) {
	metricsBin := buildMCPBinary(t, "mcps/metrics")
	codebaseBin := buildMCPBinary(t, "mcps/codebase")
	pushoverBin := buildMCPBinary(t, "mcps/pushover")
	t.Logf("built metrics=%s codebase=%s pushover=%s", metricsBin, codebaseBin, pushoverBin)

	s := incidentScenario
	s.MCPServers[0].Command = metricsBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = pushoverBin
	runScenario(t, s)
}

// --- Content Pipeline Scenario ---

var contentPipelineScenario = Scenario{
	Name: "ContentPipeline",
	Directive: `You manage a content production pipeline for a tech company blog.

Spawn and maintain 3 threads:
1. "researcher" — given a topic, gathers information and stores research notes.
   Tools: storage_store, storage_get, storage_list, send, done
2. "writer" — uses research to generate blog posts and social media content.
   Tools: creative_generate_post, creative_generate_image, storage_get, send, done
3. "publisher" — schedules and publishes content across social channels.
   Tools: social_post, social_get_channels, schedule_get_schedule, schedule_update_slot, send, done

Workflow:
- You receive a topic via console and tell researcher to gather info.
- When research is done, tell writer to draft a blog post and social posts.
- When content is ready, tell publisher to schedule and post across channels.

`,
	MCPServers: []MCPServerConfig{
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "creative", Command: "", Env: map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "social", Command: "", Env: map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"}},
		{Name: "schedule", Command: "", Env: map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed social channels
		writeJSONFile(t, dir, "channels.json", []map[string]string{
			{"id": "twitter", "name": "Twitter/X"},
			{"id": "linkedin", "name": "LinkedIn"},
			{"id": "instagram", "name": "Instagram"},
		})
		// Empty schedule
		writeJSONFile(t, dir, "schedule.json", []map[string]any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
		},
		{
			Name:    "Content production — topic to published posts",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New content topic: 'Why AI agents are replacing SaaS dashboards'. Research it, write content, and publish.")
						injected = true
					}
					// Check if content was generated (audit trail from creative/social)
					audit, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
					posts, _ := os.ReadFile(filepath.Join(dir, "posts.json"))
					return len(audit) > 50 || (len(posts) > 2 && strings.Contains(string(posts), "AI"))
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify content was generated
				data, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if len(data) == 0 {
					t.Error("expected audit trail of creative/social actions")
				}
			},
		},
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_ContentPipeline(t *testing.T) {
	storageBin := buildMCPBinary(t, "mcps/storage")
	creativeBin := buildMCPBinary(t, "mcps/creative")
	socialBin := buildMCPBinary(t, "mcps/social")
	scheduleBin := buildMCPBinary(t, "mcps/schedule")
	t.Logf("built storage=%s creative=%s social=%s schedule=%s", storageBin, creativeBin, socialBin, scheduleBin)

	s := contentPipelineScenario
	s.MCPServers[0].Command = storageBin
	s.MCPServers[1].Command = creativeBin
	s.MCPServers[2].Command = socialBin
	s.MCPServers[3].Command = scheduleBin
	runScenario(t, s)
}

// --- Trading Scenario ---

var tradingScenario = Scenario{
	Name: "Trading",
	Directive: `You manage a simple trading portfolio. Starting cash: $10,000.
Available symbols: AAPL, GOOGL, MSFT, TSLA.

Spawn and maintain 3 threads:
1. "data-feed" — reads prices periodically, stores history, reports significant moves to you.
   Tools: market_get_prices, market_get_history, storage_store, send, done
2. "analyst" — analyzes price data, identifies buy/sell signals based on price changes.
   Tools: market_get_history, market_get_prices, storage_get, storage_store, send, done
3. "executor" — places trades and manages stop-losses based on analyst signals.
   Tools: market_place_order, market_get_portfolio, market_set_stop_loss, market_get_orders, send, done

Workflow:
- Data-feed monitors prices and reports to you.
- You ask analyst to evaluate when significant moves occur.
- If analyst recommends a trade, you tell executor to place it with exact symbol, side, qty.
- Executor sets stop-losses on new positions (10% below buy price).

`,
	MCPServers: []MCPServerConfig{
		{Name: "market", Command: "", Env: map[string]string{"MARKET_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed initial prices
		writeJSONFile(t, dir, "prices.json", map[string]float64{
			"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 420.10, "TSLA": 178.90,
		})
		// Seed price history (simulated recent data)
		var history []map[string]any
		now := time.Now()
		symbols := map[string]float64{"AAPL": 180.0, "GOOGL": 140.0, "MSFT": 415.0, "TSLA": 185.0}
		for i := 10; i >= 1; i-- {
			ts := now.Add(-time.Duration(i) * time.Minute).UTC().Format(time.RFC3339)
			for sym, base := range symbols {
				drift := (float64(10-i) / 10.0) * 5.0 // gradual increase
				history = append(history, map[string]any{
					"symbol": sym, "price": base + drift, "timestamp": ts,
				})
			}
		}
		writeJSONFile(t, dir, "history.json", history)
		// Portfolio: $10k cash, no holdings
		writeJSONFile(t, dir, "portfolio.json", map[string]any{
			"cash": 10000.0, "holdings": map[string]any{},
		})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
		},
		{
			Name:    "Trading — buy signal and execution",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Market is open. AAPL shows strong upward trend from $180 to $185.50 over the last 10 periods. Buy 10 shares of AAPL now and set a stop-loss at $170.")
						injected = true
					}
					// Check if any order was placed
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "filled")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				if !strings.Contains(string(data), "buy") {
					t.Error("expected at least one buy order")
				}
			},
		},
		{
			Name:    "Stop-loss — price drop triggers sell",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Crash TSLA price to trigger stop-loss (if they bought it)
				// Or crash whatever they bought
				writeJSONFile(t, dir, "prices.json", map[string]float64{
					"AAPL": 150.00, "GOOGL": 110.00, "MSFT": 380.00, "TSLA": 120.00,
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("MARKET CRASH: Prices just dropped hard. AAPL is now $150. Sell all AAPL positions immediately to limit losses.")
						injected = true
					}
					// Check if portfolio was updated (stop-loss triggered or manual sell)
					data, err := os.ReadFile(filepath.Join(dir, "orders.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "sell")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				if !strings.Contains(string(data), "sell") {
					t.Error("expected at least one sell order after crash")
				}
			},
		},
	},
	Timeout:    8 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_Trading(t *testing.T) {
	marketBin := buildMCPBinary(t, "mcps/market")
	storageBin := buildMCPBinary(t, "mcps/storage")
	t.Logf("built market=%s storage=%s", marketBin, storageBin)

	s := tradingScenario
	s.MCPServers[0].Command = marketBin
	s.MCPServers[1].Command = storageBin
	runScenario(t, s)
}

// --- Onboarding Scenario ---

var onboardingScenario = Scenario{
	Name: "Onboarding",
	Directive: `You manage new customer onboarding for a SaaS platform.

Spawn and maintain 3 threads:
1. "intake" — fetches signup CSV files, reads customer data, reports to you.
   Tools: files_fetch_file, files_read_csv, files_list_files, files_file_status, send, done
2. "provisioner" — stores customer account records using storage tools.
   Tools: codebase_write_file, codebase_list_files, storage_store, storage_get, send, done
3. "welcome" — sends onboarding notifications to new customers.
   Tools: pushover_send_notification, storage_get, send, done

When you receive a signup file URL, tell intake to fetch and read it. Then tell provisioner to create accounts. Then tell welcome to notify customers.`,
	MCPServers: []MCPServerConfig{
		{Name: "files", Command: "", Env: map[string]string{"FILES_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		os.MkdirAll(filepath.Join(dir, "accounts"), 0755)
		// Signup CSV
		csv := "name,email,plan\nAlice Johnson,alice@startup.io,pro\nBob Chen,bob@bigcorp.com,enterprise\nCarol Davis,carol@freelance.me,starter\n"
		os.WriteFile(filepath.Join(dir, "signups-batch-1.csv"), []byte(csv), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
		},
		{
			Name:    "Onboarding — signup to welcome message",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "signups-batch-1.csv")
						th.InjectConsole("New signups file: " + csvPath + ". Please onboard these customers.")
						injected = true
					}
					// Check if accounts were provisioned (config files or storage entries)
					entries, _ := os.ReadDir(filepath.Join(dir, "accounts"))
					store, _ := os.ReadFile(filepath.Join(dir, "store.json"))
					s := strings.ToLower(string(store))
					return len(entries) >= 2 || strings.Contains(s, "alice") || strings.Contains(s, "bob")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify accounts provisioned via files or storage
				entries, _ := os.ReadDir(filepath.Join(dir, "accounts"))
				store, _ := os.ReadFile(filepath.Join(dir, "store.json"))
				hasFiles := len(entries) >= 2
				hasStore := strings.Contains(string(store), "alice") || strings.Contains(string(store), "bob")
				if !hasFiles && !hasStore {
					t.Error("expected accounts provisioned (config files or storage entries)")
				}
			},
		},
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_Onboarding(t *testing.T) {
	filesBin := buildMCPBinary(t, "mcps/files")
	codebaseBin := buildMCPBinary(t, "mcps/codebase")
	storageBin := buildMCPBinary(t, "mcps/storage")
	pushoverBin := buildMCPBinary(t, "mcps/pushover")
	t.Logf("built files=%s codebase=%s storage=%s pushover=%s", filesBin, codebaseBin, storageBin, pushoverBin)

	s := onboardingScenario
	s.MCPServers[0].Command = filesBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = storageBin
	s.MCPServers[3].Command = pushoverBin
	runScenario(t, s)
}
