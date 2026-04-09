package main

import (
	"encoding/json"
	"fmt"
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
	MCPServers []MCPServerConfig  // {{dataDir}} in Env values is replaced at runtime
	Providers  []ProviderConfig   // multi-provider pool config (optional)
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
	thinker := newScenarioThinker(t, apiKey, s.Directive, mcpServers, s.Providers)

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

func newScenarioThinker(t *testing.T, apiKey, directive string, mcpServers []MCPServerConfig, providerConfigs ...[]ProviderConfig) *Thinker {
	t.Helper()

	tmpDir := t.TempDir()

	cfg := &Config{
		path:       filepath.Join(tmpDir, "config.json"),
		Directive:  directive,
		MCPServers: mcpServers,
	}
	// Apply provider configs if provided
	if len(providerConfigs) > 0 && len(providerConfigs[0]) > 0 {
		cfg.Providers = providerConfigs[0]
	}
	cfg.Save()

	memStore := &MemoryStore{
		apiKey: apiKey,
		path:   filepath.Join(tmpDir, "memory.jsonl"),
	}

	// Build provider pool from config + env vars
	pool, err := buildProviderPool(cfg)
	if err != nil {
		t.Fatalf("no LLM provider: %v", err)
	}
	provider := pool.Default()

	bus := NewEventBus()
	thinker := &Thinker{
		apiKey:   apiKey,
		pool:     pool,
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

	thinker.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, ModeAutonomous, thinker.registry, "", nil, nil, pool)}

	go thinker.registry.EmbedAll(memStore)

	thinker.handleTools = mainToolHandler(thinker)
	thinker.rebuildPrompt = func(toolDocs string) string {
		return buildSystemPrompt(cfg.GetDirective(), ModeAutonomous, thinker.registry, toolDocs, thinker.mcpServers, nil, thinker.pool)
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

// --- Multi-Provider DevTeam Scenario ---
// Uses fireworks (default, cheap) for coordination + support + qa,
// and openai (gpt-4.1, powerful) for the dev coding thread.

var devTeamMultiProviderScenario = Scenario{
	Name: "DevTeamMultiProvider",
	Directive: `You manage a small development team maintaining a Todo SaaS app.
The codebase is in the "app/" directory. It is a Go package with todo.go and todo_test.go.

You have TWO providers available:
- fireworks (default) — fast and cheap, use for coordination, support, and QA
- openai — powerful (gpt-4.1), use for the dev thread that writes code

Spawn and maintain 3 threads:
1. "support" — monitors helpdesk tickets, triages them (bug vs feature), reports to main with recommendations.
   Tools: helpdesk_list_tickets, helpdesk_reply_ticket, helpdesk_close_ticket, send, done
2. "dev" — reads/writes code, implements features and fixes. MUST use provider="openai" for better code quality. Always reads existing code before modifying.
   Tools: codebase_read_file, codebase_write_file, codebase_list_files, codebase_search, send, done
3. "qa" — runs the test suite and reports results. Triggered by main after dev finishes.
   Tools: codebase_run_tests, codebase_read_file, send, done

Workflow:
- Support finds a ticket and tells you what it is
- You decide what to do and tell dev to implement it
- After dev is done, tell qa to run tests
- If tests fail, send dev back to fix. If pass, tell support to close the ticket.`,
	Providers: []ProviderConfig{
		{Name: "fireworks", Default: true},
		{Name: "openai"},
	},
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Command: "", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		seedTodoApp(t, dir)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned, dev on openai",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return len(threadIDs(th)) >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify pool has both providers
				if th.pool == nil {
					t.Error("expected provider pool")
					return
				}
				if th.pool.Count() < 2 {
					t.Errorf("expected 2 providers in pool, got %d", th.pool.Count())
				}
				if th.pool.Get("fireworks") == nil {
					t.Error("expected fireworks in pool")
				}
				if th.pool.Get("openai") == nil {
					t.Error("expected openai in pool")
				}
				// Log each thread's actual provider
				threads := th.threads.List()
				for _, thread := range threads {
					t.Logf("thread %s: provider=%s model=%s", thread.ID, thread.Provider, thread.Model)
				}
				// Check if dev got openai
				for _, thread := range threads {
					if thread.ID == "dev" && thread.Provider == "openai" {
						t.Logf("OK: dev thread correctly using openai")
					} else if thread.ID == "dev" {
						t.Logf("NOTE: dev thread using %s (directive asked for openai)", thread.Provider)
					}
				}
			},
		},
		{
			Name:    "Feature request — add priority field (dev uses openai)",
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
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_DevTeamMultiProvider(t *testing.T) {
	// Require both API keys
	if os.Getenv("FIREWORKS_API_KEY") == "" {
		t.Skip("FIREWORKS_API_KEY not set")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	helpdeskBin := buildMCPBinary(t, "mcps/helpdesk")
	codebaseBin := buildMCPBinary(t, "mcps/codebase")
	t.Logf("built helpdesk=%s codebase=%s", helpdeskBin, codebaseBin)

	s := devTeamMultiProviderScenario
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

// --- Lead Enrichment Scenario ---

var leadEnrichmentScenario = Scenario{
	Name: "LeadEnrichment",
	Directive: `You manage a lead enrichment pipeline.

Your job:
1. Read all leads from the "Lead Pipeline" spreadsheet using the sheets tools.
2. For each lead with status "new":
   a. Create a contact in the CRM (crm_create_contact) with name, email, company, website.
   b. Scrape the lead's website (webscraper_extract_info) to get company details.
   c. Update the CRM contact (crm_update_contact) with the enrichment data (industry, employee_count, location, description) and set status to "enriched".
   d. Update the spreadsheet row (sheets_update_cell) to set the status column to "enriched".
3. After all leads are processed, you are done.

Process all leads. Do not skip any.`,
	MCPServers: []MCPServerConfig{
		{Name: "sheets", Command: "", Env: map[string]string{"SHEETS_DATA_DIR": "{{dataDir}}"}},
		{Name: "crm", Command: "", Env: map[string]string{"CRM_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Command: "", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed the spreadsheet with 5 leads
		writeJSONFile(t, dir, "sheets.json", map[string]*struct {
			Columns []string            `json:"columns"`
			Rows    []map[string]string `json:"rows"`
		}{
			"Lead Pipeline": {
				Columns: []string{"name", "email", "website", "company", "status"},
				Rows: []map[string]string{
					{"name": "Alice Smith", "email": "alice@acmecorp.com", "website": "https://acmecorp.com", "company": "Acme Corp", "status": "new"},
					{"name": "Bob Chen", "email": "bob@globex.io", "website": "https://globex.io", "company": "Globex", "status": "new"},
					{"name": "Carol Davis", "email": "carol@initech.com", "website": "https://initech.com", "company": "Initech", "status": "new"},
					{"name": "Dan Wilson", "email": "dan@umbrella.dev", "website": "https://umbrella.dev", "company": "Umbrella Labs", "status": "new"},
					{"name": "Eve Park", "email": "eve@northwind.co", "website": "https://northwind.co", "company": "Northwind", "status": "new"},
				},
			},
		})

		// Seed website data for the scraper
		writeJSONFile(t, dir, "sites.json", map[string]*struct {
			Title       string `json:"title"`
			Description string `json:"description"`
			Body        string `json:"body"`
			Industry    string `json:"industry"`
			Employees   string `json:"employees"`
			Location    string `json:"location"`
			Founded     string `json:"founded"`
		}{
			"https://acmecorp.com": {
				Title:       "Acme Corp — Industrial Solutions",
				Description: "Leading provider of industrial automation and robotics systems for manufacturing.",
				Body:        "Acme Corp builds next-generation automation platforms for factories worldwide. Founded in 2015, we serve over 200 enterprise customers across North America and Europe.",
				Industry:    "Industrial Automation",
				Employees:   "500-1000",
				Location:    "San Francisco, CA",
				Founded:     "2015",
			},
			"https://globex.io": {
				Title:       "Globex — AI-Powered Analytics",
				Description: "We help businesses make data-driven decisions with real-time AI analytics.",
				Body:        "Globex provides a unified analytics platform powered by machine learning. Our team of 80 engineers and data scientists builds tools used by Fortune 500 companies.",
				Industry:    "SaaS / Analytics",
				Employees:   "50-100",
				Location:    "Austin, TX",
				Founded:     "2020",
			},
			"https://initech.com": {
				Title:       "Initech — Enterprise Software Consulting",
				Description: "Initech delivers custom enterprise software solutions and digital transformation services.",
				Body:        "For over a decade, Initech has helped mid-market companies modernize their technology stack. We specialize in ERP integration, cloud migration, and custom application development.",
				Industry:    "IT Consulting",
				Employees:   "200-500",
				Location:    "Chicago, IL",
				Founded:     "2012",
			},
			"https://umbrella.dev": {
				Title:       "Umbrella Labs — Biotech Research Platform",
				Description: "Umbrella Labs accelerates drug discovery with AI-powered molecular simulation.",
				Body:        "Our computational biology platform reduces drug discovery timelines from years to months. Backed by $50M in Series B funding, we partner with 15 pharmaceutical companies.",
				Industry:    "Biotech / Life Sciences",
				Employees:   "100-200",
				Location:    "Boston, MA",
				Founded:     "2019",
			},
			"https://northwind.co": {
				Title:       "Northwind — Sustainable Supply Chain",
				Description: "Northwind optimizes global supply chains for sustainability and efficiency.",
				Body:        "We provide end-to-end supply chain visibility with carbon footprint tracking. Our platform is used by 300+ retailers and manufacturers committed to sustainable operations.",
				Industry:    "Supply Chain / Logistics",
				Employees:   "150-300",
				Location:    "Seattle, WA",
				Founded:     "2017",
			},
		})

		// Start with empty CRM
		writeJSONFile(t, dir, "contacts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Sheet read — agent discovers 5 leads",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Wait for the agent to have called read_sheet (check audit)
				data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), "read_sheet")
			},
		},
		{
			Name:    "CRM creation — all 5 leads added",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "contacts.json"))
				if err != nil {
					return false
				}
				var contacts []json.RawMessage
				json.Unmarshal(data, &contacts)
				return len(contacts) >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "contacts.json"))
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				if len(contacts) < 5 {
					t.Errorf("expected 5 contacts, got %d", len(contacts))
				}
				// Verify all have emails
				emails := map[string]bool{}
				for _, c := range contacts {
					emails[c["email"]] = true
				}
				for _, expected := range []string{"alice@acmecorp.com", "bob@globex.io", "carol@initech.com", "dan@umbrella.dev", "eve@northwind.co"} {
					if !emails[expected] {
						t.Errorf("missing contact with email %s", expected)
					}
				}
			},
		},
		{
			Name:    "Enrichment — CRM contacts updated with company info",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "contacts.json"))
				if err != nil {
					return false
				}
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				enriched := 0
				for _, c := range contacts {
					if c["industry"] != "" && c["location"] != "" {
						enriched++
					}
				}
				return enriched >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "contacts.json"))
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				for _, c := range contacts {
					if c["industry"] == "" {
						t.Errorf("contact %s (%s) missing industry", c["id"], c["email"])
					}
					if c["location"] == "" {
						t.Errorf("contact %s (%s) missing location", c["id"], c["email"])
					}
				}
				// Verify scraper was actually called (check audit)
				auditData, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				auditStr := string(auditData)
				if !strings.Contains(auditStr, "extract_info") && !strings.Contains(auditStr, "fetch_page") {
					t.Error("expected webscraper tools to be called")
				}
			},
		},
		{
			Name:    "Sheet update — all leads marked enriched",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "sheets.json"))
				if err != nil {
					return false
				}
				// Count rows with status=enriched
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				sheetData, ok := sheets["Lead Pipeline"]
				if !ok {
					return false
				}
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(sheetData, &sheet)
				enriched := 0
				for _, row := range sheet.Rows {
					if row["status"] == "enriched" {
						enriched++
					}
				}
				return enriched >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "sheets.json"))
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(sheets["Lead Pipeline"], &sheet)
				for i, row := range sheet.Rows {
					if row["status"] != "enriched" {
						t.Errorf("row %d (%s) status=%q, expected enriched", i, row["name"], row["status"])
					}
				}
			},
		},
	},
	Timeout:    6 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_LeadEnrichment(t *testing.T) {
	sheetsBin := buildMCPBinary(t, "mcps/sheets")
	crmBin := buildMCPBinary(t, "mcps/crm")
	scraperBin := buildMCPBinary(t, "mcps/webscraper")
	t.Logf("built sheets=%s crm=%s webscraper=%s", sheetsBin, crmBin, scraperBin)

	s := leadEnrichmentScenario
	s.MCPServers[0].Command = sheetsBin
	s.MCPServers[1].Command = crmBin
	s.MCPServers[2].Command = scraperBin
	runScenario(t, s)
}

// --- Website Build + Deploy Scenario ---

func seedWebsiteBrief(t *testing.T, dir string) {
	t.Helper()

	// Design brief
	writeJSONFile(t, dir, "brief.json", map[string]any{
		"company": "NovaPay",
		"tagline": "Payments infrastructure for the AI economy",
		"sections": []map[string]any{
			{
				"id": "hero", "heading": "Accept AI-to-AI payments",
				"subheading": "NovaPay handles billing between autonomous agents, with real-time settlement and fraud detection.",
				"cta":        "Get Started",
			},
			{
				"id": "features", "items": []map[string]string{
					{"title": "Agent Wallets", "desc": "Every AI agent gets a programmable wallet with spending limits and approval flows."},
					{"title": "Real-time Settlement", "desc": "Sub-second settlement between agents. No batching, no delays."},
					{"title": "Fraud Detection", "desc": "ML-powered anomaly detection built for machine-speed transactions."},
				},
			},
			{
				"id": "pricing", "plans": []map[string]any{
					{"name": "Starter", "price": "$0", "desc": "1,000 transactions/mo", "features": []string{"Agent wallets", "Basic analytics", "Email support"}},
					{"name": "Growth", "price": "$49/mo", "desc": "50,000 transactions/mo", "features": []string{"Everything in Starter", "Real-time dashboard", "Webhooks", "Priority support"}},
					{"name": "Enterprise", "price": "Custom", "desc": "Unlimited", "features": []string{"Everything in Growth", "SLA", "Dedicated account manager", "Custom integrations"}},
				},
			},
			{
				"id": "footer", "links": []string{"Docs", "Pricing", "Blog", "GitHub", "Twitter"},
			},
		},
		"brand": map[string]string{"primary": "#6C5CE7", "secondary": "#00CEC9", "dark": "#2D3436", "light": "#DFE6E9"},
	})

	// Assets
	writeJSONFile(t, dir, "assets.json", []map[string]string{
		{"name": "logo", "url": "/logo.svg", "desc": "NovaPay logo"},
		{"name": "hero-bg", "url": "/hero-bg.svg", "desc": "Abstract gradient background"},
	})

	// App directory
	os.MkdirAll(filepath.Join(dir, "app", "src"), 0755)

	// test.sh — validates project structure (searches recursively)
	os.WriteFile(filepath.Join(dir, "test.sh"), []byte(`#!/bin/bash
cd app || exit 1
[ -f package.json ] || { echo "ERROR: no package.json"; exit 1; }
# Find entry point
found_entry=0
for f in src/index.tsx src/index.jsx src/main.tsx src/main.jsx; do
  [ -f "$f" ] && { found_entry=1; break; }
done
[ "$found_entry" -eq 1 ] || { echo "ERROR: no entry point (src/index.tsx or src/main.tsx)"; exit 1; }
# Find App component
found_app=0
for f in src/App.tsx src/App.jsx; do
  [ -f "$f" ] && { found_app=1; break; }
done
[ "$found_app" -eq 1 ] || { echo "ERROR: no App component (src/App.tsx)"; exit 1; }
# Check component files have exports (skip entry points)
count=0
while IFS= read -r f; do
  base=$(basename "$f")
  # Skip entry points — they render to DOM, no export needed
  case "$base" in index.tsx|index.jsx|main.tsx|main.jsx) count=$((count+1)); continue;; esac
  grep -q "export" "$f" || { echo "ERROR: $f has no export"; exit 1; }
  count=$((count + 1))
done < <(find src -name "*.tsx" -o -name "*.jsx" 2>/dev/null)
[ "$count" -ge 2 ] || { echo "ERROR: need at least 2 component files, found $count"; exit 1; }
echo "BUILD OK: $count components"
mkdir -p dist
echo "<html>bundled</html>" > dist/index.html
`), 0755)
}

var websiteBuildScenario = Scenario{
	Name: "WebsiteBuild",
	Directive: `You are building and deploying a React landing page for NovaPay.

Read the design brief first, then build a complete React application with Bun as the bundler.

Spawn 3 threads:
1. "architect" — reads the design brief and assets, plans the component structure, creates the project scaffold (package.json with react/react-dom deps, src/index.tsx entry point, src/App.tsx main component, and a basic index.html). Reports the plan to main when done.
   Tools: brief_get_brief, brief_get_assets, codebase_write_file, codebase_list_files, send, done
2. "builder" — implements each React component based on the brief. Creates Hero, Features, Pricing, and Footer components as separate .tsx files in src/. Includes inline CSS or a styles.css file. Runs the build check to verify all files are valid. Fixes any errors. Reports done when build passes.
   Tools: brief_get_brief, codebase_read_file, codebase_write_file, codebase_list_files, codebase_run_tests, send, done
3. "deployer" — creates a site on the hosting platform, deploys the app when the build is ready, and confirms it's live with the URL.
   Tools: hosting_create_site, hosting_deploy, hosting_get_status, hosting_get_url, hosting_list_sites, send, done

Workflow:
- First, tell architect to read the brief and scaffold the project.
- When architect reports done, tell builder to implement all sections from the brief.
- Builder should create: Hero.tsx, Features.tsx, Pricing.tsx, Footer.tsx (at minimum), import them in App.tsx, and run the build check.
- When builder confirms the build passes, tell deployer to create a site called "novapay-landing" and deploy.
- Deployer confirms the live URL.

IMPORTANT: All files go in the "app/" directory. package.json must include "react" and "react-dom" as dependencies. Every .tsx file must have an export.`,
	MCPServers: []MCPServerConfig{
		{Name: "brief", Command: "", Env: map[string]string{"BRIEF_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "hosting", Command: "", Env: map[string]string{"HOSTING_DATA_DIR": "{{dataDir}}", "CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedWebsiteBrief,
	Phases: []Phase{
		{
			Name:    "Scaffold — package.json + App.tsx created",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if _, err := os.Stat(filepath.Join(dir, "app", "package.json")); err != nil {
					return false
				}
				if _, err := os.Stat(filepath.Join(dir, "app", "src", "App.tsx")); err != nil {
					if _, err2 := os.Stat(filepath.Join(dir, "app", "src", "App.jsx")); err2 != nil {
						return false
					}
				}
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify package.json is valid and has react
				data, _ := os.ReadFile(filepath.Join(dir, "app", "package.json"))
				var pkg map[string]any
				if err := json.Unmarshal(data, &pkg); err != nil {
					t.Errorf("package.json is not valid JSON: %v", err)
				}
				deps, _ := pkg["dependencies"].(map[string]any)
				if deps == nil {
					t.Error("package.json missing dependencies")
				} else if deps["react"] == nil {
					t.Error("package.json missing react dependency")
				}
			},
		},
		{
			Name:    "Components — 4+ tsx/jsx files with exports",
			Timeout: 240 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Count tsx/jsx files recursively under app/src/
				count := 0
				filepath.Walk(filepath.Join(dir, "app", "src"), func(path string, info os.FileInfo, err error) error {
					if err != nil || info.IsDir() {
						return nil
					}
					if strings.HasSuffix(info.Name(), ".tsx") || strings.HasSuffix(info.Name(), ".jsx") {
						count++
					}
					return nil
				})
				return count >= 4
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Log component files (exports checked in build phase, builder will fix missing ones)
				filepath.Walk(filepath.Join(dir, "app", "src"), func(path string, info os.FileInfo, err error) error {
					if err != nil || info.IsDir() {
						return nil
					}
					if strings.HasSuffix(info.Name(), ".tsx") || strings.HasSuffix(info.Name(), ".jsx") {
						data, _ := os.ReadFile(path)
						hasExport := strings.Contains(string(data), "export")
						t.Logf("component %s (%d bytes, export=%v)", info.Name(), len(data), hasExport)
					}
					return nil
				})
			},
		},
		{
			Name:    "Build — test.sh passes",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
		},
		{
			Name:    "Deploy — site is live",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "sites.json"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), `"live"`)
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify site is live with URL
				data, _ := os.ReadFile(filepath.Join(dir, "sites.json"))
				var sites []map[string]string
				json.Unmarshal(data, &sites)
				if len(sites) == 0 {
					t.Error("no sites created")
					return
				}
				site := sites[0]
				if site["status"] != "live" {
					t.Errorf("site status=%s, expected live", site["status"])
				}
				if site["url"] == "" {
					t.Error("site has no URL")
				}
				t.Logf("Site deployed: %s → %s", site["name"], site["url"])

				// Verify deployment record
				dData, _ := os.ReadFile(filepath.Join(dir, "deployments.json"))
				var deploys []map[string]any
				json.Unmarshal(dData, &deploys)
				if len(deploys) == 0 {
					t.Error("no deployment records")
				} else {
					files, _ := deploys[0]["files"].([]any)
					t.Logf("Deployed %d files", len(files))
				}
			},
		},
	},
	Timeout:    10 * time.Minute,
	MaxThreads: 5,
}

func TestScenario_WebsiteBuild(t *testing.T) {
	briefBin := buildMCPBinary(t, "mcps/brief")
	codebaseBin := buildMCPBinary(t, "mcps/codebase")
	hostingBin := buildMCPBinary(t, "mcps/hosting")
	t.Logf("built brief=%s codebase=%s hosting=%s", briefBin, codebaseBin, hostingBin)

	s := websiteBuildScenario
	s.MCPServers[0].Command = briefBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = hostingBin
	runScenario(t, s)
}

// --- Learning Agent Scenario ---

var learningAgentScenario = Scenario{
	Name: "LearningAgent",
	Directive: `You manage a warehouse. You do NOT know the business rules — discover them by trying actions and learning from failures.

CRITICAL RULES FOR LEARNING:
1. When ANY action fails, you MUST call [[remember text="..."]] with the rule you learned. This is mandatory.
2. After learning 2+ rules, call [[evolve directive="..."]] to update your directive with all learned rules.
3. Your memory persists across sessions. Your conversation does NOT. Only remembered facts survive.

Process orders and shipments as requested via console events. When something fails, learn why, remember it, and retry correctly.`,
	MCPServers: []MCPServerConfig{
		{Name: "warehouse", Command: "", Env: map[string]string{"WAREHOUSE_DATA_DIR": "{{dataDir}}"}, MainAccess: true},
	},
	DataSetup: func(t *testing.T, dir string) {
		writeJSONFile(t, dir, "stock.json", map[string]int{
			"widgets":   500,
			"gadgets":   200,
			"chemicals": 300,
			"batteries": 150,
		})
		writeJSONFile(t, dir, "orders.json", []any{})
		writeJSONFile(t, dir, "shipments.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Phase 1: Order fails — learns and remembers max qty rule",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Order 200 widgets immediately.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(data, &orders)
					hasFailed := false
					hasSuccess := false
					for _, o := range orders {
						if o["status"] == "failed" {
							hasFailed = true
						}
						if o["status"] == "fulfilled" {
							hasSuccess = true
						}
					}
					return hasFailed && hasSuccess && th.memory.Count() > 0
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 1: %d entries", th.memory.Count())
				if th.memory.Count() == 0 {
					t.Error("agent did not use [[remember]] after learning qty rule")
				}
			},
		},
		{
			Name:    "Phase 2: Ship to Japan + remember customs rule",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Ship the fulfilled widgets order to Japan, weight 20kg. Remember any rules you discover.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
					var shipments []map[string]any
					json.Unmarshal(data, &shipments)
					for _, s := range shipments {
						if s["status"] == "shipped" {
							return true
						}
					}
					return false
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 2: %d entries", th.memory.Count())
			},
		},
		{
			Name:    "Phase 2b: Force hazardous rule discovery",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("Order 50 chemicals. Remember any new rules you discover about ordering.")
						injected = true
					}
					data, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(data, &orders)
					for _, o := range orders {
						if o["item"] == "chemicals" && o["status"] == "fulfilled" {
							return true
						}
					}
					return false
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 2b: %d entries", th.memory.Count())
				// Should have at least 2 memories now (qty + hazardous or customs)
				if th.memory.Count() < 2 {
					t.Logf("NOTE: expected 2+ memories, got %d", th.memory.Count())
				}
			},
		},
		{
			Name:    "Phase 3: Context reset — apply knowledge from memory only",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Reset order/shipment files for clean phase 3
				writeJSONFile(t, dir, "orders.json", []any{})
				writeJSONFile(t, dir, "shipments.json", []any{})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						// Clear conversation history — agent must rely on memory
						th.messages = th.messages[:1]
						t.Logf("conversation reset — %d memory entries available", th.memory.Count())
						th.InjectConsole("Order 150 chemicals and ship them to Germany, weight 30kg. Apply everything you know about warehouse rules.")
						injected = true
					}
					orderData, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
					var orders []map[string]any
					json.Unmarshal(orderData, &orders)
					chemFulfilled := 0
					for _, o := range orders {
						if o["item"] == "chemicals" && o["status"] == "fulfilled" {
							chemFulfilled++
						}
					}
					shipData, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
					var shipments []map[string]any
					json.Unmarshal(shipData, &shipments)
					germanyShipped := false
					for _, s := range shipments {
						dest, _ := s["destination"].(string)
						if strings.EqualFold(dest, "germany") && s["status"] == "shipped" {
							germanyShipped = true
						}
					}
					return chemFulfilled >= 2 && germanyShipped
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Count failures in phase 3 — fewer failures = better memory recall
				orderData, _ := os.ReadFile(filepath.Join(dir, "orders.json"))
				var orders []map[string]any
				json.Unmarshal(orderData, &orders)
				failures := 0
				successes := 0
				for _, o := range orders {
					if o["status"] == "failed" {
						failures++
					}
					if o["status"] == "fulfilled" {
						successes++
					}
				}
				t.Logf("phase 3 orders: %d fulfilled, %d failed (fewer failures = better memory)", successes, failures)

				shipData, _ := os.ReadFile(filepath.Join(dir, "shipments.json"))
				var shipments []map[string]any
				json.Unmarshal(shipData, &shipments)
				shipOK := 0
				shipFail := 0
				for _, s := range shipments {
					if s["status"] == "shipped" {
						shipOK++
					} else {
						shipFail++
					}
				}
				t.Logf("phase 3 shipments: %d shipped, %d failed", shipOK, shipFail)
			},
		},
		{
			Name:    "Phase 4: Final summary",
			Timeout: 10 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				return true // always pass — just log results
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				directive := th.config.GetDirective()
				evolved := len(directive) > 700
				t.Logf("directive evolved: %v (%d chars)", evolved, len(directive))
				t.Logf("final memory count: %d", th.memory.Count())

				auditData, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(auditData)), "\n")
				t.Logf("total audit trail: %d entries", len(lines))

				if th.memory.Count() < 2 {
					t.Error("expected at least 2 memory entries from learning")
				}
			},
		},
	},
	Timeout:    10 * time.Minute,
	MaxThreads: 3,
}

func TestScenario_LearningAgent(t *testing.T) {
	warehouseBin := buildMCPBinary(t, "mcps/warehouse")
	t.Logf("built warehouse=%s", warehouseBin)

	s := learningAgentScenario
	s.MCPServers[0].Command = warehouseBin
	runScenario(t, s)
}

// --- Emergent Behavior Scenario ---

func seedStoreData(t *testing.T, dir string) {
	t.Helper()

	writeJSONFile(t, dir, "sales.json", map[string]any{
		"summary": "Revenue down 23% month-over-month. 3 of top 5 products showing sharp decline.",
		"daily": []map[string]any{
			{"date": "2026-04-01", "revenue": 3200, "orders": 42},
			{"date": "2026-04-02", "revenue": 2800, "orders": 38},
			{"date": "2026-04-03", "revenue": 2100, "orders": 29},
			{"date": "2026-04-04", "revenue": 1900, "orders": 25},
			{"date": "2026-04-05", "revenue": 1700, "orders": 22},
			{"date": "2026-04-06", "revenue": 1500, "orders": 19},
			{"date": "2026-04-07", "revenue": 1400, "orders": 17},
		},
		"by_product": []map[string]any{
			{"name": "Wireless Earbuds Pro", "units_sold": 0, "revenue": 0, "note": "NO SALES — check inventory"},
			{"name": "USB-C Hub 7-in-1", "units_sold": 0, "revenue": 0, "note": "NO SALES — check inventory"},
			{"name": "Laptop Stand Adjustable", "units_sold": 45, "revenue": 2250, "trend": "stable"},
			{"name": "Mechanical Keyboard RGB", "units_sold": 12, "revenue": 1080, "trend": "declining — was 30/week"},
			{"name": "Webcam 4K", "units_sold": 0, "revenue": 0, "note": "NO SALES — check inventory"},
			{"name": "Phone Case Premium", "units_sold": 89, "revenue": 1335, "trend": "stable"},
			{"name": "Desk Lamp LED", "units_sold": 34, "revenue": 680, "trend": "stable"},
		},
	})

	writeJSONFile(t, dir, "inventory.json", map[string]any{
		"products": []map[string]any{
			{"name": "Wireless Earbuds Pro", "stock": 0, "price": 79.99, "status": "OUT OF STOCK", "last_restocked": "2026-03-01"},
			{"name": "USB-C Hub 7-in-1", "stock": 0, "price": 49.99, "status": "OUT OF STOCK", "last_restocked": "2026-03-05"},
			{"name": "Laptop Stand Adjustable", "stock": 120, "price": 49.99, "status": "in stock"},
			{"name": "Mechanical Keyboard RGB", "stock": 45, "price": 89.99, "status": "in stock"},
			{"name": "Webcam 4K", "stock": 0, "price": 129.99, "status": "OUT OF STOCK", "last_restocked": "2026-02-20"},
			{"name": "Phone Case Premium", "stock": 230, "price": 14.99, "status": "in stock"},
			{"name": "Desk Lamp LED", "stock": 67, "price": 19.99, "status": "in stock"},
		},
	})

	writeJSONFile(t, dir, "reviews.json", map[string]any{
		"average_rating": 3.2,
		"recent": []map[string]any{
			{"product": "Wireless Earbuds Pro", "rating": 1, "text": "Wanted to buy but OUT OF STOCK for over a month! Going to Amazon instead.", "date": "2026-04-05"},
			{"product": "USB-C Hub 7-in-1", "rating": 1, "text": "Says out of stock. This was my favorite hub. Very disappointing.", "date": "2026-04-04"},
			{"product": "Mechanical Keyboard RGB", "rating": 3, "text": "Good keyboard but $89.99 is too expensive. Same one is $69 on Amazon.", "date": "2026-04-03"},
			{"product": "Webcam 4K", "rating": 1, "text": "OUT OF STOCK AGAIN. Third time I've tried to order. Lost a customer.", "date": "2026-04-06"},
			{"product": "Phone Case Premium", "rating": 5, "text": "Great case, fast shipping, good price!", "date": "2026-04-05"},
			{"product": "Laptop Stand Adjustable", "rating": 4, "text": "Solid product but shipping was slow — 8 days.", "date": "2026-04-02"},
			{"product": "Desk Lamp LED", "rating": 4, "text": "Nice lamp. Would buy again.", "date": "2026-04-01"},
		},
	})

	writeJSONFile(t, dir, "competitors.json", map[string]any{
		"comparison": []map[string]any{
			{"product": "Wireless Earbuds Pro", "our_price": 79.99, "amazon_price": 74.99, "best_buy_price": 79.99},
			{"product": "USB-C Hub 7-in-1", "our_price": 49.99, "amazon_price": 39.99, "best_buy_price": 44.99},
			{"product": "Mechanical Keyboard RGB", "our_price": 89.99, "amazon_price": 69.99, "best_buy_price": 74.99},
			{"product": "Webcam 4K", "our_price": 129.99, "amazon_price": 109.99, "best_buy_price": 119.99},
			{"product": "Phone Case Premium", "our_price": 14.99, "amazon_price": 14.99, "best_buy_price": 16.99},
			{"product": "Laptop Stand Adjustable", "our_price": 49.99, "amazon_price": 49.99, "best_buy_price": 54.99},
		},
	})

	writeJSONFile(t, dir, "analytics.json", map[string]any{
		"period":           "last 7 days",
		"unique_visitors":  12400,
		"page_views":       34200,
		"conversion_rate":  "1.4% (was 3.2% last month)",
		"bounce_rate":      "62% (was 45% last month)",
		"top_search_terms": []string{"wireless earbuds", "usb-c hub", "webcam 4k", "keyboard", "portable monitor usb-c"},
		"cart_abandonment": "78% (was 52% last month)",
		"note":             "Traffic is healthy but conversions dropped. Most searched products are out of stock. Unusual spike in searches for 'portable monitor usb-c' — we don't carry this product. Check traffic sources for details.",
	})

	writeJSONFile(t, dir, "traffic.json", map[string]any{
		"period": "last 7 days",
		"sources": []map[string]any{
			{"source": "google organic", "visits": 5200, "conversion": "1.8%"},
			{"source": "direct", "visits": 3100, "conversion": "2.1%"},
			{"source": "social media", "visits": 1800, "conversion": "0.9%"},
			{"source": "techgadgetblog.com/best-usb-c-monitors-2026", "visits": 1400, "conversion": "0.1%", "note": "ANOMALY: High traffic, near-zero conversion. Blog recommends 'UltraView Portable Monitor 15.6\" USB-C' at $199 — we don't carry it. 89% of these visitors search our store for it then leave."},
			{"source": "email campaigns", "visits": 900, "conversion": "3.2%"},
		},
		"trending_searches_with_zero_results": []string{"portable monitor", "usb-c monitor", "ultraview monitor"},
	})

	writeJSONFile(t, dir, "suppliers.json", map[string]any{
		"suppliers": []map[string]any{
			{
				"name": "TechSource Direct", "status": "DELAYED",
				"products":      []string{"Wireless Earbuds Pro", "USB-C Hub 7-in-1", "Webcam 4K"},
				"normal_lead":   "3-5 days",
				"current_lead":  "14-18 days",
				"reason":        "Warehouse fire at distribution center. Backlog expected until mid-April.",
				"reliability":   "Usually excellent — this is an unusual event",
				"recommendation": "Use alt_supplier for urgent restocks (+15% cost, 3-5 day delivery)",
			},
			{
				"name": "AltSupply Express", "status": "OPERATIONAL",
				"products":     []string{"Wireless Earbuds Pro", "USB-C Hub 7-in-1", "Webcam 4K", "UltraView Portable Monitor"},
				"normal_lead":  "3-5 days",
				"current_lead": "3-5 days",
				"surcharge":    "15%",
				"note":         "Can also supply UltraView Portable Monitor 15.6\" USB-C at wholesale $120 (MSRP $199)",
			},
			{
				"name": "GenericParts Co", "status": "OPERATIONAL",
				"products":    []string{"Laptop Stand Adjustable", "Desk Lamp LED", "Phone Case Premium"},
				"normal_lead": "2-3 days",
				"current_lead": "2-3 days",
			},
		},
	})

	writeJSONFile(t, dir, "segments.json", map[string]any{
		"segments": []map[string]any{
			{"name": "power_buyers", "count": 340, "avg_order": 127, "frequency": "2.3x/month", "note": "Highest value — 40% of revenue. Many have stopped buying (out-of-stock items). 78 haven't purchased in 3 weeks."},
			{"name": "deal_seekers", "count": 890, "avg_order": 34, "frequency": "1.1x/month", "note": "Price-sensitive. Respond well to promotions. Keyboard price increase lost 40% of this segment."},
			{"name": "new_visitors", "count": 1200, "avg_order": 0, "frequency": "0", "note": "1,200 new visitors this week, mostly from techgadgetblog.com. Almost none converted — they're looking for a product we don't sell."},
			{"name": "returning_loyal", "count": 460, "avg_order": 62, "frequency": "1.8x/month", "note": "Stable segment. Good retention. Would respond well to loyalty rewards."},
		},
	})
}

var emergentScenario = Scenario{
	Name: "Emergent",
	Directive: `You run a small online electronics store. Sales have been declining. Diagnose the root causes and take action to turn things around. Go deep — surface-level fixes won't be enough.`,
	MCPServers: []MCPServerConfig{
		{Name: "store", Command: "", Env: map[string]string{"STORE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedStoreData,
	Phases: []Phase{
		{
			Name:    "Deep investigation — agent explores multiple data layers",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				tools := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						tools[a] = true
					}
				}
				// Must dig into at least 5 different data sources (not just the obvious ones)
				return len(tools) >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				tools := map[string]bool{}
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					if a := entry["action"]; a != "" {
						tools[a] = true
					}
				}
				t.Logf("data sources explored: %v (%d)", tools, len(tools))
			},
		},
		{
			Name:    "Multi-layered action — fixes surface + deeper issues",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				s := string(data)
				// Must take at least 3 different ACTION types
				actionTypes := 0
				if strings.Contains(s, "\"action\":\"restock_item\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"adjust_price\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"send_promotion\"") {
					actionTypes++
				}
				if strings.Contains(s, "\"action\":\"add_product\"") {
					actionTypes++
				}
				return actionTypes >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				var actions []string
				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					switch entry["action"] {
					case "restock_item":
						actions = append(actions, fmt.Sprintf("RESTOCK: %s ×%s (supplier: %s)", entry["product"], entry["quantity"], entry["supplier"]))
					case "adjust_price":
						actions = append(actions, fmt.Sprintf("PRICE: %s → $%s", entry["product"], entry["new_price"]))
					case "send_promotion":
						actions = append(actions, fmt.Sprintf("PROMO: \"%s\" (%s to %s)", entry["subject"], entry["discount"], entry["target_segment"]))
					case "add_product":
						actions = append(actions, fmt.Sprintf("NEW PRODUCT: %s at $%s", entry["name"], entry["price"]))
					}
				}
				for _, a := range actions {
					t.Logf("  %s", a)
				}
				// Check for smart decisions
				usedAltSupplier := strings.Contains(string(data), "alt_supplier")
				addedProduct := strings.Contains(string(data), "add_product")
				t.Logf("discovered alt supplier: %v", usedAltSupplier)
				t.Logf("added new product (blog opportunity): %v", addedProduct)
			},
		},
		{
			Name:    "Emergence score — threads, memory, strategy, creativity",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				score := 0
				if th.threads != nil && th.threads.Count() > 0 {
					score += 2
				}
				if th.memory.Count() > 0 {
					score++
				}
				directive := th.config.GetDirective()
				if len(directive) > 200 {
					score++
				}
				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				s := string(data)
				if strings.Contains(s, "alt_supplier") {
					score++ // discovered supply chain workaround
				}
				if strings.Contains(s, "add_product") {
					score++ // spotted market opportunity
				}
				actions := strings.Count(s, "\"action\":")
				if actions >= 6 {
					score++ // took comprehensive action
				}
				return score >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Log("=== EMERGENCE REPORT ===")
				t.Logf("threads active: %d", th.threads.Count())
				t.Logf("memory entries: %d", th.memory.Count())
				directive := th.config.GetDirective()
				t.Logf("directive evolved: %v (%d chars)", len(directive) > 200, len(directive))

				data, _ := os.ReadFile(filepath.Join(dir, "actions.jsonl"))
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				s := string(data)

				score := 0
				if th.threads.Count() > 0 {
					score += 2
					t.Log("  ✓ spawned worker threads (self-organization)")
				}
				if th.memory.Count() > 0 {
					score++
					t.Log("  ✓ remembered findings (persistent learning)")
				}
				if len(directive) > 200 {
					score++
					t.Log("  ✓ evolved directive (self-improvement)")
				}
				if strings.Contains(s, "alt_supplier") {
					score++
					t.Log("  ✓ discovered alt supplier workaround (problem-solving)")
				}
				if strings.Contains(s, "add_product") {
					score++
					t.Log("  ✓ spotted new product opportunity (creativity)")
				}
				if len(lines) >= 8 {
					score++
					t.Log("  ✓ took comprehensive multi-step action (initiative)")
				}

				t.Logf("EMERGENCE SCORE: %d/7", score)
				t.Logf("total tool calls: %d", len(lines))

				for _, line := range lines {
					var entry map[string]string
					json.Unmarshal([]byte(line), &entry)
					t.Logf("  [%s] %s", entry["action"], entry)
				}
			},
		},
	},
	Timeout:    10 * time.Minute,
	MaxThreads: 12,
}

func TestScenario_Emergent(t *testing.T) {
	storeBin := buildMCPBinary(t, "mcps/store")
	t.Logf("built store=%s", storeBin)

	s := emergentScenario
	s.MCPServers[0].Command = storeBin
	runScenario(t, s)
}
