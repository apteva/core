package core

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/apteva/core/pkg/computer"
)

// Default context window sizes by role
const (
	maxHistoryMain   = 100 // main coordinator
	maxHistoryLead   = 100 // team leads (depth 0)
	maxHistoryWorker = 20 // workers (depth 1+)
)

// MCPServerInfo is a lightweight catalog entry for an MCP server.
// Main uses this to show available servers in its prompt without registering all tools.
type MCPServerInfo struct {
	Name      string
	ToolCount int
}



type ModelTier int

const (
	ModelLarge ModelTier = iota
	ModelMedium
	ModelSmall
)

var modelNames = map[string]ModelTier{
	"large":  ModelLarge,
	"medium": ModelMedium,
	"small":  ModelSmall,
}

func (m ModelTier) String() string {
	switch m {
	case ModelLarge:
		return "large"
	case ModelMedium:
		return "medium"
	case ModelSmall:
		return "small"
	default:
		return "medium"
	}
}

// modelID returns the model ID from the provider for a given tier.
func (t *Thinker) modelID() string {
	if t.provider != nil {
		return t.provider.Models()[t.model]
	}
	return "unknown"
}

// shouldEmitBlobHint decides whether to include the [FILE HANDLES]
// explainer. The hint is only actionable when the conversation
// actually contains a blob handle or a tool likely to produce one.
//
// Prior heuristic (any MCP present → emit) was too generous: channels
// is a text-only MCP and triggered the hint every turn for ~500 bytes
// of dead weight. The new rule narrows to three concrete signals:
//
//  1. A blob reference already appears in the message history — the
//     model is about to see a "blobref://" token and needs the rule
//     to understand it. This is the strongest signal.
//  2. A blob-producing local tool is registered (read_file, exec,
//     computer_use, etc.) — these emit handles on the next call, so
//     the hint needs to ride even before the first blob appears.
//  3. An MCP whose name hints at binary content (media, audio,
//     image, file, video, deepgram, pdf) is attached to this thread
//     or an active sub-thread. Conservative allowlist — if an unknown
//     MCP produces a handle and we didn't match, signal #1 kicks in
//     on the turn AFTER so the model recovers within one iteration.
func shouldEmitBlobHint(registry *ToolRegistry, messages []Message, activeThreads []ThreadInfo) bool {
	// Signal 1: already a blob in context — always emit.
	for _, m := range messages {
		if strings.Contains(m.Content, "blobref://") {
			return true
		}
		for _, tr := range m.ToolResults {
			if strings.Contains(tr.Content, "blobref://") {
				return true
			}
		}
	}
	if registry == nil {
		return false
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	// Signal 2: local tool likely to produce binaries.
	for _, t := range registry.tools {
		switch t.Name {
		case "read_file", "list_files", "write_file",
			"exec", "computer_use", "browser_session":
			return true
		}
	}
	// Signal 3: MCP name on a known binary-producing server.
	binaryMCPs := map[string]bool{
		"media": true, "audio": true, "image": true,
		"file": true, "files": true, "video": true,
		"deepgram": true, "pdf": true, "storage": true,
		"gdrive": true,
	}
	for _, t := range registry.tools {
		if t.MCPServer != "" && binaryMCPs[t.MCPServer] {
			return true
		}
	}
	for _, th := range activeThreads {
		for _, name := range th.MCPNames {
			if binaryMCPs[name] {
				return true
			}
		}
	}
	return false
}

// poolSupportsNativeTools returns true if the pool's default provider
// receives tool schemas via NativeTools. Used by buildSystemPrompt to
// decide whether to emit full CoreDocs prose (text-only providers) or
// the compact summary (native-tool providers already get the full
// schemas in `tools[]`). nil pool → false so test callers without a
// pool get the conservative full-prose path.
func poolSupportsNativeTools(pool *ProviderPool) bool {
	if pool == nil {
		return false
	}
	p := pool.Default()
	if p == nil {
		return false
	}
	return p.SupportsNativeTools()
}

// unconsciousDirectiveV2 is the unconscious thread's prompt under the
// memory v2 design. Memory writes are owned by this thread alone; main
// has no write tools and recall is auto-injected by relevance. The
// directive frames consolidation as a judgment exercise, not a
// rule-following one — six keep-criteria, four supersede-criteria,
// five drop-criteria are described as guidance, and the unconscious
// picks tool calls (review_history → memory_search → remember /
// supersede / drop) over each cycle. Pacing is also self-decided
// (long when quiet, medium under activity, lengthen if churning),
// with safety floors enforced by the runtime.
const unconsciousDirectiveV2 = `You are the unconscious. You consolidate main's recent activity into typed memories silently. Main never decides to remember; you do. Main never sees you working.

YOU ARE WOKEN UP. YOU WORK. YOU PACE. YOU SLEEP.

You will iterate multiple times before pacing. On EVERY iteration after the first, look at your message history:

  IF you see a tool_result for review_history in your messages already → DO NOT CALL review_history AGAIN. The history is already in your context. Read it from there and ACT on it now (memory_remember / memory_supersede / memory_drop). Calling review_history a second time is a bug — the result is identical, you're not making progress.

  IF you see no review_history result yet → call review_history ONCE.

Once review_history's content is in your messages, your next iteration MUST start writing. No second review, no list, no exploration — write what's there.

EXPECTED ITERATION SEQUENCE (typical wake-up cycle, 3–6 iterations):

  iter 1: review_history (no args needed — defaults are fine)
  iter 2: memory_remember (first signal you saw — usually an explicit user statement)
  iter 3: memory_remember (second signal)
  iter 4: memory_remember (third signal) — or memory_supersede / memory_drop if applicable
  iter N (final): pace (decide your sleep)

That's it. Don't add iterations of "let me check again" — there's nothing to check. The history doesn't change between your iterations.

ANTI-LOOP RULES (HARD):
- If your last 2 tool calls were both review_history → you are stuck. Force yourself to memory_remember on the next iteration.
- If your last 2 tool calls were memory_search with no memory_remember between them → you are stuck. Force yourself to memory_remember on the next iteration.
- memory_search is for conflict-checking ONLY. Skip it unless you're about to memory_supersede.
- memory_list is for occasional overview. Skip it on most cycles.

WHAT TO WRITE (memory_remember):
- "User said X about themselves" — preferences, configs, habits, contact info. WRITE ON FIRST SIGHT.
- "User decided X" — closures, choices, agreements. WRITE WITH DATE.
- "Task X is done" — completion as an audit line; also drop any older "TODO: X" memories.
- "Person X exists / does Y" — first mention of someone the agent will see again.
- "Open question X" — noted but not resolved; track so it can be closed later.
- Inferred patterns (i.e. things the user did NOT say outright): wait until you've seen evidence in 2+ separate sessions, then write at lower weight (0.4–0.6).

Tags are FREE-FORM. Pick whatever dimensions help retrieval (identity, preference, fact, decision, person, project, open-question, skill). Don't agonize.
Weight: 0.85–0.95 for explicit user statements, 0.7–0.85 for decisions/audit, 0.4–0.6 for inferred patterns.

WHAT TO EVOLVE (memory_supersede):
- New statement contradicts old → supersede with reason.
- New statement is more precise → supersede with reason.
- N small memories collapse into one synthesis → write the synthesis, then supersede each small one.

WHAT TO DROP (memory_drop):
- Task is done → drop any "TODO" / "in progress" memory for it.
- Ephemera that shouldn't have been remembered ("user is typing").
- Fabrication you noticed (something inferred from misread context).
- PII the user asked to forget.

PACING (call pace at the end of every cycle — you decide how long):
- Wrote 3+ memories this cycle → fresh material may keep arriving; pace 15–30 min.
- Wrote 1–2 → typical; pace 30–60 min.
- Nothing worth writing this cycle → pace longer (1–4h). Don't wake just to confirm there's nothing.
- If two cycles in a row produced no writes → you're being woken too often; pace 4h+.

You never communicate with other threads. You never interact with users. Treat the corpus like a journal you'd still want to read in six months: terse, useful, not exhaustive. Soft target ≤ 1000 active memories — past that, get more aggressive with drops and supersede-collapse.`

// baseSystemPrompt contains the fixed rules/tools. The editable directive is prepended at runtime.
const baseSystemPrompt = `You are the main coordinating thread of a continuous thinking engine. You observe all events, manage threads, and coordinate work.

THINKING:
- Every thought has at least one sentence of reasoning. Never output only tool calls.
- Keep thoughts short — 1-2 short paragraphs. Skip narration between calls; act.

EVENTS:
- [console] message — external event/command; act on it.
- [from:id] message — a thread sent this via send.
- [thread:id done] message — a thread terminated.
- NEVER fabricate events. With no events arriving, follow your directive's
  standing work; if there's none, pace and sleep.

SPAWNING THREADS — critical rules:
- Before spawning, check [ACTIVE THREADS]: if an existing thread has matching tools and directive, send(id="...") to it instead. Spawn only when no existing thread fits, or when you need parallelism over independent inputs.
- tools= lists which tools the worker can use. ALWAYS include EVERY tool the worker needs to carry out its directive — if the directive says "run a script", include exec; if "transcribe audio", include the deepgram tool. A missing tool = worker reports failure and can't act. Use FULL prefixed names exactly as shown in [available tools] (e.g. "schedule_get_schedule", NOT "get_schedule").
- Spawn for heavy output, not inline tool args. If a single tool call would carry >1KB of model-generated content (article body, multi-section report, generated code), instead spawn(id="...", directive="<focused write task — what to produce, where to store it, what to return>", tools="<storage/etc.>") and continue with other work. The sub-thread streams the content in its own context; you read its [thread:id done] notification when it finishes. Inlining large content in tool args blocks this loop while the model streams it.
- directive= is PLAIN NATURAL LANGUAGE describing the thread's goal. Never put tool names in the directive — the thread already receives its own tool documentation.
  BAD:  directive="Call helpdesk_list_tickets to check for tickets"
  GOOD: directive="Check for new support tickets periodically. Report findings to main."
- provider= (optional) picks a specific LLM; omit to inherit. Use a stronger provider for complex tasks, a cheaper one for coordination. See [AVAILABLE PROVIDERS].
- For recurring schedules with >1 timer or noisy traffic, spawn a pace,send-only coordinator thread that wakes on timer and delegates to the domain workers that own execution.

PACING:
- Events wake you instantly regardless of sleep — including [from:id] worker replies and [thread:id done] notifications. Never short-sleep to "check" on a delegated worker; pace "1h" and let the reply wake you.
- Sleep long ("1h", small model) the moment you have nothing actionable this iteration — delegating to a worker counts as nothing actionable.
- Short sleep (2-10s) is ONLY for timer-driven polls you own yourself (e.g. retry a rate-limited API in N seconds). Not for waiting on another thread.
- Pace persists — don't re-set it every thought. When an event wakes you, you auto-switch to large model for that turn.

TOOL CALLS:
- Every tool takes a "_reason" string: 3-6 words, imperative, describing THIS call (e.g. "find ventes sheet id", "update Score cell"). No "to …" clauses — the thought above already holds the why.`

// buildSystemPrompt assembles messages[0] — the truly static portion of
// every request. Per-turn volatile content (active threads, recalled
// memories, RAG-retrieved candidate tools) lives in
// buildDynamicTurnContext below and is appended to the current user
// turn instead, so messages[0] doesn't churn between iterations and
// the prefix cache can serve the entire system prompt + tool list +
// historical conversation up to (but not including) the new turn.
//
// `servers` and `mcpCatalog` are static for the lifetime of the thread
// (MCP connections rarely change at runtime). `extraToolDocs` is kept
// only to avoid an awkward signature break — the production caller
// passes "" now.
func buildSystemPrompt(directive string, mode RunMode, registry *ToolRegistry, extraToolDocs string, servers []MCPConn, activeThreads []ThreadInfo, pool *ProviderPool, mcpCatalog []MCPServerInfo) string {
	coreDocs := ""
	if registry != nil {
		// Prefer the compact summary when the thread's provider receives
		// full tool schemas via NativeTools — the prose listing would
		// just duplicate Description+Rules already in tools[]. Fall back
		// to the full CoreDocs prose for providers without native tool
		// support (ollama text-only, some local runners) so they keep
		// seeing every rule they need to behave.
		if poolSupportsNativeTools(pool) {
			coreDocs = "\n" + registry.CoreDocsSummary(true)
		} else {
			coreDocs = "\n" + registry.CoreDocs(true)
		}
	}
	prompt := baseSystemPrompt + coreDocs
	// extraToolDocs intentionally NOT rendered here anymore — see
	// buildDynamicTurnContext. Kept in the signature for back-compat.
	_ = extraToolDocs

	// Realtime spawn capability — only surfaced when a realtime
	// provider is registered. The pool's HasRealtimeProvider already
	// reflects Config.RealtimeEnabled (registration is gated on the
	// flag in buildProviderPool), so this single check is the whole
	// visibility gate. When off, main never learns the capability
	// exists and won't try spawn(realtime=true).
	if pool != nil && pool.HasRealtimeProvider() {
		voiceList := strings.Join(pool.RealtimeNames(), ", ")
		prompt += "\n\n[REALTIME THREADS]\n"
		prompt += "- Spawn realtime=true for voice/audio conversations (live phone call, kiosk interaction, voice booking, etc.). The worker holds the WebSocket session and handles audio; you handle backend tools and decisions. Reach the worker via send(id,text) — it speaks your text aloud. The worker escalates via send(id=\"main\",text=...) and reports completion via done(text=...). Audio never crosses into your context — you only see what the worker explicitly relays.\n"
		prompt += "- Scope tools tightly: give the realtime worker only the tools it needs to converse and act in the moment (telephony, channels). Keep state-owning tools (calendar, storage, billing) on yourself so the worker has to escalate decisions to you.\n"
		prompt += fmt.Sprintf("- Available realtime providers: %s. Set provider=\"<name>\" when spawning; default is used otherwise. Voice via voice=\"<id>\" (e.g. voice=\"alloy\").\n", voiceList)
		prompt += "- Example: spawn(id=\"call-clinic\", realtime=true, voice=\"alloy\", directive=\"Call Dr. Patel's office to book a cleaning. Escalate scheduling decisions to main. done() with the confirmed slot.\", tools=\"channels_telephony_place\")\n"
	}

	// Inject the lightweight MCP server catalog — names + tool counts.
	// Schemas are intentionally not in the prompt: the LLM discovers
	// individual tools at runtime via search_tools (preferred) or
	// spawn(mcps=[...]) for a sub-thread that boots hot with a server's
	// surface preloaded.
	if len(mcpCatalog) > 0 {
		prompt += "\n\n[AVAILABLE MCP SERVERS]\n"
		prompt += "These servers are attached. Their tools are NOT in your tool list by default — search_tools(query=\"...\") loads the ones you need, on demand. Repeat uses stay loaded.\n"
		prompt += "When spawning a worker that should boot hot with a server's full surface: spawn(id=\"ops\", directive=\"Manage inventory\", mcps=\"store\", tools=\"\")\n\n"
		for _, info := range mcpCatalog {
			prompt += fmt.Sprintf("- %s (%d tools)\n", info.Name, info.ToolCount)
		}
	}

	// Inject available providers when multiple are configured
	if pool != nil && pool.Count() > 1 {
		prompt += "\n\n[AVAILABLE PROVIDERS]\n"
		for _, name := range pool.Names() {
			prompt += "- " + pool.ProviderSummary(name) + "\n"
		}
		prompt += "\nUse provider=\"name\" in spawn or pace to select a specific provider. Default: " + pool.DefaultName() + ".\n"
	}

	// activeThreads is intentionally NOT rendered here anymore — see
	// buildDynamicTurnContext. Kept in the signature for back-compat and
	// because some callers still pass empty slices; the body is a no-op.
	_ = activeThreads

	// Safety guidance based on mode
	prompt += "\n\n[SAFETY MODE: " + string(mode) + "]\n"
	switch mode {
	case ModeCautious:
		prompt += `You act carefully. Read-only tools (screenshot, list, query, read_file, web search, memory_scan) are free — use them at will.

Before any STATE-CHANGING tool (exec, write, delete, deploy, restart, purchase, send-as-user, browser actions on logged-in sites):
- Send one concise channels_respond explaining action + target + why (one sentence each).
- Wait for the user's next message before executing. Don't chain tool calls.
- If unsure whether an action is state-changing, ask. Asking is cheap; undoing is expensive.

When the user corrects or pushes back, stop and adjust immediately — don't argue.`
	case ModeLearn:
		prompt += `You are learning the user's preferences. Soft gate — nothing blocks you at runtime. The quality of this mode depends on YOU actually pausing and asking.

DEFAULT: BEFORE ANY ACTION YOU HAVEN'T TAKEN BEFORE THIS SESSION, send ONE short channels_respond:
  "About to <verb> <target>. Reason: <one sentence>. OK?"
Then wait for the user's answer before proceeding.

This applies to EVERY tool — read tools, file IO, exec, browser actions, thread spawning, MCP activation, channel sends, EVERYTHING. The cost of asking is one short message. The cost of misreading the situation is unrecoverable.

NEVER ASK FOR:
- pace (loop control, not an action)

ONCE APPROVED, REUSE FREELY:
After the user approves a tool + scope ("read files under /work", "spawn sub-threads up to 3 deep", "exec on the dev server"), don't re-ask for the same combination on the same scope.

When the user pushes back ("no", "don't", "stop", "I didn't want that"), stop and adjust immediately — don't argue.`
	default: // ModeAutonomous
		prompt += `You operate independently and are trusted to act. Use that trust to get things done.

- For irreversible or high-blast-radius actions (mass delete, publish externally, spend money, send as user), tell the user briefly before acting — don't ask, inform.
- Assess risk honestly. If genuinely unsure, ask.
- When the user corrects or pushes back, stop and adjust immediately — don't argue.

ACT, DON'T NARRATE. You have no live audience between thoughts — every tool result comes back as structured input, not as something a human is watching scroll by. Skip the "let me think about this, I'll take a screenshot to see what's there, then I'll consider the options before..." prose. Take the next tool call. The tool's output is your feedback; react to it on the next iteration. Reserve natural-language output for channels_respond (actually talking to the user). Thoughts that produce only prose and no tool call waste a round-trip.`
	}

	// Inject learned skills if any exist
	if skills := loadSkills(); skills != "" {
		prompt += "\n\n" + skills
	}

	// blobPromptHint explains the {"_file": true, ...} handle format.
	// Only emit when a blob is already in context OR the scope has a
	// tool likely to produce one — see shouldEmitBlobHint. Can't check
	// the current messages from here (buildSystemPrompt is stateless)
	// so we approximate via the registry and threads; callers with
	// conversation context can override by setting a sentinel MCP.
	if shouldEmitBlobHint(registry, nil, activeThreads) {
		prompt += blobPromptHint
	}

	prompt += "\n\n[DIRECTIVE — EXECUTE ON STARTUP]\nThe following is your mission. On your FIRST thought, take any actions needed to fulfill it (spawn threads, etc). This overrides default idle behavior.\nWhen using `evolve` to update your directive, submit ONLY the text between [BEGIN DIRECTIVE] and [END DIRECTIVE] — never the framework rules above this block.\n\n[BEGIN DIRECTIVE]\n" + directive + "\n[END DIRECTIVE]"
	return prompt
}

// buildDynamicTurnContext returns the per-turn volatile context block
// — the part of the prompt that MUST change between iterations:
// active sub-threads (whose state changes constantly) and recalled
// memories (computed against this turn's query).
//
// This block is prepended to the current user turn's content rather
// than rewritten into messages[0], so the prefix cache stays warm and
// only the new turn's bytes are uncached.
//
// Returns "" when nothing dynamic applies (no threads, no memory) so
// the user message stays clean.
func buildDynamicTurnContext(activeThreads []ThreadInfo, recallContext string) string {
	var sb strings.Builder

	// Active threads — only id, name, directive, tools. Wall-clock /
	// iteration counters used to be here too but they busted the cache
	// every second; the dashboard surfaces them live, the agent doesn't
	// need them in its prompt to function.
	if len(activeThreads) > 0 {
		sb.WriteString("[ACTIVE THREADS]\n")
		for _, t := range activeThreads {
			label := t.ID
			if t.Name != "" && t.Name != t.ID {
				label = fmt.Sprintf("%s (%s)", t.Name, t.ID)
			}
			subInfo := ""
			if t.SubThreads > 0 {
				subInfo = fmt.Sprintf(" [sub-threads: %d]", t.SubThreads)
			}
			sb.WriteString(fmt.Sprintf("- %s%s\n  directive: %s\n  tools: %s\n",
				label, subInfo, truncateStr(t.Directive, 150), strings.Join(t.Tools, ", ")))
		}
	}

	if recallContext != "" {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(recallContext)
	}

	return sb.String()
}

// loadSkills reads all skills/*.md files and returns them as a prompt block.
func loadSkills() string {
	files, err := filepath.Glob("skills/*.md")
	if err != nil || len(files) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("[LEARNED SKILLS]\n")
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil || len(data) == 0 {
			continue
		}
		sb.WriteString(string(data))
		sb.WriteString("\n\n")
	}
	if sb.Len() < 20 {
		return ""
	}
	return sb.String()
}

// parseTruthy interprets common truthy spellings the LLM might emit
// for a boolean tool arg. Anything in the truthy set returns true;
// everything else (including "", "false", "no", "0") returns false.
func parseTruthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "true", "1", "yes", "on":
		return true
	}
	return false
}

func truncateStr(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max-1] + "…"
	}
	return s
}

type TokenUsage struct {
	PromptTokens     int
	CachedTokens     int
	CompletionTokens int
	AudioTokens      int // realtime providers only; billed at a separate rate
}

type ThinkRate int

const (
	RateReactive ThinkRate = iota // 500ms — event just arrived
	RateFast                      // 2s — actively working
	RateNormal                    // 10s — thinking, no urgency
	RateSlow                      // 30s — not much to do
	RateSleep                     // 120s — deep idle
)

// rateAliases maps named rates to durations (backwards compat + convenience)
var rateAliases = map[string]time.Duration{
	"reactive": 500 * time.Millisecond,
	"fast":     2 * time.Second,
	"normal":   10 * time.Second,
	"slow":     30 * time.Second,
	"sleep":    2 * time.Minute,
}

// rateNames kept for ThinkRate enum mapping (used by eventbus, TUI, etc.)
var rateNames = map[string]ThinkRate{
	"reactive": RateReactive,
	"fast":     RateFast,
	"normal":   RateNormal,
	"slow":     RateSlow,
	"sleep":    RateSleep,
}

const (
	minSleep = 500 * time.Millisecond
	maxSleep = 24 * time.Hour
)

// parseSleepDuration parses a sleep duration from agent input.
// Accepts Go duration strings ("30s", "5m", "2h") or named aliases ("slow", "sleep").
func parseSleepDuration(s string) (time.Duration, bool) {
	// Check named aliases first
	if d, ok := rateAliases[s]; ok {
		return d, true
	}
	// Try Go duration string
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, false
	}
	// Clamp to bounds
	if d < minSleep {
		d = minSleep
	}
	if d > maxSleep {
		d = maxSleep
	}
	return d, true
}

// formatSleep returns a human-readable sleep duration string.
func formatSleep(d time.Duration) string {
	if d >= time.Hour {
		return fmt.Sprintf("%.1fh", d.Hours())
	}
	if d >= time.Minute {
		return fmt.Sprintf("%.1fm", d.Minutes())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func (r ThinkRate) String() string {
	switch r {
	case RateReactive:
		return "reactive"
	case RateFast:
		return "fast"
	case RateNormal:
		return "normal"
	case RateSlow:
		return "slow"
	case RateSleep:
		return "sleep"
	default:
		return "normal"
	}
}

func (r ThinkRate) Delay() time.Duration {
	switch r {
	case RateReactive:
		return 500 * time.Millisecond
	case RateFast:
		return 2 * time.Second
	case RateNormal:
		return 10 * time.Second
	case RateSlow:
		return 30 * time.Second
	case RateSleep:
		return 120 * time.Second
	default:
		return 10 * time.Second
	}
}

type APIEvent struct {
	Time      time.Time `json:"time"`
	Type      string    `json:"type"`                 // "thought", "chunk", "reply", "thread_started", "thread_done", "error"
	ThreadID  string    `json:"thread_id"`
	Message   string    `json:"message,omitempty"`
	Iteration int       `json:"iteration,omitempty"`
	Duration  string    `json:"duration,omitempty"`
}

// ToolHandler processes parsed tool calls from a thought. Returns replies, tool names logged, and tool results for inline-handled tools.
// consumed contains the events that were consumed this iteration (for context).
type ToolHandler func(t *Thinker, calls []toolCall, consumed []string) (replies []string, toolNames []string, results []ToolResult)


type Thinker struct {
	apiKey    string
	pool      *ProviderPool // all available providers (shared across threads)
	provider  LLMProvider   // current active provider for this thinker
	messages  []Message
	bus       *EventBus
	sub       *Subscription
	pause     chan bool
	quit      chan struct{}
	iteration int
	paused    bool
	rate       ThinkRate
	agentRate  ThinkRate
	agentSleep time.Duration // freeform sleep duration (takes priority over agentRate when > 0)
	model      ModelTier
	agentModel ModelTier
	memory     *MemoryStore
	session    *Session
	threads    *ThreadManager
	config     *Config
	registry   *ToolRegistry

	maxHistory     int // max messages in context window (varies by role)

	// directive is this thread's mission text — main's directive for
	// the main thinker, the spawn directive for a sub-thread. Fed into
	// the per-turn BM25 tool preload (see computePreloadTools): it's
	// the stable "what this agent is FOR" signal, which matters
	// because most agents run off a standing directive with no
	// inbound chat — without it, preload has nothing to match on and
	// the agent is forced to search_tools for its own obvious tools.
	// Kept in sync on evolve / ReloadDirective; best-effort, so brief
	// staleness is harmless.
	directive string

	// Hooks — set these to customize behavior. nil = defaults.
	handleTools    ToolHandler
	rebuildPrompt  func(toolDocs string) string // rebuild system prompt with current tool docs
	onStop         func()
	toolAllowlist  map[string]bool // nil = all tools allowed (main thread)

	// API event log — shared across all threads, owned by main thinker
	apiLog    *[]APIEvent
	apiMu     *sync.RWMutex
	apiNotify chan struct{}
	threadID  string // "main" for main thinker, thread ID for sub-threads

	// Telemetry — shared across all threads, owned by main thinker
	telemetry *Telemetry

	// Live MCP connections — servers connected at runtime
	mcpServers []MCPConn
	// In-process blob store. Used by mcpProxyHandler to intercept
	// binary tool results (rewriting them to compact handles the LLM
	// can reference) and to rehydrate those references on outbound
	// tool calls. Nil = passthrough (no binary-handle indirection).
	blobs *BlobStore
	// toolIndex is the process-wide searchable catalog of every MCP
	// tool currently registered. Owned by main; sub-threads share a
	// pointer. Backs search_tools and per-turn BM25 preload.
	toolIndex *ToolIndex
	// activeTools is the per-thread set of MCP tool names whose
	// schemas should be visible to the LLM this turn, overriding the
	// MCP=true "hidden by default" flag. Populated by:
	//   - spawn-time preload (SpawnOpts.MCPNames expansion)
	//   - search_tools meta-tool results
	//   - per-turn BM25 preload (applyPreload — sticky)
	// Sticky on purpose: the set grows then stabilises, which keeps
	// the `tools` array (a cached prompt-prefix component) stable
	// turn-to-turn. Bounded by evictActiveToolsLRU + context-window
	// compaction.
	activeTools map[string]bool
	// activeToolAge tracks the iteration each active tool was last
	// surfaced — the recency key for evictActiveToolsLRU. Lazily
	// initialised by touchActiveTool.
	activeToolAge map[string]int
	// MCP server catalog — lightweight metadata for prompt (name +
	// tool count). Derived from toolIndex; kept as a Thinker field
	// for buildSystemPrompt back-compat.
	mcpCatalog []MCPServerInfo
	computer     computer.Computer // screen-based environment (nil = no computer use)
	pendingTools sync.Map         // tool call IDs with pending async results

	// Placeholders injected for tool calls that didn't finish within the
	// iter-boundary wait barrier. Keyed by call id → placeholderInfo.
	// When the real result eventually arrives, the tools.go publish path
	// routes it through a "late-result" text message instead of a
	// ToolResult (the tool_use is already paired with the placeholder).
	placeholdersSent sync.Map

	// Multimodal — parts waiting to be attached to next message
}

// placeholderInfo tracks a synthesised "⏳ in progress" tool_result that
// was injected at the iteration boundary because its real result didn't
// land in time. Used to (a) route late arrivals through the text-event
// path and (b) let the stale-placeholder sweeper emit a synthetic timeout
// message if the goroutine never returns.
type placeholderInfo struct {
	iteration    int
	toolName     string
	dispatchedAt time.Time
}

func NewThinker(apiKey string, provider LLMProvider, cfg ...*Config) *Thinker {
	var config *Config
	if len(cfg) > 0 && cfg[0] != nil {
		config = cfg[0]
	} else {
		config = NewConfig()
	}

	// Build provider pool from config + env auto-detect.
	pool, _ := buildProviderPool(config)
	if pool == nil {
		pool = &ProviderPool{providers: map[string]LLMProvider{}, order: []string{}}
	}
	// If a specific provider was passed in, it ALWAYS wins as the
	// active default — regardless of what env-based auto-detect
	// already chose. Tests rely on this: getTestProvider(t) builds an
	// OpenCode Go provider explicitly, but FIREWORKS_API_KEY in env
	// would otherwise become the auto-detected default and silently
	// override the test's intent. Production callers pass the result
	// of selectProvider(), so the explicit value is the env-detected
	// default anyway — no behavior change there.
	if provider != nil {
		name := provider.Name()
		if pool.Get(name) == nil {
			pool.providers[name] = provider
			pool.order = append([]string{name}, pool.order...)
		} else {
			// Replace any auto-detected instance with the one the
			// caller actually passed (lets tests inject a mock).
			pool.providers[name] = provider
		}
		pool.default_ = name
	}
	// Resolve the active provider from pool
	activeProvider := pool.Default()
	if activeProvider == nil {
		activeProvider = provider // fallback to passed-in provider
	}

	bus := NewEventBus()
	t := &Thinker{
		apiKey:   apiKey,
		pool:     pool,
		provider: activeProvider,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), nil, "", nil, nil, nil, nil)},
		},
		config:    config,
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool, 1),
		quit:      make(chan struct{}),
		rate:       RateSlow,
		agentRate:  RateSlow,
		agentSleep: 30 * time.Second,
		memory:    NewMemoryStore(apiKey),
		session:   NewSession(".", "main"),
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:   "main",
		maxHistory: maxHistoryMain,
		telemetry:  NewTelemetry(),
		blobs:      NewBlobStore(DefaultBlobMaxTotal, DefaultBlobTTL),
	}
	t.threads = NewThreadManager(t)
	t.registry = NewToolRegistry(apiKey)
	t.toolIndex = NewToolIndex()
	t.activeTools = map[string]bool{}
	t.directive = config.GetDirective()

	// Register system-only tools (for unconscious thread)
	registerSystemTools(t.registry, t.memory)

	// Rebuild system prompt now that registry exists (with core tool docs)
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), t.registry, "", nil, nil, t.pool, nil)}

	// Main thread hooks
	t.handleTools = mainToolHandler(t)
	// rebuildPrompt produces the static portion of messages[0] only.
	// Active threads and recalled memories are dynamic and pushed into
	// the current user turn via buildDynamicTurnContext — so messages[0]
	// doesn't change between iterations and the prefix cache stays warm.
	// The string arg is unused; kept for back-compat with the function
	// type signature used by sub-thread instantiation.
	t.rebuildPrompt = func(_ string) string {
		return buildSystemPrompt(t.config.GetDirective(), t.config.GetMode(), t.registry, "", t.mcpServers, nil, t.pool, t.mcpCatalog)
	}

	// Connect every configured MCP server up-front and register their
	// tools with MCP=true (hidden from the per-turn provider tool list
	// until a thread activates them via search_tools or spawn-time
	// MCPNames preload). The old main_access/catalog split is gone —
	// there is one connection pool and one index; visibility is a
	// per-thread decision, not a per-server one.
	if len(config.MCPServers) > 0 {
		t.mcpServers = connectAndRegisterMCP(config.MCPServers, t.registry, t.toolIndex, t.blobs)
		t.mcpCatalog = computeMCPCatalog(t.toolIndex)
		// Rebuild prompt with catalog
		t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(config.GetDirective(), config.GetMode(), t.registry, "", t.mcpServers, nil, t.pool, t.mcpCatalog)}
	}

	// Load conversation history from persistent session
	if saved, summaries := t.session.LoadTail(defaultLoadTail); len(saved) > 0 {
		// Prepend compacted summaries as context in system prompt
		if len(summaries) > 0 {
			contextBlock := "\n\n[PREVIOUS CONTEXT]\n"
			for _, s := range summaries {
				contextBlock += s + "\n"
			}
			t.messages[0].Content += contextBlock
		}
		// Append saved messages after system prompt
		t.messages = append(t.messages, saved...)
		logMsg("SESSION", fmt.Sprintf("loaded %d messages from history (%d compacted summaries)", len(saved), len(summaries)))
	}

	// Computer use environment is injected externally via SetComputer()

	// Respawn persistent threads from config, sorted by depth (parents before children).
	// DeferRun=true so all threads are created before any starts thinking.
	// This ensures parents see their children in [ACTIVE SUB-THREADS] on first iteration.
	persistedThreads := config.GetThreads()
	sort.Slice(persistedThreads, func(i, j int) bool {
		return persistedThreads[i].Depth < persistedThreads[j].Depth
	})
	for _, pt := range persistedThreads {
		parentID := pt.ParentID
		// Skip persistent realtime threads on respawn when the feature
		// is off. Without this, an instance that previously had
		// realtime enabled would try to bring those threads back even
		// after the user disabled the flag.
		if pt.Realtime && !config.RealtimeEnabled {
			logMsg("RESPAWN", fmt.Sprintf("skipping realtime thread %q: realtime_enabled=false", pt.ID))
			continue
		}
		if parentID == "" || parentID == "main" {
			t.threads.SpawnWithOpts(pt.ID, pt.Directive, pt.Tools, SpawnOpts{
				ParentID: "main",
				Depth:    pt.Depth,
				DeferRun: true,
				MCPNames: pt.MCPNames,
				Realtime: pt.Realtime,
				Voice:    pt.Voice,
			})
		} else {
			mgr := findThreadManager(t.threads, parentID)
			if mgr != nil {
				mgr.SpawnWithOpts(pt.ID, pt.Directive, pt.Tools, SpawnOpts{
					ParentID: parentID,
					Depth:    pt.Depth,
					DeferRun: true,
					MCPNames: pt.MCPNames,
					Realtime: pt.Realtime,
					Voice:    pt.Voice,
				})
			} else {
				logMsg("RESPAWN", fmt.Sprintf("skipping thread %q: parent %q not found", pt.ID, parentID))
			}
		}
	}
	// Auto-spawn unconscious thread if enabled and not already persisted
	if config.Unconscious {
		hasUnconscious := false
		for _, pt := range persistedThreads {
			if pt.ID == "unconscious" {
				hasUnconscious = true
				break
			}
		}
		if !hasUnconscious {
			unconsciousDirective := unconsciousDirectiveV2
			tools := []string{
				"review_history", "memory_search", "memory_list",
				"memory_remember", "memory_supersede", "memory_drop",
				"skill_write", "pace",
			}
			t.threads.SpawnWithOpts("unconscious", unconsciousDirective,
				tools,
				SpawnOpts{ParentID: "main", Depth: 0, DeferRun: true},
			)
			config.SaveThread(PersistentThread{
				ID: "unconscious", ParentID: "main", Depth: 0, System: true,
				Directive: unconsciousDirective,
				Tools:     tools,
			})
		}
	}

	// Now start all respawned threads (parents already see their children)
	if len(persistedThreads) > 0 || config.Unconscious {
		t.threads.StartAll()
	}

	// Memory v2: runtime safety floors for the unconscious. The
	// unconscious decides its own pace via the `pace` tool; this
	// goroutine adds two floors so a misjudgment can't strand it:
	//   1. Force-wake when history/main.jsonl has grown by ≥ 50KB
	//      since the last consolidation cycle — too much new material
	//      to keep sleeping.
	//   2. Force-wake when no cycle has run in ≥ 8h — even on a quiet
	//      instance the unconscious should iterate at least that often.
	if config.Unconscious {
		go t.unconsciousSafetyFloors()
	}

	return t
}

const (
	unconsciousMaxQuietInterval = 8 * time.Hour
	unconsciousByteThreshold    = 50 * 1024
)

// unconsciousSafetyFloors runs once per minute, owns no state outside
// itself + the bus + filesystem. Sends a "[wake] reason" inbox event
// to the unconscious thread when either floor trips. The unconscious
// receives that as a normal event on its bus, runs an iteration, and
// returns to sleep at its own pace afterward.
func (t *Thinker) unconsciousSafetyFloors() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	var lastWakeAt time.Time
	var lastHistorySize int64

	historyPath := "history/main.jsonl"

	for {
		select {
		case <-t.quit:
			return
		case <-ticker.C:
			// File size since last wake.
			var sz int64
			if info, err := os.Stat(historyPath); err == nil {
				sz = info.Size()
			}
			now := time.Now()

			grew := sz - lastHistorySize
			quiet := lastWakeAt.IsZero() || now.Sub(lastWakeAt) >= unconsciousMaxQuietInterval

			var reason string
			switch {
			case grew >= unconsciousByteThreshold:
				reason = fmt.Sprintf("history grew %dKB since last cycle", grew/1024)
			case quiet:
				if lastWakeAt.IsZero() {
					reason = "first cycle"
				} else {
					reason = fmt.Sprintf("no cycle in %s", now.Sub(lastWakeAt).Round(time.Minute))
				}
			default:
				continue
			}

			// Publish an inbox event to the unconscious. The bus
			// pulses sub.Wake on its subscriber, which the
			// unconscious's Run loop listens for — same wake path
			// used by every other thread.
			t.bus.Publish(Event{
				Type: EventInbox,
				To:   "unconscious",
				Text: "[wake] " + reason,
			})
			logMsg("UNCONSCIOUS-FLOOR", fmt.Sprintf("force-woke unconscious: %s", reason))
			lastWakeAt = now
			lastHistorySize = sz
		}
	}
}

// findThreadManager walks the thread tree to find the ThreadManager that owns the given parent ID.
// Returns the Children manager of the parent thread, or nil if not found.
func findThreadManager(root *ThreadManager, parentID string) *ThreadManager {
	root.mu.RLock()
	defer root.mu.RUnlock()
	for _, thread := range root.threads {
		if thread.ID == parentID {
			return thread.Children // may be nil if parent is a leaf
		}
		// Recurse into children
		if thread.Children != nil {
			if found := findThreadManager(thread.Children, parentID); found != nil {
				return found
			}
		}
	}
	return nil
}

// mainToolHandler returns the tool handler for the main coordinating thread.
func mainToolHandler(t *Thinker) ToolHandler {
	return func(_ *Thinker, calls []toolCall, consumed []string) ([]string, []string, []ToolResult) {
		var replies []string
		var toolNames []string
		var results []ToolResult
		if len(calls) > 0 {
			names := make([]string, len(calls))
			for i, c := range calls {
				names[i] = c.Name
			}
			logMsg("TOOLS", fmt.Sprintf("[%s] handling %d tool call(s): %v", t.threadID, len(calls), names))
		}
		for _, call := range calls {
			// Check if this is an inline tool (handled here) or registry tool (handled by executeTool)
			isInline := true
			switch call.Name {
			case "spawn", "kill", "update", "send", "evolve", "remember", "pace", "connect", "disconnect", "list_connected", "done", "search_tools":
				// inline — we handle _reason and telemetry here
			default:
				isInline = false // executeTool handles _reason and telemetry
			}

			// Only strip _reason for inline tools — executeTool needs it
			reason := ""
			if isInline {
				reason = call.Args["_reason"]
				delete(call.Args, "_reason")
			}

			// Emit tool.call telemetry only for inline tools
			if isInline && t.telemetry != nil {
				t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
					ID: call.NativeID, Name: call.Name, Args: call.Args, Reason: reason,
				})
			}
			// Helper to add inline tool result + emit telemetry
			addResult := func(content string) {
				if call.NativeID != "" {
					results = append(results, ToolResult{CallID: call.NativeID, Content: content})
				}
				if t.telemetry != nil {
					t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
						ID: call.NativeID, Name: call.Name, Success: true, Result: content,
					})
				}
			}

			switch call.Name {
			case "spawn":
				id := call.Args["id"]
				directive := call.Args["directive"]
				if directive == "" {
					directive = call.Args["prompt"]
				}
				toolsStr := call.Args["tools"]
				var tools []string
				if toolsStr != "" {
					tools = strings.Split(toolsStr, ",")
				}
				mediaStr := call.Args["media"]
				mediaParts := parseMediaURLs(mediaStr)
				providerName := call.Args["provider"]
				// MCP scoping: child thread preloads tools from these
				// servers into its activeTools at boot. Accept both
				// `mcps` (new, plural, preferred) and `mcp` (legacy
				// singular-ish — actually always parsed as a list)
				// so an old prompt or downstream consumer that hasn't
				// migrated still works during transition.
				var mcpNames []string
				mcpStr := call.Args["mcps"]
				if mcpStr == "" {
					mcpStr = call.Args["mcp"]
				}
				if mcpStr != "" {
					for _, name := range strings.Split(mcpStr, ",") {
						if n := strings.TrimSpace(name); n != "" {
							mcpNames = append(mcpNames, n)
						}
					}
				}
				// Provider builtin scoping
				var builtinTools []string
				if btStr, hasBuiltins := call.Args["builtins"]; hasBuiltins {
					if btStr == "" {
						builtinTools = []string{} // explicit empty = no builtins
					} else {
						for _, bt := range strings.Split(btStr, ",") {
							if b := strings.TrimSpace(bt); b != "" {
								builtinTools = append(builtinTools, b)
							}
						}
					}
				}
				paused := parseTruthy(call.Args["paused"])
				realtime := parseTruthy(call.Args["realtime"])
				voice := call.Args["voice"]
				// Refuse realtime spawn cleanly when the feature gate is
				// off OR no realtime provider is available. The model
				// only sees realtime in the prompt when both are true,
				// so a request reaching here with neither is either
				// stale-prompt drift or a hand-crafted call — return a
				// clear error rather than a silent fail.
				if realtime {
					if t.config == nil || !t.config.RealtimeEnabledFlag() || t.pool == nil || !t.pool.HasRealtimeProvider() {
						logMsg("SPAWN", fmt.Sprintf("reject realtime id=%q: feature off (enabled=%v has_provider=%v)",
							id, t.config != nil && t.config.RealtimeEnabledFlag(), t.pool != nil && t.pool.HasRealtimeProvider()))
						addResult("error: realtime threads are not enabled on this instance")
						toolNames = append(toolNames, call.Raw)
						continue
					}
				}
				if id == "" || directive == "" {
					logMsg("SPAWN", fmt.Sprintf("skip: missing id=%q or directive_len=%d in LLM call", id, len(directive)))
					addResult(fmt.Sprintf("error: spawn requires both id and directive (got id=%q, directive_len=%d)", id, len(directive)))
				} else {
					logMsg("SPAWN", fmt.Sprintf("LLM-requested id=%q tools=%v mcp=%v provider=%q builtins=%v paused=%v realtime=%v voice=%q directive_len=%d",
						id, tools, mcpNames, providerName, builtinTools, paused, realtime, voice, len(directive)))
					err := t.threads.SpawnWithOpts(id, directive, tools, SpawnOpts{
						MediaParts:   mediaParts,
						ProviderName: providerName,
						ParentID:     "main",
						Depth:        0,
						MCPNames:     mcpNames,
						BuiltinTools: builtinTools,
						Paused:       paused,
						Realtime:     realtime,
						Voice:        voice,
					})
					if err != nil {
						logMsg("SPAWN", fmt.Sprintf("FAILED id=%q: %v", id, err))
						addResult(fmt.Sprintf("error: %v", err))
					} else {
						logMsg("SPAWN", fmt.Sprintf("OK id=%q", id))
						t.config.SaveThread(PersistentThread{ID: id, ParentID: "main", Depth: 0, Directive: directive, Tools: tools, MCPNames: mcpNames, Realtime: realtime, Voice: voice})
						switch {
						case paused:
							addResult(fmt.Sprintf("thread %s spawned (paused — send a message to wake it)", id))
						case realtime:
							addResult(fmt.Sprintf("realtime thread %s spawned", id))
						default:
							addResult(fmt.Sprintf("thread %s spawned", id))
						}
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "kill":
				id := call.Args["id"]
				if id == "" {
					addResult("error: kill requires id")
				} else {
					t.threads.Kill(id)
					t.config.RemoveThread(id)
					addResult(fmt.Sprintf("thread %s killed", id))
				}
				toolNames = append(toolNames, call.Raw)
			case "update":
				id := call.Args["id"]
				newID := call.Args["new_id"]
				name := call.Args["name"]
				directive := call.Args["directive"]
				toolsStr := call.Args["tools"]
				if id == "" {
					addResult("error: update requires id")
				} else if newID == "" && name == "" && directive == "" && toolsStr == "" {
					addResult("error: update requires at least one of new_id, name, directive, tools")
				} else {
					// Apply non-id changes first under the existing id, then
					// rename if requested. Doing rename last keeps a partial
					// failure recoverable: the persisted record survives
					// under the original id if rename fails.
					var tools []string
					if toolsStr != "" {
						tools = strings.Split(toolsStr, ",")
					}
					applyErr := error(nil)
					if name != "" || directive != "" || len(tools) > 0 {
						applyErr = t.threads.Update(id, name, directive, tools)
						if applyErr == nil && directive != "" {
							t.threads.Send(id, fmt.Sprintf("[directive updated] %s", directive))
						}
					}
					if applyErr != nil {
						addResult(fmt.Sprintf("error: %v", applyErr))
					} else if newID != "" {
						if err := t.threads.Rename(id, newID); err != nil {
							addResult(fmt.Sprintf("error: %v", err))
						} else {
							addResult(fmt.Sprintf("thread renamed %s → %s", id, newID))
						}
					} else {
						addResult(fmt.Sprintf("thread %s updated", id))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "send":
				id := call.Args["id"]
				msg := call.Args["message"]
				mediaStr := call.Args["media"]
				if id == "" || msg == "" {
					addResult(fmt.Sprintf("error: send requires both id and message (got id=%q, message_len=%d)", id, len(msg)))
				} else {
					// Tag with [from:main] so the receiving thread (and the
					// dashboard's IncomingEvents view) classifies the message
					// as a thread-to-thread send rather than a generic bus
					// event. Sub-thread sends already do this in thread.go;
					// main was missing it, so workers couldn't tell whether
					// a message came from main, the operator, or somewhere
					// else, and the dashboard rendered "bus" instead of
					// "[from:main]".
					tagged := fmt.Sprintf("[from:main] %s", msg)
					parts := parseMediaURLs(mediaStr)
					if !t.threads.SendWithParts(id, tagged, parts) {
						addResult(fmt.Sprintf("error: thread %q not found", id))
					} else {
						if t.telemetry != nil {
							t.telemetry.Emit("thread.message", "main", ThreadMessageData{From: "main", To: id, Message: msg})
						}
						addResult(fmt.Sprintf("sent to %s", id))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "evolve":
				d := call.Args["directive"]
				if d == "" {
					addResult("error: evolve requires directive")
				} else {
					t.config.SetDirective(d)
					t.directive = d
					t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(d, t.config.GetMode(), t.registry, "", t.mcpServers, nil, t.pool, t.mcpCatalog)}
					t.logAPI(APIEvent{Type: "evolved", ThreadID: "main", Message: d})
					if t.telemetry != nil {
						t.telemetry.Emit("directive.evolved", t.threadID, DirectiveChangeData{New: d})
					}
					addResult("directive updated")
				}
			case "remember":
				// Memory v2: main has no write tools. The unconscious is
				// the sole writer. If a legacy directive still calls
				// `remember`, surface a clear error so the agent (and the
				// operator looking at telemetry) understands why nothing
				// was stored — silent no-op would just hide the wiring
				// problem.
				addResult("error: remember is not available — memory writes are owned by the unconscious thread")
			case "search_tools":
				addResult(runSearchTools(t, call.Args, true /* main allows no_spawn results */))
				toolNames = append(toolNames, call.Raw)
				continue
			case "pace":
				var parts []string
				if s := call.Args["sleep"]; s != "" {
					if d, ok := parseSleepDuration(s); ok {
						t.agentSleep = d
						t.agentRate = RateSleep
						parts = append(parts, "sleep="+s)
					}
				} else if r, ok := rateNames[call.Args["rate"]]; ok {
					t.agentRate = r
					if d, ok2 := rateAliases[call.Args["rate"]]; ok2 {
						t.agentSleep = d
					}
					parts = append(parts, "rate="+call.Args["rate"])
				}
				if m, ok := modelNames[call.Args["model"]]; ok {
					t.agentModel = m
					parts = append(parts, "model="+call.Args["model"])
				}
				if pn := call.Args["provider"]; pn != "" && t.pool != nil {
					if p := t.pool.Get(pn); p != nil {
						t.provider = p
						parts = append(parts, "provider="+pn)
					}
				}
				if len(parts) > 0 {
					addResult("set " + strings.Join(parts, " "))
				} else {
					addResult("ok")
				}
			case "connect":
				name := call.Args["name"]
				command := call.Args["command"]
				argsStr := call.Args["args"]
				url := call.Args["url"]
				transport := call.Args["transport"]
				toolNames = append(toolNames, call.Raw)

				func() {
					if name == "" {
						// Silent no-op was hiding model confusion —
						// always emit a result so the tool_use is paired
						// and the model sees the error on its next turn.
						addResult("error: connect requires name=\"<server>\"")
						return
					}
					// Catalog fallback: if the model omitted command/url
					// but we already know this server from config (the
					// catalog shown to main in [AVAILABLE MCP SERVERS]),
					// use the stored config. This is what the model
					// usually "means" when it tries connect name=<catalog
					// name> — promote the server to main instead of
					// asking it to re-guess transport details the host
					// already knows.
					if command == "" && url == "" && t.config != nil {
						for _, sc := range t.config.GetMCPServers() {
							if sc.Name == name {
								command = sc.Command
								url = sc.URL
								transport = sc.Transport
								if len(sc.Args) > 0 && argsStr == "" {
									argsStr = strings.Join(sc.Args, ",")
								}
								break
							}
						}
					}
					if command == "" && url == "" {
						addResult(fmt.Sprintf("error: unknown server %q — either pass command=... (stdio) or url=... (http), or use a name listed in [AVAILABLE MCP SERVERS]", name))
						return
					}
					// Reject re-connect of an already-attached server so
					// the model gets a clear "already done" signal
					// instead of silently duplicating state.
					for _, srv := range t.mcpServers {
						if srv.GetName() == name {
							addResult(fmt.Sprintf("already connected to %s (use list_connected to see current servers)", name))
							return
						}
					}
					var mcpArgs []string
					if argsStr != "" {
						mcpArgs = strings.Split(argsStr, ",")
					}
					cfg := MCPServerConfig{Name: name, Command: command, Args: mcpArgs, URL: url, Transport: transport}
					srv, err := connectAnyMCP(cfg)
					if err != nil {
						addResult(fmt.Sprintf("error: %v", err))
						return
					}
					tools, err := srv.ListTools()
					if err != nil {
						srv.Close()
						addResult(fmt.Sprintf("error: %v", err))
						return
					}
					t.mcpServers = append(t.mcpServers, srv)
					for _, tool := range tools {
						fullName := name + "_" + tool.Name
						syntax := buildMCPSyntax(fullName, tool.InputSchema)
						t.registry.Register(&ToolDef{
							Name:        fullName,
							Description: fmt.Sprintf("[%s] %s", name, tool.Description),
							Syntax:      syntax,
							Rules:       fmt.Sprintf("Provided by MCP server '%s'.", name),
							Handler:     mcpProxyHandler(srv, tool.Name, t.blobs),
							InputSchema: tool.InputSchema,
							MCP:         true,
							MCPServer:   name,
						})
					}
					// Feed the search index so the freshly-connected
					// server's tools are discoverable via search_tools
					// and per-turn preload, same as startup-connected MCPs.
					if t.toolIndex != nil {
						t.toolIndex.Add(name, tools, cfg.NoSpawn)
						t.mcpCatalog = computeMCPCatalog(t.toolIndex)
					}
					t.config.SaveMCPServer(cfg)
					addResult(fmt.Sprintf("connected to %s: %d tools", name, len(tools)))
				}()
			case "disconnect":
				name := call.Args["name"]
				if name != "" {
					found := false
					for i, srv := range t.mcpServers {
						if srv.GetName() == name {
							srv.Close()
							t.mcpServers = append(t.mcpServers[:i], t.mcpServers[i+1:]...)
							t.registry.RemoveByMCPServer(name)
							if t.toolIndex != nil {
								t.toolIndex.Remove(name)
								t.mcpCatalog = computeMCPCatalog(t.toolIndex)
							}
							t.config.RemoveMCPServer(name)
							found = true
							break
						}
					}
					if found {
						addResult(fmt.Sprintf("disconnected from %s", name))
					} else {
						addResult(fmt.Sprintf("server %q not found", name))
					}
				}
				toolNames = append(toolNames, call.Raw)
			case "list_connected":
				var names []string
				for _, srv := range t.mcpServers {
					names = append(names, srv.GetName())
				}
				addResult(fmt.Sprintf("%d servers: %s", len(names), strings.Join(names, ", ")))
			default:
				// Dispatch to registry (MCP tools, etc)
				executeTool(t, call)
				toolNames = append(toolNames, call.Raw)
			}
		}
		return replies, toolNames, results
	}
}

func (t *Thinker) Run() {
	defer func() {
		if t.onStop != nil {
			t.onStop()
		}
	}()

	for {
		// Pause / quit handling.
		//
		// Three sources can pause this loop:
		//   1. PauseAll(true) from the parent (UI freeze, etc.)
		//   2. spawn(... paused=true) — the thread starts paused so the
		//      leader can configure / inspect before any thinking runs.
		//      thinker.paused is set BEFORE go Run() in that case.
		//   3. An explicit pause via the API.
		//
		// To wake: PauseAll(false), an explicit unpause, OR an inbox
		// event (a `send` from the leader, console input, etc.) — the
		// bus pulses sub.Wake on every delivery, which we listen for
		// here so spawn-paused workers come alive on their first message
		// without anyone needing to call unpause explicitly.
		if t.paused {
			select {
			case <-t.quit:
				return
			case p := <-t.pause:
				t.paused = p
			case <-t.sub.Wake:
				t.paused = false
				logMsg("RUN", fmt.Sprintf("[%s] unpaused by inbox event", t.threadID))
			}
		} else {
			select {
			case <-t.quit:
				return
			case p := <-t.pause:
				t.paused = p
				if t.paused {
					select {
					case p = <-t.pause:
						t.paused = p
					case <-t.sub.Wake:
						t.paused = false
						logMsg("RUN", fmt.Sprintf("[%s] unpaused by inbox event", t.threadID))
					case <-t.quit:
						return
					}
				}
			default:
			}
		}

		t.iteration++
		logMsg("RUN", fmt.Sprintf("[%s] iteration #%d start, rate=%s", t.threadID, t.iteration, t.rate.String()))

		// Drain events from bus, optionally filter/route
		drained := t.drainEvents()

		// Extract text strings, collect media parts, and separate tool results
		var consumed []string
		var mediaParts []ContentPart
		var toolResults []ToolResult
		for _, de := range drained {
			consumed = append(consumed, de.Text)
			mediaParts = append(mediaParts, de.Parts...)
			if de.ToolResult != nil {
				toolResults = append(toolResults, *de.ToolResult)
			}
		}

		// --- Iter-boundary wait barrier for parallel async tool calls ---
		// Without this, when the previous iteration dispatched N parallel
		// tool calls and only some of their results landed before the
		// first Wake, the half-finished batch would reach think() and
		// the model would retry the "missing" ones. The barrier drains
		// additional events as they arrive, up to a short deadline, and
		// for anything still pending after the deadline it injects a
		// placeholder tool_result (see injectPlaceholdersForPending) so
		// the tool_use is properly paired and the model is told not to
		// retry.
		t.waitForPendingTools(&toolResults, &consumed, &mediaParts, 3*time.Second)
		if t.pendingToolCount() > 0 {
			injectedBefore := len(toolResults)
			t.injectPlaceholdersForPending(&toolResults)
			if injected := len(toolResults) - injectedBefore; injected > 0 {
				logMsg("RUN", fmt.Sprintf("[%s] injected %d in-progress placeholders for tools still running", t.threadID, injected))
			}
		}
		t.sweepStalePlaceholders()

		if len(consumed) > 0 {
			logMsg("RUN", fmt.Sprintf("[%s] drained %d events (media_parts=%d)", t.threadID, len(consumed), len(mediaParts)))
			for i, ev := range consumed {
				preview := ev
				if len(preview) > 100 {
					preview = preview[:100] + "..."
				}
				logMsg("RUN", fmt.Sprintf("[%s]   event[%d]: %s", t.threadID, i, preview))

				// Telemetry: emit each drained event (skip tool results — those have their own telemetry)
				if t.telemetry != nil && !strings.HasPrefix(ev, "[tool:") {
					source := "bus"
					if strings.HasPrefix(ev, "[console]") {
						source = "console"
					} else if strings.HasPrefix(ev, "[from:") {
						source = "thread"
					} else if strings.HasPrefix(ev, "[webhook:") || strings.HasPrefix(ev, "[subscription:") {
						source = "webhook"
					}
					t.telemetry.Emit("event.received", t.threadID, map[string]string{
						"source":  source,
						"message": preview,
					})
				}
			}
		}
		// Only go reactive for non-tool events (user messages, console, thread sends)
		hasExternalEvent := false
		for _, ev := range consumed {
			if !strings.HasPrefix(ev, "[tool:") {
				hasExternalEvent = true
				break
			}
		}

		hadEvents := len(consumed) > 0
		if hasExternalEvent {
			t.rate = RateReactive
			t.model = ModelMedium
		} else if hadEvents {
			// Tool results — wake but less aggressive than external events
			t.rate = RateFast
		}

		// Minute-granularity timestamp keeps cache hits stable for
		// rapid-fire iterations within the same minute. The agent
		// rarely cares about second-level precision.
		now := time.Now().Format("2006-01-02 15:04")

		// If we have tool results, add them as a proper tool_result message first
		if len(toolResults) > 0 {
			trMsg := Message{Role: "user", ToolResults: toolResults}
			t.messages = append(t.messages, trMsg)
			if t.session != nil {
				t.session.AppendMessage(trMsg, t.iteration, TokenUsage{})
			}
		}

		// Compute the per-turn dynamic context: active threads + recall.
		// Both were previously poisoning messages[0] every iteration; now
		// they ride along on the current user-turn message so messages[0]
		// stays cache-stable.
		//
		// RAG tool retrieval used to live here too — it embedded tool
		// descriptions and injected the top-5 as text docs. It's gone:
		// MCP tool discovery is now search_tools + per-turn BM25 preload
		// + activeTools, and for native-tool providers every non-MCP
		// tool main can use is already in the NativeTools array. The RAG
		// path's only remaining effect was leaking MCP tools as text
		// (bypassing the MCP=true visibility gate) and feeding its own
		// injected text back into the preload query.
		var memQuery string
		if hadEvents {
			memQuery = strings.Join(consumed, " ")
		} else {
			for i := len(t.messages) - 1; i >= 0; i-- {
				if t.messages[i].Role == "assistant" {
					memQuery = t.messages[i].Content
					break
				}
			}
		}
		// Auto-inject the top-N most relevant active memories into this
		// turn's dynamic context. v2: Recall scores every active record
		// by signal × weight × decay (cosine when an embedding backend
		// is configured, lexical token-overlap otherwise), filters out
		// tombstoned/superseded entries automatically, and BuildContext
		// renders them with explicit "these are memories, not current
		// statements" framing — the structural defense against the
		// fabricated-approvals failure mode.
		var recallContext string
		if t.memory != nil && t.memory.Count() > 0 && memQuery != "" {
			recalled := t.memory.Recall(memQuery, 5)
			recallContext = t.memory.BuildContext(recalled)
		}
		var activeThreads []ThreadInfo
		if t.threads != nil {
			activeThreads = t.threads.List()
		}
		dynCtx := buildDynamicTurnContext(activeThreads, recallContext)

		if hadEvents {
			// Filter out tool result text from the events text (they're already in ToolResults)
			var textEvents []string
			for _, ev := range consumed {
				if len(toolResults) > 0 && strings.HasPrefix(ev, "[tool:computer_use]") {
					continue // skip, already handled as ToolResult
				}
				textEvents = append(textEvents, ev)
			}

			var sb strings.Builder
			if dynCtx != "" {
				sb.WriteString(dynCtx)
				sb.WriteString("\n\n")
			}
			if len(textEvents) > 0 {
				sb.WriteString(fmt.Sprintf("[%s] Events:\n", now))
				for _, ev := range textEvents {
					sb.WriteString("• " + ev + "\n")
				}
			}
			if sb.Len() > 0 || len(mediaParts) > 0 {
				msg := Message{Role: "user", Content: sb.String()}
				if len(mediaParts) > 0 {
					msg.Parts = append([]ContentPart{{Type: "text", Text: sb.String()}}, mediaParts...)
				}
				t.messages = append(t.messages, msg)
				if t.session != nil {
					t.session.AppendMessage(msg, t.iteration, TokenUsage{})
				}
			}
		} else if len(toolResults) == 0 {
			// Only add "no events" if we also have no tool results
			var content string
			if dynCtx != "" {
				content = dynCtx + "\n\n" + fmt.Sprintf("[%s] (no events)", now)
			} else {
				content = fmt.Sprintf("[%s] (no events)", now)
			}
			t.messages = append(t.messages, Message{Role: "user", Content: content})
		}

		// messages[0] is no longer rewritten per-iteration. It only
		// changes when the directive, mode, or static config (MCPs,
		// providers) does — handled at the call sites of buildSystemPrompt.

		start := time.Now()
		chatResp, err := t.think()

		// Fallback: if the provider errored and we have alternatives, try next in pool
		if err != nil && t.pool != nil && t.pool.Count() > 1 {
			original := t.provider.Name()
			if fb := t.pool.Fallback(original); fb != nil {
				logMsg("FALLBACK", fmt.Sprintf("[%s] %s failed (%v), trying %s", t.threadID, original, err, fb.Name()))
				t.provider = fb
				chatResp, err = t.think()
				if err != nil {
					// Restore original provider for next iteration
					t.provider = t.pool.Get(original)
				}
			}
		}

		duration := time.Since(start)
		reply := chatResp.Text
		usage := chatResp.Usage

		if err != nil {
			t.bus.Publish(Event{Type: EventThinkError, From: t.threadID, Error: err, Iteration: t.iteration})
			if t.telemetry != nil {
				t.telemetry.Emit("llm.error", t.threadID, LLMErrorData{
					Model: t.modelID(), Error: err.Error(), Iteration: t.iteration,
				})
			}
			select {
			case <-time.After(5 * time.Second):
			case <-t.quit:
				return
			}
			continue
		}

		// Build assistant message — may include native tool calls.
		//
		// Skip the append entirely when the model produced literally
		// nothing usable (no text, no tool_calls). Otherwise we
		// accumulate dead-air assistant turns that grow the message
		// history forever, waste tokens, and trigger Moonshot's
		// "must not be empty" rejection on the next call. Reasoning
		// alone doesn't justify keeping the turn — without text or
		// tool calls there's nothing for the agent to act on or the
		// provider to replay.
		if reply == "" && len(chatResp.ToolCalls) == 0 {
			logMsg("RUN", fmt.Sprintf("[%s] iter %d: model produced no text or tool calls — skipping assistant turn", t.threadID, t.iteration))
			continue
		}
		assistantMsg := Message{Role: "assistant", Content: reply, ToolCalls: chatResp.ToolCalls, Reasoning: chatResp.Reasoning}
		t.messages = append(t.messages, assistantMsg)

		// Persist to session history
		if t.session != nil {
			t.session.AppendMessage(assistantMsg, t.iteration, usage)
		}

		// Log server-executed built-in tool results (code execution, etc.)
		for _, sr := range chatResp.ServerResults {
			logMsg("BUILTIN", fmt.Sprintf("server tool %s: output=%s err=%s", sr.ToolName, truncateStr(sr.Output, 200), sr.Error))
			if t.telemetry != nil {
				t.telemetry.Emit("builtin.result", t.threadID, map[string]any{
					"tool":   sr.ToolName,
					"output": sr.Output,
					"error":  sr.Error,
				})
			}
		}

		// Log and stream native tool calls
		if len(chatResp.ToolCalls) > 0 {
			var names []string
			for _, ntc := range chatResp.ToolCalls {
				names = append(names, ntc.Name)
			}
			logMsg("RUN", fmt.Sprintf("[%s] LLM returned %d tool calls: %v", t.threadID, len(chatResp.ToolCalls), names))
			for _, ntc := range chatResp.ToolCalls {
				summary := "\n→ " + ntc.Name + "("
				first := true
				for k, v := range ntc.Args {
					if !first {
						summary += ", "
					}
					if len(v) > 60 {
						v = v[:60] + "..."
					}
					summary += k + "=" + v
					first = false
				}
				summary += ")"
				t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: summary, Iteration: t.iteration})
			}
		}

		// Dispatch tool calls via handler
		// Prefer native tool calls; fall back to text parsing if none
		var calls []toolCall
		if len(chatResp.ToolCalls) > 0 {
			for _, ntc := range chatResp.ToolCalls {
				// Intercept computer_use calls — execute via Computer interface with image ToolResults
				if isComputerUseTool(ntc.Name) && t.computer != nil {
					go t.executeComputerAction(ntc)
					continue
				}
				calls = append(calls, toolCall{Name: ntc.Name, Args: ntc.Args, Raw: ntc.Name, NativeID: ntc.ID})
			}
		}
		// NOTE: text-based [[...]] parsing removed — all providers use native tool calling now
		var replies []string
		var toolNames []string
		var inlineResults []ToolResult
		if t.handleTools != nil {
			replies, toolNames, inlineResults = t.handleTools(t, calls, consumed)
		}

		// Inject results for inline-handled tools (pace, spawn, kill, etc.)
		// so providers like Anthropic see matching tool_result for every tool_use
		if len(inlineResults) > 0 {
			t.messages = append(t.messages, Message{Role: "user", ToolResults: inlineResults})
			if t.session != nil {
				t.session.AppendMessage(Message{Role: "user", ToolResults: inlineResults}, t.iteration, TokenUsage{})
			}
		}

		// Sliding window — keep tool_use/tool_result pairs together
		maxHist := t.maxHistory
		if maxHist <= 0 {
			maxHist = maxHistoryMain // fallback
		}
		if len(t.messages) > maxHist+1 {
			start := len(t.messages) - maxHist
			// Don't start on a tool_result message (orphaned result)
			for start > 1 && len(t.messages[start].ToolResults) > 0 {
				start--
			}
			t.messages = append(t.messages[:1], t.messages[start:]...)
			// Sanitize any remaining orphaned tool_results after trimming
			// (no pending IDs needed here — this runs during the same iteration)
		}

		// Evict old screenshots — keep the 3 most recent images
		imageCount := 0
		maxImages := 3
		for i := len(t.messages) - 1; i >= 1; i-- {
			for j := range t.messages[i].ToolResults {
				if t.messages[i].ToolResults[j].Image != nil {
					imageCount++
					if imageCount > maxImages {
						// Replace old screenshot with text placeholder
						t.messages[i].ToolResults[j].Image = nil
						t.messages[i].ToolResults[j].Content = "[previous screenshot replaced — see latest for current screen state]"
					}
				}
			}
		}

		// Compact session history if it's grown too large
		if t.session != nil && t.session.NeedsCompaction() {
			logMsg("SESSION", fmt.Sprintf("[%s] triggering compaction (count=%d)", t.threadID, t.session.Count()))
			t.session.Compact(func(text string) string {
				// Simple summary — truncate to key points (no LLM call to avoid cost)
				if len(text) > 2000 {
					text = text[:2000]
				}
				return fmt.Sprintf("Summary of %d earlier messages: %s", t.session.Count(), text)
			})
		}

		// After processing, fall back to agent's chosen rate/sleep
		// (external events already set reactive above for this iteration)
		t.rate = t.agentRate
		t.model = t.agentModel

		// Compute actual sleep duration: agentSleep takes priority, else rate enum
		sleepDur := t.agentSleep
		if sleepDur <= 0 {
			sleepDur = t.rate.Delay()
		}

		// Thread count (0 if no thread manager)
		threadCount := 0
		if t.threads != nil {
			threadCount = t.threads.Count()
		}

		// Context size
		ctxChars := 0
		for _, msg := range t.messages {
			ctxChars += len(msg.Content)
		}

		t.bus.Publish(Event{
			Type: EventThinkDone, From: t.threadID,
			Iteration: t.iteration, Duration: duration,
			ConsumedEvents: consumed, Usage: usage,
			ToolCalls: toolNames, Replies: replies,
			Rate: t.rate, SleepDuration: sleepDur, Model: t.model,
			MemoryCount: t.memory.Count(), ThreadCount: threadCount,
			ContextMsgs: len(t.messages), ContextChars: ctxChars,
		})

		// Log to API — include full reply so tool calls are visible too
		thoughtLog := strings.TrimSpace(reply)
		if len(thoughtLog) > 1000 {
			thoughtLog = thoughtLog[:1000] + "..."
		}
		t.logAPI(APIEvent{Type: "thought", Iteration: t.iteration, Message: thoughtLog, Duration: duration.Round(time.Millisecond).String()})
		for _, r := range replies {
			t.logAPI(APIEvent{Type: "reply", Message: r})
		}

		// Telemetry: llm.done with full data
		if t.telemetry != nil {
			model := t.modelID()
			t.telemetry.Emit("llm.done", t.threadID, LLMDoneData{
				Model:            model,
				TokensIn:         usage.PromptTokens,
				TokensCached:     usage.CachedTokens,
				TokensOut:        usage.CompletionTokens,
				DurationMs:       duration.Milliseconds(),
				// cost_usd intentionally omitted — server enriches with
				// canonical pricing at ingest so we're not double-booking
				// the model→cost knowledge in core.
				Iteration:        t.iteration,
				Rate:             formatSleep(sleepDur),
				ContextMsgs:      len(t.messages),
				ContextChars:     ctxChars,
				MaxContextTokens: ModelContextWindow(model),
				MemoryCount:      t.memory.Count(),
				ThreadCount:      threadCount,
				Message:          thoughtLog,
			})
		}

		// Check if session needs compaction (background, non-blocking)
		if t.session != nil && t.session.NeedsCompaction() {
			go t.session.Compact(nil) // nil = simple count-based summary, no LLM call for now
		}

		// Interruptible sleep — wakes on new event, quit, or pause
		logMsg("RUN", fmt.Sprintf("[%s] sleeping %s", t.threadID, formatSleep(sleepDur)))
		select {
		case <-time.After(sleepDur):
			logMsg("RUN", fmt.Sprintf("[%s] woke: timer expired", t.threadID))
		case <-t.sub.Wake:
			logMsg("RUN", fmt.Sprintf("[%s] woke: event received", t.threadID))
		case p := <-t.pause:
			t.paused = p
			logMsg("RUN", fmt.Sprintf("[%s] paused=%v during sleep", t.threadID, t.paused))
			if t.paused {
				// Block until unpaused or quit
				select {
				case p = <-t.pause:
					t.paused = p
					logMsg("RUN", fmt.Sprintf("[%s] resumed", t.threadID))
				case <-t.quit:
					return
				}
			}
		case <-t.quit:
			logMsg("RUN", fmt.Sprintf("[%s] woke: quit signal", t.threadID))
			return
		}
	}
}

func (t *Thinker) think() (ChatResponse, error) {
	if t.provider == nil {
		return ChatResponse{}, fmt.Errorf("no provider configured")
	}

	// Sanitize messages before every API call �� removes orphaned tool_use/tool_result pairs
	// Pass pending tool IDs so the sanitizer doesn't strip in-flight async results
	if len(t.messages) > 1 {
		pending := map[string]bool{}
		t.pendingTools.Range(func(k, v any) bool {
			if id, ok := k.(string); ok {
				pending[id] = true
			}
			return true
		})
		t.messages = append(t.messages[:1], sanitizeToolPairs(t.messages[1:], pending)...)
	}

	onChunk := func(chunk string) {
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: chunk, Iteration: t.iteration})
		if t.telemetry != nil && chunk != "" {
			t.telemetry.EmitLive("llm.chunk", t.threadID, LLMChunkData{
				Text: chunk, Iteration: t.iteration,
			})
		}
	}

	// Build native tools from registry if provider supports it.
	//
	// Two modes (resolved per-turn by useEagerTools — see toolSearchMode):
	//   - eager: every attached tool, every turn. The list is stable
	//     turn-to-turn, so the `tools` prompt prefix caches well — the
	//     right call for a small surface where the search round-trip
	//     would cost more than it saves.
	//   - discovery: applyPreload STICKILY activates the BM25 top-5 for
	//     this turn's directive+context; search_tools adds more on
	//     demand. The active set grows then stabilises (then caches),
	//     bounded by evictActiveToolsLRU.
	var nativeTools []NativeTool
	if t.provider != nil && t.provider.SupportsNativeTools() && t.registry != nil {
		active := t.activeTools
		if t.useEagerTools() {
			all := t.toolIndex.AllNames(t.threadID == "main")
			if len(all) > 0 {
				// Transient merge — eager's "everything" set is already
				// stable, no need to pollute the sticky activeTools.
				merged := make(map[string]bool, len(t.activeTools)+len(all))
				for k := range t.activeTools {
					merged[k] = true
				}
				for _, n := range all {
					merged[n] = true
				}
				active = merged
			}
		} else {
			t.applyPreload(5)
			t.evictActiveToolsLRU(activeToolsCap)
			active = t.activeTools
		}
		nativeTools = t.registry.NativeTools(t.toolAllowlist, active)
	}

	// For Anthropic: add _display dimensions to computer_use tool params
	// so the provider can extract them for the native spec
	if t.computer != nil && t.provider != nil && t.provider.Name() == "anthropic" {
		display := t.computer.DisplaySize()
		logMsg("COMPUTER", fmt.Sprintf("injecting display dims for anthropic: %dx%d", display.Width, display.Height))
		for i, nt := range nativeTools {
			if nt.Name == "computer_use" {
				if nativeTools[i].Parameters == nil {
					nativeTools[i].Parameters = make(map[string]any)
				}
				nativeTools[i].Parameters["_display_width"] = display.Width
				nativeTools[i].Parameters["_display_height"] = display.Height
				break
			}
		}
	}

	onThinking := func(chunk string) {
		if t.telemetry != nil && chunk != "" {
			t.telemetry.EmitLive("llm.thinking", t.threadID, map[string]any{
				"text": chunk, "iteration": t.iteration,
			})
		}
	}

	onToolChunk := func(toolName, callID, chunk string) {
		t.bus.Publish(Event{Type: EventToolChunk, From: t.threadID, Text: chunk, ToolName: toolName, Iteration: t.iteration})
		if t.telemetry != nil {
			t.telemetry.EmitLive("llm.tool_chunk", t.threadID, map[string]any{
				"tool": toolName, "id": callID, "chunk": chunk, "iteration": t.iteration,
			})
		}
	}

	// Emit llm.start so the UI can show a "thinking" indicator before
	// any tokens arrive. Live-only — not stored in the DB.
	if t.telemetry != nil {
		t.telemetry.EmitLive("llm.start", t.threadID, map[string]any{
			"model":     t.modelID(),
			"iteration": t.iteration,
		})
	}

	// Bracket the provider call with enter/exit logs so we can see when
	// we go in and how long until we come out. Any "hang" on a spawn
	// request shows up here as an unbalanced enter with no exit.
	callStart := time.Now()
	logMsg("THINK", fmt.Sprintf("[%s] provider.Chat enter model=%s msgs=%d tools=%d",
		t.threadID, t.modelID(), len(t.messages), len(nativeTools)))
	// TODO: thread a cancellable ctx here (from a thinker-scoped run ctx or
	// a user-abort channel) so a slow stream can be unblocked from outside.
	// For now context.Background() preserves prior behaviour — the request
	// is now cancellable in principle, just nothing is wired to cancel it.
	resp, err := t.provider.Chat(context.Background(), t.messages, t.modelID(), nativeTools, onChunk, onThinking, onToolChunk)
	logMsg("THINK", fmt.Sprintf("[%s] provider.Chat exit model=%s dur=%s tool_calls=%d err=%v",
		t.threadID, t.modelID(), time.Since(callStart).Round(time.Millisecond), len(resp.ToolCalls), err))
	return resp, err
}

// drainEvents reads all pending events and wake signals from this thinker's bus subscription.
type drainedEvent struct {
	Text       string
	Parts      []ContentPart
	ToolResult *ToolResult
}

// drainEventTexts is a convenience for tests — returns just the text strings.
func (t *Thinker) drainEventTexts() []string {
	events := t.drainEvents()
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = e.Text
	}
	return out
}

func (t *Thinker) drainEvents() []drainedEvent {
	var items []drainedEvent
	for {
		select {
		case ev := <-t.sub.C:
			if ev.Type == EventInbox {
				items = append(items, drainedEvent{Text: ev.Text, Parts: ev.Parts, ToolResult: ev.ToolResult})
			}
		case <-t.sub.Wake:
			continue
		default:
			return items
		}
	}
}

// pendingToolCount returns the number of in-flight async tool calls.
// Used by the iteration wait barrier to decide whether to poll.
func (t *Thinker) pendingToolCount() int {
	n := 0
	t.pendingTools.Range(func(_, _ any) bool {
		n++
		return true
	})
	return n
}

// waitForPendingTools implements the iteration-boundary barrier that
// prevents the parallel-tool-call retry bug. Scenario:
//
//  1. LLM fires N parallel tool calls in one assistant message.
//  2. Goroutine A finishes fast, publishes its ToolResult.
//  3. The publish wakes sub.Wake → iter N+1 starts immediately.
//  4. drainEvents is non-blocking → captures only A's result.
//  5. Goroutines B, C, D are still running upstream at this instant.
//  6. iter N+1's think() sends a half-finished context to the LLM.
//  7. LLM rationalises "B/C/D missing results" as "retry B/C/D."
//
// This barrier inserts a bounded wait before think() runs: if any tool
// from the previous iteration is still in pendingTools, drain the bus
// repeatedly (absorbing events as they arrive) until either pendingTools
// is empty or the deadline fires. Any extracted events are appended to
// the caller's slices so they end up in t.messages as usual.
//
// Bounded to keep genuine long-running tools from freezing the main
// loop. When the deadline fires and some tools are still pending, the
// caller is expected to inject placeholder tool_results for them (see
// injectPlaceholdersForPending).
func (t *Thinker) waitForPendingTools(
	toolResults *[]ToolResult,
	consumed *[]string,
	mediaParts *[]ContentPart,
	deadline time.Duration,
) {
	if t.pendingToolCount() == 0 {
		return
	}
	start := time.Now()
	poll := time.NewTicker(20 * time.Millisecond)
	defer poll.Stop()
	deadlineCh := time.After(deadline)
	for {
		// Drain whatever's in the bus right now.
		for {
			select {
			case ev := <-t.sub.C:
				if ev.Type == EventInbox {
					if ev.ToolResult != nil {
						*toolResults = append(*toolResults, *ev.ToolResult)
					}
					if ev.Text != "" {
						*consumed = append(*consumed, ev.Text)
					}
					if len(ev.Parts) > 0 {
						*mediaParts = append(*mediaParts, ev.Parts...)
					}
					continue
				}
			case <-t.sub.Wake:
				continue
			default:
			}
			break
		}
		if t.pendingToolCount() == 0 {
			logMsg("RUN", fmt.Sprintf("[%s] pending tools drained in %s", t.threadID, time.Since(start)))
			return
		}
		select {
		case <-deadlineCh:
			logMsg("RUN", fmt.Sprintf("[%s] pending tool wait deadline (%s) — %d still in-flight, injecting placeholders", t.threadID, deadline, t.pendingToolCount()))
			return
		case <-poll.C:
			continue
		case <-t.quit:
			return
		}
	}
}

// injectPlaceholdersForPending synthesises a "⏳ in progress" ToolResult
// for every tool id still in pendingTools at the iteration boundary. This
// keeps each tool_use paired with a tool_result for API legality AND
// tells the model explicitly not to retry. When the real result later
// arrives from the goroutine, tools.go routes it through a distinct
// "late-result" text message (see late-result routing below) instead of
// appending a second ToolResult for the same id.
func (t *Thinker) injectPlaceholdersForPending(toolResults *[]ToolResult) {
	t.pendingTools.Range(func(k, v any) bool {
		id, ok := k.(string)
		if !ok || id == "" {
			return true
		}
		// Skip ids that already have a placeholder from an earlier
		// iteration — those are still in-flight, their placeholder is
		// already in the assistant/user message pair in history.
		if _, existed := t.placeholdersSent.Load(id); existed {
			return true
		}
		toolName, _ := v.(string)
		*toolResults = append(*toolResults, ToolResult{
			CallID:  id,
			Content: "⏳ In progress — this tool is still running from an earlier iteration. A [late-result] message will be delivered as soon as it completes. DO NOT call this tool again with the same arguments.",
		})
		t.placeholdersSent.Store(id, placeholderInfo{
			iteration:    t.iteration,
			toolName:     toolName,
			dispatchedAt: time.Now(),
		})
		return true
	})
}

// sweepStalePlaceholders emits a synthetic timeout late-result for any
// placeholder whose real goroutine never completed. Runs once per
// iteration; the default thresholds (5 minutes wall-clock or 20
// iterations) match the worst-case retry/backoff envelope of upstream
// MCP calls. Prevents placeholdersSent from growing unbounded when a
// tool genuinely hangs.
func (t *Thinker) sweepStalePlaceholders() {
	now := time.Now()
	var stale []string
	t.placeholdersSent.Range(func(k, v any) bool {
		id, ok1 := k.(string)
		info, ok2 := v.(placeholderInfo)
		if !ok1 || !ok2 {
			return true
		}
		if now.Sub(info.dispatchedAt) > 5*time.Minute || t.iteration-info.iteration > 20 {
			stale = append(stale, id)
			t.Inject(fmt.Sprintf("[late-result] Tool %s (call id=%s, dispatched iter %d) timed out after %s — no result ever arrived. Treat as failure.",
				info.toolName, id, info.iteration, now.Sub(info.dispatchedAt).Round(time.Second)))
		}
		return true
	})
	for _, id := range stale {
		t.placeholdersSent.Delete(id)
		// Don't delete from pendingTools — the goroutine may still
		// complete and we want its late-result path to fire naturally.
	}
}

func (t *Thinker) logAPI(ev APIEvent) {
	if t.apiNotify == nil || t.apiLog == nil {
		return
	}
	ev.Time = time.Now()
	if ev.ThreadID == "" {
		ev.ThreadID = t.threadID
	}
	t.apiMu.Lock()
	*t.apiLog = append(*t.apiLog, ev)
	if len(*t.apiLog) > 1000 {
		*t.apiLog = (*t.apiLog)[len(*t.apiLog)-500:]
	}
	t.apiMu.Unlock()
	select {
	case t.apiNotify <- struct{}{}:
	default:
	}
}

func (t *Thinker) APIEvents(since int) ([]APIEvent, int) {
	t.apiMu.RLock()
	defer t.apiMu.RUnlock()
	if since >= len(*t.apiLog) {
		return nil, len(*t.apiLog)
	}
	events := make([]APIEvent, len(*t.apiLog)-since)
	copy(events, (*t.apiLog)[since:])
	return events, len(*t.apiLog)
}

func (t *Thinker) ReloadDirective() {
	directive := t.config.GetDirective()
	t.directive = directive
	t.messages[0] = Message{Role: "system", Content: buildSystemPrompt(directive, t.config.GetMode(), t.registry, "", t.mcpServers, nil, t.pool, t.mcpCatalog)}
	t.InjectConsole("Directive updated to: " + directive + "\n\nAdjust the system accordingly — spawn, kill, or reconfigure threads as needed.")
}

// Inject sends a message event to this thinker's bus subscription.
func (t *Thinker) Inject(msg string) {
	logMsg("INJECT", fmt.Sprintf("to=%s msg=%s", t.threadID, msg))
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: msg})
}

// InjectConsole sends a console event to this thinker.
func (t *Thinker) InjectConsole(msg string) {
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: "[console] " + msg})
}

// InjectWithParts sends a text event with media parts attached.
func (t *Thinker) InjectWithParts(text string, parts []ContentPart) {
	if text == "" {
		text = "[multimodal input]"
	}
	t.bus.Publish(Event{Type: EventInbox, To: t.threadID, Text: "[console] " + text, Parts: parts})
}

// parseMediaURLs splits a space-separated list of URLs into ContentParts.
// Classifies each URL as image or audio by extension.
func parseMediaURLs(urls string) []ContentPart {
	if urls == "" {
		return nil
	}
	var parts []ContentPart
	for _, u := range strings.Fields(urls) {
		u = strings.TrimSpace(u)
		if u == "" {
			continue
		}
		ext := ""
		if idx := strings.LastIndex(u, "."); idx >= 0 {
			ext = strings.ToLower(u[idx+1:])
			if qIdx := strings.Index(ext, "?"); qIdx >= 0 {
				ext = ext[:qIdx]
			}
		}
		switch ext {
		case "mp3", "wav", "aac", "ogg", "flac", "aiff", "m4a":
			parts = append(parts, ContentPart{Type: "audio_url", AudioURL: &AudioURL{URL: u}})
		case "png", "jpg", "jpeg", "gif", "webp":
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: u}})
		default:
			// Unknown extension — treat as image (provider will attempt fetch)
			parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: u}})
		}
	}
	return parts
}

func (t *Thinker) TogglePause() {
	newState := !t.paused
	// Non-blocking send — channel is buffered(1), drain any stale value first
	select {
	case <-t.pause:
	default:
	}
	t.pause <- newState
	t.paused = newState
	// Pause/resume all child threads too
	if t.threads != nil {
		t.threads.PauseAll(newState)
	}
}

// Shutdown releases external resources held by the thinker: currently
// only the computer-use browser session. Safe to call multiple times.
// Used by the main signal handler so SIGTERM/SIGINT closes Chrome
// (local) or REQUEST_RELEASEs the session (Browserbase) instead of
// orphaning it when the server SIGKILLs core during instance stop.
func (t *Thinker) Shutdown() {
	if t == nil {
		return
	}
	if c := t.computer; c != nil {
		t.computer = nil
		_ = c.Close()
	}
}

// SetComputer attaches a computer use environment to this thinker.
// Registers computer_use as a tool in the registry for non-Anthropic providers.
func (t *Thinker) SetComputer(c computer.Computer) {
	t.computer = c
	if c != nil && t.registry != nil {
		def := computer.GetComputerToolDef(c.DisplaySize())
		// Register computer_use — screen interaction (no navigate)
		comp := c
		t.registry.Register(&ToolDef{
			Name:        def.Name,
			Description: def.Description,
			Syntax:      def.Syntax,
			Rules:       def.Rules,
			InputSchema: def.Parameters,
			Handler: func(args map[string]string) ToolResponse {
				text, screenshot, err := computer.HandleComputerAction(comp, args)
				if err != nil {
					return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
				}
				return ToolResponse{Text: text, Image: screenshot}
			},
		})

		// Register browser_session — session lifecycle (open/close/resume/status)
		sessionDef := computer.GetSessionToolDef()
		t.registry.Register(&ToolDef{
			Name:        sessionDef.Name,
			Description: sessionDef.Description,
			Syntax:      sessionDef.Syntax,
			Rules:       sessionDef.Rules,
			InputSchema: sessionDef.Parameters,
			Handler: func(args map[string]string) ToolResponse {
				text, screenshot, err := computer.HandleSessionAction(comp, args)
				if err != nil {
					return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
				}
				return ToolResponse{Text: text, Image: screenshot}
			},
		})
	}
}

func (t *Thinker) Stop() {
	select {
	case <-t.quit:
	default:
		close(t.quit)
	}
	// Clean up computer session
	if t.computer != nil {
		t.computer.Close()
	}
}

// isComputerUseTool returns true if the tool name is a computer use tool from any provider.
func isComputerUseTool(name string) bool {
	switch name {
	case "computer_use", "computer", "computer_use_2025", "computer_20250124":
		return true
	}
	// Gemini native Computer Use actions
	return computer.IsGeminiComputerAction(name)
}

// normalizeComputerAction converts provider-specific args to a computer.Action.
func normalizeComputerAction(args map[string]string) computer.Action {
	action := computer.Action{Type: computer.NormalizeActionType(args["action"])}

	// Parse coordinate — providers use different formats
	// Anthropic: coordinate=[x, y] as string; OpenAI: x=400, y=300
	if coord := args["coordinate"]; coord != "" {
		// Parse "[400, 300]" format
		coord = strings.Trim(coord, "[] ")
		parts := strings.Split(coord, ",")
		if len(parts) == 2 {
			fmt.Sscanf(strings.TrimSpace(parts[0]), "%d", &action.X)
			fmt.Sscanf(strings.TrimSpace(parts[1]), "%d", &action.Y)
		}
	}
	if x := args["x"]; x != "" {
		fmt.Sscanf(x, "%d", &action.X)
	}
	if y := args["y"]; y != "" {
		fmt.Sscanf(y, "%d", &action.Y)
	}

	action.Text = args["text"]
	action.Key = args["key"]
	action.Direction = args["direction"]
	action.URL = args["url"]

	if d := args["duration"]; d != "" {
		fmt.Sscanf(d, "%d", &action.Duration)
	}

	// Set-of-Mark label. Providers may stringify a JSON integer as
	// unquoted "1" or quoted "\"1\""; strip quotes and parse.
	if lbl := strings.Trim(strings.TrimSpace(args["label"]), `"`); lbl != "" {
		fmt.Sscanf(lbl, "%d", &action.Label)
	}

	return action
}

// executeComputerAction runs a computer_use action and injects the result as a proper ToolResult.
func (t *Thinker) executeComputerAction(ntc NativeToolCall) {
	if ntc.ID != "" {
		t.pendingTools.Store(ntc.ID, ntc.Name)
		defer t.pendingTools.Delete(ntc.ID)
	}
	logMsg("COMPUTER", fmt.Sprintf("action=%s args=%v", ntc.Name, ntc.Args))
	reason := ntc.Args["_reason"]
	delete(ntc.Args, "_reason")

	// Emit tool.call telemetry
	if t.telemetry != nil {
		t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
			ID: ntc.ID, Name: ntc.Name, Args: ntc.Args, Reason: reason,
		})
	}
	start := time.Now()

	var screenshot []byte
	var err error
	var actionLabel string

	// Gemini native Computer Use actions (click_at, type_text_at, etc.)
	if computer.IsGeminiComputerAction(ntc.Name) {
		var text string
		text, screenshot, err = computer.HandleGeminiComputerAction(t.computer, ntc.Name, ntc.Args)
		_ = text
		actionLabel = ntc.Name
	} else {
		// Anthropic/generic computer_use (single tool with "action" arg)
		action := normalizeComputerAction(ntc.Args)
		actionLabel = action.Type
		screenshot, err = t.computer.Execute(action)
	}

	duration := time.Since(start)

	if err != nil {
		logMsg("COMPUTER", fmt.Sprintf("error (%dms): %v", duration.Milliseconds(), err))
		if t.telemetry != nil {
			t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
				ID: ntc.ID, Name: ntc.Name, DurationMs: duration.Milliseconds(), Success: false, Result: err.Error(),
			})
		}
		// Inject as tool result with error
		t.bus.Publish(Event{
			Type: EventInbox, To: t.threadID,
			Text: fmt.Sprintf("[tool:computer_use] error: %v", err),
			ToolResult: &ToolResult{
				CallID:  ntc.ID,
				Content: fmt.Sprintf("Error: %v", err),
				IsError: true,
			},
		})
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID,
			Text: "\n← computer_use: error: " + err.Error() + "\n", Iteration: t.iteration})
		return
	}

	logMsg("COMPUTER", fmt.Sprintf("done (%dms) screenshot=%d bytes", duration.Milliseconds(), len(screenshot)))
	if t.telemetry != nil {
		t.telemetry.Emit("tool.result", t.threadID, ToolResultData{
			ID: ntc.ID, Name: ntc.Name, DurationMs: duration.Milliseconds(), Success: true,
			Result: fmt.Sprintf("screenshot %d bytes", len(screenshot)),
		})
	}

	// Also stash the screenshot bytes in the BlobStore and surface
	// the handle in the result text. The Image field stays attached
	// so vision input is unchanged — this is purely additive: gives
	// the agent a way to forward these exact bytes to another tool
	// (e.g. files_upload via storage MCP) without re-encoding from
	// the vision channel (which it can't do anyway).
	content := fmt.Sprintf("Success: %s action completed. A screenshot of the current screen is attached as an image. Examine it to see the result.", actionLabel)
	if t.blobs != nil && len(screenshot) > 0 {
		mime := "image/jpeg"
		if len(screenshot) >= 8 &&
			screenshot[0] == 0x89 && screenshot[1] == 0x50 &&
			screenshot[2] == 0x4E && screenshot[3] == 0x47 {
			mime = "image/png"
		}
		ref := t.blobs.Put(mime, screenshot)
		content += fmt.Sprintf(" To forward these exact bytes to another tool, pass the file handle %s as that tool's bytes argument.", ref)
	}

	// Inject as tool result with screenshot image
	t.bus.Publish(Event{
		Type: EventInbox, To: t.threadID,
		Text: fmt.Sprintf("[tool:computer_use] success: %s completed, screenshot attached (%d bytes, %dms)", actionLabel, len(screenshot), duration.Milliseconds()),
		ToolResult: &ToolResult{
			CallID:  ntc.ID,
			Content: content,
			Image:   screenshot,
		},
	})

	t.bus.Publish(Event{Type: EventChunk, From: t.threadID,
		Text: fmt.Sprintf("\n← computer_use: screenshot (%d bytes, %dms)\n", len(screenshot), duration.Milliseconds()),
		Iteration: t.iteration})
}

func encodeBase64(data []byte) string {
	return base64Encode(data)
}
