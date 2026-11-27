package core

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// AutomaticToolLoadingConfig controls request-time schema loading.
// An omitted config enables bounded automatic loading and memory relevance;
// an explicit enabled:false retains legacy preload with search as a fallback.
type AutomaticToolLoadingConfig struct {
	Enabled         bool     `json:"enabled"`
	WorkflowTools   []string `json:"workflow_tools,omitempty"` // canonical MCP identities
	MaxTools        int      `json:"max_tools,omitempty"`
	MaxSchemaTokens int      `json:"max_schema_tokens,omitempty"` // approximate, JSON bytes / 4
	IncludeMemory   bool     `json:"include_memory,omitempty"`
}

func validateAutomaticToolLoading(c *AutomaticToolLoadingConfig) error {
	if c == nil {
		return nil
	}
	if c.MaxTools < 0 || c.MaxTools > 20 {
		return fmt.Errorf("automatic_tool_loading.max_tools must be between 0 (default) and 20")
	}
	if c.MaxSchemaTokens < 0 || c.MaxSchemaTokens > 16000 {
		return fmt.Errorf("automatic_tool_loading.max_schema_tokens must be between 0 (default) and 16000")
	}
	if len(c.WorkflowTools) > 20 {
		return fmt.Errorf("automatic_tool_loading.workflow_tools exceeds 20 tools")
	}
	for _, name := range c.WorkflowTools {
		if !validToolName(name) || len(name) > 256 {
			return fmt.Errorf("automatic_tool_loading.workflow_tools requires valid canonical names")
		}
	}
	return nil
}

// defaultAutomaticSchemaTokens fits roughly six to eight typical app tool
// schemas. At 2,000, workers with large Processes and CRM schemas got two to
// four tools and searched for the rest.
const defaultAutomaticSchemaTokens = 6000

func (c *Config) GetAutomaticToolLoading() AutomaticToolLoadingConfig {
	out := AutomaticToolLoadingConfig{Enabled: true, IncludeMemory: true}
	if c != nil {
		c.mu.RLock()
		if c.AutomaticToolLoading != nil {
			out = *c.AutomaticToolLoading
			out.WorkflowTools = append([]string(nil), out.WorkflowTools...)
		}
		c.mu.RUnlock()
	}
	if out.MaxTools == 0 {
		out.MaxTools = 8
	}
	if out.MaxSchemaTokens == 0 {
		out.MaxSchemaTokens = defaultAutomaticSchemaTokens
	}
	return out
}

type automaticToolSelection struct {
	policy  string
	context string
	// directive is a hash of the directive the selection was made for.
	directive string
	names     []string
	reasons   map[string]string
}

type automaticToolQuery struct{ Source, Text string }

func boundedToolQuery(text string) string {
	const maxBytes = 8192
	if len(text) > maxBytes {
		text = text[:maxBytes]
	}
	return strings.TrimSpace(text)
}

func (t *Thinker) automaticToolQueries(c AutomaticToolLoadingConfig) []automaticToolQuery {
	var out []automaticToolQuery
	add := func(source, text string) {
		if text = boundedToolQuery(text); text != "" {
			out = append(out, automaticToolQuery{source, text})
		}
	}
	// Prefer the durable current instruction so clearing lastInboundForPreload
	// after a request does not cause a different selection on continuation.
	current, _ := recallQueryForTurn(nil, t.messages, "")
	if current == "" {
		current = t.lastInboundForPreload
	}
	add("instruction", current)
	// Only this thread's active executions contribute task state. Do not scan
	// other workers' contexts or use assistant narration as a relevance signal.
	ids := t.currentEventExecutions()
	if t.config != nil && len(ids) > 0 {
		t.config.mu.RLock()
		var task []string
		for _, ex := range t.config.EventExecutions {
			if containsString(ids, ex.ExecutionID) && !executionTerminal(ex.Status) {
				task = append(task, boundedToolQuery(ex.Reason))
				if ex.RequiredFirstAction != nil && !ex.RequiredFirstAction.Satisfied {
					task = append(task, ex.RequiredFirstAction.Tool)
				}
			}
			if len(task) >= 20 {
				break
			}
		}
		t.config.mu.RUnlock()
		sort.Strings(task)
		add("task", strings.Join(task, "\n"))
	}
	add("directive", t.directive)
	// Reuse recall that already ran for this model request. Never perform an
	// extra memory/model request, and never treat remembered names as grants.
	// A worker thread with an assigned task already has its instruction, task
	// and directive; recall there picks tools from unrelated past work and
	// spends the budget the task needs.
	if c.IncludeMemory && !t.assignedWorker(ids) {
		add("memory", t.memoryRecall.context)
	}
	return out
}

// assignedWorker reports a thread other than main that is working on a
// tracked task, such as a workflow step worker.
func (t *Thinker) assignedWorker(executions []string) bool {
	return t.threadID != "main" && len(executions) > 0
}

// exactToolNames returns canonical MCP tool names written in text, in order.
// A task that names its tools should get them before relevance guesses.
func (t *Thinker) exactToolNames(text string) []string {
	if t.toolIndex == nil {
		return nil
	}
	var out []string
	seen := map[string]bool{}
	for _, token := range toolNameTokens(text) {
		if seen[token] || !strings.Contains(token, "_") {
			continue
		}
		seen[token] = true
		if entry, ok := t.toolIndex.Get(token); ok {
			out = append(out, entry.Name)
		}
	}
	return out
}

// automaticQueryNormalizer recognizes references to existing non-MCP tools
// and schema fields as context, not failed MCP lookups. Unknown names stay
// intact so the exact-name safeguard still applies. Registry/index identities
// take precedence over parameter vocabulary, including denied identities.
func (t *Thinker) automaticQueryNormalizer() func(string) string {
	nonMCP, parameters, identities := map[string]bool{}, map[string]bool{}, map[string]bool{}
	for _, def := range t.registry.AllTools() {
		identities[strings.ToLower(def.Name)] = true
		if !t.toolAuthorized(def.Name) || (def.SystemOnly && !t.systemThread) {
			continue
		}
		if entry, ok := t.toolIndex.Get(def.Name); ok && entry.NoSpawn && t.threadID != "main" && !t.allowNoSpawn {
			continue
		}
		if !def.MCP {
			nonMCP[strings.ToLower(def.Name)] = true
		}
		if props, ok := def.native.Parameters["properties"].(map[string]any); ok {
			for name := range props {
				parameters[strings.ToLower(name)] = true
			}
		}
	}
	t.toolIndex.mu.RLock()
	for name := range t.toolIndex.byName {
		identities[name] = true
	}
	for name := range t.toolIndex.aliases {
		identities[name] = true
	}
	for name := range t.toolIndex.localNames {
		identities[name] = true
	}
	t.toolIndex.mu.RUnlock()
	return func(query string) string {
		var terms []string
		for _, token := range toolNameTokens(query) {
			if nonMCP[token] {
				continue
			}
			if !identities[token] && parameters[token] {
				token = strings.ReplaceAll(token, "_", " ")
			}
			terms = append(terms, token)
		}
		return strings.Join(terms, " ")
	}
}

// prepareAutomaticTools returns an independent, bounded visibility overlay.
// Search/use activation and required/always schemas remain under their existing
// rules. Disabling the experiment removes this overlay on the next request.
func (t *Thinker) prepareAutomaticTools(c AutomaticToolLoadingConfig) map[string]bool {
	started := time.Now()
	if !c.Enabled || validateAutomaticToolLoading(&c) != nil || t.toolIndex == nil || t.registry == nil {
		t.automaticTools = automaticToolSelection{}
		return nil
	}
	queries := t.automaticToolQueries(c)
	rawPolicy, _ := json.Marshal(c)
	// Retrieval and required-action bookkeeping can change on every tool
	// continuation. They are selection inputs, not phase boundaries.
	// Hash the full instruction, even though search uses a bounded excerpt.
	// Different long tasks with the same opening are different phases.
	instruction, _ := recallQueryForTurn(nil, t.messages, "")
	if instruction == "" {
		instruction = t.lastInboundForPreload
	}
	phaseQueries := []automaticToolQuery{{Source: "instruction", Text: instruction}, {Source: "directive", Text: t.directive}}
	executions := append([]string(nil), t.currentEventExecutions()...)
	sort.Strings(executions)
	rawContext, _ := json.Marshal(struct {
		Queries    []automaticToolQuery
		Executions []string
	}{phaseQueries, executions})
	policy, contextKey := string(rawPolicy), promptCacheShortHash(rawContext)
	previous := t.automaticTools
	if previous.policy != policy {
		previous = automaticToolSelection{}
	}
	// Tools chosen for an earlier directive are matched again against the
	// current one instead of being carried forward, so what an agent gets
	// follows from its directive and not from what it was given in the past.
	directiveKey := promptCacheShortHash([]byte(t.directive))
	if previous.directive != directiveKey && len(previous.names) > 0 {
		var kept []string
		for _, name := range previous.names {
			if previous.reasons[name] != "directive" {
				kept = append(kept, name)
			}
		}
		previous.names = kept
	}
	normalize := t.automaticQueryNormalizer()
	allowNoSpawn := t.threadID == "main" || t.allowNoSpawn
	available := t.authorizedDiscoveryServers(allowNoSpawn)
	var intents []DiscoveryIntent
	for _, name := range c.WorkflowTools {
		if _, ok := t.toolIndex.Get(name); !ok {
			continue
		}
		intents = append(intents, compileDiscoveryIntent(name, nil, DiscoveryAccessAny, 1, "workflow", 100, available)...)
	}
	// Revalidate frozen names against the live catalog, grants and schema
	// budget each time. Explicit search can still introduce newly needed tools.
	sticky := requestStabilityEnabled("APTEVA_STABLE_TOOL_SELECTION")
	if (sticky || previous.context == contextKey) && len(previous.names) > 0 {
		for _, name := range previous.names {
			intents = append(intents, compileDiscoveryIntent(name, nil, DiscoveryAccessAny, 1, previous.reasons[name], 90, available)...)
		}
	}
	if len(previous.names) == 0 || previous.context != contextKey {
		// Tool names written in the instruction, task or directive load at
		// workflow priority, for up to half the tool budget, so the rest
		// stays open for relevance matches.
		exactLimit := c.MaxTools / 2
		if exactLimit < 1 {
			exactLimit = 1
		}
		exactSeen := map[string]bool{}
		for _, q := range queries {
			// Short acknowledgements are not evidence for new capabilities.
			// Explicit tool/server references still work even as one-word input.
			if sticky && len(previous.names) > 0 && !t.automaticToolExpansion(q.Text, available) {
				continue
			}
			if q.Source != "memory" {
				for _, name := range t.exactToolNames(q.Text) {
					if len(exactSeen) >= exactLimit || exactSeen[name] {
						continue
					}
					exactSeen[name] = true
					intents = append(intents, compileDiscoveryIntent(name, nil, DiscoveryAccessAny, 1, q.Source, 100, available)...)
				}
			}
			priority := 70
			if q.Source == "memory" {
				priority = 20
			}
			if q.Source == "directive" {
				// A directive is many instructions. Match each sentence on
				// its own; one search over the whole text lets generic
				// words outvote the sentence that names an action.
				if ordered := t.directiveToolOrder(q.Text, normalize, available, allowNoSpawn); len(ordered) > 0 {
					intents = append(intents, DiscoveryIntent{Query: "directive", Access: DiscoveryAccessPreferRead, Limit: len(ordered), Source: q.Source, Priority: priority, preset: ordered})
				}
				continue
			}
			intents = append(intents, compileDiscoveryIntent(normalize(q.Text), nil, DiscoveryAccessPreferRead, c.MaxTools, q.Source, priority, available)...)
		}
	}
	memoryMax := c.MaxTools / 4
	if memoryMax < 1 {
		memoryMax = 1
	}
	memorySchemaMax := c.MaxSchemaTokens / 4
	if memorySchemaMax < 1 {
		memorySchemaMax = 1
	}
	discovery := t.discoverTools(DiscoveryRequest{Intents: intents, AllowNoSpawn: allowNoSpawn, MaxTools: c.MaxTools, MaxSchemaTokens: c.MaxSchemaTokens, MemoryMaxTools: memoryMax, MemoryMaxSchemaTokens: memorySchemaMax})
	selected, reasons := map[string]bool{}, map[string]string{}
	for _, name := range discovery.Loaded {
		selected[name] = true
	}
	for _, group := range discovery.Results {
		for _, candidate := range group.Candidates {
			if candidate.Loaded && reasons[candidate.Name] == "" {
				reasons[candidate.Name] = group.Source
			}
		}
	}
	t.automaticTools = automaticToolSelection{policy: policy, context: contextKey, directive: directiveKey, names: append([]string(nil), discovery.Loaded...), reasons: reasons}
	if t.telemetry != nil {
		t.telemetry.Emit("tool.autoload", t.threadID, map[string]any{"iteration": t.iteration, "selected": discovery.Loaded, "reasons": reasons, "not_loaded": automaticToolsNotLoaded(discovery), "schema_tokens_est": discovery.SchemaTokens, "max_tools": c.MaxTools, "max_schema_tokens": c.MaxSchemaTokens, "skipped_budget": discovery.SkippedBudget, "catalog_revision": discovery.CatalogRevision, "duration_us": time.Since(started).Microseconds(), "context_hash": contextKey})
	}
	return selected
}

func (t *Thinker) automaticToolExpansion(query string, servers []string) bool {
	for _, server := range servers {
		if containsServerReference(query, server) {
			return true
		}
	}
	return len(toolNameTokens(query)) >= 3
}

const (
	directiveUnitLimit      = 48
	directiveUnitCandidates = 3
	automaticNotLoadedLimit = 12
)

// directiveUnits splits a directive into the sentences and bullets it is made
// of. Headings, list markers and fragments under three words are dropped.
func directiveUnits(text string) []string {
	var out []string
	add := func(unit string) {
		unit = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(unit), "#-*+>0123456789.) \t"))
		if len(strings.Fields(unit)) >= 3 && len(out) < directiveUnitLimit {
			out = append(out, unit)
		}
	}
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		start := 0
		for i := 0; i < len(line); i++ {
			// End a sentence at . ! ? ; followed by a space, so "18:00",
			// "v1.2" and "e.g.x" stay whole.
			if strings.IndexByte(".!?;", line[i]) >= 0 && (i+1 == len(line) || line[i+1] == ' ') {
				add(line[start:i])
				start = i + 1
			}
		}
		add(line[start:])
	}
	return out
}

// directiveToolOrder ranks tools for a directive sentence by sentence. Every
// sentence's best match comes before any sentence's second-best. Within one
// rank a tool that a sentence names comes first, then the tool with the most
// support across sentences.
//
// A sentence names a tool when it mentions the tool's server and every word
// of the tool's own name: "alert the operator in Conversations" names
// conversations_alert.
func (t *Thinker) directiveToolOrder(directive string, normalize func(string) string, available []string, allowNoSpawn bool) []IndexEntry {
	if t == nil || t.toolIndex == nil {
		return nil
	}
	authorized := t.discoveryAuthorizedPredicate()
	usable := func(entry IndexEntry) bool {
		return (allowNoSpawn || !entry.NoSpawn) && (authorized == nil || authorized(entry)) && normalizeToolLoadMode(entry.LoadMode) != ToolLoadAlways
	}
	type ranked struct {
		entry IndexEntry
		named bool
		unit  int
	}
	var ranks [directiveUnitCandidates][]ranked
	for unit, sentence := range directiveUnits(directive) {
		words := map[string]bool{}
		for _, word := range discoveryQueryTokens(sentence) {
			words[word] = true
		}
		var list []ranked
		seen := map[string]bool{}
		for _, server := range available {
			if !containsServerReference(sentence, server) {
				continue
			}
			for _, name := range t.toolIndex.ToolsForServer(server) {
				entry, ok := t.toolIndex.Get(name)
				if !ok || !usable(entry) || seen[entry.Name] || !namedBy(entry, words) {
					continue
				}
				seen[entry.Name] = true
				entry.Match = "named"
				list = append(list, ranked{entry: entry, named: true, unit: unit})
			}
		}
		// A sentence that mentions two servers is searched in each. Take
		// their hits in turn so neither server's matches crowd out the other.
		var searches [][]IndexEntry
		for _, intent := range compileDiscoveryIntent(normalize(sentence), nil, DiscoveryAccessPreferRead, directiveUnitCandidates, "directive", 0, available) {
			found := t.toolIndex.searchDetailedWithOptions(intent.Query, intent.Limit, allowNoSpawn, authorized, indexSearchOptions{Servers: stringSetLower(intent.Servers), Access: intent.Access})
			searches = append(searches, found.Hits)
		}
		for position := 0; position < directiveUnitCandidates; position++ {
			for _, hits := range searches {
				if position >= len(hits) || seen[hits[position].Name] {
					continue
				}
				seen[hits[position].Name] = true
				list = append(list, ranked{entry: hits[position], unit: unit})
			}
		}
		for rank := 0; rank < len(list) && rank < directiveUnitCandidates; rank++ {
			ranks[rank] = append(ranks[rank], list[rank])
		}
	}
	var out []IndexEntry
	placed := map[string]bool{}
	for _, tier := range ranks {
		// A tool that several sentences point to outranks one strong match
		// from a single sentence, so add up its scores within the rank.
		named, weight := map[string]bool{}, map[string]float64{}
		for _, candidate := range tier {
			named[candidate.entry.Name] = named[candidate.entry.Name] || candidate.named
			weight[candidate.entry.Name] += candidate.entry.Score
		}
		sort.SliceStable(tier, func(i, j int) bool {
			a, b := tier[i].entry.Name, tier[j].entry.Name
			if named[a] != named[b] {
				return named[a]
			}
			if weight[a] != weight[b] {
				return weight[a] > weight[b]
			}
			return tier[i].unit < tier[j].unit
		})
		for _, candidate := range tier {
			if !placed[candidate.entry.Name] {
				placed[candidate.entry.Name] = true
				out = append(out, candidate.entry)
			}
		}
	}
	return out
}

// namedBy reports whether words hold every word of the tool's own name.
func namedBy(entry IndexEntry, words map[string]bool) bool {
	local := entry.LocalName
	if local == "" {
		local = entry.Name
	}
	parts := discoveryQueryTokens(local)
	if len(parts) == 0 {
		return false
	}
	for _, part := range parts {
		if !words[part] {
			return false
		}
	}
	return true
}

// automaticToolsNotLoaded lists the leading candidates that were considered
// and left out, each with its score and the reason, so a missing tool can be
// told apart from one that was never a candidate.
func automaticToolsNotLoaded(discovery DiscoveryResult) []map[string]any {
	var out []map[string]any
	seen := map[string]bool{}
	for _, name := range discovery.Loaded {
		seen[name] = true
	}
	for rank := 0; len(out) < automaticNotLoadedLimit; rank++ {
		more := false
		for _, group := range discovery.Results {
			if rank >= len(group.Candidates) {
				continue
			}
			more = true
			candidate := group.Candidates[rank]
			if candidate.Loaded || seen[candidate.Name] || len(out) >= automaticNotLoadedLimit {
				continue
			}
			seen[candidate.Name] = true
			reason := candidate.SkipReason
			if reason == "" {
				reason = "not_reached"
			}
			out = append(out, map[string]any{"name": candidate.Name, "source": group.Source, "rank": rank + 1, "score": candidate.Score, "match": candidate.Match, "schema_tokens_est": candidate.entry.SchemaCost, "reason": reason})
		}
		if !more {
			break
		}
	}
	return out
}
