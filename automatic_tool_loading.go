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
		out.MaxSchemaTokens = 2000
	}
	return out
}

type automaticToolSelection struct {
	policy  string
	context string
	names   []string
	reasons map[string]string
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
	if c.IncludeMemory {
		add("memory", t.memoryRecall.context)
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
	rawContext, _ := json.Marshal(queries)
	policy, contextKey := string(rawPolicy), promptCacheShortHash(rawContext)
	previous := t.automaticTools
	if previous.policy != policy {
		previous = automaticToolSelection{}
	}
	selected := map[string]bool{}
	reasons := map[string]string{}
	var order []string
	tokens, skippedBudget := 0, 0
	add := func(name, reason string) {
		if selected[name] {
			return
		}
		entry, ok := t.toolIndex.Get(name)
		if !ok || (!(t.threadID == "main" || t.allowNoSpawn) && entry.NoSpawn) || !t.toolAuthorized(name) {
			return
		}
		def := t.registry.Get(name)
		if def == nil || !def.MCP || (def.SystemOnly && !t.systemThread) {
			return
		}
		// Measure the actual cached native schema, including Core's injected fields.
		raw, err := json.Marshal(def.native)
		if err != nil {
			return
		}
		cost := (len(raw) + 3) / 4
		if len(order) >= c.MaxTools || tokens+cost > c.MaxSchemaTokens {
			skippedBudget++
			return
		}
		selected[name], reasons[name] = true, reason
		order = append(order, name)
		tokens += cost
	}
	for _, name := range c.WorkflowTools {
		add(name, "workflow")
	}
	if previous.context == contextKey {
		for _, name := range previous.names {
			add(name, previous.reasons[name])
		}
	}
	// Round-robin sources so a long directive cannot consume all slots before
	// current task/memory relevance is considered. Search only authorized hits;
	// unresolved-name suggestions never become silent automatic replacements.
	normalize := t.automaticQueryNormalizer()
	ranked := make([][]IndexEntry, len(queries))
	allowNoSpawn := t.threadID == "main" || t.allowNoSpawn
	for i, q := range queries {
		ranked[i] = t.searchAuthorizedTools(normalize(q.Text), c.MaxTools, allowNoSpawn)
	}
	for rank := 0; rank < c.MaxTools; rank++ {
		for i, q := range queries {
			if rank < len(ranked[i]) {
				add(ranked[i][rank].Name, q.Source)
			}
		}
	}
	for _, name := range previous.names {
		add(name, "retained")
	}
	t.automaticTools = automaticToolSelection{policy: policy, context: contextKey, names: order, reasons: reasons}
	if t.telemetry != nil {
		t.telemetry.Emit("tool.autoload", t.threadID, map[string]any{"iteration": t.iteration, "selected": order, "reasons": reasons, "schema_tokens_est": tokens, "max_tools": c.MaxTools, "max_schema_tokens": c.MaxSchemaTokens, "skipped_budget": skippedBudget, "duration_us": time.Since(started).Microseconds(), "context_hash": contextKey})
	}
	return selected
}
