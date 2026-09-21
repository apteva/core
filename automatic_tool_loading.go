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
	for _, q := range queries {
		priority := 70
		if q.Source == "memory" {
			priority = 20
		}
		intents = append(intents, compileDiscoveryIntent(normalize(q.Text), nil, DiscoveryAccessPreferRead, c.MaxTools, q.Source, priority, available)...)
	}
	if previous.context == contextKey {
		for _, name := range previous.names {
			reason := previous.reasons[name]
			priority := 60
			if reason == "workflow" {
				priority = 100
			}
			intents = append(intents, compileDiscoveryIntent(name, nil, DiscoveryAccessAny, 1, reason, priority, available)...)
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
	t.automaticTools = automaticToolSelection{policy: policy, context: contextKey, names: append([]string(nil), discovery.Loaded...), reasons: reasons}
	if t.telemetry != nil {
		t.telemetry.Emit("tool.autoload", t.threadID, map[string]any{"iteration": t.iteration, "selected": discovery.Loaded, "reasons": reasons, "schema_tokens_est": discovery.SchemaTokens, "max_tools": c.MaxTools, "max_schema_tokens": c.MaxSchemaTokens, "skipped_budget": discovery.SkippedBudget, "catalog_revision": discovery.CatalogRevision, "duration_us": time.Since(started).Microseconds(), "context_hash": contextKey})
	}
	return selected
}
