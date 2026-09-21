package core

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// ToolAccessClass is conservative discovery metadata, never an execution
// permission. Unknown tools are excluded from strict read-only discovery.
type ToolAccessClass string

const (
	ToolAccessRead    ToolAccessClass = "read"
	ToolAccessCreate  ToolAccessClass = "create"
	ToolAccessUpdate  ToolAccessClass = "update"
	ToolAccessDelete  ToolAccessClass = "delete"
	ToolAccessExecute ToolAccessClass = "execute"
	ToolAccessUnknown ToolAccessClass = "unknown"
)

type DiscoveryAccess string

const (
	DiscoveryAccessAny        DiscoveryAccess = "any"
	DiscoveryAccessPreferRead DiscoveryAccess = "prefer_read"
	DiscoveryAccessReadOnly   DiscoveryAccess = "read_only"
)

func normalizeDiscoveryAccess(access DiscoveryAccess) DiscoveryAccess {
	switch access {
	case DiscoveryAccessReadOnly, DiscoveryAccessPreferRead:
		return access
	default:
		return DiscoveryAccessAny
	}
}

func annotationBool(annotations map[string]any, key string) (bool, bool) {
	value, ok := annotations[key]
	if !ok {
		return false, false
	}
	switch value := value.(type) {
	case bool:
		return value, true
	case string:
		parsed, err := strconv.ParseBool(value)
		return parsed, err == nil
	default:
		return false, false
	}
}

func classifyToolAccess(tool mcpToolDef) ToolAccessClass {
	readOnly, hasReadOnly := annotationBool(tool.Annotations, "readOnlyHint")
	destructive, hasDestructive := annotationBool(tool.Annotations, "destructiveHint")
	if hasReadOnly && readOnly {
		if hasDestructive && destructive {
			return ToolAccessUnknown
		}
		return ToolAccessRead
	}
	if hasDestructive && destructive {
		return ToolAccessDelete
	}
	words := strings.FieldsFunc(strings.ToLower(tool.Name), func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	classes := []struct {
		class ToolAccessClass
		verbs map[string]bool
	}{
		{ToolAccessDelete, wordSet("delete", "remove", "archive")},
		{ToolAccessCreate, wordSet("create", "add", "post", "send", "upload", "publish")},
		{ToolAccessUpdate, wordSet("update", "set", "edit", "assign", "patch", "move")},
		{ToolAccessRead, wordSet("get", "list", "read", "search", "find", "query", "count", "sum", "top", "topics", "inspect", "lookup", "fetch", "view", "status", "download", "history")},
		{ToolAccessExecute, wordSet("run", "exec", "execute", "deploy", "trigger", "call")},
	}
	for _, candidate := range classes {
		for _, word := range words {
			if candidate.verbs[word] {
				if hasReadOnly && !readOnly && candidate.class == ToolAccessRead {
					return ToolAccessUnknown
				}
				return candidate.class
			}
		}
	}
	return ToolAccessUnknown
}

func wordSet(words ...string) map[string]bool {
	out := make(map[string]bool, len(words))
	for _, word := range words {
		out[word] = true
	}
	return out
}

type DiscoveryIntent struct {
	Query    string
	Servers  []string
	Access   DiscoveryAccess
	Limit    int
	Source   string
	Priority int
}

type DiscoveryRequest struct {
	Intents               []DiscoveryIntent
	AllowNoSpawn          bool
	AllowSuggestions      bool
	MaxTools              int
	MaxSchemaTokens       int
	MemoryMaxTools        int
	MemoryMaxSchemaTokens int
	Activate              bool
}

type DiscoveryToolResult struct {
	Name       string          `json:"name"`
	Server     string          `json:"server"`
	Access     ToolAccessClass `json:"access"`
	Match      string          `json:"match,omitempty"`
	Score      float64         `json:"score,omitempty"`
	Summary    string          `json:"summary,omitempty"`
	Loaded     bool            `json:"loaded"`
	SkipReason string          `json:"skip_reason,omitempty"`
	Suggestion bool            `json:"suggestion,omitempty"`
	entry      IndexEntry
}

type DiscoveryIntentResult struct {
	Query              string                `json:"query"`
	Source             string                `json:"source,omitempty"`
	Access             DiscoveryAccess       `json:"access"`
	RequestedServers   []string              `json:"requested_servers,omitempty"`
	RepresentedServers []string              `json:"represented_servers,omitempty"`
	MissingServers     []string              `json:"missing_servers,omitempty"`
	AvailableServers   []string              `json:"available_servers,omitempty"`
	Coverage           string                `json:"coverage"`
	Candidates         []DiscoveryToolResult `json:"candidates"`
	Unresolved         []string              `json:"unresolved_names,omitempty"`
	Ambiguous          map[string][]string   `json:"ambiguous_names,omitempty"`
	priority           int
	unavailableExact   bool
}

type DiscoveryResult struct {
	CatalogRevision uint64                  `json:"catalog_revision"`
	Results         []DiscoveryIntentResult `json:"results"`
	Loaded          []string                `json:"loaded"`
	SchemaTokens    int                     `json:"schema_tokens_est"`
	SkippedBudget   int                     `json:"skipped_budget"`
}

func (t *Thinker) authorizedDiscoveryServers(allowNoSpawn bool) []string {
	if t == nil || t.toolIndex == nil {
		return nil
	}
	var visible []string
	for _, server := range t.toolIndex.Servers() {
		for _, name := range t.toolIndex.ToolsForServer(server) {
			entry, ok := t.toolIndex.Get(name)
			if ok && (allowNoSpawn || !entry.NoSpawn) && t.toolAuthorized(name) {
				visible = append(visible, server)
				break
			}
		}
	}
	return visible
}

func stringSetLower(values []string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		if value = strings.ToLower(strings.TrimSpace(value)); value != "" {
			out[value] = true
		}
	}
	return out
}

func sortedSet(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func containsServerReference(query, server string) bool {
	query, server = strings.ToLower(query), strings.ToLower(server)
	if strings.Contains(query, server+"_") {
		return true
	}
	for _, token := range strings.FieldsFunc(query, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_')
	}) {
		if token == server {
			return true
		}
	}
	return false
}

// compileDiscoveryIntent splits a compound request by named authorized server.
// This is deterministic and local; it adds no model or embedding round trip.
func compileDiscoveryIntent(query string, servers []string, access DiscoveryAccess, limit int, source string, priority int, available []string) []DiscoveryIntent {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}
	requested := stringSetLower(servers)
	if len(requested) == 0 {
		for _, server := range available {
			if containsServerReference(query, server) {
				requested[strings.ToLower(server)] = true
			}
		}
	}
	if limit <= 0 {
		limit = 5
	}
	base := DiscoveryIntent{Query: query, Access: normalizeDiscoveryAccess(access), Limit: limit, Source: source, Priority: priority}
	if len(requested) == 0 {
		return []DiscoveryIntent{base}
	}
	var out []DiscoveryIntent
	for _, server := range sortedSet(requested) {
		intent := base
		intent.Servers = []string{server}
		out = append(out, intent)
	}
	return out
}

func (t *Thinker) discoveryAuthorizedPredicate() func(string) bool {
	if t == nil || t.toolAllowlist == nil {
		return nil
	}
	return func(name string) bool { return t.toolAuthorized(name) }
}

func (t *Thinker) discoverTools(request DiscoveryRequest) DiscoveryResult {
	var result DiscoveryResult
	for attempt := 0; attempt < 3; attempt++ {
		result = t.discoverToolsOnce(request)
		if t == nil || t.toolIndex == nil || t.toolIndex.Revision() == result.CatalogRevision {
			break
		}
	}
	if request.Activate {
		for _, name := range result.Loaded {
			t.touchActiveTool(name)
		}
	}
	return result
}

// discoverToolsOnce uses revision checks around the whole multi-intent pass.
// A concurrent reconnect retries from a fresh catalog rather than returning a
// response assembled from different server generations.
func (t *Thinker) discoverToolsOnce(request DiscoveryRequest) DiscoveryResult {
	result := DiscoveryResult{}
	if t == nil || t.toolIndex == nil {
		return result
	}
	result.CatalogRevision = t.toolIndex.Revision()
	available := t.authorizedDiscoveryServers(request.AllowNoSpawn)
	availableSet := stringSetLower(available)
	authorized := t.discoveryAuthorizedPredicate()
	for _, intent := range request.Intents {
		intent.Access = normalizeDiscoveryAccess(intent.Access)
		group := DiscoveryIntentResult{Query: intent.Query, Source: intent.Source, Access: intent.Access, RequestedServers: append([]string(nil), intent.Servers...), AvailableServers: append([]string(nil), available...), priority: intent.Priority}
		serverScope := map[string]bool{}
		for _, server := range intent.Servers {
			server = strings.ToLower(strings.TrimSpace(server))
			if availableSet[server] {
				serverScope[server] = true
			}
		}
		var found indexSearchResult
		if len(intent.Servers) == 0 || len(serverScope) > 0 {
			found = t.toolIndex.searchDetailedWithOptions(intent.Query, intent.Limit, request.AllowNoSpawn, authorized, indexSearchOptions{Servers: serverScope, Access: intent.Access})
		}
		group.Unresolved, group.Ambiguous = found.Unresolved, found.Ambiguous
		for _, unresolved := range found.Unresolved {
			if entry, ok := t.toolIndex.Get(unresolved); ok {
				outsideServerScope := len(serverScope) > 0 && !serverScope[strings.ToLower(entry.Server)]
				if !t.toolAuthorized(entry.Name) || outsideServerScope || intent.Access == DiscoveryAccessReadOnly && entry.Access != ToolAccessRead {
					group.unavailableExact = true
				}
			}
		}
		appendCandidates := func(entries []IndexEntry, suggestion bool) {
			for _, entry := range entries {
				if suggestion && !request.AllowSuggestions {
					continue
				}
				summary := entry.Description
				if len(summary) > 240 {
					summary = summary[:237] + "..."
				}
				group.Candidates = append(group.Candidates, DiscoveryToolResult{Name: entry.Name, Server: entry.Server, Access: entry.Access, Match: entry.Match, Score: entry.Score, Summary: summary, Suggestion: suggestion, entry: entry})
			}
		}
		appendCandidates(found.Hits, false)
		appendCandidates(found.Suggestions, true)
		result.Results = append(result.Results, group)
	}
	result.plan(t, request)
	return result
}

func (result *DiscoveryResult) plan(t *Thinker, request DiscoveryRequest) {
	if request.MaxTools <= 0 {
		request.MaxTools = 20
	}
	if request.MaxSchemaTokens <= 0 {
		request.MaxSchemaTokens = 16000
	}
	selected := map[string]bool{}
	memorySelected := 0
	memoryTokens := 0
	priorities := map[int]bool{}
	for _, group := range result.Results {
		priorities[group.priority] = true
	}
	var orderedPriorities []int
	for priority := range priorities {
		orderedPriorities = append(orderedPriorities, priority)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(orderedPriorities)))
	for _, priority := range orderedPriorities {
		maxRank := 0
		for _, group := range result.Results {
			if group.priority == priority && len(group.Candidates) > maxRank {
				maxRank = len(group.Candidates)
			}
		}
		for rank := 0; rank < maxRank; rank++ {
			for groupIndex := range result.Results {
				group := &result.Results[groupIndex]
				if group.priority != priority || rank >= len(group.Candidates) {
					continue
				}
				candidate := &group.Candidates[rank]
				if selected[candidate.Name] {
					candidate.Loaded = true
					continue
				}
				entry, ok := t.toolIndex.Get(candidate.Name)
				if !ok || !t.toolAuthorized(candidate.Name) || (!request.AllowNoSpawn && entry.NoSpawn) {
					candidate.SkipReason = "unavailable_in_scope"
					continue
				}
				if t.registry != nil {
					def := t.registry.Get(candidate.Name)
					if def == nil || !def.MCP || def.SystemOnly && !t.systemThread {
						candidate.SkipReason = "unavailable_in_scope"
						continue
					}
				}
				if group.Source == "memory" && request.MemoryMaxTools > 0 && memorySelected >= request.MemoryMaxTools {
					candidate.SkipReason = "memory_budget"
					continue
				}
				if group.Source == "memory" && request.MemoryMaxSchemaTokens > 0 && memoryTokens+entry.SchemaCost > request.MemoryMaxSchemaTokens {
					candidate.SkipReason = "memory_budget"
					continue
				}
				if len(selected) >= request.MaxTools || result.SchemaTokens+entry.SchemaCost > request.MaxSchemaTokens {
					candidate.SkipReason = "schema_budget"
					result.SkippedBudget++
					continue
				}
				selected[candidate.Name], candidate.Loaded = true, true
				result.Loaded = append(result.Loaded, candidate.Name)
				result.SchemaTokens += entry.SchemaCost
				if group.Source == "memory" {
					memorySelected++
					memoryTokens += entry.SchemaCost
				}
			}
		}
	}
	for groupIndex := range result.Results {
		group := &result.Results[groupIndex]
		represented := map[string]bool{}
		loaded := false
		for _, candidate := range group.Candidates {
			represented[strings.ToLower(candidate.Server)] = true
			loaded = loaded || candidate.Loaded
		}
		group.RepresentedServers = sortedSet(represented)
		for _, server := range group.RequestedServers {
			if !represented[strings.ToLower(server)] {
				group.MissingServers = append(group.MissingServers, server)
			}
		}
		sort.Strings(group.MissingServers)
		switch {
		case group.unavailableExact || len(group.RequestedServers) > 0 && len(stringSetLower(group.RequestedServers)) > 0 && len(intersectStrings(group.RequestedServers, group.AvailableServers)) == 0:
			group.Coverage = "unavailable_in_scope"
		case len(group.Candidates) == 0:
			group.Coverage = "lexical_miss"
		case !loaded:
			group.Coverage = "schema_budget_exhausted"
		case len(group.MissingServers) > 0:
			group.Coverage = "partial"
		default:
			group.Coverage = "satisfied"
		}
	}
}

func intersectStrings(left, right []string) []string {
	rightSet := stringSetLower(right)
	var out []string
	for _, value := range left {
		if rightSet[strings.ToLower(value)] {
			out = append(out, value)
		}
	}
	return out
}

type discoveryQueryInput struct {
	Query       string          `json:"query"`
	Servers     []string        `json:"servers"`
	ServerNames []string        `json:"server_names"`
	Access      DiscoveryAccess `json:"access"`
	Limit       int             `json:"limit"`
}

func parseStringList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var values []string
	if json.Unmarshal([]byte(raw), &values) == nil {
		return values
	}
	for _, value := range strings.Split(raw, ",") {
		if value = strings.TrimSpace(value); value != "" {
			values = append(values, value)
		}
	}
	return values
}

func parseDiscoveryQueries(raw string) ([]discoveryQueryInput, error) {
	var values []json.RawMessage
	if err := json.Unmarshal([]byte(raw), &values); err != nil {
		return nil, err
	}
	out := make([]discoveryQueryInput, 0, len(values))
	for _, value := range values {
		var text string
		if json.Unmarshal(value, &text) == nil {
			out = append(out, discoveryQueryInput{Query: text})
			continue
		}
		var input discoveryQueryInput
		if err := json.Unmarshal(value, &input); err != nil {
			return nil, err
		}
		input.Servers = append(input.Servers, input.ServerNames...)
		out = append(out, input)
	}
	return out, nil
}
