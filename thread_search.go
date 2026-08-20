package core

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// list_threads answers the queries the [ACTIVE THREADS] digest can no longer
// answer inline once a fleet outgrows the roster budget — above all "does a
// thread already own this work?", which both system prompts require before any
// spawn.
//
// Unlike the roster block, a tool result lands in DURABLE history: it is
// prefix-stable and therefore cacheable, but it also persists. Sizing is
// bounded on both axes below.
const (
	// listThreadsDefaultLimit is one comfortable screenful.
	listThreadsDefaultLimit = 25
	// listThreadsMaxLimit is stated in the tool's Rules so the model does not
	// request 500 and silently receive a degraded answer.
	listThreadsMaxLimit = 50
	// listThreadsMaxChars keeps a result under largeToolResultThresholdBytes
	// (16 KB). At or above that, shouldArchiveToolResult SHA-archives the
	// result and prepareHistoricalToolResults projects it down to a bounded
	// preview after toolResultFullRetentionCalls — the roster would silently
	// truncate mid-task. Headroom covers the header and truncation notice.
	listThreadsMaxChars = 14 << 10
)

// listThreadsQuery is the parsed, validated form of the tool's arguments.
type listThreadsQuery struct {
	filter string
	scope  string // "tree" (default, includes descendants) or "children"
	limit  int
	offset int
}

// parseListThreadsQuery is lenient about how the model spells things: limit
// and offset fall back to defaults rather than erroring, since a rejected call
// costs a whole turn and the intent is unambiguous.
func parseListThreadsQuery(args map[string]string) listThreadsQuery {
	q := listThreadsQuery{
		filter: strings.TrimSpace(args["filter"]),
		scope:  strings.ToLower(strings.TrimSpace(args["scope"])),
		limit:  listThreadsDefaultLimit,
	}
	if q.scope != "children" {
		q.scope = "tree"
	}
	if v := strings.TrimSpace(args["limit"]); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.limit = n
		}
	}
	if q.limit > listThreadsMaxLimit {
		q.limit = listThreadsMaxLimit
	}
	if v := strings.TrimSpace(args["offset"]); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.offset = n
		}
	}
	return q
}

// threadSearchScore searches the fields an operator would actually use.
// Exact phrases rank first. Otherwise a natural multi-keyword query matches
// any whitespace/punctuation-separated term and ranks by overlap. Models
// routinely emit `billing invoice acme` rather than one literal substring;
// requiring that whole phrase creates false negatives and duplicate owners.
func threadSearchScore(t ThreadInfo, filter string) int {
	if filter == "" {
		return 1
	}
	var fields []string
	fields = append(fields, t.ID, t.Name, t.Directive)
	fields = append(fields, t.Tools...)
	fields = append(fields, t.MCPNames...)
	haystack := strings.ToLower(strings.Join(fields, "\n"))
	needle := strings.ToLower(strings.TrimSpace(filter))
	terms := strings.FieldsFunc(needle, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-'
	})
	if strings.Contains(haystack, needle) {
		return 1000 + len(terms)
	}
	seen := make(map[string]bool, len(terms))
	score := 0
	for _, term := range terms {
		term = strings.TrimSpace(term)
		if term == "" || seen[term] {
			continue
		}
		seen[term] = true
		if strings.Contains(haystack, term) {
			score++
		}
	}
	return score
}

func threadMatchesFilter(t ThreadInfo, filter string) bool {
	return threadSearchScore(t, filter) > 0
}

// listThreadsEntry renders one search hit. It carries the routing fields the
// roster block omits (parent, depth, pending wake) because a search result is
// often the model's only view of a thread it did not spawn itself.
func listThreadsEntry(t ThreadInfo) string {
	label := t.ID
	if t.Name != "" && t.Name != t.ID {
		label = fmt.Sprintf("%s (%s)", t.Name, t.ID)
	}
	meta := fmt.Sprintf("parent=%s depth=%d", t.ParentID, t.Depth)
	if t.SubThreads > 0 {
		meta += fmt.Sprintf(" sub-threads=%d", t.SubThreads)
	}
	if t.Realtime {
		meta += " realtime"
	}
	if t.NextWakeAt.IsZero() {
		meta += " wake=events-only"
	} else {
		meta += " wake=" + t.NextWakeAt.UTC().Format("2006-01-02T15:04Z")
	}
	mcpLine := ""
	if len(t.MCPNames) > 0 {
		mcpLine = "\n  mcp scopes: " + strings.Join(t.MCPNames, ", ")
	}
	return fmt.Sprintf("- %s %s\n  directive: %s\n  tools: %s%s\n",
		label, meta, truncateStr(t.Directive, 150), strings.Join(t.Tools, ", "), mcpLine)
}

// runListThreads executes a parsed query against a manager and renders the
// result. tm may be nil (a thread with no children), which is reported as an
// ordinary empty result rather than an error — "I lead nobody" is a valid,
// useful answer to "does an owner exist?".
func runListThreads(tm *ThreadManager, args map[string]string) string {
	q := parseListThreadsQuery(args)
	if tm == nil {
		return "0 threads: this thread has no sub-threads."
	}

	var all []ThreadInfo
	if q.scope == "children" {
		all = tm.ListAgentVisible()
	} else {
		all = tm.ListTreeAgentVisible()
	}

	type scoredThread struct {
		info  ThreadInfo
		score int
	}
	matched := make([]scoredThread, 0, len(all))
	for _, t := range all {
		if score := threadSearchScore(t, q.filter); score > 0 {
			matched = append(matched, scoredThread{info: t, score: score})
		}
	}
	// ListTreeAgentVisible supplies id order. Preserve that for an unfiltered
	// listing; for a query, rank stronger matches first and use id as a stable
	// tie-breaker so pagination remains deterministic.
	if q.filter != "" {
		sort.SliceStable(matched, func(i, j int) bool {
			if matched[i].score != matched[j].score {
				return matched[i].score > matched[j].score
			}
			return matched[i].info.ID < matched[j].info.ID
		})
	}

	scopeNote := "including all descendants"
	if q.scope == "children" {
		scopeNote = "direct children only"
	}
	if len(matched) == 0 {
		if q.filter == "" {
			return fmt.Sprintf("0 threads (%s). No sub-threads exist.", scopeNote)
		}
		return fmt.Sprintf("0 of %d threads textually match %q (%s). This filtered result does not prove that no related owner exists; broaden the filter or list all before spawning when ownership is still uncertain.",
			len(all), q.filter, scopeNote)
	}

	start := q.offset
	if start > len(matched) {
		start = len(matched)
	}
	end := start + q.limit
	if end > len(matched) {
		end = len(matched)
	}
	page := matched[start:end]

	var sb strings.Builder
	if q.filter == "" {
		sb.WriteString(fmt.Sprintf("%d threads (%s), showing %d-%d.\n", len(matched), scopeNote, start+1, end))
	} else {
		sb.WriteString(fmt.Sprintf("%d of %d threads match %q (%s), showing %d-%d.\n",
			len(matched), len(all), q.filter, scopeNote, start+1, end))
	}

	// Enforce the byte budget on rendered output, not on the entry count.
	// Entry width varies by an order of magnitude with the tool list, so a
	// count cap alone cannot keep the result under the archive threshold.
	rendered := 0
	for i, match := range page {
		entry := listThreadsEntry(match.info)
		if sb.Len()+len(entry) > listThreadsMaxChars {
			sb.WriteString(fmt.Sprintf("… output truncated at %d of %d shown (size limit). Narrow with filter= or page with offset=%d.\n",
				rendered, len(page), start+i))
			break
		}
		sb.WriteString(entry)
		rendered++
	}

	if end < len(matched) {
		sb.WriteString(fmt.Sprintf("%d more match. Page with offset=%d.\n", len(matched)-end, end))
	}
	return sb.String()
}
