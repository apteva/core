package core

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// search_tools is the meta-tool every thread has in its scaffolding.
// It searches the process-wide ToolIndex for MCP tools matching the
// query, activates the top k matches for this thread (their full
// schemas appear in the next provider call's tool list), and returns
// a compact JSON summary the LLM can read.
//
// Activation is sticky for the rest of the thread's life (subject to
// context-window compaction): once the LLM has called a tool we keep
// its schema visible so a follow-up call has no search overhead. The
// per-turn BM25 preload (see thinker.go iteration loop) handles the
// transient case — tools the user's current message is about that
// the LLM hasn't yet searched for.

// registerSearchTool adds search_tools to a registry as a Core tool
// so it appears in every thread's tool list regardless of allowlist.
// Called from registerDefaults so main and every sub-thread share
// the same wired-in meta-tool.
func registerSearchTool(r *ToolRegistry) {
	r.Register(&ToolDef{
		Name: "search_tools",
		Description: "Search for MCP tools by keyword and load their schemas into your context. " +
			"Use when you need a capability you don't currently have visible — file upload, " +
			"posting to a channel, fetching from an integration, etc. Returns up to k matches " +
			"with name + summary; their full schemas become available for you to call on the " +
			"next turn. Call multiple search_tools in parallel if you need several capabilities.",
		Syntax: `[[search_tools query="upload file" k="5"]]`,
		Rules: `query is required; k defaults to 5 and caps at 20. Loaded tools persist for the rest of this thread's conversation (subject to compaction). Schemas appear on the next thinking turn — you cannot call a discovered tool in the same turn you searched for it.`,
		Core:   true,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{
					"type":        "string",
					"description": "Keywords describing the capability you need.",
				},
				"k": map[string]any{
					"type":        "integer",
					"description": "Maximum number of tools to load (default 5, max 20).",
				},
			},
			"required": []string{"query"},
		},
		// Handler is intentionally nil — search_tools runs as an inline
		// tool through main/threadToolHandler because it mutates the
		// Thinker's activeTools set, which a registry Handler closure
		// can't reach without smuggling state through globals.
	})
}

// searchToolsResult is the JSON shape returned to the LLM. Compact
// on purpose: each hit's full schema lands in the tool list next
// turn, so the LLM doesn't need a paragraph of description here —
// just enough to confirm it found what it was looking for.
type searchToolsResult struct {
	Query  string                `json:"query"`
	Hits   []searchToolHit       `json:"hits"`
	Loaded []string              `json:"loaded"` // names whose schemas are now in context
	Note   string                `json:"note,omitempty"`
}

type searchToolHit struct {
	Name    string `json:"name"`
	Server  string `json:"server"`
	Summary string `json:"summary"`
}

// computePreloadTools runs a BM25 search against the most recent
// user turn's text and returns up to k tool names whose schemas
// should be transiently surfaced this turn. Sub-threads pass
// allowNoSpawn=false implicitly (their thread-id is not "main");
// main passes allowNoSpawn=true.
//
// Returns nil silently when there's no index, no last user text, or
// no matches — preload is best-effort context biasing, never a
// hard requirement.
func (t *Thinker) computePreloadTools(k int) []string {
	if t == nil || t.toolIndex == nil {
		return nil
	}
	text := t.lastUserText()
	if text == "" {
		return nil
	}
	allowNoSpawn := t.threadID == "main"
	hits := t.toolIndex.Search(text, k, allowNoSpawn)
	if len(hits) == 0 {
		return nil
	}
	out := make([]string, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.Name)
	}
	return out
}

// lastUserText returns the most recent user-role message's text, or
// empty string if no user message exists yet. The first user turn
// after a system message is what bootstraps the preload signal; on
// subsequent iterations the most recent tool_result-bearing user
// message also feeds in (it's the running record of the work).
func (t *Thinker) lastUserText() string {
	if t == nil {
		return ""
	}
	for i := len(t.messages) - 1; i >= 0; i-- {
		m := t.messages[i]
		if m.Role != "user" {
			continue
		}
		if m.Content != "" {
			return m.Content
		}
		// tool_result-only user turns have empty Content; we want the
		// user's actual prose. Walk further back.
	}
	return ""
}

// runSearchTools executes the search and mutates t.activeTools. Used
// by both mainToolHandler and threadToolHandler. allowNoSpawn is
// false for sub-threads (they cannot see no_spawn-flagged servers)
// and true for main.
func runSearchTools(t *Thinker, args map[string]string, allowNoSpawn bool) string {
	query := args["query"]
	if query == "" {
		return `{"error":"query is required"}`
	}
	k := 5
	if raw, ok := args["k"]; ok && raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			if n > 20 {
				n = 20
			}
			k = n
		}
	}
	if t.toolIndex == nil {
		return `{"error":"tool index not initialised — no MCPs attached"}`
	}
	hits := t.toolIndex.Search(query, k, allowNoSpawn)
	if t.activeTools == nil {
		t.activeTools = map[string]bool{}
	}
	res := searchToolsResult{Query: query}
	for _, h := range hits {
		t.activeTools[h.Name] = true
		summary := h.Description
		if len(summary) > 240 {
			summary = summary[:237] + "..."
		}
		res.Hits = append(res.Hits, searchToolHit{
			Name: h.Name, Server: h.Server, Summary: summary,
		})
		res.Loaded = append(res.Loaded, h.Name)
	}
	if len(res.Hits) == 0 {
		// Tell the LLM what *is* attached so it can refine the query or
		// reach for the gateway's install/list_apps tool to add what's
		// missing. Cheaper than another search round-trip.
		servers := t.toolIndex.Servers()
		var visible []string
		for _, s := range servers {
			// Reuse Search to honour the no_spawn filter — we don't want
			// a sub-thread learning that `apteva-server` exists.
			if hits := t.toolIndex.Search(s, 1, allowNoSpawn); len(hits) > 0 {
				visible = append(visible, s)
			}
		}
		if len(visible) > 0 {
			res.Note = fmt.Sprintf("no matches; attached servers: %v", visible)
		} else {
			res.Note = "no matches; no MCP servers visible to this thread"
		}
	}
	out, _ := json.Marshal(res)
	return string(out)
}
