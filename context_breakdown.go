package core

import (
	"encoding/json"
	"strings"
)

// buildComposition is a read-only instrumentation pass over a thread's
// current messages + live registry state. It produces a size breakdown
// of everything the LLM actually sees on a call:
//
//  1. System text (messages[0].Content) split into the sections that
//     buildSystemPrompt emits. No double-pass required — we just
//     look for the known "[SECTION]" markers in the text that was
//     already built.
//
//  2. Native tool schemas (registry.NativeTools). Each tool's name +
//     description + JSON-serialized Parameters contributes to the
//     provider's `tools[]` payload; we estimate its on-wire size by
//     marshalling it.
//
//  3. The separate "[memories]" system message and any other
//     role=system messages appended after the main prompt (they all
//     travel as prompt tokens too).
//
//  4. All remaining user / assistant / tool messages rolled up so the
//     user sees grand_total = everything sent on the next call.
//
// Pure read, pure compute. Nothing on the hot path calls this.
type PromptComposition struct {
	System         SystemBreakdown    `json:"system"`
	NativeTools    []NativeToolSize   `json:"native_tools"`
	NativeBytes    int                `json:"native_bytes"`
	ExtraSystem    []ExtraSystemBlock `json:"extra_system"` // role=system msgs after messages[0]
	ExtraBytes     int                `json:"extra_bytes"`
	ConvBytes      int                `json:"conv_bytes"` // user/assistant/tool/other messages
	GrandTotal     int                `json:"grand_total"`
	ModelMaxTokens int                `json:"model_max_tokens,omitempty"`
}

// SystemBreakdown is the per-section byte count of messages[0].Content,
// i.e. the main system prompt. Unknown text that falls between known
// markers lands in Other so the total always reconciles with Total.
type SystemBreakdown struct {
	Base            int `json:"base"`              // up to "CORE TOOLS — always available:"
	CoreTools       int `json:"core_tools"`        // "CORE TOOLS — always available:" to next [ marker
	RetrievedTools  int `json:"retrieved_tools"`   // [available tools — matched to your current context] block (RAG per-turn)
	MCPServers      int `json:"mcp_servers"`       // [AVAILABLE MCP SERVERS] block
	MCPToolDocs     int `json:"mcp_tool_docs"`     // [MCP TOOLS — available for sub-threads] block
	Providers       int `json:"providers"`         // [AVAILABLE PROVIDERS] block
	ActiveThreads   int `json:"active_threads"`    // [ACTIVE THREADS] block
	SafetyMode      int `json:"safety_mode"`       // [SAFETY MODE: ...] block
	Skills          int `json:"skills"`            // [LEARNED SKILLS] block
	BlobHint        int `json:"blob_hint"`         // [FILE HANDLES] block
	PreviousContext int `json:"previous_context"`  // [PREVIOUS CONTEXT] block (from session load)
	Directive       int `json:"directive"`         // [DIRECTIVE — EXECUTE ON STARTUP] block
	Other           int `json:"other"`             // text not matching any known marker
	Total           int `json:"total"`
}

// NativeToolSize describes one entry in the tools[] payload sent to
// the LLM. Kind separates core loop tools from local / MCP-main-access
// tools so the user can see which flavor is burning bytes.
type NativeToolSize struct {
	Name  string `json:"name"`
	Kind  string `json:"kind"`  // "core" | "mcp" | "local"
	Bytes int    `json:"bytes"`
}

// ExtraSystemBlock describes a role=system message that sits AFTER
// messages[0] — the [memories] block appended each iteration is the
// canonical example, but anything the thinker injects ends up here.
type ExtraSystemBlock struct {
	Preview string `json:"preview"` // first ~60 chars, newlines stripped
	Bytes   int    `json:"bytes"`
}

// sectionMarkers are the bracketed headers that buildSystemPrompt
// emits. They're matched with a "\n\n" prefix so the lookup hits the
// real section header and not a prose reference (baseSystemPrompt's
// text mentions "[ACTIVE THREADS]" and "[AVAILABLE PROVIDERS]" inline
// as cross-references; without the \n\n requirement those would be
// misattributed and wreck the breakdown).
var sectionMarkers = []struct {
	marker string
	label  string
}{
	{"\n\n[available tools — matched to your current context]", "retrieved_tools"},
	{"\n\n[AVAILABLE MCP SERVERS]", "mcp_servers"},
	{"\n[MCP TOOLS — available for sub-threads]", "mcp_tool_docs"},
	{"\n\n[AVAILABLE PROVIDERS]", "providers"},
	{"\n\n[ACTIVE THREADS]", "active_threads"},
	{"\n\n[SAFETY MODE:", "safety_mode"},
	{"\n\n[LEARNED SKILLS]", "skills"},
	{"\n\n[FILE HANDLES]", "blob_hint"},
	{"\n\n[PREVIOUS CONTEXT]", "previous_context"},
	{"\n\n[DIRECTIVE — EXECUTE ON STARTUP]", "directive"},
}

// coreToolsMarker separates the base preamble from the tool listing
// that CoreDocs appends. It's inside the "base" region as written by
// buildSystemPrompt so we handle it specially.
const coreToolsMarker = "CORE TOOLS — always available:"

func buildComposition(t *Thinker, msgs []Message) PromptComposition {
	// Non-nil slices so JSON marshals to [] not null — the dashboard
	// uses .length / .map on these and blows up on null.
	comp := PromptComposition{
		NativeTools: []NativeToolSize{},
		ExtraSystem: []ExtraSystemBlock{},
	}
	if t == nil {
		return comp
	}
	if len(msgs) > 0 && msgs[0].Role == "system" {
		comp.System = breakdownSystem(msgs[0].Content)
	}

	// Native tools — same code path the provider uses on the next call.
	if t.registry != nil && t.provider != nil && t.provider.SupportsNativeTools() {
		nativeTools := t.registry.NativeTools(t.toolAllowlist, t.activeTools)
		for _, nt := range nativeTools {
			size := nativeToolSize(nt)
			kind := classifyToolKind(t, nt.Name)
			comp.NativeTools = append(comp.NativeTools, NativeToolSize{
				Name:  nt.Name,
				Kind:  kind,
				Bytes: size,
			})
			comp.NativeBytes += size
		}
	}

	// Extra system messages (appended per iteration — [memories] etc.)
	// and all remaining non-system messages rolled up.
	for i := 1; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role == "system" {
			comp.ExtraSystem = append(comp.ExtraSystem, ExtraSystemBlock{
				Preview: truncateForPreview(m.Content, 60),
				Bytes:   messageBytes(m),
			})
			comp.ExtraBytes += messageBytes(m)
			continue
		}
		comp.ConvBytes += messageBytes(m)
	}

	comp.GrandTotal = comp.System.Total + comp.NativeBytes + comp.ExtraBytes + comp.ConvBytes

	return comp
}

// breakdownSystem splits the system prompt into sections by locating
// each known [MARKER] header. The region before the first marker is
// the base (+ core tools). Any content we can't attribute lands in
// Other so Total always reconciles.
func breakdownSystem(text string) SystemBreakdown {
	out := SystemBreakdown{Total: len(text)}
	if text == "" {
		return out
	}

	// Locate every known marker in the string.
	type hit struct {
		start int
		label string
	}
	var hits []hit
	for _, sm := range sectionMarkers {
		idx := strings.Index(text, sm.marker)
		if idx >= 0 {
			hits = append(hits, hit{start: idx, label: sm.label})
		}
	}
	// Sort by position.
	for i := 1; i < len(hits); i++ {
		for j := i; j > 0 && hits[j].start < hits[j-1].start; j-- {
			hits[j], hits[j-1] = hits[j-1], hits[j]
		}
	}

	// Base region = start to first marker (or the whole string if none).
	baseEnd := len(text)
	if len(hits) > 0 {
		baseEnd = hits[0].start
	}
	baseRegion := text[:baseEnd]
	// Split base into (preamble) + (core tool docs). CoreDocs is
	// appended right after baseSystemPrompt, so the marker is inside
	// the base region.
	if idx := strings.Index(baseRegion, coreToolsMarker); idx >= 0 {
		out.Base = idx
		out.CoreTools = len(baseRegion) - idx
	} else {
		out.Base = len(baseRegion)
	}

	// Slice between each hit and the next.
	for i, h := range hits {
		end := len(text)
		if i+1 < len(hits) {
			end = hits[i+1].start
		}
		size := end - h.start
		switch h.label {
		case "retrieved_tools":
			out.RetrievedTools += size
		case "mcp_servers":
			out.MCPServers += size
		case "mcp_tool_docs":
			out.MCPToolDocs += size
		case "providers":
			out.Providers += size
		case "active_threads":
			out.ActiveThreads += size
		case "safety_mode":
			out.SafetyMode += size
		case "skills":
			out.Skills += size
		case "blob_hint":
			out.BlobHint += size
		case "previous_context":
			out.PreviousContext += size
		case "directive":
			out.Directive += size
		default:
			out.Other += size
		}
	}

	// Reconcile: the labeled regions plus base/coreTools should sum to
	// Total. Anything off by rounding (shouldn't happen — slicing is
	// exact) we bucket into Other so the UI's bar never lies about the
	// total length.
	sum := out.Base + out.CoreTools + out.RetrievedTools + out.MCPServers + out.MCPToolDocs +
		out.Providers + out.ActiveThreads + out.SafetyMode + out.Skills +
		out.BlobHint + out.PreviousContext + out.Directive + out.Other
	if sum != out.Total {
		out.Other += out.Total - sum
	}
	return out
}

// nativeToolSize estimates how many bytes a tool costs in the
// provider's `tools[]` payload. JSON-serialize the pieces — this is
// very close to what the provider SDK does on the wire. If marshal
// fails for some reason (shouldn't — the same struct travels to the
// real provider) we fall back to a simpler sum so the total is at
// least monotonic.
func nativeToolSize(nt NativeTool) int {
	if b, err := json.Marshal(nt); err == nil {
		return len(b)
	}
	size := len(nt.Name) + len(nt.Description)
	if nt.Parameters != nil {
		if b, err := json.Marshal(nt.Parameters); err == nil {
			size += len(b)
		}
	}
	return size
}

// classifyToolKind bucketizes a tool by name + its registry metadata.
// Core tools (pace, send, spawn, …) are flagged Core; MCP tools we
// identify via the MCPServer field; everything else is local/builtin.
func classifyToolKind(t *Thinker, name string) string {
	if t == nil || t.registry == nil {
		return "local"
	}
	def := t.registry.Get(name)
	if def == nil {
		return "local"
	}
	if def.Core {
		return "core"
	}
	if def.MCPServer != "" {
		return "mcp"
	}
	return "local"
}

// messageBytes = Content + each part's Text. Matches totalChars in
// the /context response so the UI can show message bytes without
// re-walking the array.
func messageBytes(m Message) int {
	n := len(m.Content)
	for _, p := range m.Parts {
		n += len(p.Text)
	}
	// Tool calls + tool results ride in structured fields that travel
	// as JSON on the wire; approximate via the string content only
	// since provider SDKs re-serialize those fields on send.
	for _, tr := range m.ToolResults {
		n += len(tr.Content)
	}
	return n
}

func truncateForPreview(s string, max int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > max {
		return s[:max-1] + "…"
	}
	return s
}
