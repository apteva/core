package core

import (
	"testing"
)

// Pins the regression where sanitizeToolPairs preserved orphan tool
// results carrying an image. The "screenshots must survive" exception
// produced wire payloads with unpaired tool_call_ids that strict
// providers (Moonshot via opencode-go, Anthropic) rejected with HTTP
// 400. Old screenshots are already evicted to text placeholders by
// thinker.go's image-cap, so dropping orphan-with-image entirely is
// safe and matches the protocol.
//
// If anyone reintroduces the `tr.Image != nil` exception in
// sanitizeToolPairs, these tests fail and the Patreon login flow
// (and any other long-running multi-turn image-tool flow) starts
// 400ing on opencode-go again.

func TestSanitize_OrphanToolResult_NoImage_Dropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{{CallID: "screenshot_tool:1", Content: "stale"}}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected orphan tool_result to be dropped, got %d msgs", len(out))
	}
}

// The regression. Before the fix this kept the orphan because
// `tr.Image != nil`, leaving a tool_call_id with no matching tool_use
// in the wire payload. Moonshot 400s on this; Fireworks accepts it.
func TestSanitize_OrphanToolResult_WithImage_Dropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:8", Content: "stale screenshot", Image: []byte("fake-png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected orphan tool_result with image to be dropped, got %d msgs (regression: image-bearing orphan being preserved again)", len(out))
	}
}

func TestSanitize_OrphanToolUse_NoResult_Dropped(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:1", Name: "screenshot_tool"}}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 1 {
		t.Fatalf("expected assistant message to be kept, got %d", len(out))
	}
	if len(out[0].ToolCalls) != 0 {
		t.Fatalf("expected orphan tool_call to be dropped, got %d", len(out[0].ToolCalls))
	}
}

// Symmetric regression. Before the fix the assistant tool_use was
// kept when its only matching ToolResult had an image — but if that
// image-bearing result was itself dropped (which it now is), there
// was no real match. Now both must be dropped together.
func TestSanitize_OrphanToolUse_KeptByImageException_Dropped(t *testing.T) {
	in := []Message{
		// Assistant emits a tool_call.
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:8", Name: "screenshot_tool"}}},
		// Its result is an orphan-with-image (the matching tool_use is
		// elsewhere in the original conversation but rolled off the
		// LoadTail window). The previous code path: image-exception
		// kept the result → toolResultIDs included :8 → assistant
		// tool_call also kept → both wrong-shape on wire.
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:99", Image: []byte("fake-png")}, // orphan
		}},
	}
	out := sanitizeToolPairs(in)
	// Both orphans should be dropped: assistant turn loses its tool_call,
	// user turn loses its only tool_result and is therefore omitted.
	for _, m := range out {
		for _, tr := range m.ToolResults {
			if tr.CallID == "screenshot_tool:99" {
				t.Fatalf("orphan image tool_result :99 leaked through")
			}
		}
		for _, tc := range m.ToolCalls {
			if tc.ID == "screenshot_tool:8" && len(out) > 1 {
				// It's only valid to keep :8 if its result is also kept.
				found := false
				for _, m2 := range out {
					for _, tr := range m2.ToolResults {
						if tr.CallID == "screenshot_tool:8" {
							found = true
						}
					}
				}
				if !found {
					t.Fatalf("orphan tool_call :8 kept without matching result")
				}
			}
		}
	}
}

func TestSanitize_PairedToolsWithImage_Kept(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:1", Name: "screenshot_tool"}}},
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:1", Content: "ok", Image: []byte("png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 2 {
		t.Fatalf("expected both paired messages kept, got %d", len(out))
	}
	if len(out[0].ToolCalls) != 1 || out[0].ToolCalls[0].ID != "screenshot_tool:1" {
		t.Fatalf("assistant tool_call dropped from a valid pair")
	}
	if len(out[1].ToolResults) != 1 || out[1].ToolResults[0].CallID != "screenshot_tool:1" {
		t.Fatalf("user tool_result dropped from a valid pair")
	}
	if out[1].ToolResults[0].Image == nil {
		t.Fatalf("image stripped from valid tool_result")
	}
}

// In-flight async tool_calls dispatched as goroutines
// are tracked in pendingTools. The sanitizer must keep their
// assistant turn so the iteration-barrier placeholder can pair with
// them later — without this guard, sanitize would strip the call
// before its result lands.
func TestSanitize_InFlightToolUse_KeptViaPending(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:7", Name: "screenshot_tool"}}},
	}
	pending := map[string]bool{"screenshot_tool:7": true}
	out := sanitizeToolPairs(in, pending)
	if len(out) != 1 || len(out[0].ToolCalls) != 1 {
		t.Fatalf("in-flight tool_use stripped despite pending: out=%+v", out)
	}
}

func TestSanitize_AllOrphanResults_MessageDropped(t *testing.T) {
	in := []Message{
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "a:1", Content: "x"},
			{CallID: "b:2", Content: "y", Image: []byte("png")},
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 0 {
		t.Fatalf("expected message with all-orphan results to be dropped, got %d", len(out))
	}
}

// Mixed message: one tool_result paired, one orphan. The orphan is
// dropped; the message survives with just the paired result.
func TestSanitize_MixedToolResults_OnlyOrphanDropped(t *testing.T) {
	in := []Message{
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "a:1", Name: "x"}}},
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "a:1", Content: "ok"},                               // paired → keep
			{CallID: "ghost:99", Content: "stale", Image: []byte("png")}, // orphan → drop
		}},
	}
	out := sanitizeToolPairs(in)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages kept, got %d", len(out))
	}
	if len(out[1].ToolResults) != 1 || out[1].ToolResults[0].CallID != "a:1" {
		t.Fatalf("expected only paired tool_result kept, got %+v", out[1].ToolResults)
	}
}

// End-to-end referential-integrity check: post-sanitize, every
// tool_result.CallID points to an assistant tool_calls[].id earlier
// in the slice, and every assistant tool_call has a tool_result later
// (or is in pending). This is the property Moonshot enforces and the
// reason the orphan-with-image rule had to die.
func TestSanitize_PostconditionReferentialIntegrity(t *testing.T) {
	// Simulates a LoadTail window where a screenshot tool_result for
	// screenshot_tool:8 made the cut but its assistant tool_call did not.
	in := []Message{
		// Window starts mid-conversation; the assistant turn for :8 is
		// outside the slice (rolled off by LoadTail). This is exactly
		// the shape that triggered the Patreon-login 400.
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "screenshot_tool:8", Content: "stale", Image: []byte("png")},
		}},
		// Healthy pair inside the window.
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:9", Name: "screenshot_tool"}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "screenshot_tool:9", Content: "ok"}}},
		// Orphan tool_call with no result.
		// not in pendingTools because the run died).
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "screenshot_tool:10", Name: "screenshot_tool"}}},
	}
	out := sanitizeToolPairs(in)

	// Build id sets from the sanitized output.
	uses := map[string]int{} // id → index of assistant turn
	for i, m := range out {
		for _, tc := range m.ToolCalls {
			uses[tc.ID] = i
		}
	}
	for i, m := range out {
		for _, tr := range m.ToolResults {
			useIdx, ok := uses[tr.CallID]
			if !ok {
				t.Errorf("post-sanitize: tool_result %s has no preceding tool_use (Moonshot would 400)", tr.CallID)
			} else if useIdx >= i {
				t.Errorf("post-sanitize: tool_result %s appears at msg[%d] but its tool_use is at msg[%d] (must be earlier)", tr.CallID, i, useIdx)
			}
		}
	}
	for _, m := range out {
		for _, tc := range m.ToolCalls {
			matched := false
			for _, m2 := range out {
				for _, tr := range m2.ToolResults {
					if tr.CallID == tc.ID {
						matched = true
					}
				}
			}
			if !matched {
				t.Errorf("post-sanitize: tool_use %s has no matching tool_result (Moonshot would 400)", tc.ID)
			}
		}
	}
}
