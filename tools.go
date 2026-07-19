package core

import (
	"fmt"
	"regexp"
	"strings"
	"time"
)

type toolCall struct {
	Name     string
	Args     map[string]string
	Raw      string // original matched text (or synthetic for native calls)
	NativeID string // provider-assigned ID for native tool calls (empty for text-parsed)
}

// [[tool_name key="val" key2="val2"]] — values can span multiple lines, escaped quotes allowed
var toolCallRe = regexp.MustCompile(`(?s)\[\[([\w-]+)((?:\s+\w+="(?:[^"\\]|\\.)*")*)\]\]`)
var argRe = regexp.MustCompile(`(?s)(\w+)="((?:[^"\\]|\\.)*)"`)

// stripToolCalls removes [[...]] tool call syntax from text for display
func stripToolCalls(text string) string {
	cleaned := toolCallRe.ReplaceAllString(text, "")
	return collapseWhitespace(cleaned)
}

func parseToolCalls(text string) []toolCall {
	matches := toolCallRe.FindAllStringSubmatch(text, -1)
	var calls []toolCall
	for _, m := range matches {
		name := m[1]
		args := make(map[string]string)
		for _, a := range argRe.FindAllStringSubmatch(m[2], -1) {
			// Unescape \" in values
			val := strings.ReplaceAll(a[2], `\"`, `"`)
			args[a[1]] = val
		}
		calls = append(calls, toolCall{Name: name, Args: args, Raw: m[0]})
	}
	return calls
}

// toolArgsSummary builds a short string representation of tool args.
func toolArgsSummary(call toolCall) string {
	argsSummary := ""
	for k, v := range call.Args {
		if len(argsSummary) > 0 {
			argsSummary += ", "
		}
		val := v
		if len(val) > 50 {
			val = val[:50] + "..."
		}
		argsSummary += k + "=" + val
	}
	return argsSummary
}

func executeTool(t *Thinker, call toolCall) {
	// Extract _reason before dispatch (observability field, not passed to handler)
	reason := call.Args["_reason"]
	delete(call.Args, "_reason")
	if !t.acquireToolSlot() {
		return
	}

	// Telemetry: tool.call
	if t.telemetry != nil {
		t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
			ID: call.NativeID, Name: call.Name, Args: call.Args, Reason: reason,
		})
	}

	// Track pending async tool call. Value carries the tool name so the
	// iter-boundary placeholder injector can label its synthetic
	// tool_result and the stale-placeholder sweeper can emit a useful
	// timeout message if the goroutine never returns.
	if call.NativeID != "" {
		t.pendingTools.Store(call.NativeID, call.Name)
	}

	go func() {
		defer t.releaseToolSlot()
		logMsg("TOOL", fmt.Sprintf("dispatch %s reason=%q args=%v", call.Name, reason, call.Args))
		start := time.Now()
		defer func() {
			if call.NativeID != "" {
				t.pendingTools.Delete(call.NativeID)
			}
		}()
		defer func() {
			if r := recover(); r != nil {
				logMsg("TOOL", fmt.Sprintf("PANIC %s: %v", call.Name, r))
				t.Inject(fmt.Sprintf("[tool:%s] error: panic: %v", call.Name, r))
				if t.telemetry != nil {
					result := fmt.Sprintf("panic: %v", r)
					t.telemetry.Emit("tool.result", t.threadID, newToolResultData(
						call.NativeID, call.Name, time.Since(start).Milliseconds(), false,
						result, result, 0,
					))
				}
			}
		}()
		wakePolicy := WakeOnResultAlways
		if t.registry != nil {
			if def := t.registry.Get(call.Name); def != nil && def.WakeOnResult != "" {
				wakePolicy = def.WakeOnResult
			}
		}
		var resp ToolResponse
		if t.registry != nil {
			dispatchArgs := toolDispatchArgs(t, call)
			if res, ok := t.registry.Dispatch(call.Name, dispatchArgs); ok {
				resp = res
			} else {
				resp = ToolResponse{Text: fmt.Sprintf("unknown tool %q", call.Name), IsError: true}
			}
		} else {
			resp = ToolResponse{Text: fmt.Sprintf("unknown tool %q", call.Name), IsError: true}
		}
		resultPreview := resp.Text
		if len(resultPreview) > 200 {
			resultPreview = resultPreview[:200] + "..."
		}
		logMsg("TOOL", fmt.Sprintf("result %s (%dms): %s", call.Name, time.Since(start).Milliseconds(), resultPreview))

		// Telemetry: tool.result
		if t.telemetry != nil {
			success := !resp.IsError && !strings.HasPrefix(resp.Text, "error") && !strings.HasPrefix(resp.Text, "unknown")
			t.telemetry.Emit("tool.result", t.threadID, newToolResultData(
				call.NativeID, call.Name, time.Since(start).Milliseconds(), success,
				resp.Text, resp.Text, len(resp.Image),
			))
		}

		// Emit visual chunk for TUI
		resultPreviewForTUI := resp.Text
		if len(resultPreviewForTUI) > 120 {
			resultPreviewForTUI = resultPreviewForTUI[:120] + "..."
		}
		t.bus.Publish(Event{Type: EventChunk, From: t.threadID, Text: "\n← " + call.Name + ": " + resultPreviewForTUI + "\n", Iteration: t.status().Iteration})

		// Inject result as a proper ToolResult event (text + optional image).
		// MCP tools may opt into success-only silent delivery via
		// _meta["io.apteva/wakeOnResult"]="on_error"; those results are
		// still queued for history pairing, but do not wake the model.
		resultText := resp.Text
		toolResult := ToolResult{
			CallID:   call.NativeID,
			ToolName: call.Name,
			Content:  resultText,
			Image:    resp.Image,
			IsError:  resp.IsError,
		}

		// Late-result routing. If the iter-boundary barrier already
		// injected a placeholder tool_result for this call id (because
		// this goroutine didn't finish in time), we CANNOT publish a
		// second ToolResult for the same id — the tool_use is already
		// paired with the placeholder, and adding a second result
		// recreates the exact duplicate-pair state that confuses the
		// model. Instead, publish the real result as a text event
		// prefixed [late-result] so it lands as a plain user message in
		// the next drain. The model gets the real answer with a clear
		// "this is the delayed result" label.
		if _, hasPlaceholder := t.placeholdersSent.LoadAndDelete(call.NativeID); hasPlaceholder {
			lateText := fmt.Sprintf("[late-result] Tool %s (call id=%s) completed: %s", call.Name, call.NativeID, resultText)
			t.bus.Publish(Event{
				Type: EventInbox, To: t.threadID,
				Text: lateText,
			})
			return
		}

		if !t.executionGate(ExecutionPhaseToolAfter, ExecutionGate{
			Tool:    call.Name,
			CallID:  call.NativeID,
			Summary: "Tool result ready",
			Result:  resultText,
		}) {
			return
		}

		if wakePolicy == WakeOnResultOnError && !resp.IsError {
			t.queueSilentToolResult(toolResult)
			return
		}

		t.bus.Publish(Event{
			Type: EventInbox, To: t.threadID,
			Text:       fmt.Sprintf("[tool:%s] %s", call.Name, resultText),
			ToolResult: &toolResult,
		})
	}()
}

func toolDispatchArgs(t *Thinker, call toolCall) map[string]string {
	if t == nil || t.registry == nil {
		return call.Args
	}
	def := t.registry.Get(call.Name)
	if def == nil || !def.MCP || def.MCPServer != "channels" {
		return call.Args
	}
	// The Channels MCP uses this opaque runtime value to route an answer back
	// to the durable Apteva conversation that triggered this thinker. It is
	// injected after telemetry and absent from the model-visible schema.
	out := make(map[string]string, len(call.Args)+1)
	for key, value := range call.Args {
		out[key] = value
	}
	out["_apteva_caller_context"] = t.threadID
	return out
}

func collapseWhitespace(s string) string {
	lines := strings.Split(s, "\n")
	var out []string
	blank := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			blank++
			if blank <= 1 {
				out = append(out, "")
			}
			continue
		}
		blank = 0
		out = append(out, trimmed)
	}
	return strings.Join(out, "\n")
}
