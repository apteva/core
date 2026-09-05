package core

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

type toolCall struct {
	generation   uint64
	admitted     bool
	executionIDs []string
	Name         string
	Args         map[string]string
	Raw          string // original matched text (or synthetic for native calls)
	NativeID     string // provider-assigned ID for native tool calls (empty for text-parsed)
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
	executionIDs := call.executionIDs
	if executionIDs == nil {
		executionIDs = t.currentEventExecutions()
	}
	generation := call.generation
	if !call.admitted {
		generation = t.toolGeneration.Load()
	}
	if generation != t.toolGeneration.Load() {
		return
	}
	if !t.acquireToolSlot() {
		t.pendingTools.Delete(call.NativeID)
		return
	}
	if generation != t.toolGeneration.Load() {
		t.releaseToolSlot()
		return
	}
	t.asyncToolsActive.Add(1)

	// Telemetry: tool.call
	if t.telemetry != nil {
		t.telemetry.Emit("tool.call", t.threadID, ToolCallData{
			ID: call.NativeID, Name: call.Name, Args: call.Args, Reason: reason, ExecutionIDs: executionIDs,
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
		defer t.asyncToolsActive.Add(-1)
		ctx, cancel := context.WithTimeout(t.toolContext(), 3*time.Minute)
		t.toolLifecycleMu.Lock()
		if generation != t.toolGeneration.Load() {
			t.toolLifecycleMu.Unlock()
			cancel()
			return
		}
		t.toolCancels.Store(call.NativeID, context.CancelFunc(cancel))
		t.toolLifecycleMu.Unlock()
		defer func() { cancel(); t.toolCancels.Delete(call.NativeID) }()
		defer func() {
			select {
			case t.toolCompleted <- struct{}{}:
			default:
			}
		}()
		logMsg("TOOL", fmt.Sprintf("dispatch %s", call.Name))
		start := time.Now()
		defer func() {
			if call.NativeID != "" {
				t.pendingTools.Delete(call.NativeID)
			}
		}()
		defer func() {
			if r := recover(); r != nil {
				logMsg("TOOL", fmt.Sprintf("PANIC %s: %v", call.Name, r))
				t.publishToolFailure(call, generation, executionIDs, fmt.Sprintf("panic: %v", r))
				if t.telemetry != nil {
					result := fmt.Sprintf("panic: %v", r)
					data := newToolResultData(
						call.NativeID, call.Name, time.Since(start).Milliseconds(), false,
						result, result, 0,
					)
					data.ExecutionIDs = executionIDs
					t.telemetry.Emit("tool.result", t.threadID, data)
				}
			}
		}()
		releaseBudget, budgetErr := t.acquireExecutionBudget(ctx)
		if budgetErr != nil {
			t.publishToolFailure(call, generation, executionIDs, budgetErr.Error())
			return
		}
		defer releaseBudget()
		wakePolicy := WakeOnResultAlways
		if t.registry != nil {
			if def := t.registry.Get(call.Name); def != nil && def.WakeOnResult != "" {
				wakePolicy = def.WakeOnResult
			}
		}
		var resp ToolResponse
		if t.registry != nil {
			dispatchArgs := toolDispatchArgs(t, call)
			if t.telemetry != nil {
				if def := t.registry.Get(call.Name); def != nil && def.MCP {
					typedArgs := mcpArgumentsFromStrings(dispatchArgs, def.InputSchema)
					t.telemetry.Emit("tool.arguments", t.threadID, newToolArgumentsData(call.NativeID, call.Name, "mcp_typed", typedArgs))
				}
			}
			if res, ok := t.registry.DispatchContext(ctx, call.Name, dispatchArgs); ok {
				resp = res
			} else {
				resp = ToolResponse{Text: fmt.Sprintf("unknown tool %q", call.Name), IsError: true}
			}
		} else {
			resp = ToolResponse{Text: fmt.Sprintf("unknown tool %q", call.Name), IsError: true}
		}
		if generation != t.toolGeneration.Load() || t.toolContext().Err() != nil {
			return
		}
		logMsg("TOOL", fmt.Sprintf("result %s (%dms): bytes=%d error=%v", call.Name, time.Since(start).Milliseconds(), len(resp.Text), resp.IsError))

		// Telemetry: tool.result
		if t.telemetry != nil {
			success := !resp.IsError && !strings.HasPrefix(resp.Text, "error") && !strings.HasPrefix(resp.Text, "unknown")
			data := newToolResultData(
				call.NativeID, call.Name, time.Since(start).Milliseconds(), success,
				resp.Text, resp.Text, len(resp.Image),
			)
			data.ExecutionIDs = executionIDs
			t.telemetry.Emit("tool.result", t.threadID, data)
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

		if !t.executionGate(ExecutionPhaseToolAfter, ExecutionGate{
			Tool:    call.Name,
			CallID:  call.NativeID,
			Summary: "Tool result ready",
			Result:  resultText,
		}) {
			return
		}
		t.toolLifecycleMu.Lock()
		defer t.toolLifecycleMu.Unlock()
		if generation != t.toolGeneration.Load() {
			return
		}
		// Remove pending atomically with publishing the terminal result, so a
		// placeholder cannot be inserted between these operations.
		t.pendingTools.Delete(call.NativeID)
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
				Text: lateText, ExecutionIDs: executionIDs, Parts: toolResponseParts(resp), ToolGeneration: &generation,
			})
			return
		}

		if wakePolicy == WakeOnResultOnError && !resp.IsError && !t.realtimeMode {
			t.queueSilentToolResult(toolResult)
			return
		}

		t.bus.Publish(Event{
			Type: EventInbox, To: t.threadID,
			Text:           fmt.Sprintf("[tool:%s] %s", call.Name, resultText),
			ToolResult:     &toolResult,
			ToolGeneration: &generation,
			Parts:          resp.Parts,
			ExecutionIDs:   executionIDs,
		})
	}()
}

func toolDispatchArgs(t *Thinker, call toolCall) map[string]string {
	if t == nil || t.registry == nil {
		return call.Args
	}
	def := t.registry.Get(call.Name)
	if def == nil || !def.MCP || (!def.MCPApp && def.MCPServer != "channels") {
		return call.Args
	}
	// Apteva app MCPs receive the current opaque thread id through the trusted
	// gateway. It is injected after telemetry and absent from the model-visible
	// schema, so the model cannot forge its caller identity.
	out := make(map[string]string, len(call.Args)+2)
	for key, value := range call.Args {
		out[key] = value
	}
	out["_apteva_caller_thread"] = t.threadID
	// The provider-native call id survives transport retries and is absent from
	// the model-visible schema. The app gateway promotes it to the trusted
	// X-Apteva-Tool-Call-ID header for sidecar idempotency ledgers.
	if strings.TrimSpace(call.NativeID) != "" {
		out["_apteva_tool_call_id"] = call.NativeID
	}
	// Compatibility for the built-in Channels MCP while it still routes the
	// legacy hidden argument directly instead of the generic caller header.
	if def.MCPServer == "channels" {
		out["_apteva_caller_context"] = t.threadID
	}
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
