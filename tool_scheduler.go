package core

import (
	"context"
	"fmt"
)

// Admission is bounded and independent of execution slots. In particular a
// saturated tool pool must not block the owner's later coordination calls.
func queueTool(t *Thinker, call toolCall) {
	t.initializeToolSlots()
	t.toolQueueOnce.Do(func() {
		t.toolQueue = make(chan toolCall, 128)
		go func() {
			for {
				select {
				case <-t.toolContext().Done():
					for {
						select {
						case pending := <-t.toolQueue:
							pending.trace.finish("cancelled")
						default:
							return
						}
					}
				case next := <-t.toolQueue:
					executeTool(t, next)
				}
			}
		}()
	})
	call.generation = t.toolGeneration.Load()
	call.admitted = true
	call.Args = copyStringMap(call.Args)
	call.executionIDs = t.currentEventExecutions()
	t.resolveToolCall(&call)
	t.queueToolTrace(&call)
	if call.NativeID != "" {
		t.pendingTools.Store(call.NativeID, call.Name)
	}
	select {
	case <-t.toolContext().Done():
		call.trace.finish("cancelled")
		t.pendingTools.Delete(call.NativeID)
	case t.toolQueue <- call:
	default:
		call.trace.finish("rejected")
		t.pendingTools.Delete(call.NativeID)
		result := ToolResult{CallID: call.NativeID, ToolName: call.Name, IsError: true, Content: "Tool queue is full; wait for current work before retrying."}
		t.bus.Publish(Event{Type: EventInbox, To: t.threadID, ToolResult: &result, ExecutionIDs: call.executionIDs})
	}
}

func (t *Thinker) toolContext() context.Context {
	t.toolCtxOnce.Do(func() { t.toolCtx, t.toolCancel = context.WithCancel(context.Background()) })
	return t.toolCtx
}

// Ordinary calls retain their existing admission behavior. Realtime calls can
// also leave a saturated execution queue when their individual call is cancelled.
func (t *Thinker) acquireToolSlotForContext(ctx context.Context) bool {
	if ctx == nil {
		return t.acquireToolSlot()
	}
	t.initializeToolSlots()
	select {
	case t.toolSem <- struct{}{}:
		return true
	case <-ctx.Done():
		return false
	case <-t.quit:
		return false
	}
}

// toolResponseParts renders a late result's media as message parts. Its call
// is already paired with a placeholder, so images travel labelled in the
// message instead of in a tool result.
func toolResponseParts(resp ToolResponse, toolName string) []ContentPart {
	parts := append([]ContentPart(nil), resp.Parts...)
	parts = append(parts, toolImageParts(labelToolImages(resp.Images, toolName))...)
	if resp.IsError {
		parts = append(parts, ContentPart{Type: "text", Text: fmt.Sprintf("Tool failed: %s", resp.Text)})
	}
	return parts
}

// Reset revokes work admitted to the old conversation without stopping the owner.
func (t *Thinker) invalidateTools() {
	t.toolLifecycleMu.Lock()
	defer t.toolLifecycleMu.Unlock()
	t.toolGeneration.Add(1)
	t.resetMCPFailures()
	t.toolCancels.Range(func(_, v any) bool { v.(context.CancelFunc)(); return true })
	t.pendingTools.Clear()
	t.placeholdersSent.Clear()
	t.drainSilentToolResults()
}

func (t *Thinker) publishToolFailure(call toolCall, generation uint64, executions []string, message string) {
	t.toolLifecycleMu.Lock()
	defer t.toolLifecycleMu.Unlock()
	if generation != t.toolGeneration.Load() || t.toolContext().Err() != nil {
		return
	}
	t.pendingTools.Delete(call.NativeID)
	event := Event{Type: EventInbox, To: t.threadID, Text: fmt.Sprintf("[tool:%s] error: %s", call.Name, message), ExecutionIDs: executions, ToolGeneration: &generation}
	if _, late := t.placeholdersSent.LoadAndDelete(call.NativeID); late {
		event.Text = "[late-result] " + event.Text
	} else {
		event.ToolResult = &ToolResult{CallID: call.NativeID, ToolName: call.Name, Content: message, IsError: true}
	}
	t.bus.Publish(event)
}
