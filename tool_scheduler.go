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
					return
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
	if call.NativeID != "" {
		t.pendingTools.Store(call.NativeID, call.Name)
	}
	select {
	case <-t.toolContext().Done():
		t.pendingTools.Delete(call.NativeID)
	case t.toolQueue <- call:
	default:
		t.pendingTools.Delete(call.NativeID)
		result := ToolResult{CallID: call.NativeID, ToolName: call.Name, IsError: true, Content: "Tool queue is full; wait for current work before retrying."}
		t.bus.Publish(Event{Type: EventInbox, To: t.threadID, ToolResult: &result, ExecutionIDs: call.executionIDs})
	}
}

func (t *Thinker) toolContext() context.Context {
	t.toolCtxOnce.Do(func() { t.toolCtx, t.toolCancel = context.WithCancel(context.Background()) })
	return t.toolCtx
}

func toolResponseParts(resp ToolResponse) []ContentPart {
	parts := append([]ContentPart(nil), resp.Parts...)
	if resp.Image != nil {
		parts = append(parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64," + base64Encode(resp.Image)}})
	}
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
