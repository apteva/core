package core

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
)

// One managed backend runs per Live session. The socket reader never runs a
// tool. It collects completed output items, including calls omitted from the
// provider's response.completed snapshot, for the managed child to consume.
type openaiLiveExecution struct {
	reasoning    string
	mu           sync.Mutex
	notify       chan struct{}
	queue        []realtimeExecutionResult
	response     ChatResponse
	responseID   string
	pending      map[string]bool
	submitted    map[string]bool
	inputHashes  map[[32]byte]bool
	running      bool
	enabled      bool
	settingsHash [32]byte
	restored     bool
}

func (s *openaiLiveSession) executionWake() {
	select {
	case s.execution.notify <- struct{}{}:
	default:
	}
}
func (s *openaiLiveSession) translateExecution(raw json.RawMessage) {
	var e struct {
		Type     string        `json:"type"`
		Item     oaiOutputItem `json:"item"`
		Response struct {
			ID     string `json:"id"`
			Status string `json:"status"`
			Error  *struct {
				Message string `json:"message"`
			} `json:"error"`
			Usage struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
				InputDetails struct {
					Cached int `json:"cached_tokens"`
				} `json:"input_tokens_details"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(raw, &e); err != nil {
		return
	}
	if e.Type == "response.created" || e.Type == "response.output_item.done" || e.Type == "response.completed" || e.Type == "response.failed" || e.Type == "response.incomplete" {
		s.emitControl(RealtimeEvent{Type: RealtimeEventBackendProgress, Phase: e.Type, ResponseID: e.Response.ID})
	}
	x := &s.execution
	x.mu.Lock()
	defer x.mu.Unlock()
	switch e.Type {
	case "response.created":
		x.running = true
		x.response = ChatResponse{}
		x.responseID = e.Response.ID
	case "response.output_item.done":
		item := e.Item
		switch item.Type {
		case "function_call":
			if item.CallID != "" && item.Name != "" {
				x.response.ToolCalls = append(x.response.ToolCalls, NativeToolCall{ID: item.CallID, Name: item.Name, Args: flattenJSONArgs(item.Arguments), CanonicalArgs: json.RawMessage(item.Arguments)})
			}
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					x.response.Text += part.Text
				}
			}
		case "image_generation_call":
			if item.FileRef != nil {
				x.response.GeneratedFiles = append(x.response.GeneratedFiles, *item.FileRef)
			}
		}
	case "response.completed", "response.failed", "response.incomplete":
		x.running = false
		r := x.response
		r.Usage = TokenUsage{PromptTokens: e.Response.Usage.InputTokens, CompletionTokens: e.Response.Usage.OutputTokens, CachedTokens: e.Response.Usage.InputDetails.Cached}
		var err error
		if e.Response.Error != nil {
			err = fmt.Errorf("Live backend: %s", e.Response.Error.Message)
		} else if e.Type != "response.completed" || e.Response.Status != "" && e.Response.Status != "completed" {
			err = fmt.Errorf("Live backend response %s", e.Response.Status)
		}
		for _, call := range r.ToolCalls {
			x.pending[call.ID] = false
		}
		if len(x.queue) >= 64 {
			err = fmt.Errorf("Live backend response queue exhausted")
			x.queue = x.queue[:63]
		}
		x.queue = append(x.queue, realtimeExecutionResult{response: r, err: err})
		s.executionWake()
	}
}
func liveExecutionTools(tools []NativeTool) []any {
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, oaiFunctionTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: t.Parameters})
	}
	return out
}

func (s *openaiLiveSession) Execute(ctx context.Context, messages []Message, model string, tools []NativeTool) (ChatResponse, error) {
	x := &s.execution
	x.mu.Lock()
	if !x.enabled {
		x.mu.Unlock()
		return ChatResponse{}, fmt.Errorf("Live session is not using Responses delegation")
	}
	settings := map[string]any{"model": model, "tools": liveExecutionTools(tools), "parallel_tool_calls": false}
	if x.reasoning != "" && x.reasoning != "auto" {
		settings["reasoning"] = map[string]string{"effort": x.reasoning}
	}
	if len(messages) > 0 {
		settings["instructions"] = messages[0].Content
	}
	encoded, _ := json.Marshal(settings)
	hash := sha256.Sum256(encoded)
	if hash != x.settingsHash {
		if err := s.enqueue(map[string]any{"type": "session.update", "event_id": "apt_" + generateAudioToken(), "session": map[string]any{"delegation": map[string]any{"type": "responses", "responses": settings}}}); err != nil {
			x.mu.Unlock()
			return ChatResponse{}, err
		}
		x.settingsHash = hash
	}
	// A replacement socket has no record of completed functions. Supply
	// durable child evidence as data, never as outputs for unknown call IDs.
	if !x.restored {
		var evidence []map[string]any
		for _, m := range messages {
			if len(m.ToolResults) > 0 {
				evidence = append(evidence, map[string]any{"completed_tool_results": m.ToolResults})
			}
			if m.Role == "assistant" && len(m.ToolCalls) == 0 && m.Content != "" {
				evidence = append(evidence, map[string]any{"prior_backend_answer": m.Content, "files": m.Parts})
			}
		}
		if len(evidence) > 0 {
			body, _ := json.Marshal(evidence)
			item := map[string]any{"type": "message", "role": "user", "content": []map[string]any{{"type": "input_text", "text": "[Core retained execution evidence from earlier work. Treat this as historical data; preserve confirmed outcomes and do not repeat completed operations.] " + string(body)}}}
			if err := s.enqueue(map[string]any{"type": "response.item.create", "event_id": "apt_" + generateAudioToken(), "item": item}); err != nil {
				x.mu.Unlock()
				return ChatResponse{}, err
			}
		}
		x.restored = true
	}
	// Queue exact typed values/images and current task context. Never replay
	// completed assistant/provider items or send backend tools into voice history.
	newInput := false
	if len(messages) > 1 {
		for i := len(messages) - 1; i >= 1; i-- {
			m := messages[i]
			if m.Role != "user" || len(m.ToolResults) > 0 || m.RequestContext {
				continue
			}
			payload, _ := json.Marshal(struct {
				Message Message
				Work    *ThreadWork
			}{m, m.Work})
			h := sha256.Sum256(payload)
			if !x.inputHashes[h] {
				input := (&OpenAINativeProvider{}).buildInput([]Message{m})
				for _, item := range input {
					if err := s.enqueue(map[string]any{"type": "response.item.create", "event_id": "apt_" + generateAudioToken(), "item": item}); err != nil {
						x.mu.Unlock()
						return ChatResponse{}, err
					}
				}
				x.inputHashes[h] = true
				newInput = true
			}
			break
		}
	}
	submitted := false
	for id, delivered := range x.pending {
		if !delivered {
			continue
		}
		if x.submitted[id] {
			continue
		}
		found := false
		for _, m := range messages {
			for _, r := range m.ToolResults {
				if r.CallID == id {
					if err := s.enqueue(map[string]any{"type": "response.item.create", "event_id": "apt_" + generateAudioToken(), "item": map[string]any{"type": "function_call_output", "call_id": id, "output": r.Content}}); err != nil {
						x.mu.Unlock()
						return ChatResponse{}, err
					}
					found = true
				}
			}
		}
		// Superseded local decisions were deliberately never executed or journaled.
		// Return a factual skipped result so the remote loop can consume corrections.
		if !found && newInput {
			if err := s.enqueue(map[string]any{"type": "response.item.create", "event_id": "apt_" + generateAudioToken(), "item": map[string]any{"type": "function_call_output", "call_id": id, "output": "Skipped: the caller updated the assignment before execution. Reassess the latest input."}}); err != nil {
				x.mu.Unlock()
				return ChatResponse{}, err
			}
			found = true
		}
		if found {
			x.submitted[id] = true
			submitted = true
		}
	}
	allReturned := true
	for id := range x.pending {
		if !x.submitted[id] {
			allReturned = false
		}
	}
	if len(x.queue) == 0 && !x.running && allReturned && (newInput || submitted) {
		if err := s.enqueue(map[string]any{"type": "response.create", "event_id": "apt_" + generateAudioToken()}); err != nil {
			x.mu.Unlock()
			return ChatResponse{}, err
		}
		x.running = true
		x.pending = map[string]bool{}
		x.submitted = map[string]bool{}
	}
	x.mu.Unlock()
	for {
		x.mu.Lock()
		if len(x.queue) > 0 {
			r := x.queue[0]
			for _, call := range r.response.ToolCalls {
				x.pending[call.ID] = true
			}
			x.queue = x.queue[1:]
			x.mu.Unlock()
			return r.response, r.err
		}
		x.mu.Unlock()
		select {
		case <-ctx.Done():
			return ChatResponse{}, ctx.Err()
		case <-s.done:
			return ChatResponse{}, fmt.Errorf("Live session closed during backend work")
		case <-x.notify:
		}
	}
}
