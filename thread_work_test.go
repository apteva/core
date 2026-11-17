package core

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestManagedAssignmentUsesNormalThreadAndDurableInbox(t *testing.T) {
	parent := newTestThinkerFull()
	parent.session = NewSession(t.TempDir(), "main")
	parent.registry = NewToolRegistry("")
	p := newRecordingThreadEventProvider()
	parent.provider = p
	defer parent.Stop()
	if err := parent.threads.SpawnWithOpts("worker", "Return concise results", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	defer parent.threads.KillAll()
	worker := parent.threads.threads["worker"]
	if worker.Parent != parent || worker.ParentID != "main" || worker.Thinker.bus != parent.bus {
		t.Fatal("assignment bypassed managed thread tree")
	}
	if err := parent.threads.Start("worker"); err != nil {
		t.Fatal(err)
	}
	for revision := uint64(1); revision <= 2; revision++ {
		w := ThreadWork{ID: "task", Revision: revision, Origin: "main", Kind: "request"}
		if err := parent.threads.QueueWork("worker", w, "Answer this assignment", nil); err != nil {
			t.Fatal(err)
		}
		select {
		case <-p.requests:
		case <-time.After(time.Second):
			t.Fatal("normal thinking loop did not run")
		}
		deadline := time.After(time.Second)
		found := false
		for !found {
			select {
			case <-deadline:
				t.Fatal("result not returned to parent")
			case <-parent.sub.Wake:
				for _, e := range parent.sub.DrainTargeted() {
					if e.Work != nil && e.Work.Kind == "result" && e.Work.Revision == revision {
						if e.Text != "event handled" || e.Work.Worker != "worker" {
							t.Fatalf("result: %+v", e)
						}
						found = true
					}
				}
			}
		}
	}
	time.Sleep(30 * time.Millisecond)
	if p.calls.Load() != 2 {
		t.Fatalf("idle worker repeated inference: %d", p.calls.Load())
	}
	if parent.threads.Count() != 1 {
		t.Fatal("assignment completion terminated worker")
	}
	saved := readSessionEntriesForTest(t, worker.Thinker.session.path)
	hasCorrelation := false
	for _, m := range saved {
		hasCorrelation = hasCorrelation || m.Work != nil
	}
	if !hasCorrelation {
		t.Fatal("assignment correlation not persisted")
	}
}

func TestQueuedToolRejectsSupersededAssignment(t *testing.T) {
	th := newTestThinkerFull()
	defer th.Stop()
	th.registry = NewToolRegistry("")
	th.work = &threadWorkState{}
	var executed atomic.Int32
	th.registry.Register(&ToolDef{Name: "operation", Handler: func(map[string]string) ToolResponse { executed.Add(1); return ToolResponse{Text: "done"} }})
	old := ThreadWork{ID: "task", Revision: 1, Origin: "main", Kind: "request"}
	th.work.observe(&old)
	newer := old
	newer.Revision = 2
	th.work.advance(newer)
	executeTool(th, toolCall{Name: "operation", NativeID: "call", Args: map[string]string{}, work: &old})
	deadline := time.After(time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("skipped operation did not return its matching result")
		case <-th.sub.Wake:
			for _, e := range th.sub.DrainTargeted() {
				if e.ToolResult != nil {
					if !e.ToolResult.IsError || !strings.Contains(e.ToolResult.Content, "assignment changed") {
						t.Fatalf("result=%+v", e.ToolResult)
					}
					if executed.Load() != 0 {
						t.Fatal("superseded operation executed")
					}
					return
				}
			}
		}
	}
}

func TestSupersededWorkKeepsGeneratedFileReferences(t *testing.T) {
	th := newTestThinkerFull()
	th.session = NewSession(t.TempDir(), "worker")
	defer th.Stop()
	ref := FileRef{File: true, Ref: "blobref://artifact", Filename: "result.png", MimeType: "image/png", Size: 42}
	th.preserveSupersededWork(ChatResponse{GeneratedFiles: []FileRef{ref}, ServerResults: []ServerToolResult{{ToolName: "lookup", Output: "confirmed"}}, ToolCalls: []NativeToolCall{{ID: "stale", Name: "operation"}}})
	entries := readSessionEntriesForTest(t, th.session.path)
	if len(entries) != 1 || len(entries[0].ToolCalls) != 0 || len(entries[0].Parts) != 1 || entries[0].Parts[0].FileRef.Ref != ref.Ref {
		t.Fatalf("completed evidence lost: %+v", entries)
	}
}

func managedSessionFixture() *openaiLiveSession {
	return &openaiLiveSession{openaiRealtimeSession: newTranslationSession(), execution: openaiLiveExecution{enabled: true, notify: make(chan struct{}, 1), pending: map[string]bool{}, submitted: map[string]bool{}, inputHashes: map[[32]byte]bool{}}}
}
func managedEvent(s *openaiLiveSession, text string) { s.translateExecution(json.RawMessage(text)) }
func drainManagedWrites(s *openaiLiveSession) []map[string]any {
	var out []map[string]any
	for len(s.outbox) > 0 {
		var e map[string]any
		_ = json.Unmarshal((<-s.outbox).data, &e)
		out = append(out, e)
	}
	return out
}

func TestLiveManagedExecutionCollectsItemsAndContinuesOnce(t *testing.T) {
	s := managedSessionFixture()
	managedEvent(s, `{"type":"response.created","response":{"id":"resp1"}}`)
	managedEvent(s, `{"type":"response.output_item.done","item":{"type":"function_call","name":"lookup","call_id":"call1","arguments":"{\"_reason\":\"Check the record\"}"}}`)
	managedEvent(s, `{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":7,"output_tokens":4}}}`)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	messages := []Message{{Role: "system", Content: "instructions"}, {Role: "user", Content: "lookup"}}
	first, err := s.Execute(ctx, messages, "model", nil)
	if err != nil || len(first.ToolCalls) != 1 || first.ToolCalls[0].ID != "call1" || first.Usage.PromptTokens != 7 {
		t.Fatalf("response=%+v error=%v", first, err)
	}
	_ = drainManagedWrites(s)
	messages = append(messages, Message{Role: "assistant", ToolCalls: first.ToolCalls}, Message{Role: "user", ToolResults: []ToolResult{{CallID: "call1", Content: "verified"}}})
	done := make(chan error, 1)
	go func() {
		r, e := s.Execute(ctx, messages, "model", nil)
		if e == nil && r.Text != "Confirmed" {
			e = context.Canceled
		}
		done <- e
	}()
	var writes []map[string]any
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		writes = append(writes, drainManagedWrites(s)...)
		ready := false
		for _, w := range writes {
			ready = ready || w["type"] == "response.create"
		}
		if ready {
			break
		}
		time.Sleep(time.Millisecond)
	}
	outputs, continuations := 0, 0
	for _, w := range writes {
		if w["type"] == "response.create" {
			continuations++
		}
		if w["type"] == "response.item.create" {
			item := w["item"].(map[string]any)
			if item["type"] == "function_call_output" {
				outputs++
			}
		}
	}
	if outputs != 1 || continuations != 1 {
		t.Fatalf("outputs=%d continuations=%d writes=%+v", outputs, continuations, writes)
	}
	managedEvent(s, `{"type":"response.created","response":{"id":"resp2"}}`)
	managedEvent(s, `{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Confirmed"}]}}`)
	managedEvent(s, `{"type":"response.completed","response":{"status":"completed","output":[]}}`)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestLiveStartupCustomVoiceAndGreetingBudget(t *testing.T) {
	payload, err := buildOpenAILiveStart(RealtimeSessionOpts{Model: "gpt-live-1", Voice: "voice_custom", InitialMessage: "Bonjour", History: []Message{{Role: "user", Content: strings.Repeat("x", 5999)}}}, "")
	if err != nil {
		t.Fatal(err)
	}
	var event map[string]any
	_ = json.Unmarshal(payload, &event)
	session := event["session"].(map[string]any)
	voice := session["audio"].(map[string]any)["output"].(map[string]any)["voice"].(map[string]any)
	if voice["id"] != "voice_custom" || len(session["input"].([]any)) != 1 {
		t.Fatalf("startup=%s", payload)
	}
}

func TestLiveDelegationNotificationJoinsTypedAssignment(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{CoreExecution: true}
	rt.backend.active = true
	rt.backend.requestID = "current"
	rt.backend.revision = 4
	rt.acceptDelegation(RealtimeEvent{DelegationID: "provider-notification"})
	if rt.backend.revision != 4 || rt.backend.want || rt.backend.requestID != "current" || rt.backend.delegation != "provider-notification" {
		t.Fatalf("notification started duplicate work: %+v", rt.backend)
	}
}

func TestLiveExecutionChildBelongsToConversationAndKeepsItsGrants(t *testing.T) {
	root := newTestThinkerFull()
	root.session = NewSession(t.TempDir(), "main")
	root.registry = NewToolRegistry("")
	root.config.RealtimeEnabled = true
	p := newRecordingThreadEventProvider()
	voice := &fakeRealtimeProvider{}
	root.provider = p
	root.pool = &ProviderPool{providers: map[string]LLMProvider{p.Name(): p}, default_: p.Name(), realtimeProviders: map[string]RealtimeProvider{voice.Name(): voice}, realtimeDefault: voice.Name()}
	root.registry.Register(&ToolDef{Name: "allowed", Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "yes"} }})
	root.registry.Register(&ToolDef{Name: "forbidden", Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "no"} }})
	if err := root.threads.SpawnWithOpts("conversation", "Speak with caller", []string{"allowed"}, SpawnOpts{Realtime: true, ProviderName: voice.Name(), DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	parent := root.threads.threads["conversation"]
	rt := parent.Realtime
	defer root.threads.KillAll()
	defer root.Stop()
	defer rt.cancel()
	rt.capabilities = RealtimeCapabilities{CoreExecution: true}
	if err := rt.configureExecution(); err != nil {
		t.Fatal(err)
	}
	child, err := rt.ensureExecutionThread()
	if err != nil {
		t.Fatal(err)
	}
	if child.ParentID != "conversation" || child.Parent != parent.Thinker || child.Depth != parent.Depth+1 || parent.Children == nil {
		t.Fatal("execution child bypassed conversation hierarchy")
	}
	if !child.Thinker.toolAuthorized("allowed") || child.Thinker.toolAuthorized("forbidden") {
		t.Fatal("execution child broadened capabilities")
	}
	if !child.WorkMode || child.Thinker.bus != root.bus || child.Thinker.execution != root.execution {
		t.Fatal("execution bypassed ordinary thread machinery")
	}
	first := newFakeRealtimeSession()
	rt.replaceSession(first)
	rt.backend.active = true
	rt.backend.requestID = "work"
	rt.backend.revision = 1
	rt.backend.delegation = "old-socket-id"
	replacement := newFakeRealtimeSession()
	rt.replaceSession(replacement)
	again, err := rt.ensureExecutionThread()
	if err != nil || again != child {
		t.Fatal("reconnect replaced the execution worker")
	}
	rt.consumeThreadWork(Event{Work: &ThreadWork{ID: "work", Revision: 1, Origin: "conversation", Worker: child.ID, Kind: "result", Status: "completed"}, Text: "Verified result"})
	if len(replacement.texts) != 1 || replacement.texts[0].Content != "Verified result" || len(first.texts) != 0 {
		t.Fatal("result routed to wrong session")
	}
	if rt.backend.active {
		t.Fatal("completed assignment stayed active")
	}
}

func TestLiveChildCompletionRequestDoesNotStopTheChild(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{CoreExecution: true}
	rt.backend.worker = &Thread{ID: "worker", Thinker: newTestThinker()}
	defer rt.backend.worker.Thinker.Stop()
	rt.backend.requestID = "task"
	rt.backend.revision = 1
	rt.toolAllowlist = map[string]bool{"done": true}
	rt.consumeThreadWork(Event{ID: "request", Work: &ThreadWork{ID: "task", Revision: 1, Origin: rt.threadID, Worker: "worker", Kind: "progress", Status: "completion_requested"}, Text: "Au revoir."})
	if !rt.completionRequested() {
		t.Fatal("explicit completion was not routed to conversation")
	}
	select {
	case <-rt.backend.worker.Thinker.quit:
		t.Fatal("completion notification stopped child before finishing phase")
	default:
	}
}

func TestManagedWorkerDuplicateOldDeliveryDoesNotRewindAdmission(t *testing.T) {
	parent := newTestThinkerFull()
	parent.registry = NewToolRegistry("")
	parent.session = NewSession(t.TempDir(), "main")
	defer parent.Stop()
	if err := parent.threads.SpawnWithOpts("worker", "Execute", nil, SpawnOpts{WorkMode: true, DeferRun: true}); err != nil {
		t.Fatal(err)
	}
	defer parent.threads.KillAll()
	old := ThreadWork{ID: "old", Revision: 1, Origin: "main", Kind: "request"}
	newer := ThreadWork{ID: "new", Revision: 1, Origin: "main", Kind: "request"}
	for _, w := range []ThreadWork{old, newer, old} {
		if err := parent.threads.QueueWork("worker", w, "same text", nil); err != nil {
			t.Fatal(err)
		}
	}
	if !parent.threads.threads["worker"].Thinker.work.current(newer) {
		t.Fatal("duplicate old delivery rewound active assignment")
	}
}

func TestLiveManagedReconnectSeedsCompletedEvidenceWithoutOldCallOutputs(t *testing.T) {
	s := managedSessionFixture()
	managedEvent(s, `{"type":"response.created","response":{"id":"new"}}`)
	managedEvent(s, `{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Already confirmed"}]}}`)
	managedEvent(s, `{"type":"response.completed","response":{"status":"completed"}}`)
	m := []Message{{Role: "system", Content: "instructions"}, {Role: "assistant", ToolCalls: []NativeToolCall{{ID: "previous-socket-call", Name: "book"}}}, {Role: "user", ToolResults: []ToolResult{{CallID: "previous-socket-call", Content: `{"booked":true}`}}}, {Role: "assistant", Content: "Booking confirmed"}, {Role: "user", Content: "Repeat the confirmation"}}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := s.Execute(ctx, m, "model", nil); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, w := range drainManagedWrites(s) {
		if w["type"] != "response.item.create" {
			continue
		}
		item := w["item"].(map[string]any)
		if item["type"] == "function_call_output" {
			t.Fatal("replayed old call output into a new session")
		}
		data, _ := json.Marshal(item)
		found = found || strings.Contains(string(data), "booked")
	}
	if !found {
		t.Fatal("replacement backend lost completed action evidence")
	}
}

func TestRealtimeExecutionRejectsUnsupportedPerThreadMode(t *testing.T) {
	rt, _ := completionFixture(t)
	rt.capabilities = RealtimeCapabilities{AudioTruncation: true}
	rt.realtimeExecution = &RealtimeExecutionConfig{Mode: "provider"}
	if err := rt.configureExecution(); err == nil {
		t.Fatal("unsupported managed mode silently fell back to direct tools")
	}
}

func TestManagedApplicationResultRequestsSpeechWithoutDoubleAnnouncingDelegations(t *testing.T) {
	for _, id := range []string{"", "provider-task"} {
		t.Run(id, func(t *testing.T) {
			rt, s := completionFixture(t)
			rt.capabilities = RealtimeCapabilities{CoreExecution: true}
			rt.backend.mode = "provider"
			rt.backend.delegation = id
			rt.backend.requestID = "task"
			rt.backend.revision = 1
			rt.backend.worker = &Thread{ID: "worker"}
			rt.consumeThreadWork(Event{Text: "Confirmed", Work: &ThreadWork{ID: "task", Revision: 1, Origin: rt.threadID, Worker: "worker", Kind: "result", Status: "completed"}})
			expected := 0
			if id == "" {
				expected = 1
			}
			if len(s.texts) != expected {
				t.Fatalf("speakable updates=%d expected=%d", len(s.texts), expected)
			}
		})
	}
}
