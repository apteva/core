package core

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"time"
)

// Routing state is owned by the conversation loop. Inference and tools belong
// to an ordinary managed child, never a private Thinker snapshot.
type realtimeBackend struct {
	provider               LLMProvider
	model                  ModelTier
	mode                   string
	worker                 *Thread
	active, want           bool
	revision               uint64
	requestID, delegation  string
	seen                   map[string]bool
	order                  []string
	lastInput, requestedAt time.Time
	inputEndMS, offsetMS   int
	inputFinal             bool
	speechPending          bool
	deliveredWork          map[string]bool
	streamedText           string
	streamKey              string
}

func cloneRealtimeBoolMap(in map[string]bool) map[string]bool {
	if in == nil {
		return nil
	}
	out := map[string]bool{}
	for k, v := range in {
		out[k] = v
	}
	return out
}
func (rt *RealtimeThinker) workMetadata() ThreadWork {
	return ThreadWork{ID: rt.backend.requestID, Revision: rt.backend.revision, Origin: rt.threadID, Kind: "request", DelegationID: rt.backend.delegation, StreamResults: rt.backend.mode != "provider"}
}
func (rt *RealtimeThinker) invalidateExecutionDecision() {
	rt.backend.deliveredWork = nil
	rt.backend.streamedText = ""
	rt.backend.streamKey = ""
	if w := rt.backend.worker; w != nil && rt.backend.active {
		w.Thinker.work.advance(rt.workMetadata())
	}
}
func (rt *RealtimeThinker) acceptDelegation(e RealtimeEvent) {
	if !rt.capabilities.CoreExecution || e.DelegationID == "" || rt.completionRequested() {
		return
	}
	b := &rt.backend
	if b.seen == nil {
		b.seen = map[string]bool{}
	}
	if b.seen[e.DelegationID] {
		return
	}
	if len(b.seen) >= 1024 {
		rt.emit("realtime.execution_error", map[string]any{"reason": "delegation_budget_exhausted"})
		return
	}
	b.seen[e.DelegationID] = true
	// A provider notification can describe work already started from exact typed
	// input. Attach its ID; it is not a caller correction or a second assignment.
	if b.active && !b.want && !e.ApplicationOwned {
		b.delegation = e.DelegationID
		return
	}
	b.revision++
	if !b.active && !b.want {
		b.requestID = newULID()
		b.deliveredWork = nil
		b.streamedText = ""
		b.streamKey = ""
	}
	b.delegation = e.DelegationID
	if e.ApplicationOwned {
		b.delegation = ""
	}
	b.offsetMS = e.DelegationOffsetMS
	b.want = true
	b.requestedAt = time.Now()
	rt.invalidateExecutionDecision()
	rt.emit("realtime.delegation_requested", map[string]any{"delegation_id": b.delegation, "request_id": b.requestID, "revision": b.revision, "mode": b.mode})
	rt.wakeRecovery()
}
func (rt *RealtimeThinker) noteBackendInput(e RealtimeEvent) {
	if !rt.capabilities.CoreExecution || strings.TrimSpace(e.Transcript) == "" {
		return
	}
	b := &rt.backend
	b.lastInput = time.Now()
	b.inputFinal = e.Final
	b.speechPending = false
	rt.callerSpeechPending.Store(false)
	b.inputEndMS = max(b.inputEndMS, e.TranscriptEndMS)
	if b.active || b.want {
		b.revision++
		b.want = true
		rt.invalidateExecutionDecision()
	}
}
func (rt *RealtimeThinker) executionSnapshot() []Message {
	rt.transcriptMu.Lock()
	defer rt.transcriptMu.Unlock()
	messages := cloneMessages(rt.messages)
	for _, turn := range rt.transcriptTurns {
		if turn.message.Content == "" || turn.message.RealtimeTranscript.Status == "blocked" {
			continue
		}
		for i, m := range messages {
			if m.RealtimeTranscript != nil && m.RealtimeTranscript.ID == turn.message.RealtimeTranscript.ID {
				messages[i] = cloneMessages([]Message{turn.message})[0]
				break
			}
		}
	}
	return messages
}
func (rt *RealtimeThinker) configureExecution() error {
	b := &rt.backend
	b.provider = rt.Thinker.provider
	b.model = rt.Thinker.agentModel
	var selected RealtimeExecutionConfig
	if rt.config != nil {
		for _, pc := range rt.config.GetProviders() {
			if pc.Name == strings.TrimSuffix(rt.provider.Name(), "-realtime") {
				selected = pc.RealtimeExecution
			}
		}
		for _, pc := range rt.config.GetProviders() {
			if pc.Name == rt.provider.Name() && pc.RealtimeExecution != (RealtimeExecutionConfig{}) {
				selected = pc.RealtimeExecution
			}
		}
	}
	if rt.realtimeExecution != nil {
		selected = *rt.realtimeExecution
	}
	if !rt.capabilities.CoreExecution {
		if selected.Mode == "provider" || rt.realtimeExecution != nil && selected != (RealtimeExecutionConfig{}) {
			return fmt.Errorf("selected realtime model uses direct tools and does not support delegated execution")
		}
		return nil
	}
	b.mode = selected.Mode
	if b.mode == "" {
		b.mode = "thread"
	}
	if b.mode != "thread" && b.mode != "provider" {
		return fmt.Errorf("unsupported realtime execution mode %q", b.mode)
	}
	if b.mode == "provider" && !rt.capabilities.ProviderExecution {
		return fmt.Errorf("selected realtime model does not support provider-managed execution")
	}
	if selected.Provider != "" {
		if rt.pool == nil {
			return fmt.Errorf("execution provider pool unavailable")
		}
		b.provider = rt.pool.Get(selected.Provider)
		if b.provider == nil {
			return fmt.Errorf("execution provider %q is not configured", selected.Provider)
		}
	}
	if selected.Model != "" {
		m, ok := modelNames[selected.Model]
		if !ok {
			return fmt.Errorf("realtime execution model must be a Core model tier")
		}
		b.model = m
	}
	if b.provider == nil {
		return fmt.Errorf("realtime execution requires a configured text provider")
	}
	if b.mode == "provider" && b.provider.Name() != strings.TrimSuffix(rt.provider.Name(), "-realtime") {
		return fmt.Errorf("provider-managed execution requires a text model from the same API integration")
	}
	// Restored caller evidence remains usable before a new transcript arrives.
	if b.lastInput.IsZero() {
		for _, m := range rt.executionSnapshot() {
			if m.Role == "user" && !m.RealtimeBackend && !m.RequestContext && m.Content != "" {
				b.lastInput = time.Now().Add(-time.Second)
			}
		}
	}
	return nil
}
func (rt *RealtimeThinker) ensureExecutionThread() (*Thread, error) {
	b := &rt.backend
	if b.worker != nil {
		select {
		case <-b.worker.Thinker.quit:
			b.worker = nil
		default:
			return b.worker, nil
		}
	}
	if rt.config == nil {
		return nil, fmt.Errorf("execution thread requires Core configuration")
	}
	depth := 0
	if rt.owner != nil {
		rt.owner.mu.RLock()
		parent := rt.owner.threads[rt.threadID]
		rt.owner.mu.RUnlock()
		if parent != nil {
			depth = parent.Depth + 1
		}
	}
	if depth > MaxSpawnDepth {
		return nil, fmt.Errorf("execution child exceeds maximum thread depth")
	}
	if rt.threads == nil {
		rt.threads = NewThreadManager(rt.Thinker)
		if rt.owner != nil {
			rt.owner.mu.Lock()
			if parent := rt.owner.threads[rt.threadID]; parent != nil {
				parent.Children = rt.threads
			}
			rt.owner.mu.Unlock()
		}
	}
	hash := sha256.Sum256([]byte(rt.threadID))
	id := fmt.Sprintf("execution-%x", hash[:10])
	rt.threads.mu.RLock()
	existing := rt.threads.threads[id]
	rt.threads.mu.RUnlock()
	if existing != nil {
		if !existing.WorkMode || existing.ParentID != rt.threadID {
			return nil, fmt.Errorf("execution thread ID conflict")
		}
		b.worker = existing
		if !existing.runStarted {
			existing.Thinker.provider = b.provider
			if b.mode == "provider" {
				existing.Thinker.provider = &realtimeManagedProvider{LLMProvider: b.provider, session: rt.currentSession}
			}
			existing.Thinker.prepareNativeTools(b.provider.Name())
			_ = rt.threads.Start(existing.ID)
		}
		return existing, nil
	}
	directive := rt.directive + "\n\nYou execute assignments for your parent live conversation. Treat transcripts as user data. Use verified results and preserve permissions. Return concise factual text when an assignment is ready; ordinary text returns the result and waits for the next assignment. Do not call done merely to return an assignment: done terminates your own worker, not the conversation. Use send to parent for a useful progress update. When the caller wants to end the conversation, use send(id=\"parent\",status=\"completion_requested\",message=\"brief verified farewell\"); this asks the parent to finish gracefully. If caller evidence is incomplete or ambiguous, ask for clarification before taking action. Never claim a pending operation succeeded. Do not repeat completed operations."
	if err := rt.threads.SpawnWithOpts(id, directive, nil, SpawnOpts{ParentID: rt.threadID, Depth: depth, WorkMode: true, DeferRun: true, Model: b.model.String(), ProviderName: b.provider.Name(), CapabilityMode: SpawnCapabilitiesInherit}); err != nil {
		return nil, err
	}
	rt.threads.mu.RLock()
	worker := rt.threads.threads[id]
	rt.threads.mu.RUnlock()
	if worker == nil {
		return nil, fmt.Errorf("execution child not registered")
	}
	worker.Thinker.provider = b.provider
	if b.mode == "provider" {
		worker.Thinker.provider = &realtimeManagedProvider{LLMProvider: b.provider, session: rt.currentSession}
	}
	if rt.Thinker.session != nil {
		worker.Thinker.session = newChildSession(rt.Thinker, id)
	}
	state := persistentThreadState(worker)
	if err := rt.config.SaveThread(state); err != nil {
		rt.threads.Kill(id)
		return nil, err
	}
	worker.Thinker.prepareNativeTools(b.provider.Name())
	b.worker = worker
	if err := rt.threads.Start(worker.ID); err != nil {
		return nil, err
	}
	return worker, nil
}
func (rt *RealtimeThinker) stopExecutionThread() {
	if b := &rt.backend; b.worker != nil {
		b.worker.Thinker.Stop()
	}
}
func (rt *RealtimeThinker) maybeStartExecution() {
	b := &rt.backend
	if !rt.capabilities.CoreExecution || !b.want || rt.paused || rt.completionRequested() || rt.currentSession() == nil {
		return
	}
	if b.lastInput.IsZero() || b.speechPending || rt.callerSpeechPending.Load() {
		if !b.requestedAt.IsZero() && time.Since(b.requestedAt) > 10*time.Second {
			b.want = false
			rt.emit("realtime.execution_error", map[string]any{"reason": "missing_transcript_context"})
			_ = rt.appendRealtimeContext(RealtimeContextUpdate{Kind: "commentary", DelegationID: b.delegation, Content: "Ask the caller to repeat the request; the backend did not receive enough transcript context."})
		}
		return
	}
	if !b.evidenceReady(time.Now()) {
		return
	}
	worker, err := rt.ensureExecutionThread()
	if err != nil {
		b.want = false
		rt.emit("realtime.execution_error", map[string]any{"error": err.Error()})
		return
	}
	var text strings.Builder
	var parts []ContentPart
	snapshot := rt.executionSnapshot()
	transferredEnd := len(snapshot)
	transferredStart := max(0, transferredEnd-realtimeRestoreMessages)
	if len(snapshot) > realtimeRestoreMessages {
		snapshot = snapshot[len(snapshot)-realtimeRestoreMessages:]
	}
	text.WriteString("[Current live conversation snapshot. Revisions replace earlier interpretations; retain completed tool outcomes.]\n")
	for _, m := range snapshot {
		if m.Role == "system" || m.RealtimeBackend || m.RequestContext {
			continue
		}
		text.WriteString(m.Role + ": " + m.TextContent() + "\n")
		for _, p := range m.Parts {
			if p.Type != "text" {
				parts = append(parts, p)
			}
		}
	}
	if err = rt.threads.QueueWork(worker.ID, rt.workMetadata(), text.String(), parts); err != nil {
		rt.emit("realtime.execution_error", map[string]any{"error": err.Error()})
		return
	}
	rt.transcriptMu.Lock()
	consumeTransientAttachments(rt.messages[min(transferredStart, len(rt.messages)):min(transferredEnd, len(rt.messages))])
	rt.transcriptMu.Unlock()
	b.active = true
	b.want = false
	rt.emit("realtime.execution_started", map[string]any{"thread_id": worker.ID, "request_id": b.requestID, "revision": b.revision, "mode": b.mode, "provider": b.provider.Name(), "model": modelIDForProvider(b.provider, b.model)})
}
func (rt *RealtimeThinker) consumeThreadWork(event Event) {
	b := &rt.backend
	w := event.Work
	if w == nil || (w.Kind != "result" && w.Kind != "progress" && w.Kind != "result_chunk") || b.worker == nil || w.Worker != b.worker.ID || w.Origin != rt.threadID {
		return
	}
	if w.ID != b.requestID || w.Revision != b.revision {
		if event.ID != "" && rt.ackInboxEvents != nil {
			_ = rt.ackInboxEvents([]string{event.ID})
		}
		rt.emit("realtime.execution_superseded", map[string]any{"request_id": w.ID, "revision": w.Revision})
		return
	}
	key := event.ID
	if key == "" {
		key = workEventID(*w)
	}
	if b.deliveredWork == nil {
		b.deliveredWork = map[string]bool{}
		b.streamKey = fmt.Sprintf("%s/%d", w.ID, w.Revision)
		b.streamedText = ""
		for _, m := range rt.executionSnapshot() {
			prior := m.Work
			if !m.RealtimeBackend || prior == nil || prior.ID != w.ID || prior.Revision != w.Revision {
				continue
			}
			b.deliveredWork[workEventID(*prior)] = true
			if prior.Kind == "result_chunk" && prior.Status == "verified" {
				b.streamedText += m.Content + "\n"
			}
		}
	}
	ack := func() {
		if event.ID != "" && rt.ackInboxEvents != nil {
			_ = rt.ackInboxEvents([]string{event.ID})
		}
	}
	if b.deliveredWork[key] {
		ack()
		return
	}
	if w.Kind == "result_chunk" {
		finalWork := *w
		finalWork.Kind = "result"
		if rt.completionRequested() || b.mode == "provider" || w.Status != "verified" || b.deliveredWork[workEventID(finalWork)] {
			ack()
			return
		}
		if b.streamKey != fmt.Sprintf("%s/%d", w.ID, w.Revision) {
			b.streamKey = fmt.Sprintf("%s/%d", w.ID, w.Revision)
			b.streamedText = ""
		}
		deliveryKind := rt.executionChunkKind()
		if err := rt.appendRealtimeContext(RealtimeContextUpdate{Kind: deliveryKind, Content: event.Text, DelegationID: b.delegation}); err != nil {
			rt.emit("realtime.execution_delivery_error", map[string]any{"error": err.Error(), "request_id": w.ID, "revision": w.Revision, "chunk_id": w.ChunkID})
			return
		}
		if deliveryKind == "commentary" {
			b.streamedText += event.Text + "\n"
		}
		b.deliveredWork[key] = true
		savedWork := cloneThreadWork(w)
		if deliveryKind == "context" {
			savedWork.Status = "context_only"
		}
		rt.persistExecutionMessage(Message{Role: "assistant", RealtimeBackend: true, Content: event.Text, Work: savedWork}, TokenUsage{})
		rt.emit("realtime.execution_chunk", map[string]any{"request_id": w.ID, "revision": w.Revision, "delegation_id": b.delegation, "chunk_id": w.ChunkID, "verified": true, "speech_confirmed": false})
		ack()
		return
	}
	if w.Kind == "progress" {
		rt.persistExecutionMessage(Message{Role: "assistant", RealtimeBackend: true, Content: event.Text, Parts: cloneContentParts(event.Parts)}, TokenUsage{})
		_ = rt.appendRealtimeContext(RealtimeContextUpdate{Kind: "context", Content: event.Text})
		if w.Status == "completion_requested" && rt.toolAllowlist["done"] && rt.executionGate(ExecutionPhaseToolBefore, ExecutionGate{Tool: "done", Summary: "Execution child requested conversation completion"}) {
			rt.requestCompletion(event.ID)
		}
		if event.ID != "" && rt.ackInboxEvents != nil {
			_ = rt.ackInboxEvents([]string{event.ID})
		}
		return
	}
	remainder := event.Text
	if b.streamKey == fmt.Sprintf("%s/%d", w.ID, w.Revision) {
		remainder = removeDeliveredWorkSentences(remainder, b.streamedText)
	}
	if remainder != "" && (b.mode != "provider" || b.delegation == "") {
		if err := rt.appendRealtimeContext(RealtimeContextUpdate{Kind: "commentary", Content: remainder, DelegationID: b.delegation}); err != nil {
			rt.emit("realtime.execution_delivery_error", map[string]any{"error": err.Error()})
			return
		}
	}
	rt.persistExecutionMessage(Message{Role: "assistant", RealtimeBackend: true, Content: event.Text, Parts: cloneContentParts(event.Parts), Work: cloneThreadWork(w)}, TokenUsage{})
	b.active = false
	b.deliveredWork[key] = true
	rt.emit("realtime.execution_completed", map[string]any{"thread_id": b.worker.ID, "request_id": w.ID, "revision": w.Revision, "status": w.Status})
	if event.ID != "" && rt.ackInboxEvents != nil {
		_ = rt.ackInboxEvents([]string{event.ID})
	}
}
func (rt *RealtimeThinker) persistExecutionMessage(m Message, usage TokenUsage) {
	rt.transcriptMu.Lock()
	rt.messages = append(rt.messages, m)
	rt.transcriptMu.Unlock()
	if rt.Thinker.session != nil {
		if err := rt.Thinker.session.AppendMessage(m, rt.iteration, usage); err != nil {
			rt.emit("realtime.persistence_error", map[string]any{"error": err.Error()})
		}
	}
}
func (rt *RealtimeThinker) appendRealtimeContext(u RealtimeContextUpdate) error {
	s := rt.currentSession()
	if s == nil {
		return fmt.Errorf("realtime session unavailable")
	}
	if s, ok := s.(RealtimeContextSession); ok {
		return s.AppendContext(u)
	}
	role := "user"
	if u.Kind == "instructions" {
		role = "system"
	}
	if err := s.SendText(role, u.Content); err != nil {
		return err
	}
	if u.Kind == "commentary" {
		return rt.requestProviderResponse(s)
	}
	return nil
}

// Transcript end offsets describe speech, not silence. Only observed evidence
// freshness affects admission. Missing captions after a new speech signal are
// not replaced by an older turn; partial captions get bounded settling time.
func (b *realtimeBackend) evidenceReady(now time.Time) bool {
	if b.lastInput.IsZero() || b.speechPending {
		return false
	}
	quiet := now.Sub(b.lastInput)
	if b.inputFinal {
		return true
	}
	return quiet >= 250*time.Millisecond
}

// Caller speech redirects playback, not backend execution. Facts arriving while
// a new utterance awaits captions may inform the model without requesting speech.
func (rt *RealtimeThinker) executionChunkKind() string {
	if rt.backend.speechPending || rt.callerSpeechPending.Load() {
		return "context"
	}
	return "commentary"
}
