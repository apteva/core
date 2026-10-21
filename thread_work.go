package core

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sync"
	"time"
)

// ThreadWork correlates an assignment and its revisions independently of a
// provider connection. IDs and statuses are runtime metadata, never model-authored.
type ThreadWork struct {
	ID       string `json:"id"`
	Revision uint64 `json:"revision"`
	Origin   string `json:"origin"`
	Worker   string `json:"worker,omitempty"`
	Kind     string `json:"kind"` // request, progress, result
	Status   string `json:"status,omitempty"`
}

func cloneThreadWork(w *ThreadWork) *ThreadWork {
	if w == nil {
		return nil
	}
	c := *w
	return &c
}

type threadWorkState struct {
	mu             sync.Mutex
	latest, active ThreadWork
	delivered      string
	instructions   string
	tools          []NativeTool
}

func (s *threadWorkState) advance(w ThreadWork) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if w.ID != s.latest.ID || w.Revision >= s.latest.Revision {
		s.latest = w
	}
}
func (s *threadWorkState) observe(w *ThreadWork) {
	if s == nil || w == nil || w.Kind != "request" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if w.ID != s.active.ID || w.Revision >= s.active.Revision {
		s.active = *w
	}
	if w.ID != s.latest.ID || w.Revision >= s.latest.Revision {
		s.latest = *w
	}
}
func (s *threadWorkState) snapshot() ThreadWork {
	if s == nil {
		return ThreadWork{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}
func (s *threadWorkState) current(w ThreadWork) bool {
	if s == nil {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return w.ID != "" && w.ID == s.latest.ID && w.Revision == s.latest.Revision
}

func workEventID(w ThreadWork) string { return fmt.Sprintf("work/%s/%d/%s", w.ID, w.Revision, w.Kind) }
func newThreadWorkEvent(w ThreadWork, text string, parts []ContentPart) PersistentThreadEvent {
	payload, _ := json.Marshal(struct {
		Text  string
		Parts []ContentPart
		Work  ThreadWork
	}{text, parts, w})
	return PersistentThreadEvent{ID: workEventID(w), Text: text, Parts: cloneContentParts(parts), Work: cloneThreadWork(&w), Hash: threadEventHash(string(payload), nil)}
}

// QueueWork uses the ordinary durable inbox. Updating the admission revision
// doesn't cancel inference; the owning loop checks it before dispatching tools.
func (tm *ThreadManager) QueueWork(id string, w ThreadWork, text string, parts []ContentPart) error {
	owner, _ := tm.findManagedThread(id)
	if owner == nil {
		return &threadNotFoundError{id: id}
	}
	owner.mu.RLock()
	thread := owner.threads[id]
	owner.mu.RUnlock()
	if thread == nil || thread.Thinker.work == nil {
		return fmt.Errorf("thread %q does not accept assignments", id)
	}
	if w.ID == "" || w.Origin != thread.ParentID || w.Kind != "request" {
		return fmt.Errorf("invalid thread assignment")
	}
	event := newThreadWorkEvent(w, text, parts)
	thread.inboxMu.Lock()
	for _, prior := range thread.inboxEvents {
		if prior.ID == event.ID {
			thread.inboxMu.Unlock()
			if prior.Hash != event.Hash {
				return &ThreadEventConflictError{ID: event.ID}
			}
			return nil
		}
	}
	thread.inboxMu.Unlock()
	// Invalidate stale decisions before the queued input can race admission.
	thread.Thinker.work.advance(w)
	_, err := owner.QueueEvents(id, []PersistentThreadEvent{event})
	return err
}

func (t *Thinker) publishThreadWork(w ThreadWork, text, status string, parts []ContentPart) error {
	if t.work == nil || !t.work.current(w) {
		return nil
	}
	w.Kind, w.Status, w.Worker = "result", status, t.threadID
	key := workEventID(w)
	t.work.mu.Lock()
	if t.work.delivered == key {
		t.work.mu.Unlock()
		return nil
	}
	t.work.mu.Unlock()
	e := newThreadWorkEvent(w, text, parts)
	var err error
	if t.owner != nil && t.owner.parent != nil {
		parent := t.owner.parent
		if parent.threadID == "main" {
			_, err = parent.QueueMainEvents([]PersistentThreadEvent{e})
		} else if parent.owner != nil {
			_, err = parent.owner.QueueEvents(parent.threadID, []PersistentThreadEvent{e})
		} else {
			parent.bus.Publish(Event{ID: e.ID, Type: EventInbox, From: t.threadID, To: parent.threadID, Text: text, Parts: parts, Work: &w})
		}
	}
	if err != nil {
		return err
	}
	t.work.mu.Lock()
	t.work.delivered = key
	t.work.mu.Unlock()
	// Assignment completion is a wait boundary, not thread termination.
	t.nextWakeAt = time.Time{}
	t.waitForEvents = true
	t.kickNextTurn = false
	t.paceDurable = true
	if t.persistPace != nil {
		err = t.persistPace(PersistentPaceState{WaitForEvents: true})
	}
	return err
}

func (t *Thinker) preserveSupersededWork(r ChatResponse) {
	if len(r.GeneratedFiles) == 0 && len(r.ServerResults) == 0 {
		return
	}
	evidence, _ := json.Marshal(r.ServerResults)
	m := Message{Role: "assistant", Content: "[Completed provider evidence from superseded work; do not repeat completed actions.] " + string(evidence), Parts: generatedFileParts(r.GeneratedFiles)}
	t.messages = append(t.messages, m)
	if t.session != nil {
		if err := t.session.AppendMessage(m, t.iteration, r.Usage); err != nil {
			t.recordPersistenceFailure(err)
		}
	}
	if t.telemetry != nil {
		t.telemetry.Emit("thread.work_superseded", t.threadID, map[string]any{"files": r.GeneratedFiles, "results": r.ServerResults})
	}
}

func newChildSession(parent *Thinker, id string) *Session {
	if parent != nil && parent.session != nil {
		return NewSession(filepath.Dir(filepath.Dir(parent.session.path)), id)
	}
	return NewSession(".", id)
}

func (s *threadWorkState) ready(w ThreadWork) bool {
	if s == nil {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	result := w
	result.Kind = "result"
	return w.ID != "" && w.ID == s.latest.ID && w.Revision == s.latest.Revision && s.delivered != workEventID(result)
}

func (t *Thinker) sendThreadWorkProgress(text, status string, parts []ContentPart) error {
	if t.work == nil || t.owner == nil {
		return fmt.Errorf("not an assigned worker")
	}
	w := t.work.snapshot()
	if !t.work.current(w) {
		return fmt.Errorf("assignment was superseded")
	}
	if status != "progress" && status != "completion_requested" {
		return fmt.Errorf("unsupported parent update status")
	}
	w.Kind, w.Status, w.Worker = "progress", status, t.threadID
	e := newThreadWorkEvent(w, text, parts)
	e.ID += "/" + newULID()
	parent := t.owner.parent
	if parent.threadID == "main" {
		_, err := parent.QueueMainEvents([]PersistentThreadEvent{e})
		return err
	}
	if parent.owner == nil {
		return fmt.Errorf("missing parent inbox")
	}
	_, err := parent.owner.QueueEvents(parent.threadID, []PersistentThreadEvent{e})
	return err
}

func (s *threadWorkState) setSetup(instructions string, tools []NativeTool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.instructions = instructions
	s.tools = append([]NativeTool(nil), tools...)
}
func (s *threadWorkState) setup() (string, []NativeTool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.instructions, append([]NativeTool(nil), s.tools...)
}
