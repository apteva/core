package core

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Thread tool continuity lets a replacement thread start with the MCP tools an
// earlier thread had loaded. A workflow engine that retires an idle worker and
// later starts the next one for the same job names the earlier thread in
// continue_tools_from; otherwise the new worker pays search_tools turns to
// rediscover tools the old one already had.
//
// Only names carry over. The new thread checks each one against its own
// grants, scopes and no_spawn rule, so continuity can never widen access.

const (
	retiredToolSetLimit = 256
	retiredToolSetTTL   = 24 * time.Hour
)

// retiredToolSets remembers the loaded tools of threads that have ended. It is
// shared by every thread of one Core process and lives in memory only.
type retiredToolSets struct {
	mu   sync.Mutex
	sets map[string]retiredToolSet
}

type retiredToolSet struct {
	names []string
	at    time.Time
}

func newRetiredToolSets() *retiredToolSets {
	return &retiredToolSets{sets: map[string]retiredToolSet{}}
}

func (r *retiredToolSets) record(threadID string, names []string, now time.Time) {
	if r == nil || threadID == "" || len(names) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, set := range r.sets {
		if now.Sub(set.at) > retiredToolSetTTL {
			delete(r.sets, id)
		}
	}
	if _, exists := r.sets[threadID]; !exists && len(r.sets) >= retiredToolSetLimit {
		oldest, oldestAt := "", now
		for id, set := range r.sets {
			if oldest == "" || set.at.Before(oldestAt) {
				oldest, oldestAt = id, set.at
			}
		}
		delete(r.sets, oldest)
	}
	r.sets[threadID] = retiredToolSet{names: append([]string(nil), names...), at: now}
}

func (r *retiredToolSets) get(threadID string, now time.Time) []string {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	set, ok := r.sets[threadID]
	if !ok || now.Sub(set.at) > retiredToolSetTTL {
		return nil
	}
	return append([]string(nil), set.names...)
}

// publishActiveTools snapshots this thread's loaded MCP tools so another
// goroutine can read them. activeTools itself belongs to the thread's loop.
func (t *Thinker) publishActiveTools() {
	if t == nil {
		return
	}
	names := make([]string, 0, len(t.activeTools))
	for name, active := range t.activeTools {
		if active {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	t.activeToolsSnapshot.Store(&names)
}

func (t *Thinker) loadedToolsSnapshot() []string {
	if t == nil {
		return nil
	}
	if names := t.activeToolsSnapshot.Load(); names != nil {
		return append([]string(nil), (*names)...)
	}
	return nil
}

// continuedTools returns the tools loaded by a live thread anywhere in this
// tree, or by a thread that ended within the retention window.
func (tm *ThreadManager) continuedTools(threadID string) []string {
	if tm == nil || threadID == "" {
		return nil
	}
	if _, thread := tm.findManagedThread(threadID); thread != nil && thread.Thinker != nil {
		return thread.Thinker.loadedToolsSnapshot()
	}
	return tm.parent.retiredTools.get(threadID, time.Now())
}

// continueActiveTools loads the named tools that this thread may use. It
// returns the names it loaded.
func (t *Thinker) continueActiveTools(names []string) []string {
	if t == nil || t.toolIndex == nil || t.registry == nil {
		return nil
	}
	var loaded []string
	for _, name := range names {
		if len(loaded) >= activeToolsCap {
			break
		}
		entry, ok := t.toolIndex.Get(name)
		if !ok || entry.NoSpawn && !t.allowNoSpawn || !t.toolAuthorized(name) {
			continue
		}
		def := t.registry.Get(name)
		if def == nil || !def.MCP || def.SystemOnly && !t.systemThread {
			continue
		}
		t.touchActiveTool(name)
		loaded = append(loaded, name)
	}
	if len(loaded) > 0 {
		t.publishActiveTools()
	}
	return loaded
}

func continuityLogLine(id, source string, loaded []string, offered int) string {
	return fmt.Sprintf("%s: continued %d of %d loaded tools from %q", id, len(loaded), offered, source)
}
