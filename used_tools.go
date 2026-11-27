package core

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// The main thread's working tools survive a Core restart. Loaded tools are
// otherwise held in memory only, so after an upgrade, crash or redeploy the
// agent started with a different set and had to search again for tools it
// used every hour, or went without them.
//
// Only names are saved, with the time of the last successful call. On
// restart each name is checked against the live catalog and the agent's
// current grants before it loads, so a saved name can never widen access.
// This is thread continuity (continue_tools_from) for the main thread, on
// disk.

const (
	usedToolsFile         = "main-tools.json"
	usedToolRetention     = 7 * 24 * time.Hour
	usedToolSaveLimit     = 64
	usedToolRestoreLimit  = 16
	usedToolResaveAfter   = 10 * time.Minute
	usedToolRestoreWindow = 10 * time.Minute
)

type usedToolStore struct {
	mu      sync.Mutex
	path    string
	tools   map[string]time.Time
	saved   map[string]time.Time
	pending []string // names still to restore, oldest use first
	started time.Time
}

type usedToolsFileData struct {
	Tools map[string]time.Time `json:"tools"`
}

// newUsedToolStore reads the saved names. A missing or damaged file is an
// empty store: the agent then starts as a fresh one would.
func newUsedToolStore(path string, now time.Time) *usedToolStore {
	s := &usedToolStore{path: path, tools: map[string]time.Time{}, saved: map[string]time.Time{}, started: now}
	raw, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	var data usedToolsFileData
	if json.Unmarshal(raw, &data) != nil {
		return s
	}
	for name, at := range data.Tools {
		if validToolName(name) && now.Sub(at) <= usedToolRetention {
			s.tools[name], s.saved[name] = at, at
		}
	}
	s.pending = s.recentLocked(usedToolRestoreLimit)
	return s
}

// recentLocked returns the most recently used names, oldest first, so that
// loading them in order leaves the latest one freshest for LRU eviction.
func (s *usedToolStore) recentLocked(limit int) []string {
	names := make([]string, 0, len(s.tools))
	for name := range s.tools {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if !s.tools[names[i]].Equal(s.tools[names[j]]) {
			return s.tools[names[i]].Before(s.tools[names[j]])
		}
		return names[i] < names[j]
	})
	if len(names) > limit {
		names = names[len(names)-limit:]
	}
	return names
}

// record notes a successful call. It writes the file when a name is new or
// its saved time is stale, not on every call.
func (s *usedToolStore) record(name string, now time.Time) {
	if s == nil || !validToolName(name) {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[name] = now
	if at, ok := s.saved[name]; ok && now.Sub(at) < usedToolResaveAfter {
		return
	}
	for other, at := range s.tools {
		if now.Sub(at) > usedToolRetention {
			delete(s.tools, other)
		}
	}
	if len(s.tools) > usedToolSaveLimit {
		keep := map[string]bool{}
		for _, kept := range s.recentLocked(usedToolSaveLimit) {
			keep[kept] = true
		}
		for other := range s.tools {
			if !keep[other] {
				delete(s.tools, other)
			}
		}
	}
	raw, err := json.Marshal(usedToolsFileData{Tools: s.tools})
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0700); err != nil {
		return
	}
	if atomicWriteFile(s.path, raw, 0600) != nil {
		return
	}
	s.saved = make(map[string]time.Time, len(s.tools))
	for other, at := range s.tools {
		s.saved[other] = at
	}
}

// takePending hands out the names not yet restored and keeps the ones the
// caller reports as still unknown to the catalog. MCP servers connect after
// start, so a name can be waiting for its server; it stops waiting after
// usedToolRestoreWindow.
func (s *usedToolStore) takePending(now time.Time, known func(string) bool) []string {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pending) == 0 {
		return nil
	}
	var ready, waiting []string
	for _, name := range s.pending {
		if known(name) {
			ready = append(ready, name)
		} else {
			waiting = append(waiting, name)
		}
	}
	if now.Sub(s.started) > usedToolRestoreWindow {
		waiting = nil
	}
	s.pending = waiting
	return ready
}

// restoreUsedTools loads the saved names this thread may still use. It runs
// on the thread's own loop, before the automatic selection, so tools the
// agent works with come back ahead of any directive match.
func (t *Thinker) restoreUsedTools() {
	if t == nil || t.usedTools == nil || t.toolIndex == nil || t.registry == nil {
		return
	}
	ready := t.usedTools.takePending(time.Now(), func(name string) bool {
		_, ok := t.toolIndex.Get(name)
		return ok
	})
	if len(ready) == 0 {
		return
	}
	loaded := t.continueActiveTools(ready)
	logMsg("TOOLS", continuityLogLine(t.threadID, "disk", loaded, len(ready)))
	if t.telemetry != nil {
		t.telemetry.Emit("tool.restored", t.threadID, map[string]any{"loaded": loaded, "offered": len(ready)})
	}
}

// clear forgets every saved name. A history reset removes the file with the
// rest of the agent's history; this drops the copy held in memory.
func (s *usedToolStore) clear() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools, s.saved, s.pending = map[string]time.Time{}, map[string]time.Time{}, nil
}
