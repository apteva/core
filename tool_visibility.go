package core

// toolAuthorizedFor evaluates a complete authorization state, including proposed
// grants during a configuration preview. Never consult the old grants there.
func (t *Thinker) toolAuthorizedFor(name string, grants, scopes map[string]bool) bool {
	if grants == nil {
		return true
	}
	entry, indexed := t.toolIndex.Get(name)
	if indexed && entry.NoSpawn && !t.allowNoSpawn {
		return false
	}
	return grants[name] || (indexed && scopes[entry.Server])
}

// visibleNativeToolSnapshot is shared by text requests and realtime session
// setup, refresh, and previews. It has no activation/LRU or manifest side effects.
// Loading policy grants visibility only within the supplied authorization scope.
func (t *Thinker) visibleNativeToolSnapshot(grants, scopes map[string]bool) ([]NativeTool, map[string]*ToolDef, map[string]bool) {
	if t == nil || t.registry == nil {
		return nil, nil, nil
	}
	var allowed map[string]bool
	if grants != nil {
		allowed = make(map[string]bool, len(grants))
		for name, enabled := range grants {
			if enabled && t.toolAuthorizedFor(name, grants, scopes) {
				allowed[name] = true
			}
		}
	}
	active := make(map[string]bool, len(t.activeTools))
	add := func(name string) {
		if t.toolAuthorizedFor(name, grants, scopes) {
			active[name] = true
		}
	}
	for name, enabled := range t.activeTools {
		if enabled {
			add(name)
		}
	}
	for _, name := range t.toolIndex.BaselineNames(t.useEagerTools(), t.threadID == "main" || t.allowNoSpawn) {
		add(name)
	}
	for name, until := range t.discoveredToolUntil {
		if t.iteration <= until {
			add(name)
		}
	}
	for _, action := range t.requiredActions(t.currentEventExecutions()) {
		if name := t.requiredToolNameFor(action, grants, scopes); name != "" {
			add(name)
		}
	}
	tools, definitions := t.registry.nativeToolSnapshot(allowed, active, t.systemThread)
	return tools, definitions, active
}
