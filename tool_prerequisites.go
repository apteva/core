package core

import (
	"encoding/json"
	"fmt"
	"strings"
)

// RequiredFirstAction is durable execution state, independent of model history
// and the LRU. The caller supplies a requirement, never an additional grant.
type RequiredFirstAction struct {
	Tool      string            `json:"tool"`
	Args      map[string]string `json:"args,omitempty"`
	Satisfied bool              `json:"satisfied,omitempty"`
}

func parseRequiredFirstAction(text string) *RequiredFirstAction {
	text = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(text), "[console]"))
	var event struct {
		First map[string]json.RawMessage `json:"required_first_action"`
	}
	if json.Unmarshal([]byte(text), &event) != nil || event.First == nil {
		return nil
	}
	var name string
	if json.Unmarshal(event.First["tool"], &name) != nil || strings.TrimSpace(name) == "" {
		return nil
	}
	action := &RequiredFirstAction{Tool: name, Args: map[string]string{}}
	values := event.First
	if args, ok := values["arguments"]; ok {
		var nested map[string]json.RawMessage
		if json.Unmarshal(args, &nested) != nil {
			return nil
		}
		values = nested
	}
	for key, raw := range values {
		if key == "tool" || key == "_reason" {
			continue
		}
		var value string
		if json.Unmarshal(raw, &value) != nil {
			value = string(raw)
		}
		action.Args[key] = value
	}
	return action
}

func (t *Thinker) requiredActions(ids []string) map[string]RequiredFirstAction {
	out := map[string]RequiredFirstAction{}
	if t.config == nil || len(ids) == 0 {
		return out
	}
	t.config.mu.RLock()
	defer t.config.mu.RUnlock()
	for _, ex := range t.config.EventExecutions {
		if containsString(ids, ex.ExecutionID) && !executionTerminal(ex.Status) && ex.RequiredFirstAction != nil {
			action := *ex.RequiredFirstAction
			action.Args = copyStringMap(action.Args)
			out[ex.ExecutionID] = action
		}
	}
	return out
}

func (t *Thinker) requiredToolName(action RequiredFirstAction) string {
	return t.requiredToolNameFor(action, t.toolAllowlist, t.toolMCPScopes)
}

func (t *Thinker) requiredToolNameFor(action RequiredFirstAction, grants, scopes map[string]bool) string {
	// Required actions accept only exact names or aliases. Authorize after the
	// index lookup so the predicate cannot recursively acquire the index lock.
	hits := t.toolIndex.Search(action.Tool, 1, t.threadID == "main" || t.allowNoSpawn)
	if len(hits) == 1 && t.toolAuthorizedFor(hits[0].Name, grants, scopes) && (strings.EqualFold(hits[0].Name, action.Tool) || t.toolIndex.isAlias(action.Tool, hits[0].Name)) {
		return hits[0].Name
	}
	return ""
}

func (ix *ToolIndex) isAlias(alias, name string) bool {
	if ix == nil {
		return false
	}
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.aliases[strings.ToLower(alias)] == name
}

func actionMatches(action RequiredFirstAction, name string, call toolCall) bool {
	if name == "" || name != call.Name {
		return false
	}
	for key, value := range action.Args {
		if call.Args[key] != value {
			return false
		}
	}
	return true
}

func (t *Thinker) requiredToolNames() map[string]bool {
	names := map[string]bool{}
	for _, action := range t.requiredActions(t.currentEventExecutions()) {
		if name := t.requiredToolName(action); name != "" {
			names[name] = true
		}
	}
	return names
}

func (t *Thinker) checkRequiredAction(call toolCall) error {
	actions, names := call.prerequisites, call.prerequisiteNames
	if actions == nil {
		actions = t.requiredActions(call.executionIDs)
		names = map[string]string{}
		for id, action := range actions {
			names[id] = t.requiredToolName(action)
		}
	}
	// Permit required actions to satisfy concurrent executions in either order.
	for id, action := range actions {
		if !action.Satisfied && actionMatches(action, names[id], call) {
			return nil
		}
	}
	for id, action := range actions {
		if action.Satisfied {
			continue
		}
		if names[id] == "" {
			return fmt.Errorf("capability_unavailable: required first action %s is not registered or authorized; do not substitute another operation", action.Tool)
		}
		return fmt.Errorf("prerequisite_required: call %s with the event's arguments and await success before dependent operations", action.Tool)
	}
	return nil
}

func (t *Thinker) satisfyRequiredAction(call toolCall) error {
	var satisfied []string
	actions, names := call.prerequisites, call.prerequisiteNames
	if actions == nil {
		actions = t.requiredActions(call.executionIDs)
		names = map[string]string{}
		for id, action := range actions {
			names[id] = t.requiredToolName(action)
		}
	}
	for id, action := range actions {
		if !action.Satisfied && actionMatches(action, names[id], call) {
			satisfied = append(satisfied, id)
		}
	}
	if len(satisfied) == 0 {
		return nil
	}
	return t.config.updateRuntime(func() {
		for i := range t.config.EventExecutions {
			ex := &t.config.EventExecutions[i]
			if containsString(satisfied, ex.ExecutionID) && ex.RequiredFirstAction != nil {
				copyAction := *ex.RequiredFirstAction
				copyAction.Satisfied = true
				ex.RequiredFirstAction = &copyAction
			}
		}
	})
}
