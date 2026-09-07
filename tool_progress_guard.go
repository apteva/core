package core

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
)

const maxRepeatedMCPFailures = 3

type mcpFailure struct {
	Target, Error string
	Count         int
}

// Group failures by operation and stable target. Narration and changing prose
// cannot disguise retrying the same invalid operation against the same entity.
// For tools without target fields, retain all arguments except observability.
func toolProgressKey(call toolCall) (string, string) {
	args := map[string]string{}
	target := map[string]string{}
	for k, v := range call.Args {
		if k == "_reason" || strings.HasPrefix(k, "_apteva_") {
			continue
		}
		args[k] = v
		if k == "id" || strings.HasSuffix(k, "_id") || k == "url" || k == "path" {
			target[k] = v
		}
	}
	if len(target) == 0 {
		target = args
	}
	raw, _ := json.Marshal(target)
	sum := sha256.Sum256(raw)
	key := fmt.Sprintf("%x", sum[:])
	return call.Name + ":" + key, key
}

func (t *Thinker) resetMCPFailures() {
	t.mcpFailureMu.Lock()
	defer t.mcpFailureMu.Unlock()
	t.mcpFailures = nil
}

func (t *Thinker) checkMCPProgress(call toolCall) error {
	key, _ := toolProgressKey(call)
	t.mcpFailureMu.Lock()
	defer t.mcpFailureMu.Unlock()
	if f := t.mcpFailures[key]; f.Count >= maxRepeatedMCPFailures {
		return fmt.Errorf("no_progress: %s repeatedly failed for this target. This operation is blocked until a successful operation on that target or a fresh external instruction. Inspect the failure or report the blocker; do not retry with different narration", call.Name)
	}
	return nil
}

func (t *Thinker) recordMCPProgress(call toolCall, response ToolResponse) {
	key, target := toolProgressKey(call)
	t.mcpFailureMu.Lock()
	defer t.mcpFailureMu.Unlock()
	if !response.IsError {
		for key, f := range t.mcpFailures {
			if f.Target == target {
				delete(t.mcpFailures, key)
			}
		}
		return
	}
	if t.mcpFailures == nil {
		t.mcpFailures = map[string]mcpFailure{}
	}
	if len(t.mcpFailures) >= 128 { // Bounded per-thread state; never grow with tool traffic.
		if _, ok := t.mcpFailures[key]; !ok {
			return
		}
	}
	previous := t.mcpFailures[key]
	if previous.Error != response.Text {
		previous = mcpFailure{Target: target, Error: response.Text}
	}
	previous.Count++
	t.mcpFailures[key] = previous
}
