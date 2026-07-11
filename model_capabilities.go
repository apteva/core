package core

import (
	"strings"
	"sync"
)

var runtimeModelCapabilities = struct {
	sync.RWMutex
	models map[string]ModelCapabilities
}{models: map[string]ModelCapabilities{}}

func resetRuntimeModelCapabilities() {
	runtimeModelCapabilities.Lock()
	runtimeModelCapabilities.models = map[string]ModelCapabilities{}
	runtimeModelCapabilities.Unlock()
}

func registerModelCapabilities(models map[string]ModelCapabilities) {
	runtimeModelCapabilities.Lock()
	defer runtimeModelCapabilities.Unlock()
	for id, caps := range models {
		runtimeModelCapabilities.models[strings.ToLower(strings.TrimSpace(id))] = caps
	}
}

func capabilitiesForModel(modelID string) (ModelCapabilities, bool) {
	runtimeModelCapabilities.RLock()
	defer runtimeModelCapabilities.RUnlock()
	caps, ok := runtimeModelCapabilities.models[strings.ToLower(strings.TrimSpace(modelID))]
	return caps, ok
}

// ModelEffectiveContextWindow applies the provider's safety percentage for
// compaction while ModelContextWindow continues to report the advertised max.
func ModelEffectiveContextWindow(modelID string) int {
	caps, ok := capabilitiesForModel(modelID)
	if !ok || caps.ContextWindow <= 0 {
		return ModelContextWindow(modelID)
	}
	percent := caps.EffectiveContextWindowPercent
	if percent <= 0 || percent > 100 {
		percent = 100
	}
	return caps.ContextWindow * percent / 100
}

func modelReasoningEffort(modelID, requested string) string {
	caps, ok := capabilitiesForModel(modelID)
	if !ok {
		return requested
	}
	return reasoningEffortForCapabilities(caps, requested)
}

func reasoningEffortForCapabilities(caps ModelCapabilities, requested string) string {
	if len(caps.SupportedReasoningLevels) == 0 || requested == "" {
		return requested
	}
	available := map[string]bool{}
	for _, level := range caps.SupportedReasoningLevels {
		available[strings.ToLower(strings.TrimSpace(level.Effort))] = true
	}
	requested = strings.ToLower(strings.TrimSpace(requested))
	if available[requested] {
		return requested
	}
	order := []string{"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
	requestedIndex := len(order) - 1
	for i, level := range order {
		if level == requested {
			requestedIndex = i
			break
		}
	}
	for i := requestedIndex; i >= 0; i-- {
		if available[order[i]] {
			return order[i]
		}
	}
	for _, level := range order {
		if available[level] {
			return level
		}
	}
	return requested
}
