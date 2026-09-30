package core

import (
	"fmt"
	"strings"
)

// Keep Live protocol changes model-scoped. Existing/pinned 3.1 sessions retain
// their endpoint, thinking settings, synchronous tools and turn lifecycle.
// https://ai.google.dev/gemini-api/docs/live-api/thinking
type googleLiveProfile struct {
	alpha        bool
	omitThinking bool
	asyncTools   bool
	toolBehavior string
}

func googleLiveProfileFor(model string) googleLiveProfile {
	switch strings.TrimPrefix(strings.TrimSpace(model), "models/") {
	case "gemini-3.8-live":
		// 3.8 defaults to asynchronous tools. Explicit blocking preserves
		// Core's existing batch/continuation contract for the standard model.
		return googleLiveProfile{alpha: true, omitThinking: true, toolBehavior: "BLOCKING"}
	case "gemini-3.8-live-extended-thinking":
		return googleLiveProfile{alpha: true, asyncTools: true, toolBehavior: "NON_BLOCKING"}
	default:
		return googleLiveProfile{}
	}
}

func (p googleLiveProfile) thinkingConfig(reasoning string) (map[string]any, error) {
	if p.omitThinking {
		return nil, nil
	}
	if !p.asyncTools {
		config := map[string]any{"includeThoughts": false}
		if level := googleLiveThinkingLevel(reasoning); level != "" {
			config["thinkingLevel"] = level
		}
		return config, nil
	}
	// The shared effort scale includes none/minimal/xhigh. Map them to the
	// nearest supported effort; never send MINIMAL to Extended Thinking.
	level := strings.ToLower(strings.TrimSpace(reasoning))
	switch level {
	case "", "auto", "none", "minimal":
		level = "low"
	case "xhigh":
		level = "high"
	case "low", "medium", "high":
	default:
		return nil, fmt.Errorf("google-realtime: unsupported Extended Thinking reasoning %q", reasoning)
	}
	return map[string]any{"thinkingLevel": level}, nil
}

func googleLiveEndpoint(endpoint, model string) string {
	if endpoint == googleRealtimeEndpoint && googleLiveProfileFor(model).alpha {
		return strings.Replace(endpoint, ".v1beta.", ".v1alpha.", 1)
	}
	return endpoint // Preserve explicit endpoint overrides, including local tests.
}

// Published rates per million tokens, including thinking in text output.
// https://ai.google.dev/gemini-api/docs/pricing (September 2026)
var googleLiveModelPricing = map[string]RealtimePricing{
	"gemini-3.1-flash-live-preview":     {TextInput: 0.75, TextOutput: 4.50, AudioInput: 3, AudioOutput: 12},
	"gemini-3.8-live":                   {TextInput: 0.75, TextOutput: 4.50, AudioInput: 3, AudioOutput: 12},
	"gemini-3.8-live-extended-thinking": {TextInput: 0.75, TextOutput: 4.50, AudioInput: 3, AudioOutput: 12},
}
