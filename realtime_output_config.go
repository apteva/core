package core

import (
	"fmt"
	"strings"
	"time"
)

// RealtimeOutputConfig is durable, provider-neutral intent. Zero values retain
// the existing provider policy; unsupported explicit safety/tool modes fail
// at session setup rather than silently weakening the requested policy.
type RealtimeOutputConfig struct {
	ToolMode    string                    `json:"tool_mode,omitempty"` // default, blocking, async
	SpeechGuard RealtimeSpeechGuardConfig `json:"speech_guard,omitempty"`
}

type RealtimeSpeechGuardConfig struct {
	Mode               string `json:"mode,omitempty"` // default, transcript_prefix
	PrefixBytes        int    `json:"prefix_bytes,omitempty"`
	MaxBufferedAudioMS int    `json:"max_buffered_audio_ms,omitempty"`
}

func (c RealtimeOutputConfig) normalized() (RealtimeOutputConfig, error) {
	c.ToolMode = strings.ToLower(strings.TrimSpace(c.ToolMode))
	c.SpeechGuard.Mode = strings.ToLower(strings.TrimSpace(c.SpeechGuard.Mode))
	switch c.ToolMode {
	case "", "default", "blocking", "async":
	default:
		return c, fmt.Errorf("realtime tool_mode must be default, blocking, or async")
	}
	switch c.SpeechGuard.Mode {
	case "", "default", "transcript_prefix":
	default:
		return c, fmt.Errorf("speech_guard mode must be default or transcript_prefix")
	}
	if c.SpeechGuard.PrefixBytes < 0 || c.SpeechGuard.PrefixBytes > 1024 {
		return c, fmt.Errorf("speech_guard prefix_bytes must be between 0 and 1024")
	}
	if c.SpeechGuard.MaxBufferedAudioMS < 0 || c.SpeechGuard.MaxBufferedAudioMS > 30000 {
		return c, fmt.Errorf("speech_guard max_buffered_audio_ms must be between 0 and 30000")
	}
	if c.ToolMode == "default" {
		c.ToolMode = ""
	}
	if c.SpeechGuard.Mode == "default" {
		c.SpeechGuard.Mode = ""
	}
	return c, nil
}

func (c RealtimeOutputConfig) isZero() bool {
	n, err := c.normalized()
	return err == nil && n == (RealtimeOutputConfig{})
}

func (c RealtimeSpeechGuardConfig) resolved() RealtimeSpeechGuardConfig {
	c.Mode = "transcript_prefix"
	if c.PrefixBytes == 0 {
		c.PrefixBytes = 32
	}
	if c.MaxBufferedAudioMS == 0 {
		c.MaxBufferedAudioMS = 5000
	}
	return c
}

func realtimeOutputConfigValue(c *RealtimeOutputConfig) RealtimeOutputConfig {
	if c == nil {
		return RealtimeOutputConfig{}
	}
	return *c
}

func cloneRealtimeOutputConfig(c *RealtimeOutputConfig) *RealtimeOutputConfig {
	if c == nil {
		return nil
	}
	result := *c
	return &result
}

// Settings reports fields actually sent by the adapter, not a claim that the
// remote service echoed or honored every field. No prompt, tools or credentials.
type realtimeSettingsSession interface{ RealtimeSettings() map[string]any }

type realtimeReadySession interface{ RealtimeReadyAt() time.Time }
