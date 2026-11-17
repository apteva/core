package core

import (
	"encoding/binary"
	"fmt"
	"strings"
)

// RealtimeMediaConfig is provider-neutral, immutable session intent. Omission
// retains PCM16/24kHz over the provider's existing transport. Explicit unsupported
// combinations fail setup; there is no hidden lossy fallback.
type RealtimeMediaConfig struct {
	Transport string      `json:"transport,omitempty"`
	Format    AudioFormat `json:"format,omitempty"`
}

func (m RealtimeMediaConfig) normalized() (RealtimeMediaConfig, error) {
	m.Transport = strings.ToLower(strings.TrimSpace(m.Transport))
	switch m.Transport {
	case "", "websocket", "webrtc":
	default:
		return m, fmt.Errorf("unsupported realtime media transport %q", m.Transport)
	}
	switch m.Format {
	case "", AudioPCM16, AudioG711ULaw, AudioG711ALaw:
	default:
		return m, fmt.Errorf("unsupported realtime audio format %q", m.Format)
	}
	return m, nil
}
func (m RealtimeMediaConfig) format() AudioFormat {
	if m.Format == "" {
		return AudioPCM16
	}
	return m.Format
}
func (m RealtimeMediaConfig) rate() int {
	if m.format() == AudioPCM16 {
		return 24000
	}
	return 8000
}
func mediaBytesPerSecond(format AudioFormat) int {
	if format == AudioG711ULaw || format == AudioG711ALaw {
		return 8000
	}
	return realtimePCMBytesPerSecond
}
func defaultMediaRate(n int) int {
	if n > 0 {
		return n
	}
	return realtimePCMBytesPerSecond
}
func (rt *RealtimeThinker) outputRate() int { return mediaBytesPerSecond(rt.opts.AudioOutFmt) }
func (rt *RealtimeThinker) configureMedia() {
	m := rt.opts.OutputConfig.Media
	rt.opts.AudioInFmt = m.format()
	rt.opts.AudioOutFmt = m.format()
	rt.opts.AudioInRate = m.rate()
	rt.opts.AudioOutRate = m.rate()
	rt.delivery.bytesPerSecond = rt.outputRate()
	rt.configureInputQueue()
}
func mediaSample(b byte, format AudioFormat) int {
	if format == AudioG711ULaw {
		u := ^b
		v := (int(u&15)*8 + 132) << ((u >> 4) & 7)
		if u&128 != 0 {
			return 132 - v
		}
		return v - 132
	}
	a := b ^ 0x55
	v := int(a&15) * 16
	exp := (a >> 4) & 7
	if exp == 0 {
		v += 8
	} else {
		v += 264
		v <<= exp - 1
	}
	if a&128 == 0 {
		return -v
	}
	return v
}
func mediaVoiced(b []byte, format AudioFormat) bool {
	if format == AudioG711ULaw || format == AudioG711ALaw {
		for _, v := range b {
			n := mediaSample(v, format)
			if n > 128 || n < -128 {
				return true
			}
		}
		return false
	}
	for i := 0; i+1 < len(b); i += 2 {
		n := int16(binary.LittleEndian.Uint16(b[i:]))
		if n > 128 || n < -128 {
			return true
		}
	}
	return false
}

type realtimeMediaProvider interface {
	ValidateRealtimeMedia(string, RealtimeMediaConfig) error
}

func (p *OpenAIRealtimeProvider) ValidateRealtimeMedia(model string, m RealtimeMediaConfig) error {
	if m == (RealtimeMediaConfig{}) {
		return nil
	}
	if !strings.HasPrefix(model, "gpt-live") {
		return fmt.Errorf("explicit media selection is not supported by this model")
	}
	_, err := m.normalized()
	return err
}

func (rt *RealtimeThinker) effectiveTurnDetectionTelemetry() map[string]any {
	if rt.capabilities.ContinuousConversation {
		owner := "provider"
		if !rt.capabilities.ProviderInterruption && rt.capabilities.RendererInterruption {
			owner = "renderer"
		}
		return map[string]any{"mode": "continuous", "provider_managed": true, "vad_overrides_applied": false, "interruption_owner": owner, "local_interruption_required": rt.capabilities.RendererInterruption && !rt.capabilities.ProviderInterruption}
	}
	return rt.opts.TurnDetection.telemetryData()
}
