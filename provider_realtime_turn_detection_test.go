package core

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRealtimeTurnDetectionProfilesResolveDeterministically(t *testing.T) {
	defaults, err := (RealtimeTurnDetectionConfig{}).normalized()
	if err != nil {
		t.Fatal(err)
	}
	defaults = defaults.resolved()
	if defaults.Profile != RealtimeTurnProfileDefault ||
		defaults.StartSensitivity != "" || defaults.PrefixPaddingMS != 0 ||
		defaults.EndSensitivity != "" || defaults.SilenceDurationMS != 0 ||
		defaults.Interruption != "" {
		t.Fatalf("default profile must preserve provider defaults: %#v", defaults)
	}

	telephony, err := (RealtimeTurnDetectionConfig{Profile: " TELEPHONY "}).normalized()
	if err != nil {
		t.Fatal(err)
	}
	telephony = telephony.resolved()
	if telephony.Profile != RealtimeTurnProfileTelephony ||
		telephony.StartSensitivity != RealtimeSensitivityLow ||
		telephony.PrefixPaddingMS != 300 ||
		telephony.EndSensitivity != RealtimeSensitivityLow ||
		telephony.SilenceDurationMS != 750 ||
		telephony.Interruption != RealtimeInterruptionAllow {
		t.Fatalf("telephony profile = %#v", telephony)
	}

	overridden := (RealtimeTurnDetectionConfig{
		Profile:           RealtimeTurnProfileTelephony,
		StartSensitivity:  RealtimeSensitivityHigh,
		PrefixPaddingMS:   420,
		EndSensitivity:    RealtimeSensitivityHigh,
		SilenceDurationMS: 900,
		Interruption:      RealtimeInterruptionDisable,
	}).resolved()
	if overridden.StartSensitivity != RealtimeSensitivityHigh ||
		overridden.PrefixPaddingMS != 420 ||
		overridden.EndSensitivity != RealtimeSensitivityHigh ||
		overridden.SilenceDurationMS != 900 ||
		overridden.Interruption != RealtimeInterruptionDisable {
		t.Fatalf("explicit values did not override profile: %#v", overridden)
	}
}

func TestRealtimeTurnDetectionRejectsInvalidConfiguration(t *testing.T) {
	tests := []RealtimeTurnDetectionConfig{
		{Profile: "concert"},
		{StartSensitivity: "extreme"},
		{EndSensitivity: "extreme"},
		{Interruption: "sometimes"},
		{PrefixPaddingMS: -1},
		{SilenceDurationMS: 60_001},
	}
	for _, config := range tests {
		if _, err := config.normalized(); err == nil {
			t.Fatalf("invalid configuration accepted: %#v", config)
		}
	}
}

func TestGoogleLiveSetupMapsDefaultAndTelephonyTurnDetection(t *testing.T) {
	setupFor := func(config RealtimeTurnDetectionConfig) map[string]any {
		t.Helper()
		payload, err := buildGoogleLiveSetup(RealtimeSessionOpts{
			Model: "gemini-3.1-flash-live-preview", TurnDetection: config,
		}, "Kore")
		if err != nil {
			t.Fatal(err)
		}
		var envelope map[string]any
		if err := json.Unmarshal(payload, &envelope); err != nil {
			t.Fatal(err)
		}
		return envelope["setup"].(map[string]any)["realtimeInputConfig"].(map[string]any)
	}

	defaultInput := setupFor(RealtimeTurnDetectionConfig{})
	defaultAAD := defaultInput["automaticActivityDetection"].(map[string]any)
	if len(defaultAAD) != 1 || defaultAAD["disabled"] != false ||
		defaultInput["activityHandling"] != "START_OF_ACTIVITY_INTERRUPTS" {
		t.Fatalf("default setup changed provider behavior: %#v", defaultInput)
	}

	telephoneInput := setupFor(RealtimeTurnDetectionConfig{Profile: RealtimeTurnProfileTelephony})
	telephoneAAD := telephoneInput["automaticActivityDetection"].(map[string]any)
	if telephoneAAD["startOfSpeechSensitivity"] != "START_SENSITIVITY_LOW" ||
		telephoneAAD["prefixPaddingMs"] != float64(300) ||
		telephoneAAD["endOfSpeechSensitivity"] != "END_SENSITIVITY_LOW" ||
		telephoneAAD["silenceDurationMs"] != float64(750) {
		t.Fatalf("telephony Gemini VAD = %#v", telephoneAAD)
	}

	noInterruption := setupFor(RealtimeTurnDetectionConfig{Interruption: RealtimeInterruptionDisable})
	if noInterruption["activityHandling"] != "NO_INTERRUPTION" {
		t.Fatalf("Gemini no-interruption mapping = %#v", noInterruption)
	}
}

func TestOpenAIRealtimeMapsSupportedTurnDetectionFields(t *testing.T) {
	payload, err := buildSessionUpdate(RealtimeSessionOpts{
		Model: "gpt-realtime-2.1",
		TurnDetection: RealtimeTurnDetectionConfig{
			Profile:      RealtimeTurnProfileTelephony,
			Interruption: RealtimeInterruptionDisable,
		},
	}, "marin")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	session := envelope["session"].(map[string]any)
	input := session["audio"].(map[string]any)["input"].(map[string]any)
	turnDetection := input["turn_detection"].(map[string]any)
	if turnDetection["prefix_padding_ms"] != float64(300) ||
		turnDetection["silence_duration_ms"] != float64(750) ||
		turnDetection["interrupt_response"] != false {
		t.Fatalf("OpenAI turn detection = %#v", turnDetection)
	}
}

func TestXAIRealtimeMapsSupportedTurnDetectionFields(t *testing.T) {
	payload, err := buildXAISessionUpdate(RealtimeSessionOpts{
		Model: "grok-voice-latest",
		TurnDetection: RealtimeTurnDetectionConfig{
			Profile: RealtimeTurnProfileTelephony,
		},
	}, "eve")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(payload, &envelope); err != nil {
		t.Fatal(err)
	}
	session := envelope["session"].(map[string]any)
	turnDetection := session["turn_detection"].(map[string]any)
	if turnDetection["prefix_padding_ms"] != float64(300) ||
		turnDetection["silence_duration_ms"] != float64(750) {
		t.Fatalf("xAI turn detection = %#v", turnDetection)
	}
}

func TestRealtimeTurnDetectionTelemetryIsEffectiveAndProviderNeutral(t *testing.T) {
	data := (RealtimeTurnDetectionConfig{Profile: RealtimeTurnProfileTelephony}).telemetryData()
	encoded, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	text := string(encoded)
	for _, expected := range []string{
		`"profile":"telephony"`,
		`"start_sensitivity":"low"`,
		`"prefix_padding_ms":300`,
		`"silence_duration_ms":750`,
		`"interruption":"interrupt"`,
	} {
		if !strings.Contains(text, expected) {
			t.Fatalf("telemetry missing %s: %s", expected, text)
		}
	}
}
