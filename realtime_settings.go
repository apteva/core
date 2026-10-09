package core

import (
	"encoding/json"
	"fmt"
)

func validateCompatibleRealtimeOutput(opts RealtimeSessionOpts) error {
	c, err := opts.OutputConfig.normalized()
	if err != nil {
		return err
	}
	if c.ToolMode == "async" {
		return fmt.Errorf("this realtime adapter does not support async tools")
	}
	if c.SpeechGuard != (RealtimeSpeechGuardConfig{}) {
		return fmt.Errorf("this realtime adapter does not support a pre-audio transcript_prefix guard")
	}
	return nil
}

func realtimeRequestedSettings(opts RealtimeSessionOpts) map[string]any {
	return map[string]any{"model": opts.Model, "turn_detection": opts.TurnDetection,
		"output_config": opts.OutputConfig, "reasoning": opts.Reasoning,
		"input_transcription": opts.TranscribeInput, "transcription_model": opts.TranscriptionModel}
}

func googleRealtimeSettings(opts RealtimeSessionOpts, wire []byte) map[string]any {
	var envelope struct {
		Setup map[string]any `json:"setup"`
	}
	_ = json.Unmarshal(wire, &envelope)
	profile, _ := googleLiveProfileWithOptions(opts)
	generation, _ := envelope.Setup["generationConfig"].(map[string]any)
	ignored := map[string]string{}
	if profile.omitThinking && opts.Reasoning != "" && opts.Reasoning != "auto" {
		ignored["reasoning"] = "model requires thinkingConfig to be omitted"
	}
	if opts.TranscriptionModel != "" {
		ignored["transcription_model"] = "uses native transcription; no selectable transcription model"
	}
	mode := "blocking"
	var scheduling any
	if profile.asyncTools {
		mode = "async"
		if !profile.interactionStatus {
			scheduling = "when_idle"
		}
	}
	return map[string]any{
		"source": "accepted_setup", "model": opts.Model,
		"turn_detection": envelope.Setup["realtimeInputConfig"],
		"thinking":       generation["thinkingConfig"], "tool_mode": mode,
		"tool_result_scheduling": scheduling,
		"input_transcription":    opts.TranscribeInput, "output_transcription": true,
		"speech_guard":                    opts.OutputConfig.SpeechGuard.resolved(),
		"speech_guard_wall_wait_limit_ms": nil,
		"transcript_audio_ordering":       "unordered; final transcript before generationComplete/interrupted",
		"speech_end_events":               false, "ignored": ignored,
	}
}

func compatibleRealtimeSettings(opts RealtimeSessionOpts, provider string, wire []byte) map[string]any {
	var envelope struct {
		Session map[string]any `json:"session"`
	}
	_ = json.Unmarshal(wire, &envelope)
	audio, _ := envelope.Session["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	turn := input["turn_detection"]
	if turn == nil {
		turn = envelope.Session["turn_detection"]
	}
	ignored := map[string]string{}
	resolved := opts.TurnDetection.resolved()
	if resolved.StartSensitivity != "" && resolved.StartSensitivity != "default" {
		ignored["start_sensitivity"] = "not supported by adapter"
	}
	if resolved.EndSensitivity != "" && resolved.EndSensitivity != "default" {
		ignored["end_sensitivity"] = "not supported by adapter"
	}
	if provider == "xai-realtime" && resolved.Interruption != "" && resolved.Interruption != "default" {
		ignored["interruption"] = "not supported by adapter"
	}
	return map[string]any{"source": "sent_session_update", "model": opts.Model,
		"turn_detection": turn, "thinking": envelope.Session["reasoning"], "tool_mode": "blocking",
		"input_transcription": input["transcription"], "output_transcription": "provider_native",
		"speech_guard": "provider_stream_with_core_transcript_checks", "speech_guard_wall_wait_limit_ms": nil,
		"transcript_audio_ordering": "provider_stream", "speech_end_events": true, "ignored": ignored}
}

func mustRealtimeSettingsJSON(session map[string]any) []byte {
	data, _ := json.Marshal(map[string]any{"session": session})
	return data
}
