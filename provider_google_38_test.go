package core

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestGoogleLiveModelSpecificSetup(t *testing.T) {
	for _, tc := range []struct{ model, reasoning, wantLevel, behavior, version string }{
		{"gemini-3.1-flash-live-preview", "minimal", "minimal", "", "v1beta"},
		{"gemini-3.8-live", "high", "", "BLOCKING", "v1alpha"},
		{"models/gemini-3.8-live", "minimal", "", "BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "low", "low", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "medium", "medium", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "high", "high", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "minimal", "low", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "none", "low", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "xhigh", "high", "NON_BLOCKING", "v1alpha"},
		{"gemini-3.8-live-extended-thinking", "auto", "low", "NON_BLOCKING", "v1alpha"},
	} {
		t.Run(tc.model+"/"+tc.reasoning, func(t *testing.T) {
			tools := googleFullSchemaSmokeTools(t)
			before := string(mustJSON(t, tools))
			raw, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: tc.model, Reasoning: tc.reasoning, Tools: tools, RestoreHistory: true}, "Kore")
			if err != nil {
				t.Fatal(err)
			}
			var envelope map[string]any
			if err := json.Unmarshal(raw, &envelope); err != nil {
				t.Fatal(err)
			}
			setup := envelope["setup"].(map[string]any)
			thinking, ok := setup["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
			if tc.wantLevel == "" && ok {
				t.Fatalf("standard 3.8 must omit thinkingConfig: %s", raw)
			}
			if tc.wantLevel != "" && thinking["thinkingLevel"] != tc.wantLevel {
				t.Fatalf("thinking=%#v", thinking)
			}
			decls := setup["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)
			for i, value := range decls {
				decl := value.(map[string]any)
				behavior, _ := decl["behavior"].(string)
				if behavior != tc.behavior {
					t.Fatalf("tool behavior=%q want %q", behavior, tc.behavior)
				}
				if string(mustJSON(t, decl["parametersJsonSchema"])) != string(mustJSON(t, tools[i].Parameters)) {
					t.Fatal("schema changed")
				}
			}
			if string(mustJSON(t, tools)) != before {
				t.Fatal("mutated shared tools")
			}
			if !strings.Contains(googleLiveEndpoint(googleRealtimeEndpoint, tc.model), "."+tc.version+".") {
				t.Fatal("wrong websocket API version")
			}
			if googleLiveEndpoint("ws://localhost/test", tc.model) != "ws://localhost/test" {
				t.Fatal("overrode custom endpoint")
			}
			_, historyFlag := setup["historyConfig"]
			if historyFlag != (tc.version == "v1beta") {
				t.Fatal("legacy history-only setup flag leaked to 3.8")
			}
		})
	}
	if _, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "gemini-3.8-live-extended-thinking", Reasoning: "invalid"}, "Kore"); err == nil {
		t.Fatal("invalid reasoning accepted")
	}
}

func google38Session() *googleRealtimeSession {
	s := newGoogleRealtimeTestSession()
	s.profile = googleLiveProfileFor("gemini-3.8-live-extended-thinking")
	return s
}

func google38Translate(t *testing.T, s *googleRealtimeSession, raw string) []RealtimeEvent {
	t.Helper()
	var message googleLiveServerMessage
	if err := json.Unmarshal([]byte(raw), &message); err != nil {
		t.Fatal(err)
	}
	s.translate(&message)
	var events []RealtimeEvent
	for len(s.events) > 0 {
		events = append(events, <-s.events)
	}
	return events
}

func TestGoogleExtendedThinkingInteractionLifecycle(t *testing.T) {
	s := google38Session()
	if got := google38Translate(t, s, `{"interactionStatus":"IDLE","serverContent":{"turnComplete":true}}`); len(got) != 0 {
		t.Fatalf("initial idle emitted events: %#v", got)
	}
	first := google38Translate(t, s, `{"serverContent":{"outputTranscription":{"text":"Checking now."},"turnComplete":true,"interactionStatus":"IN_PROGRESS"},"usageMetadata":{"responseTokenCount":3,"thoughtsTokenCount":2,"responseTokensDetails":[{"modality":"TEXT","tokenCount":3}]}}`)
	var responseID, firstItem string
	for _, event := range first {
		if event.Type == RealtimeEventResponseStarted {
			responseID = event.ResponseID
		}
		if event.Type == RealtimeEventTranscriptOutput && event.Final {
			firstItem = event.ItemID
		}
		if event.Type == RealtimeEventResponseDone {
			t.Fatal("progress utterance completed the interaction")
		}
	}
	if responseID == "" || firstItem == "" || !s.activeResponse() {
		t.Fatalf("missing correlation: %#v", first)
	}
	calls := google38Translate(t, s, `{"interactionStatus":"IN_PROGRESS","toolCall":{"functionCalls":[{"id":"a","name":"first","args":{}},{"id":"b","name":"second","args":{}}]}}`)
	if len(calls) != 2 {
		t.Fatalf("tool calls ended interaction: %#v", calls)
	}
	for _, call := range calls {
		if call.Type != RealtimeEventToolCall || call.ResponseID != responseID {
			t.Fatalf("bad tool event: %#v", call)
		}
	}
	// Fast result must reach the wire while the slow call and interaction run.
	if err := s.SendToolResult("a", "fast", false); err != nil {
		t.Fatal(err)
	}
	if len(s.outbox) != 1 || len(s.pendingResponses) != 0 {
		t.Fatal("async result waits for completion")
	}
	frame := <-s.outbox
	if strings.Contains(string(frame.data), "scheduling") || !strings.Contains(string(frame.data), `"id":"a"`) {
		t.Fatalf("bad async response: %s", frame.data)
	}
	if err := s.SendToolResult("b", "slow", false); err != nil {
		t.Fatal(err)
	}
	<-s.outbox
	if err := s.RequestResponse(); err != nil || len(s.outbox) != 0 {
		t.Fatal("duplicate continuation")
	}
	final := google38Translate(t, s, `{"interactionStatus":"IDLE","serverContent":{"outputTranscription":{"text":"All done."},"turnComplete":true}}`)
	doneCount := 0
	for _, event := range final {
		if event.ResponseID != responseID {
			t.Fatalf("interaction id changed: %#v", event)
		}
		if event.Type == RealtimeEventTranscriptOutput && event.Final && (event.ItemID == firstItem || event.Transcript != "All done.") {
			t.Fatalf("utterances merged: %#v", event)
		}
		if event.Type == RealtimeEventResponseDone {
			doneCount++
			if event.Usage.TextOutputTokens != 5 {
				t.Fatalf("lost usage at utterance boundary: %#v", event.Usage)
			}
		}
	}
	if doneCount != 1 || s.activeResponse() {
		t.Fatal("IDLE did not finish exactly once")
	}
	if len(google38Translate(t, s, `{"interactionStatus":"IDLE"}`)) != 0 {
		t.Fatal("duplicate idle emitted completion")
	}
}

func TestGoogleExtendedThinkingStatusOnlyIdleAndInterruption(t *testing.T) {
	s := google38Session()
	google38Translate(t, s, `{"interactionStatus":"IN_PROGRESS","serverContent":{"outputTranscription":{"text":"Old speech"}}}`)
	events := google38Translate(t, s, `{"serverContent":{"interrupted":true}}`)
	if len(events) != 1 || events[0].Type != RealtimeEventSpeechStarted {
		t.Fatalf("interruption=%#v", events)
	}
	events = google38Translate(t, s, `{"serverContent":{"outputTranscription":{"text":"New speech."},"turnComplete":true,"interactionStatus":"IN_PROGRESS"}}`)
	for _, event := range events {
		if event.Final && strings.Contains(event.Transcript, "Old") {
			t.Fatal("interrupted transcript leaked into next utterance")
		}
	}
	if err := s.UpdateConfiguration("new prompt", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case <-s.done:
		t.Fatal("restarted during interaction")
	default:
	}
	events = google38Translate(t, s, `{"serverContent":{"interactionStatus":"IDLE"}}`)
	if len(events) != 1 || events[0].Type != RealtimeEventResponseDone {
		t.Fatalf("status-only completion=%#v", events)
	}
	select {
	case <-s.done:
	default:
		t.Fatal("deferred restart did not happen at IDLE")
	}
}

func TestGoogleExtendedThinkingThinkerDeliversResultsWhileActive(t *testing.T) {
	s := google38Session()
	rt := newRealtimeThinker(context.Background(), newTestThinker(), NewGoogleRealtimeProvider("test"), "", nil, nil, nil)
	defer rt.cancel()
	rt.replaceSession(s)
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseStarted, ResponseID: "interaction"})
	for _, id := range []string{"a", "b"} {
		s.callNames[id] = "probe"
		rt.beginToolCall(RealtimeEvent{ResponseID: "interaction", ToolCallID: id, ToolName: "probe"}, s)
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventUtteranceDone, ResponseID: "interaction"})
	if !rt.responseInProgress() || rt.state != "working" {
		t.Fatal("progress utterance settled active work")
	}
	rt.submitToolResult(s, "a", "first", false)
	if len(s.outbox) != 1 || !rt.pendingToolWork() {
		t.Fatal("fast tool result blocked behind slow result")
	}
	rt.submitToolResult(s, "b", "second", false)
	if len(s.outbox) != 2 || rt.pendingToolWork() || !rt.responseInProgress() {
		t.Fatal("results did not complete independently of interaction")
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventUtteranceDone, ResponseID: "interaction"})
	if rt.state != "thinking" {
		t.Fatal("returned to listening before IDLE")
	}
	rt.handleSessionEvent(RealtimeEvent{Type: RealtimeEventResponseDone, ResponseID: "interaction"})
	if rt.responseInProgress() || rt.state != "listening" || len(s.outbox) != 2 {
		t.Fatal("IDLE did not settle without extra continuation")
	}
}

func TestGoogle38PricingAndThinkingUsage(t *testing.T) {
	p := NewGoogleRealtimeProvider("test")
	for _, model := range []string{"gemini-3.1-flash-live-preview", "gemini-3.8-live", "gemini-3.8-live-extended-thinking"} {
		want := RealtimePricing{TextInput: 0.75, TextOutput: 4.5, AudioInput: 3, AudioOutput: 12}
		if !reflect.DeepEqual(p.Pricing("models/"+model), want) {
			t.Fatalf("pricing %s=%#v", model, p.Pricing(model))
		}
		usage := googleRealtimeUsage(&googleLiveUsage{ThoughtsTokenCount: 100, ResponseTokensDetails: []googleLiveTokenDetail{{Modality: "AUDIO", TokenCount: 10}}})
		if usage.TextOutputTokens != 100 || usage.AudioOutputTokens != 10 {
			t.Fatalf("thinking usage=%#v", usage)
		}
	}
}

func TestGoogleExtendedThinkingAccumulatesUsageAcrossUtterances(t *testing.T) {
	s := google38Session()
	// Real Live frames carry separate, non-cumulative output counts: the
	// final utterance can have fewer tokens than its preceding progress speech.
	google38Translate(t, s, `{"serverContent":{"interactionStatus":"IN_PROGRESS","outputTranscription":{"text":"Checking now."},"turnComplete":true},"usageMetadata":{"promptTokenCount":3125,"responseTokenCount":33,"totalTokenCount":3158,"promptTokensDetails":[{"modality":"TEXT","tokenCount":2792},{"modality":"AUDIO","tokenCount":222}],"responseTokensDetails":[{"modality":"AUDIO","tokenCount":33}]}}`)
	// Snapshots before the next boundary replace one another, not add up.
	google38Translate(t, s, `{"usageMetadata":{"thoughtsTokenCount":100}}`)
	events := google38Translate(t, s, `{"serverContent":{"interactionStatus":"IDLE","outputTranscription":{"text":"Orange."},"turnComplete":true},"usageMetadata":{"promptTokenCount":5966,"responseTokenCount":38,"totalTokenCount":6736,"promptTokensDetails":[{"modality":"TEXT","tokenCount":5580},{"modality":"AUDIO","tokenCount":255}],"responseTokensDetails":[{"modality":"TEXT","tokenCount":13},{"modality":"AUDIO","tokenCount":25}],"thoughtsTokenCount":732}}`)
	var usage RealtimeUsage
	for _, event := range events {
		if event.Type == RealtimeEventResponseDone {
			usage = event.Usage
		}
	}
	want := RealtimeUsage{TotalTokens: 9894, InputTokens: 9091, OutputTokens: 71, TextInputTokens: 8372, AudioInputTokens: 477, TextOutputTokens: 745, AudioOutputTokens: 58}
	if usage != want {
		t.Fatalf("interaction usage = %#v, want %#v", usage, want)
	}
	if s.lastUsage != (RealtimeUsage{}) || s.interactionUsage != (RealtimeUsage{}) {
		t.Fatal("usage leaked into next interaction")
	}
	// A tool-only generation is also billable even without an utterance.
	google38Translate(t, s, `{"interactionStatus":"IN_PROGRESS","toolCall":{"functionCalls":[{"id":"a","name":"probe","args":{}}]},"usageMetadata":{"thoughtsTokenCount":7}}`)
	events = google38Translate(t, s, `{"interactionStatus":"IDLE","usageMetadata":{"thoughtsTokenCount":3}}`)
	if len(events) != 1 || events[0].Usage.TextOutputTokens != 10 {
		t.Fatalf("tool-selection usage lost: %#v", events)
	}
}
