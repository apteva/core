package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"strings"
	"testing"
)

func computerScreenshotJPEG(t *testing.T, targetBytes int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 1280, 800))
	for y := 0; y < 800; y += 32 {
		for x := 0; x < 1280; x += 32 {
			c := color.RGBA{R: uint8(x / 8), G: uint8(y / 4), B: uint8((x + y) / 12), A: 255}
			for yy := y; yy < y+32; yy++ {
				for xx := x; xx < x+32; xx++ {
					img.SetRGBA(xx, yy, c)
				}
			}
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatalf("encode screenshot: %v", err)
	}
	data := buf.Bytes()
	if len(data) < targetBytes {
		data = append(data, make([]byte, targetBytes-len(data))...)
	}
	return data
}

func TestContextCharsCountsStructuredAndMultimodalPayloads(t *testing.T) {
	messages := []Message{{
		Role:      "assistant",
		Content:   "content",
		Reasoning: "reasoning",
		Parts: []ContentPart{
			{Type: "text", Text: "part text"},
			{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/png;base64,abcdef", Detail: "high"}},
			{Type: "input_audio", InputAudio: &InputAudio{Data: "audio-data", Format: "wav"}},
		},
		ToolCalls:   []NativeToolCall{{ID: "call", Name: "tool", Args: map[string]string{"large": "argument"}, ThoughtSignature: "signature"}},
		ToolResults: []ToolResult{{CallID: "call", Content: "result", Image: []byte("pixels")}},
	}}
	if got := contextChars(messages); got < 100 {
		t.Fatalf("contextChars = %d; structured payloads were not fully counted", got)
	}
}

type testCompactionProvider struct {
	response string
	err      error
	calls    int
	prompts  [][]Message
}

func (p *testCompactionProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(toolName, callID, chunk string)) (ChatResponse, error) {
	p.calls++
	p.prompts = append(p.prompts, append([]Message(nil), messages...))
	if p.err != nil {
		return ChatResponse{}, p.err
	}
	return ChatResponse{
		Text: p.response,
		Usage: TokenUsage{
			PromptTokens:     123,
			CompletionTokens: 45,
		},
	}, nil
}

func (p *testCompactionProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{
		ModelLarge:  "large-test",
		ModelMedium: "medium-test",
		ModelSmall:  "small-test",
	}
}

func (p *testCompactionProvider) Name() string { return "test-compactor" }
func (p *testCompactionProvider) CostPer1M() (float64, float64, float64) {
	return 0, 0, 0
}
func (p *testCompactionProvider) SupportsNativeTools() bool { return true }
func (p *testCompactionProvider) AvailableBuiltinTools() []BuiltinTool {
	return nil
}
func (p *testCompactionProvider) SetBuiltinTools(tools []string) {}
func (p *testCompactionProvider) WithBuiltins(builtins []string) LLMProvider {
	return p
}

func TestShouldCompactBeforeLLMWhenContextFallbackExceeded(t *testing.T) {
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+5; i++ {
		messages = append(messages, Message{Role: "user", Content: strings.Repeat("x", 24_000)})
	}
	if !shouldCompactBeforeLLM("unknown-model", messages) {
		t.Fatal("expected pre-LLM compaction when fallback context pressure is exceeded")
	}
}

func TestShouldCompactBeforeLLMFalseWhenNoCompactionRoom(t *testing.T) {
	messages := []Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: strings.Repeat("x", contextPressureCharFallback+1)},
	}
	if shouldCompactBeforeLLM("unknown-model", messages) {
		t.Fatal("did not expect pre-LLM compaction when there is no older context to remove")
	}
}

func TestShouldCompactBeforeLLMIgnoresCompressedScreenshotByteSize(t *testing.T) {
	messages := []Message{{Role: "system", Content: strings.Repeat("system", 1000)}}
	for i := 0; i < contextPressureKeepRecent+5; i++ {
		messages = append(messages, Message{Role: "user", Content: "ordinary conversation"})
	}
	for i, size := range []int{105_000, 120_000, 145_000, 163_000} {
		id := fmt.Sprintf("computer-%d", i)
		messages = append(messages,
			Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "computer_computer_use"}}},
			Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: strings.Repeat("som", 1600), Image: computerScreenshotJPEG(t, size)}}},
		)
	}

	if contextChars(messages) < semanticCompactionCharFallback {
		t.Fatalf("fixture raw chars = %d, want above old false-positive threshold", contextChars(messages))
	}
	if tokens := estimatedContextTokens(messages); tokens >= 40_000 {
		t.Fatalf("estimated tokens = %d, expected screenshot context well below pressure", tokens)
	}
	if shouldCompactBeforeLLM("kimi-k2.6", messages) {
		t.Fatal("compressed screenshot bytes triggered compaction for a known large-context model")
	}
	if shouldRecoverFromEmptyResponse(TokenUsage{}, "kimi-k2.6", messages, 1) {
		t.Fatal("compressed screenshot bytes triggered empty-response pressure recovery")
	}
}

func TestShouldCompactBeforeLLMStillDetectsKnownModelTextPressure(t *testing.T) {
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+10; i++ {
		messages = append(messages, Message{Role: "user", Content: strings.Repeat("text ", 8_000)})
	}
	if !shouldCompactBeforeLLM("kimi-k2.6", messages) {
		t.Fatal("genuine text token pressure should still trigger compaction")
	}
}

func TestEvictStaleComputerScreenshotsKeepsLatestFrame(t *testing.T) {
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < 3; i++ {
		id := fmt.Sprintf("call-%d", i)
		messages = append(messages,
			Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: id, Name: "computer_computer_use"}}},
			Message{Role: "user", ToolResults: []ToolResult{{CallID: id, Content: "screen", Image: computerScreenshotJPEG(t, 110_000+i*10_000)}}},
		)
	}

	if removed := evictStaleComputerScreenshots(messages, 1); removed != 2 {
		t.Fatalf("removed = %d, want 2", removed)
	}
	images := 0
	for _, msg := range messages {
		for _, result := range msg.ToolResults {
			if len(result.Image) > 0 {
				images++
				if result.CallID != "call-2" {
					t.Fatalf("retained stale screenshot %s", result.CallID)
				}
			}
		}
	}
	if images != 1 {
		t.Fatalf("retained images = %d, want 1", images)
	}
}

func TestMessageForSessionDropsComputerPixelsOnly(t *testing.T) {
	history := []Message{{Role: "assistant", ToolCalls: []NativeToolCall{
		{ID: "computer", Name: "computer_computer_use"},
		{ID: "generated", Name: "image_generate"},
	}}}
	msg := Message{Role: "user", ToolResults: []ToolResult{
		{CallID: "computer", Content: "screen metadata", Image: []byte("screen")},
		{CallID: "generated", Content: "generated image", Image: []byte("art")},
	}}
	persisted := messageForSession(history, msg)
	if persisted.ToolResults[0].Image != nil {
		t.Fatal("computer screenshot should not be persisted")
	}
	if string(persisted.ToolResults[1].Image) != "art" {
		t.Fatal("non-computer image should remain persisted")
	}
	if string(msg.ToolResults[0].Image) != "screen" {
		t.Fatal("durable-copy sanitization mutated the live screenshot")
	}
}

func TestShouldRecoverFromEmptyResponseNearContextLimit(t *testing.T) {
	messages := []Message{{Role: "system", Content: "small"}}
	if !shouldRecoverFromEmptyResponse(TokenUsage{PromptTokens: 205_000}, "kimi-k2.6", messages, 1) {
		t.Fatal("expected recovery near 256k context limit")
	}
}

func TestShouldRecoverFromEmptyResponseAfterRepeatedEmptyUnknownUsage(t *testing.T) {
	messages := []Message{{Role: "system", Content: "small"}}
	if shouldRecoverFromEmptyResponse(TokenUsage{}, "unknown-model", messages, 1) {
		t.Fatal("did not expect recovery on first empty response without pressure signal")
	}
	if !shouldRecoverFromEmptyResponse(TokenUsage{}, "unknown-model", messages, contextPressureEmptyStreak) {
		t.Fatal("expected recovery after repeated empty responses")
	}
}

func TestSessionForceCompactArchivesRecentOversizedToolResult(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	for i := 0; i < contextPressureKeepRecent+5; i++ {
		session.Append(SessionEntry{Role: "user", Content: "message"})
	}
	session.Append(SessionEntry{
		Role:      "assistant",
		ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}},
	})
	large := strings.Repeat("y", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("z", 16_000)
	session.Append(SessionEntry{
		Role: "user",
		ToolResults: []ToolResult{{
			CallID:  "call-1",
			Content: large,
		}},
	})

	session.ForceCompact(contextPressureKeepRecent, func(text string) string {
		return "pressure summary"
	})

	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) != contextPressureKeepRecent+1 {
		t.Fatalf("entry count = %d, want %d", len(entries), contextPressureKeepRecent+1)
	}
	last := entries[len(entries)-1]
	if len(last.ToolResults) != 1 {
		t.Fatalf("expected retained tool result, got %+v", last.ToolResults)
	}
	stored := last.ToolResults[0]
	if stored.Content != "" || stored.SHA256 == "" || stored.ArchiveRef == "" || stored.OriginalBytes != len(large) {
		t.Fatalf("tool result was not reduced to durable archive metadata: %+v", stored)
	}
	object, err := session.archive.Read(stored.ArchiveRef)
	if err != nil {
		t.Fatal(err)
	}
	if object.Content != large || !strings.Contains(object.Content, "FULL_RESULT_SENTINEL") {
		t.Fatal("full result was not retained in the immutable archive")
	}
}

func TestTrimMessagesForContextPressurePreservesFullToolResultAndPair(t *testing.T) {
	large := strings.Repeat("a", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("b", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < 30; i++ {
		messages = append(messages, Message{Role: "user", Content: strings.Repeat("old", 100)})
	}
	messages = append(messages,
		Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}},
		Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}},
	)
	for i := 0; i < contextPressureKeepRecent-1; i++ {
		messages = append(messages, Message{Role: "user", Content: "later"})
	}

	trimmed := trimMessagesForContextPressure(messages, contextPressureKeepRecent)
	if len(trimmed) != contextPressureKeepRecent+2 {
		t.Fatalf("trimmed messages = %d, want %d", len(trimmed), contextPressureKeepRecent+2)
	}

	foundCall := false
	foundResult := false
	for _, msg := range trimmed {
		for _, call := range msg.ToolCalls {
			if call.ID == "call-1" {
				foundCall = true
			}
		}
		for _, result := range msg.ToolResults {
			if result.CallID == "call-1" {
				foundResult = true
				if result.Content != large {
					t.Fatalf("tool result changed: len=%d want=%d", len(result.Content), len(large))
				}
			}
		}
	}
	if !foundCall {
		t.Fatal("matching tool call was not preserved")
	}
	if !foundResult {
		t.Fatal("tool result was not preserved")
	}
}

func TestThinkerCompactForContextPressurePreservesOversizedToolResult(t *testing.T) {
	session := NewSession(t.TempDir(), "main")
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	for i := 0; i < contextPressureKeepRecent+10; i++ {
		msg := Message{Role: "user", Content: "message"}
		if i == contextPressureKeepRecent+8 {
			msg = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
		}
		if i == contextPressureKeepRecent+9 {
			msg = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
		}
		session.AppendMessage(msg, i, TokenUsage{})
	}
	thinker := &Thinker{
		session:  session,
		threadID: "main",
		messages: append([]Message{{Role: "system", Content: "system"}},
			make([]Message, contextPressureKeepRecent+10)...),
	}
	for i := 1; i < len(thinker.messages); i++ {
		thinker.messages[i] = Message{Role: "user", Content: strings.Repeat("m", 100)}
	}
	thinker.messages[len(thinker.messages)-2] = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	thinker.messages[len(thinker.messages)-1] = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}

	reduced := thinker.compactForContextPressure("test", TokenUsage{PromptTokens: 205_000}, 1)
	if !reduced {
		t.Fatal("expected context pressure compaction to report progress")
	}

	if len(thinker.messages) != contextPressureKeepRecent+1 {
		t.Fatalf("messages = %d, want %d", len(thinker.messages), contextPressureKeepRecent+1)
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 {
		t.Fatalf("expected last message tool result, got %+v", last)
	}
	if last.ToolResults[0].Content != large {
		t.Fatalf("in-memory tool result changed: len=%d want=%d", len(last.ToolResults[0].Content), len(large))
	}
}

func TestThinkerSemanticCompactionSummarizesOldContextAndPreservesLatestToolResult(t *testing.T) {
	provider := &testCompactionProvider{
		response: strings.Join([]string{
			"# Objective",
			"- Continue the lead workflow.",
			"## Completed Work",
			"- CONSULTING_LEAD_ALPHA was qualified from the older context.",
			"## Current State",
			"- Latest tool result remains in the retained tail.",
			"## Important Tool Results",
			"- None known",
			"## Decisions And Constraints",
			"- Preserve emails.",
			"## Open Tasks",
			"- Follow up.",
			"## Risks And Failed Attempts",
			"- None known",
		}, "\n"),
	}
	session := NewSession(t.TempDir(), "main")
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+8; i++ {
		content := strings.Repeat("older context ", 300)
		if i == 3 {
			content += " CONSULTING_LEAD_ALPHA email alpha@example.com qualified"
		}
		msg := Message{Role: "user", Content: content}
		messages = append(messages, msg)
		session.AppendMessage(msg, i, TokenUsage{})
	}
	callMsg := Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	resultMsg := Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
	messages = append(messages, callMsg, resultMsg)
	session.AppendMessage(callMsg, 99, TokenUsage{})
	session.AppendMessage(resultMsg, 100, TokenUsage{})

	thinker := &Thinker{
		provider:  provider,
		session:   session,
		threadID:  "main",
		messages:  messages,
		quit:      make(chan struct{}),
		telemetry: &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{PromptTokens: 205_000}, 1)
	if !reduced {
		t.Fatal("expected semantic compaction to reduce context")
	}
	if provider.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", provider.calls)
	}
	if len(provider.prompts) != 1 || !strings.Contains(provider.prompts[0][1].Content, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("compaction prompt did not include old context: %+v", provider.prompts)
	}
	if len(thinker.messages) < 4 {
		t.Fatalf("messages after compaction too short: %d", len(thinker.messages))
	}
	if !strings.Contains(thinker.messages[1].Content, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("semantic summary missing from compacted context: %q", thinker.messages[1].Content)
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 || last.ToolResults[0].Content != large {
		t.Fatalf("latest full tool result was not preserved: %+v", last.ToolResults)
	}

	events, _ := thinker.telemetry.StoredEvents(0)
	seenStarted := false
	seenDone := false
	seenContextCompat := false
	for _, ev := range events {
		switch ev.Type {
		case "llm.compaction_started":
			seenStarted = true
		case "llm.compaction_done":
			seenDone = true
			var data map[string]any
			if err := json.Unmarshal(ev.Data, &data); err != nil {
				t.Fatalf("done telemetry JSON: %v", err)
			}
			if data["mode"] != "semantic" {
				t.Fatalf("done telemetry mode = %v, want semantic", data["mode"])
			}
		case "llm.context_compacted":
			seenContextCompat = true
		}
	}
	if !seenStarted || !seenDone || !seenContextCompat {
		t.Fatalf("missing compaction telemetry: started=%v done=%v context=%v", seenStarted, seenDone, seenContextCompat)
	}

	entries := readSessionEntriesForTest(t, session.path)
	if len(entries) == 0 || entries[0].Role != "_compacted" {
		t.Fatalf("session was not compacted with summary: %+v", entries)
	}
	if !strings.Contains(entries[0].Summary, "CONSULTING_LEAD_ALPHA") {
		t.Fatalf("session summary missing semantic content: %q", entries[0].Summary)
	}
}

func TestThinkerSemanticCompactionNoReductionFallsBackToEmergencyTrim(t *testing.T) {
	provider := &testCompactionProvider{response: strings.Repeat("oversized summary ", 80_000)}
	large := strings.Repeat("z", 16_000) + "FULL_RESULT_SENTINEL" + strings.Repeat("q", 16_000)
	messages := []Message{{Role: "system", Content: "system"}}
	for i := 0; i < contextPressureKeepRecent+10; i++ {
		messages = append(messages, Message{Role: "user", Content: "old"})
	}
	messages[len(messages)-2] = Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-1", Name: "media_search"}}}
	messages[len(messages)-1] = Message{Role: "user", ToolResults: []ToolResult{{CallID: "call-1", Content: large}}}
	thinker := &Thinker{
		provider:  provider,
		threadID:  "main",
		messages:  messages,
		quit:      make(chan struct{}),
		telemetry: &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{}, 1)
	if !reduced {
		t.Fatal("expected fallback trim to reduce context")
	}
	if strings.Contains(thinker.messages[1].Content, "oversized summary") {
		t.Fatal("accepted semantic summary that did not reduce context")
	}
	last := thinker.messages[len(thinker.messages)-1]
	if len(last.ToolResults) != 1 || last.ToolResults[0].Content != large {
		t.Fatalf("fallback did not preserve latest full tool result: %+v", last.ToolResults)
	}
}

func TestThinkerCompactForContextPressureReportsNoProgress(t *testing.T) {
	large := strings.Repeat("x", contextPressureCharFallback+1)
	thinker := &Thinker{
		threadID: "main",
		messages: []Message{
			{Role: "system", Content: "system"},
			{Role: "user", Content: large},
		},
	}

	reduced := thinker.compactForContextPressure("test", TokenUsage{}, 2)
	if reduced {
		t.Fatal("did not expect progress when there is no old context to remove")
	}
	if thinker.messages[1].Content != large {
		t.Fatal("expected full message content to remain unchanged")
	}
}
