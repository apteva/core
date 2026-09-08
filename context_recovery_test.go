package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func oversizedFixture(payload int) []Message {
	return []Message{
		{Role: "system", Content: "Follow the release instructions; never duplicate published releases."},
		{Role: "user", Content: "[COMPACTED CONTEXT]\nRelease remains PENDING-RELEASE-927. Already published post POST-881: do not duplicate. Artifact /releases/final.png."},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "call-inspect", Name: "inspect", Args: map[string]string{"id": "asset-37"}}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "call-inspect", Content: strings.Repeat("payload-data ", payload/13) + "EXACT-MIDDLE-ARTIFACT-REF"}}},
		{Role: "user", Content: "Inspect the existing release and report its state. Do not publish again.", EventIDs: []string{"event-sept8"}},
	}
}
func recoveryTestThinker(t *testing.T, p LLMProvider, m []Message) *Thinker {
	t.Helper()
	th := retryTestThinker(p)
	th.messages = cloneMessages(m)
	th.session = NewSession(t.TempDir(), "main")
	th.telemetry = &Telemetry{notify: make(chan struct{}, 1), quit: make(chan struct{})}
	for _, msg := range m {
		if msg.Role != "system" {
			if err := th.session.AppendMessage(msg, 1, TokenUsage{}); err != nil {
				t.Fatal(err)
			}
		}
	}
	return th
}

func TestOversizedPreparedRequestRecoversBeforeProviderAndPersists(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "release is pending"}}
	original := oversizedFixture(1_600_000)
	th := recoveryTestThinker(t, p, original)
	resp, err := th.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "release is pending" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if p.calls != 1 {
		t.Fatalf("sent oversized input to provider: calls=%d", p.calls)
	}
	if estimatedContextTokens(p.messages[0]) >= estimatedContextTokens(original)/2 {
		t.Fatal("request did not shrink")
	}
	for _, i := range []int{0, 1, 4} {
		if p.messages[0][i].Content != original[i].Content {
			t.Fatalf("lost instructions/state at %d", i)
		}
	}
	if !strings.Contains(original[3].ToolResults[0].Content, "EXACT-MIDDLE-ARTIFACT-REF") {
		t.Fatal("original input mutated")
	}
	loaded, _ := NewSession(strings.TrimSuffix(strings.TrimSuffix(th.session.path, "/main.jsonl"), "/history"), "main").LoadTail(50)
	joined := fmt.Sprint(loaded)
	for _, want := range []string{"PENDING-RELEASE-927", "POST-881", "/releases/final.png", "Do not publish again"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("restart lost %s", want)
		}
	}
	if estimatedContextTokens(loaded) > 30000 {
		t.Fatal("restart restored oversized input")
	}
	ref := ""
	events, _ := th.telemetry.StoredEvents(0)
	for _, ev := range events {
		if ev.Type == "llm.context_recovery" {
			var d map[string]any
			_ = json.Unmarshal(ev.Data, &d)
			ref = strings.TrimPrefix(d["archive_ref"].(string), strings.TrimSuffix(th.session.path, "main.jsonl"))
		}
	}

	archived, err := th.session.archive.Read(ref)
	if err != nil || !strings.Contains(archived.Content, "EXACT-MIDDLE-ARTIFACT-REF") {
		t.Fatalf("lossless archive unavailable: %v", err)
	}
	if !th.session.EventIDs()["event-sept8"] {
		t.Fatal("checkpoint lost consumed event ID")
	}
}
func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestContextRejectionShrinksWireRequestBeforeRetry(t *testing.T) {
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "text/event-stream")
		if len(bodies) == 1 {
			fmt.Fprint(w, "data: {\"type\":\"error\",\"error\":{\"code\":\"context_length_exceeded\",\"message\":\"Your input exceeds the context window\"}}\n\n")
			return
		}
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"recovered\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":200,\"output_tokens\":2}}}\n\n")
	}))
	defer server.Close()
	p := &OpenAINativeProvider{name: "openai-codex", responsesURL: server.URL, forceStoreFalse: true, models: map[ModelTier]string{ModelLarge: "context-test", ModelSmall: "context-test"}}
	th := recoveryTestThinker(t, p, oversizedFixture(160000))
	fallback := &scriptedRetryProvider{name: "anthropic", failures: 99, failureErr: errors.New("Anthropic API error 401: invalid API key")}
	th.pool = &ProviderPool{providers: map[string]LLMProvider{p.Name(): p, "anthropic": fallback}, order: []string{p.Name(), "anthropic"}, default_: p.Name()}
	resp, err := th.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "recovered" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if len(bodies) != 2 || len(bodies[1]) >= len(bodies[0])/2 {
		t.Fatalf("unchanged retry: wire sizes %v", func() []int {
			out := []int{}
			for _, b := range bodies {
				out = append(out, len(b))
			}
			return out
		}())
	}
	if fallback.calls != 0 {
		t.Fatal("context error incorrectly invoked unauthenticated fallback")
	}
	budgets := []requestBudget{}
	events, _ := th.telemetry.StoredEvents(0)
	for _, ev := range events {
		if ev.Type == "llm.request_budget" {
			var b requestBudget
			_ = json.Unmarshal(ev.Data, &b)
			if b.Stage == "serialized" {
				budgets = append(budgets, b)
			}
		}
	}
	if len(budgets) != 2 {
		t.Fatalf("wire diagnostics count=%d", len(budgets))
	}
	for i, b := range budgets {
		if b.SerializedBytes != len(bodies[i]) || b.Fingerprint != requestFingerprint(bodies[i]) {
			t.Fatal("diagnostics differ from submitted bytes")
		}
	}
	if budgets[0].Fingerprint == budgets[1].Fingerprint {
		t.Fatal("retry reused fingerprint")
	}
}

func TestContextFailureWithoutSafeReductionPreservesState(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", failures: 99, failureErr: errors.New("context_length_exceeded")}
	m := []Message{{Role: "system", Content: "Keep all pending release state."}, {Role: "user", Content: "PENDING-927; published=POST-881; do not duplicate."}}
	th := recoveryTestThinker(t, p, m)
	before, _ := os.ReadFile(th.session.path)
	_, err := th.callLLMWithRetry(context.Background())
	var management *contextManagementError
	if !errors.As(err, &management) || !strings.Contains(providerExecutionFailureReason(err), "context_management_failed") {
		t.Fatalf("unactionable error: %v", err)
	}
	if p.calls != 1 {
		t.Fatalf("unchanged context retried %d times", p.calls)
	}
	after, _ := os.ReadFile(th.session.path)
	if string(before) != string(after) || string(mustJSON(t, m)) != string(mustJSON(t, th.messages)) {
		t.Fatal("failure discarded original state")
	}
}

func TestFallbackAuthFailureDoesNotStopTransientPrimaryRecovery(t *testing.T) {
	primary := &scriptedRetryProvider{name: "primary", failures: 2, failureErr: errors.New("HTTP 503 unavailable"), response: ChatResponse{Text: "primary recovered"}}
	fallback := &scriptedRetryProvider{name: "anthropic", failures: 99, failureErr: errors.New("Anthropic API error 401: invalid API key")}
	th := retryTestThinker(primary)
	th.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "anthropic": fallback}, order: []string{"primary", "anthropic"}, default_: "primary"}
	th.retryDelay = func(error, int) time.Duration { return 0 }
	resp, err := th.callLLMWithRetry(context.Background())
	if err != nil || resp.Text != "primary recovered" {
		t.Fatalf("response=%q err=%v", resp.Text, err)
	}
	if primary.calls != 3 || fallback.calls != 1 {
		t.Fatalf("calls primary=%d fallback=%d", primary.calls, fallback.calls)
	}
}

func TestRequestBudgetCountsToolsOpaqueStateAndOutput(t *testing.T) {
	msg := []Message{{Role: "system", Content: "instructions"}, {Role: "assistant", ProviderState: &ProviderResponseState{Provider: openAIResponsesStateProvider, Items: []json.RawMessage{json.RawMessage(`{"type":"reasoning","encrypted_content":"` + strings.Repeat("x", 400000) + `"}`)}}}}
	tools := []NativeTool{{Name: "large_schema", Description: strings.Repeat("schema", 24000)}}
	b := estimatePreparedRequest("openai-codex", "context-test", msg, tools)
	if !b.OverBudget || b.ToolTokens < 30000 || b.InputTokens < 130000 || b.ReservedOutput == 0 {
		t.Fatalf("incomplete budget: %+v", b)
	}
	var wire requestBudget
	ctx := context.WithValue(context.Background(), requestObserverKey{}, requestObserver(func(b requestBudget) { wire = b }))
	body := mustJSON(t, map[string]any{"model": "context-test", "instructions": "system", "tools": tools, "input": []any{map[string]any{"type": "reasoning", "encrypted_content": strings.Repeat("x", 400000)}, map[string]any{"type": "message", "content": []any{map[string]any{"type": "input_image", "image_url": "data:image/png;base64," + strings.Repeat("A", 400000)}}}}})
	err := observeProviderRequest(ctx, "openai-codex", "context-test", body)
	if !isContextLengthError(err) || wire.OpaqueTokens < 100000 || wire.ImageTokens != defaultImageTokenEstimate || wire.ToolTokens < 30000 {
		t.Fatalf("wire budget: %+v err=%v", wire, err)
	}
}

func TestRecoveryCheckpointArchivesHistoryOutsideLiveTail(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "ok"}}
	th := recoveryTestThinker(t, p, oversizedFixture(1600000))
	if err := th.session.Append(SessionEntry{Role: "user", Content: "OLDER-JOURNAL-ONLY-FACT", EventIDs: []string{"older-consumed"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := th.callLLMWithRetry(context.Background()); err != nil {
		t.Fatal(err)
	}
	events, _ := th.telemetry.StoredEvents(0)
	ref := ""
	for _, ev := range events {
		if ev.Type == "llm.context_recovery" {
			var d map[string]any
			_ = json.Unmarshal(ev.Data, &d)
			ref, _ = d["history_archive_ref"].(string)
		}
	}
	archived, err := th.session.archive.Read(ref)
	if err != nil || !strings.Contains(archived.Content, "OLDER-JOURNAL-ONLY-FACT") {
		t.Fatalf("out-of-window journal lost: %v", err)
	}
	if !th.session.EventIDs()["older-consumed"] {
		t.Fatal("consumed event ledger lost")
	}
}

func TestContextBudgetBlocksUnshrinkableInstructionsWithoutNetwork(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "should not run"}}
	m := []Message{{Role: "system", Content: strings.Repeat("mandatory instructions ", 50000)}, {Role: "user", Content: "Preserve pending release state."}}
	th := recoveryTestThinker(t, p, m)
	_, err := th.callLLMWithRetry(context.Background())
	if !isContextLengthError(err) || p.calls != 0 {
		t.Fatalf("over-budget instructions submitted: calls=%d err=%v", p.calls, err)
	}
	if th.messages[0].Content != m[0].Content {
		t.Fatal("instructions truncated")
	}
}

func TestProviderChainErrorPreservesBothCauses(t *testing.T) {
	primary := errors.New("context_length_exceeded")
	fallback := errors.New("Anthropic API error 401: invalid API key")
	err := &providerChainError{PrimaryName: "openai-codex", Primary: primary, FallbackName: "anthropic", Fallback: fallback}
	if !errors.Is(err, primary) || !errors.Is(err, fallback) || !isContextLengthError(err) {
		t.Fatal("original error identity lost")
	}
}

type nativeBudgetProvider struct{ scriptedRetryProvider }

func (p *nativeBudgetProvider) SupportsNativeTools() bool { return true }
func TestOversizedToolSchemasBlockBeforeProvider(t *testing.T) {
	p := &nativeBudgetProvider{scriptedRetryProvider: scriptedRetryProvider{name: "native"}}
	th := recoveryTestThinker(t, p, []Message{{Role: "system", Content: "Current instructions"}, {Role: "user", Content: "Preserve pending state"}})
	th.registry = NewToolRegistry("")
	if ok := th.registry.Register(&ToolDef{Name: "huge_schema", Description: strings.Repeat("required schema detail ", 30000), Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "unused"} }}); !ok {
		t.Fatal("register huge schema")
	}
	_, err := th.callLLMWithRetry(context.Background())
	if !isContextLengthError(err) || p.calls != 0 {
		t.Fatalf("oversized schemas submitted: calls=%d err=%v", p.calls, err)
	}
}

func TestRecoveryDoesNotWaitForBackgroundCompactionModel(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", response: ChatResponse{Text: "ok"}}
	th := recoveryTestThinker(t, p, oversizedFixture(1600000))
	th.session.compactMu.Lock()
	defer th.session.compactMu.Unlock()
	done := make(chan error, 1)
	go func() { _, err := th.callLLMWithRetry(context.Background()); done <- err }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("recovery waited on background summary lock")
	}
}

func TestRecoveryPreservesStructuredStateBesideLargePayloads(t *testing.T) {
	payload := string(mustJSON(t, map[string]any{"a_image": strings.Repeat("pixels", 50000), "pending_release": "PENDING-IN-MIDDLE", "receipt_id": uint64(9007199254740993), "published_post": "POST-DO-NOT-DUPLICATE", "z_document": strings.Repeat("document", 50000)}))
	m := oversizedFixture(1)
	m[3].ToolResults[0].Content = payload
	next := projectBulkyContext(m, "history/original.json")
	projectedResult := ""
	for _, msg := range next {
		if len(msg.ToolResults) > 0 {
			projectedResult = msg.ToolResults[0].Content
		}
	}
	var recovered map[string]any
	if err := json.Unmarshal([]byte(projectedResult), &recovered); err != nil {
		t.Fatal(err)
	}
	if recovered["pending_release"] != "PENDING-IN-MIDDLE" || recovered["published_post"] != "POST-DO-NOT-DUPLICATE" {
		t.Fatal("structured release state lost beside large payload")
	}
	if !strings.Contains(projectedResult, `"receipt_id":9007199254740993`) {
		t.Fatal("numeric identifier precision lost")
	}
	if len(projectedResult) > 30000 {
		t.Fatal("large leaves not reduced")
	}
	session := NewSession(t.TempDir(), "main")
	if err := session.AppendMessage(Message{Role: "user", Content: "prior state"}, 1, TokenUsage{}); err != nil {
		t.Fatal(err)
	}
	if _, err := session.checkpointRecoveredContext(next); err != nil {
		t.Fatal(err)
	}
	loaded, _ := session.LoadTail(50)
	notes := ""
	for _, msg := range loaded {
		if strings.HasPrefix(msg.Content, "[RECOVERED TOOL STATE]") {
			notes += msg.Content
		}
	}
	for _, id := range []string{"PENDING-IN-MIDDLE", "POST-DO-NOT-DUPLICATE", "9007199254740993"} {
		if !strings.Contains(notes, id) {
			t.Fatalf("restart lost structured state %s", id)
		}
	}
	if m[3].ToolResults[0].Content != payload {
		t.Fatal("source payload changed")
	}
}

func TestRecoveryPreservesInFlightCall(t *testing.T) {
	p := &scriptedRetryProvider{name: "primary", failures: 1, failureErr: errors.New("context_length_exceeded"), response: ChatResponse{Text: "pending work preserved"}}
	args := strings.Repeat("pending argument ", 3000)
	m := []Message{{Role: "system", Content: "Preserve pending work."}, {Role: "assistant", ToolCalls: []NativeToolCall{{ID: "in-flight", Name: "long_operation", Args: map[string]string{"payload": args}}}}}
	for i := 0; i < 30; i++ {
		m = append(m, Message{Role: "assistant", Content: strings.Repeat("old progress note ", 200)})
	}
	m = append(m, Message{Role: "user", Content: "Report current progress."})
	th := recoveryTestThinker(t, p, m)
	th.pendingTools.Store("in-flight", "long_operation")
	if _, err := th.callLLMWithRetryMessages(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, msg := range th.messages {
		for _, call := range msg.ToolCalls {
			if call.ID == "in-flight" {
				found = true
				if call.Args["payload"] != args {
					t.Fatal("in-flight call arguments altered")
				}
			}
		}
	}
	if !found {
		t.Fatal("in-flight native call lost during summary")
	}
}
