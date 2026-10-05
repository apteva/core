package core

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func efficiencyPayload(t *testing.T, id string) string {
	t.Helper()
	rows := make([]map[string]any, 1800)
	for i := range rows {
		rows[i] = map[string]any{"index": i, "description": strings.Repeat("historical procedure details ", 3)}
	}
	return string(mustJSON(t, map[string]any{
		"rows": rows, "run": map[string]any{"id": id, "state": "waiting", "approval_state": "pending", "published_post": "POST-881", "receipt_id": uint64(9007199254740993)},
		"read_reference": map[string]any{"tool": "processes_step_get", "arguments": map[string]any{"process_id": "process-1", "run_id": id, "step_id": "step-1"}},
	}))
}

func efficiencyHistory(t *testing.T, thinker *Thinker) []Message {
	t.Helper()
	messages := []Message{{Role: "system", Content: "Use observations exactly; approval pending is not completion."}, {Role: "user", Content: "Inspect the frozen run and report its state."}}
	for i, id := range []string{"old-one", "old-two", "current"} {
		callID := "inspect-" + id
		messages = append(messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: callID, Name: "inspect", Args: map[string]string{"id": id}}}})
		result := Message{Role: "user", ToolResults: []ToolResult{{CallID: callID, ToolName: "inspect", Content: efficiencyPayload(t, id)}}}
		if thinker != nil {
			result = thinker.archiveToolResultMessage(result)
		}
		messages = append(messages, result)
		if thinker != nil && i < 2 {
			for j := 0; j < 3; j++ {
				thinker.markToolResultsConsumed(messages)
			}
		}
	}
	return messages
}

func TestContextEfficiencyRetentionProtectsFreshStateAndArchive(t *testing.T) {
	th := &Thinker{threadID: "worker", session: NewSession(t.TempDir(), "worker")}
	original := efficiencyHistory(t, th)
	before := estimatedContextTokens(original)
	projected := th.prepareToolResultRequest(original)
	if after := estimatedContextTokens(projected); after >= before/2 {
		t.Fatalf("input did not shrink: before=%d after=%d", before, after)
	}
	for _, i := range []int{3, 5} {
		r := projected[i].ToolResults[0]
		for _, field := range []string{"pending", "POST-881", "9007199254740993", "processes_step_get"} {
			if !strings.Contains(r.Content, field) {
				t.Fatalf("receipt lost %s", field)
			}
		}
		archived, err := th.session.archive.Read(r.ArchiveRef)
		if err != nil || archived.Content != original[i].ToolResults[0].Content {
			t.Fatal("archive lost original", err)
		}
	}
	if !reflect.DeepEqual(projected[7], original[7]) {
		t.Fatal("fresh result changed before consumption")
	}
	if th.promptCacheEpoch != 1 {
		t.Fatal("expected one batched pressure checkpoint", th.promptCacheEpoch)
	}
	th.prepareToolResultRequest(original)
	if th.promptCacheEpoch != 1 {
		t.Fatal("unchanged continuation reset the cache")
	}
}

func TestContextEfficiencyToolPhaseIgnoresRecallChurn(t *testing.T) {
	th := automaticTicketThinker(t)
	th.messages = []Message{{Role: "user", Content: "Read tickets and assign a ticket to an agent"}}
	first := th.prepareNativeTools("openai-codex")
	for _, recall := range []string{"create feedback areas", "tickets_tickets_get", "tickets_tickets_update"} {
		th.memoryRecall.context = recall
		th.iteration++
		if next := th.prepareNativeTools("openai-codex"); !reflect.DeepEqual(first, next) {
			t.Fatal("recall changed phase manifest")
		}
	}
	th.toolAllowlist = map[string]bool{"tickets_tickets_get": true}
	if nativeToolSet(th.prepareNativeTools("openai-codex"))["tickets_tickets_update"] {
		t.Fatal("phase retention bypassed revoked grants")
	}
}

func TestContextEfficiencyRetentionProtectsPendingCall(t *testing.T) {
	th := &Thinker{threadID: "worker", session: NewSession(t.TempDir(), "worker")}
	original := efficiencyHistory(t, th)
	th.pendingTools.Store("inspect-old-one", true)
	projected := th.prepareToolResultRequest(original)
	if !reflect.DeepEqual(projected[3], original[3]) {
		t.Fatal("pending call evidence projected")
	}
	if !projected[5].ToolResults[0].ContentIsPreview {
		t.Fatal("eligible completed observation stayed large")
	}
	th.pendingTools.Delete("inspect-old-one")
	projected = th.prepareToolResultRequest(original)
	if !projected[3].ToolResults[0].ContentIsPreview {
		t.Fatal("completed call did not become eligible")
	}
}

type efficiencyBudgetProvider struct {
	*scriptedRetryProvider
	model string
}

func (p *efficiencyBudgetProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{ModelLarge: p.model, ModelMedium: p.model, ModelSmall: p.model}
}

func TestContextEfficiencyFallbackSkipsSmallWindowWithoutRewriting(t *testing.T) {
	for _, compatible := range []bool{false, true} {
		t.Run(map[bool]string{false: "retry-primary", true: "use-large-fallback"}[compatible], func(t *testing.T) {
			primary := &efficiencyBudgetProvider{&scriptedRetryProvider{name: "primary", failures: 1, failureErr: errors.New("HTTP 429 rate limit"), response: ChatResponse{Text: "primary recovered"}}, "gpt-6.1-sol"}
			small := &efficiencyBudgetProvider{&scriptedRetryProvider{name: "small", response: ChatResponse{Text: "must not run"}}, "claude-fable-5"}
			large := &efficiencyBudgetProvider{&scriptedRetryProvider{name: "large", response: ChatResponse{Text: "large recovered"}}, "gpt-6.1-sol"}
			original := []Message{{Role: "system", Content: "Keep instructions"}, {Role: "user", Content: strings.Repeat("large current input ", 40000)}}
			th := recoveryTestThinker(t, primary, original)
			th.retryDelay = func(error, int) time.Duration { return 0 }
			th.pool = &ProviderPool{providers: map[string]LLMProvider{"primary": primary, "small": small}, order: []string{"primary", "small"}, default_: "primary"}
			if compatible {
				th.pool.providers["large"] = large
				th.pool.order = append(th.pool.order, "large")
			}
			journal, _ := os.ReadFile(th.session.path)
			response, err := th.callLLMWithRetry(context.Background())
			if err != nil || response.Text == "" {
				t.Fatal("recovery failed", err)
			}
			if small.calls != 0 || compatible && large.calls != 1 || !compatible && primary.calls != 2 {
				t.Fatalf("wrong routing: primary=%d small=%d large=%d", primary.calls, small.calls, large.calls)
			}
			after, _ := os.ReadFile(th.session.path)
			if string(after) != string(journal) || !reflect.DeepEqual(th.messages, original) {
				t.Fatal("unsuitable fallback rewrote live history")
			}
		})
	}
}

func TestContextEfficiencySummaryConsumesWholeOversizedMessage(t *testing.T) {
	p := &scriptedRetryProvider{name: "summary", response: ChatResponse{Text: "Approval pending; POST-881 exists; do not duplicate."}}
	th := retryTestThinker(p)
	message := Message{Role: "user", Content: strings.Repeat("prefix α ", 18000) + "MIDDLE-EXACT-STATE" + strings.Repeat(" suffix β", 18000)}
	if _, err := th.summarizeRecoveryPrefix(context.Background(), p, []Message{message}); err != nil {
		t.Fatal(err)
	}
	var reconstructed strings.Builder
	for _, request := range p.messages {
		var fragments []Message
		if err := json.Unmarshal([]byte(request[1].Content), &fragments); err != nil {
			t.Fatal(err)
		}
		for _, fragment := range fragments {
			_, body, ok := strings.Cut(fragment.Content, "\n")
			if !ok {
				t.Fatal("missing fragment marker")
			}
			reconstructed.WriteString(body)
		}
	}
	expected := string(mustJSON(t, message))
	if reconstructed.String() != expected {
		t.Fatal("summary omitted or changed original message bytes")
	}
}

func TestContextEfficiencyConfigValidationDoesNotCancel(t *testing.T) {
	api, th := newTestAPI()
	th.beginRuntime()
	defer th.endRuntime()
	ctx, finish := th.runtimeRequest(context.Background())
	defer finish()
	for _, test := range []struct {
		body   string
		status int
	}{
		{"{", 400}, {"{} {}", 400}, {"{}", 200}, {"null", 200}, {`{"reset":{}}`, 200}, {`{"automatic_tool_loading":{"enabled":true,"max_tools":21}}`, 400},
		{string(mustJSON(t, map[string]any{"directive": th.config.GetDirective()})), 200},
	} {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(test.body)))
		if w.Code != test.status {
			t.Fatal(test.body, w.Code, w.Body.String())
		}
		if ctx.Err() != nil {
			t.Fatal("invalid/no-op update cancelled inference", test.body)
		}
	}
	done := make(chan int, 1)
	go func() {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(`{"directive":"Updated real instruction"}`)))
		done <- w.Code
	}()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("real instruction change did not invalidate request")
	}
	if !th.applyRuntimeMutations() {
		t.Fatal("change not classified as invalidating")
	}
	if code := <-done; code != 200 {
		t.Fatal(code)
	}
}

func TestContextEfficiencyOutputBudgetAndUnknownCapabilities(t *testing.T) {
	p := &AnthropicProvider{}
	if p.outputTokenLimit("claude-fable-5") != 16384 {
		t.Fatal("unexpected default output reservation")
	}
	p.maxOutputTokens = 8000
	if p.outputTokenLimit("claude-fable-5") != 8000 {
		t.Fatal("configured output reservation ignored")
	}
	p.maxOutputTokens = 64000
	if p.outputTokenLimit("legacy") != 4096 {
		t.Fatal("model output maximum ignored")
	}
	b := estimatePreparedRequest("anthropic", "unknown-budget-model", nil, nil)
	if b.ContextWindowSource != "unknown_model_default" {
		t.Fatal("unknown window claimed as advertised", b)
	}
}

func TestContextEfficiencyPhaseHashesFullInstruction(t *testing.T) {
	th := automaticTicketThinker(t)
	th.messages = []Message{{Role: "user", Content: strings.Repeat("same opening ", 900) + "Task A"}}
	th.prepareNativeTools("openai-codex")
	phase := th.automaticTools.context
	th.messages[0].Content = strings.Repeat("same opening ", 900) + "Task B"
	th.prepareNativeTools("openai-codex")
	if phase == th.automaticTools.context {
		t.Fatal("different long task reused phase")
	}
}

func TestContextEfficiencyAnthropicBudgetIgnoresResponsesState(t *testing.T) {
	messages := []Message{{Role: "assistant", Content: "Observed exact state", Reasoning: strings.Repeat("private reasoning ", 10000)}}
	plain := cloneMessages(messages)
	plain[0].Reasoning = ""
	a := estimatePreparedRequest("anthropic", "claude-sonnet-4-6", messages, nil)
	b := estimatePreparedRequest("anthropic", "claude-sonnet-4-6", plain, nil)
	if a.InputTokens != b.InputTokens || messages[0].Reasoning == "" {
		t.Fatal("budget includes unsupported replay or changes source")
	}
}
