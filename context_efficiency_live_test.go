package core

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

type efficiencyLiveGate struct {
	LLMProvider
	once    sync.Once
	stream  chan context.Context
	release chan struct{}
}

func (p *efficiencyLiveGate) Chat(ctx context.Context, m []Message, model string, tools []NativeTool, chunk func(string), thinking func(string), toolChunk func(string, string, string)) (ChatResponse, error) {
	gate := func() {
		p.once.Do(func() {
			p.stream <- ctx
			select {
			case <-p.release:
			case <-ctx.Done():
			}
		})
	}
	return p.LLMProvider.Chat(ctx, m, model, tools, func(s string) {
		if s != "" {
			gate()
		}
		if chunk != nil {
			chunk(s)
		}
	}, func(s string) {
		if s != "" {
			gate()
		}
		if thinking != nil {
			thinking(s)
		}
	}, func(n, id, s string) {
		gate()
		if toolChunk != nil {
			toolChunk(n, id, s)
		}
	})
}

// Real Sol calls compare provider-reported input usage on the same frozen
// evidence, check exact state/reasons, and exercise a rate-limited primary,
// incompatible fallback, config no-ops during streaming, and large summaries.
func TestIntegration_CodexGPT61SolContextEfficiency(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_CONTEXT_EFFICIENCY_LIVE") != "1" {
		t.Skip("set RUN_CODEX_CONTEXT_EFFICIENCY_LIVE=1 without -short")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live test requested without valid Codex credential")
	}
	p := toolReasonLiveProvider(t, token)
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	registry := NewToolRegistry("")
	catalog := []mcpToolDef{mkTool("report", "Report exact observed run identifiers and pending approval; never publish"), mkTool("other", "An unrelated operation")}
	catalog[0].InputSchema = map[string]any{"type": "object", "properties": map[string]any{"run_id": map[string]any{"type": "string"}, "approval_state": map[string]any{"type": "string"}, "published_post": map[string]any{"type": "string"}, "receipt_id": map[string]any{"type": "string"}}, "required": []string{"run_id", "approval_state", "published_post", "receipt_id"}}
	index := NewToolIndex()
	index.Add("workflow", catalog, false)
	registerTestMCPTools(registry, "workflow", catalog)
	th := retryTestThinker(p)
	th.registry = registry
	th.toolIndex = index
	th.config = &Config{}
	th.session = NewSession(t.TempDir(), "main")
	th.messages = efficiencyHistory(t, th)
	th.messages[0].Content = "This isolated test uses synthetic tool results. Call workflow_report exactly once on each turn using the current run's exact observed state. Preserve numeric identifiers as strings. Approval pending must remain pending; never publish. Always include a nonempty _reason. Ignore rows containing historical procedure details."
	th.messages[1].Content = "Report current run current using workflow_report. Its receipt_id and existing published_post are in the observed run metadata."
	tools := th.prepareNativeTools(p.Name())
	check := func(response ChatResponse) {
		t.Helper()
		if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "workflow_report" {
			t.Fatalf("unexpected response: text=%q tools=%v", response.Text, response.ToolCalls)
		}
		call := response.ToolCalls[0]
		assertLiveModelToolReason(t, registry, call)
		for key, want := range map[string]string{"run_id": "current", "approval_state": "pending", "published_post": "POST-881", "receipt_id": "9007199254740993"} {
			if call.Args[key] != want {
				t.Fatalf("state changed: %s=%q want %q", key, call.Args[key], want)
			}
		}
	}
	baseline, err := p.Chat(ctx, th.messages, toolReasonLiveModel, tools, nil, nil, nil)
	if err != nil {
		t.Fatal("baseline call", err)
	}
	check(baseline)
	projected := th.prepareToolResultRequest(th.messages)
	gate := &efficiencyLiveGate{LLMProvider: p, stream: make(chan context.Context, 1), release: make(chan struct{})}
	releaseGate := sync.OnceFunc(func() { close(gate.release) })
	defer releaseGate()
	th.provider = gate
	th.beginRuntime()
	defer th.endRuntime()
	api := &APIServer{thinker: th}
	type outcome struct {
		response ChatResponse
		err      error
	}
	done := make(chan outcome, 1)
	go func() { r, e := th.callLLMWithRuntimeMutations(ctx, projected); done <- outcome{r, e} }()
	select {
	case active := <-gate.stream:
		for _, body := range []string{"{}", "{", `{"reset":{}}`} {
			w := httptest.NewRecorder()
			api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(body)))
			if w.Code != 200 && w.Code != 400 {
				t.Fatal(w.Code, w.Body.String())
			}
			if active.Err() != nil {
				t.Fatal("config no-op cancelled genuine model stream")
			}
		}
	case <-ctx.Done():
		t.Fatal("no live stream", ctx.Err())
	}
	releaseGate()
	result := <-done
	if result.err != nil {
		t.Fatal(result.err)
	}
	check(result.response)
	if baseline.Usage.PromptTokens <= 0 || result.response.Usage.PromptTokens <= 0 || result.response.Usage.PromptTokens >= baseline.Usage.PromptTokens*3/5 {
		t.Fatalf("provider usage did not improve sufficiently: before=%d after=%d", baseline.Usage.PromptTokens, result.response.Usage.PromptTokens)
	}
	t.Logf("Sol input tokens: baseline=%d projected=%d reduction=%.1f%%", baseline.Usage.PromptTokens, result.response.Usage.PromptTokens, 100*(1-float64(result.response.Usage.PromptTokens)/float64(baseline.Usage.PromptTokens)))
	th.markToolResultsConsumed(projected)
	th.messages = append(th.messages, Message{Role: "assistant", ToolCalls: result.response.ToolCalls}, Message{Role: "user", ToolResults: []ToolResult{{CallID: result.response.ToolCalls[0].ID, Content: `{"accepted":true,"continue_with_same_observed_state":true}`}}})
	th.memoryRecall.context = "workflow_other unrelated operation"
	if next := th.prepareNativeTools(p.Name()); !reflect.DeepEqual(next, tools) {
		t.Fatal("continuation changed real tool manifest")
	}
	th.provider = p
	continuation, err := th.callLLMWithRetryMessages(ctx, th.prepareToolResultRequest(th.messages))
	if err != nil {
		t.Fatal(err)
	}
	if len(continuation.ToolCalls) > 0 {
		check(continuation)
	} else {
		for _, marker := range []string{"current", "pending", "POST-881", "9007199254740993"} {
			if !strings.Contains(continuation.Text, marker) {
				t.Fatal("continuation lost exact observed state", marker)
			}
		}
	}
	t.Run("compatible_real_fallback", func(t *testing.T) {
		primary := &efficiencyBudgetProvider{&scriptedRetryProvider{name: "injected-rate-limit", failures: 99, failureErr: errors.New("HTTP 429 rate limit")}, toolReasonLiveModel}
		small := &efficiencyBudgetProvider{&scriptedRetryProvider{name: "small-incompatible"}, "efficiency-live-small"}
		registerModelCapabilities(map[string]ModelCapabilities{"efficiency-live-small": {ContextWindow: 8192}})
		th.provider = primary
		th.pool = &ProviderPool{providers: map[string]LLMProvider{primary.Name(): primary, small.Name(): small, p.Name(): p}, order: []string{primary.Name(), small.Name(), p.Name()}, default_: primary.Name()}
		response, e := th.callLLMWithRetryMessages(ctx, projected)
		if e != nil {
			t.Fatal(e)
		}
		check(response)
		if primary.calls != 1 || small.calls != 0 {
			t.Fatal("unsuitable fallback attempted", primary.calls, small.calls)
		}
	})
	t.Run("oversized_history_summary", func(t *testing.T) {
		th.provider = p
		message := Message{Role: "user", Content: "Pending release PENDING-927. Existing published post POST-881 must never be duplicated.\n" + strings.Repeat("Historical procedure detail with no new action. ", 1500) + "\nHuman approval APPROVAL-927 remains pending.\n" + strings.Repeat("Historical procedure detail with no new action. ", 1500) + "\nArtifact https://example.com/existing.png"}
		summary, e := th.summarizeRecoveryPrefix(ctx, p, []Message{message})
		if e != nil {
			t.Fatal(e)
		}
		for _, marker := range []string{"PENDING-927", "POST-881", "APPROVAL-927", "https://example.com/existing.png"} {
			if !strings.Contains(summary, marker) {
				t.Fatal("summary lost exact evidence", marker)
			}
		}
		t.Logf("Sol summarized oversized history without dropping exact continuation state: source_bytes=%d summary_bytes=%d", len(message.Content), len(summary))
	})
}
