package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func setGrokBuildTestEnv(t *testing.T, baseURL string) {
	t.Helper()
	t.Setenv("GROK_BUILD_ACCESS_TOKEN", "grok-access")
	t.Setenv("GROK_BUILD_PROVIDER_ID", "connection-1")
	t.Setenv("GROK_BUILD_USER_ID", "user-1")
	t.Setenv("GROK_BUILD_ACCOUNT_EMAIL", "agent@example.com")
	t.Setenv("GROK_BUILD_PRINCIPAL_TYPE", "user")
	t.Setenv("GROK_BUILD_PRINCIPAL_ID", "principal-1")
	t.Setenv("GROK_BUILD_BASE_URL", baseURL)
	t.Setenv("SERVER_URL", "")
	t.Setenv("APTEVA_API_KEY", "")
}

func writeResponsesStream(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, event := range events {
		_, _ = io.WriteString(w, "data: "+event+"\n\n")
	}
}

func TestGrokBuildProviderConstructionAndModelOverrides(t *testing.T) {
	setGrokBuildTestEnv(t, "")
	provider := createProviderByName("grok-build")
	grok, ok := provider.(*OpenAINativeProvider)
	if !ok {
		t.Fatalf("provider = %T, want *OpenAINativeProvider", provider)
	}
	if grok.Name() != "grok-build" || grok.responsesEndpoint() != grokBuildDefaultBaseURL+"/responses" {
		t.Fatalf("provider = name %q endpoint %q", grok.Name(), grok.responsesEndpoint())
	}
	if input, cached, output := grok.CostPer1M(); input != 0 || cached != 0 || output != 0 {
		t.Fatalf("subscription cost = (%v, %v, %v), want zero", input, cached, output)
	}
	if got := grok.AvailableBuiltinTools(); len(got) != 0 {
		t.Fatalf("builtins = %#v, want none", got)
	}
	grok.SetBuiltinTools([]string{"web_search_preview", "code_interpreter"})
	if len(grok.builtinTools) != 0 {
		t.Fatalf("configured provider builtins leaked into Grok Build: %#v", grok.builtinTools)
	}
	applyModelOverrides(grok, map[string]string{
		"large": "grok-large", "medium": "grok-medium", "small": "grok-small",
	})
	if grok.Models()[ModelLarge] != "grok-large" || grok.Models()[ModelMedium] != "grok-medium" || grok.Models()[ModelSmall] != "grok-small" {
		t.Fatalf("models = %#v", grok.Models())
	}
}

func TestGrokBuildNormalizesSearchToolsRootSchemaOnlyForGrok(t *testing.T) {
	registry := NewToolRegistry("test")
	search := registry.Get("search_tools")
	if search == nil {
		t.Fatal("search_tools was not registered")
	}
	original, _ := cloneJSONValue(search.native.Parameters).(map[string]any)
	if _, ok := original["anyOf"]; !ok {
		t.Fatalf("fixture no longer exercises the Grok root-composition incompatibility: %#v", original)
	}

	setGrokBuildTestEnv(t, "")
	grok := NewGrokBuildProvider("token").(*OpenAINativeProvider)
	grokTools := grok.buildAPITools("grok-4.7", []NativeTool{search.native})
	grokTool, ok := grokTools[0].(oaiFunctionTool)
	if !ok {
		t.Fatalf("Grok tool = %T, want oaiFunctionTool", grokTools[0])
	}
	if grokTool.Parameters["type"] != "object" {
		t.Fatalf("Grok search_tools root type = %#v", grokTool.Parameters["type"])
	}
	for _, keyword := range []string{"allOf", "anyOf", "oneOf"} {
		if _, ok := grokTool.Parameters[keyword]; ok {
			t.Fatalf("Grok search_tools retained root %s: %#v", keyword, grokTool.Parameters)
		}
	}
	properties, _ := grokTool.Parameters["properties"].(map[string]any)
	if properties["query"] == nil || properties["queries"] == nil || properties["_reason"] == nil {
		t.Fatalf("Grok normalization lost search_tools properties: %#v", properties)
	}
	queries, _ := properties["queries"].(map[string]any)
	items, _ := queries["items"].(map[string]any)
	if _, ok := items["oneOf"]; !ok {
		t.Fatalf("Grok normalization changed a nested property schema: %#v", queries)
	}

	// The registry schema is shared by every provider. Grok normalization must
	// operate on a copy so OpenAI/Codex retain their existing schema exactly.
	if !reflect.DeepEqual(search.native.Parameters, original) {
		t.Fatalf("Grok normalization mutated the registry schema\n got: %#v\nwant: %#v", search.native.Parameters, original)
	}
	codex := &OpenAINativeProvider{sessionProfile: &openAICodexSessionProviderProfile}
	codexTools := codex.buildAPITools("gpt-5.5", []NativeTool{search.native})
	codexTool, ok := codexTools[0].(oaiFunctionTool)
	if !ok || !reflect.DeepEqual(codexTool.Parameters, original) {
		t.Fatalf("Codex search_tools schema changed: %#v", codexTools[0])
	}
}

func TestIntegration_GrokBuildAcceptsSearchToolsSchema(t *testing.T) {
	if os.Getenv("RUN_GROK_BUILD_SCHEMA_SMOKE") != "1" {
		t.Skip("set RUN_GROK_BUILD_SCHEMA_SMOKE=1 with Grok Build runtime-token environment")
	}
	registry := NewToolRegistry("grok-build-live")
	search := registry.Get("search_tools")
	if search == nil {
		t.Fatal("search_tools was not registered")
	}
	provider := NewGrokBuildProvider(os.Getenv("GROK_BUILD_ACCESS_TOKEN"))
	response, err := provider.Chat(
		context.Background(),
		[]Message{{Role: "user", Content: "Reply with exactly SCHEMA_OK. Do not call a tool."}},
		"grok-4.7",
		[]NativeTool{search.native},
		nil, nil, nil,
	)
	if err != nil {
		t.Fatalf("Grok Build rejected the normalized search_tools schema: %v", err)
	}
	if strings.TrimSpace(response.Text) == "" && len(response.ToolCalls) == 0 {
		t.Fatalf("Grok Build accepted the request but returned no text or tool call: %+v", response)
	}
	t.Logf("Grok Build accepted normalized search_tools schema: model=%s text=%q tool_calls=%d", response.Model, response.Text, len(response.ToolCalls))
}

func TestGrokBuildIsConfigOnlyAndNotAPIKeyAutoDiscovered(t *testing.T) {
	setGrokBuildTestEnv(t, "")
	for _, name := range []string{
		"APTEVA_MANAGED_LLM_URL", "OPENCODE_GO_API_KEY", "FIREWORKS_API_KEY",
		"ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY",
		"VENICE_API_KEY", "NVIDIA_API_KEY", "OLLAMA_HOST", "CORE_PROVIDER",
	} {
		t.Setenv(name, "")
	}
	pool, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: "grok-build"}}})
	if err != nil || pool.Get("grok-build") == nil {
		t.Fatalf("explicit Grok Build config: pool=%v err=%v", pool, err)
	}
	if _, err := buildProviderPool(&Config{}); err == nil {
		t.Fatal("GROK_BUILD_ACCESS_TOKEN alone was auto-discovered; provider must be server-configured")
	}
}

func TestGrokBuildResponsesHeadersStreamingToolsContinuationAndUsage(t *testing.T) {
	type capturedRequest struct {
		headers http.Header
		body    map[string]any
	}
	var mu sync.Mutex
	var requests []capturedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests = append(requests, capturedRequest{headers: r.Header.Clone(), body: body})
		requestNumber := len(requests)
		mu.Unlock()
		if requestNumber == 1 {
			writeResponsesStream(w,
				`{"type":"response.reasoning_summary_text.delta","delta":"checking"}`,
				`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"first","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","item_id":"fc_a","delta":"{\"value\":1}"}`,
				`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_a","call_id":"call_a","name":"first","arguments":"{\"value\":1}","status":"completed"}}`,
				`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_b","call_id":"call_b","name":"second","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","item_id":"fc_b","delta":"{\"value\":2}"}`,
				`{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_b","call_id":"call_b","name":"second","arguments":"{\"value\":2}","status":"completed"}}`,
				`{"type":"response.completed","response":{"usage":{"input_tokens":40,"output_tokens":8,"input_tokens_details":{"cached_tokens":17}}}}`,
			)
			return
		}
		writeResponsesStream(w,
			`{"type":"response.output_text.delta","delta":"done"}`,
			`{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done"}]}}`,
			`{"type":"response.completed","response":{"usage":{"input_tokens":52,"output_tokens":3,"input_tokens_details":{"cached_tokens":31,"cache_write_tokens":4}}}}`,
		)
	}))
	defer srv.Close()

	setGrokBuildTestEnv(t, srv.URL+"/v1")
	parallel := true
	provider := NewGrokBuildProvider("grok-access").(*OpenAINativeProvider)
	provider.modelCapabilities = map[string]ModelCapabilities{
		"grok-4.6": {
			DefaultReasoningLevel: "high",
			SupportedReasoningLevels: []ModelReasoningCapability{
				{Effort: "low"}, {Effort: "medium"}, {Effort: "high"}, {Effort: "xhigh"},
			},
			SupportsParallelToolCalls: &parallel,
		},
	}
	provider = provider.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).(*OpenAINativeProvider)
	tools := []NativeTool{
		{Name: "first", Parameters: map[string]any{"type": "object"}},
		{Name: "second", Parameters: map[string]any{"type": "object"}},
	}
	var thinking strings.Builder
	first, err := provider.Chat(context.Background(), []Message{
		{Role: "system", Content: "Use the supplied functions."},
		{Role: "user", Content: "Call both."},
	}, "grok-4.6", tools, nil, func(chunk string) { thinking.WriteString(chunk) }, nil)
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if thinking.String() != "checking" || first.Reasoning != "checking" {
		t.Fatalf("reasoning callback=%q response=%q", thinking.String(), first.Reasoning)
	}
	if len(first.ToolCalls) != 2 || first.ToolCalls[0].ID != "call_a" || first.ToolCalls[1].ID != "call_b" {
		t.Fatalf("parallel tool calls = %+v", first.ToolCalls)
	}
	if first.Usage.PromptTokens != 40 || first.Usage.CompletionTokens != 8 || first.Usage.CachedTokens != 17 {
		t.Fatalf("first usage = %+v", first.Usage)
	}
	if first.ProviderState == nil || first.ProviderState.Provider != grokBuildStateProvider {
		t.Fatalf("provider state = %#v", first.ProviderState)
	}

	second, err := provider.Chat(context.Background(), []Message{
		{Role: "system", Content: "Use the supplied functions."},
		{Role: "user", Content: "Call both."},
		{Role: "assistant", ToolCalls: first.ToolCalls, ProviderState: first.ProviderState},
		{Role: "user", ToolResults: []ToolResult{
			{CallID: "call_a", ToolName: "first", Content: `{"ok":1}`},
			{CallID: "call_b", ToolName: "second", Content: `{"ok":2}`},
		}},
	}, "grok-4.6", tools, nil, nil, nil)
	if err != nil {
		t.Fatalf("continuation Chat: %v", err)
	}
	if second.Text != "done" || second.Usage.CachedTokens != 31 || second.Usage.CacheWriteTokens != 4 {
		t.Fatalf("second response = %+v", second)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	for i, request := range requests {
		headers := request.headers
		for name, want := range map[string]string{
			"Authorization":            "Bearer grok-access",
			"X-XAI-Token-Auth":         "xai-grok-cli",
			"x-grok-client-version":    grokBuildClientVersion,
			"x-grok-client-identifier": "apteva-core",
			"x-grok-client-mode":       "headless",
			"x-authenticateresponse":   "authenticate-response",
			"x-userid":                 "user-1",
			"x-email":                  "agent@example.com",
			"x-grok-model-override":    "grok-4.6",
		} {
			if got := headers.Get(name); got != want {
				t.Fatalf("request %d header %s = %q, want %q", i+1, name, got, want)
			}
		}
		if got := headers.Get("ChatGPT-Account-ID"); got != "" {
			t.Fatalf("request %d leaked ChatGPT-Account-ID = %q", i+1, got)
		}
		if _, ok := request.body["prompt_cache_key"]; ok {
			t.Fatalf("request %d sent prompt_cache_key: %#v", i+1, request.body)
		}
		if _, ok := request.body["prompt_cache_retention"]; ok {
			t.Fatalf("request %d sent prompt_cache_retention: %#v", i+1, request.body)
		}
		if request.body["store"] != false || request.body["parallel_tool_calls"] != true {
			t.Fatalf("request %d stateless/parallel fields = %#v", i+1, request.body)
		}
		reasoning, _ := request.body["reasoning"].(map[string]any)
		if reasoning["effort"] != "high" || reasoning["summary"] != "concise" {
			t.Fatalf("request %d reasoning = %#v", i+1, reasoning)
		}
	}
	input, _ := requests[1].body["input"].([]any)
	var calls, outputs int
	for _, raw := range input {
		item, _ := raw.(map[string]any)
		switch item["type"] {
		case "function_call":
			calls++
		case "function_call_output":
			outputs++
		}
	}
	if calls != 2 || outputs != 2 {
		t.Fatalf("continuation input has %d calls and %d outputs: %#v", calls, outputs, input)
	}
}

func TestGrokBuildReasoningUsesOfficialCapabilityShape(t *testing.T) {
	// Grok Build's upstream Responses fixture asserts that reasoning effort is
	// nested at /reasoning/effort for every supported effort. Keep this test
	// aligned with xai-org/grok-build@4247f661, responses_tests.rs:
	// test_responses_request_carries_reasoning_effort_nested.
	setGrokBuildTestEnv(t, "")
	provider := NewGrokBuildProvider("token").(*OpenAINativeProvider)
	provider.modelCapabilities = map[string]ModelCapabilities{
		"grok-4.6": {
			DefaultReasoningLevel: "high",
			SupportedReasoningLevels: []ModelReasoningCapability{
				{Effort: "xhigh"}, {Effort: "high"}, {Effort: "medium"}, {Effort: "low"},
			},
		},
	}
	if got := provider.requestReasoning("grok-4.6"); got == nil || got.Effort != "high" || got.Summary != "concise" {
		t.Fatalf("auto reasoning = %#v, want official Grok 4.6 high/concise shape", got)
	}
	xhigh := provider.WithReasoning(ReasoningSettings{Level: ReasoningXHigh}).(*OpenAINativeProvider)
	if got := xhigh.requestReasoning("grok-4.6"); got == nil || got.Effort != "xhigh" || got.Summary != "concise" {
		t.Fatalf("xhigh reasoning = %#v", got)
	}
	none := provider.WithReasoning(ReasoningSettings{Level: ReasoningNone}).(*OpenAINativeProvider)
	if got := none.requestReasoning("grok-4.6"); got == nil || got.Effort != "low" || got.Summary != "" {
		t.Fatalf("unsupported none reasoning = %#v, want capability-clamped low", got)
	}
}

func TestGrokBuildProactiveRefreshUpdatesSharedIdentityAndBaseURL(t *testing.T) {
	var inferenceRequests atomic.Int32
	inference := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inferenceRequests.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer refreshed-token" {
			t.Errorf("authorization = %q", got)
		}
		if got := r.Header.Get("x-userid"); got != "refreshed-user" {
			t.Errorf("x-userid = %q", got)
		}
		if got := r.Header.Get("x-email"); got != "refreshed@example.com" {
			t.Errorf("x-email = %q", got)
		}
		writeResponsesStream(w,
			`{"type":"response.output_text.delta","delta":"ok"}`,
			`{"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1}}}`,
		)
	}))
	defer inference.Close()

	var refreshRequests atomic.Int32
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		refreshRequests.Add(1)
		if r.URL.Path != "/api/providers/connection-1/auth/runtime-token" {
			t.Errorf("runtime path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer server-key" {
			t.Errorf("server authorization = %q", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]string{
			"provider": "grok-build", "access_token": "refreshed-token",
			"account_id": "account-2", "account_email": "refreshed@example.com",
			"user_id": "refreshed-user", "principal_type": "organization",
			"principal_id": "principal-2", "base_url": inference.URL + "/v1",
		})
	}))
	defer runtime.Close()

	setGrokBuildTestEnv(t, "https://unused.invalid/v1")
	t.Setenv("SERVER_URL", runtime.URL)
	t.Setenv("APTEVA_API_KEY", "server-key")
	provider := NewGrokBuildProvider("old-token").(*OpenAINativeProvider)
	clone := provider.WithReasoning(ReasoningSettings{Level: ReasoningLow}).(*OpenAINativeProvider)
	if _, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "grok-4.6", nil, nil, nil, nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if refreshRequests.Load() != 1 || inferenceRequests.Load() != 1 {
		t.Fatalf("refresh requests=%d inference requests=%d", refreshRequests.Load(), inferenceRequests.Load())
	}
	state := clone.currentSessionSnapshot()
	if state.AccessToken != "refreshed-token" || state.UserID != "refreshed-user" || state.AccountEmail != "refreshed@example.com" || state.PrincipalType != "organization" || state.PrincipalID != "principal-2" || state.ResponsesURL != inference.URL+"/v1/responses" {
		t.Fatalf("shared refreshed state = %+v", state)
	}
}

func TestGrokBuildForcesRefreshAndRetriesAuthFailures(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var proactive, forced, inference atomic.Int32
			var server *httptest.Server
			server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case strings.HasSuffix(r.URL.Path, "/auth/runtime-token"):
					if r.URL.Query().Get("force") == "1" {
						forced.Add(1)
						_ = json.NewEncoder(w).Encode(map[string]string{
							"provider": "grok-build", "access_token": "fresh-token",
							"user_id": "fresh-user", "account_email": "fresh@example.com",
							"base_url": server.URL + "/v1",
						})
						return
					}
					proactive.Add(1)
					_ = json.NewEncoder(w).Encode(map[string]string{
						"provider": "grok-build", "access_token": "stale-token",
						"user_id": "stale-user", "base_url": server.URL + "/v1",
					})
				case r.URL.Path == "/v1/responses":
					inference.Add(1)
					if r.Header.Get("Authorization") != "Bearer fresh-token" {
						w.WriteHeader(status)
						_, _ = io.WriteString(w, `{"error":"expired"}`)
						return
					}
					if got := r.Header.Get("x-userid"); got != "fresh-user" {
						t.Errorf("retry x-userid = %q", got)
					}
					writeResponsesStream(w, `{"type":"response.completed","response":{"usage":{}}}`)
				default:
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()

			setGrokBuildTestEnv(t, server.URL+"/v1")
			t.Setenv("SERVER_URL", server.URL)
			t.Setenv("APTEVA_API_KEY", "server-key")
			provider := NewGrokBuildProvider("initial-token").(*OpenAINativeProvider)
			if _, err := provider.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "grok-4.6", nil, nil, nil, nil); err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if proactive.Load() != 1 || forced.Load() != 1 || inference.Load() != 2 {
				t.Fatalf("proactive=%d forced=%d inference=%d", proactive.Load(), forced.Load(), inference.Load())
			}
		})
	}
}

func TestGrokBuildConcurrentForcedRefreshIsCoalescedAcrossClones(t *testing.T) {
	var requests atomic.Int32
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
		_ = json.NewEncoder(w).Encode(map[string]string{
			"provider": "grok-build", "access_token": "same-token",
		})
	}))
	defer server.Close()

	setGrokBuildTestEnv(t, "")
	t.Setenv("SERVER_URL", server.URL)
	t.Setenv("APTEVA_API_KEY", "server-key")
	provider := NewGrokBuildProvider("same-token").(*OpenAINativeProvider)
	clones := make([]*OpenAINativeProvider, 20)
	levels := []ReasoningLevel{ReasoningAuto, ReasoningLow, ReasoningMedium, ReasoningHigh, ReasoningXHigh}
	for i := range clones {
		clones[i] = provider.WithReasoning(ReasoningSettings{Level: levels[i%len(levels)]}).(*OpenAINativeProvider)
	}
	var wg sync.WaitGroup
	errs := make(chan error, len(clones))
	for _, clone := range clones {
		wg.Add(1)
		go func(p *OpenAINativeProvider) {
			defer wg.Done()
			errs <- p.refreshRuntimeToken(context.Background(), true)
		}(clone)
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("refresh request did not start")
	}
	time.Sleep(25 * time.Millisecond)
	close(release)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := requests.Load(); got != 1 {
		t.Fatalf("refresh HTTP requests = %d, want 1", got)
	}
}

func TestGrokBuildHeadersDoNotLeakToOtherProviders(t *testing.T) {
	var captured []http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = append(captured, r.Header.Clone())
		if strings.HasSuffix(r.URL.Path, "/chat/completions") {
			writeResponsesStream(w, `{"choices":[{"delta":{"content":"ok"}}]}`, `[DONE]`)
			return
		}
		writeResponsesStream(w, `{"type":"response.completed","response":{"usage":{}}}`)
	}))
	defer srv.Close()

	openai := &OpenAINativeProvider{name: "openai", apiKey: "openai-key", responsesURL: srv.URL + "/responses"}
	if _, err := openai.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "gpt-test", nil, nil, nil, nil); err != nil {
		t.Fatalf("OpenAI Chat: %v", err)
	}
	codex := &OpenAINativeProvider{name: "openai-codex", apiKey: "codex-key", accountID: "account-1", responsesURL: srv.URL + "/responses", forceStoreFalse: true}
	if _, err := codex.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "gpt-test", nil, nil, nil, nil); err != nil {
		t.Fatalf("Codex Chat: %v", err)
	}
	xai := NewXAIProvider("xai-key").(*XAIProvider)
	xai.compat.url = srv.URL + "/chat/completions"
	if _, err := xai.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "grok-test", nil, nil, nil, nil); err != nil {
		t.Fatalf("xAI Chat: %v", err)
	}
	if len(captured) != 3 {
		t.Fatalf("captured requests = %d", len(captured))
	}
	for i, headers := range captured {
		for _, name := range []string{
			"X-XAI-Token-Auth", "x-grok-client-version", "x-grok-client-identifier",
			"x-grok-client-mode", "x-authenticateresponse", "x-userid", "x-email",
			"x-grok-model-override",
		} {
			if values, ok := headers[http.CanonicalHeaderKey(name)]; ok && len(values) > 0 {
				t.Fatalf("request %d leaked %s = %#v", i+1, name, values)
			}
		}
	}
	if got := captured[1].Get("ChatGPT-Account-ID"); got != "account-1" {
		t.Fatalf("Codex account header changed = %q", got)
	}
}
