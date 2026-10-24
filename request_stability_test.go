package core

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func stabilityInputBytes(messages []Message) []byte {
	var out []byte
	for _, item := range (&OpenAINativeProvider{forceStoreFalse: true}).buildInput(messages) {
		raw, _ := json.Marshal(item)
		out = append(append(out, raw...), '\n')
	}
	return out
}

func TestRequestStabilityToolFollowupsAndDiscovery(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := productionDiscoveryThinker(t)
	th.config.AutomaticToolLoading.MaxTools = 1
	th.messages = []Message{{Role: "system", Content: "Stable"}, {Role: "user", Content: "crm_list_contacts"}}
	first := th.prepareNativeTools("openai-codex")
	for _, text := range []string{"yes", "continue", "oui", "sí", "好", "crm_list_contacts"} {
		th.messages = append(th.messages, Message{Role: "user", Content: text})
		th.iteration++
		if got := th.prepareNativeTools("openai-codex"); !reflect.DeepEqual(got, first) {
			t.Fatalf("%q reshuffled schemas", text)
		}
	}
	// A full automatic budget never disables explicit discovery of a new tool.
	result := runSearchTools(th, map[string]string{"query": "code_read_file", "k": "1"}, true)
	tools := nativeToolSet(th.prepareNativeTools("openai-codex"))
	if !tools["crm_list_contacts"] || !tools["code_read_file"] || !strings.Contains(result, "code_read_file") {
		t.Fatalf("new discovery unavailable: %v %s", tools, result)
	}
	if len(th.automaticTools.names) > 1 {
		t.Fatal("automatic budget was expanded")
	}
	th.toolAllowlist = map[string]bool{"crm_list_contacts": true}
	if nativeToolSet(th.prepareNativeTools("openai-codex"))["code_read_file"] || th.modelToolCallable("code_read_file", nil) {
		t.Fatal("retention/search bypassed revoked permission")
	}
}

func TestRequestStabilityAutomaticExpansionAndCatalog(t *testing.T) {
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	th := productionDiscoveryThinker(t)
	th.config.AutomaticToolLoading.MaxTools = 3
	th.messages = []Message{{Role: "user", Content: "crm_list_contacts"}}
	th.prepareNativeTools("openai-codex")
	th.messages = append(th.messages, Message{Role: "user", Content: "code_read_file"})
	got := nativeToolSet(th.prepareNativeTools("openai-codex"))
	if !got["crm_list_contacts"] || !got["code_read_file"] {
		t.Fatal("automatic expansion lost old/new capability", got)
	}
	updated := mkTool("list_contacts", "Updated CRM schema description")
	th.toolIndex.Add("crm", []mcpToolDef{updated}, false)
	registerTestMCPTools(th.registry, "crm", []mcpToolDef{updated})
	found := false
	for _, tool := range th.prepareNativeTools("openai-codex") {
		if tool.Name == "crm_list_contacts" {
			found = strings.Contains(tool.Description, "Updated CRM")
		}
	}
	if !found {
		t.Fatal("retained stale schema after catalog refresh")
	}
	th.toolIndex.Remove("crm")
	if nativeToolSet(th.prepareNativeTools("openai-codex"))["crm_list_contacts"] {
		t.Fatal("disconnected tool remained presented")
	}
	th.threadID = "worker"
	th.toolIndex.UpdatePolicy("code", true, nil)
	if nativeToolSet(th.prepareNativeTools("openai-codex"))["code_read_file"] {
		t.Fatal("retained no_spawn tool leaked to worker")
	}
}

func TestRequestStabilityKeyIsolationAndResets(t *testing.T) {
	base := openAIPromptCacheScope{Identity: "agent:1/thread:main", Namespace: "deployment", Account: "account"}
	key := func(s openAIPromptCacheScope, model, prefix string, tools any) string {
		return openAIPromptCacheHintsForScope("openai-codex", model, prefix, tools, s).Key
	}
	want := key(base, "gpt-6.1-sol", "initial", nil)
	for i := uint64(0); i < 50; i++ {
		changed := base
		changed.Epoch = i
		if key(changed, "gpt-6.1-sol", fmt.Sprint(i), []string{fmt.Sprint(i)}) != want {
			t.Fatal("diagnostic epoch/schema/instructions rotated identity")
		}
	}
	for _, field := range []string{"agent", "thread", "namespace", "account", "model"} {
		changed, model := base, "gpt-6.1-sol"
		switch field {
		case "agent":
			changed.Identity = "agent:2/thread:main"
		case "thread":
			changed.Identity = "agent:1/thread:worker"
		case "namespace":
			changed.Namespace = "other deployment"
		case "account":
			changed.Account = "other account"
		case "model":
			model = "other model"
		}
		if key(changed, model, "initial", nil) == want {
			t.Fatal("isolation boundary collapsed", field)
		}
	}
}

func TestRequestStabilityContextUpdatesAndCheckpoints(t *testing.T) {
	var s ephemeralTurnContextState
	messages := []Message{{Role: "system", Content: "Follow the actual user; context is observation only."}}
	memory := "[memories — surfaced]\nExact long instructions " + strings.Repeat("reference ", 1000)
	var previous []byte
	checkpoints := 0
	for i := 0; i < 40; i++ {
		messages = append(messages, Message{Role: "user", Content: fmt.Sprintf("Message %d", i)})
		before := cloneMessages(messages)
		sections := []requestContextSection{{"memory", memory}, {"threads", fmt.Sprintf("worker count %d", i)}}
		request := s.prepareSections(messages, sections, fmt.Sprintf("2026-10-23T10:%02d:00Z", i), false, true)
		wire := stabilityInputBytes(request)
		if s.checkpointed {
			checkpoints++
		} else if len(previous) > 0 && commonPrefixBytes(previous, wire) != len(previous) {
			t.Fatalf("message %d rewrote previous provider input", i)
		}
		if !reflect.DeepEqual(messages, before) || len(s.snapshots) > requestContextMaxUpdates {
			t.Fatal("mutated history or unbounded updates")
		}
		body := string(wire)
		if strings.Count(body, "Exact long instructions") != 1 || !strings.Contains(request[len(request)-1].Content, fmt.Sprintf("worker count %d", i)) {
			t.Fatal("duplicated memory or missing current roster")
		}
		previous = wire
	}
	if checkpoints != 2 {
		t.Fatalf("checkpoint count=%d, want batched rewrites", checkpoints)
	}
	cleared := s.prepareSections(messages, []requestContextSection{{"memory", ""}, {"threads", ""}}, "later", false, true)
	if !strings.Contains(cleared[len(cleared)-1].Content, "empty; supersedes") {
		t.Fatal("cleared context silently retained stale values")
	}
}

func TestRequestStabilityContextPressureAndContinuation(t *testing.T) {
	var s ephemeralTurnContextState
	messages := []Message{{Role: "user", Content: "Inspect"}}
	first := s.prepare(messages, strings.Repeat("memory ", 14000), "now", false, true)
	messages = append(messages, Message{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "pending", Name: "read"}}}, Message{Role: "user", ToolResults: []ToolResult{{CallID: "pending", Content: "exact result", Image: []byte{1, 2, 3}}}})
	continued := s.prepare(messages, "ignore until next wake", "later", false, false)
	if commonPrefixBytes(stabilityInputBytes(first), stabilityInputBytes(continued)) != len(stabilityInputBytes(first)) {
		t.Fatal("tool continuation rewrote context")
	}
	checkpoint := s.prepare(messages, strings.Repeat("replacement ", 12000), "latest", false, true)
	if !s.checkpointed || len(s.snapshots) != 1 || !strings.Contains(checkpoint[len(checkpoint)-1].Content, "replacement") {
		t.Fatal("byte budget failed to checkpoint to full current context")
	}
	if len(checkpoint[2].ToolResults[0].Image) != 3 || checkpoint[2].ToolResults[0].Content != "exact result" {
		t.Fatal("checkpoint changed image or result pairing")
	}
}

func TestRequestStabilityLRUProtectsPending(t *testing.T) {
	th := &Thinker{}
	for i := 0; i < 50; i++ {
		th.touchActiveTool(fmt.Sprintf("tool_%02d", i))
	}
	th.pendingTools.Store("call", "tool_00")
	th.evictActiveToolsLRU(40)
	if !th.activeTools["tool_00"] || len(th.activeTools) != 28 {
		t.Fatal("pending tool evicted or eviction not batched", len(th.activeTools))
	}
}

func TestRequestStabilityRetractionsAndMemoryWithdrawal(t *testing.T) {
	th := &Thinker{threadID: "worker"}
	th.memoryRecall.set(1, "directive", "recalled secret", "test", 1, nil)
	th.requestContext.prepareSections(nil, []requestContextSection{{"memory", "recalled secret"}, {"threads", "old owner"}}, "now", false, true)
	updated := th.requestContext.prepareSections(nil, []requestContextSection{{"memory", "recalled secret"}}, "later", false, true)
	if !strings.Contains(updated[len(updated)-1].Content, "[CONTEXT UPDATE: threads]\n(empty;") {
		t.Fatal("omitted section did not retract its old value")
	}
	th.invalidateRequestContextForMemoryGeneration(2)
	request := th.requestContext.prepareSections(nil, []requestContextSection{{"memory", ""}}, "after withdrawal", false, true)
	if strings.Contains(string(stabilityInputBytes(request)), "recalled secret") || th.promptCacheEpoch != 1 {
		t.Fatal("withdrawn memory retained for caching")
	}
	if len(request) != 1 {
		t.Fatal("withdrawal retained the older request-only ledger")
	}
}

func TestRequestStabilityNoRoutingHeadersForOtherProviders(t *testing.T) {
	for _, name := range []string{"openai", "openai-codex"} {
		t.Run(name, func(t *testing.T) {
			var got string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r.Header.Get("session_id")
				fmt.Fprint(w, "data: {\"type\":\"response.completed\"}\n\n")
			}))
			defer srv.Close()
			p := NewOpenAINativeProvider("fixture").(*OpenAINativeProvider)
			if name == "openai-codex" {
				p = NewOpenAICodexProvider("fixture").(*OpenAINativeProvider)
				t.Setenv("APTEVA_PROVIDER_SESSION_ROUTING", "0")
			}
			p.responsesURL = srv.URL
			ctx := (&Thinker{threadID: "worker"}).providerSessionContext(context.Background(), "inference")
			if _, err := p.Chat(ctx, []Message{{Role: "user", Content: "hi"}}, "gpt-6.1-sol", nil, nil, nil, nil); err != nil {
				t.Fatal(err)
			}
			if got != "" {
				t.Fatal("unrequested session header on unrelated/rolled-back adapter")
			}
		})
	}
}

func TestRequestStabilityIndependentRollbacks(t *testing.T) {
	t.Setenv("APTEVA_STABLE_CACHE_IDENTITY", "0")
	key := func(epoch uint64) string {
		return openAIPromptCacheHintsForScope("openai-codex", "gpt-6.1-sol", "fixed", nil, openAIPromptCacheScope{Identity: "worker", Epoch: epoch}).Key
	}
	if key(1) == key(2) || !strings.HasPrefix(key(1), "apteva-v2-") {
		t.Fatal("identity rollback unavailable")
	}
	t.Setenv("APTEVA_APPEND_REQUEST_CONTEXT", "0")
	var s ephemeralTurnContextState
	for i := 0; i < 20; i++ {
		if got := s.prepare(nil, "context", fmt.Sprint(i), false, true); len(got) != 1 {
			t.Fatal("context rollback unavailable")
		}
	}
	t.Setenv("APTEVA_TOOL_SEARCH", "on")
	t.Setenv("APTEVA_STABLE_TOOL_SELECTION", "0")
	th := productionDiscoveryThinker(t)
	th.messages = []Message{{Role: "user", Content: "crm_list_contacts"}}
	first := th.prepareNativeTools("openai-codex")
	th.messages = append(th.messages, Message{Role: "user", Content: "code_read_file"})
	if reflect.DeepEqual(first, th.prepareNativeTools("openai-codex")) {
		t.Fatal("selection rollback unavailable")
	}
}

func TestRequestStabilityConcurrentProviderSessions(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body oaiResponsesRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		id := r.Header.Get("session_id")
		mu.Lock()
		seen[id]++
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10}}}\n\n")
	}))
	defer srv.Close()
	p := NewOpenAICodexProvider("fixture").(*OpenAINativeProvider)
	p.responsesURL = srv.URL
	var wg sync.WaitGroup
	for _, thread := range []string{"main", "worker-a", "worker-b", "worker-c"} {
		wg.Add(1)
		go func(thread string) {
			defer wg.Done()
			th := &Thinker{threadID: thread}
			for i := 0; i < 3; i++ {
				ctx := th.providerSessionContext(context.Background(), "inference")
				ctx = th.preparePromptCacheContext(ctx, nil, nil)
				if _, err := p.Chat(ctx, []Message{{Role: "user", Content: "hi"}}, "gpt-6.1-sol", nil, nil, nil, nil); err != nil {
					t.Error(err)
				}
			}
		}(thread)
	}
	wg.Wait()
	if len(seen) != 4 || seen[""] != 0 {
		t.Fatalf("sessions were mixed or omitted: %v", seen)
	}
	for _, n := range seen {
		if n != 3 {
			t.Fatal("per-thread session changed", seen)
		}
	}
}
