package core

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestOpenCodeGoSessionHeadersSurviveModelSwitchAndOptionalRetry(t *testing.T) {
	var ids []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("x-opencode-session")
		if id == "" || !strings.HasPrefix(r.UserAgent(), "apteva-core/") {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"type":"MissingSessionID"}}`)
			return
		}
		ids = append(ids, id)
		if r.Header.Get("X-Test-Preserved") != "yes" {
			t.Error("lost caller header")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] == "kimi-k3" && body["reasoning_effort"] != nil {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"message":"reasoning_effort not supported"}}`)
			return
		}
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()
	p := newOpenCodeGoTestProvider(t, srv.URL)
	owner := &Thinker{threadID: "main"}
	headers := map[string]string{"X-Test-Preserved": "yes", "X-OpenCode-Session": "must-be-overridden"}
	ctx := withOpenAICompatRequestOptions(owner.providerSessionContext(context.Background(), "inference"), openAICompatRequestOptions{Headers: headers})
	for _, model := range []string{"glm-5.2", "kimi-k3", "kimi-k3", "glm-5.2"} {
		clone := p.WithReasoning(ReasoningSettings{Level: ReasoningHigh}).WithBuiltins(nil)
		if _, err := clone.Chat(ctx, []Message{{Role: "user", Content: "test"}}, model, nil, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if len(ids) != 5 {
		t.Fatalf("requests=%d, want 5 including retry", len(ids))
	}
	for _, id := range ids {
		if id != providerSessionFromContext(ctx) {
			t.Fatalf("session changed: %v", ids)
		}
	}
	if headers["X-OpenCode-Session"] != "must-be-overridden" || len(headers) != 2 {
		t.Fatal("caller headers mutated")
	}
}

func TestProviderSessionStableAndIsolated(t *testing.T) {
	t.Setenv("AGENT_ID", "1118")
	t.Setenv("SERVER_URL", "http://deployment-a")
	a := &Thinker{threadID: "main"}
	id := providerSessionFromContext(a.providerSessionContext(nil, "inference"))
	a.promptCacheEpoch = 99
	if got := providerSessionFromContext(a.providerSessionContext(nil, "inference")); got != id {
		t.Fatal("cache epoch changed routing")
	}
	restarted := &Thinker{threadID: "main"}
	if got := providerSessionFromContext(restarted.providerSessionContext(nil, "inference")); got != id {
		t.Fatal("restart changed routing")
	}
	worker := &Thinker{threadID: "worker"}
	if providerSessionFromContext(worker.providerSessionContext(nil, "inference")) == id {
		t.Fatal("workers share routing")
	}
	if providerSessionFromContext(a.providerSessionContext(nil, "session-compaction")) == id {
		t.Fatal("auxiliary shares foreground routing")
	}
	t.Setenv("AGENT_ID", "1119")
	if providerSessionFromContext(a.providerSessionContext(nil, "inference")) == id {
		t.Fatal("agents share routing")
	}
	t.Setenv("AGENT_ID", "1118")
	t.Setenv("SERVER_URL", "http://deployment-b")
	if providerSessionFromContext(a.providerSessionContext(nil, "inference")) == id {
		t.Fatal("deployments share routing")
	}
}

func TestOpenCodeGoSessionsConcurrentAndStandalone(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		seen[body["model"].(string)] = r.Header.Get("x-opencode-session")
		mu.Unlock()
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()
	p := newOpenCodeGoTestProvider(t, srv.URL)
	var wg sync.WaitGroup
	for _, name := range []string{"main", "worker"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			owner := &Thinker{threadID: name}
			_, err := p.Chat(owner.providerSessionContext(nil, "inference"), nil, name, nil, nil, nil, nil)
			if err != nil {
				t.Error(err)
			}
		}(name)
	}
	wg.Wait()
	for _, model := range []string{"standalone-1", "standalone-2"} {
		_, err := p.WithReasoning(ReasoningSettings{}).Chat(context.Background(), nil, model, nil, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	if seen["main"] == "" || seen["worker"] == "" || seen["main"] == seen["worker"] {
		t.Fatal("concurrent sessions not isolated")
	}
	if seen["standalone-1"] == "" || seen["standalone-1"] != seen["standalone-2"] {
		t.Fatal("standalone session missing or unstable")
	}
}

func TestOpenCodeGoAuxiliarySessionHeaders(t *testing.T) {
	var id string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id = r.Header.Get("x-opencode-session")
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()
	owner := &Thinker{threadID: "worker", provider: newOpenCodeGoTestProvider(t, srv.URL)}
	if _, err := owner.summarizePersistentSession("Earlier coding work"); err != nil {
		t.Fatal(err)
	}
	if id != providerSessionFromContext(owner.providerSessionContext(nil, "session-compaction")) {
		t.Fatal("persistent summary lacks its session")
	}
	if _, err := owner.summarizeRecoveryPrefix(context.Background(), owner.provider, []Message{{Role: "user", Content: "Earlier coding work"}}); err != nil {
		t.Fatal(err)
	}
	if id != providerSessionFromContext(owner.providerSessionContext(nil, "recovery-compaction")) {
		t.Fatal("recovery summary lacks its session")
	}
}

func TestOpenCodeGoSessionHeaderDoesNotLeakToOtherProviders(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-opencode-session") != "" {
			t.Error("OpenCode header leaked to another provider")
		}
		writeOpenCodeGoTestStream(w)
	}))
	defer srv.Close()
	p := &OpenAICompatProvider{name: "venice", url: srv.URL}
	owner := &Thinker{threadID: "main"}
	if _, err := p.Chat(owner.providerSessionContext(nil, "inference"), nil, "test", nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
}
