package core

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

func TestAPIMemoryScopeCompatibilityAndProvenance(t *testing.T) {
	api, thinker := newTestAPI()
	m, _ := newHistoryTestStore(t)
	thinker.memory = m
	source := MemorySource{ThreadID: "chat-a", EntryID: "source-id", Role: "user"}
	if _, err := m.RememberWithID("stable", "cobalt", nil, 1, MemoryMetadata{Scope: "chat-a", Sources: []MemorySource{source}}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ body, scope string }{
		{`{"id":"stable","content":"updated"}`, "chat-a"},
		{`{"id":"stable","content":"published","scope":""}`, ""},
		{`{"id":"stable","content":"private","scope":"chat-a"}`, "chat-a"},
	} {
		w := httptest.NewRecorder()
		api.memoryRoot(w, httptest.NewRequest(http.MethodPost, "/memory", strings.NewReader(tc.body)))
		if w.Code != 200 {
			t.Fatal(w.Body.String())
		}
		active := m.Active()
		if len(active) != 1 || active[0].Scope != tc.scope || len(active[0].Sources) != 1 || active[0].Sources[0] != source {
			t.Fatalf("scope/provenance lost: %+v", active)
		}
	}
	w := httptest.NewRecorder()
	api.memoryRoot(w, httptest.NewRequest(http.MethodGet, "/memory", nil))
	var items []memoryListItem
	if err := json.Unmarshal(w.Body.Bytes(), &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Scope != "chat-a" || len(items[0].Sources) != 1 {
		t.Fatalf("missing API provenance: %s", w.Body.String())
	}
	w = httptest.NewRecorder()
	api.memoryRoot(w, httptest.NewRequest(http.MethodPost, "/memory", strings.NewReader(`{"content":"invalid","scope":"../bad"}`)))
	if w.Code != 400 {
		t.Fatal("accepted invalid scope")
	}
}

func TestAPIMemoryForgetSourceAuthAndPersistence(t *testing.T) {
	api, thinker := newTestAPI()
	m, _ := newHistoryTestStore(t)
	thinker.memory = m
	api.apiKey = "test-key"
	if _, err := m.Remember("private", nil, 1, MemoryMetadata{Scope: "chat-a"}); err != nil {
		t.Fatal(err)
	}
	body := `{"thread_id":"chat-a","reason":"requested"}`
	w := httptest.NewRecorder()
	api.apiAuth(api.memoryForgetSource)(w, httptest.NewRequest(http.MethodPost, "/memory/forget-source", strings.NewReader(body)))
	if w.Code != 401 || m.Count() != 1 {
		t.Fatal("unauthenticated forgetting allowed")
	}
	req := httptest.NewRequest(http.MethodPost, "/memory/forget-source", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	w = httptest.NewRecorder()
	api.apiAuth(api.memoryForgetSource)(w, req)
	if w.Code != 200 || m.Count() != 0 || !m.SourceForgotten("chat-a") {
		t.Fatalf("forget failed: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "not erased") {
		t.Fatal("missing logical deletion disclosure")
	}
}

func TestAPIMemoryPolicyConfigRoundTripAndRollback(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")
	put := func(body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest(http.MethodPut, "/config", strings.NewReader(body)))
		return w
	}
	if w := put(`{"memory_policy":{"exclude_threads":["private-chat"]}}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	reloaded := &Config{path: thinker.config.path}
	if err := reloaded.load(); err != nil {
		t.Fatal(err)
	}
	if p := reloaded.GetMemoryPolicy(); len(p.ExcludeThreads) != 1 || p.ExcludeThreads[0] != "private-chat" {
		t.Fatalf("policy not persisted: %+v", p)
	}
	if w := put(`{}`); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if len(thinker.config.GetMemoryPolicy().ExcludeThreads) != 1 {
		t.Fatal("omitted policy cleared setting")
	}
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest(http.MethodGet, "/config", nil))
	if !strings.Contains(w.Body.String(), `"exclude_threads":["private-chat"]`) {
		t.Fatalf("GET missing policy: %s", w.Body.String())
	}
	if w := put(`{"memory_policy":{"exclude_threads":["../bad"]}}`); w.Code != 400 {
		t.Fatal("accepted invalid thread")
	}
	thinker.config.path = t.TempDir()
	if w := put(`{"memory_policy":{"exclude_threads":[]}}`); w.Code != 500 {
		t.Fatalf("expected save failure: %d", w.Code)
	}
	if len(thinker.config.GetMemoryPolicy().ExcludeThreads) != 1 {
		t.Fatal("failed persistence changed memory policy")
	}
}
