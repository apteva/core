package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestAPI() (*APIServer, *Thinker) {
	t := &Thinker{
		apiKey:    "test",
		messages:  []Message{{Role: "system", Content: "test"}},
		events:    make(chan ThinkEvent, 100),
		inbox:     make(chan string, 50),
		wakeup:    make(chan struct{}, 1),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    &MemoryStore{path: "/dev/null"},
		config:    &Config{Directive: "test directive"},
	}
	t.threads = NewThreadManager(t)
	api := &APIServer{thinker: t, startTime: time.Now()}
	return api, t
}

func TestAPI_Health(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	api.health(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]bool
	json.Unmarshal(w.Body.Bytes(), &body)
	if !body["ok"] {
		t.Error("expected ok: true")
	}
}

func TestAPI_Status(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.iteration = 5
	thinker.rate = RateFast
	thinker.model = ModelLarge

	req := httptest.NewRequest("GET", "/status", nil)
	w := httptest.NewRecorder()
	api.status(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["iteration"].(float64) != 5 {
		t.Errorf("expected iteration 5, got %v", body["iteration"])
	}
	if body["rate"] != "fast" {
		t.Errorf("expected rate fast, got %v", body["rate"])
	}
	if body["model"] != "large" {
		t.Errorf("expected model large, got %v", body["model"])
	}
}

func TestAPI_Threads_Empty(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body []any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 0 {
		t.Errorf("expected empty array, got %d items", len(body))
	}
}

func TestAPI_Threads_WithThreads(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.threads.Spawn("test-thread", "test prompt", []string{"web"}, true)
	defer thinker.threads.Kill("test-thread")

	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	var body []map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 1 {
		t.Fatalf("expected 1 thread, got %d", len(body))
	}
	if body[0]["id"] != "test-thread" {
		t.Errorf("expected id test-thread, got %v", body[0]["id"])
	}
}

func TestAPI_PostEvent(t *testing.T) {
	api, thinker := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"message": "test command"})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	// Check it was injected
	items := thinker.drainInbox()
	if len(items) != 1 {
		t.Fatalf("expected 1 item in inbox, got %d", len(items))
	}
	if items[0] != "[console] test command" {
		t.Errorf("expected '[console] test command', got %q", items[0])
	}
}

func TestAPI_PostEvent_EmptyMessage(t *testing.T) {
	api, _ := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"message": ""})
	req := httptest.NewRequest("POST", "/event", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 400 {
		t.Errorf("expected 400 for empty message, got %d", w.Code)
	}
}

func TestAPI_PostEvent_InvalidJSON(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("POST", "/event", bytes.NewReader([]byte("not json")))
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 400 {
		t.Errorf("expected 400 for invalid JSON, got %d", w.Code)
	}
}

func TestAPI_PostEvent_WrongMethod(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/event", nil)
	w := httptest.NewRecorder()
	api.postEvent(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405 for GET, got %d", w.Code)
	}
}

func TestAPI_Config_Get(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/config", nil)
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body map[string]string
	json.Unmarshal(w.Body.Bytes(), &body)
	if body["directive"] != "test directive" {
		t.Errorf("expected 'test directive', got %q", body["directive"])
	}
}

func TestAPI_Config_Put(t *testing.T) {
	api, thinker := newTestAPI()
	payload, _ := json.Marshal(map[string]string{"directive": "new directive"})
	req := httptest.NewRequest("PUT", "/config", bytes.NewReader(payload))
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if thinker.config.GetDirective() != "new directive" {
		t.Errorf("directive not updated, got %q", thinker.config.GetDirective())
	}
}

func TestAPI_Config_WrongMethod(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("DELETE", "/config", nil)
	w := httptest.NewRecorder()
	api.config(w, req)

	if w.Code != 405 {
		t.Errorf("expected 405, got %d", w.Code)
	}
}
