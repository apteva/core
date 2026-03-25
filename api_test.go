package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func newTestAPI() (*APIServer, *Thinker) {
	bus := NewEventBus()
	t := &Thinker{
		apiKey:    "test",
		messages:  []Message{{Role: "system", Content: "test"}},
		bus:       bus,
		sub:       bus.Subscribe("main", 100),
		pause:     make(chan bool),
		quit:      make(chan struct{}),
		rate:      RateSlow,
		agentRate: RateSlow,
		memory:    &MemoryStore{path: "/dev/null"},
		config:    &Config{Directive: "test directive"},
		apiLog:    &[]APIEvent{},
		apiMu:     &sync.RWMutex{},
		apiNotify: make(chan struct{}, 1),
		threadID:  "main",
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

func TestAPI_Threads_MainOnly(t *testing.T) {
	api, _ := newTestAPI()
	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var body []map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 1 {
		t.Fatalf("expected 1 thread (main), got %d", len(body))
	}
	if body[0]["id"] != "main" {
		t.Errorf("expected id 'main', got %v", body[0]["id"])
	}
}

func TestAPI_Threads_WithSubThreads(t *testing.T) {
	api, thinker := newTestAPI()
	thinker.threads.Spawn("test-thread", "test prompt", []string{"web"})
	defer thinker.threads.Kill("test-thread")

	req := httptest.NewRequest("GET", "/threads", nil)
	w := httptest.NewRecorder()
	api.threads(w, req)

	var body []map[string]any
	json.Unmarshal(w.Body.Bytes(), &body)
	if len(body) != 2 {
		t.Fatalf("expected 2 threads (main + test-thread), got %d", len(body))
	}
	if body[0]["id"] != "main" {
		t.Errorf("expected first thread 'main', got %v", body[0]["id"])
	}
	if body[1]["id"] != "test-thread" {
		t.Errorf("expected second thread 'test-thread', got %v", body[1]["id"])
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
	items := thinker.drainEvents()
	if len(items) != 1 {
		t.Fatalf("expected 1 item in events, got %d", len(items))
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

// Full HTTP server integration test — verifies routing and real HTTP round-trips
func TestAPI_FullServer(t *testing.T) {
	api, _ := newTestAPI()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", api.health)
	mux.HandleFunc("/status", api.status)
	mux.HandleFunc("/threads", api.threads)
	mux.HandleFunc("/event", api.postEvent)
	mux.HandleFunc("/config", api.config)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// Health
	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("health: expected 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Status
	resp, err = http.Get(srv.URL + "/status")
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	var status map[string]any
	json.NewDecoder(resp.Body).Decode(&status)
	resp.Body.Close()
	if _, ok := status["uptime_seconds"]; !ok {
		t.Error("status missing uptime_seconds")
	}

	// Post event
	payload, _ := json.Marshal(map[string]string{"message": "hello"})
	resp, err = http.Post(srv.URL+"/event", "application/json", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("event: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("event: expected 200, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Config round-trip
	payload, _ = json.Marshal(map[string]string{"directive": "full server test"})
	req, _ := http.NewRequest("PUT", srv.URL+"/config", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("config PUT: %v", err)
	}
	resp.Body.Close()

	resp, err = http.Get(srv.URL + "/config")
	if err != nil {
		t.Fatalf("config GET: %v", err)
	}
	var cfg map[string]string
	json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if cfg["directive"] != "full server test" {
		t.Errorf("config round-trip failed, got %q", cfg["directive"])
	}

	t.Log("All endpoints working via real HTTP server")
}
