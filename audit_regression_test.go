package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAuditMemoryReset(t *testing.T) {
	api, th := newTestAPI()
	defer th.telemetry.Stop()
	ms := &MemoryStore{path: filepath.Join(t.TempDir(), "memory.jsonl"), byID: map[string]int{}, active: map[string]MemoryRecord{}}
	if err := ms.append(MemoryRecord{ID: "remembered", Content: "must be forgotten"}); err != nil {
		t.Fatal(err)
	}
	th.memory = ms
	before := ms.Generation()
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(`{"reset":{"memory":true}}`)))
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if ms.Count() != 0 || len(ms.Active()) != 0 || ms.Generation() == before {
		t.Fatalf("reset left count=%d active=%d generation=%d (before=%d)", ms.Count(), len(ms.Active()), ms.Generation(), before)
	}
}

func TestAuditCorruptMemoryLoad(t *testing.T) {
	if file := os.Getenv("AUDIT_CORRUPT_MEMORY_FILE"); file != "" {
		ms := &MemoryStore{path: file, byID: map[string]int{}}
		ms.load()
		return
	}
	path := filepath.Join(t.TempDir(), "memory.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":\"ok\",\"content\":\"valid\"}\n{\"id\":\"torn"), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	exe, _ := os.Executable()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestAuditCorruptMemoryLoad$")
	cmd.Env = append(os.Environ(), "AUDIT_CORRUPT_MEMORY_FILE="+path)
	out, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatal("memory.load hung on a torn final JSONL record")
	}
	if err != nil {
		t.Fatalf("child: %v %s", err, out)
	}
}

func TestAuditGeminiCallIDs(t *testing.T) {
	stream := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{}}}]}}]}\n\ndata: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\n"
	a, err := parseGeminiStream(strings.NewReader(stream), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := parseGeminiStream(strings.NewReader(stream), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.ToolCalls[0].ID == b.ToolCalls[0].ID {
		t.Fatalf("two separate turns both use call ID %q", a.ToolCalls[0].ID)
	}
}

func TestAuditTelemetryCursorRollover(t *testing.T) {
	tel := &Telemetry{notify: make(chan struct{}, 1)}
	for i := 0; i < 1500; i++ {
		tel.Emit("test", "main", i)
	}
	_, cursor := tel.StoredEvents(0)
	for i := 0; i < 501; i++ {
		tel.Emit("test", "main", i)
	}
	for i := 0; i < 500; i++ {
		tel.Emit("test", "main", i)
	}
	events, next := tel.StoredEvents(cursor)
	if len(events) != 1001 {
		t.Fatalf("cursor=%d next=%d retained=%d returned=%d; retained unseen events skipped", cursor, next, len(tel.log), len(events))
	}
}

func TestAuditLateToolImage(t *testing.T) {
	th := retryTestThinker(nil)
	th.registry = NewToolRegistry("")
	th.registry.Register(&ToolDef{Name: "image_test", Handler: func(map[string]string) ToolResponse { return ToolResponse{Text: "screenshot", Image: []byte{1, 2, 3}} }})
	th.placeholdersSent.Store("call1", placeholderInfo{iteration: 1, dispatchedAt: time.Now()})
	executeTool(th, toolCall{Name: "image_test", NativeID: "call1", Args: map[string]string{}})
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		for _, e := range th.sub.DrainTargeted() {
			if strings.Contains(e.Text, "late-result") {
				if len(e.Parts) == 0 && (e.ToolResult == nil || len(e.ToolResult.Image) == 0) {
					t.Fatal("late result contains text only; image was discarded")
				}
				return
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("result not received")
}

func TestAuditStalePlaceholderReinjection(t *testing.T) {
	th := retryTestThinker(nil)
	th.iteration = 30
	th.pendingTools.Store("call1", "slow")
	th.placeholdersSent.Store("call1", placeholderInfo{iteration: 1, toolName: "slow", dispatchedAt: time.Now().Add(-6 * time.Minute)})
	th.sweepStalePlaceholders()
	var results []ToolResult
	th.injectPlaceholdersForPending(&results)
	if len(results) > 0 {
		t.Fatal("already paired timed-out call received a second placeholder")
	}
}

func TestAuditTruncatedStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call1\",\"function\":{\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n")
	}))
	defer server.Close()
	p := &OpenAICompatProvider{name: "test", url: server.URL, models: map[ModelTier]string{ModelLarge: "test"}}
	resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "test"}}, "test", nil, nil, nil, nil)
	if err == nil {
		t.Fatalf("truncated stream accepted: tool calls=%+v", resp.ToolCalls)
	}
}

func TestAuditNestedThreadDelete(t *testing.T) {
	api, th := newTestAPI()
	defer th.telemetry.Stop()
	leader := retryTestThinker(nil)
	leader.threadID = "leader"
	leader.config = th.config
	children := NewThreadManager(leader)
	leader.threads = children
	worker := retryTestThinker(nil)
	worker.threadID = "worker"
	children.threads["worker"] = &Thread{ID: "worker", Thinker: worker}
	th.threads.threads["leader"] = &Thread{ID: "leader", Thinker: leader, Children: children}
	w := httptest.NewRecorder()
	api.threadAction(w, httptest.NewRequest("DELETE", "/threads/worker", nil))
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if findThinkerByID(th, "worker") != nil {
		t.Fatal("DELETE returned 200/killed but nested worker remains alive")
	}
}

func TestAuditResetConcurrentReaders(t *testing.T) {
	th := retryTestThinker(nil)
	th.messages = append(th.messages, Message{Role: "assistant", Content: "answer"})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			th.publishContextStatus()
		}
	}()
	for i := 0; i < 1000; i++ {
		_, _ = resetThinkerContext(th)
	}
	<-done
}

func TestAuditRealtimeOverflow(t *testing.T) {
	th := newTestThinkerFull()
	th.sub = th.bus.Subscribe("main", 1)
	session := newFakeRealtimeSession()
	rt := newRealtimeThinker(context.Background(), th, &fakeRealtimeProvider{}, "", nil, nil, nil)
	rt.replaceSession(session)
	for i := 0; i < 3; i++ {
		th.bus.Publish(Event{Type: EventInbox, To: "main", Text: fmt.Sprintf("message %d", i)})
	}
	done := make(chan struct{})
	go func() { defer close(done); rt.Run() }()
	defer func() { th.Stop(); <-done }()
	time.Sleep(100 * time.Millisecond)
	session.mu.Lock()
	got := len(session.texts)
	session.mu.Unlock()
	if got != 3 {
		t.Fatalf("voice runtime consumed %d/3 queued events; overflow is never drained", got)
	}
}

func TestAuditModeUpdate(t *testing.T) {
	api, th := newTestAPI()
	defer th.telemetry.Stop()
	th.config.Mode = ModeAutonomous
	th.ReloadDirectiveQuiet()
	before := th.messages[0].Content
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(`{"mode":"cautious"}`)))
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if th.config.GetMode() != ModeCautious {
		t.Fatal("mode not persisted")
	}
	if th.messages[0].Content == before {
		t.Fatal("config says cautious but model still sees autonomous prompt")
	}
}

func TestAuditRejectedConfigPartialWrite(t *testing.T) {
	api, th := newTestAPI()
	defer th.telemetry.Stop()
	before := th.config.GetDirective()
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest("PUT", "/config", strings.NewReader(`{"directive":"changed despite rejection","mcp_servers":[{"name":"bad","tool_loading":{"default":"invalid"}}]}`)))
	if w.Code != 400 {
		t.Fatalf("status=%d %s", w.Code, w.Body.String())
	}
	if th.config.GetDirective() != before {
		t.Fatalf("400 response left directive changed to %q", th.config.GetDirective())
	}
}

func TestAuditMCPNativeImage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"image","data":"AQID","mimeType":"image/png"}]}}`)
	}))
	defer server.Close()
	srv := &MCPHTTPServer{Name: "test", url: server.URL, client: server.Client()}
	resp, err := srv.CallTool("screenshot", map[string]string{})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Image) == 0 && resp.Text == "" {
		t.Fatal("standard MCP image content silently became an empty result")
	}
}

type auditCompactionProvider struct{ scriptedRetryProvider }

func (p *auditCompactionProvider) Chat(ctx context.Context, m []Message, model string, tools []NativeTool, a, b func(string), c func(string, string, string)) (ChatResponse, error) {
	if len(m) > 0 && strings.Contains(m[0].Content, "Compact autonomous agent history") {
		return ChatResponse{}, fmt.Errorf("simulated compaction outage")
	}
	return ChatResponse{Text: "finished"}, nil
}
func TestAuditCompactionFailure(t *testing.T) {
	th := newTestThinkerFull()
	th.provider = &auditCompactionProvider{}
	th.session = NewSession(t.TempDir(), "main")
	for i := 0; i < 501; i++ {
		if err := th.session.AppendMessage(Message{Role: "user", Content: fmt.Sprintf("irreplaceable instruction %d", i)}, i, TokenUsage{}); err != nil {
			t.Fatal(err)
		}
	}
	done := make(chan struct{})
	go func() { defer close(done); th.Run() }()
	defer func() { th.Stop(); <-done }()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if th.session.Count() < 500 {
			data, _ := os.ReadFile(th.session.path)
			if strings.Contains(string(data), "Compacted ") {
				t.Fatal("failed semantic compaction fell through to count-only summary; older history deleted")
			}
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestAuditMCPStreamingResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"completed\"}]}}\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	srv := &MCPHTTPServer{Name: "test", url: server.URL, client: &http.Client{Timeout: 100 * time.Millisecond}}
	resp, err := srv.CallTool("test", map[string]string{})
	if err != nil || resp.Text != "completed" {
		t.Fatalf("complete SSE result was ignored until stream timeout: %v", err)
	}
}

func BenchmarkAuditMemoryRecall(b *testing.B) {
	for _, n := range []int{1000, 10000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			ms := &MemoryStore{byID: map[string]int{}}
			for i := 0; i < n; i++ {
				ms.records = append(ms.records, MemoryRecord{ID: fmt.Sprint(i), Content: fmt.Sprintf("Project %d production deployment uses Go workers and daily customer reports in Europe", i), Weight: .8, TS: time.Now()})
			}
			ms.rebuildActiveLocked()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ms.RecallMatchesForContexts([]string{"production deployment customer reports", "Go workers in Europe"}, 8)
			}
		})
	}
}
func BenchmarkAuditEventPublish(b *testing.B) {
	for _, n := range []int{1, 100, 1000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			bus := NewEventBus()
			for i := 0; i < n; i++ {
				bus.Subscribe(fmt.Sprint(i), 1)
			}
			ev := Event{Type: EventChunk, Text: "token"}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bus.Publish(ev)
			}
		})
	}
}

type auditRoundTripper func(*http.Request) (*http.Response, error)

func (f auditRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestAuditGeminiResponseName(t *testing.T) {
	old := llmHTTPClient
	defer func() { llmHTTPClient = old }()
	var body geminiRequest
	llmHTTPClient = &http.Client{Transport: auditRoundTripper(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		return &http.Response{StatusCode: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`data: {"candidates":[{"finishReason":"STOP"}]}

`))}, nil
	})}
	p := NewGoogleProvider("dummy")
	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "look up"},
		{Role: "assistant", ToolCalls: []NativeToolCall{{ID: "gemini_1", Name: "lookup", Args: map[string]string{}}}},
		{Role: "user", ToolResults: []ToolResult{{CallID: "gemini_1", ToolName: "lookup", Content: "done"}}},
	}, "test", nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range body.Contents {
		for _, part := range c.Parts {
			if part.FunctionResponse != nil && part.FunctionResponse.Name != "lookup" {
				t.Fatalf("response name=%q, want actual function name lookup", part.FunctionResponse.Name)
			}
		}
	}
}
