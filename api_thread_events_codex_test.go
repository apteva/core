package core

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type recordingLiveThreadEventProvider struct {
	LLMProvider
	mu       sync.Mutex
	requests [][]Message
}

func (p *recordingLiveThreadEventProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	p.requests = append(p.requests, cloneMessages(messages))
	p.mu.Unlock()
	return p.LLMProvider.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
}

func (p *recordingLiveThreadEventProvider) firstRequest() []Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.requests) == 0 {
		return nil
	}
	return cloneMessages(p.requests[0])
}

// TestCodexAPIThreadStartsWithIdempotentEventSmoke proves the API contract
// through the real Codex provider: the event is present on the first turn,
// drives exactly one model-selected tool call, and an identical POST retry
// neither redelivers the event nor wakes another workflow.
//
//	RUN_CODEX_THREAD_EVENTS_SMOKE=1 go test -v -run TestCodexAPIThreadStartsWithIdempotentEventSmoke -timeout 5m .
func TestCodexAPIThreadStartsWithIdempotentEventSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_THREAD_EVENTS_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_THREAD_EVENTS_SMOKE=1 to run the Codex thread-events smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex thread-events smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}

	t.Chdir(t.TempDir())
	provider := &recordingLiveThreadEventProvider{LLMProvider: NewOpenAICodexProvider(token)}
	cfg := &Config{
		path:      filepath.Join(t.TempDir(), "config.json"),
		Directive: "Coordinate API-created threads.",
		Mode:      ModeAutonomous,
	}
	thinker := NewThinker("", provider, cfg)
	defer func() {
		thinker.threads.KillAll()
		thinker.Stop()
	}()
	api := &APIServer{thinker: thinker}

	var probeCalls atomic.Int32
	var probeText atomic.Value
	thinker.registry.Register(&ToolDef{
		Name:        "thread_event_probe",
		Description: "Record the received API event exactly once. Call only when explicitly required by the thread directive.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{"type": "string", "description": "Exact event text."},
			},
			"required": []string{"text"},
		},
		Handler: func(args map[string]string) ToolResponse {
			probeCalls.Add(1)
			probeText.Store(args["text"])
			return ToolResponse{Text: `{"recorded":true,"instruction":"event processed; do not call this tool again"}`}
		},
	})

	const eventText = "Codex event payload 7F31"
	body := map[string]any{
		"directive": strings.Join([]string{
			"# Role",
			"Validate one API-created inbox event directly on this thread. Do not spawn, send, evolve, or delegate.",
			"",
			"# Workflow",
			"- When the event containing 'Codex event payload 7F31' arrives, call thread_event_probe exactly once with that exact text.",
			"- Wait for its real result. After success, reply exactly THREAD_EVENT_CODEX_OK, then wait for future events.",
		}, "\n"),
		"tools": []string{"thread_event_probe"},
		"events": []any{map[string]any{
			"id": "codex-event:7f31", "message": eventText,
		}},
	}
	w := postThreadForTest(t, api, "codex-event-thread", body)
	if w.Code != 200 {
		t.Fatalf("spawn status=%d body=%s", w.Code, w.Body.String())
	}
	response := decodeThreadEventResponse(t, w.Body.Bytes())
	if ids := responseEventIDs(t, response, "accepted"); len(ids) != 1 || ids[0] != "codex-event:7f31" {
		t.Fatalf("accepted=%v", ids)
	}

	deadline := time.Now().Add(150 * time.Second)
	sawFinal := false
	for time.Now().Before(deadline) {
		if probeCalls.Load() > 1 {
			t.Fatalf("Codex repeated event tool call: %d", probeCalls.Load())
		}
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != "codex-event-thread" || event.Type != "llm.done" {
				continue
			}
			var data LLMDoneData
			if json.Unmarshal(event.Data, &data) == nil && strings.Contains(data.Message, "THREAD_EVENT_CODEX_OK") {
				sawFinal = true
				break
			}
		}
		if probeCalls.Load() == 1 && sawFinal {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if probeCalls.Load() != 1 || !sawFinal {
		t.Fatalf("Codex workflow incomplete: probe_calls=%d final=%v", probeCalls.Load(), sawFinal)
	}
	if text, _ := probeText.Load().(string); text != eventText {
		t.Fatalf("probe text=%q want %q", text, eventText)
	}
	if request := provider.firstRequest(); len(request) == 0 {
		t.Fatal("real Codex received no first request")
	} else if _, found := messageContaining(request, eventText); !found {
		t.Fatalf("real Codex first request omitted event: %#v", request)
	}

	retry := postThreadForTest(t, api, "codex-event-thread", body)
	if retry.Code != 200 {
		t.Fatalf("retry status=%d body=%s", retry.Code, retry.Body.String())
	}
	retryResponse := decodeThreadEventResponse(t, retry.Body.Bytes())
	if ids := responseEventIDs(t, retryResponse, "duplicates"); len(ids) != 1 || ids[0] != "codex-event:7f31" {
		t.Fatalf("retry duplicates=%v", ids)
	}
	time.Sleep(2 * time.Second)
	if probeCalls.Load() != 1 {
		t.Fatalf("retry redelivered event: probe_calls=%d", probeCalls.Load())
	}
	t.Logf("Codex processed the creation event once and ignored the retry: probe_calls=%d", probeCalls.Load())
}
