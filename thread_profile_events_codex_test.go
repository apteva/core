package core

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The first REAL inference completes at the provider boundary and is held
// before returning to core. This makes reconciliation during an active request
// deterministic without replacing any model response with a fake one.
type profileLiveProvider struct {
	LLMProvider
	mu           sync.Mutex
	requests     [][]Message
	firstContext context.Context
	firstReady   chan error
	releaseFirst chan struct{}
}

func (p *profileLiveProvider) Chat(ctx context.Context, messages []Message, model string, tools []NativeTool, onChunk func(string), onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	p.mu.Lock()
	first := len(p.requests) == 0
	p.requests = append(p.requests, cloneMessages(messages))
	if first {
		p.firstContext = ctx
	}
	p.mu.Unlock()
	response, err := p.LLMProvider.Chat(ctx, messages, model, tools, onChunk, onThinking, onToolChunk)
	if first {
		p.firstReady <- err
		if err != nil {
			return response, err
		}
		select {
		case <-ctx.Done():
			return ChatResponse{}, ctx.Err()
		case <-p.releaseFirst:
		}
	}
	return response, err
}

// RUN_CODEX_PROFILE_EVENTS=1 go test -run TestCodexThreadProfileEvents -v -timeout 6m .
func TestCodexThreadProfileEvents(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_PROFILE_EVENTS") != "1" {
		t.Skip("set RUN_CODEX_PROFILE_EVENTS=1 for live Codex profile/event tests")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live Codex test requires a valid local Codex token or OPENAI_CODEX_ACCESS_TOKEN")
	}
	for _, changed := range []bool{false, true} {
		t.Run(fmt.Sprintf("profile_changed_%v", changed), func(t *testing.T) {
			t.Chdir(t.TempDir())
			provider := &profileLiveProvider{LLMProvider: NewOpenAICodexProvider(token), firstReady: make(chan error, 1), releaseFirst: make(chan struct{})}
			cfg := &Config{path: filepath.Join(t.TempDir(), "config.json"), Directive: "Coordinate test workers."}
			parent := NewThinker("", provider, cfg)
			// Keep all test inference on the explicitly requested Codex provider.
			useThreadEventProvider(parent, provider)
			defer func() { parent.threads.KillAll(); parent.Stop(); parent.blobs.Close() }()
			api := &APIServer{thinker: parent}
			var probes atomic.Int32
			observed := make(chan map[string]string, 4)
			parent.registry.Register(&ToolDef{Name: "profile_delivery_probe", Description: "Record the event token and the active profile version exactly once.", InputSchema: map[string]any{"type": "object", "properties": map[string]any{"token": map[string]any{"type": "string"}, "profile": map[string]any{"type": "string"}}, "required": []string{"token", "profile"}}, Handler: func(args map[string]string) ToolResponse {
				probes.Add(1)
				copy := map[string]string{}
				for k, v := range args {
					copy[k] = v
				}
				observed <- copy
				return ToolResponse{Text: `{"recorded":true,"instruction":"Do not record again. Call pace with clear_wake=true and wait for new events."}`}
			}})
			const desired = "PROFILE_VERSION_V2. This is an event-driven test. Do not spawn, send, delegate, or evolve. If no PROFILE_EVENT_TOKEN has arrived, reply READY only, without tools. When an inbox event contains PROFILE_EVENT_TOKEN, call profile_delivery_probe exactly once using its exact token and profile=PROFILE_VERSION_V2. After its success call pace(clear_wake=true). Do not repeat completed events."
			old := desired
			tools := []string{"profile_delivery_probe"}
			if changed {
				old = "PROFILE_VERSION_V1. Reply READY only, without tools, and wait for instructions."
				tools = []string{}
			}
			rec := postThreadForTest(t, api, "codex-profile-events", map[string]any{"directive": old, "tools": tools})
			requireProfileHTTP(t, rec, 200)
			select {
			case err := <-provider.firstReady:
				if err != nil {
					t.Fatalf("first real Codex inference: %v", err)
				}
			case <-time.After(90 * time.Second):
				t.Fatal("first Codex inference timed out")
			}
			const eventToken = "PROFILE_EVENT_TOKEN_4F91"
			events := profileEventPayload("live-profile-event", eventToken)
			rec = putProfileEvents(api, "codex-profile-events", desired, []string{"profile_delivery_probe"}, events)
			requireProfileHTTP(t, rec, 200)
			provider.mu.Lock()
			firstCtx := provider.firstContext
			provider.mu.Unlock()
			if changed && firstCtx.Err() == nil {
				t.Fatal("changed profile did not cancel obsolete inference")
			}
			if !changed && firstCtx.Err() != nil {
				t.Fatal("unchanged profile canceled active inference")
			}
			close(provider.releaseFirst)
			select {
			case args := <-observed:
				if args["token"] != eventToken || args["profile"] != "PROFILE_VERSION_V2" {
					t.Fatalf("real model saw wrong event/profile: %+v", args)
				}
			case <-time.After(120 * time.Second):
				t.Fatal("Codex did not execute the event/profile probe")
			}
			// The duplicate update must not interrupt the result continuation or repeat
			// the local probe. Wait for explicit pace so periodic work cannot mask this.
			retry := putProfileEvents(api, "codex-profile-events", desired, []string{"profile_delivery_probe"}, events)
			requireProfileHTTP(t, retry, 200)
			response := decodeThreadEventResponse(t, retry.Body.Bytes())
			if len(responseEventIDs(t, response, "duplicates")) != 1 {
				t.Fatal("retry not deduplicated")
			}
			deadline := time.Now().Add(90 * time.Second)
			for time.Now().Before(deadline) {
				_, thread := parent.threads.findManagedThread("codex-profile-events")
				if thread.Thinker.status().WaitForEvents && !thread.Thinker.status().LLMActive {
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			_, thread := parent.threads.findManagedThread("codex-profile-events")
			if !thread.Thinker.status().WaitForEvents {
				t.Fatal("Codex did not settle after processing event")
			}
			if probes.Load() != 1 {
				t.Fatalf("probe count=%d", probes.Load())
			}
			provider.mu.Lock()
			requests := append([][]Message(nil), provider.requests...)
			provider.mu.Unlock()
			if len(requests) < 2 {
				t.Fatal("no replacement/following model request")
			}
			if text := messagesText(requests[1]); !strings.Contains(text, eventToken) || !strings.Contains(text, "PROFILE_VERSION_V2") {
				t.Fatal("next real Codex inference did not see event and profile together")
			}
			t.Logf("Real Codex: changed=%v requests=%d probe_calls=%d; matching profile/event and duplicate suppression verified", changed, len(requests), probes.Load())
		})
	}
}
