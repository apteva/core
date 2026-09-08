package core

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

type rejectContextOnceProvider struct {
	LLMProvider
	calls    int
	requests [][]Message
}

func (p *rejectContextOnceProvider) Chat(ctx context.Context, m []Message, model string, tools []NativeTool, onChunk, onThinking func(string), onToolChunk func(string, string, string)) (ChatResponse, error) {
	p.calls++
	p.requests = append(p.requests, cloneMessages(m))
	if p.calls == 1 {
		return ChatResponse{}, errors.New("OpenAI stream error (context_length_exceeded): injected rejection for recovery regression")
	}
	return p.LLMProvider.Chat(ctx, m, model, tools, onChunk, onThinking, onToolChunk)
}

// The initial rejection is injected; summaries and continuations use real
// Codex. All state and payloads are synthetic, with no external tool effects.
func TestCodexContextRecoveryPreservesReleaseState(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_CODEX_CONTEXT_RECOVERY") != "1" {
		t.Skip("set RUN_CODEX_CONTEXT_RECOVERY=1 for live Codex recovery checks")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Fatal("live Codex requested but saved authentication unavailable")
	}
	for _, kind := range []string{"bulky_result", "semantic_history", "structured_result"} {
		t.Run(kind, func(t *testing.T) {
			p := &rejectContextOnceProvider{LLMProvider: &fixedModelProvider{LLMProvider: NewOpenAICodexProvider(token), model: "gpt-5.6-terra"}}
			m := oversizedFixture(160000)
			if kind == "structured_result" {
				m[1].Content = "The tool output contains the authoritative release state."
				m[3].ToolResults[0].Content = string(mustJSON(t, map[string]any{"a_image": strings.Repeat("pixels", 15000), "pending_release": "PENDING-RELEASE-927", "published_post": "POST-881", "artifact_path": "/releases/final.png", "z_document": strings.Repeat("document", 15000)}))
			}
			if kind == "semantic_history" {
				m = []Message{{Role: "system", Content: "Inspect prior release state and answer the user's exact request. Never invent completion or republish."}}
				for i := 0; i < 40; i++ {
					m = append(m, Message{Role: "assistant", Content: strings.Repeat("Earlier checks found no change to the draft; continue preserving the established release state. ", 30)})
				}
				m[6].Content = "Release remains PENDING-RELEASE-927. Already published post POST-881 must not be duplicated. Artifact path /releases/final.png."
				m = append(m, Message{Role: "user"})
			}
			m[len(m)-1].Content = "From the existing history, return only the pending release identifier, already-published post identifier, and artifact path, separated by | with no spaces. Do not take any external action."
			th := recoveryTestThinker(t, p, m)
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			resp, err := th.callLLMWithRetry(ctx)
			if err != nil {
				t.Fatal(err)
			}
			want := "PENDING-RELEASE-927|POST-881|/releases/final.png"
			if strings.TrimSpace(resp.Text) != want {
				t.Fatalf("release state lost: %q", resp.Text)
			}
			if len(resp.ToolCalls) != 0 {
				t.Fatal("unexpected tool action")
			}
			if p.calls < 2 {
				t.Fatal("recovery did not retry")
			}
			before := estimatedContextTokens(p.requests[0])
			after := estimatedContextTokens(p.requests[len(p.requests)-1])
			if after >= before*95/100 {
				t.Fatalf("request did not materially shrink: %d -> %d", before, after)
			}
			t.Logf("LIVE_CONTEXT_RECOVERY model=gpt-5.6-terra case=%s provider_calls=%d input_tokens_est=%d->%d state_verified=true", kind, p.calls, before, after)
		})
	}
}
