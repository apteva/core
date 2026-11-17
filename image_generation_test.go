package core

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestNativeImageGenerationFlagAndGateway(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			var request map[string]any
			var headers http.Header
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				headers = r.Header.Clone()
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					return
				}
				writeServiceTierTestStream(w)
			}))
			defer srv.Close()
			t.Setenv("APTEVA_IMAGE_GENERATION_GATEWAY_URL", srv.URL)
			t.Setenv("AGENT_ID", "42")
			t.Setenv("AGENT_SECRET", "agent-secret")
			p := NewOpenAICodexProvider("token").(*OpenAINativeProvider)
			p.runtimeTokenURL = ""
			p.responsesURL = srv.URL
			if err := p.configureImageGeneration(&ImageGenerationConfig{Enabled: enabled, Model: "gpt-image-2.5-flare", Quality: "low"}); err != nil {
				t.Fatal(err)
			}
			p.SetBuiltinTools([]string{"image_generation"})
			p = p.WithReasoning(ReasoningSettings{Level: ReasoningLow}).WithBuiltins([]string{"image_generation"}).(*OpenAINativeProvider)
			ctx := withBlobCallerThread(context.Background(), "real-thread")
			if _, err := p.Chat(ctx, []Message{{Role: "user", Content: "Draw a circle"}}, "gpt-6.1-sol", nil, nil, nil, nil); err != nil {
				t.Fatal(err)
			}
			tools, _ := request["tools"].([]any)
			if !enabled {
				if len(tools) != 0 || len(p.AvailableBuiltinTools()) != 0 || headers.Get("X-Agent-Secret") != "" {
					t.Fatal("disabled flag leaked image capability or gateway identity")
				}
				return
			}
			if len(tools) != 1 {
				t.Fatalf("tools=%v", tools)
			}
			tool := tools[0].(map[string]any)
			if tool["model"] != "gpt-image-2.5-flare" || tool["type"] != "image_generation" {
				t.Fatalf("tool=%v", tool)
			}
			if headers.Get("X-Apteva-File-Thread") != "real-thread" || headers.Get("X-Apteva-Caller-Agent") != "42" || headers.Get("X-Agent-Secret") != "agent-secret" {
				t.Fatal("trusted gateway context lost")
			}
		})
	}
	// Enabling the feature must never fall back to an upstream that returns bytes.
	p := NewOpenAINativeProvider("token").(*OpenAINativeProvider)
	if err := p.configureImageGeneration(&ImageGenerationConfig{Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.imageResponsesEndpoint(context.Background()); err == nil {
		t.Fatal("missing gateway context accepted")
	}
}

func TestNativeImageGenerationHistoryAndCompletion(t *testing.T) {
	p := NewOpenAINativeProvider("token").(*OpenAINativeProvider)
	if err := p.configureImageGeneration(&ImageGenerationConfig{Enabled: true}); err != nil {
		t.Fatal(err)
	}
	handle := FileRef{File: true, Ref: "blobref://test-image", Filename: "image.png", MimeType: "image/png", Size: 100}
	item := map[string]any{"type": "image_generation_call", "id": "ig_1", "status": "completed", "file_ref": handle}
	for _, withDone := range []bool{true, false} {
		var stream strings.Builder
		writeEvent := func(event any) { b, _ := json.Marshal(event); stream.WriteString("data: " + string(b) + "\n\n") }
		writeEvent(map[string]any{"type": "response.output_item.done", "item": map[string]any{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"}})
		if withDone {
			writeEvent(map[string]any{"type": "response.output_item.done", "item": item})
		}
		writeEvent(map[string]any{"type": "response.completed", "response": map[string]any{"output": []any{item}}})
		resp, err := p.streamResponse(strings.NewReader(stream.String()), nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.GeneratedFiles) != 1 || len(resp.ToolCalls) != 0 {
			t.Fatalf("images=%v tool_calls=%v", resp.GeneratedFiles, resp.ToolCalls)
		}
		msg := Message{Role: "assistant", Content: "Your image", Parts: generatedFileParts(resp.GeneratedFiles), ProviderState: resp.ProviderState}
		session := NewSession(t.TempDir(), "image-thread")
		if err := session.AppendMessage(msg, 1, resp.Usage); err != nil {
			t.Fatal(err)
		}
		persisted, err := os.ReadFile(session.path)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(persisted), "image_generation_call") || strings.Contains(string(persisted), "base64") || !strings.Contains(string(persisted), handle.Ref) {
			t.Fatal("history lost handle or retained hosted output")
		}
		input, _ := json.Marshal(p.buildInput([]Message{msg}))
		if !strings.Contains(string(input), handle.Ref) {
			t.Fatal("provider replay hid generated handle behind reasoning state")
		}
	}
	_, err := p.streamResponse(strings.NewReader("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"image_generation_call\",\"result\":\"bytes\"}}\n\n"), nil, nil, nil)
	if err == nil {
		t.Fatal("raw output without gateway handle was accepted")
	}
}

func TestNativeImageGenerationConfigurationCopies(t *testing.T) {
	cfg := NewConfig()
	cfg.Providers = []ProviderConfig{{Name: "openai-codex", ImageGeneration: &ImageGenerationConfig{Enabled: true, Model: "gpt-image-2.5-flare"}}}
	copy := cfg.GetProviders()
	copy[0].ImageGeneration.Enabled = false
	if !cfg.Providers[0].ImageGeneration.Enabled {
		t.Fatal("GetProviders aliases flag")
	}
	copy = cloneProviderConfigs(cfg.Providers)
	copy[0].ImageGeneration.Model = "changed"
	if cfg.Providers[0].ImageGeneration.Model != "gpt-image-2.5-flare" {
		t.Fatal("clone aliases model")
	}
	updated := mergeProviderConfig(cfg.Providers, ProviderConfig{Name: "openai-codex", ImageGeneration: &ImageGenerationConfig{Enabled: false}})
	if updated[0].ImageGeneration.Enabled {
		t.Fatal("feature could not be disabled")
	}
}
