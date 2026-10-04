package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This test is launched by Server's TestNativeImageGenerationLive, which owns
// a real blob gateway and isolated storage. All model calls use real upstreams;
// neither generation nor the model's file-reference tool call is fabricated.
func TestNativeImageGenerationLive(t *testing.T) {
	if testing.Short() || os.Getenv("RUN_NATIVE_IMAGE_GENERATION_LIVE") != "1" {
		t.Skip("opt in through Server's TestNativeImageGenerationLive")
	}
	if os.Getenv("APTEVA_IMAGE_GENERATION_GATEWAY_URL") == "" || os.Getenv("APTEVA_IMAGE_GENERATION_LIVE_CONSUMER_URL") == "" {
		t.Fatal("run the Server live test to provide isolated gateway/storage")
	}
	loadIntegrationEnv()
	providerName := os.Getenv("APTEVA_IMAGE_GENERATION_LIVE_PROVIDER")
	if providerName == "" {
		providerName = "openai-codex"
	}
	var p *OpenAINativeProvider
	model := os.Getenv("APTEVA_IMAGE_GENERATION_LIVE_MODEL")
	switch providerName {
	case "openai-codex":
		token := codexAccessTokenForMemorySmoke(t)
		if token == "" {
			t.Fatal("live test requested but no valid Codex credential is available")
		}
		p = NewOpenAICodexProvider(token).(*OpenAINativeProvider)
		p.runtimeTokenURL = ""
		p.serverAPIKey = ""
		if p.accountID == "" {
			home, _ := os.UserHomeDir()
			data, _ := os.ReadFile(filepath.Join(home, ".codex", "auth.json"))
			var auth struct {
				Tokens struct {
					AccountID string `json:"account_id"`
				} `json:"tokens"`
			}
			if json.Unmarshal(data, &auth) == nil {
				p.accountID = auth.Tokens.AccountID
			}
		}
		if model == "" {
			model = "gpt-6.1-sol"
		}
	case "openai":
		key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
		if key == "" {
			t.Fatal("live test requested but OPENAI_API_KEY is unavailable")
		}
		p = NewOpenAINativeProvider(key).(*OpenAINativeProvider)
		if model == "" {
			model = "gpt-5.4-mini"
		}
	default:
		t.Fatalf("unsupported live provider %q", providerName)
	}
	imageModel := os.Getenv("APTEVA_IMAGE_GENERATION_LIVE_IMAGE_MODEL")
	if imageModel == "" {
		imageModel = "gpt-image-2.5-flare"
	}
	if err := configureProviderBuiltins(p, ProviderConfig{Builtins: map[string]BuiltinToolConfig{
		"image_generation": {Enabled: true, Options: map[string]any{"model": imageModel, "quality": "low", "size": "1024x1024", "output_format": "png"}},
	}}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(withBlobCallerThread(context.Background(), "image-live"), 4*time.Minute)
	defer cancel()
	messages := []Message{{Role: "system", Content: "You are an image-generation smoke-test assistant. Generate exactly one requested image using the hosted tool. Do not describe how to generate it or use external tools."}, {Role: "user", Content: "Generate a simple blue circle on a plain white background. No text."}}
	t.Logf("real provider=%s model=%s image_model=%s; isolated gateway and blob storage", providerName, model, imageModel)
	resp, err := p.Chat(ctx, messages, model, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("live image generation: %v", err)
	}
	if len(resp.GeneratedFiles) != 1 {
		t.Fatalf("expected one generated file, got %d", len(resp.GeneratedFiles))
	}
	file := resp.GeneratedFiles[0]
	if err := validateFileRef(&file); err != nil || file.Size <= 0 || file.MimeType != "image/png" {
		t.Fatalf("invalid generated file: metadata=%+v err=%v", file, err)
	}
	assistant := Message{Role: "assistant", Content: resp.Text, Parts: generatedFileParts(resp.GeneratedFiles), Reasoning: resp.Reasoning, ProviderState: resp.ProviderState}
	session := NewSession(t.TempDir(), "image-live")
	if err := session.AppendMessage(assistant, 1, resp.Usage); err != nil {
		t.Fatal(err)
	}
	persisted, err := os.ReadFile(session.path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(persisted, []byte(`"result"`)) || bytes.Contains(persisted, []byte("base64")) || !bytes.Contains(persisted, []byte(file.Ref)) {
		t.Fatal("history retained image bytes or lost reference")
	}
	// Reconstruct from serialized state, proving the opaque handle survives.
	serialized, _ := json.Marshal(assistant)
	var restored Message
	if err := json.Unmarshal(serialized, &restored); err != nil {
		t.Fatal(err)
	}
	messages = append(messages, restored, Message{Role: "user", Content: "Call use_generated_file exactly once with the generated image's unchanged blobref reference. Do not generate another image. After the tool result, answer FILE_ACCEPTED."})
	continuation := p.clone()
	if err := continuation.ConfigureBuiltins(map[string]BuiltinToolConfig{"image_generation": {Enabled: false}}); err != nil {
		t.Fatal(err)
	}
	tool := NativeTool{Name: "use_generated_file", Description: "Consume the generated image through the file gateway.", Parameters: map[string]any{"type": "object", "properties": map[string]any{"ref": map[string]any{"type": "string", "description": "The unchanged blobref:// reference"}}, "required": []string{"ref"}, "additionalProperties": false}}
	resp, err = continuation.Chat(ctx, messages, model, []NativeTool{tool}, nil, nil, nil)
	if err != nil {
		t.Fatalf("live handle continuation: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != tool.Name || resp.ToolCalls[0].Args["ref"] != file.Ref {
		t.Fatal("real model did not pass the generated handle unchanged to the consumer")
	}
	call := resp.ToolCalls[0]
	consumerBody, _ := json.Marshal(map[string]string{"ref": call.Args["ref"]})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, os.Getenv("APTEVA_IMAGE_GENERATION_LIVE_CONSUMER_URL"), bytes.NewReader(consumerBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Agent-Secret", imageGatewayAgentSecret())
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resultBytes, readErr := io.ReadAll(io.LimitReader(httpResp.Body, 4096))
	httpResp.Body.Close()
	if readErr != nil || httpResp.StatusCode != 200 || string(resultBytes) != "FILE_ACCEPTED" {
		t.Fatalf("gateway consumer failed: status=%d err=%v", httpResp.StatusCode, readErr)
	}
	messages = append(messages, Message{Role: "assistant", Content: resp.Text, ToolCalls: resp.ToolCalls, ProviderState: resp.ProviderState}, Message{Role: "tool", ToolResults: []ToolResult{{CallID: call.ID, ToolName: call.Name, Content: "FILE_ACCEPTED"}}})
	resp, err = continuation.Chat(ctx, messages, model, []NativeTool{tool}, nil, nil, nil)
	if err != nil || !strings.Contains(resp.Text, "FILE_ACCEPTED") {
		t.Fatalf("live tool-result continuation failed: %v", err)
	}
	t.Log(fmt.Sprintf("verified generation, byte-free persisted handle, real LLM tool handoff, gateway consumption and continuation; image_bytes=%d", file.Size))
}
