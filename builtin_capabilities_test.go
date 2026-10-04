package core

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestBuiltinCapabilitiesOpenAIWireAndPrecedence(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		writeServiceTierTestStream(w)
	}))
	defer srv.Close()
	p := NewOpenAINativeProvider("test").(*OpenAINativeProvider)
	p.responsesURL = srv.URL
	if err := configureProviderBuiltins(p, ProviderConfig{
		BuiltinTools: []string{"web_search_preview", "code_interpreter"},
		Builtins: map[string]BuiltinToolConfig{
			"web_search":  {Enabled: false},
			"file_search": {Enabled: true, Options: map[string]any{"vector_store_ids": []string{"vs_test"}, "max_num_results": 3}},
		},
	}); err != nil {
		t.Fatal(err)
	}
	function := NativeTool{Name: "local", Parameters: map[string]any{"type": "object"}}
	if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "gpt-5.4-mini", []NativeTool{function}, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	byType := map[string]map[string]any{}
	for _, item := range tools {
		tool := item.(map[string]any)
		byType[tool["type"].(string)] = tool
	}
	if len(tools) != 3 || byType["function"]["name"] != "local" || byType["web_search"] != nil || byType["web_search_preview"] != nil {
		t.Fatalf("incorrect enabled tools: %#v", tools)
	}
	if byType["code_interpreter"]["container"].(map[string]any)["type"] != "auto" || byType["file_search"]["vector_store_ids"].([]any)[0] != "vs_test" {
		t.Fatalf("missing native options: %#v", tools)
	}
	if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"web_search": {Enabled: true, Options: map[string]any{"search_context_size": "low"}}}); err != nil {
		t.Fatal(err)
	}
	tools = p.configuredOpenAIBuiltins()
	if tools[0].(map[string]any)["type"] != "web_search" || tools[0].(map[string]any)["search_context_size"] != "low" {
		t.Fatalf("generic search should use current API type/options: %#v", tools)
	}
}

func TestBuiltinCapabilitiesImageGateAndAtomicValidation(t *testing.T) {
	for _, name := range []string{"openai", "openai-codex"} {
		t.Run(name, func(t *testing.T) {
			p := &OpenAINativeProvider{name: name}
			p.SetBuiltinTools([]string{"image_generation"})
			if p.imageGenerationEnabled() || len(p.configuredOpenAIBuiltins()) != 0 {
				t.Fatal("legacy name alone enabled images")
			}
			image := BuiltinToolConfig{Enabled: true, Options: map[string]any{"model": "gpt-image-2.5-flare", "quality": "low", "output_format": "png"}}
			if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"image_generation": image, "file_search": {Enabled: true}}); err == nil {
				t.Fatal("file search accepted without vector stores")
			}
			if p.imageGenerationEnabled() || p.builtinConfigs != nil {
				t.Fatal("failed validation partially enabled images")
			}
			if err := configureProviderBuiltins(p, ProviderConfig{ImageGeneration: &ImageGenerationConfig{Enabled: false}, Builtins: map[string]BuiltinToolConfig{"image_generation": image}}); err != nil {
				t.Fatal(err)
			}
			if !p.imageGenerationEnabled() || p.imageGeneration.Model != "gpt-image-2.5-flare" || p.imageGenerationTool()["quality"] != "low" {
				t.Fatalf("generic image configuration lost: %#v", p.imageGeneration)
			}
			if err := configureProviderBuiltins(p, ProviderConfig{ImageGeneration: &ImageGenerationConfig{Enabled: true}, Builtins: map[string]BuiltinToolConfig{"image_generation": {Enabled: false}}}); err != nil {
				t.Fatal(err)
			}
			if p.imageGenerationEnabled() {
				t.Fatal("generic false did not override legacy true")
			}
			if p.WithBuiltins([]string{"image_generation"}).(*OpenAINativeProvider).imageGenerationEnabled() {
				t.Fatal("child list bypassed the image gate")
			}
			if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"image_generation": image}); err != nil {
				t.Fatal(err)
			}
			if err := p.ConfigureBuiltins(nil); err != nil || p.imageGenerationEnabled() {
				t.Fatal("replacing generic configuration retained its image flag")
			}
		})
	}
}

func TestBuiltinCapabilitiesRejectUnsupportedConfiguration(t *testing.T) {
	for _, test := range []struct {
		name     string
		provider LLMProvider
		configs  map[string]BuiltinToolConfig
	}{
		{"unknown", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"unrecognized": {Enabled: true}}},
		{"option injection", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"web_search": {Enabled: true, Options: map[string]any{"type": "image_generation"}}}},
		{"aliases", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"code_interpreter": {Enabled: true}, "code_execution": {Enabled: true}}},
		{"non JSON", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"web_search": {Options: map[string]any{"filters": make(chan int)}}}},
		{"empty vector id", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"file_search": {Enabled: true, Options: map[string]any{"vector_store_ids": []string{" "}}}}},
		{"image option type", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"image_generation": {Enabled: true, Options: map[string]any{"quality": 1}}}},
		{"image model", NewOpenAINativeProvider("test"), map[string]BuiltinToolConfig{"image_generation": {Enabled: true, Options: map[string]any{"model": "gpt-6.1-sol"}}}},
		{"anthropic image", NewAnthropicProvider("test"), map[string]BuiltinToolConfig{"image_generation": {Enabled: true}}},
		{"google option", NewGoogleProvider("test"), map[string]BuiltinToolConfig{"web_search": {Options: map[string]any{"filters": map[string]any{}}}}},
		{"grok build", NewGrokBuildProvider("test"), map[string]BuiltinToolConfig{"web_search": {Enabled: true}}},
		{"compat", NewOllamaProvider("http://localhost:11434"), map[string]BuiltinToolConfig{"web_search": {Enabled: true}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := configureProviderBuiltins(test.provider, ProviderConfig{Name: test.provider.Name(), Builtins: test.configs}); err == nil {
				t.Fatal("invalid configuration accepted")
			}
		})
	}
	if _, err := buildProviderPool(&Config{Providers: []ProviderConfig{{Name: "openai-realtime", Builtins: map[string]BuiltinToolConfig{"web_search": {Enabled: true}}}}}); err == nil {
		t.Fatal("realtime generic configuration silently ignored")
	}
}

func TestBuiltinCapabilitiesChildSelectionAndCloning(t *testing.T) {
	for _, provider := range []LLMProvider{NewOpenAINativeProvider("test"), NewAnthropicProvider("test"), NewGoogleProvider("test")} {
		t.Run(provider.Name(), func(t *testing.T) {
			configurer := provider.(builtinConfigurer)
			if err := configurer.ConfigureBuiltins(map[string]BuiltinToolConfig{"web_search": {Enabled: true}, "code_execution": {Enabled: false}}); err != nil {
				t.Fatal(err)
			}
			assertEnabled := func(p LLMProvider, search bool) {
				t.Helper()
				for _, capability := range builtinCapabilityInfo(p) {
					if capability.Name == "web_search" && capability.Enabled != search || capability.Name == "code_execution" && capability.Enabled {
						t.Fatalf("incorrect child flags: %#v", builtinCapabilityInfo(p))
					}
				}
			}
			assertEnabled(provider.WithBuiltins(nil), true)
			assertEnabled(provider.WithBuiltins([]string{}), false)
			assertEnabled(provider.WithBuiltins([]string{"google_search", "code_interpreter"}), true)
			assertEnabled(provider, true)
		})
	}
	p := NewOpenAINativeProvider("test").(*OpenAINativeProvider)
	options := map[string]any{"container": map[string]any{"type": "auto", "file_ids": []any{"file-1"}}}
	if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"code_execution": {Enabled: true, Options: options}, "image_generation": {Enabled: true}}); err != nil {
		t.Fatal(err)
	}
	options["container"].(map[string]any)["type"] = "changed"
	clone := p.WithReasoning(ReasoningSettings{Level: ReasoningLow}).(*OpenAINativeProvider)
	clone.builtinConfigs["code_execution"].Options["container"].(map[string]any)["file_ids"].([]any)[0] = "changed"
	original := p.builtinConfigs["code_execution"].Options["container"].(map[string]any)
	if original["type"] != "auto" || original["file_ids"].([]any)[0] != "file-1" || !clone.imageGenerationEnabled() {
		t.Fatal("options alias input/reasoning clone or image flag was lost")
	}
	if p.WithBuiltins([]string{}).(*OpenAINativeProvider).imageGenerationEnabled() || !p.WithBuiltins([]string{"image_generation"}).(*OpenAINativeProvider).imageGenerationEnabled() || !p.imageGenerationEnabled() {
		t.Fatal("child image selection changed parent or lost gate")
	}
}

func TestBuiltinCapabilitiesAnthropicWire(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer srv.Close()
	p := NewAnthropicProvider("test").(*AnthropicProvider)
	p.url = srv.URL
	if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"web_search": {Enabled: true, Options: map[string]any{"max_uses": 2, "allowed_domains": []string{"example.com"}}}, "code_execution": {Enabled: true}}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "claude-sonnet-4-6", nil, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	tools := body["tools"].([]any)
	search := tools[1].(map[string]any)
	if len(tools) != 2 || search["type"] != "web_search_20250305" || search["name"] != "web_search" || search["max_uses"] != float64(2) || search["allowed_domains"].([]any)[0] != "example.com" {
		t.Fatalf("wrong hosted Anthropic tool shape: %#v", tools)
	}
}

func TestBuiltinCapabilitiesGoogleWireAndResults(t *testing.T) {
	p := NewGoogleProvider("test").(*GoogleProvider)
	if err := p.ConfigureBuiltins(map[string]BuiltinToolConfig{"web_search": {Enabled: true}, "code_execution": {Enabled: true}}); err != nil {
		t.Fatal(err)
	}
	originalClient := llmHTTPClient
	t.Cleanup(func() { llmHTTPClient = originalClient })
	stream := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"executableCode\":{\"language\":\"PYTHON\",\"code\":\"print(2)\"}},{\"codeExecutionResult\":{\"outcome\":\"OUTCOME_OK\",\"output\":\"2\\n\"}},{\"text\":\"2\"}]},\"finishReason\":\"STOP\"}]}\n\n"
	called := false
	llmHTTPClient = &http.Client{Transport: auditRoundTripper(func(r *http.Request) (*http.Response, error) {
		called = true
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		tools := request["tools"].([]any)
		if len(tools) != 3 || len(tools[0].(map[string]any)["functionDeclarations"].([]any)) != 1 || !reflect.DeepEqual(tools[1], map[string]any{"googleSearch": map[string]any{}}) || !reflect.DeepEqual(tools[2], map[string]any{"codeExecution": map[string]any{}}) {
			t.Fatalf("Gemini omitted empty builtin objects or function tool: %#v", tools)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(stream)), Request: r}, nil
	})}
	resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, "gemini-2.5-flash", []NativeTool{{Name: "local", Parameters: map[string]any{"type": "object"}}}, nil, nil, nil)
	if err != nil || !called || resp.Text != "2" || len(resp.ServerResults) != 1 || resp.ServerResults[0].Code != "print(2)" || resp.ServerResults[0].Output != "2\n" {
		t.Fatalf("Gemini hosted result lost: response=%#v error=%v called=%v", resp, err, called)
	}
}

func TestBuiltinCapabilitiesConfigPersistenceAndDiscovery(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test")
	api, thinker := newTestAPI()
	thinker.config.path = filepath.Join(t.TempDir(), "config.json")
	put := func(payload string, want int) {
		t.Helper()
		w := httptest.NewRecorder()
		api.config(w, httptest.NewRequest(http.MethodPut, "/config", bytes.NewBufferString(payload)))
		if w.Code != want {
			t.Fatalf("PUT status=%d want=%d: %s", w.Code, want, w.Body.String())
		}
	}
	put(`{"provider":{"name":"openai","builtins":{"web_search_preview":{"enabled":true,"options":{"search_context_size":"low"}},"code_execution":{"enabled":true}}}}`, 200)
	put(`{"provider":{"name":"openai","builtins":{"web_search":{"enabled":false}}}}`, 200)
	put(`{"provider":{"name":"openai","builtins":{"file_search":{"enabled":true}}}}`, 400)
	w := httptest.NewRecorder()
	api.config(w, httptest.NewRequest(http.MethodGet, "/config", nil))
	var response struct {
		Providers    []ProviderConfig               `json:"providers"`
		Capabilities map[string][]BuiltinCapability `json:"builtin_capabilities"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Providers) != 1 || len(response.Capabilities["openai"]) != 4 {
		t.Fatalf("GET dropped capabilities or provider: %s", w.Body.String())
	}
	for _, capability := range response.Capabilities["openai"] {
		if capability.Enabled != (capability.Name == "code_execution") {
			t.Fatalf("GET incorrect enabled flag: %#v", capability)
		}
	}
	restored := &Config{path: thinker.config.path}
	if err := restored.load(); err != nil {
		t.Fatal(err)
	}
	got := restored.GetProviders()[0].Builtins
	if len(got) != 2 || got["web_search"].Enabled || !got["code_execution"].Enabled {
		t.Fatalf("persisted config lost disable, retained alias, or committed invalid PUT: %#v", got)
	}
	got["code_execution"] = BuiltinToolConfig{Enabled: false}
	if !restored.GetProviders()[0].Builtins["code_execution"].Enabled {
		t.Fatal("GetProviders exposes mutable configuration")
	}
}
