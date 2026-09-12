package core

import "testing"

func TestOpenAIBaseURLDefaultsToPublicAPI(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "")
	if got := openAIBaseURL(); got != "https://api.openai.com/v1" {
		t.Fatalf("openAIBaseURL() = %q, want the public API root", got)
	}
}

func TestOpenAIBaseURLHonoursEnvAndTrimsSlash(t *testing.T) {
	for _, tc := range []struct{ set, want string }{
		{"https://gateway.internal/v1", "https://gateway.internal/v1"},
		{"https://gateway.internal/v1/", "https://gateway.internal/v1"},
		{"  https://gateway.internal/v1  ", "https://gateway.internal/v1"},
	} {
		t.Setenv("OPENAI_BASE_URL", tc.set)
		if got := openAIBaseURL(); got != tc.want {
			t.Errorf("openAIBaseURL() with %q = %q, want %q", tc.set, got, tc.want)
		}
	}
}

// The bug this guards: OPENAI_BASE_URL was accepted by the UI and
// injected into the core process, but every OpenAI call site used a
// hardcoded api.openai.com constant, so requests silently went to the
// real API and came back 401 with OpenAI's own error text.
func TestOpenAIProvidersRouteThroughBaseURL(t *testing.T) {
	t.Setenv("OPENAI_BASE_URL", "https://gateway.internal/v1")

	compat, ok := NewOpenAIProvider("sk-test").(*OpenAICompatProvider)
	if !ok {
		t.Fatal("NewOpenAIProvider did not return *OpenAICompatProvider")
	}
	if want := "https://gateway.internal/v1/chat/completions"; compat.url != want {
		t.Errorf("chat completions URL = %q, want %q", compat.url, want)
	}

	native, ok := NewOpenAINativeProvider("sk-test").(*OpenAINativeProvider)
	if !ok {
		t.Fatal("NewOpenAINativeProvider did not return *OpenAINativeProvider")
	}
	if want := "https://gateway.internal/v1/responses"; native.responsesURL != want {
		t.Errorf("responses URL = %q, want %q", native.responsesURL, want)
	}
}

func TestEmbeddingBackendHonoursOpenAIBaseURL(t *testing.T) {
	t.Setenv("FIREWORKS_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OPENAI_BASE_URL", "https://gateway.internal/v1")

	backend := detectEmbeddingBackend()
	if backend == nil {
		t.Fatal("detectEmbeddingBackend() = nil, want the OpenAI backend")
	}
	if want := "https://gateway.internal/v1/embeddings"; backend.URL != want {
		t.Errorf("embeddings URL = %q, want %q", backend.URL, want)
	}
}

// A gateway behind OPENAI_BASE_URL may only serve its non-OpenAI models
// (Claude, Llama, …) on /chat/completions, while Apteva defaults the
// "openai" provider to the Responses API. OPENAI_API_STYLE switches it.
func TestOpenAIAPIStyleSelectsChatCompletions(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test")
	t.Setenv("OPENAI_BASE_URL", "https://gateway.internal/v1")

	t.Setenv("OPENAI_API_STYLE", "")
	if _, ok := createProviderByName("openai").(*OpenAINativeProvider); !ok {
		t.Error("default must stay on the Responses API provider")
	}

	for _, style := range []string{"chat", "chat_completions", "chat-completions", "completions", "  CHAT  "} {
		t.Setenv("OPENAI_API_STYLE", style)
		p, ok := createProviderByName("openai").(*OpenAICompatProvider)
		if !ok {
			t.Errorf("OPENAI_API_STYLE=%q did not select the chat completions provider", style)
			continue
		}
		if want := "https://gateway.internal/v1/chat/completions"; p.url != want {
			t.Errorf("OPENAI_API_STYLE=%q url = %q, want %q", style, p.url, want)
		}
	}

	// An unrecognised value must not silently switch APIs.
	t.Setenv("OPENAI_API_STYLE", "responses")
	if _, ok := createProviderByName("openai").(*OpenAINativeProvider); !ok {
		t.Error("unrecognised OPENAI_API_STYLE must fall back to the Responses API")
	}
}
