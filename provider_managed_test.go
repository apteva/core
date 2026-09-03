package core

import "testing"

func TestManagedProviderUsesLocalGatewayCredential(t *testing.T) {
	t.Setenv("APTEVA_MANAGED_LLM_URL", "http://127.0.0.1:5280/api/llm/chat/completions")
	t.Setenv("APTEVA_API_KEY", "core_test")
	provider, ok := createProviderByName("managed").(*OpenAICompatProvider)
	if !ok || provider == nil {
		t.Fatal("managed provider was not created")
	}
	if provider.url != "http://127.0.0.1:5280/api/llm/chat/completions" || provider.apiKey != "core_test" {
		t.Fatalf("managed provider=%+v", provider)
	}
	if provider.name != "managed" || provider.authHeader != "Bearer" {
		t.Fatalf("managed identity/auth=%q/%q", provider.name, provider.authHeader)
	}
}
