package core

import (
	"net"
	"net/url"
	"testing"
)

func TestBlockedOutboundIP(t *testing.T) {
	for _, raw := range []string{"127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"} {
		if !blockedOutboundIP(net.ParseIP(raw)) {
			t.Errorf("%s was not blocked", raw)
		}
	}
	if blockedOutboundIP(net.ParseIP("8.8.8.8")) {
		t.Fatal("public address was blocked")
	}
}

func TestValidatePublicHTTPURLRejectsPrivateAndUnsafeURLs(t *testing.T) {
	for _, raw := range []string{
		"http://127.0.0.1/private",
		"http://169.254.169.254/latest/meta-data",
		"file:///etc/passwd",
		"http://user:pass@example.com/",
	} {
		if _, err := validatePublicHTTPURL(raw); err == nil {
			t.Errorf("unsafe URL accepted: %s", raw)
		}
	}
}

func TestSameOriginURL(t *testing.T) {
	base, _ := url.Parse("https://example.com/mcp")
	same, _ := url.Parse("https://example.com/next")
	differentPort, _ := url.Parse("https://example.com:8443/next")
	differentHost, _ := url.Parse("https://other.example/next")
	if !sameOriginURL(base, same) || sameOriginURL(base, differentPort) || sameOriginURL(base, differentHost) {
		t.Fatal("same-origin comparison produced the wrong result")
	}
}

func TestRuntimeMCPURLRequiresConfigOrAllowlist(t *testing.T) {
	t.Setenv("APTEVA_MCP_CONNECT_ALLOWLIST", "allowed.example,https://second.example:8443")
	cfg := &Config{MCPServers: []MCPServerConfig{{Name: "local", URL: "http://127.0.0.1:5280/api/apps/x/mcp"}}}
	if !runtimeMCPURLAllowed(cfg, "http://127.0.0.1:5280/api/apps/x/mcp") {
		t.Fatal("exact configured URL was rejected")
	}
	if !runtimeMCPURLAllowed(cfg, "https://allowed.example/mcp") {
		t.Fatal("allowlisted host was rejected")
	}
	if !runtimeMCPURLAllowed(cfg, "https://second.example:8443/mcp") {
		t.Fatal("allowlisted origin was rejected")
	}
	if runtimeMCPURLAllowed(cfg, "https://blocked.example/mcp") {
		t.Fatal("unconfigured URL was accepted")
	}
}

func TestFetchMediaRejectsPrivateAddress(t *testing.T) {
	if _, _, err := fetchMediaAsBase64("http://127.0.0.1/secret"); err == nil {
		t.Fatal("private media URL was fetched")
	}
}
