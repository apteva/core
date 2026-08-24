package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// MCPHTTPServer connects to an MCP server via Streamable HTTP transport.
// Per MCP spec 2025-03-26: POST for requests, single endpoint.
//
// Compatibility notes for real-world hosted MCP servers (observed against
// Composio's backend.composio.dev):
//   - Some servers host the MCP endpoint at a sub-path and respond to the
//     parent path with HTTP 307 → Location (appending "/mcp"). Go's default
//     client strips the POST body on 307 redirects, so we disable auto-
//     redirects and re-issue the POST manually with the body intact.
//     After the first successful hop we store the resolved URL and skip the
//     redirect on every subsequent call.
//   - Some servers return SSE-framed responses (`Content-Type: text/event-stream`
//     with one or more `event: message\ndata: {...}\n\n` frames) instead of
//     plain JSON, even for POST requests. We parse both.
type MCPHTTPServer struct {
	Name      string
	mu        sync.Mutex // guards url (after redirect)
	url       string
	sessionID string
	nextID    atomic.Int64
	client    *http.Client
}

func connectMCPHTTP(name, url string) (*MCPHTTPServer, error) {
	srv := &MCPHTTPServer{
		Name: name,
		url:  url,
		client: &http.Client{
			Timeout: 3 * time.Minute,
			// Disable auto-redirects so we can re-issue POSTs with the body
			// preserved. http.Client would otherwise follow 307/308 but drop
			// the body for non-idempotent methods.
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}

	// Initialize
	result, headers, err := srv.callWithHeaders("initialize", map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name":    "apteva-core",
			"version": "1.0.0",
		},
	})
	if err != nil {
		return nil, fmt.Errorf("initialize: %w", err)
	}
	_ = result

	// Store session ID if provided
	if sid := headers.Get("Mcp-Session-Id"); sid != "" {
		srv.sessionID = sid
	}

	// Send initialized notification
	srv.notify("notifications/initialized", nil)

	return srv, nil
}

// decodeMCPBody extracts the JSON-RPC payload from either a plain JSON body
// or an SSE-framed `event: message\ndata: {...}` body. Returns the raw JSON
// bytes suitable for unmarshaling into jsonRPCResponse.
func decodeMCPBody(contentType string, body []byte) ([]byte, error) {
	ct := strings.ToLower(contentType)
	trimmed := strings.TrimSpace(string(body))
	if strings.Contains(ct, "text/event-stream") ||
		strings.HasPrefix(trimmed, "event:") ||
		strings.HasPrefix(trimmed, "data:") {
		// SSE: walk lines and take the last `data:` payload.
		var last string
		for _, line := range strings.Split(trimmed, "\n") {
			line = strings.TrimRight(line, "\r")
			if strings.HasPrefix(line, "data:") {
				last = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
		}
		if last == "" {
			return nil, fmt.Errorf("SSE body had no data: line")
		}
		return []byte(last), nil
	}
	return body, nil
}

// doPOST issues a POST to the current URL, manually following 307/308
// redirects up to 3 hops while preserving the body. Returns the response
// headers + raw body bytes. Updates srv.url in-place to the post-redirect
// URL so subsequent calls skip the redirect entirely.
func (s *MCPHTTPServer) doPOST(body []byte, includeAccept bool) (http.Header, []byte, int, error) {
	s.mu.Lock()
	currentURL := s.url
	s.mu.Unlock()

	for attempt := 0; attempt < 4; attempt++ {
		httpReq, err := http.NewRequest("POST", currentURL, bytes.NewReader(body))
		if err != nil {
			return nil, nil, 0, fmt.Errorf("request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if includeAccept {
			// Accept both JSON and SSE — some servers (Composio) respond
			// with SSE even for POSTs.
			httpReq.Header.Set("Accept", "application/json, text/event-stream")
		}
		if agentID := os.Getenv("AGENT_ID"); agentID != "" && isAptevaAppMCPURL(currentURL) {
			httpReq.Header.Set("X-Apteva-Caller-Agent", agentID)
		}
		if s.sessionID != "" {
			httpReq.Header.Set("Mcp-Session-Id", s.sessionID)
		}

		resp, err := s.client.Do(httpReq)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("http: %w", err)
		}
		if resp.StatusCode == http.StatusTemporaryRedirect || resp.StatusCode == http.StatusPermanentRedirect {
			loc := resp.Header.Get("Location")
			resp.Body.Close()
			if loc == "" {
				return nil, nil, resp.StatusCode, fmt.Errorf("redirect with no Location header")
			}
			// Location may be relative (RFC 7231 §7.1.2). Resolve against
			// the URL we just POSTed to so "/mcp/inner" → full URL.
			base, berr := url.Parse(currentURL)
			if berr != nil {
				return nil, nil, resp.StatusCode, fmt.Errorf("parse current URL: %w", berr)
			}
			target, perr := url.Parse(loc)
			if perr != nil {
				return nil, nil, resp.StatusCode, fmt.Errorf("parse Location: %w", perr)
			}
			resolved := base.ResolveReference(target)
			if !sameOriginURL(base, resolved) {
				return nil, nil, resp.StatusCode, fmt.Errorf("cross-origin MCP redirect refused: %s", resolved.Redacted())
			}
			currentURL = resolved.String()
			continue
		}
		respBody, rerr := io.ReadAll(io.LimitReader(resp.Body, 4_000_000))
		resp.Body.Close()
		if rerr != nil {
			return resp.Header, nil, resp.StatusCode, fmt.Errorf("read: %w", rerr)
		}

		// Pin the post-redirect URL so the next call skips the redirect.
		if attempt > 0 {
			s.mu.Lock()
			s.url = currentURL
			s.mu.Unlock()
			logMsg("MCP-HTTP", fmt.Sprintf("resolved redirect → %s", currentURL))
		}
		return resp.Header, respBody, resp.StatusCode, nil
	}
	return nil, nil, 0, fmt.Errorf("too many redirects")
}

func isAptevaAppMCPURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := u.Hostname()
	if host != "127.0.0.1" && host != "localhost" && host != "::1" {
		return false
	}
	return (strings.HasPrefix(u.Path, "/api/apps/") || strings.HasPrefix(u.Path, "/api/environment-app-gateway/")) &&
		strings.HasSuffix(u.Path, "/mcp")
}

func (s *MCPHTTPServer) callWithHeaders(method string, params any) (json.RawMessage, http.Header, error) {
	id := s.nextID.Add(1)
	logMsg("MCP-HTTP", fmt.Sprintf("call %s id=%d url=%s", method, id, s.url))

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	data, _ := json.Marshal(req)

	headers, body, status, err := s.doPOST(data, true)
	if err != nil {
		return nil, headers, err
	}
	if status < 200 || status >= 300 {
		snippet := string(body)
		if len(snippet) > 500 {
			snippet = snippet[:500] + "…"
		}
		logMsg("MCP-HTTP", fmt.Sprintf("error %d: %s", status, snippet))
		return nil, headers, fmt.Errorf("HTTP %d: %s", status, snippet)
	}

	payload, err := decodeMCPBody(headers.Get("Content-Type"), body)
	if err != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("decode error: %v body=%s", err, string(body[:min(len(body), 200)])))
		return nil, headers, fmt.Errorf("decode: %w", err)
	}

	var rpcResp jsonRPCResponse
	if err := json.Unmarshal(payload, &rpcResp); err != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("parse error: %v payload=%s", err, string(payload[:min(len(payload), 200)])))
		return nil, headers, fmt.Errorf("parse: %w", err)
	}

	if rpcResp.Error != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message))
		return nil, headers, fmt.Errorf("MCP error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}

	resultPreview := string(rpcResp.Result)
	if len(resultPreview) > 200 {
		resultPreview = resultPreview[:200] + "..."
	}
	logMsg("MCP-HTTP", fmt.Sprintf("ok id=%d result=%s", id, resultPreview))

	// For tools/list, also dump the full description of every tool —
	// the truncated preview above hides whether dynamic blocks (like
	// the channels MCP's AVAILABLE COMPONENTS section) actually
	// reached the agent. Costs one extra log line per tool per
	// connect but makes that question answerable from the log
	// alone.
	if method == "tools/list" {
		var listed struct {
			Tools []struct {
				Name        string         `json:"name"`
				Description string         `json:"description"`
				InputSchema map[string]any `json:"inputSchema"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(rpcResp.Result, &listed); err == nil {
			for _, t := range listed.Tools {
				descPreview := t.Description
				if len(descPreview) > 2000 {
					descPreview = descPreview[:2000] + "..."
				}
				logMsg("MCP-HTTP", fmt.Sprintf("  tool name=%q description=%q schema_props=%v",
					t.Name, descPreview, schemaPropKeys(t.InputSchema)))
			}
		}
	}
	return rpcResp.Result, headers, nil
}

// schemaPropKeys extracts the property keys from an MCP tool's
// inputSchema for the dump log — quick way to verify a new arg
// (like respond's components) actually shows up in the schema the
// agent receives.
func schemaPropKeys(schema map[string]any) []string {
	props, _ := schema["properties"].(map[string]any)
	keys := make([]string, 0, len(props))
	for k := range props {
		keys = append(keys, k)
	}
	return keys
}

func (s *MCPHTTPServer) call(method string, params any) (json.RawMessage, error) {
	result, _, err := s.callWithHeaders(method, params)
	return result, err
}

func (s *MCPHTTPServer) notify(method string, params any) {
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	data, _ := json.Marshal(req)
	// Best-effort — ignore response. doPOST still handles redirects.
	_, _, _, _ = s.doPOST(data, false)
}

func (s *MCPHTTPServer) ListTools() ([]mcpToolDef, error) {
	result, err := s.call("tools/list", nil)
	if err != nil {
		return nil, err
	}
	var list mcpToolsListResult
	if err := json.Unmarshal(result, &list); err != nil {
		return nil, fmt.Errorf("parse tools: %w", err)
	}
	return list.Tools, nil
}

func (s *MCPHTTPServer) CallTool(name string, args map[string]string) (ToolResponse, error) {
	return s.CallToolTyped(name, args, nil)
}

func (s *MCPHTTPServer) CallToolTyped(name string, args map[string]string, inputSchema map[string]any) (ToolResponse, error) {
	arguments := mcpArgumentsFromStrings(args, inputSchema)
	result, err := s.call("tools/call", map[string]any{
		"name":      name,
		"arguments": arguments,
	})
	if err != nil {
		return ToolResponse{}, err
	}

	var callResult mcpCallResult
	if err := json.Unmarshal(result, &callResult); err != nil {
		return ToolResponse{}, fmt.Errorf("parse result: %w", err)
	}

	var texts []string
	for _, c := range callResult.Content {
		if c.Type == "text" {
			texts = append(texts, c.Text)
		}
	}
	return ToolResponse{Text: strings.Join(texts, "\n"), IsError: callResult.IsError}, nil
}

func (s *MCPHTTPServer) GetName() string { return s.Name }

func (s *MCPHTTPServer) Close() {
	// No process to kill — just stop using it
}
