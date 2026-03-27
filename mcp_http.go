package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// MCPHTTPServer connects to an MCP server via Streamable HTTP transport.
// Per MCP spec 2025-03-26: POST for requests, single endpoint.
type MCPHTTPServer struct {
	Name      string
	url       string
	sessionID string
	nextID    atomic.Int64
	client    *http.Client
}

func connectMCPHTTP(name, url string) (*MCPHTTPServer, error) {
	srv := &MCPHTTPServer{
		Name:   name,
		url:    url,
		client: &http.Client{Timeout: 30 * time.Second},
	}

	// Initialize
	result, headers, err := srv.callWithHeaders("initialize", map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name":    "cogito",
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

	httpReq, err := http.NewRequest("POST", s.url, bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if s.sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", s.sessionID)
	}

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1000))
		logMsg("MCP-HTTP", fmt.Sprintf("error %d: %s", resp.StatusCode, string(body)))
		return nil, resp.Header, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("read error: %v", err))
		return nil, resp.Header, fmt.Errorf("read: %w", err)
	}

	var rpcResp jsonRPCResponse
	if err := json.Unmarshal(body, &rpcResp); err != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("parse error: %v body=%s", err, string(body[:min(len(body), 200)])))
		return nil, resp.Header, fmt.Errorf("parse: %w", err)
	}

	if rpcResp.Error != nil {
		logMsg("MCP-HTTP", fmt.Sprintf("rpc error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message))
		return nil, resp.Header, fmt.Errorf("MCP error %d: %s", rpcResp.Error.Code, rpcResp.Error.Message)
	}

	resultPreview := string(rpcResp.Result)
	if len(resultPreview) > 200 {
		resultPreview = resultPreview[:200] + "..."
	}
	logMsg("MCP-HTTP", fmt.Sprintf("ok id=%d result=%s", id, resultPreview))
	return rpcResp.Result, resp.Header, nil
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

	httpReq, err := http.NewRequest("POST", s.url, bytes.NewReader(data))
	if err != nil {
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if s.sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", s.sessionID)
	}

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return
	}
	resp.Body.Close()
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

func (s *MCPHTTPServer) CallTool(name string, args map[string]string) (string, error) {
	arguments := make(map[string]any)
	for k, v := range args {
		arguments[k] = v
	}

	result, err := s.call("tools/call", map[string]any{
		"name":      name,
		"arguments": arguments,
	})
	if err != nil {
		return "", err
	}

	var callResult mcpCallResult
	if err := json.Unmarshal(result, &callResult); err != nil {
		return "", fmt.Errorf("parse result: %w", err)
	}

	var texts []string
	for _, c := range callResult.Content {
		if c.Type == "text" {
			texts = append(texts, c.Text)
		}
	}
	return strings.Join(texts, "\n"), nil
}

func (s *MCPHTTPServer) GetName() string { return s.Name }

func (s *MCPHTTPServer) Close() {
	// No process to kill — just stop using it
}
