package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// MCP JSON-RPC types
type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type mcpToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpToolsListResult struct {
	Tools []mcpToolDef `json:"tools"`
}

type mcpCallResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

// MCPServerConfig is stored in config.json
type MCPServerConfig struct {
	Name       string            `json:"name"`
	Command    string            `json:"command,omitempty"`    // stdio transport
	Args       []string          `json:"args,omitempty"`       // stdio transport
	Env        map[string]string `json:"env,omitempty"`        // stdio transport
	Transport  string            `json:"transport,omitempty"`  // "stdio" (default) or "http"
	URL        string            `json:"url,omitempty"`        // http transport
	MainAccess bool              `json:"main_access,omitempty"` // if true, tools are callable by main thread (not just sub-threads)
}

// MCPConn is the interface for any MCP server connection (stdio or HTTP)
type MCPConn interface {
	GetName() string
	ListTools() ([]mcpToolDef, error)
	CallTool(name string, args map[string]string) (string, error)
	Close()
}

// MCPServer manages a running MCP server subprocess (stdio transport)
type MCPServer struct {
	Name    string
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	scanner *bufio.Scanner
	mu      sync.Mutex
	nextID  atomic.Int64
	pending map[int64]chan jsonRPCResponse
	pendMu  sync.Mutex
}

func connectMCP(cfg MCPServerConfig) (*MCPServer, error) {
	cmd := exec.Command(cfg.Command, cfg.Args...)

	// Set environment
	cmd.Env = os.Environ()
	for k, v := range cfg.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %q: %w", cfg.Command, err)
	}

	srv := &MCPServer{
		Name:    cfg.Name,
		cmd:     cmd,
		stdin:   stdin,
		scanner: bufio.NewScanner(stdout),
		pending: make(map[int64]chan jsonRPCResponse),
	}
	srv.scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	// Read responses in background
	go srv.readLoop()

	// Initialize
	_, err = srv.call("initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name":    "apteva-core",
			"version": "1.0.0",
		},
	})
	if err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("initialize: %w", err)
	}

	// Send initialized notification (no response expected)
	srv.notify("notifications/initialized", nil)

	return srv, nil
}

func (s *MCPServer) readLoop() {
	for s.scanner.Scan() {
		line := s.scanner.Text()
		if line == "" {
			continue
		}
		var resp jsonRPCResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			continue
		}
		s.pendMu.Lock()
		if ch, ok := s.pending[resp.ID]; ok {
			ch <- resp
			delete(s.pending, resp.ID)
		}
		s.pendMu.Unlock()
	}
}

func (s *MCPServer) call(method string, params any) (json.RawMessage, error) {
	id := s.nextID.Add(1)

	ch := make(chan jsonRPCResponse, 1)
	s.pendMu.Lock()
	s.pending[id] = ch
	s.pendMu.Unlock()

	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	s.mu.Lock()
	data, _ := json.Marshal(req)
	_, err := fmt.Fprintf(s.stdin, "%s\n", data)
	s.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}

	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	case <-time.After(30 * time.Second):
		s.pendMu.Lock()
		delete(s.pending, id)
		s.pendMu.Unlock()
		return nil, fmt.Errorf("MCP call timed out after 30s")
	}
}

func (s *MCPServer) notify(method string, params any) {
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	s.mu.Lock()
	data, _ := json.Marshal(req)
	fmt.Fprintf(s.stdin, "%s\n", data)
	s.mu.Unlock()
}

// ListTools calls tools/list on the MCP server
func (s *MCPServer) ListTools() ([]mcpToolDef, error) {
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

// CallTool invokes a tool on the MCP server
func (s *MCPServer) CallTool(name string, args map[string]string) (string, error) {
	// Convert string args to any for JSON
	// Parse string values that look like JSON arrays/objects/numbers/booleans
	// so they're sent as proper JSON types to the MCP server
	arguments := make(map[string]any)
	for k, v := range args {
		if len(v) > 0 && (v[0] == '[' || v[0] == '{') {
			var parsed any
			if json.Unmarshal([]byte(v), &parsed) == nil {
				arguments[k] = parsed
				continue
			}
		}
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

func (s *MCPServer) GetName() string { return s.Name }

func (s *MCPServer) Close() {
	s.stdin.Close()
	s.cmd.Process.Kill()
	s.cmd.Wait()
}

// mcpProxyHandler returns a tool handler that proxies calls to an MCP server
func mcpProxyHandler(server MCPConn, toolName string) func(args map[string]string) ToolResponse {
	return func(args map[string]string) ToolResponse {
		result, err := server.CallTool(toolName, args)
		if err != nil {
			return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
		}
		return ToolResponse{Text: result}
	}
}

// buildMCPSyntax generates [[tool arg1="..." arg2="..."]] syntax from MCP schema
func buildMCPSyntax(name string, schema map[string]any) string {
	var parts []string
	if props, ok := schema["properties"].(map[string]any); ok {
		keys := make([]string, 0, len(props))
		for k := range props {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf(`%s="..."`, k))
		}
	}
	if len(parts) == 0 {
		return fmt.Sprintf("[[%s]]", name)
	}
	return fmt.Sprintf("[[%s %s]]", name, strings.Join(parts, " "))
}

// connectAnyMCP connects to an MCP server using the appropriate transport.
func connectAnyMCP(cfg MCPServerConfig) (MCPConn, error) {
	if cfg.Transport == "http" || cfg.URL != "" {
		return connectMCPHTTP(cfg.Name, cfg.URL)
	}
	return connectMCP(cfg)
}

// connectAndRegisterMCP connects to MCP servers from config and registers tools
func connectAndRegisterMCP(configs []MCPServerConfig, registry *ToolRegistry, memory *MemoryStore) []MCPConn {
	var servers []MCPConn
	for _, cfg := range configs {
		srv, err := connectAnyMCP(cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "MCP %s: %v\n", cfg.Name, err)
			continue
		}

		tools, err := srv.ListTools()
		if err != nil {
			fmt.Fprintf(os.Stderr, "MCP %s tools: %v\n", cfg.Name, err)
			srv.Close()
			continue
		}

		for _, tool := range tools {
			// Prefix with server name to avoid collisions
			fullName := cfg.Name + "_" + tool.Name
			syntax := buildMCPSyntax(fullName, tool.InputSchema)

			registry.Register(&ToolDef{
				Name:        fullName,
				Description: fmt.Sprintf("[%s] %s", cfg.Name, tool.Description),
				Syntax:      syntax,
				Rules:       fmt.Sprintf("Provided by MCP server '%s'.", cfg.Name),
				Handler:     mcpProxyHandler(srv, tool.Name),
				InputSchema: tool.InputSchema,
				MCP:         !cfg.MainAccess,
				MCPServer:   cfg.Name,
			})
		}

		// Embed new tools
		if memory != nil {
			go func(name string, tools []mcpToolDef) {
				for _, tool := range tools {
					fullName := name + "_" + tool.Name
					emb, err := memory.embed(fullName + ": " + tool.Description)
					if err == nil {
						t := registry.Get(fullName)
						if t != nil {
							t.Embedding = emb
						}
					}
				}
			}(cfg.Name, tools)
		}

		servers = append(servers, srv)
		fmt.Fprintf(os.Stderr, "MCP %s (%s): %d tools registered\n", cfg.Name, cfg.transport(), len(tools))
	}
	return servers
}

func (c MCPServerConfig) transport() string {
	if c.Transport == "http" || c.URL != "" {
		return "http"
	}
	return "stdio"
}
