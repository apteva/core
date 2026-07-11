package core

import (
	"bufio"
	"encoding/base64"
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
	Meta        map[string]any `json:"_meta,omitempty"`
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

// MCPServerConfig is stored in config.json.
//
// Note: the legacy `main_access` field is intentionally absent. Earlier
// versions of core split MCPs into "main" (tools eagerly registered to
// the main thread) and "catalog" (tools held off main, attachable only
// by spawning a sub-thread with mcp="name"). That distinction is gone:
// every MCP attached here is connected and indexed, every thread
// activates the subset it needs via search_tools or spawn-time MCPNames.
// Old configs containing main_access:true|false deserialize cleanly —
// json.Unmarshal silently drops the unknown field.
type MCPServerConfig struct {
	Name      string            `json:"name"`
	Command   string            `json:"command,omitempty"`   // stdio transport
	Args      []string          `json:"args,omitempty"`      // stdio transport
	Env       map[string]string `json:"env,omitempty"`       // stdio transport
	Transport string            `json:"transport,omitempty"` // "stdio" (default) or "http"
	URL       string            `json:"url,omitempty"`       // http transport
	// NoSpawn, when true, hides this server's tools from sub-thread
	// search_tools results and refuses sub-thread spawn(mcps=[...])
	// attachments. Used for infrastructure-level servers the host
	// wires in for main-thread-only responsibilities (management
	// gateway, outbound channel bridges) where letting a worker
	// invoke them would be a privilege escalation. Core has no
	// knowledge of which names are "system" — the host sets this
	// flag when registering those entries. The privileged HTTP spawn
	// path (POST /threads/{id}) sets SpawnOpts.BypassNoSpawn to
	// punch through this filter for system-initiated workers
	// (channelchat's chat-handling thread needs `channels`).
	NoSpawn bool `json:"no_spawn,omitempty"`
}

// MCPConn is the interface for any MCP server connection (stdio or HTTP)
type MCPConn interface {
	GetName() string
	ListTools() ([]mcpToolDef, error)
	CallTool(name string, args map[string]string) (ToolResponse, error)
	Close()
}

type schemaAwareMCPConn interface {
	CallToolTyped(name string, args map[string]string, inputSchema map[string]any) (ToolResponse, error)
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
	deadErr error
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
	srv.scanner.Buffer(make([]byte, 64*1024), 16<<20)

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
	err := s.scanner.Err()
	if err == nil {
		err = io.EOF
	}
	s.failPending(fmt.Errorf("MCP connection closed: %w", err))
}

func (s *MCPServer) failPending(err error) {
	s.pendMu.Lock()
	defer s.pendMu.Unlock()
	if s.deadErr == nil {
		s.deadErr = err
	}
	for id, ch := range s.pending {
		ch <- jsonRPCResponse{ID: id, Error: &jsonRPCError{Code: -32000, Message: s.deadErr.Error()}}
		delete(s.pending, id)
	}
}

func (s *MCPServer) call(method string, params any) (json.RawMessage, error) {
	id := s.nextID.Add(1)

	ch := make(chan jsonRPCResponse, 1)
	s.pendMu.Lock()
	if s.deadErr != nil {
		err := s.deadErr
		s.pendMu.Unlock()
		return nil, err
	}
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
		s.pendMu.Lock()
		delete(s.pending, id)
		s.pendMu.Unlock()
		return nil, fmt.Errorf("write: %w", err)
	}

	select {
	case resp := <-ch:
		if resp.Error != nil {
			return nil, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	case <-time.After(mcpCallTimeout):
		s.pendMu.Lock()
		delete(s.pending, id)
		s.pendMu.Unlock()
		return nil, fmt.Errorf("MCP call timed out after %s", mcpCallTimeout)
	}
}

// mcpCallTimeout is the deadline for a single tool invocation. Bumped
// from 30s because legitimate long-running tools (audio transcription,
// large-file downloads, OCR) can legitimately take a minute or two.
// Short enough that a hung MCP still surfaces as a timeout error within
// a reasonable window for retry.
const mcpCallTimeout = 3 * time.Minute

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
func (s *MCPServer) CallTool(name string, args map[string]string) (ToolResponse, error) {
	return s.CallToolTyped(name, args, nil)
}

func (s *MCPServer) CallToolTyped(name string, args map[string]string, inputSchema map[string]any) (ToolResponse, error) {
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

func (s *MCPServer) GetName() string { return s.Name }

func (s *MCPServer) Close() {
	s.failPending(fmt.Errorf("MCP connection closed"))
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
	}
}

// mcpProxyHandler returns a tool handler that proxies calls to an MCP
// server. If blobs is non-nil, the handler transparently rehydrates
// file-ref arguments into real _binary envelopes before dispatch and
// rewrites any _binary envelope returned by the tool into a compact
// _file handle — so large binaries never traverse the LLM context.
// Pass nil for blobs to get straight proxy behaviour (legacy path).
func mcpProxyHandler(server MCPConn, toolName string, opts ...any) func(args map[string]string) ToolResponse {
	var inputSchema map[string]any
	var blobs *BlobStore
	for _, opt := range opts {
		switch v := opt.(type) {
		case map[string]any:
			inputSchema = v
		case *BlobStore:
			blobs = v
		}
	}
	return func(args map[string]string) ToolResponse {
		if blobs != nil {
			args = blobs.RehydrateFileRefs(args)
		}
		var (
			resp ToolResponse
			err  error
		)
		if inputSchema != nil {
			if typed, ok := server.(schemaAwareMCPConn); ok {
				resp, err = typed.CallToolTyped(toolName, args, inputSchema)
			} else {
				resp, err = server.CallTool(toolName, args)
			}
		} else {
			resp, err = server.CallTool(toolName, args)
		}
		if err != nil {
			return ToolResponse{Text: fmt.Sprintf("error: %v", err), IsError: true}
		}
		var image []byte
		result, image := extractMCPResultImage(resp.Text)
		if blobs != nil {
			result = blobs.RewriteBinaryToHandle(result)
		}
		return ToolResponse{Text: result, Image: image, IsError: resp.IsError}
	}
}

func extractMCPResultImage(text string) (string, []byte) {
	var obj map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &obj); err != nil {
		return text, nil
	}
	raw, ok := obj["screenshot"]
	if !ok {
		return text, nil
	}
	image, ok := decodeNestedBinaryImage(raw)
	if !ok {
		return text, nil
	}
	obj["screenshot"] = "attached as image"
	delete(obj, "screenshot_b64")
	if out, err := json.Marshal(obj); err == nil {
		return string(out), image
	}
	return text, image
}

func decodeNestedBinaryImage(v any) ([]byte, bool) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	if binary, _ := m["_binary"].(bool); !binary {
		return nil, false
	}
	mime, _ := m["mimeType"].(string)
	if !strings.HasPrefix(mime, "image/") {
		return nil, false
	}
	b64, _ := m["base64"].(string)
	if b64 == "" {
		return nil, false
	}
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, false
	}
	return data, true
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

// connectAnyMCPWithRetry wraps connectAnyMCP with a bounded retry
// schedule. Defense in depth against transient startup races: when
// apteva-server restarts both the agent and the app sidecars, the
// agent can boot a fraction of a second before the app MCP proxy is
// ready. Without this, that race silently drops the MCP server from
// the agent's connected list for the rest of the process lifetime.
//
// Retry budget: 4 attempts at 0s, 1s, 2s, 4s — total ~7s wait worst
// case, which is well under any reasonable user-perceivable boot
// pause and well over the 0.1–0.5s window where transient failures
// typically resolve. A genuinely broken URL still fails fast on the
// final attempt.
func connectAnyMCPWithRetry(cfg MCPServerConfig) (MCPConn, error) {
	delays := []time.Duration{0, time.Second, 2 * time.Second, 4 * time.Second}
	var lastErr error
	for i, d := range delays {
		if d > 0 {
			time.Sleep(d)
		}
		srv, err := connectAnyMCP(cfg)
		if err == nil {
			if i > 0 {
				fmt.Fprintf(os.Stderr, "MCP %s: connected on attempt %d/%d\n", cfg.Name, i+1, len(delays))
			}
			return srv, nil
		}
		lastErr = err
		fmt.Fprintf(os.Stderr, "MCP %s: attempt %d/%d failed: %v\n", cfg.Name, i+1, len(delays), err)
	}
	return nil, lastErr
}

// connectAndRegisterMCP connects to MCP servers from config and
// registers tools into the registry and (if non-nil) the index. Every
// MCP tool is registered with MCP=true so it stays hidden from the
// per-turn provider tool list until a thread explicitly activates it
// via search_tools or spawn-time MCPNames preload. The index supplies
// the searchable surface for those activations.
//
// If blobs is non-nil, every registered tool is wrapped so that binary
// arguments and results flow through the blob store (see
// mcpProxyHandler). Pass nil for blobs to register plain proxies.
func connectAndRegisterMCP(configs []MCPServerConfig, registry *ToolRegistry, index *ToolIndex, blobs *BlobStore) []MCPConn {
	var servers []MCPConn
	for _, cfg := range configs {
		srv, err := connectAnyMCPWithRetry(cfg)
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
				Handler:     mcpProxyHandler(srv, tool.Name, tool.InputSchema, blobs),
				InputSchema: tool.InputSchema,
				MCP:         true, // hidden until activated; old MainAccess flag is gone
				MCPServer:   cfg.Name,
				WakeOnResult: normalizeWakeOnResultPolicy(
					func() any {
						if tool.Meta == nil {
							return nil
						}
						return tool.Meta[wakeOnResultMetaKey]
					}(),
				),
			})
		}

		// Mirror into the searchable index. Done after registry.Register
		// so the index's "this tool exists" claim is always consistent
		// with what the registry can actually dispatch.
		if index != nil {
			index.Add(cfg.Name, tools, cfg.NoSpawn)
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
