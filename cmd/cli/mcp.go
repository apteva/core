package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
)

// mcpServer is a minimal MCP HTTP server exposing tools for the core to call.
type mcpServer struct {
	port     int
	listener net.Listener

	// Channels for TUI communication
	respond chan string          // cli_respond: core sends text to display
	askCh   chan string          // cli_ask: core sends a question
	askReply chan string         // cli_ask: user sends answer back
	statusCh chan statusUpdate   // cli_status: core updates status line

	mu       sync.Mutex
	closed   bool
}

type statusUpdate struct {
	Line  string
	Level string // "info", "warn", "alert"
}

func newMCPServer() (*mcpServer, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	s := &mcpServer{
		port:     ln.Addr().(*net.TCPAddr).Port,
		listener: ln,
		respond:  make(chan string, 64),
		askCh:    make(chan string, 1),
		askReply: make(chan string, 1),
		statusCh: make(chan statusUpdate, 16),
	}
	return s, nil
}

func (s *mcpServer) url() string {
	return fmt.Sprintf("http://127.0.0.1:%d", s.port)
}

func (s *mcpServer) serve() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handle)
	http.Serve(s.listener, mux)
}

func (s *mcpServer) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		s.listener.Close()
	}
}

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Result  any         `json:"result,omitempty"`
	Error   *rpcError   `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (s *mcpServer) handle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}

	var req rpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON-RPC", http.StatusBadRequest)
		return
	}

	// Notifications (no ID) — just ack
	if req.ID == nil {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	var result any
	var rpcErr *rpcError

	switch req.Method {
	case "initialize":
		result = map[string]any{
			"protocolVersion": "2025-03-26",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo": map[string]string{
				"name":    "apteva-cli",
				"version": "1.0.0",
			},
		}

	case "tools/list":
		result = map[string]any{
			"tools": []map[string]any{
				{
					"name":        "respond",
					"description": "Send a message to the root user's terminal. You MUST use this tool to reply to any message from [cli]. This is the ONLY way to communicate back to the root user. When a root user connects (you receive '[cli] operator connected'), always greet them with a short welcome message using this tool.",
					"inputSchema": map[string]any{
						"type":     "object",
						"required": []string{"text"},
						"properties": map[string]any{
							"text": map[string]any{
								"type":        "string",
								"description": "The message to display on the root user's terminal",
							},
						},
					},
				},
				{
					"name":        "ask",
					"description": "Ask the root user a question and wait for their typed response. Use this when you need input or clarification from the operator. Blocks until they reply.",
					"inputSchema": map[string]any{
						"type":     "object",
						"required": []string{"question"},
						"properties": map[string]any{
							"question": map[string]any{
								"type":        "string",
								"description": "The question to ask the root user",
							},
						},
					},
				},
				{
					"name":        "status",
					"description": "Update the status line on the root user's terminal. Use for brief state indicators.",
					"inputSchema": map[string]any{
						"type":     "object",
						"required": []string{"line"},
						"properties": map[string]any{
							"line": map[string]any{
								"type":        "string",
								"description": "Status text to display",
							},
							"level": map[string]any{
								"type":        "string",
								"description": "Severity: info, warn, or alert",
								"enum":        []string{"info", "warn", "alert"},
							},
						},
					},
				},
			},
		}

	case "tools/call":
		result, rpcErr = s.handleToolCall(req.Params)

	default:
		rpcErr = &rpcError{Code: -32601, Message: "method not found"}
	}

	resp := rpcResponse{JSONRPC: "2.0", ID: req.ID}
	if rpcErr != nil {
		resp.Error = rpcErr
	} else {
		resp.Result = result
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *mcpServer) handleToolCall(params json.RawMessage) (any, *rpcError) {
	var call struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(params, &call); err != nil {
		return nil, &rpcError{Code: -32602, Message: "invalid params"}
	}

	textResult := func(text string) any {
		return map[string]any{
			"content": []map[string]string{{"type": "text", "text": text}},
		}
	}

	switch call.Name {
	case "respond":
		text, _ := call.Arguments["text"].(string)
		if text == "" {
			return nil, &rpcError{Code: -32602, Message: "text required"}
		}
		s.respond <- text
		return textResult("delivered"), nil

	case "ask":
		question, _ := call.Arguments["question"].(string)
		if question == "" {
			return nil, &rpcError{Code: -32602, Message: "question required"}
		}
		s.askCh <- question
		// Block until user replies
		answer := <-s.askReply
		return textResult(answer), nil

	case "status":
		line, _ := call.Arguments["line"].(string)
		level, _ := call.Arguments["level"].(string)
		if level == "" {
			level = "info"
		}
		s.statusCh <- statusUpdate{Line: line, Level: level}
		return textResult("ok"), nil

	default:
		return nil, &rpcError{Code: -32602, Message: fmt.Sprintf("unknown tool: %s", call.Name)}
	}
}
