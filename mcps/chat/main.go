// MCP server for a chat interface.
// Communicates via stdio JSON-RPC (MCP protocol).
//
// State is read/written in a directory set by CHAT_DATA_DIR env var:
//   messages.json — array of {user, text} objects (pending user messages)
//
// All replies are appended to audit.jsonl in the same directory.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Result  any    `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type UserMessage struct {
	User string `json:"user"`
	Text string `json:"text"`
}

type AuditEntry struct {
	Time    string            `json:"time"`
	Tool    string            `json:"tool"`
	Args    map[string]string `json:"args"`
}

var dataDir string

func respond(id int64, result any) {
	resp := jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result}
	data, _ := json.Marshal(resp)
	fmt.Println(string(data))
}

func respondError(id int64, code int, msg string) {
	resp := jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &struct{ Code int `json:"code"`; Message string `json:"message"` }{code, msg},
	}
	data, _ := json.Marshal(resp)
	fmt.Println(string(data))
}

func audit(tool string, args map[string]string) {
	entry := AuditEntry{
		Time: time.Now().UTC().Format(time.RFC3339),
		Tool: tool,
		Args: args,
	}
	data, _ := json.Marshal(entry)
	f, err := os.OpenFile(filepath.Join(dataDir, "audit.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	f.WriteString(string(data) + "\n")
}

func loadMessages() []UserMessage {
	data, err := os.ReadFile(filepath.Join(dataDir, "messages.json"))
	if err != nil {
		return nil
	}
	var msgs []UserMessage
	json.Unmarshal(data, &msgs)
	return msgs
}

// consumeMessages returns all pending messages and clears the file.
func consumeMessages() []UserMessage {
	msgs := loadMessages()
	if len(msgs) > 0 {
		os.WriteFile(filepath.Join(dataDir, "messages.json"), []byte("[]"), 0644)
	}
	return msgs
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "get_messages":
		msgs := consumeMessages()
		if msgs == nil {
			msgs = []UserMessage{}
		}
		data, _ := json.Marshal(msgs)
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": string(data)}},
		})

	case "send_reply":
		user := args["user"]
		message := args["message"]
		if message == "" {
			respondError(id, -32602, "message is required")
			return
		}
		if user == "" {
			user = "user"
		}
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("Reply sent to %s", user)}},
		})

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("CHAT_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var req jsonRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			continue
		}

		if req.ID == nil {
			continue
		}
		id := *req.ID

		switch req.Method {
		case "initialize":
			respond(id, map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":   map[string]any{"tools": map[string]any{}},
				"serverInfo":     map[string]string{"name": "chat", "version": "1.0.0"},
			})

		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "get_messages",
						"description": "Get pending user messages. Returns a JSON array of messages with user and text fields. Messages are consumed once read — calling again returns only new messages.",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
						},
					},
					{
						"name":        "send_reply",
						"description": "Send a reply message to a user. This is how you respond to user questions and requests.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"user":    map[string]string{"type": "string", "description": "Username to reply to"},
								"message": map[string]string{"type": "string", "description": "Reply message text"},
							},
							"required": []string{"message"},
						},
					},
				},
			})

		case "tools/call":
			var params struct {
				Name      string            `json:"name"`
				Arguments map[string]string `json:"arguments"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				respondError(id, -32602, "invalid params")
				continue
			}
			handleToolCall(id, params.Name, params.Arguments)

		default:
			respondError(id, -32601, fmt.Sprintf("unknown method: %s", req.Method))
		}
	}
}
