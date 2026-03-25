// MCP server for a fake helpdesk ticket system.
// Communicates via stdio JSON-RPC (MCP protocol).
//
// State is read from JSON files in a directory set by HELPDESK_DATA_DIR env var:
//   tickets.json  — array of {id, question} objects (open tickets)
//   kb.json       — map of keyword → answer (knowledge base)
//
// All tool calls are appended to audit.jsonl in the same directory.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

type Ticket struct {
	ID       string `json:"id"`
	Question string `json:"question"`
}

type AuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
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

func loadTickets() []Ticket {
	data, err := os.ReadFile(filepath.Join(dataDir, "tickets.json"))
	if err != nil {
		return nil
	}
	var tickets []Ticket
	json.Unmarshal(data, &tickets)
	return tickets
}

func loadKB() map[string]string {
	data, err := os.ReadFile(filepath.Join(dataDir, "kb.json"))
	if err != nil {
		return nil
	}
	var kb map[string]string
	json.Unmarshal(data, &kb)
	return kb
}

// removeTicket removes a ticket by ID from tickets.json
func removeTicket(id string) bool {
	tickets := loadTickets()
	found := false
	var remaining []Ticket
	for _, t := range tickets {
		if t.ID == id {
			found = true
			continue
		}
		remaining = append(remaining, t)
	}
	if !found {
		return false
	}
	data, _ := json.MarshalIndent(remaining, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "tickets.json"), data, 0644)
	return true
}

func lookupKB(query string) string {
	kb := loadKB()
	if kb == nil {
		return ""
	}
	query = strings.ToLower(query)
	for keyword, answer := range kb {
		kw := strings.ToLower(keyword)
		// Match if query contains keyword or keyword contains a word from the query
		if strings.Contains(query, kw) {
			return answer
		}
		for _, word := range strings.Fields(query) {
			// Strip punctuation from word edges
			word = strings.Trim(word, "?!.,;:'\"")
			if len(word) >= 4 && strings.Contains(kw, word) {
				return answer
			}
		}
	}
	return ""
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "list_tickets":
		tickets := loadTickets()
		if tickets == nil {
			tickets = []Ticket{}
		}
		data, _ := json.Marshal(tickets)
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": string(data)}},
		})

	case "reply_ticket":
		ticketID := args["id"]
		message := args["message"]
		if ticketID == "" || message == "" {
			respondError(id, -32602, "id and message are required")
			return
		}
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("Reply sent to ticket %s", ticketID)}},
		})

	case "close_ticket":
		ticketID := args["id"]
		if ticketID == "" {
			respondError(id, -32602, "id is required")
			return
		}
		removed := removeTicket(ticketID)
		if !removed {
			respond(id, map[string]any{
				"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("Ticket %s not found", ticketID)}},
			})
			return
		}
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("Ticket %s closed", ticketID)}},
		})

	case "lookup_kb":
		query := args["query"]
		if query == "" {
			respondError(id, -32602, "query is required")
			return
		}
		answer := lookupKB(query)
		if answer == "" {
			respond(id, map[string]any{
				"content": []map[string]string{{"type": "text", "text": "No results found"}},
			})
			return
		}
		respond(id, map[string]any{
			"content": []map[string]string{{"type": "text", "text": answer}},
		})

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("HELPDESK_DATA_DIR")
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

		// Notifications (no ID) — acknowledge silently
		if req.ID == nil {
			continue
		}
		id := *req.ID

		switch req.Method {
		case "initialize":
			respond(id, map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":   map[string]any{"tools": map[string]any{}},
				"serverInfo": map[string]string{
					"name":    "helpdesk",
					"version": "1.0.0",
				},
			})

		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "list_tickets",
						"description": "List all open support tickets. Returns a JSON array of tickets with id and question fields. Call periodically to check for new tickets.",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
						},
					},
					{
						"name":        "reply_ticket",
						"description": "Send a reply message to a support ticket. Use this to answer the customer's question.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id":      map[string]string{"type": "string", "description": "Ticket ID"},
								"message": map[string]string{"type": "string", "description": "Reply message to send to the customer"},
							},
							"required": []string{"id", "message"},
						},
					},
					{
						"name":        "close_ticket",
						"description": "Close a support ticket after it has been resolved. Removes it from the open tickets list.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id": map[string]string{"type": "string", "description": "Ticket ID to close"},
							},
							"required": []string{"id"},
						},
					},
					{
						"name":        "lookup_kb",
						"description": "Search the knowledge base for an answer to a question. Returns the best matching answer or 'No results found'.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"query": map[string]string{"type": "string", "description": "Search query — use keywords from the customer's question"},
							},
							"required": []string{"query"},
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
