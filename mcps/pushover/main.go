// MCP server for Pushover notifications.
// Communicates via stdio JSON-RPC (MCP protocol).
// Requires PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN env vars.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
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

func sendPushover(userKey, apiToken, title, message string, priority int) error {
	resp, err := http.PostForm("https://api.pushover.net/1/messages.json", url.Values{
		"token":    {apiToken},
		"user":     {userKey},
		"title":    {title},
		"message":  {message},
		"priority": {fmt.Sprintf("%d", priority)},
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("pushover API returned %d", resp.StatusCode)
	}
	return nil
}

func main() {
	userKey := os.Getenv("PUSHOVER_USER_KEY")
	apiToken := os.Getenv("PUSHOVER_API_TOKEN")

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

		// Notifications (no ID) — just acknowledge silently
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
					"name":    "pushover",
					"version": "1.0.0",
				},
			})

		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "send_notification",
						"description": "Send a push notification to the user's phone via Pushover. Use for alerts, reminders, or important updates that need immediate attention.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"title":   map[string]string{"type": "string", "description": "Notification title"},
								"message": map[string]string{"type": "string", "description": "Notification body text"},
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

			switch params.Name {
			case "send_notification":
				title := params.Arguments["title"]
				message := params.Arguments["message"]
				if message == "" {
					respondError(id, -32602, "message is required")
					continue
				}
				if title == "" {
					title = "Cogito"
				}

				if userKey == "" || apiToken == "" {
					respondError(id, -32603, "PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN not set")
					continue
				}

				err := sendPushover(userKey, apiToken, title, message, 0)
				if err != nil {
					respond(id, map[string]any{
						"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("error: %v", err)}},
						"isError": true,
					})
				} else {
					respond(id, map[string]any{
						"content": []map[string]string{{"type": "text", "text": fmt.Sprintf("Notification sent: %s", strings.TrimSpace(title+" — "+message))}},
					})
				}
			default:
				respondError(id, -32601, fmt.Sprintf("unknown tool: %s", params.Name))
			}

		default:
			respondError(id, -32601, fmt.Sprintf("unknown method: %s", req.Method))
		}
	}
}
