// MCP server for a content calendar/schedule.
// State in SCHEDULE_DATA_DIR: schedule.json, audit.jsonl
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

type Slot struct {
	ID      string `json:"id"`
	Channel string `json:"channel"`
	Topic   string `json:"topic"`
	Time    string `json:"time"`
	Status  string `json:"status"` // planned, content_ready, posted
}

type AuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

var dataDir string

func respond(id int64, result any) {
	data, _ := json.Marshal(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
	fmt.Println(string(data))
}

func respondError(id int64, code int, msg string) {
	data, _ := json.Marshal(jsonRPCResponse{
		JSONRPC: "2.0", ID: id,
		Error: &struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		}{code, msg},
	})
	fmt.Println(string(data))
}

func textResult(id int64, text string) {
	respond(id, map[string]any{
		"content": []map[string]string{{"type": "text", "text": text}},
	})
}

func audit(tool string, args map[string]string) {
	entry := AuditEntry{Time: time.Now().UTC().Format(time.RFC3339), Tool: tool, Args: args}
	data, _ := json.Marshal(entry)
	f, _ := os.OpenFile(filepath.Join(dataDir, "audit.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if f != nil {
		f.WriteString(string(data) + "\n")
		f.Close()
	}
}

func loadSchedule() []Slot {
	data, err := os.ReadFile(filepath.Join(dataDir, "schedule.json"))
	if err != nil {
		return nil
	}
	var slots []Slot
	json.Unmarshal(data, &slots)
	return slots
}

func saveSchedule(slots []Slot) {
	data, _ := json.MarshalIndent(slots, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "schedule.json"), data, 0644)
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "get_schedule":
		slots := loadSchedule()
		if slots == nil {
			slots = []Slot{}
		}
		data, _ := json.Marshal(slots)
		textResult(id, string(data))

	case "update_slot":
		slotID := args["id"]
		status := args["status"]
		if slotID == "" || status == "" {
			respondError(id, -32602, "id and status are required")
			return
		}
		slots := loadSchedule()
		found := false
		for i := range slots {
			if slots[i].ID == slotID {
				slots[i].Status = status
				found = true
				break
			}
		}
		if !found {
			textResult(id, fmt.Sprintf("Slot %s not found", slotID))
			return
		}
		saveSchedule(slots)
		textResult(id, fmt.Sprintf("Slot %s updated to %s", slotID, status))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("SCHEDULE_DATA_DIR")
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
				"serverInfo":     map[string]string{"name": "schedule", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "get_schedule",
						"description": "Get the content calendar. Returns all scheduled slots with id, channel, topic, time, and status (planned/content_ready/posted).",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "update_slot",
						"description": "Update a schedule slot's status. Use to mark slots as content_ready or posted.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id":     map[string]string{"type": "string", "description": "Slot ID"},
								"status": map[string]string{"type": "string", "description": "New status: content_ready or posted"},
							},
							"required": []string{"id", "status"},
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
