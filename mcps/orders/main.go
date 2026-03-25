// MCP server for bakery order management.
// Communicates via stdio JSON-RPC (MCP protocol).
//
// State is read/written in ORDERS_DATA_DIR:
//   orders.json — array of {id, item, qty, status} objects
//
// All tool calls appended to audit.jsonl.
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

type Order struct {
	ID     string `json:"id"`
	Item   string `json:"item"`
	Qty    int    `json:"qty"`
	Status string `json:"status"` // "pending", "preparing", "ready", "cancelled"
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

func textResult(id int64, text string) {
	respond(id, map[string]any{
		"content": []map[string]string{{"type": "text", "text": text}},
	})
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

func loadOrders() []Order {
	data, err := os.ReadFile(filepath.Join(dataDir, "orders.json"))
	if err != nil {
		return nil
	}
	var orders []Order
	json.Unmarshal(data, &orders)
	return orders
}

func saveOrders(orders []Order) {
	data, _ := json.MarshalIndent(orders, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "orders.json"), data, 0644)
}

func findOrder(id string) (*Order, int) {
	orders := loadOrders()
	for i := range orders {
		if orders[i].ID == id {
			return &orders[i], i
		}
	}
	return nil, -1
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "get_orders":
		orders := loadOrders()
		// Return only pending orders by default
		var pending []Order
		for _, o := range orders {
			if o.Status == "pending" {
				pending = append(pending, o)
			}
		}
		if pending == nil {
			pending = []Order{}
		}
		data, _ := json.Marshal(pending)
		textResult(id, string(data))

	case "get_order":
		orderID := args["id"]
		if orderID == "" {
			respondError(id, -32602, "id is required")
			return
		}
		order, _ := findOrder(orderID)
		if order == nil {
			textResult(id, fmt.Sprintf("Order %s not found", orderID))
			return
		}
		data, _ := json.Marshal(order)
		textResult(id, string(data))

	case "update_order":
		orderID := args["id"]
		status := args["status"]
		if orderID == "" || status == "" {
			respondError(id, -32602, "id and status are required")
			return
		}
		orders := loadOrders()
		found := false
		for i := range orders {
			if orders[i].ID == orderID {
				orders[i].Status = status
				found = true
				break
			}
		}
		if !found {
			textResult(id, fmt.Sprintf("Order %s not found", orderID))
			return
		}
		saveOrders(orders)
		textResult(id, fmt.Sprintf("Order %s updated to %s", orderID, status))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("ORDERS_DATA_DIR")
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
				"serverInfo":     map[string]string{"name": "orders", "version": "1.0.0"},
			})

		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "get_orders",
						"description": "Get all pending bakery orders. Returns a JSON array of orders with id, item, qty, and status fields.",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
						},
					},
					{
						"name":        "get_order",
						"description": "Get details of a specific order by ID.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id": map[string]string{"type": "string", "description": "Order ID"},
							},
							"required": []string{"id"},
						},
					},
					{
						"name":        "update_order",
						"description": "Update the status of an order. Valid statuses: preparing, ready, cancelled.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id":     map[string]string{"type": "string", "description": "Order ID"},
								"status": map[string]string{"type": "string", "description": "New status: preparing, ready, or cancelled"},
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
