// MCP server for bakery inventory management.
// Communicates via stdio JSON-RPC (MCP protocol).
//
// State is read/written in INVENTORY_DATA_DIR:
//   stock.json — map of item name → quantity
//
// All tool calls appended to audit.jsonl.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
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

func loadStock() map[string]int {
	data, err := os.ReadFile(filepath.Join(dataDir, "stock.json"))
	if err != nil {
		return map[string]int{}
	}
	var stock map[string]int
	json.Unmarshal(data, &stock)
	if stock == nil {
		stock = map[string]int{}
	}
	return stock
}

// normalizeItem finds the best matching key in stock for the given item name.
// Handles plurals and case differences.
func normalizeItem(item string, stock map[string]int) string {
	lower := strings.ToLower(item)
	// Exact match
	if _, ok := stock[lower]; ok {
		return lower
	}
	// Try removing trailing 's' (simple plural)
	if strings.HasSuffix(lower, "s") {
		singular := lower[:len(lower)-1]
		if _, ok := stock[singular]; ok {
			return singular
		}
	}
	// Try adding 's'
	if _, ok := stock[lower+"s"]; ok {
		return lower + "s"
	}
	// Try original case keys
	for k := range stock {
		if strings.EqualFold(k, item) {
			return k
		}
	}
	return lower
}

func saveStock(stock map[string]int) {
	data, _ := json.MarshalIndent(stock, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "stock.json"), data, 0644)
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "check_stock":
		item := args["item"]
		if item == "" {
			respondError(id, -32602, "item is required")
			return
		}
		stock := loadStock()
		key := normalizeItem(item, stock)
		qty, exists := stock[key]
		if !exists {
			textResult(id, fmt.Sprintf("%s: not in inventory", item))
			return
		}
		textResult(id, fmt.Sprintf("%s: %d in stock", key, qty))

	case "use_stock":
		item := args["item"]
		qtyStr := args["qty"]
		if item == "" || qtyStr == "" {
			respondError(id, -32602, "item and qty are required")
			return
		}
		qty, err := strconv.Atoi(qtyStr)
		if err != nil || qty <= 0 {
			respondError(id, -32602, "qty must be a positive integer")
			return
		}
		stock := loadStock()
		key := normalizeItem(item, stock)
		available, exists := stock[key]
		if !exists {
			textResult(id, fmt.Sprintf("FAILED: %s not in inventory", item))
			return
		}
		if available < qty {
			textResult(id, fmt.Sprintf("FAILED: only %d %s available, need %d", available, key, qty))
			return
		}
		stock[key] = available - qty
		saveStock(stock)
		textResult(id, fmt.Sprintf("OK: used %d %s, %d remaining", qty, key, stock[key]))

	case "list_stock":
		stock := loadStock()
		data, _ := json.Marshal(stock)
		textResult(id, string(data))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("INVENTORY_DATA_DIR")
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
				"serverInfo":     map[string]string{"name": "inventory", "version": "1.0.0"},
			})

		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "check_stock",
						"description": "Check how many of an item are in stock. Returns the item name and quantity available.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"item": map[string]string{"type": "string", "description": "Item name (e.g. croissant, baguette, muffin)"},
							},
							"required": []string{"item"},
						},
					},
					{
						"name":        "use_stock",
						"description": "Deduct items from inventory. Fails if insufficient stock. Returns OK with remaining count, or FAILED with reason.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"item": map[string]string{"type": "string", "description": "Item name"},
								"qty":  map[string]string{"type": "string", "description": "Quantity to deduct"},
							},
							"required": []string{"item", "qty"},
						},
					},
					{
						"name":        "list_stock",
						"description": "List all items in inventory with their current quantities.",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
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
