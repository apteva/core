// MCP server for ad budget monitoring and cost-per-lead tracking.
// State in ADS_DATA_DIR: budgets.json, spend.json, alerts.json
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
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

type AdBudget struct {
	AdID        string  `json:"ad_id"`
	DailyBudget float64 `json:"daily_budget"`
	MaxCPL      float64 `json:"max_cpl"`
	Status      string  `json:"status"` // active, paused
	UpdatedAt   string  `json:"updated_at"`
}

type SpendRecord struct {
	AdID      string  `json:"ad_id"`
	Amount    float64 `json:"amount"`
	Leads     int     `json:"leads"`
	Timestamp string  `json:"timestamp"`
}

type Alert struct {
	AdID    string  `json:"ad_id"`
	Message string  `json:"message"`
	CPL     float64 `json:"cpl"`
	MaxCPL  float64 `json:"max_cpl"`
	Time    string  `json:"time"`
	Read    bool    `json:"read"`
}

var (
	dataDir string
	budgets map[string]*AdBudget
	spend   []SpendRecord
	alerts  []Alert
)

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

func saveBudgets() {
	data, _ := json.MarshalIndent(budgets, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "budgets.json"), data, 0644)
}

func saveSpend() {
	data, _ := json.MarshalIndent(spend, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "spend.json"), data, 0644)
}

func saveAlerts() {
	data, _ := json.MarshalIndent(alerts, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "alerts.json"), data, 0644)
}

func loadAll() {
	if data, err := os.ReadFile(filepath.Join(dataDir, "budgets.json")); err == nil {
		json.Unmarshal(data, &budgets)
	}
	if data, err := os.ReadFile(filepath.Join(dataDir, "spend.json")); err == nil {
		json.Unmarshal(data, &spend)
	}
	if data, err := os.ReadFile(filepath.Join(dataDir, "alerts.json")); err == nil {
		json.Unmarshal(data, &alerts)
	}
}

func handleToolCall(id int64, name string, args map[string]string) {
	switch name {
	case "get_budgets":
		var result []AdBudget
		for _, b := range budgets {
			result = append(result, *b)
		}
		data, _ := json.Marshal(result)
		textResult(id, string(data))

	case "set_budget":
		adID := args["ad_id"]
		if adID == "" {
			respondError(id, -32602, "ad_id is required")
			return
		}
		daily, _ := strconv.ParseFloat(args["daily_budget"], 64)
		maxCPL, _ := strconv.ParseFloat(args["max_cpl"], 64)
		budgets[adID] = &AdBudget{
			AdID:        adID,
			DailyBudget: daily,
			MaxCPL:      maxCPL,
			Status:      "active",
			UpdatedAt:   time.Now().UTC().Format(time.RFC3339),
		}
		saveBudgets()
		textResult(id, fmt.Sprintf("OK: budget set for %s (daily=$%.2f, max_cpl=$%.2f)", adID, daily, maxCPL))

	case "record_spend":
		adID := args["ad_id"]
		if adID == "" {
			respondError(id, -32602, "ad_id is required")
			return
		}
		amount, _ := strconv.ParseFloat(args["amount"], 64)
		leads, _ := strconv.Atoi(args["leads"])
		spend = append(spend, SpendRecord{
			AdID:      adID,
			Amount:    amount,
			Leads:     leads,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		})
		saveSpend()
		textResult(id, fmt.Sprintf("OK: recorded $%.2f / %d leads for %s", amount, leads, adID))

	case "get_performance":
		// Aggregate spend per ad
		type perf struct {
			AdID       string  `json:"ad_id"`
			TotalSpend float64 `json:"total_spend"`
			TotalLeads int     `json:"total_leads"`
			CPL        float64 `json:"cpl"`
			MaxCPL     float64 `json:"max_cpl"`
			Status     string  `json:"status"`
			OverBudget bool    `json:"over_budget"`
		}
		agg := make(map[string]*perf)
		for _, s := range spend {
			p, ok := agg[s.AdID]
			if !ok {
				p = &perf{AdID: s.AdID}
				agg[s.AdID] = p
			}
			p.TotalSpend += s.Amount
			p.TotalLeads += s.Leads
		}
		var result []perf
		for adID, p := range agg {
			if p.TotalLeads > 0 {
				p.CPL = p.TotalSpend / float64(p.TotalLeads)
			}
			if b, ok := budgets[adID]; ok {
				p.MaxCPL = b.MaxCPL
				p.Status = b.Status
				p.OverBudget = p.CPL > b.MaxCPL && p.TotalLeads > 0
			}
			result = append(result, *p)
		}
		data, _ := json.Marshal(result)
		textResult(id, string(data))

	case "pause_ad":
		adID := args["ad_id"]
		if adID == "" {
			respondError(id, -32602, "ad_id is required")
			return
		}
		b, ok := budgets[adID]
		if !ok {
			textResult(id, fmt.Sprintf("ERROR: ad %s not found", adID))
			return
		}
		b.Status = "paused"
		b.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
		saveBudgets()

		// Create alert
		alert := Alert{
			AdID:    adID,
			Message: fmt.Sprintf("Ad %s paused — over CPL limit", adID),
			MaxCPL:  b.MaxCPL,
			Time:    time.Now().UTC().Format(time.RFC3339),
		}
		// Compute current CPL for alert
		var totalSpend float64
		var totalLeads int
		for _, s := range spend {
			if s.AdID == adID {
				totalSpend += s.Amount
				totalLeads += s.Leads
			}
		}
		if totalLeads > 0 {
			alert.CPL = totalSpend / float64(totalLeads)
		}
		alert.Message = fmt.Sprintf("Ad %s paused — CPL $%.2f exceeds limit $%.2f (%d leads, $%.2f spent)",
			adID, alert.CPL, b.MaxCPL, totalLeads, totalSpend)
		alerts = append(alerts, alert)
		saveAlerts()
		textResult(id, alert.Message)

	case "get_alerts":
		var unread []Alert
		for _, a := range alerts {
			if !a.Read {
				unread = append(unread, a)
			}
		}
		// Mark as read
		for i := range alerts {
			alerts[i].Read = true
		}
		saveAlerts()
		if unread == nil {
			unread = []Alert{}
		}
		data, _ := json.Marshal(unread)
		textResult(id, string(data))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("ADS_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
	budgets = make(map[string]*AdBudget)
	loadAll()

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
				"serverInfo":     map[string]string{"name": "ads", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "get_budgets",
						"description": "List all ad budgets with their CPL thresholds and status.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "set_budget",
						"description": "Create or update an ad budget with daily limit and max cost-per-lead.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"ad_id":        map[string]string{"type": "string", "description": "Ad identifier"},
								"daily_budget": map[string]string{"type": "string", "description": "Daily budget in dollars"},
								"max_cpl":      map[string]string{"type": "string", "description": "Maximum acceptable cost per lead in dollars"},
							},
							"required": []string{"ad_id", "max_cpl"},
						},
					},
					{
						"name":        "record_spend",
						"description": "Record ad spend and lead count for tracking. Used after processing a lead file.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"ad_id":  map[string]string{"type": "string", "description": "Ad identifier"},
								"amount": map[string]string{"type": "string", "description": "Amount spent in dollars"},
								"leads":  map[string]string{"type": "string", "description": "Number of leads generated"},
							},
							"required": []string{"ad_id", "amount", "leads"},
						},
					},
					{
						"name":        "get_performance",
						"description": "Get cost-per-lead performance for all ads. Shows total spend, leads, CPL, and whether over budget.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "pause_ad",
						"description": "Pause an ad that is over budget. Creates an alert.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"ad_id": map[string]string{"type": "string", "description": "Ad to pause"},
							},
							"required": []string{"ad_id"},
						},
					},
					{
						"name":        "get_alerts",
						"description": "Get unread alerts (ads paused, budget exceeded, etc.).",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
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
