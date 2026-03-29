// MCP server for simulated time-series monitoring and alerting.
// State in METRICS_DATA_DIR: metrics.json, thresholds.json, alerts.json
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
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

type MetricPoint struct {
	Service   string  `json:"service"`
	Metric    string  `json:"metric"`
	Value     float64 `json:"value"`
	Timestamp string  `json:"timestamp"`
}

type Threshold struct {
	Service string  `json:"service"`
	Metric  string  `json:"metric"`
	Max     float64 `json:"max"`
}

type Alert struct {
	ID           string  `json:"id"`
	Service      string  `json:"service"`
	Metric       string  `json:"metric"`
	Value        float64 `json:"value"`
	Threshold    float64 `json:"threshold"`
	Time         string  `json:"time"`
	Acknowledged bool    `json:"acknowledged"`
}

var (
	dataDir    string
	history    []MetricPoint
	thresholds []Threshold
	alerts     []Alert
	alertSeq   int
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

func saveHistory()    { saveJSON("metrics.json", history) }
func saveThresholds() { saveJSON("thresholds.json", thresholds) }
func saveAlerts()     { saveJSON("alerts.json", alerts) }

func saveJSON(name string, v any) {
	data, _ := json.MarshalIndent(v, "", "  ")
	os.WriteFile(filepath.Join(dataDir, name), data, 0644)
}

func loadAll() {
	loadJSON("metrics.json", &history)
	loadJSON("thresholds.json", &thresholds)
	loadJSON("alerts.json", &alerts)
	alertSeq = len(alerts)
}

func loadJSON(name string, v any) {
	data, err := os.ReadFile(filepath.Join(dataDir, name))
	if err != nil {
		return
	}
	json.Unmarshal(data, v)
}

// checkThresholds creates alerts for any current violations.
func checkThresholds(service string, metrics map[string]float64) {
	for _, th := range thresholds {
		if th.Service != service {
			continue
		}
		val, ok := metrics[th.Metric]
		if !ok {
			continue
		}
		if val > th.Max {
			alertSeq++
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("A-%d", alertSeq),
				Service:   service,
				Metric:    th.Metric,
				Value:     val,
				Threshold: th.Max,
				Time:      time.Now().UTC().Format(time.RFC3339),
			})
			saveAlerts()
		}
	}
}

func handleToolCall(id int64, name string, args map[string]string) {
	switch name {
	case "get_metrics":
		service := args["service"]
		if service == "" {
			respondError(id, -32602, "service is required")
			return
		}
		// Reload from disk (test phases may seed metrics.json)
		loadJSON("metrics.json", &history)

		// Start with random baseline
		now := time.Now().UTC().Format(time.RFC3339)
		base := map[string]float64{
			"cpu":        30 + rand.Float64()*20,
			"memory":     50 + rand.Float64()*15,
			"error_rate": rand.Float64() * 2,
			"latency_ms": 50 + rand.Float64()*100,
		}
		// Override with most recent seeded/recorded values for this service
		for _, pt := range history {
			if pt.Service == service {
				base[pt.Metric] = pt.Value
			}
		}
		// Record and check
		var points []MetricPoint
		for metric, val := range base {
			pt := MetricPoint{Service: service, Metric: metric, Value: val, Timestamp: now}
			points = append(points, pt)
			history = append(history, pt)
		}
		saveHistory()
		checkThresholds(service, base)

		data, _ := json.Marshal(base)
		textResult(id, string(data))

	case "get_history":
		service := args["service"]
		minutes, _ := strconv.Atoi(args["minutes"])
		if service == "" {
			respondError(id, -32602, "service is required")
			return
		}
		if minutes <= 0 {
			minutes = 10
		}
		cutoff := time.Now().Add(-time.Duration(minutes) * time.Minute)
		var result []MetricPoint
		for _, pt := range history {
			if pt.Service != service {
				continue
			}
			t, err := time.Parse(time.RFC3339, pt.Timestamp)
			if err != nil || t.Before(cutoff) {
				continue
			}
			result = append(result, pt)
		}
		data, _ := json.Marshal(result)
		textResult(id, string(data))

	case "set_threshold":
		service := args["service"]
		metric := args["metric"]
		maxVal, _ := strconv.ParseFloat(args["max"], 64)
		if service == "" || metric == "" {
			respondError(id, -32602, "service and metric are required")
			return
		}
		// Update or add
		found := false
		for i, th := range thresholds {
			if th.Service == service && th.Metric == metric {
				thresholds[i].Max = maxVal
				found = true
				break
			}
		}
		if !found {
			thresholds = append(thresholds, Threshold{Service: service, Metric: metric, Max: maxVal})
		}
		saveThresholds()
		textResult(id, fmt.Sprintf("OK: threshold %s/%s max=%.2f", service, metric, maxVal))

	case "get_alerts":
		var unacked []Alert
		for _, a := range alerts {
			if !a.Acknowledged {
				unacked = append(unacked, a)
			}
		}
		if unacked == nil {
			unacked = []Alert{}
		}
		data, _ := json.Marshal(unacked)
		textResult(id, string(data))

	case "acknowledge_alert":
		alertID := args["id"]
		if alertID == "" {
			respondError(id, -32602, "id is required")
			return
		}
		found := false
		for i, a := range alerts {
			if a.ID == alertID {
				alerts[i].Acknowledged = true
				found = true
				break
			}
		}
		if !found {
			textResult(id, fmt.Sprintf("ERROR: alert %s not found", alertID))
			return
		}
		saveAlerts()
		textResult(id, fmt.Sprintf("OK: alert %s acknowledged", alertID))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("METRICS_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
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
				"serverInfo":     map[string]string{"name": "metrics", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "get_metrics",
						"description": "Get current metrics for a service (cpu, memory, error_rate, latency_ms). Also checks thresholds and creates alerts.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"service": map[string]string{"type": "string", "description": "Service name (e.g. api, web, worker)"},
							},
							"required": []string{"service"},
						},
					},
					{
						"name":        "get_history",
						"description": "Get metric history for a service over the last N minutes.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"service": map[string]string{"type": "string", "description": "Service name"},
								"minutes": map[string]string{"type": "string", "description": "How many minutes of history (default: 10)"},
							},
							"required": []string{"service"},
						},
					},
					{
						"name":        "set_threshold",
						"description": "Set an alert threshold for a service metric. Alert fires when value exceeds max.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"service": map[string]string{"type": "string", "description": "Service name"},
								"metric":  map[string]string{"type": "string", "description": "Metric name (cpu, memory, error_rate, latency_ms)"},
								"max":     map[string]string{"type": "string", "description": "Maximum acceptable value"},
							},
							"required": []string{"service", "metric", "max"},
						},
					},
					{
						"name":        "get_alerts",
						"description": "Get all unacknowledged alerts.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "acknowledge_alert",
						"description": "Acknowledge an alert by ID.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id": map[string]string{"type": "string", "description": "Alert ID"},
							},
							"required": []string{"id"},
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
