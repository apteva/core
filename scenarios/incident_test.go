package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var incidentScenario = Scenario{
	Name: "Incident",
	Directive: `You are the on-call SRE coordinator for a web platform with services: api, web, worker.

Spawn and maintain 3 threads:
1. "monitor" — continuously reads metrics for all services, watches for threshold violations.
   Tools: metrics_get_metrics, metrics_get_history, metrics_set_threshold, metrics_get_alerts, send, done
2. "responder" — investigates alerts, reads config/logs, applies fixes.
   Tools: codebase_read_file, codebase_write_file, codebase_search, metrics_get_history, metrics_acknowledge_alert, send, done
3. "comms" — sends status updates to stakeholders via pushover.
   Tools: pushover_send_notification, send, done

On startup, have monitor set thresholds:
- cpu max 80 for all services
- error_rate max 5 for all services
- latency_ms max 200 for api

Workflow:
- Monitor checks metrics and reports alerts to you.
- You dispatch responder to investigate and fix.
- You tell comms to send status updates.
- After fix is applied, have monitor verify recovery.`,
	MCPServers: []MCPServerConfig{
		{Name: "metrics", Command: "", Env: map[string]string{"METRICS_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Create a config file the responder can read/edit
		os.MkdirAll(filepath.Join(dir, "config"), 0755)
		os.WriteFile(filepath.Join(dir, "config", "api.yaml"), []byte("max_connections: 100\ntimeout_ms: 5000\ncache_enabled: true\n"), 0644)
		os.WriteFile(filepath.Join(dir, "config", "worker.yaml"), []byte("concurrency: 10\nretry_limit: 3\n"), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — thresholds set",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Outcome only: thresholds got configured. Whether the
				// agent spawned a monitoring team to do it or set them
				// inline is its own call.
				data, err := os.ReadFile(filepath.Join(dir, "thresholds.json"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), "cpu") && strings.Contains(string(data), "error_rate")
			},
		},
		{
			Name:    "Incident — CPU spike detected and investigated",
			Timeout: 180 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Seed a CPU spike in metrics history so get_metrics returns high values
				WriteJSONFile(t, dir, "metrics.json", []map[string]any{
					{"service": "api", "metric": "cpu", "value": 95.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
					{"service": "api", "metric": "error_rate", "value": 12.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
					{"service": "api", "metric": "latency_ms", "value": 350.0, "timestamp": time.Now().UTC().Format(time.RFC3339)},
				})
			},
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("ALERT: api service showing high CPU and errors. Please investigate immediately.")
						injected = true
					}
					// Check if alerts were generated
					data, err := os.ReadFile(filepath.Join(dir, "alerts.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), "api")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Alerts should exist (acknowledged or not — the key is that they were detected)
				data, _ := os.ReadFile(filepath.Join(dir, "alerts.json"))
				if !strings.Contains(string(data), "api") {
					t.Error("expected alerts for api service")
				}
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_Incident(t *testing.T) {
	metricsBin := BuildMCPBinary(t, "mcps/metrics")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	pushoverBin := BuildMCPBinary(t, "mcps/pushover")
	t.Logf("built metrics=%s codebase=%s pushover=%s", metricsBin, codebaseBin, pushoverBin)

	s := incidentScenario
	s.MCPServers[0].Command = metricsBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = pushoverBin
	RunScenario(t, s)
}
