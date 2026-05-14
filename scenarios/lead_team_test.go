package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var leadTeamScenario = Scenario{
	Name: "LeadTeam",
	Directive: `You manage a lead processing pipeline for a business running Facebook ads.

Spawn and maintain 3 threads:
1. "file-intake" — receives file URLs, fetches them, checks for duplicates, marks as pending.
   Tools: files_fetch_file, files_list_files, files_file_status, send, done
2. "file-processor" — reads CSV files, extracts leads, records ad spend.
   Tools: files_read_csv, files_file_status, ads_record_spend, storage_store, send, done
3. "ad-monitor" — checks ad performance periodically, pauses over-budget ads, sends alerts.
   Tools: ads_get_performance, ads_get_budgets, ads_pause_ad, ads_get_alerts, send, done

When you receive a console event with a file URL, forward it to file-intake.
When file-intake finishes, tell file-processor to process it.
When file-processor finishes, tell ad-monitor to check performance.`,
	MCPServers: []MCPServerConfig{
		{Name: "files", Command: "", Env: map[string]string{"FILES_DATA_DIR": "{{dataDir}}"}},
		{Name: "ads", Command: "", Env: map[string]string{"ADS_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed ad budgets
		WriteJSONFile(t, dir, "budgets.json", map[string]*struct {
			AdID        string  `json:"ad_id"`
			DailyBudget float64 `json:"daily_budget"`
			MaxCPL      float64 `json:"max_cpl"`
			Status      string  `json:"status"`
			UpdatedAt   string  `json:"updated_at"`
		}{
			"fb-summer-2026":  {AdID: "fb-summer-2026", DailyBudget: 100, MaxCPL: 10.0, Status: "active", UpdatedAt: "2026-03-01T00:00:00Z"},
			"fb-winter-promo": {AdID: "fb-winter-promo", DailyBudget: 50, MaxCPL: 15.0, Status: "active", UpdatedAt: "2026-03-01T00:00:00Z"},
		})

		// Create CSV batch 1 — normal leads, within budget
		csv1 := "name,email,phone,ad_id,cost\n" +
			"Alice Smith,alice@example.com,555-0101,fb-summer-2026,8.50\n" +
			"Bob Jones,bob@example.com,555-0102,fb-summer-2026,9.20\n" +
			"Carol White,carol@example.com,555-0103,fb-winter-promo,12.00\n"
		os.WriteFile(filepath.Join(dir, "leads-batch-1.csv"), []byte(csv1), 0644)

		// Create CSV batch 2 — expensive leads that push fb-summer-2026 over CPL limit
		csv2 := "name,email,phone,ad_id,cost\n" +
			"Dave Brown,dave@example.com,555-0201,fb-summer-2026,25.00\n" +
			"Eve Black,eve@example.com,555-0202,fb-summer-2026,30.00\n" +
			"Frank Green,frank@example.com,555-0203,fb-summer-2026,28.00\n"
		os.WriteFile(filepath.Join(dir, "leads-batch-2.csv"), []byte(csv2), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 threads spawned",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				ids := ThreadIDs(th)
				return len(ids) >= 3
			},
		},
		{
			Name:    "File ingestion — batch 1 processed",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-1.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
					}
					// Check if file was processed
					data, err := os.ReadFile(filepath.Join(dir, "files.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), `"processed"`)
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify file record exists
				data, _ := os.ReadFile(filepath.Join(dir, "files.json"))
				if !strings.Contains(string(data), "leads-batch-1.csv") {
					t.Error("expected leads-batch-1.csv in files.json")
				}
			},
		},
		{
			Name:    "Duplicate rejection — same file rejected",
			Timeout: 90 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				startTime := time.Now()
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-1.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
						startTime = time.Now()
					}
					// Wait a bit for the system to process and verify no new file added
					return time.Since(startTime) > 15*time.Second
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "files.json"))
				// Should still only have 1 file entry (the original)
				var files map[string]any
				json.Unmarshal(data, &files)
				if len(files) != 1 {
					t.Errorf("expected 1 file record (duplicate rejected), got %d", len(files))
				}
			},
		},
		{
			Name:    "Ad monitoring — expensive batch triggers pause",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "leads-batch-2.csv")
						th.InjectConsole("New lead file: " + csvPath)
						injected = true
					}
					// Check if fb-summer-2026 was paused
					data, err := os.ReadFile(filepath.Join(dir, "budgets.json"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), `"paused"`)
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify the right ad was paused
				data, _ := os.ReadFile(filepath.Join(dir, "budgets.json"))
				var budgets map[string]json.RawMessage
				json.Unmarshal(data, &budgets)
				if b, ok := budgets["fb-summer-2026"]; ok {
					if !strings.Contains(string(b), `"paused"`) {
						t.Error("expected fb-summer-2026 to be paused")
					}
				} else {
					t.Error("fb-summer-2026 not found in budgets")
				}
				// Verify winter promo still active
				if b, ok := budgets["fb-winter-promo"]; ok {
					if !strings.Contains(string(b), `"active"`) {
						t.Error("expected fb-winter-promo to still be active")
					}
				}
				// Verify alert was created
				alertData, _ := os.ReadFile(filepath.Join(dir, "alerts.json"))
				if !strings.Contains(string(alertData), "fb-summer-2026") {
					t.Error("expected alert for fb-summer-2026")
				}
			},
		},
	},
	Timeout: 5 * time.Minute,
}

func TestScenario_LeadTeam(t *testing.T) {
	filesBin := BuildMCPBinary(t, "mcps/files")
	adsBin := BuildMCPBinary(t, "mcps/ads")
	storageBin := BuildMCPBinary(t, "mcps/storage")
	t.Logf("built files=%s ads=%s storage=%s", filesBin, adsBin, storageBin)

	s := leadTeamScenario
	s.MCPServers[0].Command = filesBin
	s.MCPServers[1].Command = adsBin
	s.MCPServers[2].Command = storageBin
	RunScenario(t, s)
}
