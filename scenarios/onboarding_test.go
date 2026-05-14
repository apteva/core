package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var onboardingScenario = Scenario{
	Name: "Onboarding",
	Directive: `You manage new customer onboarding for a SaaS platform.

Spawn and maintain 3 threads:
1. "intake" — fetches signup CSV files, reads customer data, reports to you.
   Tools: files_fetch_file, files_read_csv, files_list_files, files_file_status, send, done
2. "provisioner" — stores customer account records using storage tools.
   Tools: codebase_write_file, codebase_list_files, storage_store, storage_get, send, done
3. "welcome" — sends onboarding notifications to new customers.
   Tools: pushover_send_notification, storage_get, send, done

When you receive a signup file URL, tell intake to fetch and read it. Then tell provisioner to create accounts. Then tell welcome to notify customers.`,
	MCPServers: []MCPServerConfig{
		{Name: "files", Command: "", Env: map[string]string{"FILES_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "pushover", Command: "", Env: map[string]string{"PUSHOVER_USER_KEY": "test", "PUSHOVER_API_TOKEN": "test"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		os.MkdirAll(filepath.Join(dir, "accounts"), 0755)
		// Signup CSV
		csv := "name,email,plan\nAlice Johnson,alice@startup.io,pro\nBob Chen,bob@bigcorp.com,enterprise\nCarol Davis,carol@freelance.me,starter\n"
		os.WriteFile(filepath.Join(dir, "signups-batch-1.csv"), []byte(csv), 0644)
	},
	Phases: []Phase{
		// No "startup — N threads spawned" phase: the phase below
		// asserts on accounts getting provisioned, not on how the
		// agent structured the work.
		{
			Name:    "Onboarding — signup to welcome message",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						csvPath := "file://" + filepath.Join(dir, "signups-batch-1.csv")
						th.InjectConsole("New signups file: " + csvPath + ". Please onboard these customers.")
						injected = true
					}
					// Check if accounts were provisioned (config files or storage entries)
					entries, _ := os.ReadDir(filepath.Join(dir, "accounts"))
					store, _ := os.ReadFile(filepath.Join(dir, "store.json"))
					s := strings.ToLower(string(store))
					return len(entries) >= 2 || strings.Contains(s, "alice") || strings.Contains(s, "bob")
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify accounts provisioned via files or storage
				entries, _ := os.ReadDir(filepath.Join(dir, "accounts"))
				store, _ := os.ReadFile(filepath.Join(dir, "store.json"))
				hasFiles := len(entries) >= 2
				hasStore := strings.Contains(string(store), "alice") || strings.Contains(string(store), "bob")
				if !hasFiles && !hasStore {
					t.Error("expected accounts provisioned (config files or storage entries)")
				}
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_Onboarding(t *testing.T) {
	filesBin := BuildMCPBinary(t, "mcps/files")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	storageBin := BuildMCPBinary(t, "mcps/storage")
	pushoverBin := BuildMCPBinary(t, "mcps/pushover")
	t.Logf("built files=%s codebase=%s storage=%s pushover=%s", filesBin, codebaseBin, storageBin, pushoverBin)

	s := onboardingScenario
	s.MCPServers[0].Command = filesBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = storageBin
	s.MCPServers[3].Command = pushoverBin
	RunScenario(t, s)
}
