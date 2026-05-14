package scenarios

import (
	"os"
	"testing"
	"time"

	. "github.com/apteva/core"
)

// Token A/B measurement harnesses — NOT pass/fail behavioural tests.
// They attach a multi-MCP surface and have the agent do a light
// survey, then you diff the per-iteration `tok=in:` lines the harness
// logs across two runs:
//
//	RUN_TOKEN_AB=1 APTEVA_TOOL_SEARCH=on  go test -run TestScenario_TokenAB ...  # discovery model
//	RUN_TOKEN_AB=1 APTEVA_TOOL_SEARCH=off go test -run TestScenario_TokenAB ...  # eager: all tools every turn
//
// (Default APTEVA_TOOL_SEARCH=auto decides by surface size, so for a
// clean A/B force the mode explicitly.)
//
// Iteration 1's input-token count is the cleanest signal — the baseline
// prompt size before the agent has done anything. Eager mode carries
// the full tool surface there; the discovery model carries scaffolding
// + the BM25 preload only.
//
// Both gated behind RUN_TOKEN_AB=1 — they build many MCP binaries and
// aren't pass/fail tests. The Wait conditions are iteration-count
// based on purpose: we just need enough turns to sample, and a clean
// pass keeps the run readable.

// --- small surface: 6 MCPs (~20 tools) ---

var tokenABScenario = Scenario{
	Name: "TokenAB",
	Directive: `You are an operations assistant for a small business. Several tool
servers are connected — a helpdesk, file storage, a schedule, a CRM, a
social poster, and a web scraper.

Do a quick morning survey: check the helpdesk ticket queue, list what's
in storage, and check the schedule. Report a one-line summary of what
you found, then idle.`,
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "schedule", Env: map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"}},
		{Name: "crm", Env: map[string]string{"CRM_DATA_DIR": "{{dataDir}}"}},
		{Name: "social", Env: map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "kb.json", map[string]string{"hours": "Mon-Fri 9-5."})
		WriteJSONFile(t, dir, "tickets.json", []map[string]string{
			{"id": "t1", "question": "What are your hours?"},
		})
	},
	Phases: []Phase{{
		Name:    "Survey — sample several iterations",
		Timeout: 180 * time.Second,
		Wait: func(t *testing.T, dir string, th *Thinker) bool {
			t.Logf("  ... iteration=%d", th.Iteration())
			return th.Iteration() >= 3
		},
	}},
	Timeout: 6 * time.Minute,
}

// --- big surface: ~24 MCPs (~120+ tools) ---
//
// This is the production-shape case the small scenario can't show: a
// large attached surface where eager mode's per-turn schema cost is
// substantial. The directive is deliberately broad so no small preload
// set covers it — that's the point, it stresses the difference.

var tokenABBigScenario = Scenario{
	Name: "TokenABBig",
	Directive: `You are the back-office operations agent for a mid-sized company.
A large set of tool servers is connected — helpdesk, CRM, storage,
files, scheduling, social, content, hosting, inventory, orders,
metrics, sheets, email, and more.

Do a morning operations sweep: check the support queue, look at recent
orders, check inventory levels, and review the schedule. Then write a
one-line status summary and idle.`,
	MCPServers: []MCPServerConfig{
		{Name: "helpdesk", Env: map[string]string{"HELPDESK_DATA_DIR": "{{dataDir}}"}},
		{Name: "crm", Env: map[string]string{"CRM_DATA_DIR": "{{dataDir}}"}},
		{Name: "storage", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "files", Env: map[string]string{"FILES_DATA_DIR": "{{dataDir}}"}},
		{Name: "schedule", Env: map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"}},
		{Name: "social", Env: map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"}},
		{Name: "creative", Env: map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "cms", Env: map[string]string{"CMS_DATA_DIR": "{{dataDir}}"}},
		{Name: "hosting", Env: map[string]string{"HOSTING_DATA_DIR": "{{dataDir}}"}},
		{Name: "inventory", Env: map[string]string{"INVENTORY_DATA_DIR": "{{dataDir}}"}},
		{Name: "orders", Env: map[string]string{"ORDERS_DATA_DIR": "{{dataDir}}"}},
		{Name: "metrics", Env: map[string]string{"METRICS_DATA_DIR": "{{dataDir}}"}},
		{Name: "sheets", Env: map[string]string{"SHEETS_DATA_DIR": "{{dataDir}}"}},
		{Name: "gdocs", Env: map[string]string{"GDOCS_DATA_DIR": "{{dataDir}}"}},
		{Name: "gdrive", Env: map[string]string{"GDRIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "fake_email", Env: map[string]string{"FAKE_EMAIL_DATA_DIR": "{{dataDir}}"}},
		{Name: "fake_places", Env: map[string]string{"FAKE_PLACES_DATA_DIR": "{{dataDir}}"}},
		{Name: "ads", Env: map[string]string{"ADS_DATA_DIR": "{{dataDir}}"}},
		{Name: "market", Env: map[string]string{"MARKET_DATA_DIR": "{{dataDir}}"}},
		{Name: "media", Env: map[string]string{"MEDIA_DATA_DIR": "{{dataDir}}"}},
		{Name: "warehouse", Env: map[string]string{"WAREHOUSE_DATA_DIR": "{{dataDir}}"}},
		{Name: "store", Env: map[string]string{"STORE_DATA_DIR": "{{dataDir}}"}},
		{Name: "onboarding", Env: map[string]string{"ONBOARDING_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "kb.json", map[string]string{"hours": "Mon-Fri 9-5."})
		WriteJSONFile(t, dir, "tickets.json", []map[string]string{
			{"id": "t1", "question": "What are your hours?"},
		})
	},
	Phases: []Phase{{
		Name:    "Ops sweep — sample several iterations",
		Timeout: 240 * time.Second,
		Wait: func(t *testing.T, dir string, th *Thinker) bool {
			t.Logf("  ... iteration=%d", th.Iteration())
			return th.Iteration() >= 3
		},
	}},
	Timeout: 8 * time.Minute,
}

func TestScenario_TokenAB(t *testing.T) {
	if os.Getenv("RUN_TOKEN_AB") == "" {
		t.Skip("set RUN_TOKEN_AB=1 to run the token A/B measurement scenario")
	}
	s := tokenABScenario
	for i := range s.MCPServers {
		s.MCPServers[i].Command = BuildMCPBinary(t, "mcps/"+s.MCPServers[i].Name)
	}
	t.Logf("eager mode: %v | %d MCPs attached", os.Getenv("APTEVA_TOOL_SEARCH") == "off", len(s.MCPServers))
	RunScenario(t, s)
}

func TestScenario_TokenABBig(t *testing.T) {
	if os.Getenv("RUN_TOKEN_AB") == "" {
		t.Skip("set RUN_TOKEN_AB=1 to run the token A/B measurement scenario")
	}
	s := tokenABBigScenario
	for i := range s.MCPServers {
		s.MCPServers[i].Command = BuildMCPBinary(t, "mcps/"+s.MCPServers[i].Name)
	}
	t.Logf("eager mode: %v | %d MCPs attached", os.Getenv("APTEVA_TOOL_SEARCH") == "off", len(s.MCPServers))
	RunScenario(t, s)
}
