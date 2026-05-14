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

var autonomousSheetEnrichmentScenario = Scenario{
	Name: "AutonomousSheetEnrichment",
	Directive: `You are enriching a spreadsheet called "Companies".

Every row has a "name" and a "website" column. The "summary" and "industry"
columns are empty — your job is to fill them in for every single row.

For each row:
- Use webscraper_extract_info on the website to get the company's description and industry.
- Write the description into the "summary" column and the industry into the "industry" column
  using sheets_update_cell.

You have a strict time budget. Process every row as quickly as you can. When every row
has a non-empty summary and industry, you are done.`,
	MCPServers: []MCPServerConfig{
		{Name: "sheets", Command: "", Env: map[string]string{"SHEETS_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Command: "", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// 10 companies — enough that a sensible agent will parallelise rather
		// than walk them one at a time. Each site has static metadata the
		// webscraper MCP returns verbatim.
		WriteJSONFile(t, dir, "sheets.json", map[string]*struct {
			Columns []string            `json:"columns"`
			Rows    []map[string]string `json:"rows"`
		}{
			"Companies": {
				Columns: []string{"name", "website", "summary", "industry"},
				Rows: []map[string]string{
					{"name": "Acme Corp", "website": "https://acmecorp.com", "summary": "", "industry": ""},
					{"name": "Globex", "website": "https://globex.io", "summary": "", "industry": ""},
					{"name": "Initech", "website": "https://initech.com", "summary": "", "industry": ""},
					{"name": "Umbrella Labs", "website": "https://umbrella.dev", "summary": "", "industry": ""},
					{"name": "Northwind", "website": "https://northwind.co", "summary": "", "industry": ""},
					{"name": "Soylent", "website": "https://soylent.foo", "summary": "", "industry": ""},
					{"name": "Stark Industries", "website": "https://stark.industries", "summary": "", "industry": ""},
					{"name": "Wayne Enterprises", "website": "https://wayne.enterprises", "summary": "", "industry": ""},
					{"name": "Cyberdyne", "website": "https://cyberdyne.systems", "summary": "", "industry": ""},
					{"name": "Tyrell", "website": "https://tyrell.corp", "summary": "", "industry": ""},
				},
			},
		})

		WriteJSONFile(t, dir, "sites.json", map[string]*struct {
			Title       string `json:"title"`
			Description string `json:"description"`
			Body        string `json:"body"`
			Industry    string `json:"industry"`
			Employees   string `json:"employees"`
			Location    string `json:"location"`
			Founded     string `json:"founded"`
		}{
			"https://acmecorp.com": {
				Title: "Acme Corp", Industry: "Industrial Automation",
				Description: "Leading provider of industrial automation and robotics systems for manufacturing.",
				Body:        "Acme Corp builds next-generation automation platforms.",
				Employees:   "500-1000", Location: "San Francisco, CA", Founded: "2015",
			},
			"https://globex.io": {
				Title: "Globex", Industry: "SaaS / Analytics",
				Description: "AI-powered analytics platform for data-driven decisions.",
				Body:        "Globex unifies analytics with machine learning.",
				Employees:   "50-100", Location: "Austin, TX", Founded: "2020",
			},
			"https://initech.com": {
				Title: "Initech", Industry: "IT Consulting",
				Description: "Enterprise software consulting and digital transformation.",
				Body:        "Initech helps mid-market companies modernize their stack.",
				Employees:   "200-500", Location: "Chicago, IL", Founded: "2012",
			},
			"https://umbrella.dev": {
				Title: "Umbrella Labs", Industry: "Biotech / Life Sciences",
				Description: "AI-powered molecular simulation for drug discovery.",
				Body:        "Umbrella Labs compresses drug discovery timelines dramatically.",
				Employees:   "100-200", Location: "Boston, MA", Founded: "2019",
			},
			"https://northwind.co": {
				Title: "Northwind", Industry: "Supply Chain / Logistics",
				Description: "Sustainable supply chain optimization and visibility.",
				Body:        "Northwind tracks carbon footprints across global supply chains.",
				Employees:   "150-300", Location: "Seattle, WA", Founded: "2017",
			},
			"https://soylent.foo": {
				Title: "Soylent", Industry: "Food Technology",
				Description: "Nutritionally complete meal replacement drinks.",
				Body:        "Soylent ships engineered meals to subscribers worldwide.",
				Employees:   "80-120", Location: "Los Angeles, CA", Founded: "2013",
			},
			"https://stark.industries": {
				Title: "Stark Industries", Industry: "Aerospace & Defense",
				Description: "Advanced aerospace, defense, and clean energy systems.",
				Body:        "Stark Industries builds flagship defense and energy platforms.",
				Employees:   "10000+", Location: "Los Angeles, CA", Founded: "1939",
			},
			"https://wayne.enterprises": {
				Title: "Wayne Enterprises", Industry: "Diversified Conglomerate",
				Description: "Diversified conglomerate with investments across sectors.",
				Body:        "Wayne Enterprises operates in transportation, biotech, and R&D.",
				Employees:   "20000+", Location: "Gotham, NJ", Founded: "1890",
			},
			"https://cyberdyne.systems": {
				Title: "Cyberdyne Systems", Industry: "AI & Robotics",
				Description: "Advanced AI and autonomous systems research.",
				Body:        "Cyberdyne develops next-generation autonomous systems.",
				Employees:   "300-500", Location: "Sunnyvale, CA", Founded: "1984",
			},
			"https://tyrell.corp": {
				Title: "Tyrell Corp", Industry: "Genetic Engineering",
				Description: "Bioengineered humanoid systems and synthetic biology.",
				Body:        "Tyrell Corp builds replicants with human-level capabilities.",
				Employees:   "5000+", Location: "Los Angeles, CA", Founded: "2019",
			},
		})
	},
	Phases: []Phase{
		{
			Name:    "All rows enriched — every row has summary + industry filled in",
			Timeout: 5 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "sheets.json"))
				if err != nil {
					return false
				}
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				raw, ok := sheets["Companies"]
				if !ok {
					return false
				}
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(raw, &sheet)
				if len(sheet.Rows) == 0 {
					return false
				}
				for _, row := range sheet.Rows {
					if row["summary"] == "" || row["industry"] == "" {
						return false
					}
				}
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "sheets.json"))
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(sheets["Companies"], &sheet)
				if len(sheet.Rows) != 10 {
					t.Errorf("expected 10 rows, got %d", len(sheet.Rows))
				}
				// The webscraper MCP returns real metadata only for the 10
				// seeded URLs. If the agent hallucinates a different URL it
				// errors out and tends to write fallback text like "N/A" or
				// "Unable to fetch ..." to satisfy the directive. Reject
				// those — we want to see the actual seeded values.
				badSubstrings := []string{"N/A", "Unable", "unable", "could not fetch", "Unknown", "unknown"}
				for i, row := range sheet.Rows {
					for _, col := range []string{"summary", "industry"} {
						v := row[col]
						if v == "" {
							t.Errorf("row %d (%s) missing %s", i, row["name"], col)
							continue
						}
						for _, bad := range badSubstrings {
							if strings.Contains(v, bad) {
								t.Errorf("row %d (%s) %s=%q looks like a fallback, not real scrape data", i, row["name"], col, v)
								break
							}
						}
					}
				}
				// Verify the webscraper was actually called per row (the
				// agent might have made up values otherwise).
				entries := ReadAuditEntries(dir)
				scrapes := CountTool(entries, "extract_info") + CountTool(entries, "fetch_page")
				if scrapes < 10 {
					t.Errorf("expected ≥10 webscraper calls (one per row), got %d", scrapes)
				}
			},
		},
	},
	Timeout: 7 * time.Minute,
	// This scenario used to assert the agent spawned ≥2 concurrent
	// threads. It no longer does: whether the agent parallelises rows
	// across workers or walks them inline on main is its own call —
	// the phases below assert on the enriched-sheet OUTCOME, not the
	// mechanism. Spawning is a tool, not a requirement.
}

func TestScenario_AutonomousSheetEnrichment(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	sheetsBin := BuildMCPBinary(t, "mcps/sheets")
	scraperBin := BuildMCPBinary(t, "mcps/webscraper")
	t.Logf("built sheets=%s webscraper=%s", sheetsBin, scraperBin)

	s := autonomousSheetEnrichmentScenario
	s.MCPServers[0].Command = sheetsBin
	s.MCPServers[1].Command = scraperBin
	RunScenario(t, s)
}
