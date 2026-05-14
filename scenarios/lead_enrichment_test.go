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

var leadEnrichmentScenario = Scenario{
	Name: "LeadEnrichment",
	Directive: `You manage a lead enrichment pipeline.

Your job:
1. Read all leads from the "Lead Pipeline" spreadsheet using the sheets tools.
2. For each lead with status "new":
   a. Create a contact in the CRM (crm_create_contact) with name, email, company, website.
   b. Scrape the lead's website (webscraper_extract_info) to get company details.
   c. Update the CRM contact (crm_update_contact) with the enrichment data (industry, employee_count, location, description) and set status to "enriched".
   d. Update the spreadsheet row (sheets_update_cell) to set the status column to "enriched".
3. After all leads are processed, you are done.

Process all leads. Do not skip any.`,
	MCPServers: []MCPServerConfig{
		{Name: "sheets", Command: "", Env: map[string]string{"SHEETS_DATA_DIR": "{{dataDir}}"}},
		{Name: "crm", Command: "", Env: map[string]string{"CRM_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Command: "", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed the spreadsheet with 5 leads
		WriteJSONFile(t, dir, "sheets.json", map[string]*struct {
			Columns []string            `json:"columns"`
			Rows    []map[string]string `json:"rows"`
		}{
			"Lead Pipeline": {
				Columns: []string{"name", "email", "website", "company", "status"},
				Rows: []map[string]string{
					{"name": "Alice Smith", "email": "alice@acmecorp.com", "website": "https://acmecorp.com", "company": "Acme Corp", "status": "new"},
					{"name": "Bob Chen", "email": "bob@globex.io", "website": "https://globex.io", "company": "Globex", "status": "new"},
					{"name": "Carol Davis", "email": "carol@initech.com", "website": "https://initech.com", "company": "Initech", "status": "new"},
					{"name": "Dan Wilson", "email": "dan@umbrella.dev", "website": "https://umbrella.dev", "company": "Umbrella Labs", "status": "new"},
					{"name": "Eve Park", "email": "eve@northwind.co", "website": "https://northwind.co", "company": "Northwind", "status": "new"},
				},
			},
		})

		// Seed website data for the scraper
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
				Title:       "Acme Corp — Industrial Solutions",
				Description: "Leading provider of industrial automation and robotics systems for manufacturing.",
				Body:        "Acme Corp builds next-generation automation platforms for factories worldwide. Founded in 2015, we serve over 200 enterprise customers across North America and Europe.",
				Industry:    "Industrial Automation",
				Employees:   "500-1000",
				Location:    "San Francisco, CA",
				Founded:     "2015",
			},
			"https://globex.io": {
				Title:       "Globex — AI-Powered Analytics",
				Description: "We help businesses make data-driven decisions with real-time AI analytics.",
				Body:        "Globex provides a unified analytics platform powered by machine learning. Our team of 80 engineers and data scientists builds tools used by Fortune 500 companies.",
				Industry:    "SaaS / Analytics",
				Employees:   "50-100",
				Location:    "Austin, TX",
				Founded:     "2020",
			},
			"https://initech.com": {
				Title:       "Initech — Enterprise Software Consulting",
				Description: "Initech delivers custom enterprise software solutions and digital transformation services.",
				Body:        "For over a decade, Initech has helped mid-market companies modernize their technology stack. We specialize in ERP integration, cloud migration, and custom application development.",
				Industry:    "IT Consulting",
				Employees:   "200-500",
				Location:    "Chicago, IL",
				Founded:     "2012",
			},
			"https://umbrella.dev": {
				Title:       "Umbrella Labs — Biotech Research Platform",
				Description: "Umbrella Labs accelerates drug discovery with AI-powered molecular simulation.",
				Body:        "Our computational biology platform reduces drug discovery timelines from years to months. Backed by $50M in Series B funding, we partner with 15 pharmaceutical companies.",
				Industry:    "Biotech / Life Sciences",
				Employees:   "100-200",
				Location:    "Boston, MA",
				Founded:     "2019",
			},
			"https://northwind.co": {
				Title:       "Northwind — Sustainable Supply Chain",
				Description: "Northwind optimizes global supply chains for sustainability and efficiency.",
				Body:        "We provide end-to-end supply chain visibility with carbon footprint tracking. Our platform is used by 300+ retailers and manufacturers committed to sustainable operations.",
				Industry:    "Supply Chain / Logistics",
				Employees:   "150-300",
				Location:    "Seattle, WA",
				Founded:     "2017",
			},
		})

		// Start with empty CRM
		WriteJSONFile(t, dir, "contacts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Sheet read — agent discovers 5 leads",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Wait for the agent to have called read_sheet (check audit)
				data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), "read_sheet")
			},
		},
		{
			Name:    "CRM creation — all 5 leads added",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "contacts.json"))
				if err != nil {
					return false
				}
				var contacts []json.RawMessage
				json.Unmarshal(data, &contacts)
				return len(contacts) >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "contacts.json"))
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				if len(contacts) < 5 {
					t.Errorf("expected 5 contacts, got %d", len(contacts))
				}
				// Verify all have emails
				emails := map[string]bool{}
				for _, c := range contacts {
					emails[c["email"]] = true
				}
				for _, expected := range []string{"alice@acmecorp.com", "bob@globex.io", "carol@initech.com", "dan@umbrella.dev", "eve@northwind.co"} {
					if !emails[expected] {
						t.Errorf("missing contact with email %s", expected)
					}
				}
			},
		},
		{
			Name:    "Enrichment — CRM contacts updated with company info",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "contacts.json"))
				if err != nil {
					return false
				}
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				enriched := 0
				for _, c := range contacts {
					if c["industry"] != "" && c["location"] != "" {
						enriched++
					}
				}
				return enriched >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "contacts.json"))
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				for _, c := range contacts {
					if c["industry"] == "" {
						t.Errorf("contact %s (%s) missing industry", c["id"], c["email"])
					}
					if c["location"] == "" {
						t.Errorf("contact %s (%s) missing location", c["id"], c["email"])
					}
				}
				// Verify scraper was actually called (check audit)
				auditData, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				auditStr := string(auditData)
				if !strings.Contains(auditStr, "extract_info") && !strings.Contains(auditStr, "fetch_page") {
					t.Error("expected webscraper tools to be called")
				}
			},
		},
		{
			Name:    "Sheet update — all leads marked enriched",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "sheets.json"))
				if err != nil {
					return false
				}
				// Count rows with status=enriched
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				sheetData, ok := sheets["Lead Pipeline"]
				if !ok {
					return false
				}
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(sheetData, &sheet)
				enriched := 0
				for _, row := range sheet.Rows {
					if row["status"] == "enriched" {
						enriched++
					}
				}
				return enriched >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "sheets.json"))
				var sheets map[string]json.RawMessage
				json.Unmarshal(data, &sheets)
				var sheet struct {
					Rows []map[string]string `json:"rows"`
				}
				json.Unmarshal(sheets["Lead Pipeline"], &sheet)
				for i, row := range sheet.Rows {
					if row["status"] != "enriched" {
						t.Errorf("row %d (%s) status=%q, expected enriched", i, row["name"], row["status"])
					}
				}
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_LeadEnrichment(t *testing.T) {
	sheetsBin := BuildMCPBinary(t, "mcps/sheets")
	crmBin := BuildMCPBinary(t, "mcps/crm")
	scraperBin := BuildMCPBinary(t, "mcps/webscraper")
	t.Logf("built sheets=%s crm=%s webscraper=%s", sheetsBin, crmBin, scraperBin)

	s := leadEnrichmentScenario
	s.MCPServers[0].Command = sheetsBin
	s.MCPServers[1].Command = crmBin
	s.MCPServers[2].Command = scraperBin
	RunScenario(t, s)
}
