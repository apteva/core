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

var dailyProspectionScenario = Scenario{
	Name: "DailyProspection",
	Directive: `You run a daily SMB prospection batch. ICP: small marketing agencies in Lyon, France that don't already use AI tooling on their site.

Pipeline (run end-to-end, then report):
  1. DISCOVER — call places_search_text(query="marketing agency", location="Lyon") to get candidates. For each candidate, call places_get_place(id) to load the full record (phone, employee_band, address).
  2. ENRICH — for each candidate's website, call webscraper_extract_info(url) to load title/description/body and any extracted fields.
  3. QUALIFY — DROP a candidate if its scraped page mentions any AI vendor or tool (OpenAI, Anthropic, ChatGPT, GPT-4, Claude, Mistral, Gemini, Cohere, Copilot, intercom-fin). The point of the prospect list is "no AI yet" — keep only those without such mentions.
  4. CONTACT — for each kept candidate, derive a plausible decision-maker email of the form contact@<domain> (strip 'https://' / 'www.' from the website). Then verify_verify(email=...) — if the result is not "valid", DROP the candidate; do NOT save unverifiable contacts.
  5. UPSERT — for each verified candidate:
       a. crm_search_contacts(query=<domain>) to check if a row already exists.
       b. If a contact with that email OR matching company website is found, call crm_update_contact with the new info (status="enriched", set industry/location/description/employee_count from the scrape + place record).
       c. Otherwise crm_create_contact with name="Contact at <Company>", email, company, website, industry, employee_count, location, description, status="enriched".
  6. REPORT — write a one-line summary to stdout: total discovered, dropped (with reason), kept, created, updated.

Constraints:
  - Never call crm_create_contact without first calling crm_search_contacts to check for an existing row — that is the de-dup contract.
  - Never store an email that came back invalid from verify.
  - Process ALL candidates surfaced by search_text. Don't stop early.`,
	MCPServers: []MCPServerConfig{
		{Name: "places", Command: "", Env: map[string]string{"FAKE_PLACES_DATA_DIR": "{{dataDir}}"}},
		{Name: "webscraper", Command: "", Env: map[string]string{"SCRAPER_DATA_DIR": "{{dataDir}}"}},
		{Name: "verify", Command: "", Env: map[string]string{"FAKE_EMAIL_VERIFIER_DATA_DIR": "{{dataDir}}"}},
		{Name: "crm", Command: "", Env: map[string]string{"CRM_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// 5 candidates. lumiere-digital is the dedup target (already in CRM).
		// pixelforge mentions OpenAI in its site body → must be dropped.
		// noreply-shop has a contact form that yields a noreply alias → verify will fail.
		WriteJSONFile(t, dir, "places.json", []map[string]any{
			{
				"id": "p1", "name": "Atelier Pixel", "address": "12 rue de la République, 69001 Lyon",
				"city": "Lyon", "country": "France", "phone": "+33 4 00 00 00 01",
				"website": "https://atelier-pixel.fr", "rating": 4.6, "reviews_count": 23,
				"types": []string{"marketing agency", "design"}, "employee_band": "5-10",
			},
			{
				"id": "p2", "name": "Lumiere Digital", "address": "8 cours Lafayette, 69003 Lyon",
				"city": "Lyon", "country": "France", "phone": "+33 4 00 00 00 02",
				"website": "https://lumiere-digital.fr", "rating": 4.4, "reviews_count": 41,
				"types": []string{"marketing agency", "seo"}, "employee_band": "11-25",
			},
			{
				"id": "p3", "name": "PixelForge", "address": "30 rue Garibaldi, 69006 Lyon",
				"city": "Lyon", "country": "France", "phone": "+33 4 00 00 00 03",
				"website": "https://pixelforge.fr", "rating": 4.7, "reviews_count": 58,
				"types": []string{"marketing agency", "branding"}, "employee_band": "11-25",
			},
			{
				"id": "p4", "name": "Studio Bellecour", "address": "5 place Bellecour, 69002 Lyon",
				"city": "Lyon", "country": "France", "phone": "+33 4 00 00 00 04",
				"website": "https://studio-bellecour.fr", "rating": 4.8, "reviews_count": 17,
				"types": []string{"marketing agency", "creative"}, "employee_band": "1-5",
			},
			{
				"id": "p5", "name": "Noreply Shop", "address": "44 rue de Marseille, 69007 Lyon",
				"city": "Lyon", "country": "France", "phone": "+33 4 00 00 00 05",
				"website": "https://noreply-shop.fr", "rating": 4.1, "reviews_count": 9,
				"types": []string{"marketing agency"}, "employee_band": "5-10",
			},
		})

		// Site bodies. Tech-stack hints embedded in body so the agent has to
		// actually inspect the scrape to qualify. PixelForge mentions OpenAI
		// → must be dropped at step 3.
		WriteJSONFile(t, dir, "sites.json", map[string]map[string]string{
			"https://atelier-pixel.fr": {
				"title":       "Atelier Pixel — Branding & Web",
				"description": "Boutique marketing agency in Lyon focused on B2B branding and lightweight WordPress sites.",
				"body":        "Atelier Pixel is a 7-person studio. We use Figma, Notion, and HubSpot. Our clients are local SMBs in Lyon.",
				"industry":    "Marketing & Branding",
				"employees":   "5-10",
				"location":    "Lyon, France",
			},
			"https://lumiere-digital.fr": {
				"title":       "Lumiere Digital — SEO & Performance",
				"description": "Lyon-based SEO + paid acquisition agency for mid-market e-commerce.",
				"body":        "Lumiere Digital is a 15-person team. Tech stack: Google Analytics, Ahrefs, Klaviyo. No AI tooling — we like to keep things human.",
				"industry":    "SEO / Paid Acquisition",
				"employees":   "11-25",
				"location":    "Lyon, France",
			},
			"https://pixelforge.fr": {
				"title":       "PixelForge — AI-Native Branding Studio",
				"description": "We use ChatGPT and OpenAI workflows to deliver brand systems 5x faster.",
				"body":        "PixelForge is built on top of OpenAI GPT-4 and Anthropic Claude. Our pipelines are AI-first.",
				"industry":    "Marketing & Branding",
				"employees":   "11-25",
				"location":    "Lyon, France",
			},
			"https://studio-bellecour.fr": {
				"title":       "Studio Bellecour — Creative Marketing",
				"description": "Three-person creative shop on Place Bellecour. Identity, packaging, social.",
				"body":        "Studio Bellecour is a small creative agency. Tools we love: Adobe CC, Webflow, Mailchimp.",
				"industry":    "Marketing & Creative",
				"employees":   "1-5",
				"location":    "Lyon, France",
			},
			"https://noreply-shop.fr": {
				"title":       "Noreply Shop — Outbound Marketing",
				"description": "Direct mail and lead-gen agency for Lyon-area SMBs.",
				"body":        "Noreply Shop runs cold outbound for local clients. Stack: Mailgun, Pipedrive, Calendly. No AI.",
				"industry":    "Outbound Marketing",
				"employees":   "5-10",
				"location":    "Lyon, France",
			},
		})

		// Whitelist for fake_email_verifier. atelier-pixel + lumiere-digital
		// + studio-bellecour are deliverable. PixelForge is in here too but
		// the candidate gets dropped at step 3 before we ever verify it —
		// the assertion on contacts.json catches that. noreply-shop's
		// contact@ alias is NOT in the list, so it fails verify and the
		// agent must skip the upsert.
		WriteJSONFile(t, dir, "valid_emails.json", []string{
			"contact@atelier-pixel.fr",
			"contact@lumiere-digital.fr",
			"contact@pixelforge.fr",
			"contact@studio-bellecour.fr",
		})

		// CRM seeded with one row → forces the agent through the search +
		// update branch for lumiere-digital instead of creating a duplicate.
		WriteJSONFile(t, dir, "contacts.json", []map[string]string{
			{
				"id":             "c1",
				"name":           "Existing Contact at Lumiere Digital",
				"email":          "contact@lumiere-digital.fr",
				"company":        "Lumiere Digital",
				"website":        "https://lumiere-digital.fr",
				"industry":       "",
				"employee_count": "",
				"location":       "",
				"description":    "",
				"status":         "new",
				"created_at":     "2026-04-20T09:00:00Z",
				"enriched_at":    "",
			},
		})
	},
	Phases: []Phase{
		{
			// First Wait call doubles as the kickoff: inject a console
			// event that triggers the pipeline. Without this, the agent
			// can run autonomously but does so unevenly — sometimes
			// burning ~30k output tokens in one error-recovery iteration,
			// sometimes skipping enrichment follow-ups for created rows.
			// In production this scenario would be triggered by a cron
			// firing the same kind of event into the running agent, so
			// modelling it that way here matches reality and keeps the
			// run deterministic + cheap (~$0.02 vs ~$0.13 unsupervised).
			Name:    "Discovery — places searched + details fetched",
			Timeout: 180 * time.Second,
			Wait: (func() func(*testing.T, string, *Thinker) bool {
				kicked := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !kicked {
						th.InjectConsole("Run today's prospection batch now.")
						kicked = true
					}
					data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
					if err != nil {
						return false
					}
					return strings.Contains(string(data), `"tool":"search_text"`)
				}
			})(),
		},
		{
			Name:    "Qualification — sites scraped",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if err != nil {
					return false
				}
				// At least 4 scrapes — the agent should attempt every
				// candidate's website. We allow 4 (not 5) because some
				// agents short-circuit after deciding a candidate is
				// out-of-scope from the place record alone.
				return strings.Count(string(data), `"tool":"extract_info"`)+
					strings.Count(string(data), `"tool":"fetch_page"`) >= 4
			},
		},
		{
			Name:    "Verification — emails checked",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if err != nil {
					return false
				}
				return strings.Count(string(data), `"tool":"verify"`) >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Agent must not have verified PixelForge — it's already
				// disqualified by step 3 and verifying it would mean
				// it's still a live candidate downstream.
				data, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if strings.Contains(string(data), "pixelforge.fr") &&
					strings.Contains(string(data), `"tool":"verify"`) {
					// Allowed only if the agent explicitly went all-the-way
					// to verify before disqualifying — log it but don't
					// fail. Real fail mode is in the contacts.json check.
					t.Logf("note: agent verified pixelforge.fr — wasted call but not a correctness bug")
				}
			},
		},
		{
			Name:    "CRM upsert — exactly the right rows land",
			Timeout: 240 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "contacts.json"))
				if err != nil {
					return false
				}
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)
				// Final state should be exactly 3 contacts:
				//   c1 (lumiere-digital, updated — dedup hit the seeded row)
				//   atelier-pixel    (created)
				//   studio-bellecour (created)
				//   pixelforge       NOT present (disqualified — AI tooling)
				//   noreply-shop     NOT present (disqualified — verify failed)
				return len(contacts) == 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "contacts.json"))
				var contacts []map[string]string
				json.Unmarshal(data, &contacts)

				if len(contacts) != 3 {
					t.Errorf("expected 3 contacts (1 pre-existing updated + 2 new), got %d", len(contacts))
				}

				byDomain := map[string]map[string]string{}
				for _, c := range contacts {
					site := strings.ToLower(c["website"])
					email := strings.ToLower(c["email"])
					switch {
					case strings.Contains(site, "atelier-pixel") || strings.Contains(email, "atelier-pixel"):
						byDomain["atelier-pixel"] = c
					case strings.Contains(site, "lumiere-digital") || strings.Contains(email, "lumiere-digital"):
						byDomain["lumiere-digital"] = c
					case strings.Contains(site, "studio-bellecour") || strings.Contains(email, "studio-bellecour"):
						byDomain["studio-bellecour"] = c
					case strings.Contains(site, "pixelforge") || strings.Contains(email, "pixelforge"):
						byDomain["pixelforge"] = c
					case strings.Contains(site, "noreply-shop") || strings.Contains(email, "noreply-shop"):
						byDomain["noreply-shop"] = c
					}
				}

				// Disqualified: PixelForge (uses OpenAI) — MUST NOT be present.
				if _, ok := byDomain["pixelforge"]; ok {
					t.Errorf("pixelforge.fr was inserted into the CRM but its website mentions OpenAI/Anthropic — agent failed the AI-tooling disqualification rule")
				}
				// Unverifiable: noreply-shop email comes back invalid — MUST NOT be present.
				if _, ok := byDomain["noreply-shop"]; ok {
					t.Errorf("noreply-shop.fr was inserted into the CRM but its email failed verification — agent ignored verifier output")
				}

				// Dedup: lumiere-digital must STILL be id=c1 (the seeded row)
				// — not a fresh row, and now enriched.
				if c, ok := byDomain["lumiere-digital"]; ok {
					if c["id"] != "c1" {
						t.Errorf("lumiere-digital landed as a NEW contact (id=%q) — agent created a duplicate instead of updating c1", c["id"])
					}
					if c["industry"] == "" || c["location"] == "" {
						t.Errorf("lumiere-digital row was not enriched after update: industry=%q location=%q", c["industry"], c["location"])
					}
					if c["status"] != "enriched" {
						t.Errorf("lumiere-digital status=%q, expected 'enriched'", c["status"])
					}
				} else {
					t.Errorf("lumiere-digital row missing from CRM after run")
				}

				// New rows: atelier-pixel + studio-bellecour must both exist + be enriched.
				for _, key := range []string{"atelier-pixel", "studio-bellecour"} {
					c, ok := byDomain[key]
					if !ok {
						t.Errorf("%s missing from CRM — agent dropped a qualified lead", key)
						continue
					}
					if c["industry"] == "" || c["location"] == "" || c["description"] == "" {
						t.Errorf("%s was inserted unenriched: industry=%q location=%q description=%q",
							key, c["industry"], c["location"], c["description"])
					}
				}

				// CRM-side audit: agent must have called search_contacts at
				// least once before writing — that's the upsert contract.
				auditData, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if !strings.Contains(string(auditData), `"tool":"search_contacts"`) {
					t.Error("agent never called crm_search_contacts — upsert dedup contract violated")
				}
			},
		},
	},
	Timeout: 8 * time.Minute,
}

func TestScenario_DailyProspection(t *testing.T) {
	placesBin := BuildMCPBinary(t, "mcps/fake_places")
	scraperBin := BuildMCPBinary(t, "mcps/webscraper")
	verifyBin := BuildMCPBinary(t, "mcps/fake_email_verifier")
	crmBin := BuildMCPBinary(t, "mcps/crm")
	t.Logf("built fake_places=%s webscraper=%s fake_email_verifier=%s crm=%s",
		placesBin, scraperBin, verifyBin, crmBin)

	s := dailyProspectionScenario
	s.MCPServers[0].Command = placesBin
	s.MCPServers[1].Command = scraperBin
	s.MCPServers[2].Command = verifyBin
	s.MCPServers[3].Command = crmBin
	RunScenario(t, s)
}
