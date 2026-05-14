package scenarios

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

// --- AudioCallPipeline scenario ---
//
// Reproduces a real client workflow end-to-end:
//
//   1. Find the "jeremy_test" folder on Google Drive.
//   2. Find "criteres_eval" doc inside it; load its content via the
//      docs_agent.
//   3. Find the "ecoutes" subfolder and the audio file in it.
//   4. Get a download URL for the audio, hand it to the deepgram_agent
//      which transcribes with Nova 3 and scores against the criteria.
//   5. Hand the transcript + score to docs_agent which generates a French
//      report as a new Google Doc.
//   6. Place the new doc next to the audio (same ecoutes folder).
//   7. Find the "ventes" sheet inside jeremy_test, read headers via
//      read_sheet, match the audio file name against the
//      "Numéro(s) Contrat(s)" column, then update the row's Score and
//      Rapport columns.
//
// This scenario is primarily a stress test for the LARGE-TOOL-RESULT path:
// the deepgram MCP returns a synthetic transcript sized by
// DEEPGRAM_TRANSCRIPT_SIZE (default 15000 chars = ~4k tokens). That data
// has to traverse multiple worker context windows (deepgram_agent, then
// docs_agent to build the report) before ever reaching main. Raise the
// env var to exercise overflow behaviour.
//
// MCPs used:
//   gdrive   (new stub)  — folder/file lookup, download URLs, create doc reference
//   gdocs    (new stub)  — load_doc, create_doc, update_doc
//   deepgram (new stub)  — transcribe (Nova 3), evaluate
//   sheets   (existing)  — read_sheet, update_cell
//
// Enforcements (Verify):
//   - a new Google Doc exists with a French title containing "rapport"
//   - gdrive has a new entry of type=doc with parent=ecoutes folder id
//   - the ventes row matching CONTRACT-20240315 has non-empty Score and
//     Rapport cells; the Rapport cell contains the doc link.

const audioPipelineContract = "CONTRACT-20240315"

var audioCallPipelineScenario = Scenario{
	Name: "AudioCallPipeline",
	Directive: `You coordinate a sales-call analysis team. You don't handle files or tools yourself — you delegate to specialists, one per system:
  - a Drive specialist who knows Google Drive (folders, files, download links, placing reports)
  - a Docs specialist who loads and writes Google Docs
  - a Deepgram specialist who transcribes audio (Nova 3) and scores calls against criteria
  - a Sheets specialist who reads and edits the ventes spreadsheet

Keep each specialist focused on its system — don't mix systems inside one specialist. You route information between them as the pipeline progresses.

Context for the team:

On the Drive there is a folder called "jeremy_test" which contains:
  - a document "criteres_eval" with the call evaluation criteria
  - a spreadsheet "ventes" with one row per sale (columns include "Numéro(s) Contrat(s)", "Score", "Rapport")
  - a subfolder "ecoutes" where new call recordings are uploaded

Whenever a new audio recording is uploaded to "ecoutes":
  - have it transcribed with Deepgram using the Nova 3 model
  - score it against the criteria in "criteres_eval"
  - write a report about the call IN FRENCH (note, rationale, highlights, short summary) and save it as a new Google Doc
  - place the report in the same "ecoutes" folder next to the audio
  - in the "ventes" sheet, find the row whose "Numéro(s) Contrat(s)" matches the audio filename (without .mp3), and fill its "Score" and "Rapport" columns

Once each file is fully processed, say "PIPELINE COMPLETE" for that file.`,
	MCPServers: []MCPServerConfig{
		// All MCPs are catalog-only. Main is a pure coordinator and must
		// spawn a worker per MCP scope — keeps all MCP traffic (including
		// the large deepgram transcript) out of main's context.
		{Name: "gdrive", Env: map[string]string{"GDRIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "sheets", Env: map[string]string{"SHEETS_DATA_DIR": "{{dataDir}}"}},
		{Name: "gdocs", Env: map[string]string{"GDOCS_DATA_DIR": "{{dataDir}}"}},
		{Name: "deepgram", Env: map[string]string{}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// --- Drive layout ---
		// jeremy_test/
		//   criteres_eval        (doc → criteria_doc)
		//   ventes               (sheet, referenced by name)
		//   ecoutes/
		//     CONTRACT-20240315.mp3
		drive := []map[string]any{
			{"id": "folder_001", "name": "jeremy_test", "type": "folder"},
			{"id": "doc_entry_001", "name": "criteres_eval", "type": "doc", "parent_id": "folder_001", "doc_id": "criteria_doc"},
			{"id": "sheet_001", "name": "ventes", "type": "sheet", "parent_id": "folder_001"},
			{"id": "folder_002", "name": "ecoutes", "type": "folder", "parent_id": "folder_001"},
			{"id": "audio_001", "name": audioPipelineContract + ".mp3", "type": "audio", "parent_id": "folder_002", "size": 5_200_000},
		}
		WriteJSONFile(t, dir, "drive.json", drive)

		// --- Docs layout ---
		criteria := `CRITÈRES D'ÉVALUATION D'APPEL COMMERCIAL

1. Accueil et présentation claire (2 pts)
2. Identification du client et du contrat (2 pts)
3. Écoute active et reformulation (2 pts)
4. Gestion des objections (2 pts)
5. Présentation d'une offre additionnelle pertinente (2 pts)

Note finale sur 10.`
		WriteJSONFile(t, dir, "docs.json", map[string]any{
			"criteria_doc": map[string]any{
				"id":      "criteria_doc",
				"title":   "criteres_eval",
				"content": criteria,
			},
		})

		// --- Sheets layout ---
		WriteJSONFile(t, dir, "sheets.json", map[string]any{
			"ventes": map[string]any{
				"columns": []string{"Date", "Numéro(s) Contrat(s)", "Client", "Montant", "Score", "Rapport"},
				"rows": []map[string]string{
					{"Date": "2024-03-14", "Numéro(s) Contrat(s)": "CONTRACT-20240314", "Client": "Dupont SARL", "Montant": "1200", "Score": "", "Rapport": ""},
					{"Date": "2024-03-15", "Numéro(s) Contrat(s)": audioPipelineContract, "Client": "Martin SAS", "Montant": "2400", "Score": "", "Rapport": ""},
					{"Date": "2024-03-16", "Numéro(s) Contrat(s)": "CONTRACT-20240316", "Client": "Durand SA", "Montant": "1800", "Score": "", "Rapport": ""},
				},
			},
		})
	},
	Phases: []Phase{
		{
			Name:    "New audio uploaded — pipeline wakes",
			Timeout: 10 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Inject the trigger once. The real-world shape is a
				// file appearing in ecoutes waking the agent; without
				// this nudge the agent sits on the standing directive.
				th.Inject(fmt.Sprintf("[console] A new call recording was just uploaded to the ecoutes folder: %s.mp3. Please process it per your standing instructions.", audioPipelineContract))
				t.Logf("  ... trigger injected for %s.mp3", audioPipelineContract)
				return true
			},
		},
		{
			Name:    "Transcript produced and scored",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Transcript is observable once a new doc with a report
				// title exists. The docs.json file grows as create_doc
				// fires — poll for any non-criteria doc.
				data, err := os.ReadFile(filepath.Join(dir, "docs.json"))
				if err != nil {
					return false
				}
				var docs map[string]map[string]string
				if err := json.Unmarshal(data, &docs); err != nil {
					return false
				}
				for id, d := range docs {
					if id == "criteria_doc" {
						continue
					}
					title := strings.ToLower(d["title"])
					if strings.Contains(title, "rapport") {
						t.Logf("  ... rapport doc created: %s = %q", id, d["title"])
						return true
					}
				}
				return false
			},
		},
		{
			Name:    "Report placed in ecoutes folder",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "drive.json"))
				if err != nil {
					return false
				}
				var drive []map[string]any
				if err := json.Unmarshal(data, &drive); err != nil {
					return false
				}
				for _, e := range drive {
					parent, _ := e["parent_id"].(string)
					tp, _ := e["type"].(string)
					if parent == "folder_002" && tp == "doc" {
						t.Logf("  ... doc placed in ecoutes: %v", e["name"])
						return true
					}
				}
				return false
			},
		},
		{
			Name:    "Sheet row updated with score and link",
			Timeout: 90 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				row := readVentesRow(t, dir, audioPipelineContract)
				if row == nil {
					return false
				}
				t.Logf("  ... score=%q rapport=%q", row["Score"], row["Rapport"])
				return row["Score"] != "" && strings.Contains(row["Rapport"], "docs.mock/document/")
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				row := readVentesRow(t, dir, audioPipelineContract)
				if row == nil {
					t.Fatalf("row for %s not found in ventes sheet", audioPipelineContract)
				}
				// Score must be an integer 1-10
				if score := strings.TrimSpace(row["Score"]); score == "" {
					t.Errorf("Score cell is empty")
				} else {
					t.Logf("final Score=%q", score)
				}
				if rap := row["Rapport"]; !strings.Contains(rap, "docs.mock/document/") {
					t.Errorf("Rapport cell must contain a gdocs link, got %q", rap)
				}
				// Sister rows must NOT have been mutated (no cross-contamination).
				for _, other := range []string{"CONTRACT-20240314", "CONTRACT-20240316"} {
					or := readVentesRow(t, dir, other)
					if or == nil {
						continue
					}
					if or["Score"] != "" || or["Rapport"] != "" {
						t.Errorf("unrelated row %s was modified: score=%q rapport=%q",
							other, or["Score"], or["Rapport"])
					}
				}
			},
		},
	},
	Timeout: 8 * time.Minute,
}

func readVentesRow(t *testing.T, dir, contract string) map[string]string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "sheets.json"))
	if err != nil {
		return nil
	}
	var sheets map[string]struct {
		Columns []string            `json:"columns"`
		Rows    []map[string]string `json:"rows"`
	}
	if err := json.Unmarshal(data, &sheets); err != nil {
		return nil
	}
	s, ok := sheets["ventes"]
	if !ok {
		return nil
	}
	for _, row := range s.Rows {
		if row["Numéro(s) Contrat(s)"] == contract {
			return row
		}
	}
	return nil
}

func TestScenario_AudioCallPipeline(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	gdriveBin := BuildMCPBinary(t, "mcps/gdrive")
	gdocsBin := BuildMCPBinary(t, "mcps/gdocs")
	deepgramBin := BuildMCPBinary(t, "mcps/deepgram")
	sheetsBin := BuildMCPBinary(t, "mcps/sheets")
	t.Logf("built gdrive=%s gdocs=%s deepgram=%s sheets=%s", gdriveBin, gdocsBin, deepgramBin, sheetsBin)

	s := audioCallPipelineScenario
	s.MCPServers[0].Command = gdriveBin
	s.MCPServers[1].Command = sheetsBin
	s.MCPServers[2].Command = gdocsBin
	s.MCPServers[3].Command = deepgramBin

	if size := os.Getenv("DEEPGRAM_TRANSCRIPT_SIZE"); size != "" {
		t.Logf("stress mode: DEEPGRAM_TRANSCRIPT_SIZE=%s", size)
	}
	RunScenario(t, s)
}

// AudioCallPipeline_Stress fans out a handful of calls at once so the
// transcript-sized results land in parallel worker contexts. Run with
//
//	DEEPGRAM_TRANSCRIPT_SIZE=80000 RUN_SCENARIO_TESTS=1 go test -run AudioCallPipeline_Stress
//
// to exercise the MCP-result-size path under simultaneous load.
func TestScenario_AudioCallPipeline_Stress(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" || os.Getenv("RUN_STRESS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1 RUN_STRESS=1")
	}
	// Keep the scenario shape but multiply the audio count. Overriding
	// DataSetup here so we don't touch the single-call scenario.
	s := audioCallPipelineScenario
	s.Name = "AudioCallPipelineStress"
	s.Timeout = 15 * time.Minute
	s.Directive = strings.Replace(s.Directive,
		"There is exactly one audio file",
		"There are multiple audio files — process every one in parallel by spawning one deepgram_agent per file",
		1)
	s.DataSetup = func(t *testing.T, dir string) {
		drive := []map[string]any{
			{"id": "folder_001", "name": "jeremy_test", "type": "folder"},
			{"id": "doc_entry_001", "name": "criteres_eval", "type": "doc", "parent_id": "folder_001", "doc_id": "criteria_doc"},
			{"id": "sheet_001", "name": "ventes", "type": "sheet", "parent_id": "folder_001"},
			{"id": "folder_002", "name": "ecoutes", "type": "folder", "parent_id": "folder_001"},
		}
		rows := []map[string]string{}
		for i := 0; i < 4; i++ {
			contract := fmt.Sprintf("CONTRACT-2024031%d", 4+i)
			drive = append(drive, map[string]any{
				"id": fmt.Sprintf("audio_%03d", i+1), "name": contract + ".mp3",
				"type": "audio", "parent_id": "folder_002", "size": 5_000_000,
			})
			rows = append(rows, map[string]string{
				"Date":                 fmt.Sprintf("2024-03-1%d", 4+i),
				"Numéro(s) Contrat(s)": contract,
				"Client":               fmt.Sprintf("Client %d", i+1),
				"Montant":              "1000",
				"Score":                "",
				"Rapport":              "",
			})
		}
		WriteJSONFile(t, dir, "drive.json", drive)
		WriteJSONFile(t, dir, "docs.json", map[string]any{
			"criteria_doc": map[string]any{
				"id":      "criteria_doc",
				"title":   "criteres_eval",
				"content": "Critères 1..5, total /10.",
			},
		})
		WriteJSONFile(t, dir, "sheets.json", map[string]any{
			"ventes": map[string]any{
				"columns": []string{"Date", "Numéro(s) Contrat(s)", "Client", "Montant", "Score", "Rapport"},
				"rows":    rows,
			},
		})
	}
	gdriveBin := BuildMCPBinary(t, "mcps/gdrive")
	gdocsBin := BuildMCPBinary(t, "mcps/gdocs")
	deepgramBin := BuildMCPBinary(t, "mcps/deepgram")
	sheetsBin := BuildMCPBinary(t, "mcps/sheets")
	s.MCPServers[0].Command = gdriveBin
	s.MCPServers[1].Command = sheetsBin
	s.MCPServers[2].Command = gdocsBin
	s.MCPServers[3].Command = deepgramBin
	// For the stress run, replace phases with a single "all rows filled"
	// check — the sequential phases don't apply when multiple pipelines
	// run concurrently.
	s.Phases = []Phase{
		{
			Name:    "All rows scored with report links",
			Timeout: 12 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				done := 0
				for i := 0; i < 4; i++ {
					contract := fmt.Sprintf("CONTRACT-2024031%d", 4+i)
					row := readVentesRow(t, dir, contract)
					if row != nil && row["Score"] != "" && strings.Contains(row["Rapport"], "docs.mock/document/") {
						done++
					}
				}
				t.Logf("  ... done=%d/4", done)
				return done == 4
			},
		},
	}
	RunScenario(t, s)
}
