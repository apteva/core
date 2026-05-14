package scenarios

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var websiteBuildScenario = Scenario{
	Name: "WebsiteBuild",
	Directive: `You are building and deploying a React landing page for NovaPay.

Read the design brief first, then build a complete React application with Bun as the bundler.

Spawn 3 threads:
1. "architect" — reads the design brief and assets, plans the component structure, creates the project scaffold (package.json with react/react-dom deps, src/index.tsx entry point, src/App.tsx main component, and a basic index.html). Reports the plan to main when done.
   Tools: brief_get_brief, brief_get_assets, codebase_write_file, codebase_list_files, send, done
2. "builder" — implements each React component based on the brief. Creates Hero, Features, Pricing, and Footer components as separate .tsx files in src/. Includes inline CSS or a styles.css file. Runs the build check to verify all files are valid. Fixes any errors. Reports done when build passes.
   Tools: brief_get_brief, codebase_read_file, codebase_write_file, codebase_list_files, codebase_run_tests, send, done
3. "deployer" — creates a site on the hosting platform, deploys the app when the build is ready, and confirms it's live with the URL.
   Tools: hosting_create_site, hosting_deploy, hosting_get_status, hosting_get_url, hosting_list_sites, send, done

Workflow:
- First, tell architect to read the brief and scaffold the project.
- When architect reports done, tell builder to implement all sections from the brief.
- Builder should create: Hero.tsx, Features.tsx, Pricing.tsx, Footer.tsx (at minimum), import them in App.tsx, and run the build check.
- When builder confirms the build passes, tell deployer to create a site called "novapay-landing" and deploy.
- Deployer confirms the live URL.

IMPORTANT: All files go in the "app/" directory. package.json must include "react" and "react-dom" as dependencies. Every .tsx file must have an export.`,
	MCPServers: []MCPServerConfig{
		{Name: "brief", Command: "", Env: map[string]string{"BRIEF_DATA_DIR": "{{dataDir}}"}},
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
		{Name: "hosting", Command: "", Env: map[string]string{"HOSTING_DATA_DIR": "{{dataDir}}", "CODEBASE_DIR": "{{dataDir}}"}},
	},
	DataSetup: seedWebsiteBrief,
	Phases: []Phase{
		{
			Name:    "Scaffold — package.json + App.tsx created",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if _, err := os.Stat(filepath.Join(dir, "app", "package.json")); err != nil {
					return false
				}
				if _, err := os.Stat(filepath.Join(dir, "app", "src", "App.tsx")); err != nil {
					if _, err2 := os.Stat(filepath.Join(dir, "app", "src", "App.jsx")); err2 != nil {
						return false
					}
				}
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify package.json is valid and has react
				data, _ := os.ReadFile(filepath.Join(dir, "app", "package.json"))
				var pkg map[string]any
				if err := json.Unmarshal(data, &pkg); err != nil {
					t.Errorf("package.json is not valid JSON: %v", err)
				}
				deps, _ := pkg["dependencies"].(map[string]any)
				if deps == nil {
					t.Error("package.json missing dependencies")
				} else if deps["react"] == nil {
					t.Error("package.json missing react dependency")
				}
			},
		},
		{
			Name:    "Components — 4+ tsx/jsx files with exports",
			Timeout: 240 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Count tsx/jsx files recursively under app/src/
				count := 0
				filepath.Walk(filepath.Join(dir, "app", "src"), func(path string, info os.FileInfo, err error) error {
					if err != nil || info.IsDir() {
						return nil
					}
					if strings.HasSuffix(info.Name(), ".tsx") || strings.HasSuffix(info.Name(), ".jsx") {
						count++
					}
					return nil
				})
				return count >= 4
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Log component files (exports checked in build phase, builder will fix missing ones)
				filepath.Walk(filepath.Join(dir, "app", "src"), func(path string, info os.FileInfo, err error) error {
					if err != nil || info.IsDir() {
						return nil
					}
					if strings.HasSuffix(info.Name(), ".tsx") || strings.HasSuffix(info.Name(), ".jsx") {
						data, _ := os.ReadFile(path)
						hasExport := strings.Contains(string(data), "export")
						t.Logf("component %s (%d bytes, export=%v)", info.Name(), len(data), hasExport)
					}
					return nil
				})
			},
		},
		{
			Name:    "Build — test.sh passes",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				cmd := exec.Command("bash", "test.sh")
				cmd.Dir = dir
				return cmd.Run() == nil
			},
		},
		{
			Name:    "Deploy — site is live",
			Timeout: 180 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "sites.json"))
				if err != nil {
					return false
				}
				return strings.Contains(string(data), `"live"`)
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify site is live with URL
				data, _ := os.ReadFile(filepath.Join(dir, "sites.json"))
				var sites []map[string]string
				json.Unmarshal(data, &sites)
				if len(sites) == 0 {
					t.Error("no sites created")
					return
				}
				site := sites[0]
				if site["status"] != "live" {
					t.Errorf("site status=%s, expected live", site["status"])
				}
				if site["url"] == "" {
					t.Error("site has no URL")
				}
				t.Logf("Site deployed: %s → %s", site["name"], site["url"])

				// Verify deployment record
				dData, _ := os.ReadFile(filepath.Join(dir, "deployments.json"))
				var deploys []map[string]any
				json.Unmarshal(dData, &deploys)
				if len(deploys) == 0 {
					t.Error("no deployment records")
				} else {
					files, _ := deploys[0]["files"].([]any)
					t.Logf("Deployed %d files", len(files))
				}
			},
		},
	},
	Timeout: 10 * time.Minute,
}

func TestScenario_WebsiteBuild(t *testing.T) {
	briefBin := BuildMCPBinary(t, "mcps/brief")
	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	hostingBin := BuildMCPBinary(t, "mcps/hosting")
	t.Logf("built brief=%s codebase=%s hosting=%s", briefBin, codebaseBin, hostingBin)

	s := websiteBuildScenario
	s.MCPServers[0].Command = briefBin
	s.MCPServers[1].Command = codebaseBin
	s.MCPServers[2].Command = hostingBin
	RunScenario(t, s)
}
