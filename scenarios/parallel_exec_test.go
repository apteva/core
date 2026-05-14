package scenarios

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var parallelExecScenario = Scenario{
	Name: "ParallelExec",
	Directive: `You have five tiny projects in this working directory, named
proj_a, proj_b, proj_c, proj_d, proj_e. Each project contains a test.sh
script at its root. Running the script produces a done.txt file inside the
project directory containing the project's name in uppercase.

Your job: run test.sh for every single project, then report "ALL DONE" to
the user. The scripts are slow (a couple of seconds each) so work
efficiently — you have a strict time budget.

Tools available to you: spawn, send, done, pace, exec. Use exec with the
"dir" argument to run the script inside the correct project directory,
like: exec command="./test.sh" dir="/full/path/proj_a".

When every project has a done.txt file, you are finished.`,
	MCPServers: nil,
	DataSetup: func(t *testing.T, dir string) {
		for _, name := range []string{"a", "b", "c", "d", "e"} {
			projDir := filepath.Join(dir, "proj_"+name)
			if err := os.MkdirAll(projDir, 0755); err != nil {
				t.Fatal(err)
			}
			// test.sh sleeps then writes a marker. The sleep is long enough
			// that running them one-by-one would miss the phase timeout
			// easily, but short enough that the whole scenario still
			// finishes quickly when the agent parallelises. The marker
			// content is used in Verify to make sure the right script ran
			// for the right project (rejects "one worker ran all five").
			script := fmt.Sprintf("#!/bin/bash\nsleep 2\necho '%s_OK' > done.txt\nexit 0\n",
				strings.ToUpper(name))
			if err := os.WriteFile(filepath.Join(projDir, "test.sh"), []byte(script), 0755); err != nil {
				t.Fatal(err)
			}
		}
	},
	Phases: []Phase{
		{
			Name:    "All 5 projects tested",
			Timeout: 3 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				done := 0
				for _, name := range []string{"a", "b", "c", "d", "e"} {
					if _, err := os.Stat(filepath.Join(dir, "proj_"+name, "done.txt")); err == nil {
						done++
					}
				}
				t.Logf("  ... done=%d/5", done)
				return done >= 5
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Each project's done.txt must contain its own uppercase
				// marker. This rejects any worker that accidentally wrote
				// the wrong name into the wrong project (cross-contamination
				// between spawned workers).
				for _, name := range []string{"a", "b", "c", "d", "e"} {
					data, err := os.ReadFile(filepath.Join(dir, "proj_"+name, "done.txt"))
					if err != nil {
						t.Errorf("proj_%s missing done.txt: %v", name, err)
						continue
					}
					want := strings.ToUpper(name) + "_OK"
					if !strings.Contains(string(data), want) {
						t.Errorf("proj_%s done.txt = %q, expected substring %q",
							name, strings.TrimSpace(string(data)), want)
					}
				}
			},
		},
	},
	Timeout: 5 * time.Minute,
	// simultaneously during the scenario. Running the five scripts
	// sequentially from main would keep threads.Count() at 0 and fail
	// the check. Anything from 3-5 parallel workers is fine.
}

func TestScenario_ParallelExec(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	RunScenario(t, parallelExecScenario)
}
