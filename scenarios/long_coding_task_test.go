package scenarios

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var longCodingTaskScenario = Scenario{
	Name: "LongCodingTask",
	MCPServers: []MCPServerConfig{
		// NOTE: codebase MCP reads CODEBASE_DIR, not CODEBASE_DATA_DIR — if we
		// passed the wrong name it falls back to "." (the test's cwd) and
		// silently operates on the apteva-core package directory instead.
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	Directive: `You are a coordinator. You have a dev worker thread under you that does all the coding.
Your job: spawn a dev-worker thread with codebase tools and give it a clear goal. Then monitor its
progress, respond to questions, and only intervene when it's stuck.

The dev worker should:
1. Read the existing files (main.go, main_test.go) to understand the task.
2. Implement the missing functionality in main.go.
3. Run tests via run_tests. If they fail, read the failure, fix the code, run again.
4. Keep iterating until tests pass, or until it has made genuine progress.

Do not do the coding yourself — delegate everything to the worker. Stay on normal pace as the
coordinator. The worker should be on fast pace to iterate quickly.`,
	DataSetup: func(t *testing.T, dir string) {
		// Skeleton main.go with stub functions that must be implemented.
		os.WriteFile(filepath.Join(dir, "main.go"), []byte(`package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// WordCount returns a map of lowercased words → occurrence counts for the
// given text. Words are split on whitespace, leading/trailing punctuation is
// stripped, and empty tokens are ignored.
//
// TODO: implement.
func WordCount(text string) map[string]int {
	return nil
}

// TopN returns the top-N words by count, sorted by count descending then
// alphabetically ascending for tie-breaking. Return at most n entries.
//
// TODO: implement.
func TopN(counts map[string]int, n int) []string {
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: wordcount <file> [n]")
		os.Exit(1)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}
	n := 10
	counts := WordCount(string(data))
	for _, w := range TopN(counts, n) {
		fmt.Printf("%d\t%s\n", counts[w], w)
	}
	// references to avoid unused-import errors while stubs are empty
	_ = sort.Strings
	_ = strings.ToLower
}
`), 0644)

		// Full test suite that must pass.
		os.WriteFile(filepath.Join(dir, "main_test.go"), []byte(`package main

import (
	"reflect"
	"testing"
)

func TestWordCount_Basic(t *testing.T) {
	got := WordCount("the quick brown fox the lazy dog the")
	want := map[string]int{"the": 3, "quick": 1, "brown": 1, "fox": 1, "lazy": 1, "dog": 1}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWordCount_CaseAndPunct(t *testing.T) {
	got := WordCount("Hello, world! HELLO world.")
	want := map[string]int{"hello": 2, "world": 2}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestWordCount_Empty(t *testing.T) {
	got := WordCount("")
	if len(got) != 0 {
		t.Errorf("expected empty, got %v", got)
	}
}

func TestTopN_SortedByCountThenAlpha(t *testing.T) {
	counts := map[string]int{"a": 3, "b": 3, "c": 2, "d": 1}
	got := TopN(counts, 3)
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestTopN_LimitsToN(t *testing.T) {
	counts := map[string]int{"a": 1, "b": 2, "c": 3}
	got := TopN(counts, 2)
	if len(got) != 2 {
		t.Errorf("expected 2 results, got %v", got)
	}
	if got[0] != "c" || got[1] != "b" {
		t.Errorf("expected [c b], got %v", got)
	}
}
`), 0644)

		// test.sh is what the codebase MCP's run_tests tool shells out to.
		os.WriteFile(filepath.Join(dir, "test.sh"), []byte("#!/bin/bash\ncd \"$(dirname \"$0\")\"\ngo test ./... 2>&1\n"), 0755)

		// go.mod so `go test` works without network deps.
		os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module wordcount\n\ngo 1.21\n"), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Bootstrap — coordinator spawns dev worker",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 2 {
					th.Inject(`[console] Task: main.go in your codebase has two stub functions (WordCount and TopN). main_test.go already contains the tests. Spawn a worker with id="dev-worker" using spawn(id="dev-worker", directive="Read main.go and main_test.go, implement WordCount and TopN so all tests pass, run tests after each change, iterate until green.", mcp="codebase", tools="send,done,pace"). Set its pace to fast. Do not code yourself.`)
				}
				// main's children are Depth=0 — just confirm at least one
				// thread is alive under main.
				if th.Threads().Count() >= 1 {
					for _, thr := range th.Threads().List() {
						t.Logf("  ✓ thread alive: id=%s depth=%d", thr.ID, thr.Depth)
					}
					return true
				}
				return false
			},
		},
		{
			Name:    "Active coding — worker reads, writes, tests",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Poll the main.go contents for evidence of real implementation
				// (the stub returns nil; any map literal or loop means progress)
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				src := string(data)
				stubbed := strings.Contains(src, "return nil") &&
					!strings.Contains(src, "make(map[string]int)") &&
					!strings.Contains(src, "for _,")
				writes := 0
				reads := 0
				tests := 0
				for _, thr := range th.Threads().List() {
					_ = thr // placeholder; actual tool counts come from audit log below
				}
				// The codebase MCP does not produce audit entries like some of
				// the others; we rely on file mutations + the observer counts
				// logged in the main runner. Record what we can see here.
				t.Logf("  ... main.go bytes=%d stubbed=%v writes=%d reads=%d tests=%d threads=%v",
					len(data), stubbed, writes, reads, tests, ThreadIDs(th))
				return !stubbed && len(data) > len([]byte("package main"))+400
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				t.Logf("  main.go size: %d bytes", len(data))
				t.Logf("  threads alive: %d", th.Threads().Count())
			},
		},
		{
			Name:    "Convergence — aim for green tests",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Directly shell out to `go test` ourselves to check status.
				// We don't mutate files — this is an independent read of the
				// worker's progress.
				cmd := exec.Command("go", "test", "./...")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				passing := err == nil
				t.Logf("  ... go test: passing=%v threads=%v", passing, ThreadIDs(th))
				if passing {
					t.Logf("  output: %s", strings.TrimSpace(string(out)))
				}
				return passing
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Best-effort: if tests did not pass, log the last failure so
				// the test report shows what the worker was stuck on.
				cmd := exec.Command("go", "test", "./...")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Logf("NOTE: tests did not reach green within budget — last go test output:")
					lines := strings.Split(strings.TrimSpace(string(out)), "\n")
					if len(lines) > 20 {
						lines = lines[len(lines)-20:]
					}
					for _, l := range lines {
						t.Logf("    %s", l)
					}
				} else {
					t.Logf("PASS: worker drove tests to green")
				}
				// Always log final main.go size for capacity measurement
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				t.Logf("final main.go: %d bytes, %d lines",
					len(data), strings.Count(string(data), "\n"))
			},
		},
		{
			Name:    "Soak — agent still alive",
			Timeout: 20 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// A dead agent is caught by the scenario hard timeout;
				// here we just let the soak window elapse. Sub-thread
				// count is informational — the agent may have finished
				// and torn its workers down, which is fine.
				t.Logf("  ... sub-threads alive=%d", th.Threads().Count())
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("soak: %d sub-threads alive", th.Threads().Count())
			},
		},
	},
	// Hard budget: 5 minutes total. Phases 2+3 together already consume 4
	// minutes of polling window, but the scenario will often finish phase 3
	// early if the worker drives tests green quickly.
	Timeout: 5 * time.Minute,
}

func TestScenario_LongCodingTask(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built codebase=%s", codebaseBin)

	s := longCodingTaskScenario
	s.MCPServers[0].Command = codebaseBin
	RunScenario(t, s)
}
