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

var longCodingTaskHardScenario = Scenario{
	Name: "LongCodingTaskHard",
	MCPServers: []MCPServerConfig{
		{Name: "codebase", Command: "", Env: map[string]string{"CODEBASE_DIR": "{{dataDir}}"}},
	},
	Directive: `You are a coordinator. You have a dev worker under you that does all the coding.
Your job: spawn the worker, hand it the task, and stay out of its way. Do not write code yourself —
delegate everything. When the worker reports progress, acknowledge briefly. Only intervene if it
gets stuck for many iterations without making progress.

The worker should:
1. Read main.go and main_test.go to understand the task.
2. Implement the four stub functions (Tokenize, Parse, Eval, Evaluate) in main.go.
3. Run tests after each change via codebase_run_tests.
4. When tests fail, read the failure carefully, fix the specific bug, and re-run.
5. Iterate until all 12 tests pass. Do not give up — edge cases are the hard part.`,
	DataSetup: func(t *testing.T, dir string) {
		os.WriteFile(filepath.Join(dir, "main.go"), []byte(`package main

import (
	"fmt"
)

// --- Types ---

// TokenKind enumerates lexeme kinds the Tokenize function must recognize.
type TokenKind int

const (
	TokNumber TokenKind = iota
	TokPlus
	TokMinus
	TokStar
	TokSlash
	TokLParen
	TokRParen
	TokEOF
)

// Token is one lexeme with its kind and (for TokNumber) its numeric value.
type Token struct {
	Kind  TokenKind
	Value float64
}

// Expr is the AST node interface. Implementations represent number literals,
// binary ops, and unary ops. Pick any concrete types you want — the tests
// only call the four public functions below.
type Expr interface {
	exprNode()
}

// --- Stub functions — implement these. ---

// Tokenize produces a slice of tokens ending in a single TokEOF. Whitespace
// is skipped. Numbers may be integer or float (e.g. "3" or "3.14"). Any
// unrecognized character is an error.
//
// TODO: implement.
func Tokenize(src string) ([]Token, error) {
	return nil, fmt.Errorf("not implemented")
}

// Parse consumes the token slice and returns an AST root using standard
// arithmetic precedence: * and / bind tighter than + and -, both left-
// associative. Parentheses override precedence. Unary minus is supported
// and has higher precedence than binary ops.
//
// Returns an error if tokens are ill-formed (e.g. unmatched paren, trailing
// operator, empty input).
//
// TODO: implement.
func Parse(tokens []Token) (Expr, error) {
	return nil, fmt.Errorf("not implemented")
}

// Eval walks the AST and returns the numeric result. It returns an error on
// division by zero. Other runtime errors are at your discretion but the tests
// only check division-by-zero.
//
// TODO: implement.
func Eval(e Expr) (float64, error) {
	return 0, fmt.Errorf("not implemented")
}

// Evaluate is the public entry point: Tokenize + Parse + Eval.
//
// TODO: implement.
func Evaluate(src string) (float64, error) {
	return 0, fmt.Errorf("not implemented")
}

func main() {
	fmt.Println("expression evaluator — run tests")
}
`), 0644)

		os.WriteFile(filepath.Join(dir, "main_test.go"), []byte(`package main

import (
	"math"
	"strings"
	"testing"
)

func approx(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// --- Public API: Evaluate (end-to-end) ---

func TestEvaluate_Basic(t *testing.T) {
	got, err := Evaluate("1 + 2")
	if err != nil || !approx(got, 3) {
		t.Errorf("1+2: got %v err=%v", got, err)
	}
}

func TestEvaluate_Precedence(t *testing.T) {
	got, err := Evaluate("2 + 3 * 4")
	if err != nil || !approx(got, 14) {
		t.Errorf("2+3*4: got %v err=%v (want 14)", got, err)
	}
}

func TestEvaluate_Parens(t *testing.T) {
	got, err := Evaluate("(2 + 3) * 4")
	if err != nil || !approx(got, 20) {
		t.Errorf("(2+3)*4: got %v err=%v (want 20)", got, err)
	}
}

func TestEvaluate_LeftAssociativeMinus(t *testing.T) {
	got, err := Evaluate("10 - 3 - 2")
	if err != nil || !approx(got, 5) {
		t.Errorf("10-3-2: got %v err=%v (want 5, left-assoc)", got, err)
	}
}

func TestEvaluate_LeftAssociativeDiv(t *testing.T) {
	got, err := Evaluate("16 / 4 / 2")
	if err != nil || !approx(got, 2) {
		t.Errorf("16/4/2: got %v err=%v (want 2, left-assoc)", got, err)
	}
}

func TestEvaluate_UnaryMinus(t *testing.T) {
	got, err := Evaluate("-5 + 3")
	if err != nil || !approx(got, -2) {
		t.Errorf("-5+3: got %v err=%v (want -2)", got, err)
	}
}

func TestEvaluate_UnaryInsideParens(t *testing.T) {
	got, err := Evaluate("2 * (-3 + 1)")
	if err != nil || !approx(got, -4) {
		t.Errorf("2*(-3+1): got %v err=%v (want -4)", got, err)
	}
}

func TestEvaluate_NestedParens(t *testing.T) {
	got, err := Evaluate("((1 + 2) * (3 - 4))")
	if err != nil || !approx(got, -3) {
		t.Errorf("((1+2)*(3-4)): got %v err=%v (want -3)", got, err)
	}
}

func TestEvaluate_FloatLiteral(t *testing.T) {
	got, err := Evaluate("3.14 * 2")
	if err != nil || !approx(got, 6.28) {
		t.Errorf("3.14*2: got %v err=%v (want 6.28)", got, err)
	}
}

// --- Error paths ---

func TestEvaluate_EmptyInput(t *testing.T) {
	_, err := Evaluate("")
	if err == nil {
		t.Errorf("empty input should return error")
	}
}

func TestEvaluate_UnbalancedParen(t *testing.T) {
	_, err := Evaluate("(1 + 2")
	if err == nil {
		t.Errorf("unbalanced paren should return error")
	}
}

func TestEvaluate_DivisionByZero(t *testing.T) {
	_, err := Evaluate("1 / 0")
	if err == nil {
		t.Errorf("division by zero should return error")
	}
	if err != nil && !strings.Contains(strings.ToLower(err.Error()), "zero") &&
		!strings.Contains(strings.ToLower(err.Error()), "divide") {
		t.Logf("note: error text %q does not mention 'zero' or 'divide' — OK but weird", err.Error())
	}
}
`), 0644)

		os.WriteFile(filepath.Join(dir, "test.sh"), []byte("#!/bin/bash\ncd \"$(dirname \"$0\")\"\ngo test ./... 2>&1\n"), 0755)
		os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module exprcalc\n\ngo 1.21\n"), 0644)
	},
	Phases: []Phase{
		{
			Name:    "Bootstrap — coordinator spawns dev worker",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				if th.Iteration() <= 2 {
					th.Inject(`[console] Task: main.go in your codebase has four stub functions (Tokenize, Parse, Eval, Evaluate) for an arithmetic expression evaluator. main_test.go has 12 tests covering precedence, associativity, parens, unary minus, floats, and error cases. Spawn a worker with id="dev-worker" using spawn(id="dev-worker", directive="Implement a complete arithmetic expression evaluator: read main.go and main_test.go, then implement Tokenize, Parse, Eval, and Evaluate in main.go so every test passes. Use a standard recursive-descent parser. Run tests after each change, read failures carefully, and iterate until all 12 tests pass.", mcp="codebase", tools="send,done,pace"). Do not code yourself.`)
				}
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
			Name:    "First write — worker makes a real implementation attempt",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				src := string(data)
				stubbed := strings.Contains(src, `return nil, fmt.Errorf("not implemented")`)
				t.Logf("  ... main.go=%d bytes stubbed=%v threads=%v", len(data), stubbed, ThreadIDs(th))
				// Progress = stubs gone AND file substantially grown
				return !stubbed && len(data) > 1500
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				t.Logf("  first-draft main.go: %d bytes", len(data))
			},
		},
		{
			Name:    "Iteration — worker runs tests, reads failures, fixes bugs",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				lastSize := 0
				cycles := 0
				return func(t *testing.T, dir string, th *Thinker) bool {
					// Count distinct iterations by tracking file mutations
					data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
					if len(data) != lastSize && lastSize != 0 {
						cycles++
					}
					lastSize = len(data)

					// Run tests ourselves — independent of whatever the worker
					// thinks is happening
					cmd := exec.Command("go", "test", "./...")
					cmd.Dir = dir
					out, err := cmd.CombinedOutput()
					passing := err == nil
					passCount := 0
					failCount := 0
					for _, line := range strings.Split(string(out), "\n") {
						if strings.HasPrefix(line, "--- PASS") {
							passCount++
						}
						if strings.HasPrefix(line, "--- FAIL") {
							failCount++
						}
					}
					t.Logf("  ... go test passing=%v pass=%d fail=%d cycles=%d main.go=%d bytes threads=%v",
						passing, passCount, failCount, cycles, len(data), ThreadIDs(th))
					return passing
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				cmd := exec.Command("go", "test", "-v", "./...")
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err == nil {
					t.Logf("ALL GREEN — worker converged.")
					return
				}
				t.Logf("NOTE: tests did not reach full green within budget. Final go test output (tail):")
				lines := strings.Split(strings.TrimSpace(string(out)), "\n")
				if len(lines) > 30 {
					lines = lines[len(lines)-30:]
				}
				for _, l := range lines {
					t.Logf("    %s", l)
				}
			},
		},
		{
			Name:    "Soak — agent still alive",
			Timeout: 20 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// A dead agent is caught by the scenario hard timeout;
				// here we just let the soak window elapse. Sub-thread
				// count is informational.
				t.Logf("  ... sub-threads alive=%d", th.Threads().Count())
				return true
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("soak: %d sub-threads alive", th.Threads().Count())
				data, _ := os.ReadFile(filepath.Join(dir, "main.go"))
				t.Logf("final main.go: %d bytes, %d lines",
					len(data), strings.Count(string(data), "\n"))
			},
		},
	},
	// Hard budget: 8 minutes. Phases 2+3 together give ~5 minutes of polling;
	// bootstrap + soak add ~1.5 minutes headroom. Most of the time will be
	// spent in Phase 3 as the worker iterates on test failures.
	Timeout: 8 * time.Minute,
}

func TestScenario_LongCodingTaskHard(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}

	codebaseBin := BuildMCPBinary(t, "mcps/codebase")
	t.Logf("built codebase=%s", codebaseBin)

	s := longCodingTaskHardScenario
	s.MCPServers[0].Command = codebaseBin
	RunScenario(t, s)
}
