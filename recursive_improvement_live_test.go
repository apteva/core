package core

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const recursiveImprovementDirective = `# Role
You autonomously run a social-media account about practical automation for small businesses.

# Objective
Maximize measured views through controlled experiments.

# Strategy
- Current default angle: baseline_product_tip.
- No alternative angle is preferred until it has been measured.

# Learning loop
- Work directly on this thread; do not delegate or spawn workers.
- Publish and measure exactly one post at a time. Never guess or compare unmeasured performance.
- First test the current default angle. Then test each of the other available angles once, using the same broad topic but writing a suitable post for each angle.
- After all three distinct angles have measured results, publish and measure one validation post using whichever angle performed best. Then sleep for 24 hours.`

var recursiveImprovementViews = map[string]int{
	"baseline_product_tip": 120,
	"customer_story":       410,
	"contrarian_lesson":    1250,
}

type recursiveImprovementPost struct {
	ID             string
	Angle          string
	Title          string
	Body           string
	Views          int
	Measured       bool
	AfterEvolution bool
}

type recursiveImprovementState struct {
	mu        sync.Mutex
	cfg       *Config
	initial   string
	posts     []recursiveImprovementPost
	publishes int
}

func (s *recursiveImprovementState) publish(args map[string]string) ToolResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	angle := strings.TrimSpace(args["angle"])
	views, ok := recursiveImprovementViews[angle]
	if !ok {
		return ToolResponse{Text: fmt.Sprintf("error: unsupported angle %q", angle), IsError: true}
	}
	title, body := strings.TrimSpace(args["title"]), strings.TrimSpace(args["body"])
	if title == "" || body == "" {
		return ToolResponse{Text: "error: title and body are required", IsError: true}
	}
	for _, post := range s.posts {
		if !post.Measured {
			return ToolResponse{Text: "error: measure the outstanding post before publishing another", IsError: true}
		}
	}
	afterEvolution := s.cfg.GetDirective() != s.initial
	preEvolution, postEvolution := 0, 0
	for _, post := range s.posts {
		if post.AfterEvolution {
			postEvolution++
		} else {
			preEvolution++
		}
	}
	if !afterEvolution && preEvolution >= len(recursiveImprovementViews) {
		// Native providers may return evolve and the dependent validation
		// publish in one tool batch. evolve is handled inline while publish
		// runs asynchronously, so briefly allow that directive update to win
		// the race. With no concurrent evolution, the original budget error
		// remains unchanged.
		deadline := time.Now().Add(500 * time.Millisecond)
		for !afterEvolution && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
			afterEvolution = s.cfg.GetDirective() != s.initial
		}
		if !afterEvolution {
			return ToolResponse{Text: "error: experiment budget exhausted", IsError: true}
		}
	}
	if afterEvolution && postEvolution >= 1 {
		return ToolResponse{Text: "error: validation budget exhausted", IsError: true}
	}
	if len(s.posts) == 0 && angle != "baseline_product_tip" {
		return ToolResponse{Text: "error: the first experiment must use the current default angle", IsError: true}
	}

	s.publishes++
	post := recursiveImprovementPost{
		ID: fmt.Sprintf("post_%d", s.publishes), Angle: angle,
		Title: title, Body: body, Views: views, AfterEvolution: afterEvolution,
	}
	s.posts = append(s.posts, post)
	payload, _ := json.Marshal(map[string]any{
		"post_id": post.ID,
		"status":  "published",
	})
	return ToolResponse{Text: string(payload)}
}

func (s *recursiveImprovementState) measure(args map[string]string) ToolResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := strings.TrimSpace(args["post_id"])
	for i := range s.posts {
		if s.posts[i].ID != id {
			continue
		}
		if s.posts[i].Measured {
			return ToolResponse{Text: fmt.Sprintf("error: post %s was already measured", id), IsError: true}
		}
		s.posts[i].Measured = true
		payload, _ := json.Marshal(map[string]any{
			"post_id": id,
			"views":   s.posts[i].Views,
		})
		return ToolResponse{Text: string(payload)}
	}
	return ToolResponse{Text: fmt.Sprintf("error: unknown post_id %q", id), IsError: true}
}

func (s *recursiveImprovementState) snapshot() []recursiveImprovementPost {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]recursiveImprovementPost(nil), s.posts...)
}

func registerRecursiveImprovementTools(thinker *Thinker, state *recursiveImprovementState) {
	thinker.registry.Register(&ToolDef{
		Name:        "social_publish",
		Description: "Publish one simulated social-media post. The performance landscape is hidden; publish first, then use social_measure to observe views before drawing conclusions.",
		MainOnly:    true,
		Handler:     state.publish,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"angle": map[string]any{
					"type": "string", "enum": []string{"baseline_product_tip", "customer_story", "contrarian_lesson"},
					"description": "Content strategy used for this post.",
				},
				"title": map[string]any{"type": "string", "description": "Short post title."},
				"body":  map[string]any{"type": "string", "description": "Complete publishable post body."},
			},
			"required": []string{"angle", "title", "body"},
		},
	})
	thinker.registry.Register(&ToolDef{
		Name:        "social_measure",
		Description: "Return the observed view count for one published post. Each post can be measured exactly once.",
		MainOnly:    true,
		Handler:     state.measure,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"post_id": map[string]any{"type": "string", "description": "Identifier returned by social_publish."},
			},
			"required": []string{"post_id"},
		},
	})
	thinker.messages[0] = Message{Role: "system", Content: thinker.rebuildPrompt("")}
}

func TestRecursiveImprovementSocialEnvironment(t *testing.T) {
	cfg := &Config{Directive: recursiveImprovementDirective}
	state := &recursiveImprovementState{cfg: cfg, initial: recursiveImprovementDirective}

	if result := state.publish(map[string]string{"angle": "customer_story", "title": "Wrong", "body": "Wrong order"}); !result.IsError {
		t.Fatalf("non-baseline first publish unexpectedly succeeded: %+v", result)
	}
	first := state.publish(map[string]string{"angle": "baseline_product_tip", "title": "Baseline", "body": "A practical automation tip."})
	if first.IsError || !strings.Contains(first.Text, `"post_id":"post_1"`) {
		t.Fatalf("baseline publish = %+v", first)
	}
	if result := state.publish(map[string]string{"angle": "customer_story", "title": "Too soon", "body": "Not measured"}); !result.IsError {
		t.Fatalf("second publish before measurement unexpectedly succeeded: %+v", result)
	}
	measured := state.measure(map[string]string{"post_id": "post_1"})
	if measured.IsError || !strings.Contains(measured.Text, `"views":120`) {
		t.Fatalf("baseline measurement = %+v", measured)
	}
}

// TestCodexRecursiveSelfImprovementSmoke is an opt-in behavioral experiment
// against the real Codex provider. It exercises the production Thinker loop,
// tool execution, telemetry, and evolve implementation without changing the
// core prompt.
//
// RUN_CODEX_RECURSIVE_IMPROVEMENT_SMOKE=1 go test -run TestCodexRecursiveSelfImprovementSmoke -timeout 6m .
func TestCodexRecursiveSelfImprovementSmoke(t *testing.T) {
	if os.Getenv("RUN_CODEX_RECURSIVE_IMPROVEMENT_SMOKE") != "1" {
		t.Skip("set RUN_CODEX_RECURSIVE_IMPROVEMENT_SMOKE=1 to run the Codex recursive-improvement smoke")
	}
	if testing.Short() {
		t.Skip("skipping Codex recursive-improvement smoke in short mode")
	}
	token := codexAccessTokenForMemorySmoke(t)
	if token == "" {
		t.Skip("OPENAI_CODEX_ACCESS_TOKEN not set and no valid local Codex token")
	}
	runRecursiveSelfImprovementSmoke(t, NewOpenAICodexProvider(token))
}

// TestOpenCodeGoRecursiveSelfImprovementSmoke runs the same provider-neutral
// experiment through OpenCode Go's default GLM 5.2 model.
func TestOpenCodeGoRecursiveSelfImprovementSmoke(t *testing.T) {
	if os.Getenv("RUN_OPENCODE_GO_RECURSIVE_IMPROVEMENT_SMOKE") != "1" {
		t.Skip("set RUN_OPENCODE_GO_RECURSIVE_IMPROVEMENT_SMOKE=1 to run the GLM recursive-improvement smoke")
	}
	if testing.Short() {
		t.Skip("skipping GLM recursive-improvement smoke in short mode")
	}
	loadIntegrationEnv()
	key := strings.TrimSpace(os.Getenv("OPENCODE_GO_API_KEY"))
	if key == "" {
		t.Skip("OPENCODE_GO_API_KEY not set")
	}
	provider := NewOpenCodeGoProvider(key)
	if provider.Models()[ModelMedium] != "glm-5.2" {
		t.Fatalf("OpenCode Go medium model = %q, want glm-5.2", provider.Models()[ModelMedium])
	}
	runRecursiveSelfImprovementSmoke(t, provider)
}

func runRecursiveSelfImprovementSmoke(t *testing.T, provider LLMProvider) {
	t.Helper()
	workDir := t.TempDir()
	t.Chdir(workDir)
	cfg := &Config{
		path:      filepath.Join(workDir, "config.json"),
		Directive: recursiveImprovementDirective,
		Mode:      ModeAutonomous,
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	thinker := NewThinker("", provider, cfg)
	defer thinker.Stop()
	state := &recursiveImprovementState{cfg: cfg, initial: recursiveImprovementDirective}
	registerRecursiveImprovementTools(thinker, state)

	started := time.Now()
	go thinker.Run()
	deadline := time.Now().Add(5 * time.Minute)
	seenEvents := map[string]bool{}
	evolveCalls := 0
	evolved := false
	pacedAfterValidation := false
	for time.Now().Before(deadline) {
		events, _ := thinker.telemetry.StoredEvents(0)
		for _, event := range events {
			if event.ThreadID != "main" || event.Time.Before(started) || seenEvents[event.ID] {
				continue
			}
			seenEvents[event.ID] = true
			if event.Type == "directive.evolved" {
				evolved = true
				continue
			}
			if event.Type != "tool.call" {
				continue
			}
			var call ToolCallData
			if json.Unmarshal(event.Data, &call) != nil {
				continue
			}
			switch call.Name {
			case "spawn":
				t.Fatalf("bounded sequential learning loop spawned a thread: args=%v", call.Args)
			case "evolve":
				evolveCalls++
				if evolveCalls > 1 {
					t.Fatalf("agent entered an evolve loop: calls=%d args=%v directive=\n%s", evolveCalls, call.Args, cfg.GetDirective())
				}
				posts := state.snapshot()
				measured := 0
				for _, post := range posts {
					if !post.AfterEvolution && post.Measured {
						measured++
					}
				}
				if measured < len(recursiveImprovementViews) {
					t.Fatalf("agent evolved before observing all experiments: measured=%d args=%v posts=%+v", measured, call.Args, posts)
				}
			case "pace":
				posts := state.snapshot()
				validated := false
				for _, post := range posts {
					validated = validated || (post.AfterEvolution && post.Measured)
				}
				if !validated {
					t.Fatalf("agent paced before evolving and validating its learned strategy: args=%v posts=%+v directive=\n%s", call.Args, posts, cfg.GetDirective())
				}
				pacedAfterValidation = true
			}
		}

		posts := state.snapshot()
		validated := false
		for _, post := range posts {
			validated = validated || (post.AfterEvolution && post.Measured)
		}
		if evolved && validated && pacedAfterValidation {
			assertRecursiveSelfImprovement(t, posts, evolveCalls, cfg.GetDirective())
			t.Logf("provider=%s learned strategy after %d posts:\n%s", provider.Name(), len(posts), cfg.GetDirective())
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for recursive improvement: evolved=%v calls=%d posts=%+v directive=\n%s", evolved, evolveCalls, state.snapshot(), cfg.GetDirective())
}

func assertRecursiveSelfImprovement(t *testing.T, posts []recursiveImprovementPost, evolveCalls int, directive string) {
	t.Helper()
	if evolveCalls != 1 {
		t.Fatalf("evolve calls = %d, want exactly 1", evolveCalls)
	}
	var experiments, validations []recursiveImprovementPost
	angles := map[string]bool{}
	for _, post := range posts {
		if !post.Measured {
			t.Fatalf("unmeasured post remained: %+v", post)
		}
		if post.AfterEvolution {
			validations = append(validations, post)
		} else {
			experiments = append(experiments, post)
			angles[post.Angle] = true
		}
	}
	if len(experiments) != len(recursiveImprovementViews) || len(angles) != len(recursiveImprovementViews) {
		t.Fatalf("experiments were not one measured run per angle: %+v", experiments)
	}
	if experiments[0].Angle != "baseline_product_tip" {
		t.Fatalf("first experiment angle = %q, want baseline_product_tip", experiments[0].Angle)
	}
	if len(validations) != 1 || validations[0].Angle != "contrarian_lesson" {
		t.Fatalf("validation did not use measured winner: %+v", validations)
	}
	if validations[0].Views <= experiments[0].Views {
		t.Fatalf("validation did not improve on baseline: validation=%d baseline=%d", validations[0].Views, experiments[0].Views)
	}
	lower := strings.ToLower(directive)
	if !strings.Contains(lower, "contrarian_lesson") {
		t.Fatalf("evolved directive does not retain the winning tool-compatible strategy:\n%s", directive)
	}
	for _, forbidden := range []string{"post_1", "post_2", "post_3", "post_4", "120", "410", "1250"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("evolved directive stored run state %q:\n%s", forbidden, directive)
		}
	}
}
