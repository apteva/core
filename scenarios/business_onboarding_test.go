package scenarios

import (
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var businessOnboardingScenario = Scenario{
	Name: "BusinessOnboarding",
	Directive: `You're a new assistant being onboarded. You don't yet know anything
about the business you'll be working for — the user will teach you
through this conversation. Be conversational, listen, and make sure
what you learn persists: a fresh version of you starting tomorrow
should already know who this business is and how they talk.`,
	Phases: []Phase{
		{
			Name:    "Phase 1: User introduces business identity + products",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.Config().SetMode(ModeLearn)
						th.InjectConsole("Hey — I'm setting you up as the assistant for my bakery. We're called 'Crumb & Co.', based in Brooklyn. We specialize in long-fermented sourdough and rye, and our signature product is a 50-hour fermented country loaf.")
						injected = true
					}
					return th.Memory().Count() >= 2
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 1: %d entries", th.Memory().Count())
				for _, e := range th.Memory().Active() {
					t.Logf("  - %s", e.Content)
				}
			},
		},
		{
			Name:    "Phase 2: User teaches tone of voice + firm rules",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("A couple more things about how we work. Tone: warm and casual — like chatting with a neighbor across the counter, never corporate-speak, never exclamation marks. And two firm rules: we never discount below cost, and we never ship — pickup or local Brooklyn delivery only.")
						injected = true
					}
					return th.Memory().Count() >= 4
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				t.Logf("memory after phase 2: %d entries", th.Memory().Count())
				for _, e := range th.Memory().Active() {
					t.Logf("  - %s", e.Content)
				}
			},
		},
		{
			Name:    "Phase 3: User asks the agent to evolve its directive",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				initialLen := 0
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						initialLen = len(th.Config().GetDirective())
						th.InjectConsole("Great. Make sure who we are is baked into how you show up from the very start of any future session — a fresh you should wake up already knowing us.")
						injected = true
					}
					return len(th.Config().GetDirective()) > initialLen+200
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				d := th.Config().GetDirective()
				t.Logf("evolved directive length: %d chars", len(d))
				t.Logf("evolved directive:\n%s", d)
				low := strings.ToLower(d)
				for _, kw := range []string{"crumb", "sourdough"} {
					if !strings.Contains(low, kw) {
						t.Errorf("evolved directive missing %q — personality not baked in", kw)
					}
				}
			},
		},
		{
			Name:    "Phase 4: Context reset — persona survives via memory + directive",
			Timeout: 120 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				msgsAtInject := 0
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.ResetConversation()
						t.Logf("conversation reset — %d memory entries remain, directive=%d chars",
							th.Memory().Count(), len(th.Config().GetDirective()))
						th.InjectConsole("Quick sanity check: in one short paragraph, tell me who you are, what we sell, and how you talk to customers.")
						msgsAtInject = len(th.Messages())
						injected = true
					}
					for i := len(th.Messages()) - 1; i >= msgsAtInject; i-- {
						m := th.Messages()[i]
						if m.Role != "assistant" {
							continue
						}
						low := strings.ToLower(m.TextContent())
						if strings.Contains(low, "crumb") && strings.Contains(low, "sourdough") {
							return true
						}
					}
					return false
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				var last string
				for i := len(th.Messages()) - 1; i >= 0; i-- {
					if th.Messages()[i].Role == "assistant" {
						last = th.Messages()[i].TextContent()
						break
					}
				}
				t.Logf("post-reset self-description:\n%s", last)
				low := strings.ToLower(last)
				for _, kw := range []string{"crumb", "sourdough", "brooklyn"} {
					if !strings.Contains(low, kw) {
						t.Errorf("post-reset response missing %q — persona did not survive reset", kw)
					}
				}
				t.Logf("final memory: %d entries, directive: %d chars",
					th.Memory().Count(), len(th.Config().GetDirective()))
			},
		},
	},
	Timeout: 15 * time.Minute,
}

func TestScenario_BusinessOnboarding(t *testing.T) {
	RunScenario(t, businessOnboardingScenario)
}
