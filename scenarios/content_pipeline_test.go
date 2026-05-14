package scenarios

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var contentPipelineScenario = Scenario{
	Name: "ContentPipeline",
	Directive: `You manage a content production pipeline for a tech company blog.

Spawn and maintain 3 threads:
1. "researcher" — given a topic, gathers information and stores research notes.
   Tools: storage_store, storage_get, storage_list, send, done
2. "writer" — uses research to generate blog posts and social media content.
   Tools: creative_generate_post, creative_generate_image, storage_get, send, done
3. "publisher" — schedules and publishes content across social channels.
   Tools: social_post, social_get_channels, schedule_get_schedule, schedule_update_slot, send, done

Workflow:
- You receive a topic via console and tell researcher to gather info.
- When research is done, tell writer to draft a blog post and social posts.
- When content is ready, tell publisher to schedule and post across channels.

`,
	MCPServers: []MCPServerConfig{
		{Name: "storage", Command: "", Env: map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"}},
		{Name: "creative", Command: "", Env: map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "social", Command: "", Env: map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"}},
		{Name: "schedule", Command: "", Env: map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Seed social channels
		WriteJSONFile(t, dir, "channels.json", []map[string]string{
			{"id": "twitter", "name": "Twitter/X"},
			{"id": "linkedin", "name": "LinkedIn"},
			{"id": "instagram", "name": "Instagram"},
		})
		// Empty schedule
		WriteJSONFile(t, dir, "schedule.json", []map[string]any{})
	},
	Phases: []Phase{
		// No "startup — N threads spawned" phase: the phase below
		// asserts on the content that gets produced, not on how the
		// agent organised itself to produce it.
		{
			Name:    "Content production — topic to published posts",
			Timeout: 180 * time.Second,
			Wait: func() func(*testing.T, string, *Thinker) bool {
				injected := false
				return func(t *testing.T, dir string, th *Thinker) bool {
					if !injected {
						th.InjectConsole("New content topic: 'Why AI agents are replacing SaaS dashboards'. Research it, write content, and publish.")
						injected = true
					}
					// Check if content was generated (audit trail from creative/social)
					audit, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
					posts, _ := os.ReadFile(filepath.Join(dir, "posts.json"))
					return len(audit) > 50 || (len(posts) > 2 && strings.Contains(string(posts), "AI"))
				}
			}(),
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// Verify content was generated
				data, _ := os.ReadFile(filepath.Join(dir, "audit.jsonl"))
				if len(data) == 0 {
					t.Error("expected audit trail of creative/social actions")
				}
			},
		},
	},
	Timeout: 6 * time.Minute,
}

func TestScenario_ContentPipeline(t *testing.T) {
	storageBin := BuildMCPBinary(t, "mcps/storage")
	creativeBin := BuildMCPBinary(t, "mcps/creative")
	socialBin := BuildMCPBinary(t, "mcps/social")
	scheduleBin := BuildMCPBinary(t, "mcps/schedule")
	t.Logf("built storage=%s creative=%s social=%s schedule=%s", storageBin, creativeBin, socialBin, scheduleBin)

	s := contentPipelineScenario
	s.MCPServers[0].Command = storageBin
	s.MCPServers[1].Command = creativeBin
	s.MCPServers[2].Command = socialBin
	s.MCPServers[3].Command = scheduleBin
	RunScenario(t, s)
}
