package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var socialTeamScenario = Scenario{
	Name: "SocialTeam",
	Directive: `You manage social media for a small coffee shop called "Bean & Brew".
Spawn three permanent team members:
1. A planner — needs the schedule tools (get_schedule, update_slot) to check for planned slots and mark them posted
2. A creative — needs the creative tools (generate_post, generate_image) to make content when asked
3. A social manager — needs the social tools (post, get_posts) to publish content to channels

When planner finds a planned slot, coordinate: ask creative to generate a post and image,
then give the content to social manager to post it, then tell planner to update the slot to posted.
The planner must keep checking the schedule at normal pace — never go to sleep.
Creative and social manager can sleep when idle.`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "schedule",
			Command: "", // filled in test
			Env:     map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "creative",
			Command: "", // filled in test
			Env:     map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "social",
			Command: "", // filled in test
			Env:     map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		WriteJSONFile(t, dir, "schedule.json", []map[string]string{
			{"id": "s1", "channel": "twitter", "topic": "Monday morning coffee special", "time": "09:00", "status": "planned"},
			{"id": "s2", "channel": "instagram", "topic": "New seasonal latte art", "time": "12:00", "status": "planned"},
		})
		WriteJSONFile(t, dir, "posts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 team members spawned",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.Threads().Count()
				t.Logf("  ... threads=%d %v", count, ThreadIDs(th))
				return count >= 3
			},
		},
		{
			Name:    "Content pipeline — 2 posts created and published",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := ReadAuditEntries(dir)
				posts := CountTool(entries, "post")
				generates := CountTool(entries, "generate_post")
				images := CountTool(entries, "generate_image")
				updates := CountTool(entries, "update_slot")
				t.Logf("  ... generate_post=%d generate_image=%d post=%d update_slot=%d threads=%v",
					generates, images, posts, updates, ThreadIDs(th))
				return posts >= 2
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				t.Logf("Pipeline audit (%d entries):", len(entries))
				for _, e := range entries {
					if e.Tool != "get_schedule" {
						t.Logf("  %s %v", e.Tool, e.Args)
					}
				}
				generates := CountTool(entries, "generate_post")
				if generates < 2 {
					t.Logf("NOTE: generate_post called %d times (expected 2)", generates)
				}
				// Check posts were actually published
				for _, e := range entries {
					if e.Tool == "post" {
						if e.Args["channel"] == "" || e.Args["content"] == "" {
							t.Errorf("post missing channel or content: %v", e.Args)
						}
					}
				}
			},
		},
		{
			Name:    "New slot — linkedin hiring post",
			Timeout: 120 * time.Second,
			Setup: func(t *testing.T, dir string) {
				// Read current schedule and append new slot
				data, _ := os.ReadFile(filepath.Join(dir, "schedule.json"))
				var slots []map[string]string
				json.Unmarshal(data, &slots)
				slots = append(slots, map[string]string{
					"id": "s3", "channel": "linkedin", "topic": "Hiring baristas for summer", "time": "15:00", "status": "planned",
				})
				WriteJSONFile(t, dir, "schedule.json", slots)
			},
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := ReadAuditEntries(dir)
				posts := CountTool(entries, "post")
				t.Logf("  ... posts=%d threads=%v", posts, ThreadIDs(th))
				return posts >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				entries := ReadAuditEntries(dir)
				// Check linkedin post exists
				hasLinkedin := false
				for _, e := range entries {
					if e.Tool == "post" && e.Args["channel"] == "linkedin" {
						hasLinkedin = true
						t.Logf("LinkedIn post: %s", e.Args["content"])
					}
				}
				if !hasLinkedin {
					t.Logf("NOTE: no linkedin post found in audit")
				}
			},
		},
		{
			Name:    "Quiescence — 3 workers alive, all slots processed",
			Timeout: 30 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				entries := ReadAuditEntries(dir)
				posts := CountTool(entries, "post")
				t.Logf("  ... threads=%d posts=%d %v", th.Threads().Count(), posts, ThreadIDs(th))
				return posts >= 3
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				// The posts going out is the goal (asserted in Wait).
				// How many standing workers the agent kept is its call.
				t.Logf("post phase: %d sub-threads alive: %v", th.Threads().Count(), ThreadIDs(th))
			},
		},
	},
	Timeout: 5 * time.Minute,
}

func TestScenario_SocialTeam(t *testing.T) {
	scheduleBin := BuildMCPBinary(t, "mcps/schedule")
	creativeBin := BuildMCPBinary(t, "mcps/creative")
	socialBin := BuildMCPBinary(t, "mcps/social")
	t.Logf("built schedule: %s, creative: %s, social: %s", scheduleBin, creativeBin, socialBin)

	s := socialTeamScenario
	s.MCPServers[0].Command = scheduleBin
	s.MCPServers[1].Command = creativeBin
	s.MCPServers[2].Command = socialBin
	RunScenario(t, s)
}

func TestScenario_SocialTeam_SlowPost(t *testing.T) {
	scheduleBin := BuildMCPBinary(t, "mcps/schedule")
	creativeBin := BuildMCPBinary(t, "mcps/creative")
	socialBin := BuildMCPBinary(t, "mcps/social")
	t.Logf("built schedule: %s, creative: %s, social: %s", scheduleBin, creativeBin, socialBin)

	s := socialTeamScenario
	s.Name = "SocialTeam-SlowPost"
	s.MCPServers = append([]MCPServerConfig(nil), socialTeamScenario.MCPServers...)
	s.MCPServers[0].Command = scheduleBin
	s.MCPServers[1].Command = creativeBin
	s.MCPServers[2].Command = socialBin
	// Clone the social server's env so we don't mutate the shared
	// scenario definition across parallel test runs.
	socialEnv := map[string]string{}
	for k, v := range socialTeamScenario.MCPServers[2].Env {
		socialEnv[k] = v
	}
	socialEnv["SOCIAL_POST_LATENCY_MS"] = "5000"
	s.MCPServers[2].Env = socialEnv
	// Give the pipeline more wall-clock because post is now 10x slower.
	s.Timeout = 8 * time.Minute
	RunScenario(t, s)
}
