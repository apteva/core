package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var videoTeamScenario = Scenario{
	Name: "VideoTeam",
	Directive: `You manage a video production team for a tech company.

Your job: when new video files arrive, process them through a pipeline:
1. Upload and register the file in media
2. Extract 3 screenshots from the video
3. Create a 30-second reel
4. Store the pipeline status in storage
5. Plan social media posts: one reel post for instagram, one screenshot post for twitter, one announcement for linkedin
6. Generate creative copy for each post
7. Publish all posts

Spawn these permanent workers:
- "editor" — handles media processing (upload, screenshots, reels). Needs media tools. Reports to main when processing is done.
- "planner" — plans social media content from processed assets. Needs schedule and storage tools. Creates schedule slots then tells publisher.
- "publisher" — generates copy and publishes. Needs creative and social tools. Posts to channels.

The editor should periodically check for uploaded files (status=uploaded) using list_files, process them, then report to main.
Coordinate the pipeline: editor → planner → publisher.
When all posts are published, store a completion record in storage with key "pipeline:done".`,
	MCPServers: []MCPServerConfig{
		{
			Name:    "media",
			Command: "", // filled in test
			Env:     map[string]string{"MEDIA_DATA_DIR": "{{dataDir}}"},
		},
		{
			Name:    "storage",
			Command: "", // filled in test
			Env:     map[string]string{"STORAGE_DATA_DIR": "{{dataDir}}"},
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
		{
			Name:    "schedule",
			Command: "", // filled in test
			Env:     map[string]string{"SCHEDULE_DATA_DIR": "{{dataDir}}"},
		},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Pre-populate a video file waiting to be processed
		WriteJSONFile(t, dir, "media.json", map[string]any{
			"files": []map[string]string{
				{"id": "m1", "name": "product-demo.mp4", "type": "video", "duration": "3:24", "resolution": "1920x1080", "size": "245MB", "status": "uploaded", "uploaded_at": "2026-03-26T10:00:00Z"},
			},
			"assets": []any{},
		})
		WriteJSONFile(t, dir, "schedule.json", []any{})
		WriteJSONFile(t, dir, "posts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "Startup — 3 workers spawned",
			Timeout: 60 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				count := th.Threads().Count()
				t.Logf("  ... threads=%d %v", count, ThreadIDs(th))
				return count >= 3
			},
		},
		{
			Name:    "Video arrives — file processed",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				// Check media.json has assets (screenshots + reel)
				data, err := os.ReadFile(filepath.Join(dir, "media.json"))
				if err != nil {
					return false
				}
				var state struct {
					Assets []json.RawMessage `json:"assets"`
				}
				json.Unmarshal(data, &state)
				t.Logf("  ... assets=%d", len(state.Assets))
				return len(state.Assets) >= 4 // 3 screenshots + 1 reel
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "media.json"))
				var state struct {
					Files  []json.RawMessage `json:"files"`
					Assets []json.RawMessage `json:"assets"`
				}
				json.Unmarshal(data, &state)
				if len(state.Files) < 1 {
					t.Errorf("expected at least 1 file, got %d", len(state.Files))
				}
				if len(state.Assets) < 4 {
					t.Errorf("expected at least 4 assets (3 screenshots + 1 reel), got %d", len(state.Assets))
				}
			},
		},
		{
			Name:    "Social posts published",
			Timeout: 120 * time.Second,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "posts.json"))
				if err != nil {
					return false
				}
				var posts []json.RawMessage
				json.Unmarshal(data, &posts)
				t.Logf("  ... posts=%d", len(posts))
				return len(posts) >= 3 // instagram + twitter + linkedin
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "posts.json"))
				var posts []map[string]any
				json.Unmarshal(data, &posts)
				if len(posts) < 3 {
					t.Errorf("expected at least 3 posts, got %d", len(posts))
				}
				channels := map[string]bool{}
				for _, p := range posts {
					if ch, ok := p["channel"].(string); ok {
						channels[ch] = true
					}
				}
				t.Logf("channels posted to: %v", channels)
			},
		},
	},
	Timeout: 5 * time.Minute,
}

func TestScenario_VideoTeam(t *testing.T) {
	mediaBin := BuildMCPBinary(t, "mcps/media")
	storageBin := BuildMCPBinary(t, "mcps/storage")
	creativeBin := BuildMCPBinary(t, "mcps/creative")
	socialBin := BuildMCPBinary(t, "mcps/social")
	scheduleBin := BuildMCPBinary(t, "mcps/schedule")
	t.Logf("built media=%s storage=%s creative=%s social=%s schedule=%s",
		mediaBin, storageBin, creativeBin, socialBin, scheduleBin)

	s := videoTeamScenario
	s.MCPServers[0].Command = mediaBin
	s.MCPServers[1].Command = storageBin
	s.MCPServers[2].Command = creativeBin
	s.MCPServers[3].Command = socialBin
	s.MCPServers[4].Command = scheduleBin

	RunScenario(t, s)
}
