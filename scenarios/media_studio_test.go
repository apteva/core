package scenarios

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/apteva/core"
)

var mediaStudioScenario = Scenario{
	Name: "MediaStudio",
	Directive: `You are the coordinator of "NovaStudio", an AI content studio. You do
NOT call creative, media, or social tools directly — your whole job is to
spin up a fixed team of specialists and route work between them.

The studio has three active shows. Use these ids verbatim — do not rename:

1. cooking_show — "Chef Aurora's Kitchen"
   theme: recipe   (every caption must contain the word "recipe")
   style: warm, colorful kitchen sets with close-ups of ingredients

2. virtual_influencer — "Luna Dreams"
   theme: lifestyle   (every caption must contain "lifestyle")
   style: dreamy aesthetic vlogs, neon-lit bedrooms, pastel moodboards

3. fitness_channel — "Atlas Training"
   theme: workout   (every caption must contain "workout")
   style: high-energy gym sessions, dynamic camera, fast cuts

═══════════════════════════════════════════════════════════════════════
TEAM STRUCTURE (spawn ALL SIX of these workers immediately, one shot)
═══════════════════════════════════════════════════════════════════════

PRODUCER TEAM — one producer per show. Producers create media assets
for their assigned show and then HAND OFF to the poster team by calling
send(to=<poster>, message=...) for every network poster.

  spawn(id="producer-cooking_show",      mcp="creative,media", tools="creative_generate_image,creative_generate_post,media_upload_file,media_create_reel,send,done",  directive="<brief for cooking_show>")
  spawn(id="producer-virtual_influencer",mcp="creative,media", tools="creative_generate_image,creative_generate_post,media_upload_file,media_create_reel,send,done",  directive="<brief for virtual_influencer>")
  spawn(id="producer-fitness_channel",   mcp="creative,media", tools="creative_generate_image,creative_generate_post,media_upload_file,media_create_reel,send,done",  directive="<brief for fitness_channel>")

POSTER TEAM — one poster per social network. Each poster receives
messages from every producer and publishes to its OWN network.

  spawn(id="poster-twitter",   mcp="social", tools="social_post,send,done", directive="<poster brief>")
  spawn(id="poster-instagram", mcp="social", tools="social_post,send,done", directive="<poster brief>")
  spawn(id="poster-linkedin",  mcp="social", tools="social_post,send,done", directive="<poster brief>")

═══════════════════════════════════════════════════════════════════════
PRODUCER BRIEFS (the directive you put on each producer)
═══════════════════════════════════════════════════════════════════════

Each producer brief must:
  1. Include the show's id, title, theme word, and style.
  2. Tell the producer its FIRST actions are: (a) generate an image with
     creative_generate_image (prompt should describe the style), (b) create
     a short reel with media_create_reel.
  3. Then for each of the three posters — poster-twitter, poster-instagram,
     poster-linkedin — call send with a message containing: the project id,
     the theme word, and a draft caption tailored to that network. Example:

       send(to="poster-twitter",
            message="PUBLISH project=cooking_show theme=recipe caption=<short tweet mentioning recipe>")
       send(to="poster-instagram", message="PUBLISH project=cooking_show theme=recipe caption=...")
       send(to="poster-linkedin",  message="PUBLISH project=cooking_show theme=recipe caption=...")

  4. After sending to all 3 posters, the producer reports "<id> PRODUCED"
     back to main with send(to="main", message="<id> PRODUCED") and then
     calls done.

═══════════════════════════════════════════════════════════════════════
POSTER BRIEFS (the directive you put on each poster)
═══════════════════════════════════════════════════════════════════════

Each poster's directive MUST start with:

    "You are the <network> poster. You will receive PUBLISH messages
     from producers. The moment a PUBLISH message lands, your VERY FIRST
     action — before any reasoning — is to call social_post with
     channel=<network>, project=<project from message>, content=<the
     caption from the message>. Do not think. Do not plan. Call
     social_post immediately.

     The caption MUST contain the theme word from the PUBLISH message
     (recipe / lifestyle / workout).

     You expect exactly THREE PUBLISH messages — one per producer. After
     you have posted three times, send 'DONE' to main and call done.

     If social_post returns a REJECTED error for a duplicate, do not retry,
     just move on to the next project."

═══════════════════════════════════════════════════════════════════════
HARD CORRECTNESS RULES
═══════════════════════════════════════════════════════════════════════

  - Every social_post call must carry project="<show id>" — the exact id
    (cooking_show / virtual_influencer / fitness_channel).
  - Every caption contains the show's theme word (recipe / lifestyle /
    workout) matching the project it is tagged with.
  - Exactly 9 posts total: 3 shows × 3 networks. No more, no less.
  - No cross-contamination. No duplicates. No renames.

═══════════════════════════════════════════════════════════════════════
COORDINATION
═══════════════════════════════════════════════════════════════════════

Your only tools are spawn, send, done, pace. After spawning the six
workers, wait for three "PRODUCED" reports from the producers and three
"DONE" reports from the posters. When you have all six, your work is
finished.`,
	MCPServers: []MCPServerConfig{
		// MCP tools register MCP=true so the studio director sees only
		// scaffolding (spawn, send, done, search_tools, pace) by default.
		// The director's directive forces it to delegate via spawn; the
		// workers boot hot with mcps="creative" / "media" / "social"
		// preloading the full surface they need.
		{Name: "creative", Command: "", Env: map[string]string{"CREATIVE_DATA_DIR": "{{dataDir}}"}},
		{Name: "media", Command: "", Env: map[string]string{"MEDIA_DATA_DIR": "{{dataDir}}"}},
		{Name: "social", Command: "", Env: map[string]string{"SOCIAL_DATA_DIR": "{{dataDir}}"}},
	},
	DataSetup: func(t *testing.T, dir string) {
		// Empty media/posts files so save() doesn't race with first reader.
		WriteJSONFile(t, dir, "media.json", map[string]any{"files": []any{}, "assets": []any{}})
		WriteJSONFile(t, dir, "posts.json", []any{})
	},
	Phases: []Phase{
		{
			Name:    "All projects published to all channels",
			Timeout: 6 * time.Minute,
			Wait: func(t *testing.T, dir string, th *Thinker) bool {
				data, err := os.ReadFile(filepath.Join(dir, "posts.json"))
				if err != nil {
					return false
				}
				var posts []map[string]any
				json.Unmarshal(data, &posts)
				need := len(mediaStudioProjects) * 3
				t.Logf("  ... posts=%d / %d", len(posts), need)
				return len(posts) >= need
			},
			Verify: func(t *testing.T, dir string, th *Thinker) {
				data, _ := os.ReadFile(filepath.Join(dir, "posts.json"))
				var posts []map[string]any
				json.Unmarshal(data, &posts)
				expectedChannels := []string{"twitter", "instagram", "linkedin"}
				needed := len(mediaStudioProjects) * len(expectedChannels)
				if len(posts) != needed {
					t.Errorf("expected exactly %d posts (%d projects × %d channels), got %d",
						needed, len(mediaStudioProjects), len(expectedChannels), len(posts))
				}

				// Index by (project, channel) for cross-product check.
				type key struct{ project, channel string }
				seen := map[key]int{}
				for i, p := range posts {
					project, _ := p["project"].(string)
					channel, _ := p["channel"].(string)
					content, _ := p["content"].(string)
					if project == "" {
						t.Errorf("post[%d] missing project tag — agent did not pass project= to social_post", i)
						continue
					}
					if channel == "" {
						t.Errorf("post[%d] missing channel", i)
						continue
					}
					if content == "" {
						t.Errorf("post[%d] (%s/%s) empty content", i, project, channel)
					}
					seen[key{project, channel}]++
				}

				// Every (project, channel) must appear exactly once.
				for _, proj := range mediaStudioProjects {
					for _, ch := range expectedChannels {
						k := key{proj.ID, ch}
						n := seen[k]
						if n == 0 {
							t.Errorf("MISSING publication: project=%s channel=%s never posted", proj.ID, ch)
						}
						if n > 1 {
							t.Errorf("DUPLICATE publication: project=%s channel=%s posted %d times", proj.ID, ch, n)
						}
					}
				}

				// Every post's content must contain the theme keyword for its
				// claimed project — rejects cross-contamination where a
				// worker posts cooking content under the influencer project,
				// etc.
				themeByProj := map[string]string{}
				for _, p := range mediaStudioProjects {
					themeByProj[p.ID] = p.Theme
				}
				for i, p := range posts {
					project, _ := p["project"].(string)
					content, _ := p["content"].(string)
					theme, ok := themeByProj[project]
					if !ok {
						t.Errorf("post[%d] has unknown project %q", i, project)
						continue
					}
					if !strings.Contains(strings.ToLower(content), strings.ToLower(theme)) {
						t.Errorf("post[%d] project=%s channel=%v content does not contain theme %q: %q",
							i, project, p["channel"], theme, truncForLog(content, 100))
					}
				}

				// Verify the social MCP never had to reject a duplicate —
				// "REJECTED:" only surfaces if the agent posted twice for
				// the same (project, channel). We check the bus-captured
				// transcript by re-reading audit.jsonl.
				entries := ReadAuditEntries(dir)
				for _, e := range entries {
					if e.Tool == "post" {
						// We don't check the REJECTED substring in the
						// audit (it only records args), but we can count
						// that the number of post CALLS equals needed. If
						// the agent had issued extra posts, they'd be in
						// the audit even if rejected.
						continue
					}
				}
				postCalls := CountTool(entries, "post")
				if postCalls > needed+1 { // tolerate at most one accidental extra that got rejected
					t.Errorf("agent issued %d social_post calls for %d expected — extras suggest duplicates the MCP had to reject", postCalls, needed)
				}
			},
		},
	},
	Timeout: 9 * time.Minute,
	// Forces the full team to be alive simultaneously: 3 producers
	// (one per show) + 3 posters (one per social network) = 6 sub-threads.
	// threads.Count() excludes main, so 6 is the precise minimum that
	// proves the agent spawned the whole team before starting to route
	// work. Anything less means producers/posters ran sequentially.
	MinPeakThreads: 6,
	MaxThreads:     15,
}

func TestScenario_MediaStudio(t *testing.T) {
	if os.Getenv("RUN_SCENARIO_TESTS") == "" {
		t.Skip("set RUN_SCENARIO_TESTS=1")
	}
	creativeBin := BuildMCPBinary(t, "mcps/creative")
	mediaBin := BuildMCPBinary(t, "mcps/media")
	socialBin := BuildMCPBinary(t, "mcps/social")
	t.Logf("built creative=%s media=%s social=%s", creativeBin, mediaBin, socialBin)

	s := mediaStudioScenario
	s.MCPServers[0].Command = creativeBin
	s.MCPServers[1].Command = mediaBin
	s.MCPServers[2].Command = socialBin
	RunScenario(t, s)
}
