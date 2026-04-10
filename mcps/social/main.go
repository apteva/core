// MCP server for social media publishing.
// State in SOCIAL_DATA_DIR: posts.json, audit.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Result  any    `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type Post struct {
	ID            string `json:"id"`
	Channel       string `json:"channel"`
	Content       string `json:"content"`
	Image         string `json:"image,omitempty"`
	ScheduledTime string `json:"scheduled_time,omitempty"`
	Status        string `json:"status"` // "posted" or "scheduled"
	PostedAt      string `json:"posted_at,omitempty"`
}

type AuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

var dataDir string
var postCounter int

func initPostCounter() {
	posts := loadPosts()
	for _, p := range posts {
		var id int
		fmt.Sscanf(p.ID, "post-%d", &id)
		if id >= postCounter {
			postCounter = id + 1
		}
	}
}

func respond(id int64, result any) {
	data, _ := json.Marshal(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
	fmt.Println(string(data))
}

func respondError(id int64, code int, msg string) {
	data, _ := json.Marshal(jsonRPCResponse{
		JSONRPC: "2.0", ID: id,
		Error: &struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		}{code, msg},
	})
	fmt.Println(string(data))
}

func textResult(id int64, text string) {
	respond(id, map[string]any{
		"content": []map[string]string{{"type": "text", "text": text}},
	})
}

func audit(tool string, args map[string]string) {
	entry := AuditEntry{Time: time.Now().UTC().Format(time.RFC3339), Tool: tool, Args: args}
	data, _ := json.Marshal(entry)
	f, _ := os.OpenFile(filepath.Join(dataDir, "audit.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if f != nil {
		f.WriteString(string(data) + "\n")
		f.Close()
	}
}

func loadPosts() []Post {
	data, err := os.ReadFile(filepath.Join(dataDir, "posts.json"))
	if err != nil {
		return nil
	}
	var posts []Post
	json.Unmarshal(data, &posts)
	return posts
}

func savePosts(posts []Post) {
	data, _ := json.MarshalIndent(posts, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "posts.json"), data, 0644)
}

func handleToolCall(id int64, name string, args map[string]string) {
	audit(name, args)

	switch name {
	case "post":
		channel := args["channel"]
		content := args["content"]
		if channel == "" || content == "" {
			respondError(id, -32602, "channel and content are required")
			return
		}
		// Check daily limit: max 1 post per channel per day (posted or scheduled)
		today := time.Now().UTC().Format("2006-01-02")
		posts := loadPosts()
		for _, p := range posts {
			if p.Channel != channel {
				continue
			}
			// Check posted today
			if p.PostedAt != "" && len(p.PostedAt) >= 10 && p.PostedAt[:10] == today {
				textResult(id, fmt.Sprintf("REJECTED: Already posted to %s today (post %s). Limit is 1 per channel per day.", channel, p.ID))
				return
			}
			// Check scheduled today
			if p.ScheduledTime != "" && len(p.ScheduledTime) >= 10 && p.ScheduledTime[:10] == today {
				textResult(id, fmt.Sprintf("REJECTED: Already scheduled on %s for today (post %s at %s). Limit is 1 per channel per day.", channel, p.ID, p.ScheduledTime))
				return
			}
		}
		time.Sleep(500 * time.Millisecond) // simulate API call
		postCounter++
		scheduledTime := args["scheduled_time"]
		post := Post{
			ID:            fmt.Sprintf("post-%d", postCounter),
			Channel:       channel,
			Content:       content,
			Image:         args["image"],
			ScheduledTime: scheduledTime,
		}
		if scheduledTime != "" {
			post.Status = "scheduled"
		} else {
			post.Status = "posted"
			post.PostedAt = time.Now().UTC().Format(time.RFC3339)
		}
		posts = append(posts, post)
		savePosts(posts)
		if scheduledTime != "" {
			textResult(id, fmt.Sprintf("Scheduled on %s for %s: post ID %s", channel, scheduledTime, post.ID))
		} else {
			textResult(id, fmt.Sprintf("Published to %s: post ID %s", channel, post.ID))
		}

	case "get_channels":
		channels := []map[string]string{
			{"name": "twitter", "description": "Short posts, 280 chars max, hashtags"},
			{"name": "instagram", "description": "Visual posts with captions, hashtags"},
			{"name": "linkedin", "description": "Professional posts, longer format"},
		}
		data, _ := json.Marshal(channels)
		textResult(id, string(data))

	case "get_posts":
		posts := loadPosts()
		if posts == nil {
			posts = []Post{}
		}
		data, _ := json.Marshal(posts)
		textResult(id, string(data))

	case "get_todays_posts":
		today := time.Now().UTC().Format("2006-01-02")
		posts := loadPosts()
		var todayPosts []Post
		for _, p := range posts {
			isToday := false
			if p.PostedAt != "" && len(p.PostedAt) >= 10 && p.PostedAt[:10] == today {
				isToday = true
			}
			if p.ScheduledTime != "" && len(p.ScheduledTime) >= 10 && p.ScheduledTime[:10] == today {
				isToday = true
			}
			if isToday {
				todayPosts = append(todayPosts, p)
			}
		}
		// Build summary per channel
		channelsDone := map[string]string{} // channel → status (posted/scheduled)
		for _, p := range todayPosts {
			channelsDone[p.Channel] = p.Status
		}
		summary := fmt.Sprintf("%d posts today. ", len(todayPosts))
		for _, ch := range []string{"twitter", "linkedin", "instagram"} {
			status, ok := channelsDone[ch]
			if !ok {
				summary += fmt.Sprintf("%s: ⬜ not yet. ", ch)
			} else if status == "scheduled" {
				summary += fmt.Sprintf("%s: 📅 scheduled. ", ch)
			} else {
				summary += fmt.Sprintf("%s: ✅ posted. ", ch)
			}
		}
		result, _ := json.Marshal(map[string]any{
			"summary": summary,
			"posts":   todayPosts,
			"channels_remaining": func() []string {
				var remaining []string
				for _, ch := range []string{"twitter", "linkedin", "instagram"} {
					if _, done := channelsDone[ch]; !done {
						remaining = append(remaining, ch)
					}
				}
				return remaining
			}(),
		})
		textResult(id, string(result))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("SOCIAL_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
	initPostCounter()
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		var req jsonRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			continue
		}
		if req.ID == nil {
			continue
		}
		id := *req.ID

		switch req.Method {
		case "initialize":
			respond(id, map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":   map[string]any{"tools": map[string]any{}},
				"serverInfo":     map[string]string{"name": "social", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "post",
						"description": "Publish or schedule a post to a social media channel. Provide content and optionally an image URL and scheduled time.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"channel":        map[string]string{"type": "string", "description": "Channel: twitter, instagram, or linkedin"},
								"content":        map[string]string{"type": "string", "description": "Post text content"},
								"image":          map[string]string{"type": "string", "description": "Image URL (optional)"},
								"scheduled_time": map[string]string{"type": "string", "description": "When to post (optional, e.g. 09:00)"},
							},
							"required": []string{"channel", "content"},
						},
					},
					{
						"name":        "get_channels",
						"description": "List available social media channels with their descriptions and posting guidelines.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "get_posts",
						"description": "Get all published and scheduled posts.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "get_todays_posts",
						"description": "Check what's been posted today. Shows per-channel status (posted/not yet) and which channels still need a post. Limit: 1 post per channel per day.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
				},
			})
		case "tools/call":
			var params struct {
				Name      string            `json:"name"`
				Arguments map[string]string `json:"arguments"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				respondError(id, -32602, "invalid params")
				continue
			}
			handleToolCall(id, params.Name, params.Arguments)
		default:
			respondError(id, -32601, fmt.Sprintf("unknown method: %s", req.Method))
		}
	}
}
