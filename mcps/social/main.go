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
	PostedAt      string `json:"posted_at"`
}

type AuditEntry struct {
	Time string            `json:"time"`
	Tool string            `json:"tool"`
	Args map[string]string `json:"args"`
}

var dataDir string
var postCounter int

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
		time.Sleep(500 * time.Millisecond) // simulate API call to social platform
		postCounter++
		post := Post{
			ID:            fmt.Sprintf("post-%d", postCounter),
			Channel:       channel,
			Content:       content,
			Image:         args["image"],
			ScheduledTime: args["scheduled_time"],
			PostedAt:      time.Now().UTC().Format(time.RFC3339),
		}
		posts := loadPosts()
		posts = append(posts, post)
		savePosts(posts)
		textResult(id, fmt.Sprintf("Published to %s: post ID %s", channel, post.ID))

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

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("SOCIAL_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
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
