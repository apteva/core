// MCP server for AI content generation (simulated).
// State in CREATIVE_DATA_DIR: audit.jsonl
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

type AuditEntry struct {
	Time   string            `json:"time"`
	Tool   string            `json:"tool"`
	Args   map[string]string `json:"args"`
	Result string            `json:"result,omitempty"`
}

var dataDir string

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

func audit(tool string, args map[string]string, result string) {
	entry := AuditEntry{Time: time.Now().UTC().Format(time.RFC3339), Tool: tool, Args: args, Result: result}
	data, _ := json.Marshal(entry)
	f, _ := os.OpenFile(filepath.Join(dataDir, "audit.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if f != nil {
		f.WriteString(string(data) + "\n")
		f.Close()
	}
}

// simulatePost generates a social media post based on channel and topic.
func simulatePost(channel, topic, style string) string {
	channel = strings.ToLower(channel)
	if style == "" {
		style = "engaging and professional"
	}
	switch channel {
	case "twitter":
		return fmt.Sprintf("🚀 %s\n\nThread below 👇 #AI #agents #automation", topic)
	case "instagram":
		return fmt.Sprintf("✨ %s ✨\n\nBuilding the future of AI automation, one agent at a time.\n\n#AIagents #automation #tech #buildinpublic", topic)
	case "linkedin":
		return fmt.Sprintf("%s\n\nI've been working on this for the past few months and the results have been incredible. Here's what I learned and why it matters for the industry.\n\n#AI #agents #entrepreneurship #buildinpublic", topic)
	default:
		return fmt.Sprintf("%s — check out the full post for details.", topic)
	}
}

// simulateImage generates a fake image description/URL.
func simulateImage(topic, style string) string {
	if style == "" {
		style = "modern tech aesthetic"
	}
	return fmt.Sprintf("[Generated image: %s — style: %s — url: https://cdn.apteva.ai/media/%d.jpg]",
		topic, style, time.Now().UnixNano()%10000)
}

func handleToolCall(id int64, name string, args map[string]string) {
	switch name {
	case "generate_post":
		channel := args["channel"]
		topic := args["topic"]
		style := args["style"]
		if topic == "" {
			respondError(id, -32602, "topic is required")
			return
		}
		time.Sleep(800 * time.Millisecond) // simulate AI generation
		post := simulatePost(channel, topic, style)
		audit(name, args, post)
		textResult(id, post)

	case "generate_image":
		topic := args["topic"]
		style := args["style"]
		if topic == "" {
			respondError(id, -32602, "topic is required")
			return
		}
		time.Sleep(1200 * time.Millisecond) // simulate image generation (slower)
		img := simulateImage(topic, style)
		audit(name, args, img)
		textResult(id, img)

	default:
		audit(name, args, "")
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("CREATIVE_DATA_DIR")
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
				"serverInfo":     map[string]string{"name": "creative", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "generate_post",
						"description": "Generate social media post text using AI. Returns engaging copy tailored to the channel and topic.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"channel": map[string]string{"type": "string", "description": "Social channel: twitter, instagram, or linkedin"},
								"topic":   map[string]string{"type": "string", "description": "What the post is about"},
								"style":   map[string]string{"type": "string", "description": "Tone/style (optional, e.g. fun, professional)"},
							},
							"required": []string{"topic"},
						},
					},
					{
						"name":        "generate_image",
						"description": "Generate an AI image for a social media post. Returns an image URL/description.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"topic": map[string]string{"type": "string", "description": "What the image should depict"},
								"style": map[string]string{"type": "string", "description": "Visual style (optional, e.g. warm, minimalist)"},
							},
							"required": []string{"topic"},
						},
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
