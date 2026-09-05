package core

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

func decodeMCPResult(result mcpCallResult) (ToolResponse, error) {
	out := ToolResponse{IsError: result.IsError}
	var texts []string
	for _, c := range result.Content {
		switch c.Type {
		case "text":
			texts = append(texts, c.Text)
		case "image":
			data, err := base64.StdEncoding.DecodeString(c.Data)
			if err != nil {
				return out, fmt.Errorf("invalid MCP image: %w", err)
			}
			if c.MIMEType == "image/png" && out.Image == nil {
				out.Image = data
			} else {
				out.Parts = append(out.Parts, ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: "data:" + c.MIMEType + ";base64," + c.Data}})
			}
		case "audio":
			if c.MIMEType == "audio/wav" || c.MIMEType == "audio/mpeg" || c.MIMEType == "audio/mp3" {
				format := "mp3"
				if c.MIMEType == "audio/wav" {
					format = "wav"
				}
				out.Parts = append(out.Parts, ContentPart{Type: "input_audio", InputAudio: &InputAudio{Data: c.Data, Format: format}})
			} else {
				out.Parts = append(out.Parts, ContentPart{Type: "audio_url", AudioURL: &AudioURL{URL: "data:" + c.MIMEType + ";base64," + c.Data, MimeType: c.MIMEType}})
			}
		case "resource":
			texts = append(texts, string(c.Resource))
		case "resource_link":
			texts = append(texts, c.URI)
		default:
			// Preserve unsupported structured blocks instead of silently dropping them.
			raw, _ := json.Marshal(c)
			texts = append(texts, string(raw))
		}
	}
	if len(result.StructuredContent) > 0 {
		texts = append(texts, string(result.StructuredContent))
	}
	out.Text = strings.Join(texts, "\n")
	return out, nil
}

func listAllMCPTools(call func(string, any) (json.RawMessage, error)) ([]mcpToolDef, error) {
	var tools []mcpToolDef
	cursor := ""
	seen := map[string]bool{}
	for {
		var params any
		if cursor != "" {
			params = map[string]string{"cursor": cursor}
		}
		raw, err := call("tools/list", params)
		if err != nil {
			return nil, err
		}
		var page mcpToolsListResult
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, err
		}
		tools = append(tools, page.Tools...)
		if page.NextCursor == "" {
			return tools, nil
		}
		if seen[page.NextCursor] || len(tools) > 100000 {
			return nil, fmt.Errorf("invalid MCP pagination")
		}
		cursor = page.NextCursor
		seen[cursor] = true
	}
}

// Stop at the matching JSON-RPC response; a streaming connection may remain
// open or contain unrelated notifications. Bound the entire response size.
func readMCPResponse(resp *http.Response, request []byte) ([]byte, error) {
	const limit = 32 << 20
	reader := io.LimitReader(resp.Body, limit+1)
	if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		raw, err := io.ReadAll(reader)
		if len(raw) > limit {
			return nil, fmt.Errorf("MCP response too large")
		}
		return raw, err
	}
	var req struct {
		ID json.RawMessage `json:"id"`
	}
	_ = json.Unmarshal(request, &req)
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), limit)
	var data []string
	decode := func() ([]byte, bool) {
		raw := []byte(strings.Join(data, "\n"))
		var frame struct {
			ID     json.RawMessage `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  json.RawMessage `json:"error"`
		}
		if json.Unmarshal(raw, &frame) == nil && string(frame.ID) == string(req.ID) && (len(frame.Result) > 0 || len(frame.Error) > 0) {
			return raw, true
		}
		return nil, false
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if raw, ok := decode(); ok {
				resp.Header.Set("Content-Type", "application/json")
				return raw, nil
			}
			data = nil
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if raw, ok := decode(); ok {
		resp.Header.Set("Content-Type", "application/json")
		return raw, nil
	}
	return nil, fmt.Errorf("MCP stream ended without matching response")
}
