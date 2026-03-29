// MCP server for file fetching, deduplication, and CSV parsing.
// State in FILES_DATA_DIR: files.json (tracked files with status/hash)
package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

type FileRecord struct {
	URL       string `json:"url"`
	LocalPath string `json:"local_path"`
	Hash      string `json:"hash"`
	Status    string `json:"status"` // pending, processing, processed, error
	Size      int64  `json:"size"`
	FetchedAt string `json:"fetched_at"`
}

var (
	dataDir string
	files   map[string]*FileRecord // keyed by URL
)

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

func save() {
	data, _ := json.MarshalIndent(files, "", "  ")
	os.WriteFile(filepath.Join(dataDir, "files.json"), data, 0644)
}

func load() {
	data, err := os.ReadFile(filepath.Join(dataDir, "files.json"))
	if err != nil {
		return
	}
	json.Unmarshal(data, &files)
}

func handleToolCall(id int64, name string, args map[string]string) {
	switch name {
	case "fetch_file":
		url := args["url"]
		if url == "" {
			respondError(id, -32602, "url is required")
			return
		}

		// Check for duplicate
		if existing, ok := files[url]; ok {
			textResult(id, fmt.Sprintf("DUPLICATE: file already fetched (status=%s, hash=%s)", existing.Status, existing.Hash))
			return
		}

		// Fetch the file (supports file:// and http(s)://)
		var content []byte
		var err error
		if strings.HasPrefix(url, "file://") {
			localPath := strings.TrimPrefix(url, "file://")
			content, err = os.ReadFile(localPath)
		} else {
			var resp *http.Response
			resp, err = http.Get(url)
			if err == nil {
				defer resp.Body.Close()
				if resp.StatusCode != 200 {
					err = fmt.Errorf("HTTP %d", resp.StatusCode)
				} else {
					content, err = io.ReadAll(io.LimitReader(resp.Body, 50*1024*1024))
				}
			}
		}
		if err != nil {
			files[url] = &FileRecord{URL: url, Status: "error", FetchedAt: time.Now().UTC().Format(time.RFC3339)}
			save()
			textResult(id, fmt.Sprintf("ERROR: failed to fetch: %v", err))
			return
		}

		// Save locally
		hash := fmt.Sprintf("%x", sha256.Sum256(content))
		localName := fmt.Sprintf("%s_%s", hash[:12], filepath.Base(url))
		localPath := filepath.Join(dataDir, localName)
		os.WriteFile(localPath, content, 0644)

		files[url] = &FileRecord{
			URL:       url,
			LocalPath: localPath,
			Hash:      hash,
			Status:    "pending",
			Size:      int64(len(content)),
			FetchedAt: time.Now().UTC().Format(time.RFC3339),
		}
		save()
		textResult(id, fmt.Sprintf("OK: fetched %d bytes, hash=%s, path=%s", len(content), hash[:12], localPath))

	case "read_csv":
		path := args["path"]
		if path == "" {
			respondError(id, -32602, "path is required")
			return
		}
		f, err := os.Open(path)
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		defer f.Close()

		reader := csv.NewReader(f)
		records, err := reader.ReadAll()
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: failed to parse CSV: %v", err))
			return
		}
		if len(records) < 2 {
			textResult(id, "CSV has no data rows (only header or empty)")
			return
		}

		// Convert to array of objects using header row
		headers := records[0]
		var rows []map[string]string
		for _, record := range records[1:] {
			row := make(map[string]string)
			for j, val := range record {
				if j < len(headers) {
					row[headers[j]] = val
				}
			}
			rows = append(rows, row)
		}
		data, _ := json.Marshal(rows)
		textResult(id, string(data))

	case "list_files":
		var result []map[string]string
		for url, rec := range files {
			result = append(result, map[string]string{
				"url": url, "status": rec.Status, "hash": rec.Hash,
				"path": rec.LocalPath, "fetched_at": rec.FetchedAt,
			})
		}
		data, _ := json.Marshal(result)
		textResult(id, string(data))

	case "file_status":
		url := args["url"]
		status := args["status"]
		if url == "" || status == "" {
			respondError(id, -32602, "url and status are required")
			return
		}
		rec, ok := files[url]
		if !ok {
			textResult(id, fmt.Sprintf("ERROR: file not found: %s", url))
			return
		}
		rec.Status = status
		save()
		textResult(id, fmt.Sprintf("OK: %s → %s", url, status))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	dataDir = os.Getenv("FILES_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
	files = make(map[string]*FileRecord)
	load()

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
				"serverInfo":     map[string]string{"name": "files", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "fetch_file",
						"description": "Fetch a file from a URL (http/https/file://). Returns local path and hash. Rejects duplicates.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"url": map[string]string{"type": "string", "description": "URL to fetch (http://, https://, or file://)"},
							},
							"required": []string{"url"},
						},
					},
					{
						"name":        "read_csv",
						"description": "Read and parse a CSV file. Returns JSON array of row objects using the header row as keys.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"path": map[string]string{"type": "string", "description": "Local file path (from fetch_file result)"},
							},
							"required": []string{"path"},
						},
					},
					{
						"name":        "list_files",
						"description": "List all fetched files with their status and metadata.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "file_status",
						"description": "Update the status of a fetched file (pending, processing, processed, error).",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"url":    map[string]string{"type": "string", "description": "File URL"},
								"status": map[string]string{"type": "string", "description": "New status: pending, processing, processed, error"},
							},
							"required": []string{"url", "status"},
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
