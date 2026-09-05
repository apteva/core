// MCP server for file-based code read/write and test execution.
// Operates on CODEBASE_DIR. Runs test.sh for testing.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

var codebaseDir string

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

// safePath resolves a path relative to codebaseDir and ensures it doesn't escape.
func safePath(p string) (string, error) {
	abs := filepath.Join(codebaseDir, p)
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", err
	}
	base, _ := filepath.Abs(codebaseDir)
	rel, relErr := filepath.Rel(base, abs)
	if relErr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes codebase directory")
	}
	root, err := os.OpenRoot(base)
	if err != nil {
		return "", err
	}
	defer root.Close()
	if _, err := root.Stat(rel); err != nil && !os.IsNotExist(err) {
		return "", err
	}
	return abs, nil
}

func handleToolCall(id int64, name string, args map[string]string) {
	root, err := os.OpenRoot(codebaseDir)
	if err != nil {
		textResult(id, fmt.Sprintf("ERROR: %v", err))
		return
	}
	defer root.Close()
	relative := func(path string) string {
		base, _ := filepath.Abs(codebaseDir)
		rel, _ := filepath.Rel(base, path)
		return rel
	}
	switch name {
	case "read_file":
		path := args["path"]
		if path == "" {
			respondError(id, -32602, "path is required")
			return
		}
		abs, err := safePath(path)
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		data, err := root.ReadFile(relative(abs))
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		textResult(id, string(data))

	case "write_file":
		path := args["path"]
		content := args["content"]
		if path == "" {
			respondError(id, -32602, "path is required")
			return
		}
		abs, err := safePath(path)
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		if err := root.MkdirAll(filepath.Dir(relative(abs)), 0755); err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		if err := root.WriteFile(relative(abs), []byte(content), 0644); err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		textResult(id, fmt.Sprintf("OK: wrote %d bytes to %s", len(content), path))

	case "list_files":
		dir := args["dir"]
		if dir == "" {
			dir = "."
		}
		abs, err := safePath(dir)
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}
		var files []map[string]any
		filepath.Walk(abs, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			rel, _ := filepath.Rel(abs, path)
			if rel == "." {
				return nil
			}
			// Skip hidden dirs
			if info.IsDir() && strings.HasPrefix(info.Name(), ".") {
				return filepath.SkipDir
			}
			if !info.IsDir() {
				files = append(files, map[string]any{
					"path": filepath.Join(dir, rel),
					"size": info.Size(),
				})
			}
			return nil
		})
		data, _ := json.Marshal(files)
		textResult(id, string(data))

	case "run_tests":
		// Run test.sh in codebaseDir
		testScript := filepath.Join(codebaseDir, "test.sh")
		if _, err := os.Stat(testScript); os.IsNotExist(err) {
			textResult(id, "ERROR: test.sh not found in codebase directory")
			return
		}
		cmd := exec.Command("bash", "test.sh")
		cmd.Dir = codebaseDir
		output, err := cmd.CombinedOutput()
		passed := err == nil
		result := map[string]any{
			"passed": passed,
			"output": string(output),
		}
		data, _ := json.Marshal(result)
		textResult(id, string(data))

	case "search":
		pattern := args["pattern"]
		dir := args["dir"]
		if pattern == "" {
			respondError(id, -32602, "pattern is required")
			return
		}
		if dir == "" {
			dir = "."
		}
		abs, err := safePath(dir)
		if err != nil {
			textResult(id, fmt.Sprintf("ERROR: %v", err))
			return
		}

		var matches []map[string]any
		filepath.Walk(abs, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}
			if strings.HasPrefix(info.Name(), ".") {
				return nil
			}
			data, err := root.ReadFile(relative(path))
			if err != nil {
				return nil
			}
			lines := strings.Split(string(data), "\n")
			rel, _ := filepath.Rel(abs, path)
			for i, line := range lines {
				if strings.Contains(line, pattern) {
					matches = append(matches, map[string]any{
						"file": filepath.Join(dir, rel),
						"line": i + 1,
						"text": line,
					})
				}
			}
			return nil
		})
		data, _ := json.Marshal(matches)
		textResult(id, string(data))

	default:
		respondError(id, -32601, fmt.Sprintf("unknown tool: %s", name))
	}
}

func main() {
	codebaseDir = os.Getenv("CODEBASE_DIR")
	if codebaseDir == "" {
		codebaseDir = "."
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
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]string{"name": "codebase", "version": "1.0.0"},
			})
		case "tools/list":
			respond(id, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "read_file",
						"description": "Read a file from the codebase. Path is relative to the project root.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"path": map[string]string{"type": "string", "description": "Relative file path"},
							},
							"required": []string{"path"},
						},
					},
					{
						"name":        "write_file",
						"description": "Write or overwrite a file in the codebase. Creates parent directories as needed.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"path":    map[string]string{"type": "string", "description": "Relative file path"},
								"content": map[string]string{"type": "string", "description": "Full file content to write"},
							},
							"required": []string{"path", "content"},
						},
					},
					{
						"name":        "list_files",
						"description": "List all files in a directory (recursive). Shows paths and sizes.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"dir": map[string]string{"type": "string", "description": "Directory to list (default: project root)"},
							},
						},
					},
					{
						"name":        "run_tests",
						"description": "Run the project test suite (executes test.sh). Returns pass/fail and full output.",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
					},
					{
						"name":        "search",
						"description": "Search for a text pattern across all files. Returns matching lines with file paths and line numbers.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"pattern": map[string]string{"type": "string", "description": "Text to search for"},
								"dir":     map[string]string{"type": "string", "description": "Directory to search (default: project root)"},
							},
							"required": []string{"pattern"},
						},
					},
				},
			})
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				respondError(id, -32602, "invalid params")
				continue
			}
			// Convert all argument values to strings
			args := make(map[string]string)
			for k, v := range params.Arguments {
				switch val := v.(type) {
				case string:
					args[k] = val
				default:
					data, _ := json.Marshal(val)
					args[k] = string(data)
				}
			}
			handleToolCall(id, params.Name, args)
		default:
			respondError(id, -32601, fmt.Sprintf("unknown method: %s", req.Method))
		}
	}
}
