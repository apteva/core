// tools_files.go — File tools for workspace. Delete this file to remove file tools.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const workspaceDir = "workspace"

func init() {
	os.MkdirAll(workspaceDir, 0755)
}

// safePath resolves a path within workspace, preventing escapes.
func safePath(p string) (string, error) {
	cleaned := filepath.Clean(p)
	if filepath.IsAbs(cleaned) || strings.HasPrefix(cleaned, "..") {
		return "", fmt.Errorf("path must be relative and within workspace")
	}
	full := filepath.Join(workspaceDir, cleaned)
	return full, nil
}

func writeFileTool(args map[string]string) string {
	p := args["path"]
	content := args["content"]
	if p == "" {
		return "error: missing path"
	}
	full, err := safePath(p)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	os.MkdirAll(filepath.Dir(full), 0755)
	if err := os.WriteFile(full, []byte(content), 0644); err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), p)
}

func readFileTool(args map[string]string) string {
	p := args["path"]
	if p == "" {
		return "error: missing path"
	}
	full, err := safePath(p)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	data, err := os.ReadFile(full)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	text := string(data)
	if len(text) > maxToolResultLen {
		text = text[:maxToolResultLen] + "\n[truncated]"
	}
	return text
}

func listFilesTool(args map[string]string) string {
	p := args["path"]
	if p == "" {
		p = "."
	}
	full, err := safePath(p)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	entries, err := os.ReadDir(full)
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	var lines []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		lines = append(lines, name)
	}
	if len(lines) == 0 {
		return "(empty)"
	}
	return strings.Join(lines, "\n")
}
