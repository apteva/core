package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// registerSystemTools adds system-only tools (for the unconscious thread).
func registerSystemTools(registry *ToolRegistry, memory *MemoryStore) {
	registry.Register(&ToolDef{
		Name:        "memory_scan",
		Description: "Scan all memories with metadata. Returns entries grouped by namespace with recall count, age, and text preview.",
		Syntax:      `[[memory_scan]]`,
		Rules:       "Read-only. Use to assess memory quality before editing or pruning.",
		Core:        true,
		SystemOnly:  true,
		Handler: func(args map[string]string) ToolResponse {
			if memory == nil {
				return ToolResponse{Text: "error: no memory store"}
			}
			memory.mu.RLock()
			defer memory.mu.RUnlock()

			type entryInfo struct {
				Index     int    `json:"index"`
				Namespace string `json:"namespace"`
				Text      string `json:"text"`
				Age       string `json:"age"`
				Session   string `json:"session"`
			}

			var entries []entryInfo
			for i, e := range memory.entries {
				age := time.Since(e.Time)
				ageStr := formatAge(age)
				text := e.Text
				if len(text) > 200 {
					text = text[:200] + "..."
				}
				entries = append(entries, entryInfo{
					Index:     i,
					Namespace: e.Namespace,
					Text:      text,
					Age:       ageStr,
					Session:   e.Session,
				})
			}

			data, _ := json.MarshalIndent(map[string]any{
				"total":   len(entries),
				"entries": entries,
			}, "", "  ")
			return ToolResponse{Text: string(data)}
		},
	})

	registry.Register(&ToolDef{
		Name:        "memory_edit",
		Description: "Edit the text of an existing memory by index. Use to clarify, refine, or merge concepts.",
		Syntax:      `[[memory_edit index="0" text="Updated clearer text"]]`,
		Rules:       "Index from memory_scan. The embedding is recomputed automatically.",
		Core:        true,
		SystemOnly:  true,
		Handler: func(args map[string]string) ToolResponse {
			if memory == nil {
				return ToolResponse{Text: "error: no memory store"}
			}
			indexStr := args["index"]
			newText := args["text"]
			if indexStr == "" || newText == "" {
				return ToolResponse{Text: "error: index and text required"}
			}
			var index int
			fmt.Sscanf(indexStr, "%d", &index)

			memory.mu.Lock()
			if index < 0 || index >= len(memory.entries) {
				memory.mu.Unlock()
				return ToolResponse{Text: fmt.Sprintf("error: index %d out of range (0-%d)", index, len(memory.entries)-1)}
			}
			old := memory.entries[index].Text
			memory.entries[index].Text = newText
			memory.entries[index].Time = time.Now() // refresh timestamp
			memory.mu.Unlock()

			// Recompute embedding
			if emb, err := memory.embed(newText); err == nil {
				memory.mu.Lock()
				if index < len(memory.entries) {
					memory.entries[index].Embedding = emb
				}
				memory.mu.Unlock()
			}

			memory.rewrite()
			return ToolResponse{Text: fmt.Sprintf("edited memory %d: %q → %q", index, truncateForLog(old, 60), truncateForLog(newText, 60))}
		},
	})

	registry.Register(&ToolDef{
		Name:        "memory_prune",
		Description: "Delete memories by index. Use to remove duplicates, stale, or noisy entries.",
		Syntax:      `[[memory_prune indices="0,3,7"]]`,
		Rules:       "Comma-separated indices from memory_scan. Higher indices first to avoid shift issues.",
		Core:        true,
		SystemOnly:  true,
		Handler: func(args map[string]string) ToolResponse {
			if memory == nil {
				return ToolResponse{Text: "error: no memory store"}
			}
			indicesStr := args["indices"]
			if indicesStr == "" {
				return ToolResponse{Text: "error: indices required"}
			}

			// Parse and sort descending (delete from end first)
			var indices []int
			for _, s := range strings.Split(indicesStr, ",") {
				var idx int
				if _, err := fmt.Sscanf(strings.TrimSpace(s), "%d", &idx); err == nil {
					indices = append(indices, idx)
				}
			}
			// Sort descending
			for i := 0; i < len(indices)-1; i++ {
				for j := i + 1; j < len(indices); j++ {
					if indices[j] > indices[i] {
						indices[i], indices[j] = indices[j], indices[i]
					}
				}
			}

			memory.mu.Lock()
			deleted := 0
			for _, idx := range indices {
				if idx >= 0 && idx < len(memory.entries) {
					memory.entries = append(memory.entries[:idx], memory.entries[idx+1:]...)
					deleted++
				}
			}
			memory.mu.Unlock()

			memory.rewrite()
			return ToolResponse{Text: fmt.Sprintf("pruned %d memories (remaining: %d)", deleted, memory.Count())}
		},
	})

	registry.Register(&ToolDef{
		Name:        "skill_write",
		Description: "Write or overwrite a skill file. Skills are loaded into all thread prompts automatically.",
		Syntax:      `[[skill_write name="content-workflow" content="## Content Workflow\n1. List articles\n2. Find gaps\n3. Draft article"]]`,
		Rules:       "Name becomes the filename (skills/{name}.md). Content is markdown. Overwrites if exists.",
		Core:        true,
		SystemOnly:  true,
		Handler: func(args map[string]string) ToolResponse {
			name := args["name"]
			content := args["content"]
			if name == "" || content == "" {
				return ToolResponse{Text: "error: name and content required"}
			}
			// Sanitize name
			name = strings.ReplaceAll(name, "/", "-")
			name = strings.ReplaceAll(name, "..", "")

			dir := "skills"
			os.MkdirAll(dir, 0755)
			path := filepath.Join(dir, name+".md")
			if err := os.WriteFile(path, []byte(content), 0644); err != nil {
				return ToolResponse{Text: fmt.Sprintf("error: %v", err)}
			}
			return ToolResponse{Text: fmt.Sprintf("skill written: %s (%d bytes)", path, len(content))}
		},
	})
}
