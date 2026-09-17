package core

import (
	"sort"
	"strings"
)

// Normalize only discovery vocabulary. Canonical identities and schemas are
// untouched. Keep this small: broad stemming can conflate unrelated operations.
func discoveryTokens(text string) map[string]int {
	out := map[string]int{}
	for term, count := range indexTokens(text) {
		switch term {
		case "tickets", "files", "comments", "attachments", "fields", "tools", "areas":
			term = strings.TrimSuffix(term, "s")
		case "assignee", "assignees", "assignment", "assignments", "assigned", "assigning", "reassign":
			term = "assign"
		case "get", "fetch", "retrieve", "retrieval", "view":
			term = "read"
		}
		out[term] += count
	}
	return out
}

func discoveryQueryTokens(text string) []string {
	var out []string
	for term := range discoveryTokens(text) {
		out = append(out, term)
	}
	sort.Strings(out)
	return out
}

// Index semantic schema metadata only, not examples/default values, arbitrary
// extensions or $ref URLs. Sorted traversal and budgets bound cost and make
// truncation deterministic even for very large or cyclic in-memory schemas.
func schemaDiscoveryTokens(schema map[string]any) map[string]int {
	out := map[string]int{}
	nodes, bytes := 2048, 32768
	add := func(text string) {
		if len(text) > bytes {
			text = text[:bytes]
		}
		bytes -= len(text)
		for term := range discoveryTokens(text) {
			out[term] = 1
		}
	}
	var walk func(map[string]any, int)
	walk = func(node map[string]any, depth int) {
		if depth > 16 || nodes <= 0 || bytes <= 0 {
			return
		}
		nodes--
		for _, key := range []string{"title", "description"} {
			if text, ok := node[key].(string); ok {
				add(text)
			}
		}
		if text, ok := node["const"].(string); ok {
			add(text)
		}
		switch values := node["enum"].(type) {
		case []any:
			for _, value := range values {
				if bytes <= 0 {
					break
				}
				if text, ok := value.(string); ok {
					add(text)
				}
			}
		case []string:
			for _, text := range values {
				if bytes <= 0 {
					break
				}
				add(text)
			}
		}
		for _, key := range []string{"properties", "$defs", "definitions", "patternProperties"} {
			if fields, ok := node[key].(map[string]any); ok {
				keys := make([]string, 0, len(fields))
				for name := range fields {
					keys = append(keys, name)
				}
				sort.Strings(keys)
				for _, name := range keys {
					if nodes <= 0 || bytes <= 0 {
						break
					}
					if key == "properties" {
						add(name)
					}
					if child, ok := fields[name].(map[string]any); ok {
						walk(child, depth+1)
					}
				}
			}
		}
		for _, key := range []string{"items", "additionalProperties", "contains", "if", "then", "else", "not"} {
			if child, ok := node[key].(map[string]any); ok {
				walk(child, depth+1)
			}
		}
		for _, key := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
			switch children := node[key].(type) {
			case []any:
				for _, child := range children {
					if nodes <= 0 || bytes <= 0 {
						break
					}
					if child, ok := child.(map[string]any); ok {
						walk(child, depth+1)
					}
				}
			case []map[string]any:
				for _, child := range children {
					if nodes <= 0 || bytes <= 0 {
						break
					}
					walk(child, depth+1)
				}
			}
		}
	}
	walk(schema, 0)
	return out
}
