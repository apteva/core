package core

import (
	"encoding/json"
	"strconv"
	"strings"
)

func mcpArgumentsFromStrings(args map[string]string, inputSchema ...map[string]any) map[string]any {
	var props map[string]any
	if len(inputSchema) > 0 && inputSchema[0] != nil {
		if p, ok := inputSchema[0]["properties"].(map[string]any); ok {
			props = p
		}
	}

	out := make(map[string]any, len(args))
	for k, v := range args {
		if prop, ok := props[k]; ok {
			if parsed, ok := coerceMCPArgBySchema(v, prop); ok {
				out[k] = parsed
				continue
			}
		}
		out[k] = parseLegacyMCPArg(v)
	}
	return out
}

func coerceMCPArgBySchema(v string, prop any) (any, bool) {
	p, ok := prop.(map[string]any)
	if !ok {
		return nil, false
	}
	types := schemaTypes(p["type"])
	for _, typ := range types {
		switch typ {
		case "boolean":
			b, ok := parseBoolString(v)
			return b, ok
		case "integer":
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err == nil {
				return n, true
			}
		case "number":
			n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err == nil {
				return n, true
			}
		case "array", "object":
			var parsed any
			if json.Unmarshal([]byte(v), &parsed) == nil {
				return parsed, true
			}
		case "string":
			return v, true
		}
	}
	return nil, false
}

func schemaTypes(raw any) []string {
	switch t := raw.(type) {
	case string:
		return []string{t}
	case []any:
		out := make([]string, 0, len(t))
		for _, v := range t {
			if s, ok := v.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return t
	default:
		return nil
	}
}

func parseLegacyMCPArg(v string) any {
	if len(v) > 0 && (v[0] == '[' || v[0] == '{') {
		var parsed any
		if json.Unmarshal([]byte(v), &parsed) == nil {
			return parsed
		}
	}
	return v
}

func parseBoolString(v string) (bool, bool) {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "true", "1", "yes", "y", "on":
		return true, true
	case "false", "0", "no", "n", "off":
		return false, true
	default:
		return false, false
	}
}
