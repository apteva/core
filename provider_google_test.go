package core

import (
	"strings"
	"testing"
)

func TestGeminiToolParametersAddsArrayItems(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tags": map[string]any{
				"type": "array",
			},
			"filters": map[string]any{
				"type":  []any{"array", "null"},
				"items": map[string]any{},
			},
		},
	}

	normalized := geminiToolParameters(schema)
	props := normalized["properties"].(map[string]any)

	cases := map[string]string{
		"tags":    "string",
		"filters": "object",
	}
	for key, wantType := range cases {
		prop := props[key].(map[string]any)
		items, ok := prop["items"].(map[string]any)
		if !ok {
			t.Fatalf("%s.items missing or wrong type: %#v", key, prop["items"])
		}
		if items["type"] != wantType {
			t.Fatalf("%s.items.type = %#v, want %s", key, items["type"], wantType)
		}
	}

	originalTags := schema["properties"].(map[string]any)["tags"].(map[string]any)
	if _, ok := originalTags["items"]; ok {
		t.Fatal("geminiToolParameters mutated the original schema")
	}
}

func TestParseGeminiStreamReturnsScannerError(t *testing.T) {
	stream := strings.NewReader("data: " + strings.Repeat("x", 1024*1024+1))
	_, err := parseGeminiStream(stream, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "stream read error") {
		t.Fatalf("error = %v", err)
	}
}
