package core

import (
	"reflect"
	"strings"
	"testing"
)

func TestGeminiToolParametersInfersNestedObjectTypes(t *testing.T) {
	for _, tc := range []struct {
		name   string
		schema map[string]any
		want   map[string]any
	}{
		{
			name:   "required-only branch",
			schema: map[string]any{"required": []string{"query"}},
			want:   map[string]any{"type": "object", "required": []string{"query"}},
		},
		{
			name:   "properties without type",
			schema: map[string]any{"properties": map[string]any{"required": map[string]any{"type": "string"}}},
			want:   map[string]any{"type": "object", "properties": map[string]any{"required": map[string]any{"type": "string"}}},
		},
		{
			name:   "explicit type preserved",
			schema: map[string]any{"type": "string", "enum": []string{"a", "b"}},
			want:   map[string]any{"type": "string", "enum": []string{"a", "b"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Exercise nested properties, array items, and union branches together.
			schema := map[string]any{"type": "object", "properties": map[string]any{
				"entries": map[string]any{"type": "array", "items": map[string]any{"anyOf": []any{tc.schema}}},
			}}
			before := string(mustJSON(t, schema))
			got := geminiToolParameters(schema)
			items := got["properties"].(map[string]any)["entries"].(map[string]any)["items"].(map[string]any)
			if !reflect.DeepEqual(items["anyOf"].([]any)[0], tc.want) {
				t.Fatalf("nested schema = %#v, want %#v", items["anyOf"].([]any)[0], tc.want)
			}
			if _, ok := items["type"]; ok {
				t.Fatal("union container received an invented type")
			}
			if string(mustJSON(t, schema)) != before {
				t.Fatal("Google conversion mutated the provider-neutral schema")
			}
		})
	}
}

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

func TestGeminiToolParametersDropsUnsupportedJSONSchemaKeywords(t *testing.T) {
	schema := map[string]any{
		"type": "object", "additionalProperties": false,
		"properties": map[string]any{
			"value": map[string]any{"type": "object", "additionalProperties": false,
				"properties": map[string]any{"nested": map[string]any{"type": "array", "items": map[string]any{"type": "object", "additionalProperties": false}}}},
		},
	}
	normalized := geminiToolParameters(schema)
	raw := mustJSON(t, normalized)
	for _, keyword := range []string{"additionalProperties", "$schema", "$defs", "definitions"} {
		if strings.Contains(string(raw), keyword) {
			t.Fatalf("Gemini schema retained unsupported keyword %q: %s", keyword, raw)
		}
	}
	if _, ok := schema["additionalProperties"]; !ok {
		t.Fatal("normalization mutated original schema")
	}
}

func TestParseGeminiStreamReturnsScannerError(t *testing.T) {
	stream := strings.NewReader("data: " + strings.Repeat("x", 1024*1024+1))
	_, err := parseGeminiStream(stream, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "stream read error") {
		t.Fatalf("error = %v", err)
	}
}
