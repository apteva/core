package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

// These are JSON Schemas, not Google's restricted Schema proto. In particular
// required-only branches are legal and retain their parent property constraints.
func googleSchemaRegressionFixtures(t *testing.T) map[string]map[string]any {
	t.Helper()
	fixtures := map[string]string{
		"nested_compositions": `{
			"type":"object", "additionalProperties":false,
			"properties":{"entries":{"type":"array","items":{
				"type":"object",
				"properties":{"query":{"type":"string","minLength":1},"queries":{"type":"array","items":{"type":"string"}}},
				"anyOf":[{"required":["query"]},{"required":["queries"]}]
			}}},
			"required":["entries"]
		}`,
		"refs_and_exclusive_unions": `{
			"type":"object",
			"$defs":{"code":{"type":"string","pattern":"^[A-Z]+$"}},
			"properties":{
				"code":{"$ref":"#/$defs/code"},
				"value":{"oneOf":[{"type":"number","minimum":0},{"type":"integer","maximum":10}]},
				"bounds":{"allOf":[{"type":"number","minimum":0},{"maximum":10}]}
			}
		}`,
		"nullable_and_open_arrays": `{
			"type":"object","properties":{
				"filters":{"type":["array","null"],"items":{}},
				"tags":{"type":"array"},
				"nullable":{"anyOf":[{"type":"string"},{"type":"null"}]},
				"choice":{"enum":[1,true,null]},
				"map":{"type":"object","additionalProperties":{"type":"integer"}}
			}
		}`,
		"schema_keywords_as_data": `{
			"type":"object","properties":{
				"required":{"type":"string"},
				"properties":{"type":"object","default":{"required":["literal"],"anyOf":"literal"}},
				"anyOf":{"type":"string","const":"literal"}
			}
		}`,
	}
	out := map[string]map[string]any{}
	for name, raw := range fixtures {
		var schema map[string]any
		if err := json.Unmarshal([]byte(raw), &schema); err != nil {
			t.Fatal(err)
		}
		out[name] = schema
	}
	return out
}

func TestGeminiToolParametersPreservesJSONSchema(t *testing.T) {
	fixtures := googleSchemaRegressionFixtures(t)
	fixtures["registered_search_tools"] = NewToolRegistry("google-schema").Get("search_tools").native.Parameters
	for name, schema := range fixtures {
		t.Run(name, func(t *testing.T) {
			before := string(mustJSON(t, schema))
			got := geminiToolParameters(schema)
			if !reflect.DeepEqual(got, schema) {
				t.Fatalf("schema constraints changed:\ngot %s\nwant %s", mustJSON(t, got), before)
			}
			// Confirm isolation, including nested maps: Google must not mutate
			// registry schemas shared with OpenAI, Codex, xAI or Grok Build.
			got["properties"].(map[string]any)["google_only"] = map[string]any{"type": "boolean"}
			got["type"] = "string"
			if string(mustJSON(t, schema)) != before {
				t.Fatal("Google conversion aliases the provider-neutral schema")
			}
		})
	}
}

func TestGeminiToolParametersDefaultsOnlyRootObject(t *testing.T) {
	for _, schema := range []map[string]any{nil, {}, {"properties": map[string]any{"value": map[string]any{}}}} {
		before := string(mustJSON(t, schema))
		got := geminiToolParameters(schema)
		if got["type"] != "object" {
			t.Fatalf("function parameters need an object root: %#v", got)
		}
		if props, ok := got["properties"].(map[string]any); ok && len(props["value"].(map[string]any)) != 0 {
			t.Fatal("unconstrained child schema received invented constraints")
		}
		if string(mustJSON(t, schema)) != before {
			t.Fatal("conversion mutated input")
		}
	}
}

func TestGoogleChatAndLivePreserveFullToolSchemasOnWire(t *testing.T) {
	tools := googleFullSchemaSmokeTools(t)
	before := string(mustJSON(t, tools))
	assertDeclarations := func(declarations []any) {
		t.Helper()
		if len(declarations) != len(tools) {
			t.Fatalf("wire tool count = %d, want %d", len(declarations), len(tools))
		}
		for i, raw := range declarations {
			declaration := raw.(map[string]any)
			if _, ok := declaration["parameters"]; ok {
				t.Fatalf("%s still uses Google's restricted Schema proto", tools[i].Name)
			}
			if declaration["name"] != tools[i].Name || string(mustJSON(t, declaration["parametersJsonSchema"])) != string(mustJSON(t, tools[i].Parameters)) {
				t.Fatalf("%s schema changed on the wire: %#v", tools[i].Name, declaration)
			}
		}
	}

	originalClient := llmHTTPClient
	t.Cleanup(func() { llmHTTPClient = originalClient })
	called := false
	llmHTTPClient = &http.Client{Transport: auditRoundTripper(func(r *http.Request) (*http.Response, error) {
		called = true
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		assertDeclarations(request["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any))
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(
			"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"OK\"}]},\"finishReason\":\"STOP\"}]}\n\n")), Request: r}, nil
	})}
	_, err := NewGoogleProvider("test-key").Chat(context.Background(), []Message{{Role: "user", Content: "Hello"}}, "gemini-2.5-flash", tools, nil, nil, nil)
	if err != nil || !called {
		t.Fatalf("Chat request not completed: called=%v, error=%v", called, err)
	}
	setup, err := buildGoogleLiveSetup(RealtimeSessionOpts{Model: "gemini-3.1-flash-live-preview", Tools: tools}, "Kore")
	if err != nil {
		t.Fatal(err)
	}
	var envelope map[string]any
	if err := json.Unmarshal(setup, &envelope); err != nil {
		t.Fatal(err)
	}
	assertDeclarations(envelope["setup"].(map[string]any)["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any))
	if string(mustJSON(t, tools)) != before {
		t.Fatal("Google mutated the schemas used by other providers")
	}
}

func TestParseGeminiStreamReturnsScannerError(t *testing.T) {
	stream := strings.NewReader("data: " + strings.Repeat("x", 1024*1024+1))
	_, err := parseGeminiStream(stream, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "stream read error") {
		t.Fatalf("error = %v", err)
	}
}
