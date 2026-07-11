package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
)

// TestIntegration_Kimi_StreamShape inspects the raw SSE stream returned
// by kimi-k2.6 on Fireworks vs OpenCode Go side-by-side. The point is
// not to assert anything — it's to print a per-chunk breakdown so we
// can tell whether the two endpoints emit reasoning_content, content
// and tool_calls in the same shape, or whether one of them strips a
// field we depend on (the "no thoughts shown" symptom on instance
// #110 is the kind of thing this surfaces).
//
// Two probes per endpoint:
//
//	"freeform" — a question that should make a reasoning model emit
//	             reasoning_content tokens before the visible answer.
//	"tooluse"  — same prompt + a tool that satisfies the answer; we
//	             want to see whether the model emits reasoning around
//	             the tool call (Fireworks Kimi does; some gateways
//	             drop reasoning when tools are present).
//
// Skips automatically when keys aren't set, so CI without credentials
// is silent.
func TestIntegration_Kimi_StreamShape(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test, skipping in -short")
	}
	if os.Getenv("RUN_LLM_INTEGRATION_TESTS") != "1" {
		t.Skip("set RUN_LLM_INTEGRATION_TESTS=1 to run paid live-provider integration tests")
	}

	type endpoint struct {
		name   string
		envVar string
		url    string
		model  string
	}
	endpoints := []endpoint{
		{"fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1/chat/completions", "accounts/fireworks/models/kimi-k2p6"},
		{"opencode-go", "OPENCODE_GO_API_KEY", "https://opencode.ai/zen/go/v1/chat/completions", "kimi-k2.6"},
	}

	probes := []struct {
		label    string
		messages []map[string]any
		tools    []map[string]any
	}{
		{
			label: "freeform",
			messages: []map[string]any{
				{"role": "system", "content": "Think briefly before you answer."},
				{"role": "user", "content": "What is 17 * 23? Show your reasoning, then give the final number on its own line."},
			},
		},
		{
			label: "tooluse",
			messages: []map[string]any{
				{"role": "system", "content": "You are an agent. Use the calc tool whenever a user asks for arithmetic. Do not compute it yourself."},
				{"role": "user", "content": "What is 17 * 23?"},
			},
			tools: []map[string]any{{
				"type": "function",
				"function": map[string]any{
					"name":        "calc",
					"description": "Compute an arithmetic expression. Returns the numeric result.",
					"parameters": map[string]any{
						"type":     "object",
						"required": []string{"expression"},
						"properties": map[string]any{
							"expression": map[string]any{"type": "string", "description": "Plain arithmetic, e.g. \"17 * 23\""},
						},
					},
				},
			}},
		},
	}

	loadEnv := func() {
		// Load .env without importing godotenv noise into this scope.
		f, err := os.Open(moduleRootEnvFile())
		if err != nil {
			return
		}
		defer f.Close()
		s := bufio.NewScanner(f)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			eq := strings.IndexByte(line, '=')
			if eq <= 0 {
				continue
			}
			k := strings.TrimSpace(line[:eq])
			v := strings.Trim(strings.TrimSpace(line[eq+1:]), `"'`)
			if os.Getenv(k) == "" {
				_ = os.Setenv(k, v)
			}
		}
	}
	loadEnv()

	for _, ep := range endpoints {
		key := os.Getenv(ep.envVar)
		if key == "" {
			t.Logf("[%s] %s not set, skipping", ep.name, ep.envVar)
			continue
		}

		for _, probe := range probes {
			t.Run(ep.name+"/"+probe.label, func(t *testing.T) {
				body := map[string]any{
					"model":    ep.model,
					"messages": probe.messages,
					"stream":   true,
				}
				if len(probe.tools) > 0 {
					body["tools"] = probe.tools
				}
				raw, _ := json.Marshal(body)

				req, err := http.NewRequest("POST", ep.url, bytes.NewReader(raw))
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Authorization", "Bearer "+key)

				resp, err := http.DefaultClient.Do(req)
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				if resp.StatusCode != 200 {
					b, _ := bufio.NewReader(resp.Body).ReadString(0)
					t.Fatalf("status=%d body=%s", resp.StatusCode, b)
				}

				// Per-event counters.
				stats := struct {
					events            int
					contentChunks     int
					contentBytes      int
					reasoningChunks   int
					reasoningBytes    int
					toolCallChunks    int
					toolCallNames     []string
					seenFields        map[string]int
					firstReasoning    string
					firstContent      string
					firstToolCallArgs string
					promptTokens      int
					completionTokens  int
					cachedTokens      int
					reasoningTokens   int
				}{seenFields: map[string]int{}}

				scanner := bufio.NewScanner(resp.Body)
				scanner.Buffer(make([]byte, 1<<20), 1<<20)
				for scanner.Scan() {
					line := scanner.Text()
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					data := strings.TrimPrefix(line, "data: ")
					if data == "[DONE]" {
						break
					}
					stats.events++
					// Inspect raw fields present on the delta — this is
					// where vendor-specific shape diverges.
					var generic map[string]any
					if err := json.Unmarshal([]byte(data), &generic); err == nil {
						walkFields(generic, "", stats.seenFields)
					}

					var event struct {
						Choices []struct {
							Delta struct {
								Content          string            `json:"content"`
								ReasoningContent string            `json:"reasoning_content,omitempty"`
								Reasoning        string            `json:"reasoning,omitempty"`
								Thinking         string            `json:"thinking,omitempty"`
								ToolCalls        []json.RawMessage `json:"tool_calls,omitempty"`
							} `json:"delta"`
						} `json:"choices"`
						Usage *struct {
							PromptTokens        int `json:"prompt_tokens"`
							CompletionTokens    int `json:"completion_tokens"`
							CachedTokens        int `json:"cached_tokens"`
							PromptTokensDetails *struct {
								CachedTokens int `json:"cached_tokens"`
							} `json:"prompt_tokens_details,omitempty"`
							CompletionTokensDetails *struct {
								ReasoningTokens int `json:"reasoning_tokens"`
							} `json:"completion_tokens_details,omitempty"`
						} `json:"usage,omitempty"`
					}
					if err := json.Unmarshal([]byte(data), &event); err != nil {
						continue
					}
					if len(event.Choices) > 0 {
						d := event.Choices[0].Delta
						if d.Content != "" {
							stats.contentChunks++
							stats.contentBytes += len(d.Content)
							if stats.firstContent == "" {
								stats.firstContent = trunc(d.Content, 80)
							}
						}
						// Try every reasoning-style field we've seen in the wild.
						r := firstNonEmpty(d.ReasoningContent, d.Reasoning, d.Thinking)
						if r != "" {
							stats.reasoningChunks++
							stats.reasoningBytes += len(r)
							if stats.firstReasoning == "" {
								stats.firstReasoning = trunc(r, 80)
							}
						}
						for _, tc := range d.ToolCalls {
							stats.toolCallChunks++
							var probe struct {
								Function struct {
									Name      string `json:"name"`
									Arguments string `json:"arguments"`
								} `json:"function"`
							}
							if err := json.Unmarshal(tc, &probe); err == nil {
								if probe.Function.Name != "" {
									stats.toolCallNames = append(stats.toolCallNames, probe.Function.Name)
								}
								if probe.Function.Arguments != "" && stats.firstToolCallArgs == "" {
									stats.firstToolCallArgs = trunc(probe.Function.Arguments, 80)
								}
							}
						}
					}
					if event.Usage != nil {
						stats.promptTokens = event.Usage.PromptTokens
						stats.completionTokens = event.Usage.CompletionTokens
						stats.cachedTokens = event.Usage.CachedTokens
						if event.Usage.PromptTokensDetails != nil && event.Usage.PromptTokensDetails.CachedTokens > 0 {
							stats.cachedTokens = event.Usage.PromptTokensDetails.CachedTokens
						}
						if event.Usage.CompletionTokensDetails != nil {
							stats.reasoningTokens = event.Usage.CompletionTokensDetails.ReasoningTokens
						}
					}
				}
				if err := scanner.Err(); err != nil {
					t.Logf("scan err: %v", err)
				}

				t.Logf("==== %s / %s (model=%s) ====", ep.name, probe.label, ep.model)
				t.Logf("events                 : %d", stats.events)
				t.Logf("content chunks/bytes   : %d / %d", stats.contentChunks, stats.contentBytes)
				t.Logf("reasoning chunks/bytes : %d / %d", stats.reasoningChunks, stats.reasoningBytes)
				t.Logf("tool_call chunks       : %d  names=%v", stats.toolCallChunks, dedup(stats.toolCallNames))
				if stats.firstContent != "" {
					t.Logf("first content sample   : %q", stats.firstContent)
				}
				if stats.firstReasoning != "" {
					t.Logf("first reasoning sample : %q", stats.firstReasoning)
				}
				if stats.firstToolCallArgs != "" {
					t.Logf("first tool args sample : %q", stats.firstToolCallArgs)
				}
				t.Logf("usage tokens           : prompt=%d cached=%d completion=%d reasoning=%d",
					stats.promptTokens, stats.cachedTokens, stats.completionTokens, stats.reasoningTokens)

				// Print the union of every dotted-path field we ever saw
				// in the stream so divergences (a vendor-specific field
				// like `delta.reasoning_signature` or
				// `delta.metadata.thinking`) jump out.
				keys := make([]string, 0, len(stats.seenFields))
				for k := range stats.seenFields {
					keys = append(keys, k)
				}
				sortStrings(keys)
				t.Logf("delta paths seen       : %d", len(keys))
				for _, k := range keys {
					t.Logf("    %s   ×%d", k, stats.seenFields[k])
				}
			})
		}
	}
}

// --- small helpers -----------------------------------------------------

func trunc(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}

func dedup(xs []string) []string {
	seen := map[string]bool{}
	out := xs[:0]
	for _, x := range xs {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}

func sortStrings(xs []string) {
	// stdlib sort.Strings, but inlined to avoid an extra import on a
	// file that's already long.
	for i := 1; i < len(xs); i++ {
		for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
			xs[j-1], xs[j] = xs[j], xs[j-1]
		}
	}
}

// walkFields records the dotted path of every leaf field in a JSON
// document (objects flatten with ".", arrays collapse to "[]"). Used
// to surface the union of provider-specific field names a stream
// emits — gives us a quick "what fields does this gateway send" view.
func walkFields(v any, prefix string, dst map[string]int) {
	switch t := v.(type) {
	case map[string]any:
		for k, vv := range t {
			path := k
			if prefix != "" {
				path = prefix + "." + k
			}
			walkFields(vv, path, dst)
		}
	case []any:
		for _, vv := range t {
			walkFields(vv, prefix+"[]", dst)
		}
	default:
		if prefix != "" {
			dst[prefix]++
		}
	}
}

// moduleRootEnvFile returns the path to the .env adjacent to this
// test source. Avoids depending on godotenv (which the cache test
// already pulls in via getAPIKey, but this test stays self-contained).
func moduleRootEnvFile() string {
	wd, _ := os.Getwd()
	// We expect this file to live in core/, where .env sits.
	return fmt.Sprintf("%s/.env", wd)
}
