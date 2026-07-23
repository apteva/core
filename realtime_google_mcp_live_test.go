package core

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode"
)

// inertRealtimeTextProvider supplies the ordinary text-provider half of a
// Thinker. The paid test only exercises its separately registered Gemini Live
// provider, but constructing a real Thinker keeps the thread, MCP, registry,
// telemetry, execution-gate, and session paths identical to production.
type inertRealtimeTextProvider struct{}

func (*inertRealtimeTextProvider) Chat(context.Context, []Message, string, []NativeTool, func(string), func(string), func(string, string, string)) (ChatResponse, error) {
	return ChatResponse{Text: "unused"}, nil
}
func (*inertRealtimeTextProvider) Models() map[ModelTier]string {
	return map[ModelTier]string{
		ModelLarge: "inert", ModelMedium: "inert", ModelSmall: "inert",
	}
}
func (*inertRealtimeTextProvider) Name() string                           { return "inert-live-test" }
func (*inertRealtimeTextProvider) CostPer1M() (float64, float64, float64) { return 0, 0, 0 }
func (*inertRealtimeTextProvider) SupportsNativeTools() bool              { return true }
func (*inertRealtimeTextProvider) AvailableBuiltinTools() []BuiltinTool   { return nil }
func (*inertRealtimeTextProvider) SetBuiltinTools([]string)               {}
func (p *inertRealtimeTextProvider) WithBuiltins([]string) LLMProvider    { return p }

type geminiLiveMCPCall struct {
	Name string
	Args map[string]any
}

func newGeminiLiveMCPServer(t *testing.T, calls *atomic.Int64, received chan<- geminiLiveMCPCall) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read request", http.StatusBadRequest)
			return
		}
		var request struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if request.ID == nil {
			w.WriteHeader(http.StatusAccepted)
			return
		}

		var result any
		switch request.Method {
		case "initialize":
			result = map[string]any{
				"protocolVersion": "2025-03-26",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]string{"name": "gemini-live-probe", "version": "1.0"},
			}
		case "tools/list":
			result = map[string]any{"tools": []map[string]any{{
				"name":        "lookup_code",
				"description": "Look up the deterministic marker for a supplied validation code. Use this whenever the caller asks to validate a code.",
				"inputSchema": map[string]any{
					"type":     "object",
					"required": []string{"code"},
					"properties": map[string]any{
						"code": map[string]any{"type": "string", "description": "Validation code to look up."},
					},
				},
			}}}
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			if err := json.Unmarshal(request.Params, &params); err != nil {
				http.Error(w, "bad tool params", http.StatusBadRequest)
				return
			}
			calls.Add(1)
			select {
			case received <- geminiLiveMCPCall{Name: params.Name, Args: params.Arguments}:
			default:
			}
			result = map[string]any{
				"content": []map[string]any{{
					"type": "text",
					"text": "GEMINI_MCP_OK",
				}},
			}
		default:
			result = map[string]any{}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"jsonrpc": "2.0",
			"id":      request.ID,
			"result":  result,
		})
	})
	return httptest.NewServer(mux)
}

// TestGoogleRealtimeLiveMCPThread is an opt-in paid end-to-end Core smoke.
// It proves that a production Gemini Live session can run as a realtime Core
// thread, call a tool exposed by an actual MCP server, receive the asynchronous
// result through Core's ordinary tool path, continue speaking, and emit PCM.
//
// The complete interaction transcript is printed with `go test -v`.
//
//	RUN_GOOGLE_REALTIME_MCP_SMOKE=1 GOOGLE_API_KEY=... \
//	  go test -v -run TestGoogleRealtimeLiveMCPThread -timeout 3m .
func TestGoogleRealtimeLiveMCPThread(t *testing.T) {
	if os.Getenv("RUN_GOOGLE_REALTIME_MCP_SMOKE") != "1" {
		t.Skip("set RUN_GOOGLE_REALTIME_MCP_SMOKE=1 to run the paid Gemini Live + MCP thread smoke")
	}
	if testing.Short() {
		t.Skip("skipping paid Gemini Live + MCP thread smoke in short mode")
	}
	if strings.TrimSpace(os.Getenv("GOOGLE_API_KEY")) == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	t.Chdir(t.TempDir())
	var mcpCalls atomic.Int64
	mcpReceived := make(chan geminiLiveMCPCall, 2)
	mcpServer := newGeminiLiveMCPServer(t, &mcpCalls, mcpReceived)
	defer mcpServer.Close()

	cfg := &Config{
		path:            filepath.Join(t.TempDir(), "config.json"),
		Directive:       "Coordinate the live integration test.",
		Mode:            ModeAutonomous,
		RealtimeEnabled: true,
		Providers: []ProviderConfig{{
			Name: "google-realtime", Default: true,
		}},
		MCPServers: []MCPServerConfig{{
			Name: "liveprobe", Transport: "http", URL: mcpServer.URL + "/mcp",
		}},
	}
	parent := NewThinker("", &inertRealtimeTextProvider{}, cfg)
	defer parent.Stop()
	defer parent.threads.KillAll()
	defer func() {
		for _, server := range parent.mcpServers {
			server.Close()
		}
	}()

	if parent.pool.RealtimeByName("google-realtime") == nil {
		t.Fatal("Google realtime provider was not registered")
	}
	if parent.registry.Get("liveprobe_lookup_code") == nil {
		t.Fatal("MCP tool was not registered in the Core tool registry")
	}

	const (
		threadID = "gemini-live-mcp"
		userTurn = "Use the connected code lookup for ALPHA-7. After it returns, tell me the exact marker."
	)
	audioIn := make(chan []byte, 4)
	audioOut := make(chan RealtimeAudioFrame, 128)
	audioControl := make(chan string, 8)
	err := parent.threads.SpawnWithOpts(
		threadID,
		`You are a concise live voice agent.
For the caller's first request, call liveprobe_lookup_code exactly once with the requested code.
Wait for the real tool result, then say the returned marker clearly.
Never invent the marker, expose tool mechanics, or call done. Keep the live conversation open.`,
		[]string{"liveprobe_lookup_code"},
		SpawnOpts{
			Realtime:       true,
			Conversation:   true,
			Ephemeral:      true,
			ProviderName:   "google-realtime",
			MCPNames:       []string{"liveprobe"},
			AudioIn:        audioIn,
			AudioOut:       audioOut,
			AudioControl:   audioControl,
			InitialMessage: userTurn,
		},
	)
	if err != nil {
		t.Fatalf("spawn Gemini Live thread: %v", err)
	}
	parent.threads.realtimeBridgeConnected(threadID)

	trace := []string{"USER: " + userTurn}
	cursor := 0
	audioBytes := 0
	toolCallSeen := false
	toolResultSeen := false
	var assistantTurns []string
	var mcpCall geminiLiveMCPCall
	deadline := time.NewTimer(2 * time.Minute)
	defer deadline.Stop()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case frame := <-audioOut:
			audioBytes += len(frame.Audio)
			parent.threads.realtimePlaybackProgress(threadID, frame.ItemID, frame.AudioEndMS)
		case call := <-mcpReceived:
			mcpCall = call
		case <-ticker.C:
			events, next := parent.telemetry.StoredEvents(cursor)
			cursor = next
			for _, event := range events {
				if event.ThreadID != threadID {
					continue
				}
				switch event.Type {
				case "tool.call":
					var data ToolCallData
					if json.Unmarshal(event.Data, &data) == nil && data.Name == "liveprobe_lookup_code" {
						toolCallSeen = true
						encoded, _ := json.Marshal(data.Args)
						trace = append(trace, "GEMINI → MCP: "+data.Name+" "+string(encoded))
					}
				case "tool.result":
					var data ToolResultData
					if json.Unmarshal(event.Data, &data) == nil && data.Name == "liveprobe_lookup_code" {
						toolResultSeen = data.Success && strings.Contains(data.Result, "GEMINI_MCP_OK")
						trace = append(trace, "MCP → GEMINI: "+data.Result)
					}
				case "realtime.assistant":
					var data map[string]any
					if json.Unmarshal(event.Data, &data) == nil {
						if text, _ := data["text"].(string); strings.TrimSpace(text) != "" {
							assistantTurns = append(assistantTurns, strings.TrimSpace(text))
							trace = append(trace, "ASSISTANT: "+strings.TrimSpace(text))
						}
					}
				case "realtime.error":
					var data map[string]any
					_ = json.Unmarshal(event.Data, &data)
					t.Fatalf("Gemini Live error: %v\ntranscript:\n%s", data["error"], strings.Join(trace, "\n"))
				}
			}

			assistantText := strings.Join(assistantTurns, " ")
			// Spoken underscores have no canonical acoustic representation.
			// Gemini's output transcription may render the same marker as
			// GEMINI\_MCP\_OK or GEMINI-MCP-OK. Preserve the raw transcript
			// for diagnostics and compare only its alphanumeric form.
			if toolCallSeen && toolResultSeen && strings.Contains(canonicalSpokenText(assistantText), "GEMINIMCPOK") && audioBytes > 0 {
				if mcpCalls.Load() != 1 {
					t.Fatalf("MCP calls = %d, want exactly 1\ntranscript:\n%s", mcpCalls.Load(), strings.Join(trace, "\n"))
				}
				if mcpCall.Name != "lookup_code" || !strings.EqualFold(strings.TrimSpace(stringValue(mcpCall.Args["code"])), "ALPHA-7") {
					t.Fatalf("MCP received %#v, want lookup_code(ALPHA-7)\ntranscript:\n%s", mcpCall, strings.Join(trace, "\n"))
				}
				t.Logf("GEMINI LIVE + MCP FULL TRANSCRIPT\n%s\nAUDIO OUTPUT: %d PCM bytes", strings.Join(trace, "\n"), audioBytes)
				return
			}
		case <-deadline.C:
			t.Fatalf(
				"timed out: mcp_calls=%d tool_call=%v tool_result=%v audio_bytes=%d assistant=%q\ntranscript:\n%s",
				mcpCalls.Load(), toolCallSeen, toolResultSeen, audioBytes,
				strings.Join(assistantTurns, " "), strings.Join(trace, "\n"),
			)
		}
	}
}

func stringValue(value any) string {
	text, _ := value.(string)
	return text
}

func canonicalSpokenText(text string) string {
	var out strings.Builder
	for _, r := range strings.ToUpper(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			out.WriteRune(r)
		}
	}
	return out.String()
}
