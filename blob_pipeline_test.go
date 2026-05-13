package core

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestIntegration_BlobPipeline exercises the end-to-end blob-handle flow
// with a real LLM (Fireworks/Kimi) and two fake MCP tools:
//
//  1. drive_download_file — returns a _binary JSON envelope.
//  2. audio_transcribe — expects to receive the file bytes as input and
//     returns a transcript.
//
// The test asserts three properties that together prove the pipeline
// works:
//
//   - The tool-result the LLM sees after download is a compact _file
//     handle (no base64 payload leaked into the message history).
//   - When the LLM calls transcribe with a ref, the fake MCP tool
//     observes a full _binary envelope carrying the original bytes.
//   - The final assistant reply contains the transcript.
//
// Skipped in short mode or when FIREWORKS_API_KEY is unset. Uses the
// same conventions as the other Integration_ tests in this package.
func TestIntegration_BlobPipeline(t *testing.T) {
	tp := getTestProvider(t)

	// ---- Fake MCP server state ----
	const audioPayload = "this-is-fake-mp3-bytes-for-the-test"
	const expectedTranscript = "HELLO FROM THE FAKE TRANSCRIBER"

	mcp := newPipelineMCP(t, audioPayload, expectedTranscript)
	blobs := NewBlobStore(4*1024*1024, time.Minute)
	defer blobs.Close()

	// Wrap the two tools through mcpProxyHandler so they go through
	// exactly the same interception code path production MCP tools use.
	downloadH := mcpProxyHandler(mcp, "download_file", blobs)
	transcribeH := mcpProxyHandler(mcp, "transcribe", blobs)

	// ---- Native tool schemas the LLM sees ----
	tools := []NativeTool{
		{
			Name:        "drive_download_file",
			Description: "Download a file by id. Returns a file handle of the form {\"_file\": true, \"ref\": \"blobref://...\", \"mimeType\": ..., \"size\": ...}. Do NOT attempt to read or decode the bytes yourself.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "string", "description": "The file id to download."},
				},
				"required": []string{"id"},
			},
		},
		{
			Name:        "audio_transcribe",
			Description: "Transcribe audio. Pass a file handle ref (\"blobref://...\") from a previous download as the 'audio' argument. Returns the transcript text.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"audio":    map[string]any{"type": "string", "description": "The blobref:// reference returned by drive_download_file."},
					"language": map[string]any{"type": "string", "description": "Language code (e.g. \"en\")."},
				},
				"required": []string{"audio"},
			},
		},
	}

	provider := tp.Provider
	model := provider.Models()[ModelLarge]

	messages := []Message{
		{Role: "system", Content: "You are a tool-using assistant." + blobPromptHint +
			"\n\nWhen the user asks you to download and transcribe a file, call drive_download_file first, then audio_transcribe with the ref from the download result. Do NOT attempt to base64-decode, inline, or copy any bytes yourself — the system injects them automatically. After transcription, reply to the user with the transcript text."},
		{Role: "user", Content: `Download the file with id "audio-42.mp3" from drive and transcribe it in English. Tell me what the transcript says.`},
	}

	// ---- Conversation loop ----
	// We bound iterations so a misbehaving LLM can't spin forever.
	var handleSeenInHistory bool
	var transcribeCalled bool

	for iter := 0; iter < 6; iter++ {
		resp, err := provider.Chat(context.Background(), messages, model, tools, func(string) {}, nil, nil)
		if err != nil {
			t.Fatalf("iter %d Chat error: %v", iter, err)
		}
		t.Logf("iter %d: text=%q toolCalls=%d", iter, truncForBlobLog(resp.Text, 120), len(resp.ToolCalls))

		// No tool calls — LLM is answering the user. Pipeline terminal.
		if len(resp.ToolCalls) == 0 {
			assistantReply := resp.Text
			if !transcribeCalled {
				t.Fatalf("LLM ended without calling transcribe. Final text: %q", assistantReply)
			}
			if !strings.Contains(strings.ToUpper(assistantReply), "HELLO") {
				t.Errorf("expected transcript fragment in reply, got: %q", assistantReply)
			}
			break
		}

		// Append the assistant turn (text + tool calls) as-is.
		messages = append(messages, Message{
			Role: "assistant", Content: resp.Text, Reasoning: resp.Reasoning, ToolCalls: resp.ToolCalls,
		})

		// Dispatch each tool call through the mcpProxyHandler.
		toolResults := make([]ToolResult, 0, len(resp.ToolCalls))
		for _, tc := range resp.ToolCalls {
			var out ToolResponse
			switch tc.Name {
			case "drive_download_file":
				out = downloadH(tc.Args)
				// Sanity: what the LLM is ABOUT to see must be a handle.
				if strings.Contains(out.Text, `"_binary"`) || strings.Contains(out.Text, "base64") {
					t.Fatalf("binary payload leaked into download result (LLM would see it):\n%s", out.Text)
				}
				if strings.Contains(out.Text, `"_file"`) && strings.Contains(out.Text, "blobref://") {
					handleSeenInHistory = true
				}
			case "audio_transcribe":
				transcribeCalled = true
				out = transcribeH(tc.Args)
			default:
				t.Fatalf("unexpected tool call: %s", tc.Name)
			}
			toolResults = append(toolResults, ToolResult{
				CallID: tc.ID, Content: out.Text,
			})
		}
		messages = append(messages, Message{Role: "user", ToolResults: toolResults})
	}

	if !handleSeenInHistory {
		t.Fatal("download never produced a _file handle in the tool result")
	}

	// ---- Assertions on what the fake MCP actually saw ----
	recv := mcp.capturedTranscribeArgs()
	if recv == nil {
		t.Fatal("transcribe handler never invoked by the LLM")
	}
	audioArg := recv["audio"]
	if audioArg == "" {
		t.Fatal("transcribe received no audio arg")
	}
	var env map[string]any
	if err := json.Unmarshal([]byte(audioArg), &env); err != nil {
		t.Fatalf("transcribe's audio arg is not an envelope (no rehydration?): %q\nerr: %v", audioArg, err)
	}
	if env["_binary"] != true {
		t.Fatalf("expected rehydrated _binary envelope, got: %v", env)
	}
	decoded, _ := base64.StdEncoding.DecodeString(toString(env["base64"]))
	if string(decoded) != audioPayload {
		t.Errorf("bytes delivered to transcribe differ from original download:\ngot  %q\nwant %q",
			string(decoded), audioPayload)
	}

	// Double-check: somewhere in the message history the LLM was shown
	// the handle, and never the raw base64 payload.
	for _, m := range messages {
		for _, tr := range m.ToolResults {
			if strings.Contains(tr.Content, audioPayload) {
				t.Errorf("raw payload leaked into LLM message history (ToolResult.Content)")
			}
			if strings.Contains(tr.Content, "base64") && strings.Contains(tr.Content, "_binary") {
				t.Errorf("unconverted _binary envelope reached the message history: %s", tr.Content)
			}
		}
	}
}

// ---------- Test harness: fake MCP with two tools ----------

// pipelineMCP implements MCPConn and responds to the two tools the
// blob-pipeline test invokes. It captures the args passed to
// "transcribe" so the test can assert what the downstream tool
// actually saw after rehydration.
type pipelineMCP struct {
	t            *testing.T
	audioPayload string
	transcript   string

	mu                 sync.Mutex
	lastTranscribeArgs map[string]string
}

func newPipelineMCP(t *testing.T, audio, transcript string) *pipelineMCP {
	return &pipelineMCP{t: t, audioPayload: audio, transcript: transcript}
}

func (m *pipelineMCP) GetName() string                  { return "pipeline-fake" }
func (m *pipelineMCP) ListTools() ([]mcpToolDef, error) { return nil, nil }
func (m *pipelineMCP) Close()                           {}

func (m *pipelineMCP) CallTool(name string, args map[string]string) (string, error) {
	switch name {
	case "download_file":
		// Always returns a _binary envelope regardless of id.
		env := map[string]any{
			"_binary":  true,
			"base64":   base64.StdEncoding.EncodeToString([]byte(m.audioPayload)),
			"mimeType": "audio/mpeg",
			"size":     len(m.audioPayload),
		}
		b, _ := json.Marshal(env)
		return string(b), nil

	case "transcribe":
		m.mu.Lock()
		// Shallow copy; values are strings, safe to share.
		m.lastTranscribeArgs = make(map[string]string, len(args))
		for k, v := range args {
			m.lastTranscribeArgs[k] = v
		}
		m.mu.Unlock()

		// Validate: the audio arg should be a _binary envelope carrying
		// our exact payload. Fail the test loudly if not — that means
		// rehydration didn't run.
		audio := args["audio"]
		var env map[string]any
		if err := json.Unmarshal([]byte(audio), &env); err != nil {
			return "", fmt.Errorf("transcribe received non-envelope audio arg: %q", audio)
		}
		if env["_binary"] != true {
			return "", fmt.Errorf("transcribe audio arg missing _binary: %v", env)
		}
		decoded, err := base64.StdEncoding.DecodeString(toString(env["base64"]))
		if err != nil {
			return "", fmt.Errorf("transcribe base64 decode: %w", err)
		}
		if string(decoded) != m.audioPayload {
			return "", fmt.Errorf("transcribe received wrong bytes: got %q", string(decoded))
		}
		return m.transcript, nil
	}
	return "", fmt.Errorf("unknown tool %q", name)
}

func (m *pipelineMCP) capturedTranscribeArgs() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastTranscribeArgs
}

func truncForBlobLog(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}
