package core

import (
	"fmt"
	"net"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/joho/godotenv"
)

func joinOrNone(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return strings.Join(s, ",")
}

// Version + BuildTime are populated by cmd/apteva-core at startup with the
// values that ldflags injected into its `package main`. Default to "dev"
// so direct `go test` runs (which never call core.SetVersion) still work.
var (
	Version   = "dev"
	BuildTime = "dev"
)

// SetVersion is called by cmd/apteva-core/main.go to forward the
// -ldflags-injected build info from the binary's `package main` into core.
// Keeping the actual ldflag targets in `package main` of cmd/apteva-core
// preserves the existing Dockerfile flags ("-X main.Version=...") unchanged.
func SetVersion(version, buildTime string) {
	Version = version
	BuildTime = buildTime
}

// ContentPart represents a multimodal content block (OpenAI Chat Completions format).
type ContentPart struct {
	Type       string      `json:"type"`                  // "text", "image_url", "input_audio", "audio_url"
	Text       string      `json:"text,omitempty"`        // type=text
	ImageURL   *ImageURL   `json:"image_url,omitempty"`   // type=image_url
	InputAudio *InputAudio `json:"input_audio,omitempty"` // type=input_audio
	AudioURL   *AudioURL   `json:"audio_url,omitempty"`   // type=audio_url
}

type ImageURL struct {
	URL    string `json:"url"`              // https:// or data:image/...;base64,...
	Detail string `json:"detail,omitempty"` // "low", "high", "auto"
}

type InputAudio struct {
	Data   string `json:"data"`   // base64
	Format string `json:"format"` // "wav", "mp3"
}

type AudioURL struct {
	URL      string `json:"url"`                 // https:// or data:audio/...;base64,...
	MimeType string `json:"mime_type,omitempty"` // "audio/mp3", "audio/wav", etc. (auto-detected if empty)
}

type Message struct {
	Role        string           `json:"role"`
	Content     string           `json:"content"`
	Parts       []ContentPart    `json:"parts,omitempty"`        // multimodal content
	ToolCalls   []NativeToolCall `json:"tool_calls,omitempty"`   // assistant messages: structured tool calls
	ToolResults []ToolResult     `json:"tool_results,omitempty"` // user messages: results for prior tool calls
	// RequestContext marks transient retrieval context. It is carried only in
	// the prepared provider request and is never written to session history.
	RequestContext bool `json:"-"`
	// Reasoning is the model's chain-of-thought from the turn that
	// produced this message. We replay it back to the provider on
	// subsequent turns because Moonshot (Kimi K2.6 via OpenCode Go)
	// requires `reasoning_content` to be present on assistant
	// tool_call messages when thinking mode is enabled — without it
	// every multi-turn tool flow 400s. Other providers ignore it.
	Reasoning string `json:"reasoning,omitempty"`
}

// TextContent returns the text content of a message, whether plain Content or from Parts.
func (m Message) TextContent() string {
	if len(m.Parts) == 0 {
		return m.Content
	}
	for _, p := range m.Parts {
		if p.Type == "text" {
			return p.Text
		}
	}
	return m.Content
}

// HasParts returns true if this message has multimodal content.
func (m Message) HasParts() bool {
	return len(m.Parts) > 0
}

type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type Delta struct {
	Content string `json:"content"`
}

type StreamChoice struct {
	Delta Delta `json:"delta"`
}

type PromptTokensDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type StreamEvent struct {
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

// Run is the apteva-core entrypoint. cmd/apteva-core/main.go calls this
// after wiring -ldflags Version/BuildTime via SetVersion.
func Run() {
	godotenv.Load()
	initLogger()
	defer func() {
		if r := recover(); r != nil {
			msg := fmt.Sprintf("top-level panic: %v\n%s", r, string(debug.Stack()))
			logMsg("CRASH", msg)
			fmt.Fprintf(os.Stderr, "%s\n", msg)
			os.Exit(2)
		}
	}()
	installSignalLogger()

	cfg := NewConfig()
	if err := cfg.LoadError(); err != nil {
		fmt.Fprintf(os.Stderr, "invalid core config: %v\n", err)
		return
	}

	provider, err := selectProvider(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	models := provider.Models()
	fmt.Fprintf(os.Stderr, "LLM provider: %s (large=%s, medium=%s, small=%s)\n", provider.Name(), models[ModelLarge], models[ModelMedium], models[ModelSmall])

	// Keep apiKey for backward compat (memory embeddings use it)
	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	thinker := NewThinker(apiKey, provider, cfg)

	// Startup summary
	var mcpNames []string
	for _, m := range cfg.MCPServers {
		mcpNames = append(mcpNames, m.Name)
	}
	var threadNames []string
	for _, t := range cfg.GetThreads() {
		threadNames = append(threadNames, t.ID)
	}
	logMsg("BOOT", fmt.Sprintf("provider=%s mode=%s mcp=[%s] threads=[%s] directive=%d chars",
		provider.Name(), cfg.GetMode(), joinOrNone(mcpNames), joinOrNone(threadNames), len(cfg.GetDirective())))

	go thinker.Run()

	apiPort := os.Getenv("API_PORT")
	if apiPort == "" {
		apiPort = "3210"
	}
	// Always bind loopback. The core API has agent-mutation endpoints
	// (events, config writes) auth'd by APTEVA_API_KEY env, which is
	// fine against same-machine processes but not safe against network
	// attackers — the only legitimate caller is the parent
	// apteva-server's reverse proxy. Operators wanting external core
	// access should reach it through the server's /instances/<id>/*
	// proxy, which has its own auth layer.
	apiAddr := "127.0.0.1:" + apiPort
	apiListener, err := net.Listen("tcp", apiAddr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "core API listen failed on %s: %v\n", apiAddr, err)
		thinker.Stop()
		return
	}
	go func() {
		if err := serveAPI(thinker, apiListener); err != nil && !strings.Contains(err.Error(), "closed network connection") {
			logMsg("CRASH", fmt.Sprintf("core API stopped: %v", err))
			thinker.Stop()
		}
	}()

	// Check for --headless flag or NO_TUI env var
	headless := os.Getenv("NO_TUI") != ""
	for _, arg := range os.Args[1:] {
		if arg == "--headless" {
			headless = true
		}
	}

	if headless {
		fmt.Fprintf(os.Stderr, "apteva-core running headless (API on :%s)\n", apiPort)
		// Start console logger unless suppressed via NO_CONSOLE
		if os.Getenv("NO_CONSOLE") == "" {
			console := NewConsoleLogger(thinker.telemetry)
			go console.Run()
		}
		<-thinker.quit
		logMsg("EXIT", "thinker quit channel closed")
	} else {
		p := tea.NewProgram(newModel(thinker), tea.WithAltScreen())
		if _, err := p.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}

func installSignalLogger() {
	ch := make(chan os.Signal, 4)
	signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP, syscall.SIGQUIT)
	go func() {
		sig := <-ch
		logMsg("SIGNAL", fmt.Sprintf("received %s pid=%d", sig, os.Getpid()))
		if logFile != nil {
			_ = logFile.Sync()
		}
		signal.Reset(sig)
		if s, ok := sig.(syscall.Signal); ok {
			_ = syscall.Kill(os.Getpid(), s)
		}
	}()
}
