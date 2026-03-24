package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/joho/godotenv"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	CachedTokens int `json:"cached_tokens"`
}

type Usage struct {
	PromptTokens        int                 `json:"prompt_tokens"`
	CompletionTokens    int                 `json:"completion_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type StreamEvent struct {
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

func main() {
	godotenv.Load()

	apiKey := os.Getenv("FIREWORKS_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "FIREWORKS_API_KEY not set")
		os.Exit(1)
	}

	thinker := NewThinker(apiKey)
	go thinker.Run()

	p := tea.NewProgram(newModel(thinker), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
