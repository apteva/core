package core

import "testing"

func TestAnthropicMaxTokens(t *testing.T) {
	tests := []struct {
		model string
		want  int
	}{
		{"claude-haiku-4-5-20251001", 64000},
		{"claude-sonnet-4-6", 64000},
		{"claude-opus-4-7", 64000},
		{"claude-fable-5", 64000},
		{"claude-mythos-5-preview", 64000},
		{"claude-3-5-sonnet-20241022", 4096},
		{"claude-3-haiku-20240307", 4096},
		{"unknown-model", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := anthropicMaxTokens(tt.model); got != tt.want {
				t.Fatalf("anthropicMaxTokens(%q) = %d, want %d", tt.model, got, tt.want)
			}
		})
	}
}
