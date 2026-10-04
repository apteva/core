package core

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// ImageGenerationConfig is an explicit experimental feature gate. Merely
// listing image_generation in builtin_tools never enables it.
type ImageGenerationConfig struct {
	Enabled      bool   `json:"enabled"`
	Model        string `json:"model,omitempty"`
	Size         string `json:"size,omitempty"`
	Quality      string `json:"quality,omitempty"`
	OutputFormat string `json:"output_format,omitempty"`
}

func cloneImageGenerationConfig(c *ImageGenerationConfig) *ImageGenerationConfig {
	if c == nil {
		return nil
	}
	copy := *c
	return &copy
}

func (p *OpenAINativeProvider) configureImageGeneration(c *ImageGenerationConfig) error {
	if c != nil && c.Enabled {
		if p.Name() != "openai" && p.Name() != "openai-codex" {
			return fmt.Errorf("provider %q does not support native image_generation", p.Name())
		}
		if c.Model != "" && !strings.HasPrefix(c.Model, "gpt-image-") {
			return fmt.Errorf("image_generation.model must be a GPT Image model")
		}
		switch c.OutputFormat {
		case "", "png", "jpeg", "webp":
		default:
			return fmt.Errorf("invalid image_generation.output_format")
		}
		switch c.Quality {
		case "", "auto", "low", "medium", "high", "xhigh", "max":
		default:
			return fmt.Errorf("invalid image_generation.quality")
		}
	}
	p.imageGeneration = cloneImageGenerationConfig(c)
	return nil
}

func (p *OpenAINativeProvider) imageGenerationEnabled() bool {
	return p.imageGeneration != nil && p.imageGeneration.Enabled && (p.Name() == "openai" || p.Name() == "openai-codex")
}

func (p *OpenAINativeProvider) imageGenerationTool() map[string]any {
	c := p.imageGeneration
	tool := map[string]any{"type": "image_generation", "action": "generate"}
	for key, value := range map[string]string{"model": c.Model, "size": c.Size, "quality": c.Quality, "output_format": c.OutputFormat} {
		if value != "" {
			tool[key] = value
		}
	}
	return tool
}

func imageGatewayAgentSecret() string {
	if secret := os.Getenv("AGENT_SECRET"); secret != "" {
		return secret
	}
	return os.Getenv("INSTANCE_SECRET")
}

func (p *OpenAINativeProvider) imageResponsesEndpoint(ctx context.Context) (string, error) {
	if !p.imageGenerationEnabled() {
		return p.responsesEndpoint(), nil
	}
	thread, _ := ctx.Value(blobCallerThreadKey{}).(string)
	if thread == "" || os.Getenv("AGENT_ID") == "" || imageGatewayAgentSecret() == "" {
		return "", fmt.Errorf("native image generation requires trusted agent/thread gateway context")
	}
	endpoint := strings.TrimSpace(os.Getenv("APTEVA_IMAGE_GENERATION_GATEWAY_URL"))
	if endpoint == "" {
		serverURL := strings.TrimRight(strings.TrimSpace(os.Getenv("SERVER_URL")), "/")
		if serverURL == "" {
			return "", fmt.Errorf("native image generation requires SERVER_URL")
		}
		endpoint = serverURL + "/api/runtime-image-responses"
	}
	return endpoint, nil
}

func generatedFileParts(files []FileRef) []ContentPart {
	var parts []ContentPart
	for _, file := range files {
		copy := file
		parts = append(parts, ContentPart{Type: "file_ref", FileRef: &copy})
	}
	return parts
}
