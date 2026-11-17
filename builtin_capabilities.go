package core

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// BuiltinToolConfig enables a provider-hosted capability explicitly. Options
// belong to that provider's tool contract, not to a local function tool.
type BuiltinToolConfig struct {
	Enabled bool           `json:"enabled"`
	Options map[string]any `json:"options,omitempty"`
}

// BuiltinCapability describes Core's adapter support, not guaranteed access
// for every model/account. The upstream still validates model availability.
type BuiltinCapability struct {
	Name         string                   `json:"name"`
	Type         string                   `json:"type"`
	Enabled      bool                     `json:"enabled"`
	Experimental bool                     `json:"experimental,omitempty"`
	OptionNames  []string                 `json:"option_names,omitempty"`
	Options      map[string]BuiltinOption `json:"options,omitempty"`
}

type builtinConfigurer interface {
	ConfigureBuiltins(map[string]BuiltinToolConfig) error
	BuiltinCapabilities() []BuiltinCapability
}

func cloneBuiltinConfigs(in map[string]BuiltinToolConfig) map[string]BuiltinToolConfig {
	if in == nil {
		return nil
	}
	out := make(map[string]BuiltinToolConfig, len(in))
	for name, cfg := range in {
		if cfg.Options != nil {
			cfg.Options = cloneJSONValue(cfg.Options).(map[string]any)
		}
		out[name] = cfg
	}
	return out
}

func canonicalBuiltinName(name string) string {
	switch name {
	case "web_search_preview", "google_search":
		return "web_search"
	case "code_interpreter":
		return "code_execution"
	default:
		return name
	}
}

func validateBuiltinConfigs(provider string, in map[string]BuiltinToolConfig, catalog []BuiltinCapability) (map[string]BuiltinToolConfig, error) {
	known := map[string]BuiltinCapability{}
	for _, capability := range catalog {
		known[capability.Name] = capability
	}
	out := map[string]BuiltinToolConfig{}
	for _, name := range sortedBuiltinNames(in) {
		cfg := in[name]
		canonical := canonicalBuiltinName(name)
		capability, ok := known[canonical]
		if !ok {
			return nil, fmt.Errorf("provider %q does not support builtin %q", provider, name)
		}
		if _, exists := out[canonical]; exists {
			return nil, fmt.Errorf("duplicate builtin aliases for %q", canonical)
		}
		allowed := map[string]bool{}
		for _, key := range capability.OptionNames {
			allowed[key] = true
		}
		for key := range cfg.Options {
			if !allowed[key] {
				return nil, fmt.Errorf("builtin %q has unsupported option %q for provider %q", name, key, provider)
			}
		}
		// Ensure options are a JSON document before retaining or sending them.
		if _, err := json.Marshal(cfg.Options); err != nil {
			return nil, fmt.Errorf("builtin %q options must be JSON: %w", name, err)
		}
		out[canonical] = cfg
	}
	if err := validateBuiltinOptionShapes(out, catalog); err != nil {
		return nil, err
	}
	return cloneBuiltinConfigs(out), nil
}

func sortedBuiltinNames(configs map[string]BuiltinToolConfig) []string {
	names := make([]string, 0, len(configs))
	for name := range configs {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Generic entries override the legacy name list, including explicit disables.
func effectiveBuiltinConfigs(legacy []string, configured map[string]BuiltinToolConfig) map[string]BuiltinToolConfig {
	out := map[string]BuiltinToolConfig{}
	for _, name := range legacy {
		out[canonicalBuiltinName(name)] = BuiltinToolConfig{Enabled: true}
	}
	for name, cfg := range configured {
		out[name] = cfg
	}
	return out
}

func narrowBuiltinConfigs(configured map[string]BuiltinToolConfig, names []string) map[string]BuiltinToolConfig {
	out := cloneBuiltinConfigs(configured)
	selected := map[string]bool{}
	for _, name := range names {
		selected[canonicalBuiltinName(name)] = true
	}
	for name, cfg := range out {
		cfg.Enabled = cfg.Enabled && selected[name]
		out[name] = cfg
	}
	return out
}

func builtinWireTool(capability BuiltinCapability, cfg BuiltinToolConfig, named bool) map[string]any {
	tool := map[string]any{"type": capability.Type}
	if named {
		tool["name"] = capability.Name
	}
	for key, value := range cfg.Options {
		tool[key] = cloneJSONValue(value)
	}
	return tool
}

func configureProviderBuiltins(p LLMProvider, pc ProviderConfig) error {
	if len(pc.BuiltinTools) > 0 {
		p.SetBuiltinTools(pc.BuiltinTools)
	}
	if native, ok := p.(*OpenAINativeProvider); ok {
		if err := native.configureImageGeneration(pc.ImageGeneration); err != nil {
			return err
		}
	} else if pc.ImageGeneration != nil && pc.ImageGeneration.Enabled {
		return fmt.Errorf("provider %q does not support native image_generation", pc.Name)
	}
	if len(pc.Builtins) == 0 {
		return nil
	}
	configurable, ok := p.(builtinConfigurer)
	if !ok {
		return fmt.Errorf("provider %q does not support configurable hosted builtins", pc.Name)
	}
	return configurable.ConfigureBuiltins(pc.Builtins)
}

func builtinCapabilityInfo(p LLMProvider) []BuiltinCapability {
	if configurable, ok := p.(builtinConfigurer); ok {
		return configurable.BuiltinCapabilities()
	}
	return nil
}

func openAIBuiltinCatalog() []BuiltinCapability {
	return describeBuiltinOptions([]BuiltinCapability{
		{Name: "web_search", Type: "web_search", OptionNames: []string{"search_context_size", "user_location", "filters", "external_web_access"}},
		{Name: "code_execution", Type: "code_interpreter", OptionNames: []string{"container"}},
		{Name: "file_search", Type: "file_search", OptionNames: []string{"vector_store_ids", "max_num_results", "filters", "ranking_options"}},
		{Name: "image_generation", Type: "image_generation", Experimental: true, OptionNames: []string{"model", "size", "quality", "output_format"}},
	})
}

func (p *OpenAINativeProvider) BuiltinCapabilities() []BuiltinCapability {
	if p.Name() != "openai" && p.Name() != "openai-codex" {
		return nil
	}
	catalog := openAIBuiltinCatalog()
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	for i := range catalog {
		catalog[i].Enabled = effective[catalog[i].Name].Enabled
		if catalog[i].Name == "image_generation" {
			catalog[i].Enabled = p.imageGenerationEnabled()
		}
	}
	return catalog
}

func (p *OpenAINativeProvider) ConfigureBuiltins(in map[string]BuiltinToolConfig) error {
	configs, err := validateBuiltinConfigs(p.Name(), in, p.BuiltinCapabilities())
	if err != nil {
		return err
	}
	var preparedImage *ImageGenerationConfig
	if cfg, exists := configs["image_generation"]; exists {
		imageConfig := ImageGenerationConfig{Enabled: cfg.Enabled}
		for key, value := range cfg.Options {
			text, ok := value.(string)
			if !ok {
				return fmt.Errorf("image_generation option %q must be a string", key)
			}
			switch key {
			case "model":
				imageConfig.Model = text
			case "size":
				imageConfig.Size = text
			case "quality":
				imageConfig.Quality = text
			case "output_format":
				imageConfig.OutputFormat = text
			}
		}
		// Validate on an isolated adapter so a later validation failure cannot
		// partially enable an image capability on the live provider.
		validator := &OpenAINativeProvider{name: p.Name()}
		if err := validator.configureImageGeneration(&imageConfig); err != nil {
			return err
		}
		preparedImage = validator.imageGeneration
	}
	if cfg := configs["file_search"]; cfg.Enabled {
		encoded, _ := json.Marshal(cfg.Options["vector_store_ids"])
		var ids []string
		if json.Unmarshal(encoded, &ids) != nil || len(ids) == 0 {
			return fmt.Errorf("file_search requires non-empty vector_store_ids")
		}
		for _, id := range ids {
			if strings.TrimSpace(id) == "" {
				return fmt.Errorf("file_search vector_store_ids cannot be empty")
			}
		}
	}
	if preparedImage != nil {
		p.imageGeneration = preparedImage
	} else if _, wasConfigured := p.builtinConfigs["image_generation"]; wasConfigured {
		p.imageGeneration = nil
	}
	p.builtinConfigs = configs
	return nil
}

func (p *OpenAINativeProvider) configuredOpenAIBuiltins() []any {
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	var tools []any
	for _, capability := range p.BuiltinCapabilities() {
		if capability.Name == "image_generation" {
			continue
		}
		cfg := effective[capability.Name]
		if !cfg.Enabled {
			continue
		}
		tool := builtinWireTool(capability, cfg, false)
		// Preserve the old preview type when selected through the legacy list.
		if capability.Name == "web_search" {
			if _, generic := p.builtinConfigs[capability.Name]; !generic {
				for _, name := range p.builtinTools {
					if name == "web_search_preview" {
						tool["type"] = name
					}
				}
			}
		}
		if capability.Name == "code_execution" && tool["container"] == nil {
			tool["container"] = map[string]any{"type": "auto"}
		}
		tools = append(tools, tool)
	}
	if p.imageGenerationEnabled() {
		tools = append(tools, p.imageGenerationTool())
	}
	return tools
}

func anthropicBuiltinCatalog() []BuiltinCapability {
	return describeBuiltinOptions([]BuiltinCapability{
		{Name: "code_execution", Type: "code_execution_20250825"},
		{Name: "web_search", Type: "web_search_20250305", OptionNames: []string{"max_uses", "allowed_domains", "blocked_domains", "user_location"}},
	})
}

func (p *AnthropicProvider) ConfigureBuiltins(in map[string]BuiltinToolConfig) error {
	configs, err := validateBuiltinConfigs(p.Name(), in, anthropicBuiltinCatalog())
	if err == nil {
		p.builtinConfigs = configs
	}
	return err
}

func (p *AnthropicProvider) BuiltinCapabilities() []BuiltinCapability {
	catalog := anthropicBuiltinCatalog()
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	for i := range catalog {
		catalog[i].Enabled = effective[catalog[i].Name].Enabled
	}
	return catalog
}

func (p *AnthropicProvider) configuredAnthropicBuiltins() []any {
	var tools []any
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	for _, capability := range p.BuiltinCapabilities() {
		if cfg := effective[capability.Name]; cfg.Enabled {
			tools = append(tools, builtinWireTool(capability, cfg, true))
		}
	}
	return tools
}

func googleBuiltinCatalog() []BuiltinCapability {
	return []BuiltinCapability{{Name: "web_search", Type: "googleSearch"}, {Name: "code_execution", Type: "codeExecution"}}
}

func (p *GoogleProvider) ConfigureBuiltins(in map[string]BuiltinToolConfig) error {
	configs, err := validateBuiltinConfigs(p.Name(), in, googleBuiltinCatalog())
	if err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.builtinConfigs = configs
	return nil
}

func (p *GoogleProvider) BuiltinCapabilities() []BuiltinCapability {
	p.mu.RLock()
	defer p.mu.RUnlock()
	catalog := googleBuiltinCatalog()
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	for i := range catalog {
		catalog[i].Enabled = effective[catalog[i].Name].Enabled
	}
	return catalog
}

func (p *GoogleProvider) configuredGoogleBuiltins() []geminiToolDecl {
	p.mu.RLock()
	defer p.mu.RUnlock()
	effective := effectiveBuiltinConfigs(p.builtinTools, p.builtinConfigs)
	var tools []geminiToolDecl
	if effective["web_search"].Enabled {
		tools = append(tools, geminiToolDecl{GoogleSearch: map[string]any{}})
	}
	if effective["code_execution"].Enabled {
		tools = append(tools, geminiToolDecl{CodeExecution: map[string]any{}})
	}
	return tools
}
