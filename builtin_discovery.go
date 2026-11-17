package core

import (
	"encoding/json"
	"fmt"
	"math"
)

// BuiltinOption describes an option's JSON shape. Account/model availability
// remains an upstream decision; this contract only describes adapter support.
type BuiltinOption struct {
	Type     string   `json:"type"`
	Enum     []string `json:"enum,omitempty"`
	Required bool     `json:"required,omitempty"`
	Default  any      `json:"default,omitempty"`
}

func describeBuiltinOptions(catalog []BuiltinCapability) []BuiltinCapability {
	for i := range catalog {
		c := &catalog[i]
		c.Options = map[string]BuiltinOption{}
		for _, name := range c.OptionNames {
			option := BuiltinOption{Type: "object"}
			switch name {
			case "model", "size":
				option.Type = "string"
			case "quality":
				option = BuiltinOption{Type: "string", Enum: []string{"auto", "low", "medium", "high", "xhigh", "max"}}
			case "output_format":
				option = BuiltinOption{Type: "string", Enum: []string{"png", "jpeg", "webp"}}
			case "search_context_size":
				option = BuiltinOption{Type: "string", Enum: []string{"low", "medium", "high"}}
			case "external_web_access":
				option.Type = "boolean"
			case "max_num_results", "max_uses":
				option.Type = "integer"
			case "vector_store_ids":
				option = BuiltinOption{Type: "string_array", Required: true}
			case "allowed_domains", "blocked_domains":
				option.Type = "string_array"
			case "container":
				option = BuiltinOption{Type: "object_or_string", Default: map[string]any{"type": "auto"}}
			}
			c.Options[name] = option
		}
	}
	return catalog
}

// ProviderBuiltinCatalog is credential-free and safe before any agent exists.
func ProviderBuiltinCatalog() map[string]any {
	return map[string]any{"version": 1, "providers": map[string][]BuiltinCapability{
		"openai": openAIBuiltinCatalog(), "openai-codex": openAIBuiltinCatalog(),
		"anthropic": anthropicBuiltinCatalog(), "google": googleBuiltinCatalog(),
	}}
}

func validateBuiltinOptionShapes(configs map[string]BuiltinToolConfig, catalog []BuiltinCapability) error {
	for _, capability := range catalog {
		cfg, exists := configs[capability.Name]
		if !exists {
			continue
		}
		for key, option := range capability.Options {
			value, present := cfg.Options[key]
			if !present {
				if cfg.Enabled && option.Required {
					return fmt.Errorf("%s requires %s", capability.Name, key)
				}
				continue
			}
			valid := false
			switch option.Type {
			case "string":
				_, valid = value.(string)
			case "boolean":
				_, valid = value.(bool)
			case "integer":
				raw, _ := json.Marshal(value)
				var number float64
				valid = json.Unmarshal(raw, &number) == nil && number >= 1 && math.Trunc(number) == number
			case "object":
				_, valid = value.(map[string]any)
			case "object_or_string":
				_, object := value.(map[string]any)
				_, str := value.(string)
				valid = object || str
			case "string_array":
				raw, _ := json.Marshal(value)
				var strings []string
				valid = value != nil && json.Unmarshal(raw, &strings) == nil
			}
			if !valid {
				return fmt.Errorf("%s.%s must be %s", capability.Name, key, option.Type)
			}
			if len(option.Enum) > 0 {
				found := false
				for _, allowed := range option.Enum {
					if value == allowed {
						found = true
					}
				}
				if !found {
					return fmt.Errorf("invalid %s.%s", capability.Name, key)
				}
			}
		}
	}
	return nil
}

// ValidateProviderBuiltins reuses the adapter validation without credentials,
// requests, configuration files, or other agent startup side effects.
func ValidateProviderBuiltins(name string, configs map[string]BuiltinToolConfig) (map[string]BuiltinToolConfig, error) {
	var adapter builtinConfigurer
	switch name {
	case "openai", "openai-codex":
		adapter = &OpenAINativeProvider{name: name}
	case "anthropic":
		adapter = &AnthropicProvider{}
	case "google":
		adapter = &GoogleProvider{}
	default:
		return nil, fmt.Errorf("provider %q does not support configurable hosted builtins", name)
	}
	canonical, err := validateBuiltinConfigs(name, configs, adapter.BuiltinCapabilities())
	if err != nil {
		return nil, err
	}
	if err = adapter.ConfigureBuiltins(canonical); err != nil {
		return nil, err
	}
	return canonical, nil
}
