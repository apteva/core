package core

import "fmt"

// AutomaticToolLoadingConfig is the agent's tool-loading configuration.
//
// WorkflowTools is the only field Core acts on: the tools named there stay
// loaded in search mode. They are the operator's explicit choice.
//
// Core once guessed which tools a turn needed by matching the instruction,
// the directive and recalled memory against the catalog. It no longer does:
// the model loads what it needs with search_tools. The remaining fields
// configured that guessing; they are still read and written so stored
// configurations and older Servers keep working, and they have no effect.
type AutomaticToolLoadingConfig struct {
	Enabled         bool     `json:"enabled"`
	WorkflowTools   []string `json:"workflow_tools,omitempty"` // canonical MCP identities, kept loaded
	MaxTools        int      `json:"max_tools,omitempty"`
	MaxSchemaTokens int      `json:"max_schema_tokens,omitempty"`
	IncludeMemory   bool     `json:"include_memory,omitempty"`
}

func validateAutomaticToolLoading(c *AutomaticToolLoadingConfig) error {
	if c == nil {
		return nil
	}
	if c.MaxTools < 0 || c.MaxTools > 20 {
		return fmt.Errorf("automatic_tool_loading.max_tools must be between 0 (default) and 20")
	}
	if c.MaxSchemaTokens < 0 || c.MaxSchemaTokens > 16000 {
		return fmt.Errorf("automatic_tool_loading.max_schema_tokens must be between 0 (default) and 16000")
	}
	if len(c.WorkflowTools) > 20 {
		return fmt.Errorf("automatic_tool_loading.workflow_tools exceeds 20 tools")
	}
	for _, name := range c.WorkflowTools {
		if !validToolName(name) || len(name) > 256 {
			return fmt.Errorf("automatic_tool_loading.workflow_tools requires valid canonical names")
		}
	}
	return nil
}

// defaultAutomaticSchemaTokens is the default reported for the unused
// max_schema_tokens field.
const defaultAutomaticSchemaTokens = 6000

func (c *Config) GetAutomaticToolLoading() AutomaticToolLoadingConfig {
	out := AutomaticToolLoadingConfig{Enabled: true, IncludeMemory: true}
	if c != nil {
		c.mu.RLock()
		if c.AutomaticToolLoading != nil {
			out = *c.AutomaticToolLoading
			out.WorkflowTools = append([]string(nil), out.WorkflowTools...)
		}
		c.mu.RUnlock()
	}
	if out.MaxTools == 0 {
		out.MaxTools = 8
	}
	if out.MaxSchemaTokens == 0 {
		out.MaxSchemaTokens = defaultAutomaticSchemaTokens
	}
	return out
}
