package main

import (
	"encoding/json"
	"os"
	"sync"
)

const configFile = "config.json"

type PersistentThread struct {
	ID        string   `json:"id"`
	ParentID  string   `json:"parent_id,omitempty"` // empty = child of main
	Depth     int      `json:"depth,omitempty"`      // 0 = main's direct child
	System    bool     `json:"system,omitempty"`     // system thread (can't be killed by LLM)
	Directive string   `json:"directive"`
	Tools     []string `json:"tools"`
	MCPNames  []string `json:"mcp_names,omitempty"` // MCP servers to connect on respawn
}

// RunMode controls the agent's safety behavior via system prompt guidance.
type RunMode string

const (
	ModeAutonomous RunMode = "autonomous" // agent operates freely, asks when it thinks it should
	ModeCautious   RunMode = "cautious"   // agent asks before destructive/external actions
	ModeLearn      RunMode = "learn"      // agent actively asks about new tool types, builds safety profile
)

// ProviderConfig persists a provider and its model selections.
type ProviderConfig struct {
	Name         string            `json:"name"`                    // "google", "openai", "anthropic", "fireworks", "ollama"
	Default      bool              `json:"default,omitempty"`       // true = default provider (first match wins)
	Models       map[string]string `json:"models,omitempty"`        // "large" → model ID, "medium" → ..., "small" → ...
	BuiltinTools []string          `json:"builtin_tools,omitempty"` // e.g. ["code_execution"]
}

// ComputerConfig holds the configuration for a computer use environment.
type ComputerConfig struct {
	Type      string `json:"type"`                 // "browserbase", "service"
	URL       string `json:"url,omitempty"`        // for "service" type
	APIKey    string `json:"api_key,omitempty"`    // for "browserbase"
	ProjectID string `json:"project_id,omitempty"` // for "browserbase"
	Width     int    `json:"width,omitempty"`      // display width (default 1280)
	Height    int    `json:"height,omitempty"`     // display height (default 800)
}

type Config struct {
	mu          sync.RWMutex
	path        string
	Directive   string             `json:"directive"`
	Mode        RunMode            `json:"mode,omitempty"`
	Unconscious bool               `json:"unconscious,omitempty"` // enable background memory consolidation thread
	Providers   []ProviderConfig   `json:"providers,omitempty"`   // multi-provider pool
	Provider    *ProviderConfig    `json:"provider,omitempty"`    // legacy single-provider (auto-migrated to Providers on load)
	Computer    *ComputerConfig    `json:"computer,omitempty"`
	Threads     []PersistentThread `json:"threads,omitempty"`
	MCPServers  []MCPServerConfig  `json:"mcp_servers,omitempty"`
}

func NewConfig() *Config {
	c := &Config{
		path:      configFile,
		Directive: "Idle. Waiting for configuration via directive.",
	}
	c.load()
	return c
}

func (c *Config) load() {
	data, err := os.ReadFile(c.path)
	if err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	json.Unmarshal(data, c)

	// Migrate legacy single Provider → Providers array
	if c.Provider != nil && c.Provider.Name != "" && len(c.Providers) == 0 {
		c.Provider.Default = true
		c.Providers = []ProviderConfig{*c.Provider}
		c.Provider = nil
	}
}

func (c *Config) Save() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.path, data, 0644)
}

func (c *Config) GetDirective() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.Directive
}

func (c *Config) SetDirective(d string) {
	c.mu.Lock()
	c.Directive = d
	c.mu.Unlock()
	c.Save()
}

func (c *Config) ClearThreads() {
	c.mu.Lock()
	c.Threads = nil
	c.mu.Unlock()
	c.Save()
}

func (c *Config) SaveThread(pt PersistentThread) {
	c.mu.Lock()
	// Update if exists, otherwise append
	found := false
	for i, t := range c.Threads {
		if t.ID == pt.ID {
			c.Threads[i] = pt
			found = true
			break
		}
	}
	if !found {
		c.Threads = append(c.Threads, pt)
	}
	c.mu.Unlock()
	c.Save()
}

func (c *Config) RemoveThread(id string) {
	c.mu.Lock()
	for i, t := range c.Threads {
		if t.ID == id {
			c.Threads = append(c.Threads[:i], c.Threads[i+1:]...)
			break
		}
	}
	c.mu.Unlock()
	c.Save()
}

func (c *Config) SaveMCPServer(cfg MCPServerConfig) {
	c.mu.Lock()
	found := false
	for i, s := range c.MCPServers {
		if s.Name == cfg.Name {
			c.MCPServers[i] = cfg
			found = true
			break
		}
	}
	if !found {
		c.MCPServers = append(c.MCPServers, cfg)
	}
	c.mu.Unlock()
	c.Save()
}

func (c *Config) RemoveMCPServer(name string) {
	c.mu.Lock()
	for i, s := range c.MCPServers {
		if s.Name == name {
			c.MCPServers = append(c.MCPServers[:i], c.MCPServers[i+1:]...)
			break
		}
	}
	c.mu.Unlock()
	c.Save()
}

func (c *Config) GetThreads() []PersistentThread {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]PersistentThread, len(c.Threads))
	copy(out, c.Threads)
	return out
}

func (c *Config) GetMCPServers() []MCPServerConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]MCPServerConfig, len(c.MCPServers))
	copy(out, c.MCPServers)
	return out
}

func (c *Config) GetMode() RunMode {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.Mode == "" {
		return ModeAutonomous
	}
	return c.Mode
}

func (c *Config) SetMode(m RunMode) {
	c.mu.Lock()
	c.Mode = m
	c.mu.Unlock()
	c.Save()
}

// GetProviders returns a copy of the providers list.
func (c *Config) GetProviders() []ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]ProviderConfig, len(c.Providers))
	for i, p := range c.Providers {
		cp := ProviderConfig{Name: p.Name, Default: p.Default, BuiltinTools: p.BuiltinTools}
		if p.Models != nil {
			cp.Models = make(map[string]string)
			for k, v := range p.Models {
				cp.Models[k] = v
			}
		}
		out[i] = cp
	}
	return out
}

// GetDefaultProvider returns the default provider config, or nil.
func (c *Config) GetDefaultProvider() *ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, p := range c.Providers {
		if p.Default {
			cp := p
			return &cp
		}
	}
	if len(c.Providers) > 0 {
		cp := c.Providers[0]
		return &cp
	}
	return nil
}

// GetProvider returns the persisted default provider config, or nil.
// Backward-compatible wrapper around GetDefaultProvider.
func (c *Config) GetProvider() *ProviderConfig {
	return c.GetDefaultProvider()
}

// GetProviderByName returns a provider config by name, or nil.
func (c *Config) GetProviderByName(name string) *ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, p := range c.Providers {
		if p.Name == name {
			cp := p
			return &cp
		}
	}
	return nil
}

// SetProvider adds or updates a provider in the list. If it's the only one, marks it default.
func (c *Config) SetProvider(pc *ProviderConfig) {
	c.mu.Lock()
	found := false
	for i, p := range c.Providers {
		if p.Name == pc.Name {
			c.Providers[i] = *pc
			found = true
			break
		}
	}
	if !found {
		c.Providers = append(c.Providers, *pc)
	}
	// If only one provider, make it default
	if len(c.Providers) == 1 {
		c.Providers[0].Default = true
	}
	c.Provider = nil // clear legacy field
	c.mu.Unlock()
	c.Save()
}

// SetProviderName adds or updates a provider by name with default flag.
func (c *Config) SetProviderName(name string) {
	c.mu.Lock()
	found := false
	for i, p := range c.Providers {
		if p.Name == name {
			found = true
			_ = i
			break
		}
	}
	if !found {
		pc := ProviderConfig{Name: name}
		if len(c.Providers) == 0 {
			pc.Default = true
		}
		c.Providers = append(c.Providers, pc)
	}
	c.Provider = nil
	c.mu.Unlock()
	c.Save()
}

// SetProviderModel updates a single model tier for a provider (default if not specified).
func (c *Config) SetProviderModel(tier string, modelID string) {
	c.mu.Lock()
	if len(c.Providers) == 0 {
		c.Providers = []ProviderConfig{{Name: "unknown", Default: true}}
	}
	// Update the default provider
	for i, p := range c.Providers {
		if p.Default || i == 0 {
			if c.Providers[i].Models == nil {
				c.Providers[i].Models = make(map[string]string)
			}
			c.Providers[i].Models[tier] = modelID
			break
		}
	}
	c.Provider = nil
	c.mu.Unlock()
	c.Save()
}

// SetDefaultProvider marks a provider as default (clears default on others).
func (c *Config) SetDefaultProvider(name string) {
	c.mu.Lock()
	for i := range c.Providers {
		c.Providers[i].Default = c.Providers[i].Name == name
	}
	c.mu.Unlock()
	c.Save()
}

// RemoveProvider removes a provider by name.
func (c *Config) RemoveProvider(name string) {
	c.mu.Lock()
	for i, p := range c.Providers {
		if p.Name == name {
			c.Providers = append(c.Providers[:i], c.Providers[i+1:]...)
			break
		}
	}
	c.mu.Unlock()
	c.Save()
}

