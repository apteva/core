package main

import (
	"encoding/json"
	"os"
	"sync"
)

const configFile = "config.json"

type PersistentThread struct {
	ID        string   `json:"id"`
	Directive string   `json:"directive"`
	Tools     []string `json:"tools"`
}

// RunMode controls how tool calls are handled.
type RunMode string

const (
	ModeAutonomous RunMode = "autonomous" // tools execute immediately
	ModeSupervised RunMode = "supervised" // tools require user approval
)

// DefaultAutoApprove lists tools that never need approval (internal reasoning).
var DefaultAutoApprove = []string{"think", "done", "pace", "recall", "remember", "send"}

// ProviderConfig persists the active provider and model selections.
type ProviderConfig struct {
	Name   string            `json:"name"`             // "google", "openai", "anthropic", "fireworks", "ollama"
	Models map[string]string `json:"models,omitempty"` // "large" → model ID, "small" → model ID
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
	AutoApprove []string           `json:"auto_approve,omitempty"`
	Provider    *ProviderConfig    `json:"provider,omitempty"`
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

// GetProvider returns the persisted provider config, or nil.
func (c *Config) GetProvider() *ProviderConfig {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.Provider == nil {
		return nil
	}
	// Return a copy
	cp := &ProviderConfig{Name: c.Provider.Name}
	if c.Provider.Models != nil {
		cp.Models = make(map[string]string)
		for k, v := range c.Provider.Models {
			cp.Models[k] = v
		}
	}
	return cp
}

// SetProvider persists the provider and model selection to config.json.
func (c *Config) SetProvider(pc *ProviderConfig) {
	c.mu.Lock()
	c.Provider = pc
	c.mu.Unlock()
	c.Save()
}

// SetProviderName updates just the provider name, preserving models.
func (c *Config) SetProviderName(name string) {
	c.mu.Lock()
	if c.Provider == nil {
		c.Provider = &ProviderConfig{}
	}
	c.Provider.Name = name
	c.mu.Unlock()
	c.Save()
}

// SetProviderModel updates a single model tier in the config.
func (c *Config) SetProviderModel(tier string, modelID string) {
	c.mu.Lock()
	if c.Provider == nil {
		c.Provider = &ProviderConfig{}
	}
	if c.Provider.Models == nil {
		c.Provider.Models = make(map[string]string)
	}
	c.Provider.Models[tier] = modelID
	c.mu.Unlock()
	c.Save()
}

// IsAutoApproved returns true if a tool should skip approval in supervised mode.
func (c *Config) IsAutoApproved(toolName string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	list := c.AutoApprove
	if len(list) == 0 {
		list = DefaultAutoApprove
	}
	for _, name := range list {
		if name == toolName {
			return true
		}
	}
	return false
}
