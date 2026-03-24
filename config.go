package main

import (
	"encoding/json"
	"os"
	"sync"
)

const configFile = "config.json"

type PersistentThread struct {
	ID       string   `json:"id"`
	Directive string   `json:"directive"`
	Tools    []string `json:"tools"`
	Thinking bool     `json:"thinking"`
}

type Config struct {
	mu         sync.RWMutex
	path       string
	Directive  string             `json:"directive"`
	Threads    []PersistentThread `json:"threads,omitempty"`
	MCPServers []MCPServerConfig  `json:"mcp_servers,omitempty"`
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

func (c *Config) GetThreads() []PersistentThread {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]PersistentThread, len(c.Threads))
	copy(out, c.Threads)
	return out
}
