package main

import (
	"encoding/json"
	"os"
	"sync"
)

const configFile = "config.json"

type Config struct {
	mu        sync.RWMutex
	path      string
	Directive string `json:"directive"` // user-editable system directive
}

func NewConfig() *Config {
	c := &Config{
		path:      configFile,
		Directive: "You are a general-purpose thinking engine. Observe events, coordinate threads, and help users with their tasks.",
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
