package core

import (
	"fmt"
	"reflect"
)

// Prepare replacement transports and their schemas privately. Nothing live is
// removed until every replacement is ready and the configuration commit succeeds.
func (a *APIServer) reconcileMCPTransaction(desired []MCPServerConfig, commit func([]MCPServerConfig) error) error {
	t := a.thinker
	current := map[string]MCPServerConfig{}
	for _, c := range t.config.GetMCPServers() {
		current[c.Name] = c
	}
	want := map[string]MCPServerConfig{}
	for _, c := range desired {
		if c.Name == "" {
			return fmt.Errorf("MCP name required")
		}
		if _, exists := want[c.Name]; exists {
			return fmt.Errorf("duplicate MCP %q", c.Name)
		}
		if err := validateMCPToolLoading(c); err != nil {
			return err
		}
		want[c.Name] = c
	}
	// Host-owned connections are outside the editable set.
	for name, c := range current {
		if c.NoSpawn {
			if _, ok := want[name]; !ok {
				want[name] = c
				desired = append(desired, c)
			}
		}
	}
	live := map[string]MCPConn{}
	for _, srv := range t.mcpServers {
		live[srv.GetName()] = srv
	}
	replacement := map[string]bool{}
	var connect []MCPServerConfig
	for _, c := range desired {
		old, known := current[c.Name]
		unchanged := old.URL == c.URL && old.Command == c.Command && old.Transport == c.Transport && reflect.DeepEqual(old.Args, c.Args) && reflect.DeepEqual(old.Env, c.Env)
		if live[c.Name] != nil && (!known || c.NoSpawn || unchanged) {
			continue
		}
		replacement[c.Name] = true
		connect = append(connect, c)
	}
	stagedRegistry := &ToolRegistry{tools: map[string]*ToolDef{}}
	stagedIndex := NewToolIndex()
	staged := connectAndRegisterMCP(connect, stagedRegistry, stagedIndex, t.blobs)
	cleanup := func() {
		for _, srv := range staged {
			srv.Close()
		}
	}
	if len(staged) != len(connect) {
		cleanup()
		return fmt.Errorf("MCP replacement failed: connected %d of %d; previous configuration retained", len(staged), len(connect))
	}
	if err := commit(desired); err != nil {
		cleanup()
		return err
	}
	var kept []MCPConn
	for _, srv := range t.mcpServers {
		name := srv.GetName()
		_, wanted := want[name]
		if wanted && !replacement[name] {
			kept = append(kept, srv)
			continue
		}
		t.registry.RemoveByMCPServer(name)
		if t.toolIndex != nil {
			t.toolIndex.Remove(name)
		}
		srv.Close()
	}
	for _, def := range stagedRegistry.tools {
		t.registry.Register(def)
	}
	if t.toolIndex != nil {
		t.toolIndex.mu.Lock()
		t.toolIndex.entries = append(t.toolIndex.entries, stagedIndex.entries...)
		t.toolIndex.mu.Unlock()
		for _, c := range desired {
			t.toolIndex.UpdatePolicy(c.Name, c.NoSpawn, c.ToolLoading)
		}
	}
	t.mcpServers = append(kept, staged...)
	t.mcpCatalog = computeMCPCatalog(t.toolIndex)
	reason := "mcp_configuration_changed"
	for _, c := range desired {
		if !toolLoadingEqual(current[c.Name], c) {
			reason = "tool_loading_policy_changed"
			break
		}
	}
	t.resetPromptCache(reason)
	return nil
}
