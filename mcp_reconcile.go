package core

import (
	"fmt"
	"reflect"
	"strings"
)

// MCPReplacementError reports the connections a configuration change needed
// that could not be made. The previous configuration stays in place.
type MCPReplacementError struct {
	Connected, Attempted int
	Failed               []string
}

func (e *MCPReplacementError) Error() string {
	return fmt.Sprintf("MCP replacement failed: connected %d of %d (failed: %s); previous configuration retained",
		e.Connected, e.Attempted, strings.Join(e.Failed, ", "))
}

// failedMCPNames returns the names in attempted that have no connection in
// connected, in attempted order.
func failedMCPNames(attempted []MCPServerConfig, connected []MCPConn) []string {
	ok := map[string]bool{}
	for _, srv := range connected {
		ok[srv.GetName()] = true
	}
	var failed []string
	for _, c := range attempted {
		if !ok[c.Name] {
			failed = append(failed, c.Name)
		}
	}
	return failed
}

// Prepare replacement transports and their schemas privately. Nothing live is
// removed until every replacement is ready and the configuration commit succeeds.
func (a *APIServer) reconcileMCPTransaction(desired []MCPServerConfig, commit func([]MCPServerConfig) error, finalize ...func(bool, string)) error {
	t := a.thinker
	beforeCatalog, _, _, _, _, err := t.semanticToolCatalog()
	if err != nil {
		return err
	}
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
	var connect, retry []MCPServerConfig
	for _, c := range desired {
		old, known := current[c.Name]
		unchanged := old.URL == c.URL && old.Command == c.Command && old.Transport == c.Transport && reflect.DeepEqual(old.Args, c.Args) && reflect.DeepEqual(old.Env, c.Env)
		if live[c.Name] != nil && (!known || c.NoSpawn || unchanged) {
			continue
		}
		replacement[c.Name] = true
		// A saved connection that is unchanged but not live already failed
		// before this update. Try it once more, but its failure belongs to
		// whatever broke it, not to the change being applied, so it must not
		// block that change.
		if live[c.Name] == nil && known && unchanged {
			retry = append(retry, c)
			continue
		}
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
		return &MCPReplacementError{Connected: len(staged), Attempted: len(connect), Failed: failedMCPNames(connect, staged)}
	}
	if len(retry) > 0 {
		retried := connectAndRegisterMCPWith(retry, stagedRegistry, stagedIndex, t.blobs, connectAnyMCP)
		if failed := failedMCPNames(retry, retried); len(failed) > 0 {
			logMsg("MCP", fmt.Sprintf("still unavailable, kept in configuration: %s", strings.Join(failed, ", ")))
		}
		staged = append(staged, retried...)
	}
	for _, def := range stagedRegistry.tools {
		old := t.registry.Get(def.Name)
		if old == nil {
			continue
		}
		_, retained := want[old.MCPServer]
		if !old.MCP || (retained && !replacement[old.MCPServer] && (old.MCPServer != def.MCPServer || old.MCPLocalName != def.MCPLocalName)) {
			cleanup()
			return fmt.Errorf("MCP tool identity collision for %q; previous configuration retained", def.Name)
		}
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
		t.toolIndex.changedLocked()
		t.toolIndex.rebuildNamesLocked()
		t.toolIndex.mu.Unlock()
		for _, c := range desired {
			t.toolIndex.UpdatePolicy(c.Name, c.NoSpawn, c.ToolLoading)
			t.toolIndex.replaceServerAliases(c.Name, c.ToolAliases)
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
	afterCatalog, _, _, _, _, err := t.semanticToolCatalog()
	if err != nil {
		return err
	}
	if len(finalize) > 0 {
		finalize[0](beforeCatalog != afterCatalog, reason)
	} else {
		t.refreshRuntimePrompt(reason, beforeCatalog != afterCatalog)
	}
	return nil
}
