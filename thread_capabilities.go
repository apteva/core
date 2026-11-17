package core

import (
	"fmt"
	"sort"
	"strings"
)

// SpawnCapabilityMode controls how a new thread receives its hard capability
// envelope. Auto preserves the public API distinction between omitted fields
// (inherit) and explicitly supplied tools/MCP fields (legacy strict profile).
type SpawnCapabilityMode uint8

const (
	SpawnCapabilitiesAuto SpawnCapabilityMode = iota
	SpawnCapabilitiesInherit
	SpawnCapabilitiesExplicit
)

type threadCapabilityEnvelope struct {
	tools map[string]bool
	mcps  map[string]bool
}

func resolvedSpawnCapabilityMode(tools []string, opts SpawnOpts) SpawnCapabilityMode {
	if opts.CapabilityMode != SpawnCapabilitiesAuto {
		return opts.CapabilityMode
	}
	if tools != nil || opts.Tools != nil || opts.MCPNames != nil {
		return SpawnCapabilitiesExplicit
	}
	return SpawnCapabilitiesInherit
}

func sortedCapabilityNames(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for name, enabled := range values {
		if enabled {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

func (t *Thinker) serverIsDelegable(server string) bool {
	if t == nil || t.toolIndex == nil {
		return false
	}
	for _, name := range t.toolIndex.ToolsForServer(server) {
		if entry, ok := t.toolIndex.Get(name); ok && !entry.NoSpawn {
			return true
		}
	}
	return false
}

// delegableCapabilities returns the authority a child may receive from this
// thinker. It is deliberately not the current model-visible schema set:
// discovery/loading stays lazy, while this envelope is the hard ceiling.
func (t *Thinker) delegableCapabilities() threadCapabilityEnvelope {
	envelope := threadCapabilityEnvelope{tools: map[string]bool{}, mcps: map[string]bool{}}
	if t == nil || t.registry == nil {
		return envelope
	}

	if t.toolAllowlist == nil {
		// Root/main owns the configured universe, but host-only administration,
		// system tools, and no_spawn MCPs are never delegated implicitly.
		for _, def := range t.registry.AllTools() {
			if def == nil || def.SystemOnly || def.MCP {
				continue
			}
			if def.MainOnly && def.Name != "spawn" {
				continue
			}
			envelope.tools[def.Name] = true
		}
		if t.toolIndex != nil {
			for _, server := range t.toolIndex.Servers() {
				if t.serverIsDelegable(server) {
					envelope.mcps[server] = true
				}
			}
		}
		return envelope
	}

	// A worker can pass on only its own exact grants and MCP scopes. Management
	// companions are reconstructed from spawn rather than copied independently.
	for name, enabled := range t.toolAllowlist {
		if !enabled || name == "kill" || name == "update" || name == "list_threads" {
			continue
		}
		def := t.registry.Get(name)
		if def == nil || def.SystemOnly || (def.MainOnly && def.Name != "spawn") {
			continue
		}
		if def.MCP {
			entry, ok := t.toolIndex.Get(name)
			if ok && entry.NoSpawn {
				continue
			}
		}
		envelope.tools[name] = true
	}
	for server, enabled := range t.toolMCPScopes {
		if enabled && t.serverIsDelegable(server) {
			envelope.mcps[server] = true
		}
	}
	return envelope
}

func (e threadCapabilityEnvelope) authorizesTool(t *Thinker, name string) bool {
	if e.tools[name] {
		return true
	}
	if t == nil || t.toolIndex == nil {
		return false
	}
	entry, ok := t.toolIndex.Get(name)
	return ok && !entry.NoSpawn && e.mcps[entry.Server]
}

func knownMCPServer(t *Thinker, name string) bool {
	if t == nil || t.toolIndex == nil {
		return false
	}
	return len(t.toolIndex.ToolsForServer(name)) > 0
}

func resolveExplicitChildTools(parent *Thinker, requested []string, bypass bool) ([]string, error) {
	if parent == nil || parent.registry == nil {
		return compactStringList(requested), nil
	}
	envelope := parent.delegableCapabilities()
	var out []string
	for _, name := range compactStringList(requested) {
		if bypass {
			// Authenticated API profiles preserve their historical ability to
			// stage names before the corresponding integration is connected.
			out = append(out, name)
			continue
		}
		def := parent.registry.Get(name)
		if def == nil {
			// Preserve the old tolerant behavior for stale/unknown names. They
			// grant nothing and are therefore safe to omit.
			continue
		}
		if def.MCP && parent.toolIndex != nil {
			if entry, ok := parent.toolIndex.Get(name); ok && entry.NoSpawn {
				// no_spawn is an absolute agent-delegation deny. Preserve the
				// historical tolerant behavior by dropping it instead of failing
				// the rest of an otherwise valid legacy profile.
				continue
			}
		}
		if envelope.authorizesTool(parent, name) {
			out = append(out, name)
			continue
		}
		return nil, fmt.Errorf("capability_not_delegable: tool %q is outside the parent capability ceiling", name)
	}
	return out, nil
}

func resolveExplicitChildMCPs(parent *Thinker, requested []string, bypass bool) ([]string, error) {
	if parent == nil || parent.toolIndex == nil {
		return compactStringList(requested), nil
	}
	envelope := parent.delegableCapabilities()
	var out []string
	for _, name := range compactStringList(requested) {
		if bypass {
			out = append(out, name)
			continue
		}
		if !knownMCPServer(parent, name) {
			// A stale saved server name remains non-fatal, matching the previous
			// spawn contract, but cannot become a latent future grant.
			continue
		}
		if !parent.serverIsDelegable(name) {
			continue
		}
		if envelope.mcps[name] {
			out = append(out, name)
			continue
		}
		return nil, fmt.Errorf("capability_not_delegable: MCP server %q is outside the parent capability ceiling", name)
	}
	return out, nil
}

func resolveSpawnCapabilities(parent *Thinker, tools []string, opts SpawnOpts) (resolvedTools, resolvedMCPs []string, inherited bool, err error) {
	mode := resolvedSpawnCapabilityMode(tools, opts)
	if mode == SpawnCapabilitiesInherit {
		envelope := parent.delegableCapabilities()
		return sortedCapabilityNames(envelope.tools), sortedCapabilityNames(envelope.mcps), true, nil
	}

	requestedTools := append([]string(nil), tools...)
	requestedTools = append(requestedTools, opts.Tools...)
	bypassCeiling := opts.BypassCapabilityCeiling || opts.System
	resolvedTools, err = resolveExplicitChildTools(parent, requestedTools, bypassCeiling || opts.BypassNoSpawn)
	if err != nil {
		return nil, nil, false, err
	}
	resolvedMCPs, err = resolveExplicitChildMCPs(parent, opts.MCPNames, bypassCeiling || opts.BypassNoSpawn)
	if err != nil {
		return nil, nil, false, err
	}
	return resolvedTools, resolvedMCPs, false, nil
}

func capabilityModeLabel(inherited bool) string {
	if inherited {
		return "inherited"
	}
	return "explicit"
}

func formatInheritedMCPServers(names []string) string {
	if len(names) == 0 {
		return ""
	}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	var b strings.Builder
	b.WriteString("\n\n[DELEGABLE MCP SERVERS — inherited automatically by new children]\n")
	for _, name := range sorted {
		b.WriteString("- ")
		b.WriteString(name)
		b.WriteByte('\n')
	}
	return b.String()
}
