package core

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
)

type ToolIdentity struct {
	Name       string `json:"name"`
	Server     string `json:"mcp_server,omitempty"`
	Tool       string `json:"mcp_tool,omitempty"`
	SchemaHash string `json:"schema_hash"`
}

func toolIdentity(def *ToolDef) ToolIdentity {
	if def == nil {
		return ToolIdentity{}
	}
	return ToolIdentity{Name: def.Name, Server: def.MCPServer, Tool: def.MCPLocalName, SchemaHash: def.schemaHash}
}

func (t *Thinker) currentToolManifestHash() string {
	t.presentedToolsMu.RLock()
	defer t.presentedToolsMu.RUnlock()
	return t.toolManifestHash
}

func (t *Thinker) resolveToolCall(call *toolCall) {
	call.prerequisites = t.requiredActions(call.executionIDs)
	call.prerequisiteNames = map[string]string{}
	for id, action := range call.prerequisites {
		call.prerequisiteNames[id] = t.requiredToolName(action)
	}
	if t.registry == nil {
		return
	}
	call.definition = t.registry.Get(call.Name)
	t.presentedToolsMu.RLock()
	expected, wasPresented := t.presentedIdentities[call.Name]
	expectedDefinition := t.presentedDefinitions[call.Name]
	call.manifestHash = t.toolManifestHash
	t.presentedToolsMu.RUnlock()
	if wasPresented && (expected != toolIdentity(call.definition) || (expectedDefinition != nil && expectedDefinition != call.definition)) {
		call.resolutionError = "capability_changed: this tool's registered identity or schema changed after the model request; rediscover it before calling"
	}
}

func manifestDigest(tools []ToolIdentity) string {
	raw, _ := json.Marshal(tools)
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("%x", sum[:])
}

// Emit the catalog once per mount/policy revision, shared across the main
// thinker and its workers. Per-request manifests reference its schema hashes.
func (t *Thinker) recordToolCatalog() {
	if t.telemetry == nil || t.toolIndex == nil || t.registry == nil {
		return
	}
	ix := t.toolIndex
	ix.mu.Lock()
	if ix.revision == ix.diagnosedRevision {
		ix.mu.Unlock()
		return
	}
	revision := ix.revision
	ix.diagnosedRevision = revision
	entries := append([]IndexEntry(nil), ix.entries...)
	aliases := copyStringMap(ix.aliases)
	ix.mu.Unlock()
	catalog := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		def := t.registry.Get(e.Name)
		if def == nil {
			continue
		}
		catalog = append(catalog, map[string]any{"identity": toolIdentity(def), "description": e.Description, "parameters": def.native.Parameters, "loading": e.LoadMode, "no_spawn": e.NoSpawn})
	}
	t.telemetry.Emit("tool.catalog", t.threadID, map[string]any{"revision": revision, "tools": catalog, "aliases": aliases})
}
