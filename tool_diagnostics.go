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

// A semantic content identity is independent of transport/activation revisions.
func (t *Thinker) semanticToolCatalog() (string, json.RawMessage, int, int, uint64, error) {
	if t.toolIndex == nil || t.registry == nil {
		return "", nil, 0, 0, 0, nil
	}
	ix := t.toolIndex
	ix.mu.RLock()
	revision := ix.revision
	entries := append([]IndexEntry(nil), ix.entries...)
	aliases := copyStringMap(ix.aliases)
	ix.mu.RUnlock()
	catalog := make([]map[string]any, 0, len(entries))
	for _, e := range entries {
		def := t.registry.Get(e.Name)
		if def == nil {
			continue
		}
		catalog = append(catalog, map[string]any{"identity": toolIdentity(def), "description": e.Description, "parameters": def.native.Parameters, "loading": normalizeToolLoadMode(e.LoadMode), "no_spawn": e.NoSpawn, "access": e.Access})
	}
	raw, err := json.Marshal(map[string]any{"tools": catalog, "aliases": aliases})
	if err != nil {
		return "", nil, 0, 0, revision, err
	}
	hash, body, tools, aliasCount, err := canonicalToolCatalog(raw)
	return hash, body, tools, aliasCount, revision, err
}

func (t *Thinker) recordToolCatalog() {
	if t.telemetry == nil || t.toolIndex == nil || t.registry == nil {
		return
	}
	t.catalogSnapshotMu.Lock()
	defer t.catalogSnapshotMu.Unlock()
	ix := t.toolIndex
	ix.mu.RLock()
	revision := ix.revision
	ix.mu.RUnlock()
	if t.catalogSnapshotRecorded && t.catalogSnapshotRevision == revision {
		return
	}
	hash, body, tools, aliases, revision, err := t.semanticToolCatalog()
	if err != nil {
		return
	}
	ix.mu.RLock()
	stable := ix.revision == revision
	ix.mu.RUnlock()
	if !stable {
		return
	}
	t.telemetry.recordCatalog(t.threadID, hash, body, revision, tools, aliases)
	t.catalogSnapshotRecorded = true
	t.catalogSnapshotRevision = revision
	t.catalogSnapshotHash = hash
}
func (t *Thinker) currentToolCatalogHash() string {
	t.catalogSnapshotMu.Lock()
	defer t.catalogSnapshotMu.Unlock()
	return t.catalogSnapshotHash
}

func (t *Telemetry) recordCatalog(thread, hash string, body json.RawMessage, revision uint64, tools, aliases int) {
	t.catalogMu.Lock()
	defer t.catalogMu.Unlock()
	if t.catalogSnapshots == nil {
		t.catalogSnapshots = map[string]catalogDiagnosticSnapshot{}
	}
	fields := map[string]any{"catalog_hash": hash, "catalog_format": toolCatalogFormat, "revision": revision, "tool_count": tools, "alias_count": aliases}
	if _, seen := t.catalogSnapshots[hash]; !seen {
		var full map[string]json.RawMessage
		if json.Unmarshal(body, &full) != nil {
			return
		}
		fields["tools"] = full["tools"]
		fields["aliases"] = full["aliases"]
	} else {
		fields["catalog_reference"] = true
	}
	// Serialize the initial snapshot before its references in the telemetry log.
	t.Emit("tool.catalog", thread, fields)
	t.catalogSnapshots[hash] = catalogDiagnosticSnapshot{Thread: thread, Body: body, Revision: revision, Tools: tools, Aliases: aliases}
}

// Keep one in-memory payload per content hash so Server can recover a catalog
// after retention, a restore, or a lost first snapshot, without logging it again
// on ordinary requests. Recovery is requested by the authenticated ingest peer.
type catalogDiagnosticSnapshot struct {
	Thread         string
	Body           json.RawMessage
	Revision       uint64
	Tools, Aliases int
}

func (t *Telemetry) replayMissingCatalogs(hashes []string) {
	t.catalogMu.Lock()
	defer t.catalogMu.Unlock()
	if len(hashes) > 64 {
		hashes = hashes[:64]
	}
	seen := map[string]bool{}
	for _, hash := range hashes {
		if seen[hash] {
			continue
		}
		seen[hash] = true
		snapshot, ok := t.catalogSnapshots[hash]
		if !ok {
			continue
		}
		var full map[string]json.RawMessage
		if json.Unmarshal(snapshot.Body, &full) != nil {
			continue
		}
		t.Emit("tool.catalog", snapshot.Thread, map[string]any{"catalog_hash": hash, "catalog_format": toolCatalogFormat, "revision": snapshot.Revision, "tool_count": snapshot.Tools, "alias_count": snapshot.Aliases, "tools": full["tools"], "aliases": full["aliases"], "catalog_recovery": true})
	}
}
