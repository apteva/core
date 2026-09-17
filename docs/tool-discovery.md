# Tool discovery reliability

Discovery indexes the full tool description and bounded semantic input-schema metadata: property names, descriptions, titles and enum/const values, including nested objects, arrays and schema alternatives. Defaults and examples are excluded. Schema traversal stops at depth 16, 2048 schema nodes or 32 KiB of indexed schema text; it never modifies the schema sent to a provider.

Exact canonical names and configured aliases resolve before descriptive ranking. Unambiguous raw MCP names also resolve: `tickets_get` finds `tickets_tickets_get`, including in longer queries. To avoid interpreting ordinary prose as an exact request, bare local names such as `get` resolve as aliases only when they are the entire query. Canonical identities take precedence, then configured aliases, then local names. Local-name ambiguity is determined across the catalog, but only authorized candidates are disclosed. Reconnect/removal rebuilds the local-name map. Aliases never become executable identities or change permissions.

Known names return only their requested authorized identities. A denied name never falls back to another operation. Unknown names alone have no matches. For an unknown name accompanied by descriptive words, `search_tools` reports `unresolved_names` and may return a separate `suggestions` array, with `match: "descriptive_suggestion"`. Suggestions have their canonical schemas loaded in the same call, but the response explicitly says they are not equivalent replacements and must be checked against the requested operation. `hits` remains reserved for exact/alias matches or ordinary descriptive results. Ambiguous local names return `ambiguous_names` with authorized candidates instead of silently choosing one. All loading remains bounded by `k`.

Automatic preload and the internal `Search` API consume only ordinary matches, never unresolved-name suggestions: preload cannot silently omit the warning. The live registry and existing capability checks remain authoritative for presentation and execution.

Descriptive ranking uses BM25 for descriptions, a moderate local operation-name boost and a separate schema-field boost. Server vocabulary has only a small namespace weight, even when repeated in the local name. A small discovery-only vocabulary normalization handles common plurals, assignment terms and read/get/fetch terms. Canonical names are never normalized or rewritten. Repeated schema terms cannot inflate their weight, and deterministic canonical-name ordering resolves score ties.

A server may configure discovery aliases with raw local tool names as targets:

```json
{
  "name": "tasks",
  "url": "http://localhost:5280/api/apps/tasks/mcp",
  "tool_aliases": {"occurrence_reader": "get"}
}
```

Aliases return the canonical `tasks_get` schema and dispatch identity. They do not create additional executable names or grant permissions. Conflicting aliases and conflicting registered MCP identities are rejected. Reconnects remove obsolete aliases.

## Execution requirements and retention

Tracked inbox events may supply an explicit first action:

```json
{
  "type": "task.ready",
  "required_first_action": {
    "tool": "tasks_get",
    "task_id": "the-occurrence-id"
  }
}
```

Core stores that requirement with the durable execution, outside the model's conversation history. Authorized required tools remain in the model surface across discovery eviction, compaction, context refresh, and restart while the execution is active. The requirement never grants access to a tool that the receiving worker lacks.

Until a matching first action succeeds and its success is persisted, dependent MCP operations are rejected. A missing prerequisite produces an explicit capability failure. Control and reporting tools remain available so the model can explain the blocker. Multiple execution prerequisites can be read in either order.

New search results are also retained through the promised next model request, including retries in the same iteration. Worker permissions are reapplied when merging retained schemas. Tools actually used refresh their LRU age; ties evict deterministically.

Three equal MCP failures for the same operation and stable target block further dispatch of that operation. Changing `_reason` or other explanatory prose cannot bypass a target-based failure guard. A successful operation on that target or a fresh external instruction clears it. This prevents repeated failed RPCs; it is not a guarantee about all future model choices.

## Diagnostics

- `tool.catalog`: mounted identities, full descriptions, model schema parameters, loading policy, and aliases, once per catalog revision.
- `tool.discovery`: query, ordered hits, match mode, descriptive scores, activated names, and optional unresolved names, ambiguous names and suggestions.
- `tool.manifest`: exact model-visible names and schema/identity hashes for each request.
- `llm.start.tool_manifest_hash`: joins the model request to its manifest.
- `tool.arguments`: existing raw provider, parsed, and typed argument stages, joined by tool call ID.
- `tool.dispatch`: the immutable MCP server/local name actually resolved, model manifest hash, and typed arguments.
- `tool.blocked`: prerequisite and repeated-failure rejections.

The request schema list and registry definitions are captured atomically. A connection, identity, or schema replacement between the model request and admission returns `capability_changed` rather than silently retargeting the call. Admitted calls hold the resolved immutable definition.

## Tests

```sh
# Local regressions, including production catalog exact-name coverage.
go test -short ./...
go test -race -short ./...

# Real Codex, using existing saved authentication and only local mock MCPs.
RUN_CODEX_DISCOVERY=1 go test -run '^TestDiscoveryCodexProductionCatalog$' -v -count=1 -timeout 15m .

# Lookup overhead, excluding network/model inference.
go test -run '^$' -bench BenchmarkDiscoveryProductionCatalog -benchmem .
```

Live cases cover exact lookup, Lily's verbose query, configured aliases, scoped workers, missing readers, incoming prerequisites, missing prerequisites, and descriptive discovery of an opaque reader. Each successful case must read a unique receipt through the MCP HTTP boundary and report that receipt. Tests inspect raw model function names and resolved dispatch telemetry, and reject any task mutation.

`testdata/patreon_tool_catalog.json` contains 142 descriptions recovered from production startup logs. Those logs truncated long descriptions at 2000 bytes; the fixture is not a complete historical request/schema snapshot. The live MCPs use representative local argument schemas. A separate regression verifies discovery can find a unique term beyond 40 KB in a full description.

## Tickets / Kimi K3 regression

`tool_discovery_tickets_test.go` uses the original terse incident descriptions,
including an update tool whose assignment capability appears only in its schema.
It covers the reported queries, preload, alias collisions/reconnects, suggestion
labeling, capability filtering, and nested/bounded schema indexing.

```sh
# Paid Kimi K3 inference; OPENCODE_GO_API_KEY comes from env or the local .env.
# Tickets operations run only against a local HTTP fixture.
RUN_LLM_INTEGRATION_TESTS=1 go test -run '^(TestDiscoveryKimiK3Tickets|TestIntegration_OpenCodeGo_KimiK3SessionToolRoundTrip)$' -v -count=1 -timeout 8m .
```

The Kimi scenarios force discovery to avoid masking failures with preload. Both
short-name and descriptive/schema queries must use exactly one search, then read
and assign the ticket through the canonical Core-to-MCP mapping and report the
unique receipt returned by the assignment. Any extra business operation fails.
These are controlled regressions, not an estimate of production latency savings.

Validation on September 15, 2026: both real Kimi K3 cases passed with one
`search_tools` call and exactly `tickets_get` then `tickets_update` at the MCP
boundary. Short-name case: 11.342 s; descriptive/schema case: 27.815 s (whole
fixture workflow, not search execution time). The existing Kimi K3 native
session/tool-result continuation test also passed in 3.713 s. The 142-tool
catalog benchmark measured approximately 0.18 microseconds per exact lookup
and 82 microseconds per descriptive query on the local Apple M1 Pro.

The optional `automatic_context` Kimi scenario covers the default-enabled
[automatic loading](automatic-tool-loading.md), requiring zero searches.
