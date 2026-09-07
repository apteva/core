# Tool discovery reliability

Discovery indexes the entire tool description. Exact registered names and host-configured aliases resolve before descriptive ranking, including names embedded in a longer query such as `tasks_get exact occurrence by task id`. Exact lookups return only requested authorized identities. An unavailable or unauthorized qualified name returns `capability_unavailable`; it never falls back to another operation.

Descriptive queries use BM25 length normalization and term-frequency saturation with a separate name boost. Repeating tool references in a long description cannot displace an explicit name. The normalized name lookup is built at mount/reconnect, so exact requests do not scan or score the corpus.

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
- `tool.discovery`: query, ordered hits, match mode, descriptive scores, and activated names.
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
