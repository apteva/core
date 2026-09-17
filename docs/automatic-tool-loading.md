# Automatic tool loading

Automatic tool loading is **enabled by default**, with `search_tools` retained
as a backup. When `automatic_tool_loading` is absent, the defaults are eight
automatically selected tools, approximately 2,000 schema tokens, and relevance
from already-recalled memory. Existing explicit settings, including
`enabled: false`, remain authoritative. This default applies when Core runs the
new build; it does not mutate saved agent configuration.

Example partial **Core `PUT /config`** request to customize the defaults:

```json
{
  "automatic_tool_loading": {
    "enabled": true,
    "workflow_tools": ["tickets_tickets_get"],
    "max_tools": 8,
    "max_schema_tokens": 2000,
    "include_memory": true
  }
}
```

`GET /config` reports the effective setting. Config changes use the existing
serialized runtime mutation and transactional persistence path. Failed writes
restore the prior setting, including when the optional field was absent.
Disable with `{"automatic_tool_loading":{"enabled":false}}`; omission in an
unrelated partial update preserves the setting. Each supplied settings object
replaces the previous one. There is no dashboard control for these settings yet.
The running-agent server config proxy forwards the setting to Core; the existing
server stopped-agent config whitelist does not yet accept this
field. Use a running Core or its persisted config for custom overrides.

## Selection

With no settings block, memory relevance is enabled. When supplying an explicit
settings object, set `enabled: true` and `include_memory: true` to retain those
choices; omitted boolean fields in that replacement object are false. Setting
`enabled: false` restores the legacy preload and keeps search available.

In discovery mode, this replaces the legacy 3/5-tool preload for native text
requests. It does not override eager mode (`APTEVA_TOOL_SEARCH=off`, or the
existing automatic eager threshold), and does not alter realtime tool profiles.
To exercise it on a small fixture, set `APTEVA_TOOL_SEARCH=on` for that test
process. `search_tools` remains available as a fallback.

1. Prioritize configured canonical MCP names in `workflow_tools`.
2. Search separately using the current instruction, this thread's active
   execution reason/prerequisite, standing directive, and optionally the memory
   context already selected by automatic recall. Interleave ranked sources so
   the directive cannot consume all available slots.
3. Retain selections across unchanged continuation context. On a context change,
   current candidates have priority and older selections fill remaining space.
4. Resolve every name against the live registry, reapply capabilities and
   `no_spawn`, and attach the real native schemas before the provider request.

Memory is only a relevance signal. It neither grants access nor provides a
schema. This adds no model request, embedding call or separate memory lookup.
Selection does not execute tools or append anything to conversation history.
Unknown/ambiguous exact names never automatically activate suggested substitutes.

The default automatic selection budget is 8 tools and approximately 2,000 schema
tokens. `max_tools` accepts 1–20; `max_schema_tokens` accepts 1–16,000; zero selects
the default. The estimate is serialized native schema bytes divided by four,
including injected fields, not a provider tokenizer measurement. Input queries
are capped at 8 KiB per source. Oversized tools are skipped; search remains a
fallback. Configured workflow names are prioritized within these budgets; use
existing MCP `tool_loading: always` when a schema must always be presented.

Budgets apply to this automatic selection only. Existing Core tools, eager or
always schemas, exact worker grants, required execution tools, and explicit
search/use activation keep their existing rules. An automatically selected tool
that is subsequently used can remain active under those ordinary rules after
automatic loading is disabled. Full provider-request context limits still apply.

Workers inherit the instance's setting but resolve only their granted surface.
Uninstall/reconnect and permission changes are rechecked before every request;
schema changes are remeasured. Provider request schemas still go through the
existing snapshot, normalization and dispatch identity checks.

## Telemetry and tests

`tool.autoload` records selected canonical names, selection reasons, approximate
schema tokens, configured limits, budget skips, local selection duration in
microseconds and a context hash. It does not emit raw instruction/memory text.
`tool.manifest` remains the evidence of what the model actually received.

Tests cover disabled/default compatibility, first-request visibility, memory
opt-in, execution-state scope, stable tool-result continuations, budgets,
permissions, catalog/schema changes, explicit-search coexistence, eager mode,
config round trips and persistence rollback.

```sh
go test -short ./...
go test -race -short -run 'Test(Automatic|Discovery|ApplyPreload|MCPToolLoading|API.*Config)' .

# Paid real Kimi K3; all ticket operations terminate at a local MCP fixture.
RUN_LLM_INTEGRATION_TESTS=1 go test -run '^TestDiscoveryKimiK3Tickets$' -v -count=1 -timeout 10m .
```

The `automatic_context` scenario must see both read and assignment schemas in
the first model manifest, use zero search calls, read ticket 2, assign it with
all three assignee fields, and report the unique actual assignment receipt.
The two explicit-discovery scenarios still require exactly one search each.
This demonstrates orchestration correctness; it is not a production latency or
cache-hit-rate guarantee.

Automatic retrieval distinguishes known non-MCP tool names and known top-level
schema parameter names from MCP name lookups. Those references do not suppress
the descriptive search for other workflow tools. Canonical identities and
aliases take precedence over parameter vocabulary; a denied or genuinely
unknown identity retains the strict no-substitution behavior.

Validation on September 15, 2026: the final real Kimi run completed ticket read
and assignment with zero searches in `automatic_context` (3 recorded model
requests, 6.533 s). The explicit descriptive-search comparison used one search
(5 recorded requests, 9.761 s). These single-run timings are illustrative only;
model behavior and provider latency vary. An earlier comparison hit an OpenCode
stream-idle timeout and was rerun. The first automatic trial also exposed the
Core-name/parameter classification issue described above; the added regression
and final live run pass after its correction.

The `automatic_default` Kimi scenario omits the settings block and must complete
with zero searches. `automatic_backup` intentionally limits the automatic schema
budget to one token; the model must recover with one ordinary search and still
read and assign the ticket. The search tool remains in both first-request
manifests. Default-on memory retrieval and explicit opt-out are also covered by
local regressions.

Default-on validation: real Kimi K3 completed `automatic_default` with zero
searches (12.587 s) and `automatic_backup` with one search (18.734 s), both with
successful ticket read and assignment receipts. The latter used a one-token
automatic budget to prove the query tool still recovers missing schemas. These
are controlled fixture timings, not production performance promises.

Local rollout: build `2026-09-15T08:42:52Z` was installed and verified on all 23
running local agents, starting with agent 1118. Each reported automatic loading
and memory relevance enabled with the default budgets. Stopped agents were not
started; all prior agent processes exited. The final full short suite and focused
race tests passed. Remote production was not changed.
